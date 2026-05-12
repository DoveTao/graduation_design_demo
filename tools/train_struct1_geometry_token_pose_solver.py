#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GeometryTokenPoseSolver, SoftCorrespondenceGeometryLayer, SphericalGeometryTokenBackbone
from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import pair_features, relative_pose_A_to_B_in_B, scan_frames, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head, rotvec_to_matrix

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E15_POLICY = Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/s5e15_refinement_policy.json")


def load_struct1_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    def parse_scalar(raw: str) -> Any:
        value = raw.strip()
        if not value:
            return ""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            if any(ch in value for ch in [".", "e", "E"]):
                return float(value)
            return int(value)
        except Exception:
            return value

    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    def next_container(start_idx: int) -> Any:
        for j in range(start_idx + 1, len(lines)):
            probe = lines[j]
            stripped = probe.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return [] if stripped.startswith("- ") else {}
        return {}

    for idx, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected list item in {path}: {raw}")
            parent.append(parse_scalar(stripped[2:]))
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue
        container = next_container(idx)
        parent[key] = container
        stack.append((indent, container))
    return root


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def _erp_to_bearing(x: float, y: float, width: int, height: int) -> np.ndarray:
    lon = (float(x) / max(width - 1, 1) - 0.5) * 2.0 * math.pi
    lat = (0.5 - float(y) / max(height - 1, 1)) * math.pi
    c = math.cos(lat)
    return np.asarray([c * math.sin(lon), math.sin(lat), c * math.cos(lon)], dtype=np.float32)


def _softmax_rows(scores: np.ndarray, tau: float = 0.05) -> np.ndarray:
    z = scores / max(tau, 1.0e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / np.clip(np.sum(ex, axis=1, keepdims=True), 1.0e-12, None)


def _rot_geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(R_pred.float(), R_gt.float().transpose(-1, -2))
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    c = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(c))


def _vector_angle_t(v1: torch.Tensor, v2: torch.Tensor, absolute: bool = False) -> torch.Tensor:
    a = F.normalize(v1.float(), dim=-1, eps=1e-6)
    b = F.normalize(v2.float(), dim=-1, eps=1e-6)
    dot = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
    if absolute:
        dot = torch.abs(dot)
    return torch.rad2deg(torch.arccos(dot))


def _gather_square_matrix(mat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    rows = torch.gather(mat, 1, idx.unsqueeze(-1).expand(-1, -1, mat.shape[-1]))
    return torch.gather(rows, 2, idx.unsqueeze(1).expand(-1, rows.shape[1], -1))


@dataclass
class EdgeSample:
    scene: str
    seq: str
    edge_index: int
    local_features: np.ndarray
    bearing_a: np.ndarray
    bearing_b: np.ndarray
    W_ab: np.ndarray
    W_ba: np.ndarray
    R_coarse: np.ndarray
    base_tdir: np.ndarray
    base_log_tmag: float
    base_tmag: float
    gt_R: np.ndarray
    gt_tdir: np.ndarray
    gt_tmag: float
    observability_score: float
    entropy_mean: float
    confidence_mean: float
    cycle_error_mean: float
    epipolar_residual_mean: float
    dt: float


class Struct1GeometryTokenModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], token_in_dim: int) -> None:
        super().__init__()
        hidden_dim = int(cfg["model"]["hidden_dim"])
        self.backbone_a = SphericalGeometryTokenBackbone(
            token_in_dim,
            hidden_dim,
            use_local_tangent_coords=bool(cfg["spherical_backbone"]["local_tangent_coords"]),
            use_bearing_xyz_channels=bool(cfg["spherical_backbone"]["bearing_xyz_channels"]),
            use_latlon_sincos_channels=bool(cfg["spherical_backbone"]["latlon_sincos_channels"]),
            token_encoder_layers=int(cfg["spherical_backbone"]["token_encoder_layers"]),
            n_heads=int(cfg["model"]["backbone_heads"]),
        )
        self.backbone_b = SphericalGeometryTokenBackbone(
            token_in_dim,
            hidden_dim,
            use_local_tangent_coords=bool(cfg["spherical_backbone"]["local_tangent_coords"]),
            use_bearing_xyz_channels=bool(cfg["spherical_backbone"]["bearing_xyz_channels"]),
            use_latlon_sincos_channels=bool(cfg["spherical_backbone"]["latlon_sincos_channels"]),
            token_encoder_layers=int(cfg["spherical_backbone"]["token_encoder_layers"]),
            n_heads=int(cfg["model"]["backbone_heads"]),
        )
        self.geometry_layer = SoftCorrespondenceGeometryLayer(
            hidden_dim,
            hidden_dim=hidden_dim,
            temperature=float(cfg["soft_correspondence_geometry_layer"]["temperature"]),
        )
        self.coarse_solver = GeometryTokenPoseSolver(
            hidden_dim,
            hidden_dim=hidden_dim,
            topk=int(cfg["model"]["topk_pool"]),
            scale_delta_clip=float(cfg["pose_solver"]["scale_delta_clip"]),
            rotation_refine_scale=float(cfg["pose_solver"]["rotation_refine_scale"]),
        )
        self.fine_solver = GeometryTokenPoseSolver(
            hidden_dim,
            hidden_dim=hidden_dim,
            topk=min(int(cfg["spherical_backbone"]["fine_tokens"]), int(cfg["model"]["topk_pool"])),
            scale_delta_clip=float(cfg["pose_solver"]["scale_delta_clip"]),
            rotation_refine_scale=float(cfg["pose_solver"]["rotation_refine_scale"]),
        )
        self.fine_tokens = int(cfg["spherical_backbone"]["fine_tokens"])

    def forward(
        self,
        local_features: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
        W_ab: torch.Tensor,
        W_ba: torch.Tensor,
        R_coarse: torch.Tensor,
        base_tdir: torch.Tensor,
        base_log_tmag: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tok_a = self.backbone_a(local_features, bearing_a)
        tok_b = self.backbone_b(local_features, bearing_b)
        coarse_geom = self.geometry_layer(
            tok_a["token_features"],
            tok_b["token_features"],
            tok_a["bearing"],
            tok_b["bearing"],
            interaction_feature=0.5 * (tok_a["token_features"] + tok_b["token_features"]),
            precomputed_W_ab=W_ab,
            precomputed_W_ba=W_ba,
            coarse_pose_R=R_coarse,
            coarse_pose_tdir=base_tdir,
        )
        coarse_pose = self.coarse_solver(
            coarse_geom["geometry_tokens"],
            coarse_geom["confidence"],
            coarse_rotation=R_coarse,
            coarse_tdir=base_tdir,
            base_log_tmag=base_log_tmag,
        )
        fine_k = min(self.fine_tokens, coarse_geom["geometry_tokens"].shape[1])
        fine_idx = torch.topk(coarse_geom["confidence"], k=fine_k, dim=-1).indices
        fine_gather = fine_idx.unsqueeze(-1).expand(-1, -1, tok_a["token_features"].shape[-1])
        fine_a = torch.gather(tok_a["token_features"], 1, fine_gather)
        fine_b = torch.gather(tok_b["token_features"], 1, fine_gather)
        fine_ba = torch.gather(tok_a["bearing"], 1, fine_idx.unsqueeze(-1).expand(-1, -1, 3))
        fine_bb = torch.gather(tok_b["bearing"], 1, fine_idx.unsqueeze(-1).expand(-1, -1, 3))
        fine_W_ab = _gather_square_matrix(W_ab, fine_idx)
        fine_W_ba = _gather_square_matrix(W_ba, fine_idx)
        fine_geom = self.geometry_layer(
            fine_a,
            fine_b,
            fine_ba,
            fine_bb,
            interaction_feature=0.5 * (fine_a + fine_b),
            precomputed_W_ab=fine_W_ab,
            precomputed_W_ba=fine_W_ba,
            coarse_pose_R=coarse_pose["R_BA"],
            coarse_pose_tdir=coarse_pose["tdir_B"],
        )
        fine_pose = self.fine_solver(
            fine_geom["geometry_tokens"],
            fine_geom["confidence"],
            coarse_rotation=coarse_pose["R_BA"],
            coarse_tdir=coarse_pose["tdir_B"],
            base_log_tmag=coarse_pose["final_log_tmag"],
        )
        final_tvec_B = fine_pose["tdir_B"] * fine_pose["final_tmag"].unsqueeze(-1)
        return {
            "R_BA": fine_pose["R_BA"],
            "tdir_B": fine_pose["tdir_B"],
            "tmag": fine_pose["final_tmag"],
            "final_tvec_B": final_tvec_B,
            "coarse_pose": coarse_pose,
            "coarse_geom": coarse_geom,
            "fine_pose": fine_pose,
            "fine_geom": fine_geom,
            "backbone_a": tok_a,
            "backbone_b": tok_b,
            "fine_indices": fine_idx,
            "fine_stage_used": True,
            "W_ab_used_in_pose_solver": True,
            "tdir_from_geometry_tokens": True,
        }


def _edge_softcorr(frame_i: Any, frame_j: Any, R_coarse: np.ndarray, max_matches: int, temperature: float) -> Dict[str, Any]:
    width, height = 640, 320
    img_i = load_gray(frame_i.image_path, (width, height))
    img_j = load_gray(frame_j.image_path, (width, height))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    pts_i = np.asarray(orb.get("match_points_i", []), dtype=np.float32)
    pts_j = np.asarray(orb.get("match_points_j", []), dtype=np.float32)
    if pts_i.size == 0 or pts_j.size == 0:
        pts_i = np.asarray([[width * 0.5, height * 0.5]], dtype=np.float32)
        pts_j = np.asarray([[width * 0.5, height * 0.5]], dtype=np.float32)
    if pts_i.shape[0] > max_matches:
        keep = np.linspace(0, pts_i.shape[0] - 1, num=max_matches, dtype=np.int64)
        pts_i = pts_i[keep]
        pts_j = pts_j[keep]
    N = pts_i.shape[0]
    bearing_a = np.stack([_erp_to_bearing(float(x), float(y), width, height) for x, y in pts_i], axis=0)
    bearing_b = np.stack([_erp_to_bearing(float(x), float(y), width, height) for x, y in pts_j], axis=0)
    rot_a = (R_coarse @ bearing_a.T).T
    scores = rot_a @ bearing_b.T
    W_ab = _softmax_rows(scores, tau=temperature)
    W_ba = _softmax_rows(scores.T, tau=temperature)
    disp = pts_j - pts_i
    disp_mag = np.linalg.norm(disp, axis=1, keepdims=True)
    local_features = np.concatenate(
        [
            disp / np.asarray([[width, height]], dtype=np.float32),
            disp_mag / max(float(np.percentile(disp_mag, 90)) if disp_mag.size else 1.0, 1.0e-6),
            np.full((N, 1), float(orb.get("inlier_ratio", 0.0)), dtype=np.float32),
            np.full((N, 1), float(orb.get("parallax_proxy", 0.0)), dtype=np.float32),
            np.full((N, 1), float(flow.get("median_flow_magnitude", 0.0)), dtype=np.float32),
            np.full((N, 1), float(flow.get("flow_angle_dispersion", 180.0)) / 180.0, dtype=np.float32),
            np.full((N, 1), 1.0 if orb.get("low_parallax_flag", True) else 0.0, dtype=np.float32),
            np.full((N, 1), 1.0 if float(flow.get("median_flow_magnitude", 0.0)) < 0.75 else 0.0, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    probs = np.clip(W_ab, 1.0e-9, None)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    confidence = 1.0 - entropy / max(math.log(max(N, 2)), 1.0e-6)
    cyc = W_ab @ W_ba
    cycle_err = np.mean(np.abs(cyc - np.eye(N, dtype=np.float32)), axis=1)
    matched_b = W_ab @ bearing_b
    matched_b = matched_b / np.clip(np.linalg.norm(matched_b, axis=1, keepdims=True), 1.0e-12, None)
    obs = (
        0.35 * float(np.mean(confidence))
        + 0.25 * min(1.0, float(orb.get("parallax_proxy", 0.0)) / 8.0)
        + 0.20 * min(1.0, float(flow.get("median_flow_magnitude", 0.0)) / 6.0)
        + 0.20 * max(0.0, 1.0 - float(np.mean(cycle_err)))
    )
    if N < max_matches:
        pad = max_matches - N
        bearing_a = np.pad(bearing_a, ((0, pad), (0, 0)), mode="edge")
        bearing_b = np.pad(bearing_b, ((0, pad), (0, 0)), mode="edge")
        local_features = np.pad(local_features, ((0, pad), (0, 0)), mode="edge")
        W_ab = np.pad(W_ab, ((0, pad), (0, pad)), mode="constant")
        W_ba = np.pad(W_ba, ((0, pad), (0, pad)), mode="constant")
        for i in range(N, max_matches):
            W_ab[i, i] = 1.0
            W_ba[i, i] = 1.0
    return {
        "local_features": local_features.astype(np.float32),
        "bearing_a": bearing_a.astype(np.float32),
        "bearing_b": bearing_b.astype(np.float32),
        "W_ab": W_ab.astype(np.float32),
        "W_ba": W_ba.astype(np.float32),
        "entropy_mean": float(np.mean(entropy)),
        "confidence_mean": float(np.mean(confidence)),
        "cycle_error_mean": float(np.mean(cycle_err)),
        "observability_score": float(obs),
        "epipolar_residual_mean": None,
    }


def _make_rows(cfg: Dict[str, Any]) -> Tuple[List[EdgeSample], Dict[str, Any], List[Dict[str, Any]], int]:
    frames = scan_frames(Path("data"), scene="scene01")
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    scale_factor = float(json.loads(S5E15_POLICY.read_text(encoding="utf-8"))["scale_factor"])
    cache_img: Dict[str, np.ndarray] = {}
    rows: List[EdgeSample] = []
    seq_ranges: Dict[str, List[int]] = {}
    max_matches = int(cfg["soft_correspondence_geometry_layer"]["max_matches"])
    temperature = float(cfg["soft_correspondence_geometry_layer"]["temperature"])
    for scene_seq in [str(x) for x in cfg["training"]["train_scenes"]]:
        scene, seq = scene_seq.split("/")
        seq_frames = frames[(scene, seq)]
        seq_ranges[scene_seq] = []
        for idx in range(len(seq_frames) - 1):
            a, b = seq_frames[idx], seq_frames[idx + 1]
            x = pair_features(a, b, idx, max(len(seq_frames) - 1, 1), cache_img)
            prior = predict_s5e2(s5e2, x)
            R_coarse = rotvec_to_matrix(prior[:3])
            base_tdir = _norm(prior[3:6])
            base_log_tmag = float(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]) + math.log(scale_factor)
            gt_R, gt_t = relative_pose_A_to_B_in_B(a, b)
            gt_tmag = float(np.linalg.norm(gt_t))
            gt_tdir = _norm(gt_t)
            edge = _edge_softcorr(a, b, R_coarse, max_matches=max_matches, temperature=temperature)
            matched_b = edge["W_ab"] @ edge["bearing_b"]
            matched_b = matched_b / np.clip(np.linalg.norm(matched_b, axis=1, keepdims=True), 1.0e-12, None)
            bA_rot = (R_coarse @ edge["bearing_a"].T).T
            epi = np.abs(np.sum(np.cross(np.repeat(gt_tdir[None, :], matched_b.shape[0], axis=0), matched_b) * bA_rot, axis=1))
            edge["epipolar_residual_mean"] = float(np.mean(epi))
            rows.append(
                EdgeSample(
                    scene=scene,
                    seq=seq,
                    edge_index=idx,
                    local_features=edge["local_features"],
                    bearing_a=edge["bearing_a"],
                    bearing_b=edge["bearing_b"],
                    W_ab=edge["W_ab"],
                    W_ba=edge["W_ba"],
                    R_coarse=R_coarse.astype(np.float32),
                    base_tdir=base_tdir.astype(np.float32),
                    base_log_tmag=base_log_tmag,
                    base_tmag=float(np.exp(base_log_tmag)),
                    gt_R=gt_R.astype(np.float32),
                    gt_tdir=gt_tdir.astype(np.float32),
                    gt_tmag=gt_tmag,
                    observability_score=edge["observability_score"],
                    entropy_mean=edge["entropy_mean"],
                    confidence_mean=edge["confidence_mean"],
                    cycle_error_mean=edge["cycle_error_mean"],
                    epipolar_residual_mean=edge["epipolar_residual_mean"],
                    dt=float(b.timestamp - a.timestamp),
                )
            )
            seq_ranges[scene_seq].append(len(rows) - 1)
    obs = np.asarray([r.observability_score for r in rows], dtype=np.float64)
    low_thr = float(np.quantile(obs, float(cfg["observability"]["low_threshold_quantile"]))) if obs.size else 0.0
    high_thr = float(np.quantile(obs, float(cfg["observability"]["high_threshold_quantile"]))) if obs.size else 1.0
    windows: List[Dict[str, Any]] = []
    for scene_seq, idxs in seq_ranges.items():
        scene, seq = scene_seq.split("/")
        seq_frames = frames[(scene, seq)]
        for k in [int(x) for x in cfg["kstep_geometry"]["k_values"]]:
            for start in range(0, len(idxs) - k):
                edge_ids = idxs[start : start + k]
                frame_a = seq_frames[start]
                frame_k = seq_frames[start + k]
                gt_R, gt_t = relative_pose_A_to_B_in_B(frame_a, frame_k)
                windows.append(
                    {
                        "scene": scene,
                        "seq": seq,
                        "start_index": start,
                        "k": k,
                        "edge_ids": edge_ids,
                        "gt_R": gt_R.astype(np.float32).tolist(),
                        "gt_tdir": _norm(gt_t).astype(np.float32).tolist(),
                        "gt_tmag": float(np.linalg.norm(gt_t)),
                        "gt_path_len": float(sum(rows[e].gt_tmag for e in edge_ids)),
                    }
                )
    dataset = {
        "num_windows_by_k": {str(k): sum(1 for w in windows if w["k"] == k) for k in [1, 2, 3, 5]},
        "all_windows_contiguous": True,
        "uses_eval_scene": False,
        "ready_for_training": bool(rows) and bool(windows),
        "high_threshold": high_thr,
        "low_threshold": low_thr,
        "num_rows": len(rows),
    }
    return rows, dataset, windows, int(rows[0].local_features.shape[1]) if rows else 8


def _tensorize(batch_rows: Sequence[EdgeSample], low_thr: float, high_thr: float, device: torch.device) -> Dict[str, torch.Tensor]:
    def gate_weight(obs: float) -> float:
        if obs <= low_thr:
            return 0.10
        if obs >= high_thr:
            return 1.0
        return 0.10 + 0.90 * ((obs - low_thr) / max(high_thr - low_thr, 1.0e-6))

    return {
        "local_features": torch.tensor(np.stack([r.local_features for r in batch_rows]), dtype=torch.float32, device=device),
        "bearing_a": torch.tensor(np.stack([r.bearing_a for r in batch_rows]), dtype=torch.float32, device=device),
        "bearing_b": torch.tensor(np.stack([r.bearing_b for r in batch_rows]), dtype=torch.float32, device=device),
        "W_ab": torch.tensor(np.stack([r.W_ab for r in batch_rows]), dtype=torch.float32, device=device),
        "W_ba": torch.tensor(np.stack([r.W_ba for r in batch_rows]), dtype=torch.float32, device=device),
        "R_coarse": torch.tensor(np.stack([r.R_coarse for r in batch_rows]), dtype=torch.float32, device=device),
        "base_tdir": torch.tensor(np.stack([r.base_tdir for r in batch_rows]), dtype=torch.float32, device=device),
        "base_log_tmag": torch.tensor([r.base_log_tmag for r in batch_rows], dtype=torch.float32, device=device),
        "gt_R": torch.tensor(np.stack([r.gt_R for r in batch_rows]), dtype=torch.float32, device=device),
        "gt_tdir": torch.tensor(np.stack([r.gt_tdir for r in batch_rows]), dtype=torch.float32, device=device),
        "gt_tmag": torch.tensor([r.gt_tmag for r in batch_rows], dtype=torch.float32, device=device),
        "gate_floor": torch.tensor([gate_weight(r.observability_score) for r in batch_rows], dtype=torch.float32, device=device),
        "cycle_error_mean": torch.tensor([r.cycle_error_mean for r in batch_rows], dtype=torch.float32, device=device),
        "epipolar_residual_mean": torch.tensor([r.epipolar_residual_mean for r in batch_rows], dtype=torch.float32, device=device),
        "base_tmag": torch.tensor([r.base_tmag for r in batch_rows], dtype=torch.float32, device=device),
    }


def build_model_blob(cfg: Dict[str, Any], model: Struct1GeometryTokenModel, dataset: Dict[str, Any], token_in_dim: int, optimizer_steps: int) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "token_in_dim": token_in_dim,
        "config": cfg,
        "dataset": dataset,
        "optimizer_step_count": optimizer_steps,
        "architecture": {
            "uses_spherical_tokens": True,
            "uses_soft_correspondence_geometry_layer": True,
            "uses_geometry_tokens": True,
            "W_ab_used_in_pose_solver": True,
            "tdir_from_geometry_tokens": True,
            "fine_stage_used": True,
            "uses_external_router": False,
        },
    }


def load_trained_struct1(path: Path) -> Tuple[Struct1GeometryTokenModel, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    cfg = blob["config"]
    model = Struct1GeometryTokenModel(cfg, token_in_dim=int(blob["token_in_dim"]))
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_struct1_config(Path(args.config))
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, dataset, windows, token_in_dim = _make_rows(cfg)
    out_dir = Path(args.candidate_dir) if args.candidate_dir else Path("checkpoints/STRUCT1_mainline_rebuild_with_geometry_tokens_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dataset["ready_for_training"]:
        status = {
            "real_training_executed": False,
            "optimizer_step_count": 0,
            "learned_weights_saved": False,
            "smoke_policy_only": False,
            "uses_eval_gt_for_training": False,
            "uses_orbslam3_teacher": False,
            "uses_geometry_tokens": True,
            "W_ab_used_in_pose_solver": True,
            "tdir_from_geometry_tokens": True,
            "fine_stage_used": True,
            "classification": "STRUCT1_TRAINING_BLOCKED",
        }
        write_json(out_dir / "training_status.json", status)
        return status
    low_thr = float(dataset["low_threshold"])
    high_thr = float(dataset["high_threshold"])
    model = Struct1GeometryTokenModel(cfg, token_in_dim=token_in_dim).to(device)
    for name, param in model.named_parameters():
        if name.startswith("backbone_a.") or name.startswith("backbone_b.") or name.startswith("geometry_layer."):
            param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    batch_size = int(cfg["training"]["batch_size"])
    max_steps = int(cfg["training"]["max_steps"])
    optimizer_steps = 0
    for _ in range(max_steps):
        batch_rows = random.sample(rows, k=min(batch_size, len(rows)))
        batch = _tensorize(batch_rows, low_thr, high_thr, device)
        out = model(
            batch["local_features"],
            batch["bearing_a"],
            batch["bearing_b"],
            batch["W_ab"],
            batch["W_ba"],
            batch["R_coarse"],
            batch["base_tdir"],
            batch["base_log_tmag"],
        )
        pred_dot = torch.sum(out["tdir_B"] * batch["gt_tdir"], dim=-1).clamp(-1.0, 1.0)
        abs_dot = torch.abs(pred_dot)
        loss_rot = _rot_geodesic_deg(out["R_BA"], batch["gt_R"]).mean() / 180.0
        weight = batch["gate_floor"]
        loss_tdir = torch.sum(weight * (1.0 - pred_dot)) / torch.clamp(weight.sum(), min=1.0)
        loss_abs = torch.mean((0.2 + 0.8 * weight) * (1.0 - abs_dot))
        loss_anti = torch.sum(weight * torch.relu(0.15 - pred_dot)) / torch.clamp(weight.sum(), min=1.0)
        loss_tmag = torch.mean(torch.abs(out["fine_pose"]["final_log_tmag"] - torch.log(batch["gt_tmag"].clamp_min(1.0e-6))))
        loss_cycle = out["fine_geom"]["cycle_error"].mean()
        loss_epi = out["fine_geom"]["epipolar_residual"].mean()
        loss_inv = torch.mean(torch.linalg.norm(out["fine_geom"]["matched_bearing_a"] - out["fine_geom"]["bearing_b"], dim=-1))
        conf_reg = torch.mean(out["fine_geom"]["entropy"] * out["fine_geom"]["confidence"])
        window_terms: List[torch.Tensor] = []
        sampled_windows = random.sample(windows, k=min(int(cfg["training"]["kstep_batch_size"]), len(windows)))
        for w in sampled_windows:
            w_rows = [rows[i] for i in w["edge_ids"]]
            wb = _tensorize(w_rows, low_thr, high_thr, device)
            wout = model(wb["local_features"], wb["bearing_a"], wb["bearing_b"], wb["W_ab"], wb["W_ba"], wb["R_coarse"], wb["base_tdir"], wb["base_log_tmag"])
            R_chain = torch.eye(3, device=device)
            t_chain = torch.zeros(3, device=device)
            path_sum = torch.zeros((), device=device)
            for i in range(wout["R_BA"].shape[0]):
                R_step = wout["R_BA"][i]
                t_step = wout["final_tvec_B"][i]
                R_chain = R_step @ R_chain
                t_chain = R_step @ t_chain + t_step
                path_sum = path_sum + torch.linalg.norm(t_step)
            gt_R = torch.tensor(w["gt_R"], dtype=torch.float32, device=device)
            gt_tdir = torch.tensor(w["gt_tdir"], dtype=torch.float32, device=device)
            gt_path = torch.tensor(float(w["gt_path_len"]), dtype=torch.float32, device=device)
            window_terms.append(
                0.35 * _rot_geodesic_deg(R_chain.unsqueeze(0), gt_R.unsqueeze(0)).mean()
                + _vector_angle_t(t_chain.unsqueeze(0), gt_tdir.unsqueeze(0)).mean()
                + 0.25 * torch.abs(path_sum - gt_path)
            )
        loss_kstep = torch.stack(window_terms).mean() if window_terms else torch.zeros((), device=device)
        loss_path = torch.mean(torch.abs(torch.linalg.norm(out["final_tvec_B"], dim=-1) - batch["gt_tmag"]))
        loss = (
            float(cfg["losses"]["w_rot"]) * loss_rot
            + float(cfg["losses"]["w_tdir_signed"]) * loss_tdir
            + float(cfg["losses"]["w_tdir_abs_aux"]) * loss_abs
            + float(cfg["losses"]["w_antiparallel"]) * loss_anti
            + float(cfg["losses"]["w_tmag_log"]) * loss_tmag
            + float(cfg["losses"]["w_softcorr_cycle"]) * loss_cycle
            + float(cfg["losses"]["w_epipolar_residual"]) * loss_epi
            + float(cfg["losses"]["w_inverse_consistency"]) * loss_inv
            + float(cfg["losses"]["w_kstep_composition"]) * loss_kstep
            + float(cfg["losses"]["w_path_length_consistency"]) * loss_path
            + float(cfg["losses"]["w_confidence_entropy_regularization"]) * conf_reg
        )
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        grads_finite = True
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_finite = False
                break
        if not grads_finite:
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        optimizer_steps += 1
    torch.save(build_model_blob(cfg, model, dataset, token_in_dim, optimizer_steps), out_dir / "struct1_model.pt")
    status = {
        "real_training_executed": True,
        "optimizer_step_count": optimizer_steps,
        "learned_weights_saved": True,
        "smoke_policy_only": False,
        "uses_eval_gt_for_training": False,
        "uses_orbslam3_teacher": False,
        "uses_geometry_tokens": True,
        "W_ab_used_in_pose_solver": True,
        "tdir_from_geometry_tokens": True,
        "fine_stage_used": True,
        "coarse_token_count": int(cfg["spherical_backbone"]["coarse_tokens"]),
        "fine_token_count": int(cfg["spherical_backbone"]["fine_tokens"]),
        "softcorr_temperature": float(cfg["soft_correspondence_geometry_layer"]["temperature"]),
        "classification": "STRUCT1_REAL_TRAINING_COMPLETE" if optimizer_steps >= max_steps else "STRUCT1_SHORT_REAL_TRAINING",
    }
    write_json(out_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--candidate-dir", default="")
    return p.parse_args()


if __name__ == "__main__":
    # Allowed training classifications:
    # STRUCT1_REAL_TRAINING_COMPLETE
    # STRUCT1_SHORT_REAL_TRAINING
    # STRUCT1_TRAINING_BLOCKED
    # STRUCT1_TRAINING_ERROR
    run(parse_args())
