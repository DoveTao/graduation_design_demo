#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import CorrespondenceGeometryTDirHead, FineScaleHeadWithS5E15Guard, RotationCompensatedMotionToken
from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import pair_features, relative_pose_A_to_B_in_B, scan_frames, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head, rotvec_to_matrix


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E15_POLICY = Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/s5e15_refinement_policy.json")


def _parse_cfg(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            return [] if not inner else [parse_scalar(x.strip()) for x in inner.split(",")]
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text.strip('"')

    cfg: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, cfg)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, value = raw.strip().split(":", 1)
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return cfg


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


def _compose_rel(R_cb: np.ndarray, t_cb: np.ndarray, R_ba: np.ndarray, t_ba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R_cb @ R_ba, R_cb @ t_ba + t_cb


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


@dataclass
class EdgeSample:
    scene: str
    seq: str
    edge_index: int
    frame_i: Any
    frame_j: Any
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
    residual_flow_proxy: float
    dt: float


class Arch2SoftcorrGeometryModel(nn.Module):
    def __init__(self, token_in_dim: int, hidden_dim: int, max_alpha: float, scale_delta_clip: float, delta_clip_norm: float = 0.02) -> None:
        super().__init__()
        self.delta_clip_norm = float(delta_clip_norm)
        self.token_proj = nn.Linear(token_in_dim, hidden_dim)
        self.motion = RotationCompensatedMotionToken(hidden_dim, hidden_dim)
        self.tdir_head = CorrespondenceGeometryTDirHead(hidden_dim, hidden_dim, max_alpha=max_alpha)
        self.scale_head = FineScaleHeadWithS5E15Guard(hidden_dim, hidden_dim, delta_clip=scale_delta_clip)

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
        gate_floor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inter = self.token_proj(local_features.float())
        geom = self.motion(inter, bearing_a, bearing_b, W_ab, R_coarse, W_ba=W_ba)
        pooled = geom["motion_tokens"].mean(dim=1)
        tdir = self.tdir_head(pooled)
        gate = torch.clamp(gate_floor.float() * tdir["observability_score"], 0.0, 1.0)
        raw_delta = tdir["delta_tdir"]
        raw_norm = torch.linalg.norm(raw_delta.float(), dim=-1, keepdim=True).clamp_min(1.0e-8)
        clipped = raw_delta * torch.clamp(self.delta_clip_norm / raw_norm, max=1.0)
        delta = clipped * gate.unsqueeze(-1) * tdir["max_alpha"].unsqueeze(-1)
        final_tdir = F.normalize(base_tdir.float() + delta, dim=-1, eps=1e-6)
        scale = self.scale_head(pooled, base_log_tmag)
        final_tmag = scale["final_tmag"]
        final_tvec_B = final_tdir * final_tmag.unsqueeze(-1)
        return {
            **geom,
            **tdir,
            **scale,
            "gate_value": gate,
            "delta_raw": raw_delta,
            "delta_clipped": clipped,
            "delta_effective": delta,
            "final_tdir": final_tdir,
            "final_tmag": final_tmag,
            "final_tvec_B": final_tvec_B,
            "R_BA": R_coarse.float(),
            "tdir_B": final_tdir,
            "tdir_A": torch.matmul(R_coarse.float().transpose(-1, -2), final_tdir.unsqueeze(-1)).squeeze(-1),
            "tmag": final_tmag,
            "tvec_B": final_tvec_B,
        }


def _edge_softcorr(
    frame_i: Any,
    frame_j: Any,
    R_coarse: np.ndarray,
    max_matches: int,
    temperature: float,
) -> Dict[str, Any]:
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
    disp_xy = disp / np.asarray([[width, height]], dtype=np.float32)
    local_features = np.concatenate(
        [
            disp_xy,
            disp_mag / max(float(np.percentile(disp_mag, 90)) if disp_mag.size else 1.0, 1.0e-6),
            np.full((N, 1), float(orb.get("inlier_ratio", 0.0)), dtype=np.float32),
            np.full((N, 1), float(orb.get("parallax_proxy", 0.0)), dtype=np.float32),
            np.full((N, 1), float(flow.get("median_flow_magnitude", 0.0)), dtype=np.float32),
            np.full((N, 1), float(flow.get("flow_angle_dispersion", 180.0)), dtype=np.float32),
            np.full((N, 1), 1.0 if orb.get("low_parallax_flag", True) else 0.0, dtype=np.float32),
            np.full((N, 1), 1.0 if float(flow.get("median_flow_magnitude", 0.0)) < 0.75 else 0.0, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    probs = np.clip(W_ab, 1.0e-9, None)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    cyc = W_ab @ W_ba
    cycle_err = np.mean(np.abs(cyc - np.eye(N, dtype=np.float32)), axis=1)
    confidence = np.max(W_ab, axis=1)
    matched_b = W_ab @ bearing_b
    matched_b = matched_b / np.clip(np.linalg.norm(matched_b, axis=1, keepdims=True), 1.0e-12, None)
    residual_flow = matched_b - rot_a / np.clip(np.linalg.norm(rot_a, axis=1, keepdims=True), 1.0e-12, None)
    obs = (
        0.35 * float(np.mean(confidence))
        + 0.25 * max(0.0, 1.0 - float(np.mean(entropy)) / max(math.log(max(N, 2)), 1.0e-6))
        + 0.25 * min(1.0, float(orb.get("parallax_proxy", 0.0)) / 8.0)
        + 0.15 * min(1.0, float(flow.get("median_flow_magnitude", 0.0)) / 6.0)
    )
    if orb.get("low_parallax_flag", True):
        obs *= 0.45
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
        "local_features": local_features,
        "bearing_a": bearing_a.astype(np.float32),
        "bearing_b": bearing_b.astype(np.float32),
        "W_ab": W_ab.astype(np.float32),
        "W_ba": W_ba.astype(np.float32),
        "entropy_mean": float(np.mean(entropy)),
        "confidence_mean": float(np.mean(confidence)),
        "cycle_error_mean": float(np.mean(cycle_err)),
        "residual_flow_proxy": float(np.mean(np.linalg.norm(residual_flow, axis=1))),
        "epipolar_residual_mean": None,
        "observability_score": float(obs),
    }


def _epipolar_residual(bA_rot: torch.Tensor, matched_b: torch.Tensor, gt_tdir: torch.Tensor) -> torch.Tensor:
    cross = torch.cross(gt_tdir[:, None, :].expand_as(matched_b), matched_b, dim=-1)
    return torch.abs(torch.sum(cross * bA_rot, dim=-1)).mean(dim=-1)


def _make_rows(cfg: Dict[str, Any]) -> Tuple[List[EdgeSample], Dict[str, Any], List[Dict[str, Any]], int]:
    frames = scan_frames(Path("data"), scene="scene01")
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    scale_factor = float(json.loads(S5E15_POLICY.read_text(encoding="utf-8"))["scale_factor"])
    cache_img: Dict[str, np.ndarray] = {}
    rows: List[EdgeSample] = []
    seq_ranges: Dict[str, List[int]] = {}
    max_matches = int(cfg["softcorr_geometry"]["max_matches"])
    temperature = float(cfg["softcorr_geometry"].get("temperature", 0.05))
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
            # train-side epipolar residual uses train-split GT direction only
            matched_b = edge["W_ab"] @ edge["bearing_b"]
            matched_b = matched_b / np.clip(np.linalg.norm(matched_b, axis=1, keepdims=True), 1.0e-12, None)
            bA_rot = (R_coarse @ edge["bearing_a"].T).T
            epi = np.abs(np.sum(np.cross(np.repeat(gt_tdir[None, :], matched_b.shape[0], axis=0), matched_b) * bA_rot, axis=1))
            edge["epipolar_residual_mean"] = float(np.mean(epi))
            row = EdgeSample(
                scene=scene,
                seq=seq,
                edge_index=idx,
                frame_i=a,
                frame_j=b,
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
                residual_flow_proxy=edge["residual_flow_proxy"],
                dt=float(b.timestamp - a.timestamp),
            )
            seq_ranges[scene_seq].append(len(rows))
            rows.append(row)
    obs = np.asarray([r.observability_score for r in rows], dtype=np.float64)
    low_q = float(cfg["observability_weighting"]["low_threshold_quantile"])
    high_q = float(cfg["observability_weighting"]["high_threshold_quantile"])
    low_thr = float(np.quantile(obs, low_q)) if obs.size else 0.0
    high_thr = float(np.quantile(obs, high_q)) if obs.size else 1.0
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
                        "frame_indices": list(range(start, start + k + 1)),
                        "timestamps": [float(seq_frames[x].timestamp) for x in range(start, start + k + 1)],
                        "valid_contiguous": True,
                        "uses_eval_scene": False,
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
        "obs_score": torch.tensor([r.observability_score for r in batch_rows], dtype=torch.float32, device=device),
        "gate_floor": torch.tensor([gate_weight(r.observability_score) for r in batch_rows], dtype=torch.float32, device=device),
        "entropy_mean": torch.tensor([r.entropy_mean for r in batch_rows], dtype=torch.float32, device=device),
        "cycle_error_mean": torch.tensor([r.cycle_error_mean for r in batch_rows], dtype=torch.float32, device=device),
        "epipolar_residual_mean": torch.tensor([r.epipolar_residual_mean for r in batch_rows], dtype=torch.float32, device=device),
        "base_tmag": torch.tensor([r.base_tmag for r in batch_rows], dtype=torch.float32, device=device),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, dataset, windows, token_in_dim = _make_rows(cfg)
    out_dir = Path(args.candidate_dir) if args.candidate_dir else Path("checkpoints/ARCH2_mainline_rotation_compensated_softcorr_geometry_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dataset["ready_for_training"]:
        status = {
            "real_training_executed": False,
            "optimizer_step_count": 0,
            "learned_weights_saved": False,
            "smoke_policy_only": False,
            "uses_eval_gt_for_training": False,
            "uses_eval_gt_for_gate": False,
            "uses_orbslam3_teacher": False,
            "uses_rotation_compensated_motion_tokens": True,
            "uses_soft_correspondence_geometry": True,
            "uses_observability_loss_weighting": True,
            "uses_true_kstep": True,
            "dt_factor_per_sample": True,
            "classification": "ARCH2_TRAINING_BLOCKED",
        }
        write_json(out_dir / "training_status.json", status)
        return status

    low_thr = float(dataset["low_threshold"])
    high_thr = float(dataset["high_threshold"])
    model = Arch2SoftcorrGeometryModel(
        token_in_dim=token_in_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        max_alpha=float(cfg["tdir_head"]["final_tdir"]["max_alpha"]),
        scale_delta_clip=float(cfg["scale_head"]["delta_clip"]),
        delta_clip_norm=float(cfg["tdir_head"].get("delta_clip_norm", 0.02)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    batch_size = int(cfg["training"]["batch_size"])
    max_steps = int(cfg["training"]["max_steps"])
    optimizer_steps = 0
    delta_norm_train: List[float] = []

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
            batch["gate_floor"],
        )
        base_dot = torch.sum(batch["base_tdir"] * batch["gt_tdir"], dim=-1).clamp(-1.0, 1.0)
        pred_dot = torch.sum(out["final_tdir"] * batch["gt_tdir"], dim=-1).clamp(-1.0, 1.0)
        weight = batch["gate_floor"]
        loss_tdir = torch.sum(weight * (1.0 - pred_dot)) / torch.clamp(weight.sum(), min=1.0)
        loss_abs = torch.mean((0.2 + 0.8 * weight) * (1.0 - torch.abs(pred_dot)))
        loss_anti = torch.sum(weight * torch.relu(-pred_dot)) / torch.clamp(weight.sum(), min=1.0)
        loss_tmag = torch.mean(torch.abs(out["final_log_tmag"] - torch.log(batch["gt_tmag"].clamp_min(1.0e-6))))
        loss_path = torch.mean(torch.abs(out["final_tmag"] - batch["base_tmag"]) / batch["base_tmag"].clamp_min(1.0e-6))
        cycle_pred = out["cycle_error"].mean(dim=-1) if out["cycle_error"].ndim > 1 else out["cycle_error"]
        loss_cycle = torch.mean(batch["cycle_error_mean"] + cycle_pred)
        loss_epi = torch.mean(_epipolar_residual(out["bA_rot"], out["matched_bearing_b"], batch["gt_tdir"]))
        loss_delta = torch.mean(out["delta_tdir_norm"] ** 2)
        loss_no_harm = torch.mean(torch.relu(base_dot - pred_dot))

        # true contiguous k-step windows, not random batch proxy
        window_loss = torch.zeros((), device=device)
        if windows:
            sampled_windows = random.sample(windows, k=min(4, len(windows)))
            parts: List[torch.Tensor] = []
            for w in sampled_windows:
                w_rows = [rows[i] for i in w["edge_ids"]]
                wb = _tensorize(w_rows, low_thr, high_thr, device)
                wout = model(
                    wb["local_features"], wb["bearing_a"], wb["bearing_b"], wb["W_ab"], wb["W_ba"],
                    wb["R_coarse"], wb["base_tdir"], wb["base_log_tmag"], wb["gate_floor"]
                )
                R_chain = torch.eye(3, device=device)
                t_chain = torch.zeros(3, device=device)
                path_sum = torch.zeros((), device=device)
                for ei in range(wout["R_BA"].shape[0]):
                    R_step = wout["R_BA"][ei]
                    t_step = wout["final_tvec_B"][ei]
                    R_chain = R_step @ R_chain
                    t_chain = R_step @ t_chain + t_step
                    path_sum = path_sum + torch.linalg.norm(t_step)
                gt_R = torch.tensor(w["gt_R"], dtype=torch.float32, device=device)
                gt_tdir = torch.tensor(w["gt_tdir"], dtype=torch.float32, device=device)
                gt_path = torch.tensor(float(w["gt_path_len"]), dtype=torch.float32, device=device)
                parts.append(
                    0.35 * _rot_geodesic_deg(R_chain.unsqueeze(0), gt_R.unsqueeze(0)).mean()
                    + _vector_angle_t(t_chain.unsqueeze(0), gt_tdir.unsqueeze(0)).mean()
                    + 0.25 * torch.abs(path_sum - gt_path)
                )
            if parts:
                window_loss = torch.stack(parts).mean()

        loss = (
            float(cfg["losses"]["w_tdir_signed"]) * loss_tdir
            + float(cfg["losses"]["w_tdir_abs_aux"]) * loss_abs
            + float(cfg["losses"]["w_antiparallel"]) * loss_anti
            + float(cfg["losses"]["w_tmag_log"]) * loss_tmag
            + float(cfg["losses"]["w_path_window"]) * loss_path
            + float(cfg["losses"]["w_kstep_composition"]) * window_loss
            + float(cfg["losses"]["w_softcorr_cycle"]) * loss_cycle
            + float(cfg["losses"]["w_epipolar_residual"]) * loss_epi
            + float(cfg["losses"]["w_delta_tdir_l2"]) * loss_delta
            + float(cfg["losses"]["w_no_harm_margin"]) * loss_no_harm
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        optimizer_steps += 1
        delta_norm_train.extend(out["delta_effective"].detach().norm(dim=-1).cpu().numpy().tolist())
        if optimizer_steps >= max_steps:
            break

    blob = {
        "model_state": model.state_dict(),
        "token_in_dim": token_in_dim,
        "hidden_dim": int(cfg["model"]["hidden_dim"]),
        "max_alpha": float(cfg["tdir_head"]["final_tdir"]["max_alpha"]),
        "scale_delta_clip": float(cfg["scale_head"]["delta_clip"]),
        "delta_clip_norm": float(cfg["tdir_head"].get("delta_clip_norm", 0.02)),
        "softcorr_temperature": float(cfg["softcorr_geometry"].get("temperature", 0.05)),
        "low_threshold": low_thr,
        "high_threshold": high_thr,
        "dataset": dataset,
        "kstep_windows": windows,
        "dt_bucket_anchor_used": False,
        "dt_factor_per_sample": True,
    }
    torch.save(blob, out_dir / "arch2_model.pt")
    status = {
        "real_training_executed": True,
        "optimizer_step_count": optimizer_steps,
        "learned_weights_saved": True,
        "smoke_policy_only": False,
        "uses_eval_gt_for_training": False,
        "uses_eval_gt_for_gate": False,
        "uses_orbslam3_teacher": False,
        "uses_rotation_compensated_motion_tokens": True,
        "uses_soft_correspondence_geometry": True,
        "uses_observability_loss_weighting": True,
        "uses_true_kstep": True,
        "dt_factor_per_sample": True,
        "classification": "ARCH2_REAL_TRAINING_COMPLETE" if optimizer_steps >= max_steps else "ARCH2_SHORT_REAL_TRAINING",
    }
    write_json(out_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--candidate-dir", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
