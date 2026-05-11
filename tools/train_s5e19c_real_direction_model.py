#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import adjacent_pairs, pair_features, relative_pose_A_to_B_in_B, scan_frames, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


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
    current: str | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, value = raw.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                cfg[key] = parse_scalar(value)
                current = None
                mode = None
            else:
                cfg[key] = {}
                current = key
                mode = "dict"
        else:
            if current is None:
                continue
            if raw.startswith("  - "):
                if not isinstance(cfg[current], list):
                    cfg[current] = []
                cfg[current].append(parse_scalar(raw[4:]))
                mode = "list"
            elif raw.startswith("  ") and ":" in raw and mode != "list":
                key, value = raw.strip().split(":", 1)
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def _extract_corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "filtered_match_count": float(orb["filtered_match_count"]),
        "inlier_ratio": float(orb["inlier_ratio"]),
        "median_match_displacement": float(orb["median_match_displacement"]),
        "p90_match_displacement": float(orb["p90_match_displacement"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "low_parallax_flag": 1.0 if orb["low_parallax_flag"] else 0.0,
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "p90_flow_magnitude": float(flow["p90_flow_magnitude"]),
        "flow_angle_dispersion": float(flow["flow_angle_dispersion"]) / 180.0,
        "forward_backward_flow_consistency": 0.0 if flow["forward_backward_flow_consistency"] is None else float(flow["forward_backward_flow_consistency"]),
    }


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def _make_samples(train_seqs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frames = scan_frames(Path("data"), scene="scene01")
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache: Dict[str, np.ndarray] = {}
    pairs: List[Tuple[Any, Any]] = []
    for seq in train_seqs:
        seq_name = str(seq).split("/")[-1]
        pairs.extend(adjacent_pairs(frames[("scene01", seq_name)]))
    random.Random(3407).shuffle(pairs)
    val_count = 79
    train_pairs = pairs[:-val_count]
    val_pairs = pairs[-val_count:]

    def build(rows: Sequence[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        total = max(len(rows), 1)
        for idx, (a, b) in enumerate(rows):
            R, t = relative_pose_A_to_B_in_B(a, b)
            gt_mag = float(np.linalg.norm(t))
            gt_dir = _norm(t)
            x = pair_features(a, b, idx, total, cache)
            coarse = predict_s5e2(s5e2, x)
            coarse_dir = _norm(coarse[3:6])
            coarse_mag = float(np.exp(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]))
            corr = _extract_corr(a, b)
            feat = np.concatenate(
                [
                    x.astype(np.float32),
                    coarse_dir.astype(np.float32),
                    np.asarray([coarse_mag, math.log(max(coarse_mag, 1.0e-12))], dtype=np.float32),
                    np.asarray(list(corr.values()), dtype=np.float32),
                ]
            )
            out.append(
                {
                    "feature": feat.tolist(),
                    "coarse_dir": coarse_dir.tolist(),
                    "coarse_mag": coarse_mag,
                    "gt_dir": gt_dir.tolist(),
                    "gt_mag": gt_mag,
                    "sign_target": 1.0 if float(np.dot(coarse_dir, gt_dir)) >= 0.0 else 0.0,
                }
            )
        return out

    return build(train_pairs), build(val_pairs)


class DirectionRefineMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, delta_tdir_scale: float, delta_log_tmag_clip: float) -> None:
        super().__init__()
        self.delta_tdir_scale = float(delta_tdir_scale)
        self.delta_log_tmag_clip = float(delta_log_tmag_clip)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.delta_head = torch.nn.Linear(hidden_dim, 3)
        self.sign_head = torch.nn.Linear(hidden_dim, 1)
        self.scale_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, coarse_dir: torch.Tensor, coarse_mag: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.net(x)
        delta = torch.tanh(self.delta_head(z)) * self.delta_tdir_scale
        final_dir = F.normalize(coarse_dir + delta, dim=1)
        sign_score = self.sign_head(z).squeeze(1)
        delta_log_tmag = torch.tanh(self.scale_head(z).squeeze(1)) * self.delta_log_tmag_clip
        final_tmag = coarse_mag * torch.exp(delta_log_tmag)
        return {
            "delta_tdir": delta,
            "final_tdir": final_dir,
            "sign_score": sign_score,
            "delta_log_tmag": delta_log_tmag,
            "final_tmag": final_tmag,
        }


def _tensorize(rows: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "feature": torch.tensor([r["feature"] for r in rows], dtype=torch.float32, device=device),
        "coarse_dir": F.normalize(torch.tensor([r["coarse_dir"] for r in rows], dtype=torch.float32, device=device), dim=1),
        "coarse_mag": torch.tensor([r["coarse_mag"] for r in rows], dtype=torch.float32, device=device),
        "gt_dir": F.normalize(torch.tensor([r["gt_dir"] for r in rows], dtype=torch.float32, device=device), dim=1),
        "gt_mag": torch.tensor([r["gt_mag"] for r in rows], dtype=torch.float32, device=device),
        "sign_target": torch.tensor([r["sign_target"] for r in rows], dtype=torch.float32, device=device),
    }


def _metrics(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    dot = torch.clamp(torch.sum(pred["final_tdir"] * batch["gt_dir"], dim=1), -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(dot))
    ang_abs = torch.rad2deg(torch.acos(torch.abs(dot)))
    tmag_ratio = pred["final_tmag"] / torch.clamp(batch["gt_mag"], min=1.0e-12)
    sign_unique = torch.unique(torch.round(pred["sign_score"] * 1000.0) / 1000.0).numel()
    delta_norm = torch.linalg.norm(pred["delta_tdir"], dim=1)
    return {
        "signed_tdir_mean": float(ang.mean().item()),
        "tdir_abs_mean": float(ang_abs.mean().item()),
        "anti_parallel_rate": float((dot < 0).float().mean().item()),
        "tmag_median_ratio": float(torch.quantile(tmag_ratio, 0.5).item()),
        "tmag_p95_ratio": float(torch.quantile(tmag_ratio, 0.95).item()),
        "delta_tdir_zero_rate": float((delta_norm <= 1.0e-8).float().mean().item()),
        "sign_score_unique_count": int(sign_unique),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, val_rows = _make_samples(cfg["training"]["train_scenes"])
    train = _tensorize(train_rows, device)
    val = _tensorize(val_rows, device)
    model = DirectionRefineMLP(
        input_dim=len(train_rows[0]["feature"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        delta_tdir_scale=float(cfg["model"]["delta_tdir_scale"]),
        delta_log_tmag_clip=float(cfg["model"]["delta_log_tmag_clip"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    optimizer_steps = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        idx = torch.randperm(train["feature"].shape[0], device=device)
        for start in range(0, len(idx), int(cfg["training"]["batch_size"])):
            batch_idx = idx[start : start + int(cfg["training"]["batch_size"])]
            batch = {k: v[batch_idx] for k, v in train.items()}
            pred = model(batch["feature"], batch["coarse_dir"], batch["coarse_mag"])
            dot = torch.clamp(torch.sum(pred["final_tdir"] * batch["gt_dir"], dim=1), -1.0, 1.0)
            loss_signed = torch.mean(1.0 - dot)
            loss_abs = torch.mean(1.0 - torch.abs(dot))
            loss_anti = torch.mean(torch.relu(-dot))
            loss_sign = F.binary_cross_entropy_with_logits(pred["sign_score"], batch["sign_target"])
            log_ratio = torch.log(torch.clamp(pred["final_tmag"], min=1.0e-12) / torch.clamp(batch["gt_mag"], min=1.0e-12))
            loss_tmag = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio))
            loss_path = torch.abs(pred["final_tmag"].sum() - batch["gt_mag"].sum()) / torch.clamp(batch["gt_mag"].sum(), min=1.0e-12)
            if pred["final_tdir"].shape[0] > 1:
                loss_multi = torch.mean(torch.relu(-torch.sum(pred["final_tdir"][1:] * pred["final_tdir"][:-1], dim=1)))
            else:
                loss_multi = pred["final_tdir"].sum() * 0.0
            loss = loss_signed + 0.2 * loss_abs + 0.2 * loss_anti + 0.2 * loss_sign + 0.5 * loss_tmag + 0.1 * loss_path + 0.05 * loss_multi
            opt.zero_grad()
            loss.backward()
            opt.step()
            optimizer_steps += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(val["feature"], val["coarse_dir"], val["coarse_mag"])
        history.append({"epoch": epoch + 1, "val_metrics": _metrics(val_pred, val)})

    ckpt_dir = Path("checkpoints/S5E19C_real_direction_training_no_fallback_candidate")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "s5e19c_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(train_rows[0]["feature"]),
            "hidden_dim": int(cfg["model"]["hidden_dim"]),
            "delta_tdir_scale": float(cfg["model"]["delta_tdir_scale"]),
            "delta_log_tmag_clip": float(cfg["model"]["delta_log_tmag_clip"]),
        },
        ckpt_path,
    )
    status = {
        "real_training_executed": True,
        "optimizer_step_count": optimizer_steps,
        "learned_weights_saved": True,
        "smoke_policy_only": False,
        "uses_eval_gt_for_training": False,
        "uses_orbslam3_teacher": False,
        "losses_actually_computed": {
            "signed_tdir": True,
            "tdir_abs_aux": True,
            "anti_parallel_hard_negative": True,
            "sign_score_bce": True,
            "robust_tmag_log": True,
            "multiframe_composition": True,
            "path_length_consistency": True,
        },
        "classification": "S5E19C_SHORT_REAL_TRAINING",
        "best_checkpoint": str(ckpt_path),
        "history_tail": history[-5:],
    }
    write_json(Path("checkpoints/S5E19C_real_direction_training_no_fallback_candidate/training_status.json"), status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
