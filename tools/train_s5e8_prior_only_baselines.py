#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import angle_deg_from_rot, pair_features, read_timestamps, read_tum, repo_rel, validation_from_logs, vector_angle_deg, write_json
from s5e8_translation_geometry_lib import (
    NumericOnlyDirectionModel,
    build_eval_seq_frames,
    build_numeric_features,
    compose_direction,
    load_npz_model,
    load_direction_checkpoint,
    make_pair_samples_with_buckets,
    predict_s5e2,
    rotvec_to_matrix,
    save_direction_checkpoint,
)


OUT_DIR = Path("checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
TIMESTAMPS = Path("external_baselines/dataset/scene01_seq03/timestamps.txt")
GT_PATH = Path("external_baselines/dataset/scene01_seq03/groundtruth_tum.txt")


class NumericDataset(Dataset):
    def __init__(self, samples, s5e2, s5e3, bucket_stats):
        self.samples = samples
        self.s5e2 = s5e2
        self.s5e3 = s5e3
        self.bucket_stats = bucket_stats
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        numeric = build_numeric_features(s, self.cache, self.s5e2, self.s5e3, self.bucket_stats)
        return {
            "numeric": torch.from_numpy(numeric.astype(np.float32)),
            "prior_dir": torch.from_numpy(numeric[:3].astype(np.float32)),
            "gt_direction": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "bucket_index": torch.tensor(s.bucket_index, dtype=torch.long),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name == "small_motion" else 0.0, dtype=torch.float32),
        }


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    pn, gn = float(np.linalg.norm(t)), float(np.linalg.norm(tg))
    tdir = vector_angle_deg(t, tg, absolute=False)
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        "rot_deg": angle_deg_from_rot(R @ Rg.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tdir_cosine": cos,
        "anti_parallel_flag": bool(cos is not None and cos < 0),
        "tmag_ratio": pn / gn if gn > 1.0e-12 else None,
        "pred_step_length": pn,
        "gt_step_length": gn,
    }


def _summary(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _pct(key, q):
        vals = np.asarray([m.get(key) for m in metrics_rows if m.get(key) is not None], dtype=np.float64)
        return None if vals.size == 0 else float(np.percentile(vals, q))
    def _mean(key):
        vals = np.asarray([m.get(key) for m in metrics_rows if m.get(key) is not None], dtype=np.float64)
        return None if vals.size == 0 else float(np.mean(vals))
    pred_path = sum(float(m.get("pred_step_length") or 0.0) for m in metrics_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0.0) for m in metrics_rows)
    return {
        "signed_tdir_mean": _mean("tdir_deg"),
        "tdir_abs_mean": _mean("tdir_abs_deg"),
        "tdir_abs_p90": _pct("tdir_abs_deg", 90),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
        "tmag_median": _pct("tmag_ratio", 50),
        "tmag_p95": _pct("tmag_ratio", 95),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }


def _train_numeric_only(train_ds, val_ds, bucket_stats) -> Path:
    device = torch.device("cpu")
    model = NumericOnlyDirectionModel(numeric_dim=len(train_ds[0]["numeric"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    best = float("inf")
    path = OUT_DIR / "s5e8_numeric_only_best.pt"
    for _epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["numeric"])
            pred = compose_direction(batch["prior_dir"], out)
            dot = (pred * batch["gt_direction"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            loss = (1.0 - dot).mean() + 0.2 * (1.0 - abs_dot).mean() + 0.3 * torch.relu(-dot).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["numeric"])
                pred = compose_direction(batch["prior_dir"], out)
                dot = (pred * batch["gt_direction"]).sum(dim=1).clamp(-1.0, 1.0)
                vals.extend(torch.rad2deg(torch.arccos(dot)).cpu().tolist())
        score = float(np.mean(vals))
        if score < best:
            best = score
            save_direction_checkpoint(path, model, (64, 128), len(train_ds[0]["numeric"]), bucket_stats, "numeric_only_direction")
    return path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    train_samples, val_samples, _split_summary, bucket_stats = make_pair_samples_with_buckets()
    train_ds = NumericDataset(train_samples, s5e2, s5e3, bucket_stats)
    val_ds = NumericDataset(val_samples, s5e2, s5e3, bucket_stats)
    numeric_ckpt = _train_numeric_only(train_ds, val_ds, bucket_stats)

    timestamps = [float(x.strip()) for x in TIMESTAMPS.read_text(encoding="utf-8").splitlines() if x.strip()]
    gt = read_tum(GT_PATH)
    frames = build_eval_seq_frames("scene01", "seq03")
    numeric_model, _img_size, _num_dim, bucket_stats, _type = load_direction_checkpoint(numeric_ckpt, torch.device("cpu"))
    rows = []
    cache: Dict[str, np.ndarray] = {}
    for i in range(len(timestamps) - 1):
        fi = min(frames, key=lambda r: abs(float(r.timestamp) - float(timestamps[i])))
        fj = min(frames, key=lambda r: abs(float(r.timestamp) - float(timestamps[i + 1])))
        class S: pass
        s = S()
        s.frame_i = fi
        s.frame_j = fj
        s.edge_index = i
        s.total_edges = len(timestamps) - 1
        s.bucket_name = "normal_motion"
        s.bucket_index = 1
        x_pair = pair_features(fi, fj, i, len(timestamps) - 1, cache)
        base = predict_s5e2(s5e2, x_pair)
        R = rotvec_to_matrix(base[:3])
        prior_dir = base[3:6]
        prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
        numeric = build_numeric_features(s, cache, s5e2, s5e3, bucket_stats)
        with torch.no_grad():
            out = numeric_model(torch.from_numpy(numeric.astype(np.float32)).unsqueeze(0))
            pred_dir = compose_direction(torch.from_numpy(prior_dir.astype(np.float32)).unsqueeze(0), out).view(-1).cpu().numpy()
        gt_rel = _gt_rel(gt, timestamps[i], timestamps[i + 1])
        bucket = bucket_stats["normal_motion"]
        baselines = {
            "train_median_tmag_previous_direction": (prior_dir, float(bucket_stats["global"]["median"])),
            "motion_bucket_tmag_prior": (prior_dir, float(bucket["median"])),
            "constant_conservative_tmag_prior": (prior_dir, float(bucket_stats["global"]["clip_lo"])),
            "numeric_prior_only_model": (pred_dir, float(bucket["median"])),
        }
        row = {"edge_index": i, "timestamp_i": timestamps[i], "timestamp_j": timestamps[i + 1], "baselines": {}}
        for name, (direction, mag) in baselines.items():
            row["baselines"][name] = _metric(R, np.asarray(direction) * mag, gt_rel)
        rows.append(row)

    summary = {}
    for name in rows[0]["baselines"].keys():
        summary[name] = _summary([r["baselines"][name] for r in rows])
    payload = {
        "experiment": "S5E8_translation_geometry_diagnostic_ablation",
        "numeric_only_checkpoint": str(numeric_ckpt),
        "prior_only_baselines": summary,
        "notes": [
            "prior-only baseline 用于检查 learned model 是否真的超越数值先验。",
            "numeric-only control 用于检查图像特征是否提供了额外几何信息。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "prior_only_baselines.json", payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
