#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e8_translation_geometry_lib import (
    DirectionOnlyCandidate,
    build_numeric_features,
    compose_direction,
    load_npz_model,
    load_ordered_image_pair,
    make_pair_samples_with_buckets,
    save_direction_checkpoint,
)


OUT_DIR = Path("checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


class DirectionDataset(Dataset):
    def __init__(self, samples, image_size, s5e2, s5e3, bucket_stats):
        self.samples = samples
        self.image_size = image_size
        self.s5e2 = s5e2
        self.s5e3 = s5e3
        self.bucket_stats = bucket_stats
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = load_ordered_image_pair(s.frame_i, s.frame_j, self.image_size)
        numeric = build_numeric_features(s, self.cache, self.s5e2, self.s5e3, self.bucket_stats)
        prior_dir = numeric[:3]
        return {
            "image": image,
            "numeric": torch.from_numpy(numeric.astype(np.float32)),
            "prior_dir": torch.from_numpy(prior_dir.astype(np.float32)),
            "gt_direction": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "bucket_index": torch.tensor(s.bucket_index, dtype=torch.long),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name == "small_motion" else 0.0, dtype=torch.float32),
        }


def _loss(pred_dir, gt_dir, small_mask):
    dot = (pred_dir * gt_dir).sum(dim=1).clamp(-1.0, 1.0)
    abs_dot = dot.abs()
    signed = 1.0 - dot
    anti_soft = torch.relu(-dot)
    anti_hard = torch.relu(-dot - 0.5)
    tdir_abs = 1.0 - abs_dot
    weight = torch.where(small_mask > 0.5, torch.full_like(signed, 0.8), torch.ones_like(signed))
    loss = (signed * weight).mean() + 0.25 * tdir_abs.mean() + 0.35 * anti_soft.mean() + 0.7 * anti_hard.mean()
    return loss


def _eval(model, loader, device):
    signed, abs_signed, anti, severe, goodbad = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["image"], batch["numeric"])
            pred = compose_direction(batch["prior_dir"], out)
            dot = (pred * batch["gt_direction"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            ang = torch.rad2deg(torch.arccos(dot))
            ang_abs = torch.rad2deg(torch.arccos(abs_dot))
            signed.extend(ang.cpu().tolist())
            abs_signed.extend(ang_abs.cpu().tolist())
            anti.extend((dot < 0).float().cpu().tolist())
            severe.extend((ang > 120).float().cpu().tolist())
            goodbad.extend(((ang > 120) & (ang_abs < 45)).float().cpu().tolist())
    return {
        "signed_tdir_mean": float(np.mean(signed)),
        "signed_tdir_median": float(np.percentile(signed, 50)),
        "signed_tdir_p90": float(np.percentile(signed, 90)),
        "tdir_abs_mean": float(np.mean(abs_signed)),
        "tdir_abs_median": float(np.percentile(abs_signed, 50)),
        "tdir_abs_p90": float(np.percentile(abs_signed, 90)),
        "tdir_mean_cosine": float(np.mean(np.cos(np.deg2rad(np.asarray(signed))))),
        "anti_parallel_rate": float(np.mean(anti)),
        "severe_wrong_sign_rate": float(np.mean(severe)),
        "direction_abs_good_but_signed_bad_rate": float(np.mean(goodbad)),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    train_samples, val_samples, split_summary, bucket_stats = make_pair_samples_with_buckets()
    image_size = (64, 128)
    train_ds = DirectionDataset(train_samples, image_size, s5e2, s5e3, bucket_stats)
    val_ds = DirectionDataset(val_samples, image_size, s5e2, s5e3, bucket_stats)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    numeric_dim = len(train_ds[0]["numeric"])
    model = DirectionOnlyCandidate(numeric_dim=numeric_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_path = OUT_DIR / "s5e8_direction_only_best.pt"
    history: List[Dict[str, object]] = []
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["image"], batch["numeric"])
            pred = compose_direction(batch["prior_dir"], out)
            loss = _loss(pred, batch["gt_direction"], batch["small_motion_mask"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_metrics = _eval(model, val_loader, device)
        history.append({"epoch": epoch, "val_metrics": val_metrics})
        score = val_metrics["tdir_abs_mean"] + 0.5 * val_metrics["signed_tdir_mean"] + 40.0 * val_metrics["anti_parallel_rate"]
        if score < best_score:
            best_score = score
            save_direction_checkpoint(best_path, model, image_size, numeric_dim, bucket_stats, "direction_only_candidate")
    status = {
        "attempted": True,
        "classification": "S5E8_DIRECTION_ONLY_TRAINING_SMOKE",
        "direction_only_checkpoint": str(best_path),
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "history": history,
        "split_summary": split_summary,
        "notes": [
            "S5E8 direction-only candidate 不学习 raw tmag，只优化 direction。",
            "此实验用于检查 scale 学习是否干扰 direction 学习。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "direction_only_training_status.json", status)
    ckpt = {
        "experiment": "S5E8_translation_geometry_diagnostic_ablation",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "training": {
            "attempted": True,
            "classification": status["classification"],
            "config": "configs/s5e8_translation_geometry_diagnostic_ablation.yaml",
            "checkpoint_dir": str(OUT_DIR),
            "uses_scene01_seq03_for_training": False,
            "direction_only_checkpoint": str(best_path),
            "notes": status["notes"],
        },
        "direction_only_candidate": status,
        "prior_only_baselines": {},
        "numeric_only_control": {},
        "translation_supervision_quality": {},
        "motion_bucket_audit": {},
        "adjacent_dense_export": {"available": False, "coverage": None, "all_edges_traceable": False},
        "raw_prediction_metrics": {},
        "guarded_prediction_metrics": {},
        "raw_vs_guarded_gap": {},
        "external_eval": {},
        "raw_external_eval": {},
        "oracle_ablation": {},
        "comparison_summary": {},
        "comparison_to_orbslam3": {},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e8": True},
        "validation": validation_from_logs(),
        "allowed_final_classifications": [
            "S5E8_DIRECTION_SUPERVISION_LIMITED",
            "S5E8_SCALE_SUPERVISION_LIMITED",
            "S5E8_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E8_SMALL_MOTION_DOMINATED",
            "S5E8_DIRECTION_ONLY_IMPROVED",
            "S5E8_RAW_GEOMETRY_IMPROVED",
            "S5E8_GUARD_DEPENDENT",
            "S5E8_NO_IMPROVEMENT",
            "S5E8_REGRESSION",
        ],
        "final_classification": "S5E8_NO_IMPROVEMENT",
    }
    write_json(OUT_DIR.parent / "S5E8_translation_geometry_diagnostic_ablation_candidate.json", ckpt)
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
