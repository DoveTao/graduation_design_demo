#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e8_translation_geometry_lib import DirectionOnlyCandidate, compose_direction
from s5e9_small_motion_geometry_lib import make_samples
from s5e7_direction_scale_lib import load_npz_model, load_ordered_image_pair, build_numeric_features


OUT_DIR = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


class DirectionOrderDataset(Dataset):
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
        img_f = load_ordered_image_pair(s.frame_i, s.frame_j, self.image_size)
        img_r = load_ordered_image_pair(s.frame_j, s.frame_i, self.image_size)
        num_f = build_numeric_features(s, self.cache, self.s5e2, self.s5e3, self.bucket_stats)
        class RS: pass
        rs = RS()
        rs.frame_i = s.frame_j
        rs.frame_j = s.frame_i
        rs.edge_index = s.edge_index
        rs.total_edges = s.total_edges
        rs.bucket_name = s.bucket_name
        rs.bucket_index = s.bucket_index
        num_r = build_numeric_features(rs, self.cache, self.s5e2, self.s5e3, self.bucket_stats)
        return {
            "img_f": img_f,
            "img_r": img_r,
            "num_f": torch.from_numpy(num_f.astype(np.float32)),
            "num_r": torch.from_numpy(num_r.astype(np.float32)),
            "prior_f": torch.from_numpy(num_f[:3].astype(np.float32)),
            "prior_r": torch.from_numpy(num_r[:3].astype(np.float32)),
            "gt_f": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "gt_r": torch.from_numpy((-s.gt_direction).astype(np.float32)),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name in ("near_static", "small_motion") else 0.0, dtype=torch.float32),
        }


def _loss(pred_f, pred_r, gt_f, gt_r, small):
    dot_f = (pred_f * gt_f).sum(dim=1).clamp(-1.0, 1.0)
    dot_r = (pred_r * gt_r).sum(dim=1).clamp(-1.0, 1.0)
    order_dot = (pred_f * (-pred_r)).sum(dim=1).clamp(-1.0, 1.0)
    weight = torch.where(small > 0.5, torch.full_like(dot_f, 0.6), torch.ones_like(dot_f))
    loss_dir = ((1.0 - dot_f) * weight).mean() + ((1.0 - dot_r) * weight).mean()
    loss_order = (1.0 - order_dot).mean()
    anti = torch.relu(-dot_f).mean() + torch.relu(-dot_r).mean()
    return loss_dir + 0.25 * loss_order + 0.35 * anti


def _eval(model, loader, device):
    signed, abs_signed, anti, severe, order = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_f = model(batch["img_f"], batch["num_f"])
            out_r = model(batch["img_r"], batch["num_r"])
            pred_f = compose_direction(batch["prior_f"], out_f)
            pred_r = compose_direction(batch["prior_r"], out_r)
            dot = (pred_f * batch["gt_f"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            order_dot = (pred_f * (-pred_r)).sum(dim=1).clamp(-1.0, 1.0)
            ang = torch.rad2deg(torch.arccos(dot))
            ang_abs = torch.rad2deg(torch.arccos(abs_dot))
            signed.extend(ang.cpu().tolist())
            abs_signed.extend(ang_abs.cpu().tolist())
            anti.extend((dot < 0).float().cpu().tolist())
            severe.extend((ang > 120).float().cpu().tolist())
            order.extend(order_dot.cpu().tolist())
    order_np = np.asarray(order, dtype=np.float64)
    return {
        "signed_tdir_mean": float(np.mean(signed)),
        "signed_tdir_median": float(np.percentile(signed, 50)),
        "signed_tdir_p90": float(np.percentile(signed, 90)),
        "tdir_abs_mean": float(np.mean(abs_signed)),
        "tdir_abs_median": float(np.percentile(abs_signed, 50)),
        "tdir_abs_p90": float(np.percentile(abs_signed, 90)),
        "anti_parallel_rate": float(np.mean(anti)),
        "severe_wrong_sign_rate": float(np.mean(severe)),
        "pair_order_dir_flip_cosine_mean": float(np.mean(order_np)),
        "pair_order_dir_flip_success_rate": float(np.mean(order_np > 0.5)),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    train_samples, val_samples, split_summary, bucket_stats = make_samples()
    image_size = (64, 128)
    train_ds = DirectionOrderDataset(train_samples, image_size, s5e2, s5e3, bucket_stats)
    val_ds = DirectionOrderDataset(val_samples, image_size, s5e2, s5e3, bucket_stats)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    model = DirectionOnlyCandidate(numeric_dim=len(train_ds[0]["num_f"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_path = OUT_DIR / "s5e9_signed_direction_best.pt"
    history: List[Dict[str, object]] = []
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_f = model(batch["img_f"], batch["num_f"])
            out_r = model(batch["img_r"], batch["num_r"])
            pred_f = compose_direction(batch["prior_f"], out_f)
            pred_r = compose_direction(batch["prior_r"], out_r)
            loss = _loss(pred_f, pred_r, batch["gt_f"], batch["gt_r"], batch["small_motion_mask"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_metrics = _eval(model, val_loader, device)
        history.append({"epoch": epoch, "val_metrics": val_metrics})
        score = val_metrics["signed_tdir_mean"] + 0.5 * val_metrics["tdir_abs_mean"] + 40.0 * val_metrics["anti_parallel_rate"] - 5.0 * val_metrics["pair_order_dir_flip_success_rate"]
        if score < best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "numeric_dim": len(train_ds[0]["num_f"]), "image_size": list(image_size), "bucket_stats": bucket_stats, "model_type": "signed_direction_order_candidate"}, best_path)

    status = {
        "attempted": True,
        "classification": "S5E9_SIGNED_DIRECTION_TRAINING_SMOKE",
        "checkpoint": str(best_path),
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "history": history,
        "split_summary": split_summary,
        "notes": [
            "S5E9 signed-direction candidate 加入 pair-order anti-symmetry 约束。",
            "near_static 与 small_motion 降低 signed direction 权重。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "signed_direction_training_status.json", status)
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
