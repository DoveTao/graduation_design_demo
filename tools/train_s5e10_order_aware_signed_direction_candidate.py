#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e10_order_geometry_lib import (
    OrderAwareSignedDirectionModel,
    architecture_audit,
    make_samples,
    ordered_feature_vector,
    save_order_direction_checkpoint,
)


OUT_DIR = Path("checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate")


class OrderDataset(Dataset):
    def __init__(self, samples, bucket_stats):
        self.samples = samples
        self.bucket_stats = bucket_stats
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat_f = ordered_feature_vector(s, self.cache, self.bucket_stats)
        class RS: pass
        rs = RS()
        rs.frame_i = s.frame_j
        rs.frame_j = s.frame_i
        rs.edge_index = s.edge_index
        rs.total_edges = s.total_edges
        rs.bucket_name = s.bucket_name
        rs.bucket_index = s.bucket_index
        feat_r = ordered_feature_vector(rs, self.cache, self.bucket_stats)
        return {
            "feat_f": torch.from_numpy(feat_f.astype(np.float32)),
            "feat_r": torch.from_numpy(feat_r.astype(np.float32)),
            "gt_f": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "gt_r": torch.from_numpy((-s.gt_direction).astype(np.float32)),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name in ("near_static", "small_motion") else 0.0, dtype=torch.float32),
        }


def _loss(pred_f, pred_r, batch):
    dot_f = (pred_f["pred_dir_unit"] * batch["gt_f"]).sum(dim=1).clamp(-1.0, 1.0)
    dot_r = (pred_r["pred_dir_unit"] * batch["gt_r"]).sum(dim=1).clamp(-1.0, 1.0)
    axis_f = (pred_f["pred_axis_unit"] * batch["gt_f"]).sum(dim=1).abs().clamp(0.0, 1.0)
    axis_r = (pred_r["pred_axis_unit"] * batch["gt_r"]).sum(dim=1).abs().clamp(0.0, 1.0)
    order_dot = (pred_f["pred_dir_unit"] * (-pred_r["pred_dir_unit"])).sum(dim=1).clamp(-1.0, 1.0)
    sign_target_f = ((batch["gt_f"][:, 0:1] + batch["gt_f"][:, 1:2] + batch["gt_f"][:, 2:3]) >= 0).float()
    sign_target_r = ((batch["gt_r"][:, 0:1] + batch["gt_r"][:, 1:2] + batch["gt_r"][:, 2:3]) >= 0).float()
    sign_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_f["pred_sign_logit"], sign_target_f) + torch.nn.functional.binary_cross_entropy_with_logits(pred_r["pred_sign_logit"], sign_target_r)
    signed_weight = torch.where(batch["small_motion_mask"] > 0.5, torch.full_like(dot_f, 0.35), torch.ones_like(dot_f))
    loss = (
        ((1.0 - dot_f) * signed_weight).mean()
        + ((1.0 - dot_r) * signed_weight).mean()
        + 0.25 * ((1.0 - axis_f).mean() + (1.0 - axis_r).mean())
        + 0.25 * (1.0 - order_dot).mean()
        + 0.25 * sign_loss
        + 0.4 * (torch.relu(-dot_f).mean() + torch.relu(-dot_r).mean())
    )
    return loss


def _eval(model, loader, device):
    signed, abs_signed, anti, severe, flip, sign_acc = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pf = model(batch["feat_f"])
            pr = model(batch["feat_r"])
            dot = (pf["pred_dir_unit"] * batch["gt_f"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            flip_dot = (pf["pred_dir_unit"] * (-pr["pred_dir_unit"])).sum(dim=1).clamp(-1.0, 1.0)
            ang = torch.rad2deg(torch.arccos(dot))
            ang_abs = torch.rad2deg(torch.arccos(abs_dot))
            signed.extend(ang.cpu().tolist())
            abs_signed.extend(ang_abs.cpu().tolist())
            anti.extend((dot < 0).float().cpu().tolist())
            severe.extend((ang > 120).float().cpu().tolist())
            flip.extend(flip_dot.cpu().tolist())
            sign_pred = (pf["pred_sign_logit"] > 0).float()
            sign_gt = ((batch["gt_f"][:, 0:1] + batch["gt_f"][:, 1:2] + batch["gt_f"][:, 2:3]) >= 0).float()
            sign_acc.extend((sign_pred == sign_gt).float().cpu().tolist())
    flip_np = np.asarray(flip, dtype=np.float64)
    return {
        "signed_tdir_mean": float(np.mean(signed)),
        "signed_tdir_median": float(np.percentile(signed, 50)),
        "signed_tdir_p90": float(np.percentile(signed, 90)),
        "tdir_abs_mean": float(np.mean(abs_signed)),
        "tdir_abs_median": float(np.percentile(abs_signed, 50)),
        "tdir_abs_p90": float(np.percentile(abs_signed, 90)),
        "anti_parallel_rate": float(np.mean(anti)),
        "severe_wrong_sign_rate": float(np.mean(severe)),
        "pair_order_dir_flip_cosine_mean": float(np.mean(flip_np)),
        "pair_order_dir_flip_success_rate": float(np.mean(flip_np > 0.5)),
        "sign_accuracy": float(np.mean(sign_acc)),
        "axis_tdir_abs_mean": float(np.mean(abs_signed)),
    }


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, split_summary, bucket_stats = make_samples()
    train_ds = OrderDataset(train_samples, bucket_stats)
    val_ds = OrderDataset(val_samples, bucket_stats)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    feat_dim = len(train_ds[0]["feat_f"])
    model = OrderAwareSignedDirectionModel(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_path = OUT_DIR / "s5e10_order_signed_best.pt"
    history: List[Dict[str, object]] = []
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pf = model(batch["feat_f"])
            pr = model(batch["feat_r"])
            loss = _loss(pf, pr, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_metrics = _eval(model, val_loader, device)
        history.append({"epoch": epoch, "val_metrics": val_metrics})
        score = val_metrics["signed_tdir_mean"] + 0.5 * val_metrics["tdir_abs_mean"] + 35.0 * val_metrics["anti_parallel_rate"] - 10.0 * val_metrics["pair_order_dir_flip_success_rate"]
        if score < best_score:
            best_score = score
            save_order_direction_checkpoint(best_path, model, feat_dim, bucket_stats)
    status = {
        "attempted": True,
        "classification": "S5E10_ORDER_SIGNED_DIRECTION_TRAINING_SMOKE",
        "checkpoint": str(best_path),
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "reversed_pair_count": len(train_samples),
        "reversed_pair_fraction": 1.0,
        "history": history,
        "architecture_audit": architecture_audit(),
        "notes": [
            "S5E10 显式使用 ordered concat、差分和乘积特征。",
            "训练时引入 reversed pair anti-symmetry 约束。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "order_signed_direction_training_status.json", status)
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
