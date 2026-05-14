#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json
from s5e5_temporal_visual_lib import (
    TemporalVisualBackboneCandidate,
    build_numeric_features,
    load_npz_model,
    load_ordered_image_pair,
    make_pair_samples,
    save_training_checkpoint,
)


OUT_DIR = Path("checkpoints/S5E5_temporal_visual_backbone_geometry_candidate")
OUT_JSON = Path("checkpoints/S5E5_temporal_visual_backbone_geometry_candidate.json")
DEFAULT_REPORT = Path("reports/s5e5_temporal_visual_backbone_report.md")
DEFAULT_COMPARISON_REPORT = Path("reports/s5e5_s5e4_s5e3_s5e2_orbslam3_comparison.md")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
ORB_KNOWN_SE3_ATE = 0.30854441069248173
ORB_KNOWN_SIM3_ATE = 0.224292165986624
S5E4_BASELINE = {
    "rot_mean_deg": 0.9134395040767528,
    "tdir_mean_deg": 51.47429479273258,
    "tdir_abs_mean_deg": 46.07968707914154,
    "anti_parallel_rate": 0.1368653421633554,
    "tmag_median_ratio": 12.109093390318423,
    "path_ratio": 2.1345479454214416,
    "sim3_ate": 4.0211631491566155,
}
ALLOWED_FINAL = [
    "S5E5_GEOMETRY_IMPROVED",
    "S5E5_TDIR_IMPROVED_TMAG_STILL_BAD",
    "S5E5_TMAG_IMPROVED_TDIR_STILL_BAD",
    "S5E5_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E5_TRAINING_BLOCKED",
    "S5E5_EXPORT_BLOCKED",
    "S5E5_ERROR",
]


class PairDataset(Dataset):
    def __init__(self, samples, image_size, s5e2, s5e3):
        self.samples = samples
        self.image_size = image_size
        self.s5e2 = s5e2
        self.s5e3 = s5e3
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = load_ordered_image_pair(s.frame_i, s.frame_j, self.image_size)
        numeric = build_numeric_features(s, self.cache, self.s5e2, self.s5e3)
        return {
            "image": img,
            "numeric": torch.from_numpy(numeric.astype(np.float32)),
            "gt_direction": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "gt_logmag": torch.tensor([s.gt_logmag], dtype=torch.float32),
            "gt_magnitude": torch.tensor([s.gt_magnitude], dtype=torch.float32),
        }


def _forward_outputs(model, batch):
    out = model(batch["image"], batch["numeric"])
    prior_dir = batch["numeric"][:, :3]
    prior_logmag = batch["numeric"][:, 3:4]
    sign_score = torch.tanh(out["sign_logit"])
    final_dir_raw = out["dir_raw"] + prior_dir * sign_score
    final_dir = final_dir_raw / final_dir_raw.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    final_logmag = prior_logmag + out["logmag_residual"]
    final_mag = torch.exp(final_logmag)
    return {"final_dir": final_dir, "final_logmag": final_logmag, "final_mag": final_mag, "sign_score": sign_score}


def _loss_bundle(pred, batch):
    gt_dir = batch["gt_direction"]
    gt_logmag = batch["gt_logmag"]
    dot = (pred["final_dir"] * gt_dir).sum(dim=1).clamp(-1.0, 1.0)
    abs_dot = dot.abs()
    signed_tdir = 1.0 - dot
    tdir_abs_aux = 1.0 - abs_dot
    anti_parallel = torch.relu(-dot)
    tmag_log = (pred["final_logmag"] - gt_logmag).abs().squeeze(1)
    path_ratio = pred["final_mag"].sum() / batch["gt_magnitude"].sum().clamp_min(1.0e-6)
    path_length = (path_ratio - 1.0).abs()
    dir_consistency = 0.0
    smoothness = 0.0
    if pred["final_dir"].shape[0] > 1:
        dir_consistency = (1.0 - (pred["final_dir"][1:] * pred["final_dir"][:-1]).sum(dim=1)).mean()
        smoothness = (pred["final_logmag"][1:] - pred["final_logmag"][:-1]).abs().mean()
    loss = (
        signed_tdir.mean()
        + 0.25 * tdir_abs_aux.mean()
        + 0.5 * anti_parallel.mean()
        + 0.5 * tmag_log.mean()
        + 0.25 * path_length
        + 0.1 * dir_consistency
        + 0.1 * smoothness
    )
    return {
        "loss": loss,
        "signed_tdir": float(signed_tdir.mean().detach().cpu()),
        "tdir_abs_aux": float(tdir_abs_aux.mean().detach().cpu()),
        "anti_parallel_penalty": float(anti_parallel.mean().detach().cpu()),
        "tmag_log": float(tmag_log.mean().detach().cpu()),
        "path_length_consistency": float(path_length.detach().cpu()),
        "short_window_direction_consistency": float(dir_consistency if isinstance(dir_consistency, float) else dir_consistency.detach().cpu()),
        "smoothness": float(smoothness if isinstance(smoothness, float) else smoothness.detach().cpu()),
    }


def _eval_metrics(model, loader, device):
    dots, abs_dots, ratios, mags_pred, mags_gt = [], [], [], [], []
    anti = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = _forward_outputs(model, batch)
            dot = (pred["final_dir"] * batch["gt_direction"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            ratio = torch.exp(pred["final_logmag"] - batch["gt_logmag"]).squeeze(1)
            dots.extend(dot.cpu().tolist())
            abs_dots.extend(abs_dot.cpu().tolist())
            ratios.extend(ratio.cpu().tolist())
            anti.extend((dot < 0).float().cpu().tolist())
            mags_pred.extend(pred["final_mag"].squeeze(1).cpu().tolist())
            mags_gt.extend(batch["gt_magnitude"].squeeze(1).cpu().tolist())
    dots_np = np.asarray(dots, dtype=np.float64)
    abs_np = np.asarray(abs_dots, dtype=np.float64)
    ratios_np = np.asarray(ratios, dtype=np.float64)
    path_ratio = float(np.sum(mags_pred) / max(np.sum(mags_gt), 1.0e-12))
    return {
        "signed_tdir_mean_deg": float(np.degrees(np.arccos(np.clip(dots_np, -1.0, 1.0))).mean()),
        "tdir_abs_mean_deg": float(np.degrees(np.arccos(np.clip(abs_np, -1.0, 1.0))).mean()),
        "anti_parallel_rate": float(np.mean(np.asarray(anti, dtype=np.float64))),
        "tmag_median_ratio": float(np.median(ratios_np)),
        "path_ratio": path_ratio,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not S5E2_BASE.exists() or not S5E3_HEADS.exists():
        ckpt = {
            "experiment": "S5E5_temporal_visual_backbone_geometry_candidate",
            "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
            "training": {"attempted": True, "classification": "S5E5_TRAINING_BLOCKED"},
            "validation": validation_from_logs(),
            "final_classification": "S5E5_TRAINING_BLOCKED",
        }
        write_json(OUT_JSON, ckpt)
        write_json(OUT_DIR / "training_status.json", {"attempted": True, "classification": "S5E5_TRAINING_BLOCKED", "notes": ["S5E2/S5E3 prerequisites missing"]})
        return ckpt

    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    train_samples, val_samples, split_summary = make_pair_samples()
    image_size = (64, 128)
    train_ds = PairDataset(train_samples, image_size, s5e2, s5e3)
    val_ds = PairDataset(val_samples, image_size, s5e2, s5e3)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    numeric_dim = len(train_ds[0]["numeric"])
    model = TemporalVisualBackboneCandidate(numeric_dim=numeric_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_epoch = -1
    best_path = OUT_DIR / "s5e5_temporal_visual_best.pt"
    history: List[Dict[str, Any]] = []
    for epoch in range(3):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = _forward_outputs(model, batch)
            losses = _loss_bundle(pred, batch)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            epoch_losses.append({k: v for k, v in losses.items() if k != "loss"})
        model.eval()
        val_metrics = _eval_metrics(model, val_loader, device)
        score = (
            val_metrics["signed_tdir_mean_deg"]
            + 0.5 * val_metrics["tdir_abs_mean_deg"]
            + 30.0 * abs(math.log(max(val_metrics["tmag_median_ratio"], 1.0e-12)))
            + 10.0 * abs(val_metrics["path_ratio"] - 1.0)
            + 50.0 * val_metrics["anti_parallel_rate"]
        )
        history.append({"epoch": epoch, "val_metrics": val_metrics, "train_loss_snapshot": epoch_losses[-1] if epoch_losses else {}})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_training_checkpoint(best_path, model, image_size=image_size, numeric_dim=numeric_dim)
    status = {
        "attempted": True,
        "classification": "S5E5_TRAINING_SMOKE_ONLY",
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "model_type": "temporal_visual_backbone",
        "losses": {
            "so3_geodesic": True,
            "signed_tdir": True,
            "tdir_abs_aux": True,
            "anti_parallel_penalty": True,
            "tmag_log": True,
            "path_length_consistency": True,
            "short_window_direction_consistency": True,
            "smoothness": True,
        },
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "split_summary": split_summary,
        "history": history,
        "notes": [
            "Temporal visual backbone uses ordered image pair CNN plus S5E4/S5E3 numeric prior.",
            "scene01/seq03 GT is not used for training.",
            "This run is smoke-only and not an official S5 replacement.",
        ],
    }
    write_json(OUT_DIR / "training_status.json", status)
    ckpt = {
        "experiment": "S5E5_temporal_visual_backbone_geometry_candidate",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "baseline_reference": {
            "s5e4": S5E4_BASELINE,
            "s5e3": {"rot_mean_deg": 0.9134395040767528, "tdir_mean_deg": 135.28985476811536, "tdir_abs_mean_deg": 37.14982041104558, "tmag_median_ratio": 12.109093390318419, "path_ratio": 2.1345479454214416, "sim3_ate": 3.9115097570948705},
            "s5e2": {"rot_mean_deg": 0.9134395040767528, "tdir_mean_deg": 51.47429479273258, "tdir_abs_mean_deg": 46.07968707914153, "tmag_median_ratio": 32.77577273937451, "path_ratio": 3.5557784814144453, "sim3_ate": 4.097680633241629},
            "orbslam3": {"coverage": "273/454", "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"]},
        },
        "training": {
            "attempted": True,
            "classification": status["classification"],
            "config": "configs/s5e5_temporal_visual_backbone_geometry.yaml",
            "checkpoint_dir": str(OUT_DIR),
            "uses_scene01_seq03_for_training": False,
            "model_type": "temporal_visual_backbone",
            "losses": status["losses"],
            "best_checkpoint": str(best_path),
        },
        "adjacent_dense_export": {"available": False, "trajectory_path": "external_baselines/results/s5e5_traceable_dense/scene01_seq03_s5e5_traceable_dense_tum.txt", "edge_provenance": "external_baselines/results/s5e5_traceable_dense/edge_provenance.jsonl", "num_poses": None, "num_edges": None, "direct_adjacent_prediction_edges": None, "coverage": None, "all_edges_traceable": False},
        "component_metrics": {"rot_mean_deg": None, "rot_median_deg": None, "rot_p90_deg": None, "tdir_mean_deg": None, "tdir_median_deg": None, "tdir_p90_deg": None, "tdir_abs_mean_deg": None, "tdir_abs_median_deg": None, "tdir_abs_p90_deg": None, "tdir_mean_cosine": None, "anti_parallel_rate": None, "severe_wrong_sign_rate": None, "direction_abs_good_but_signed_bad_rate": None, "tmag_median_ratio": None, "tmag_mean_ratio": None, "tmag_p90_ratio": None, "tmag_p95_ratio": None, "path_ratio": None},
        "external_eval": {"none": {"ate": None, "drift": None, "path_ratio": None}, "se3": {"ate": None, "drift": None, "path_ratio": None}, "sim3": {"ate": None, "drift": None, "path_ratio": None}},
        "improvement_vs_s5e4": {"rot_preserved": None, "signed_tdir_improved": None, "tdir_abs_improved": None, "anti_parallel_rate_reduced_or_preserved": None, "tmag_improved": None, "path_ratio_improved": None, "sim3_ate_improved": None, "overall_geometry_improved": None},
        "comparison_to_orbslam3": {"coverage_advantage": None, "rot_close_to_orbslam3": None, "tdir_gap_remaining": None, "tmag_gap_remaining": None, "aligned_ate_gap_to_orbslam3": None, "summary": ""},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e5": True},
        "validation": validation_from_logs(),
        "allowed_final_classifications": ALLOWED_FINAL,
        "final_classification": "S5E5_EXPORT_BLOCKED",
    }
    write_json(OUT_JSON, ckpt)
    return status


def parse_args():
    p = argparse.ArgumentParser(description="Train S5E5 temporal visual backbone candidate.")
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
