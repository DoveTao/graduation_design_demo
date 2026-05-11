#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from s5e2_adjacent_dense_lib import write_json


class S5E13Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, log_scale_clip: float, rot_delta_scale: float, dir_delta_scale: float) -> None:
        super().__init__()
        self.log_scale_clip = float(log_scale_clip)
        self.rot_delta_scale = float(rot_delta_scale)
        self.dir_delta_scale = float(dir_delta_scale)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.rot_head = torch.nn.Linear(hidden_dim, 3)
        self.dir_head = torch.nn.Linear(hidden_dim, 3)
        self.sign_head = torch.nn.Linear(hidden_dim, 1)
        self.scale_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, feat: torch.Tensor, prior_rotvec: torch.Tensor, prior_dir: torch.Tensor, bucket_prior: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(feat)
        rot_delta = self.rot_head(z) * self.rot_delta_scale
        dir_delta = torch.tanh(self.dir_head(z)) * self.dir_delta_scale
        sign_logit = self.sign_head(z)
        raw_scale_delta = self.scale_head(z)
        pred_log_scale_delta = torch.tanh(raw_scale_delta) * self.log_scale_clip
        pred_axis = F.normalize(prior_dir + dir_delta, dim=1)
        soft_sign = torch.tanh(sign_logit)
        soft_dir = pred_axis * soft_sign
        pred_dir = F.normalize(soft_dir + 1.0e-4 * pred_axis, dim=1)
        pred_rotvec = prior_rotvec + rot_delta
        pred_tmag = bucket_prior * torch.exp(pred_log_scale_delta.squeeze(1))
        return {
            "pred_rotvec": pred_rotvec,
            "pred_axis": pred_axis,
            "soft_sign": soft_sign.squeeze(1),
            "pred_dir": pred_dir,
            "pred_log_scale_delta": pred_log_scale_delta.squeeze(1),
            "pred_tmag": pred_tmag,
        }


def load_config(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text

    cfg: Dict[str, Any] = {}
    current: str | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            if ":" in raw:
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
            elif raw.startswith("  ") and ":" in raw:
                if mode == "list":
                    continue
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                key, value = raw.strip().split(":", 1)
                cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def _to_tensor(rows: List[Dict[str, Any]], device: torch.device, bucket_stats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    feat = torch.tensor([r["model_feature_vector"] for r in rows], dtype=torch.float32, device=device)
    prior_rot = torch.tensor([r["prior_rotvec"] for r in rows], dtype=torch.float32, device=device)
    prior_dir = F.normalize(torch.tensor([r["prior_dir"] for r in rows], dtype=torch.float32, device=device), dim=1)
    gt_dir = F.normalize(torch.tensor([r["gt_direction"] for r in rows], dtype=torch.float32, device=device), dim=1)
    gt_mag = torch.tensor([r["gt_magnitude"] for r in rows], dtype=torch.float32, device=device)
    gt_logmag = torch.log(torch.clamp(gt_mag, min=1.0e-12))
    gt_rotvec = torch.tensor([r["gt_rotvec"] for r in rows], dtype=torch.float32, device=device)
    bucket_prior = torch.tensor([bucket_stats[r["prior_bucket"]]["median"] for r in rows], dtype=torch.float32, device=device)
    signed_weight = torch.tensor([r["signed_tdir_loss_weight"] for r in rows], dtype=torch.float32, device=device)
    abs_weight = torch.tensor([r["tdir_abs_loss_weight"] for r in rows], dtype=torch.float32, device=device)
    tmag_weight = torch.tensor([r["tmag_loss_weight"] for r in rows], dtype=torch.float32, device=device)
    rot_weight = torch.tensor([r["rot_loss_weight"] for r in rows], dtype=torch.float32, device=device)
    observable = torch.tensor([1.0 if r["observable"] else 0.0 for r in rows], dtype=torch.float32, device=device)
    return {
        "feat": feat,
        "prior_rot": prior_rot,
        "prior_dir": prior_dir,
        "gt_dir": gt_dir,
        "gt_mag": gt_mag,
        "gt_logmag": gt_logmag,
        "gt_rotvec": gt_rotvec,
        "bucket_prior": bucket_prior,
        "signed_weight": signed_weight,
        "abs_weight": abs_weight,
        "tmag_weight": tmag_weight,
        "rot_weight": rot_weight,
        "observable": observable,
    }


def _rotation_geodesic_deg(pred_rotvec: torch.Tensor, gt_rotvec: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_rotvec - gt_rotvec, dim=1) * (180.0 / math.pi)


def _metrics(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    dot = torch.clamp(torch.sum(pred["pred_dir"] * batch["gt_dir"], dim=1), -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(dot))
    ang_abs = torch.rad2deg(torch.acos(torch.abs(dot)))
    tmag_ratio = pred["pred_tmag"] / torch.clamp(batch["gt_mag"], min=1.0e-12)
    return {
        "signed_tdir_mean": float(ang.mean().item()),
        "tdir_abs_mean": float(ang_abs.mean().item()),
        "anti_parallel_rate": float((dot < 0).float().mean().item()),
        "severe_wrong_sign_rate": float((ang > 120.0).float().mean().item()),
        "rot_mean_deg": float(_rotation_geodesic_deg(pred["pred_rotvec"], batch["gt_rotvec"]).mean().item()),
        "tmag_p95_ratio": float(torch.quantile(tmag_ratio, 0.95).item()),
        "path_ratio": float(pred["pred_tmag"].sum().item() / max(batch["gt_mag"].sum().item(), 1.0e-12)),
        "observable_signed_tdir_mean": float(ang[batch["observable"] > 0.5].mean().item()) if torch.any(batch["observable"] > 0.5) else float("nan"),
    }


def _loss(model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = cfg["loss"]
    dot = torch.clamp(torch.sum(model_out["pred_dir"] * batch["gt_dir"], dim=1), -1.0, 1.0)
    signed_loss = ((1.0 - dot) * batch["signed_weight"]).sum() / torch.clamp(batch["signed_weight"].sum(), min=1.0)
    abs_loss = ((1.0 - torch.abs(dot)) * batch["abs_weight"]).sum() / torch.clamp(batch["abs_weight"].sum(), min=1.0)
    anti_penalty = (torch.relu(-dot) * batch["signed_weight"]).sum() / torch.clamp(batch["signed_weight"].sum(), min=1.0)
    log_ratio = model_out["pred_log_scale_delta"] + torch.log(torch.clamp(batch["bucket_prior"], min=1.0e-12)) - batch["gt_logmag"]
    tmag_loss = (F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction="none") * batch["tmag_weight"]).sum() / torch.clamp(batch["tmag_weight"].sum(), min=1.0)
    pred_ratio = model_out["pred_tmag"] / torch.clamp(batch["gt_mag"], min=1.0e-12)
    tmag_p95_penalty = torch.relu(torch.quantile(pred_ratio, 0.95) - 3.0)
    rot_loss = (_rotation_geodesic_deg(model_out["pred_rotvec"], batch["gt_rotvec"]) * batch["rot_weight"]).sum() / torch.clamp(batch["rot_weight"].sum(), min=1.0)
    path_consistency = torch.abs(model_out["pred_tmag"].sum() - batch["gt_mag"].sum()) / torch.clamp(batch["gt_mag"].sum(), min=1.0e-12)
    if model_out["pred_dir"].shape[0] > 1:
        smoothness = torch.mean(torch.linalg.norm(model_out["pred_dir"][1:] - model_out["pred_dir"][:-1], dim=1))
        short_window = torch.mean(torch.relu(torch.sum(model_out["pred_dir"][1:] * model_out["pred_dir"][:-1], dim=1) * -1.0))
    else:
        smoothness = model_out["pred_dir"].sum() * 0.0
        short_window = smoothness
    total = (
        loss_cfg["w_rot"] * rot_loss
        + loss_cfg["w_signed_tdir"] * signed_loss
        + loss_cfg["w_tdir_abs_aux"] * abs_loss
        + loss_cfg["w_anti_parallel"] * anti_penalty
        + loss_cfg["w_tmag"] * tmag_loss
        + loss_cfg["w_tmag_p95_penalty"] * tmag_p95_penalty
        + loss_cfg["w_path_length_consistency"] * path_consistency
        + loss_cfg["w_short_window_direction_consistency"] * short_window
        + loss_cfg["w_smoothness"] * smoothness
    )
    pieces = {
        "rot_loss": float(rot_loss.item()),
        "signed_loss": float(signed_loss.item()),
        "abs_loss": float(abs_loss.item()),
        "anti_penalty": float(anti_penalty.item()),
        "tmag_loss": float(tmag_loss.item()),
        "tmag_p95_penalty": float(tmag_p95_penalty.item()),
        "path_consistency": float(path_consistency.item()),
        "short_window": float(short_window.item()),
        "smoothness": float(smoothness.item()),
    }
    return total, pieces


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    weighted = json.loads(Path(cfg["weighted_dataset_json"]).read_text(encoding="utf-8"))
    ckpt_dir = Path("checkpoints/S5E13_real_correspondence_signed_direction_candidate")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    training_status_path = Path(cfg["training_status_json"])
    seed = int(cfg["train"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if cfg["train"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")

    train_rows = weighted["train_rows"]
    val_rows = weighted["val_rows"]
    bucket_stats = weighted["bucket_stats"]
    train_batch = _to_tensor(train_rows, device, bucket_stats)
    val_batch = _to_tensor(val_rows, device, bucket_stats)
    input_dim = len(train_rows[0]["model_feature_vector"])

    model = S5E13Model(
        input_dim=input_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        log_scale_clip=float(cfg["model"]["log_scale_clip"]),
        rot_delta_scale=float(cfg["model"]["rot_delta_scale"]),
        dir_delta_scale=float(cfg["model"]["dir_delta_scale"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    best_metric = float("inf")
    best_payload: Dict[str, Any] | None = None
    patience = int(cfg["train"]["patience"])
    stalled = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        opt.zero_grad()
        out = model(train_batch["feat"], train_batch["prior_rot"], train_batch["prior_dir"], train_batch["bucket_prior"])
        loss, pieces = _loss(out, train_batch, cfg)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            train_out = model(train_batch["feat"], train_batch["prior_rot"], train_batch["prior_dir"], train_batch["bucket_prior"])
            val_out = model(val_batch["feat"], val_batch["prior_rot"], val_batch["prior_dir"], val_batch["bucket_prior"])
            train_metrics = _metrics(train_out, train_batch)
            val_metrics = _metrics(val_out, val_batch)
        score = val_metrics["signed_tdir_mean"] + 0.5 * val_metrics["tdir_abs_mean"] + 20.0 * val_metrics["anti_parallel_rate"] + 0.2 * val_metrics["tmag_p95_ratio"]
        history.append({"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics, "train_loss_parts": pieces})
        if score < best_metric:
            best_metric = score
            stalled = 0
            best_payload = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "config": cfg,
                "history_entry": history[-1],
            }
            torch.save(best_payload, ckpt_dir / "s5e13_best.pt")
        else:
            stalled += 1
            if stalled >= patience:
                break

    classification = "S5E13_TRAINING_COMPLETE"
    if best_payload is None:
        classification = "S5E13_TRAINING_ERROR"
    elif len(history) <= 5:
        classification = "S5E13_TRAINING_SMOKE_ONLY"
    elif float(weighted.get("signed_direction_reliable_fraction_train", 0.0)) <= 0.0:
        classification = "S5E13_TRAINING_PARTIAL_NO_TRAIN_CORRESPONDENCE"

    status = {
        "attempted": True,
        "classification": classification,
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(val_rows),
        "uses_scene01_seq03_for_training": False,
        "uses_scene01_seq03_gt_for_training": False,
        "uses_correspondence_features": True,
        "uses_strict_essential_geometry": False,
        "losses": {
            "so3_geodesic": True,
            "signed_tdir_weighted": True,
            "tdir_abs_aux": True,
            "anti_parallel_penalty": True,
            "robust_tmag_log": True,
            "tmag_p95_penalty": True,
            "path_length_consistency": True,
            "short_window_direction_consistency": True,
            "smoothness": True,
        },
        "best_checkpoint": str(ckpt_dir / "s5e13_best.pt"),
        "best_val_metrics": None if best_payload is None else best_payload["history_entry"]["val_metrics"],
        "history_tail": history[-5:],
        "notes": [
            "S5E13 使用 S5E12 real correspondence / optical_flow / observability 权重。",
            "scene01/seq03 GT 没有用于训练；只用于最终 evaluation。",
            "strict essential geometry 没有作为监督输入。",
        ],
    }
    write_json(training_status_path, status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
