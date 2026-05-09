#!/usr/bin/env python3
"""Tiny JRT1a smoke training harness for prediction-space R/tdir refiners."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent


VARIANTS = [
    "A_s5_no_refiner_reference",
    "B_rot_only_residual_smoke",
    "C_tdir_only_residual_smoke",
    "D_joint_rtdir_residual_smoke",
]


class JointRefinerMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hat(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v.unbind(dim=-1)
    z = torch.zeros_like(vx)
    return torch.stack(
        [
            torch.stack([z, -vz, vy], dim=-1),
            torch.stack([vz, z, -vx], dim=-1),
            torch.stack([-vy, vx, z], dim=-1),
        ],
        dim=-2,
    )


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    omega = omega.float()
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    K = _hat(omega)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).view(1, 3, 3).expand(omega.shape[0], -1, -1)
    theta2 = theta * theta
    theta_safe = theta.clamp_min(1.0e-6)
    sin_term = torch.sin(theta_safe) / theta_safe
    cos_term = (1.0 - torch.cos(theta_safe)) / theta2.clamp_min(1.0e-8)
    a = torch.where(theta > 1.0e-6, sin_term, 1.0 - theta2 / 6.0)
    b = torch.where(theta > 1.0e-6, cos_term, 0.5 - theta2 / 24.0)
    return eye + a.unsqueeze(-1) * K + b.unsqueeze(-1) * torch.matmul(K, K)


def geodesic_rad(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    R = torch.matmul(R1.float().transpose(-1, -2), R2.float())
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    return torch.acos(cos)


def _load_dataset(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    meta = json.loads(str(arr["metadata_json"].item()))
    return {
        "features": torch.tensor(arr["features"], dtype=torch.float32),
        "R_pred": torch.tensor(arr["R_pred"], dtype=torch.float32),
        "tdir_pred": torch.tensor(arr["tdir_pred"], dtype=torch.float32),
        "log_tmag_pred": torch.tensor(arr["log_tmag_pred"], dtype=torch.float32),
        "tmag_pred": torch.tensor(arr["tmag_pred"], dtype=torch.float32),
        "R_gt": torch.tensor(arr["R_gt"], dtype=torch.float32),
        "tdir_gt": torch.tensor(arr["tdir_gt"], dtype=torch.float32),
        "tmag_gt": torch.tensor(arr["tmag_gt"], dtype=torch.float32),
        "split": torch.tensor(arr["split"], dtype=torch.long),
        "metadata": meta,
    }


def _standardize(x: torch.Tensor, train_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (x - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def _apply_variant(model: JointRefinerMLP | None, xb: torch.Tensor, R_pred: torch.Tensor, tdir_pred: torch.Tensor, variant: str):
    if model is None:
        delta = torch.zeros((xb.shape[0], 6), device=xb.device, dtype=xb.dtype)
    else:
        delta = model(xb)
    delta_rot = delta[:, :3]
    delta_tdir = delta[:, 3:6]
    if variant == "B_rot_only_residual_smoke":
        delta_tdir = torch.zeros_like(delta_tdir)
    elif variant == "C_tdir_only_residual_smoke":
        delta_rot = torch.zeros_like(delta_rot)
    elif variant == "A_s5_no_refiner_reference":
        delta_rot = torch.zeros_like(delta_rot)
        delta_tdir = torch.zeros_like(delta_tdir)
    R_ref = torch.matmul(exp_so3(delta_rot), R_pred.float())
    tdir_ref = F.normalize(tdir_pred.float() + delta_tdir.float(), dim=-1, eps=1.0e-6)
    return R_ref, tdir_ref, delta_rot, delta_tdir


def _loss(model, data: Dict[str, Any], idx: torch.Tensor, variant: str, weights: Dict[str, float]) -> torch.Tensor:
    xb = data["x_std"][idx]
    R_ref, tdir_ref, delta_rot, delta_tdir = _apply_variant(model, xb, data["R_pred"][idx], data["tdir_pred"][idx], variant)
    l_rot = geodesic_rad(R_ref, data["R_gt"][idx]).mean()
    cos = (tdir_ref * F.normalize(data["tdir_gt"][idx], dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
    l_tdir = (1.0 - cos).mean()
    l_couple = l_rot * l_tdir
    l_reg = (delta_rot.square().sum(dim=-1) + delta_tdir.square().sum(dim=-1)).mean()
    return (
        float(weights["w_rot"]) * l_rot
        + float(weights["w_tdir"]) * l_tdir
        + float(weights["w_couple"]) * l_couple
        + float(weights["w_reg"]) * l_reg
    )


def component_metrics(model, data: Dict[str, Any], idx: torch.Tensor, variant: str) -> Dict[str, float]:
    with torch.no_grad():
        xb = data["x_std"][idx]
        R_ref, tdir_ref, _dr, _dt = _apply_variant(model, xb, data["R_pred"][idx], data["tdir_pred"][idx], variant)
        rot = geodesic_rad(R_ref, data["R_gt"][idx]) * (180.0 / math.pi)
        cos = (tdir_ref * F.normalize(data["tdir_gt"][idx], dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
        tdir = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)) * (180.0 / math.pi)
        tmag = torch.abs(data["log_tmag_pred"][idx] - torch.log(data["tmag_gt"][idx].clamp_min(1.0e-12)))
    def q(v: torch.Tensor, p: float) -> float:
        return float(torch.quantile(v.float().cpu(), p).item())
    return {
        "rot_mean_deg": float(rot.mean().item()),
        "rot_median_deg": q(rot, 0.5),
        "rot_p90_deg": q(rot, 0.9),
        "rot_max_deg": float(rot.max().item()),
        "tdir_mean_deg": float(tdir.mean().item()),
        "tdir_median_deg": q(tdir, 0.5),
        "tdir_p90_deg": q(tdir, 0.9),
        "tdir_max_deg": float(tdir.max().item()),
        "tdir_mean_cosine": float(cos.mean().item()),
        "tmag_mean_log_error": float(tmag.mean().item()),
        "tmag_median_log_error": q(tmag, 0.5),
        "tmag_p90_log_error": q(tmag, 0.9),
        "tmag_max_log_error": float(tmag.max().item()),
        "num_eval_pairs": int(idx.numel()),
    }


def _fmt_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run tiny JRT1a joint R/tdir residual smoke training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-log", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    out_dir = _resolve(args.output_dir)
    train_log_path = _resolve(args.train_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config["smoke"]["seed"]))
    np.random.seed(int(config["smoke"]["seed"]))
    data = _load_dataset(_resolve(args.dataset))
    train_idx = torch.where(data["split"] == 0)[0]
    val_idx = torch.where(data["split"] == 1)[0]
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise RuntimeError(f"JRT1a train smoke needs non-empty train/val splits, got {train_idx.numel()}/{val_idx.numel()}")
    x_std, mu, sd = _standardize(data["features"], train_idx)
    data["x_std"] = x_std
    input_dim = int(data["features"].shape[1])

    smoke = config["smoke"]
    weights = config["loss_weights"]
    num_updates = int(smoke["num_updates"])
    batch_size = int(smoke["batch_size"])
    lr = float(smoke["learning_rate"])
    gen = torch.Generator().manual_seed(int(smoke["seed"]))
    base_val_metrics = component_metrics(None, data, val_idx, "A_s5_no_refiner_reference")
    entries: List[Dict[str, Any]] = []

    for variant in config["variants"]:
        if variant not in VARIANTS:
            raise RuntimeError(f"Unknown JRT1a variant: {variant}")
        if variant == "A_s5_no_refiner_reference":
            entries.append(
                {
                    "variant": variant,
                    "train_loss_initial": None,
                    "train_loss_final": None,
                    "val_component_metrics_before": base_val_metrics,
                    "val_component_metrics_after": base_val_metrics,
                    "num_updates": 0,
                    "status": "ok",
                    "error_if_any": None,
                    "model_state_path": None,
                }
            )
            continue
        model = JointRefinerMLP(input_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        first_idx = train_idx[: min(batch_size, train_idx.numel())]
        loss_initial = float(_loss(model, data, first_idx, variant, weights).detach().item())
        loss_final = loss_initial
        for _step in range(num_updates):
            pick = train_idx[torch.randint(0, train_idx.numel(), (batch_size,), generator=gen)]
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, data, pick, variant, weights)
            loss.backward()
            opt.step()
            loss_final = float(loss.detach().item())
        after = component_metrics(model, data, val_idx, variant)
        ckpt_path = out_dir / f"{variant}.pt"
        torch.save(
            {
                "variant": variant,
                "input_dim": input_dim,
                "state_dict": model.state_dict(),
                "feature_mean": mu,
                "feature_std": sd,
                "num_updates": num_updates,
            },
            ckpt_path,
        )
        entries.append(
            {
                "variant": variant,
                "train_loss_initial": _fmt_float(loss_initial),
                "train_loss_final": _fmt_float(loss_final),
                "val_component_metrics_before": base_val_metrics,
                "val_component_metrics_after": after,
                "num_updates": num_updates,
                "status": "ok",
                "error_if_any": None,
                "model_state_path": str(ckpt_path.relative_to(REPO_ROOT)),
            }
        )

    payload = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "dataset_metadata": data["metadata"],
        "variants": entries,
        "no_test_gt_used_for_training": True,
        "status": "ok" if all(v["status"] == "ok" for v in entries) else "failed",
    }
    train_log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
