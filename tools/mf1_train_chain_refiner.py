#!/usr/bin/env python3
"""Train MF1a lightweight multi-frame chain refiners."""

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
    return torch.stack([torch.stack([z, -vz, vy], -1), torch.stack([vz, z, -vx], -1), torch.stack([-vy, vx, z], -1)], -2)


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    shape = omega.shape
    flat = omega.reshape(-1, 3).float()
    theta = torch.linalg.norm(flat, dim=-1, keepdim=True)
    K = _hat(flat)
    eye = torch.eye(3, dtype=flat.dtype, device=flat.device).view(1, 3, 3).expand(flat.shape[0], -1, -1)
    theta2 = theta * theta
    theta_safe = theta.clamp_min(1.0e-6)
    a = torch.where(theta > 1.0e-6, torch.sin(theta_safe) / theta_safe, 1.0 - theta2 / 6.0)
    b = torch.where(theta > 1.0e-6, (1.0 - torch.cos(theta_safe)) / theta2.clamp_min(1.0e-8), 0.5 - theta2 / 24.0)
    return (eye + a.unsqueeze(-1) * K + b.unsqueeze(-1) * torch.matmul(K, K)).reshape(*shape[:-1], 3, 3)


def geodesic_rad(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    R = torch.matmul(R1.float().transpose(-1, -2), R2.float())
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))


class PairwiseRefiner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class TemporalConvRefiner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, 64)
        self.conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(64, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x.float()))
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.out(h)


class GRURefiner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x.float()))
        y, _ = self.gru(h)
        return self.out(y)


def load_dataset(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    return {
        "features": torch.tensor(arr["features"], dtype=torch.float32),
        "R_pred": torch.tensor(arr["R_pred"], dtype=torch.float32),
        "tdir_pred": torch.tensor(arr["tdir_pred"], dtype=torch.float32),
        "log_tmag_pred": torch.tensor(arr["log_tmag_pred"], dtype=torch.float32),
        "tmag_pred": torch.tensor(arr["tmag_pred"], dtype=torch.float32),
        "R_gt": torch.tensor(arr["R_gt"], dtype=torch.float32),
        "tdir_gt": torch.tensor(arr["tdir_gt"], dtype=torch.float32),
        "tmag_gt": torch.tensor(arr["tmag_gt"], dtype=torch.float32),
        "chain_indices": torch.tensor(arr["chain_indices"], dtype=torch.long),
        "chain_split": torch.tensor(arr["chain_split"], dtype=torch.long),
        "metadata": json.loads(str(arr["metadata_json"].item())),
    }


def _compose_positions(R_seq: torch.Tensor, t_seq: torch.Tensor) -> torch.Tensor:
    B, L = t_seq.shape[:2]
    R_cur = torch.eye(3, device=t_seq.device, dtype=t_seq.dtype).view(1, 3, 3).repeat(B, 1, 1)
    t_cur = torch.zeros(B, 3, device=t_seq.device, dtype=t_seq.dtype)
    pts = [torch.zeros(B, 3, device=t_seq.device, dtype=t_seq.dtype)]
    for i in range(L):
        R_cur = torch.matmul(R_seq[:, i], R_cur)
        t_cur = torch.matmul(R_seq[:, i], t_cur.unsqueeze(-1)).squeeze(-1) + t_seq[:, i]
        pts.append(-(torch.matmul(R_cur.transpose(-1, -2), t_cur.unsqueeze(-1)).squeeze(-1)))
    return torch.stack(pts, dim=1)


def _apply_variant(model, x: torch.Tensor, R_pred: torch.Tensor, tdir_pred: torch.Tensor, variant: str):
    B, L = x.shape[:2]
    if variant == "A_s5_chain_reference":
        delta = torch.zeros(B, L, 6, device=x.device, dtype=x.dtype)
    elif variant == "B_pairwise_jrt_style_reference":
        delta = model(x.reshape(B * L, -1)).reshape(B, L, 6)
    else:
        delta = model(x)
    delta_rot = delta[..., :3]
    delta_tdir = delta[..., 3:6]
    if variant == "B_pairwise_jrt_style_reference":
        pass
    R_ref = torch.matmul(exp_so3(delta_rot), R_pred.float())
    tdir_ref = F.normalize(tdir_pred.float() + delta_tdir.float(), dim=-1, eps=1.0e-6)
    return R_ref, tdir_ref, delta_rot, delta_tdir


def _loss(model, data: Dict[str, Any], chain_ids: torch.Tensor, variant: str, weights: Dict[str, float]) -> torch.Tensor:
    chains = data["chain_indices"][chain_ids]
    x = data["x_std"][chains]
    R_pred = data["R_pred"][chains]
    tdir_pred = data["tdir_pred"][chains]
    R_ref, tdir_ref, delta_rot, delta_tdir = _apply_variant(model, x, R_pred, tdir_pred, variant)
    R_gt = data["R_gt"][chains]
    tdir_gt = data["tdir_gt"][chains]
    tmag_gt = data["tmag_gt"][chains]
    tmag_pred = data["tmag_pred"][chains]
    l_rot = geodesic_rad(R_ref, R_gt).mean()
    cos = (tdir_ref * F.normalize(tdir_gt, dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
    l_tdir = (1.0 - cos).mean()
    gt_pos = _compose_positions(R_gt, tdir_gt * tmag_gt.unsqueeze(-1))
    pr_pos = _compose_positions(R_ref, tdir_ref * tmag_pred.unsqueeze(-1))
    l_pos = torch.linalg.norm(pr_pos - gt_pos, dim=-1).mean()
    drift = torch.linalg.norm(pr_pos[:, -1] - gt_pos[:, -1], dim=-1).mean()
    gt_len = torch.linalg.norm(gt_pos[:, 1:] - gt_pos[:, :-1], dim=-1).sum(dim=-1).clamp_min(1.0e-6)
    pr_len = torch.linalg.norm(pr_pos[:, 1:] - pr_pos[:, :-1], dim=-1).sum(dim=-1)
    path_loss = torch.abs((pr_len / gt_len) - 1.0).mean()
    reg = (delta_rot.square().sum(dim=-1) + delta_tdir.square().sum(dim=-1)).mean()
    return (
        float(weights["w_pair_rot"]) * l_rot
        + float(weights["w_pair_tdir"]) * l_tdir
        + float(weights["w_chain_pos"]) * l_pos
        + float(weights["w_drift"]) * drift
        + float(weights["w_path_ratio"]) * path_loss
        + float(weights["w_reg"]) * reg
    )


def evaluate_variant(model, data: Dict[str, Any], chain_ids: torch.Tensor, variant: str) -> Dict[str, float]:
    chains = data["chain_indices"][chain_ids]
    with torch.no_grad():
        x = data["x_std"][chains]
        R_ref, tdir_ref, _dr, _dt = _apply_variant(model, x, data["R_pred"][chains], data["tdir_pred"][chains], variant)
        R_gt = data["R_gt"][chains]
        tdir_gt = data["tdir_gt"][chains]
        tmag_gt = data["tmag_gt"][chains]
        tmag_pred = data["tmag_pred"][chains]
        rot = geodesic_rad(R_ref, R_gt) * (180.0 / math.pi)
        cos = (tdir_ref * F.normalize(tdir_gt, dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
        tdir = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)) * (180.0 / math.pi)
        tmag = torch.abs(data["log_tmag_pred"][chains] - torch.log(tmag_gt.clamp_min(1.0e-12)))
        gt_pos = _compose_positions(R_gt, tdir_gt * tmag_gt.unsqueeze(-1))
        pr_pos = _compose_positions(R_ref, tdir_ref * tmag_pred.unsqueeze(-1))
        ate = torch.sqrt(((pr_pos - gt_pos).square().sum(dim=-1)).mean(dim=-1)).mean()
        drift = torch.linalg.norm(pr_pos[:, -1] - gt_pos[:, -1], dim=-1).mean()
        gt_len = torch.linalg.norm(gt_pos[:, 1:] - gt_pos[:, :-1], dim=-1).sum(dim=-1).clamp_min(1.0e-6)
        pr_len = torch.linalg.norm(pr_pos[:, 1:] - pr_pos[:, :-1], dim=-1).sum(dim=-1)
        path_ratio = (pr_len / gt_len).mean()
    return {
        "ATE_proxy": float(ate.item()),
        "drift_proxy": float(drift.item()),
        "path_ratio_proxy": float(path_ratio.item()),
        "num_chains": int(chain_ids.numel()),
        "num_pairs": int(chain_ids.numel() * chains.shape[1]),
        "rot_mean_deg": float(rot.mean().item()),
        "tdir_mean_deg": float(tdir.mean().item()),
        "tdir_mean_cosine": float(cos.mean().item()),
        "tmag_mean_log_error": float(tmag.mean().item()),
    }


def _make_model(variant: str, dim: int):
    if variant == "B_pairwise_jrt_style_reference":
        return PairwiseRefiner(dim)
    if variant == "C_temporal_conv_chain_refiner":
        return TemporalConvRefiner(dim)
    if variant == "D_gru_chain_refiner":
        return GRURefiner(dim)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MF1a chain refiner smoke variants.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-log", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    out_dir = _resolve(args.output_dir)
    log_path = _resolve(args.train_log)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["smoke"]["seed"]))
    np.random.seed(int(config["smoke"]["seed"]))
    data = load_dataset(_resolve(args.dataset))
    train_ids = torch.where(data["chain_split"] == 0)[0]
    val_ids = torch.where(data["chain_split"] == 1)[0]
    row_train = torch.unique(data["chain_indices"][train_ids].reshape(-1))
    mu = data["features"][row_train].mean(dim=0, keepdim=True)
    sd = data["features"][row_train].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    data["x_std"] = (data["features"] - mu) / sd
    dim = int(data["features"].shape[1])
    gen = torch.Generator().manual_seed(int(config["smoke"]["seed"]))
    entries: List[Dict[str, Any]] = []
    for variant in config["variants"]:
        model = _make_model(variant, dim)
        before = evaluate_variant(model, data, val_ids, variant)
        if model is None:
            entries.append({"variant": variant, "status": "ok", "num_updates": 0, "train_loss_initial": None, "train_loss_final": None, "val_metrics_before": before, "val_metrics_after": before, "error_if_any": None})
            continue
        opt = torch.optim.Adam(model.parameters(), lr=float(config["smoke"]["learning_rate"]))
        first = train_ids[: min(int(config["smoke"]["batch_size"]), train_ids.numel())]
        initial = float(_loss(model, data, first, variant, config["loss_weights"]).detach().item())
        final = initial
        for _ in range(int(config["smoke"]["num_updates"])):
            pick = train_ids[torch.randint(0, train_ids.numel(), (int(config["smoke"]["batch_size"]),), generator=gen)]
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, data, pick, variant, config["loss_weights"])
            loss.backward()
            opt.step()
            final = float(loss.detach().item())
        after = evaluate_variant(model, data, val_ids, variant)
        ckpt = out_dir / f"{variant}.pt"
        torch.save({"variant": variant, "input_dim": dim, "state_dict": model.state_dict(), "feature_mean": mu, "feature_std": sd}, ckpt)
        entries.append({"variant": variant, "status": "ok", "num_updates": int(config["smoke"]["num_updates"]), "train_loss_initial": initial, "train_loss_final": final, "val_metrics_before": before, "val_metrics_after": after, "model_state_path": str(ckpt.relative_to(REPO_ROOT)), "error_if_any": None})
    payload = {"experiment_name": config["experiment_name"], "stage": config["stage"], "dataset_metadata": data["metadata"], "variants": entries, "status": "ok", "no_final_test": True}
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
