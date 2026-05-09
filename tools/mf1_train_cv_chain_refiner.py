#!/usr/bin/env python3
"""MF1b train-CV chain-level refiner evaluation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_locked(config: Dict[str, Any]) -> None:
    for k, v in LOCKED.items():
        if abs(float(config["locked_s5_metrics"][k]) - v) > 1.0e-9:
            raise RuntimeError(f"locked S5 metric changed for {k}")


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
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(((tr - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))


class PairwiseRefiner(nn.Module):
    def __init__(self, dim: int, out_dim: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class TemporalConvRefiner(nn.Module):
    def __init__(self, dim: int, out_dim: int = 6) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, 64)
        self.conv = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.out = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x.float()))
        return self.out(self.conv(h.transpose(1, 2)).transpose(1, 2))


class GRURefiner(nn.Module):
    def __init__(self, dim: int, out_dim: int = 6) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(F.relu(self.inp(x.float())))
        return self.out(y)


def _make_model(variant: str, dim: int):
    if variant == "A_s5_chain_reference":
        return None
    if variant == "B_pairwise_jrt_style_reference":
        return PairwiseRefiner(dim, 6)
    if variant == "C_temporal_conv_chain_refiner":
        return TemporalConvRefiner(dim, 6)
    if variant in {"D_gru_chain_refiner", "E_gru_chain_refiner_pathratio_loss"}:
        return GRURefiner(dim, 6)
    if variant == "F_gru_chain_refiner_tmag_head_diagnostic":
        return GRURefiner(dim, 7)
    raise RuntimeError(f"unknown MF1b variant: {variant}")


def _load_dataset(path: Path) -> Dict[str, Any]:
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
        "metadata": json.loads(str(arr["metadata_json"].item())),
    }


def _folds(n: int, num_folds: int, seed: int, max_train: int, max_val: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = [x.tolist() for x in np.array_split(perm, min(num_folds, n))]
    out: List[Dict[str, Any]] = []
    all_ids = set(range(n))
    for fid, val in enumerate(chunks):
        train = sorted(all_ids - set(val))
        val = sorted(val)
        if len(train) > max_train:
            train = [train[i] for i in np.linspace(0, len(train) - 1, max_train).round().astype(int).tolist()]
        if len(val) > max_val:
            val = [val[i] for i in np.linspace(0, len(val) - 1, max_val).round().astype(int).tolist()]
        out.append({"fold_id": fid, "train_chain_ids": train, "val_chain_ids": val, "num_train_chains": len(train), "num_val_chains": len(val)})
    return out


def _compose_positions(R_seq: torch.Tensor, t_seq: torch.Tensor) -> torch.Tensor:
    B, L = t_seq.shape[:2]
    R_cur = torch.eye(3, device=t_seq.device).view(1, 3, 3).repeat(B, 1, 1)
    t_cur = torch.zeros(B, 3, device=t_seq.device)
    pts = [torch.zeros(B, 3, device=t_seq.device)]
    for i in range(L):
        R_cur = torch.matmul(R_seq[:, i], R_cur)
        t_cur = torch.matmul(R_seq[:, i], t_cur.unsqueeze(-1)).squeeze(-1) + t_seq[:, i]
        pts.append(-(torch.matmul(R_cur.transpose(-1, -2), t_cur.unsqueeze(-1)).squeeze(-1)))
    return torch.stack(pts, dim=1)


def _apply(model, data: Dict[str, Any], chains: torch.Tensor, variant: str):
    x = data["x_std"][chains]
    B, L = x.shape[:2]
    if model is None:
        delta = torch.zeros(B, L, 6)
    elif variant == "B_pairwise_jrt_style_reference":
        delta = model(x.reshape(B * L, -1)).reshape(B, L, -1)
    else:
        delta = model(x)
    delta_rot = delta[..., :3]
    delta_tdir = delta[..., 3:6]
    delta_log_tmag = delta[..., 6] if delta.shape[-1] > 6 else torch.zeros(B, L)
    R_ref = torch.matmul(exp_so3(delta_rot), data["R_pred"][chains].float())
    tdir_ref = F.normalize(data["tdir_pred"][chains].float() + delta_tdir.float(), dim=-1, eps=1.0e-6)
    log_tmag_ref = data["log_tmag_pred"][chains].float() + delta_log_tmag.float()
    return R_ref, tdir_ref, log_tmag_ref, delta_rot, delta_tdir, delta_log_tmag


def _loss(model, data: Dict[str, Any], chain_ids: torch.Tensor, variant: str, weights: Dict[str, float]) -> torch.Tensor:
    chains = data["chain_indices"][chain_ids]
    R_ref, tdir_ref, log_tmag_ref, dr, dt, dlog = _apply(model, data, chains, variant)
    R_gt = data["R_gt"][chains]
    tdir_gt = data["tdir_gt"][chains]
    tmag_gt = data["tmag_gt"][chains]
    tmag_ref = torch.exp(log_tmag_ref).clamp_min(1.0e-6)
    l_rot = geodesic_rad(R_ref, R_gt).mean()
    cos = (tdir_ref * F.normalize(tdir_gt, dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
    l_tdir = (1.0 - cos).mean()
    gt_pos = _compose_positions(R_gt, tdir_gt * tmag_gt.unsqueeze(-1))
    pr_pos = _compose_positions(R_ref, tdir_ref * tmag_ref.unsqueeze(-1))
    l_pos = torch.linalg.norm(pr_pos - gt_pos, dim=-1).mean()
    drift = torch.linalg.norm(pr_pos[:, -1] - gt_pos[:, -1], dim=-1).mean()
    gt_len = torch.linalg.norm(gt_pos[:, 1:] - gt_pos[:, :-1], dim=-1).sum(dim=-1).clamp_min(1.0e-6)
    pr_len = torch.linalg.norm(pr_pos[:, 1:] - pr_pos[:, :-1], dim=-1).sum(dim=-1)
    ratio = pr_len / gt_len
    path_loss = torch.abs(ratio - 1.0).mean()
    if variant == "E_gru_chain_refiner_pathratio_loss":
        path_loss = ((ratio - 1.0) ** 2).mean() * 2.0
    reg = (dr.square().sum(dim=-1) + dt.square().sum(dim=-1)).mean() + 5.0 * dlog.square().mean()
    tmag_loss = torch.abs(log_tmag_ref - torch.log(tmag_gt.clamp_min(1.0e-12))).mean() if variant == "F_gru_chain_refiner_tmag_head_diagnostic" else torch.zeros(())
    return (
        float(weights["w_pair_rot"]) * l_rot
        + float(weights["w_pair_tdir"]) * l_tdir
        + float(weights["w_chain_pos"]) * l_pos
        + float(weights["w_drift"]) * drift
        + float(weights["w_path_ratio"]) * path_loss
        + float(weights["w_tmag"]) * tmag_loss
        + float(weights["w_reg"]) * reg
    )


def _metrics(model, data: Dict[str, Any], chain_ids: torch.Tensor, variant: str) -> Dict[str, float]:
    chains = data["chain_indices"][chain_ids]
    with torch.no_grad():
        R_ref, tdir_ref, log_tmag_ref, _dr, _dt, _dl = _apply(model, data, chains, variant)
        R_gt = data["R_gt"][chains]
        tdir_gt = data["tdir_gt"][chains]
        tmag_gt = data["tmag_gt"][chains]
        tmag_ref = torch.exp(log_tmag_ref).clamp_min(1.0e-6)
        rot = geodesic_rad(R_ref, R_gt) * (180.0 / math.pi)
        cos = (tdir_ref * F.normalize(tdir_gt, dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
        tdir = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)) * (180.0 / math.pi)
        tmag = torch.abs(log_tmag_ref - torch.log(tmag_gt.clamp_min(1.0e-12)))
        gt_pos = _compose_positions(R_gt, tdir_gt * tmag_gt.unsqueeze(-1))
        pr_pos = _compose_positions(R_ref, tdir_ref * tmag_ref.unsqueeze(-1))
        ate = torch.sqrt(((pr_pos - gt_pos).square().sum(dim=-1)).mean(dim=-1)).mean()
        drift = torch.linalg.norm(pr_pos[:, -1] - gt_pos[:, -1], dim=-1).mean()
        gt_len = torch.linalg.norm(gt_pos[:, 1:] - gt_pos[:, :-1], dim=-1).sum(dim=-1).clamp_min(1.0e-6)
        pr_len = torch.linalg.norm(pr_pos[:, 1:] - pr_pos[:, :-1], dim=-1).sum(dim=-1)
        ratio = (pr_len / gt_len).mean()
    return {
        "ATE_proxy": float(ate.item()),
        "drift_proxy": float(drift.item()),
        "path_ratio_proxy": float(ratio.item()),
        "rot_mean_deg": float(rot.mean().item()),
        "rot_p90_deg": float(torch.quantile(rot.float().cpu(), 0.9).item()),
        "tdir_mean_deg": float(tdir.mean().item()),
        "tdir_p90_deg": float(torch.quantile(tdir.float().cpu(), 0.9).item()),
        "tdir_mean_cosine": float(cos.mean().item()),
        "tmag_mean_log_error": float(tmag.mean().item()),
        "num_chains": int(chain_ids.numel()),
        "num_pairs": int(chain_ids.numel() * chains.shape[1]),
    }


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _summarize(per_fold: List[Dict[str, Any]], variants: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    keys = ["ATE_proxy", "drift_proxy", "path_ratio_proxy", "rot_mean_deg", "rot_p90_deg", "tdir_mean_deg", "tdir_p90_deg", "tdir_mean_cosine", "tmag_mean_log_error", "num_chains", "num_pairs"]
    out: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        rows = [r["metrics"] for r in per_fold if r["variant"] == v and r["status"] == "ok"]
        out[v] = {k: _mean_std([row[k] for row in rows]) for k in keys}
        out[v]["num_folds_ok"] = len(rows)
    return out


def _gate(config: Dict[str, Any], mean_cv: Dict[str, Dict[str, Any]], per_fold: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, str | None]:
    ref = mean_cv["A_s5_chain_reference"]
    gate = config["gate"]
    decisions: Dict[str, Any] = {"A_s5_chain_reference": {"gate_status": "REFERENCE", "folds_passing_direction": 0}}
    passers: List[str] = []
    any_shape_signal = False
    all_path_fail = True
    for variant, row in mean_cv.items():
        if variant == "A_s5_chain_reference":
            continue
        ate_delta = row["ATE_proxy"]["mean"] - ref["ATE_proxy"]["mean"]
        drift_delta = row["drift_proxy"]["mean"] - ref["drift_proxy"]["mean"]
        path = row["path_ratio_proxy"]["mean"]
        rot_delta = row["rot_mean_deg"]["mean"] - ref["rot_mean_deg"]["mean"]
        tdir_delta = row["tdir_mean_deg"]["mean"] - ref["tdir_mean_deg"]["mean"]
        path_ok = float(gate["safe_path_ratio_min"]) <= path <= float(gate["safe_path_ratio_max"])
        all_path_fail = all_path_fail and not path_ok
        if ate_delta < 0.0 or drift_delta < 0.0:
            any_shape_signal = True
        folds_pass = 0
        for r in [x for x in per_fold if x["variant"] == variant]:
            fr = next(x for x in per_fold if x["fold_id"] == r["fold_id"] and x["variant"] == "A_s5_chain_reference")
            m = r["metrics"]
            a = fr["metrics"]
            if (
                m["ATE_proxy"] - a["ATE_proxy"] <= float(gate["max_ATE_worsening"])
                and m["drift_proxy"] - a["drift_proxy"] <= float(gate["max_drift_worsening"])
                and float(gate["safe_path_ratio_min"]) <= m["path_ratio_proxy"] <= float(gate["safe_path_ratio_max"])
                and m["rot_mean_deg"] <= a["rot_mean_deg"]
                and m["tdir_mean_deg"] < a["tdir_mean_deg"]
            ):
                folds_pass += 1
        passed = (
            ate_delta <= float(gate["max_ATE_worsening"])
            and drift_delta <= float(gate["max_drift_worsening"])
            and path_ok
            and rot_delta <= 0.0
            and tdir_delta < 0.0
            and folds_pass >= 2
            and row["num_folds_ok"] == int(config["train_cv"]["num_folds"])
        )
        if passed:
            passers.append(variant)
        decisions[variant] = {
            "delta_ATE_proxy": ate_delta,
            "delta_drift_proxy": drift_delta,
            "path_ratio_proxy": path,
            "delta_rot_mean": rot_delta,
            "delta_tdir_mean": tdir_delta,
            "folds_passing_direction": folds_pass,
            "gate_status": "PASS" if passed else "FAIL",
        }
    if passers:
        best = min(passers, key=lambda v: mean_cv[v]["ATE_proxy"]["mean"])
        return decisions, "MF1B-TRAIN-CV-GATE-PASS", best
    if all_path_fail:
        return decisions, "MF1B-TRAJECTORY-SHAPE-SIGNAL-SCALE-FAIL" if any_shape_signal else "MF1B-PATH-RATIO-GATE-FAIL", None
    return decisions, "NO_STABLE_MF1_CHAIN_GAIN", None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MF1b train-CV chain refiner.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    _run_guard()
    config = _read_json(_resolve(args.config))
    _check_locked(config)
    data = _load_dataset(_resolve(args.dataset))
    folds = _folds(data["chain_indices"].shape[0], int(config["train_cv"]["num_folds"]), int(config["train_cv"]["seed"]), int(config["train_cv"]["max_train_chains_per_fold"]), int(config["train_cv"]["max_val_chains_per_fold"]))
    out_root = _resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    variants = list(config["variants"])
    torch.manual_seed(int(config["train_cv"]["seed"]))
    np.random.seed(int(config["train_cv"]["seed"]))
    gen = torch.Generator().manual_seed(int(config["train_cv"]["seed"]))
    per_fold: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    dim = int(data["features"].shape[1])
    for fold in folds:
        train_ids = torch.tensor(fold["train_chain_ids"], dtype=torch.long)
        val_ids = torch.tensor(fold["val_chain_ids"], dtype=torch.long)
        train_rows = torch.unique(data["chain_indices"][train_ids].reshape(-1))
        mu = data["features"][train_rows].mean(dim=0, keepdim=True)
        sd = data["features"][train_rows].std(dim=0, keepdim=True).clamp_min(1.0e-6)
        data["x_std"] = (data["features"] - mu) / sd
        for variant in variants:
            model = _make_model(variant, dim)
            if model is None:
                metrics = _metrics(model, data, val_ids, variant)
                per_fold.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "metrics": metrics})
                logs.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "num_updates": 0, "train_loss_initial": None, "train_loss_final": None})
                continue
            opt = torch.optim.AdamW(model.parameters(), lr=float(config["train_cv"]["learning_rate"]), weight_decay=float(config["train_cv"]["weight_decay"]))
            first = train_ids[: min(int(config["train_cv"]["batch_size"]), train_ids.numel())]
            initial = float(_loss(model, data, first, variant, config["loss_weights"]).detach().item())
            final = initial
            for _ in range(int(config["train_cv"]["num_updates"])):
                pick = train_ids[torch.randint(0, train_ids.numel(), (int(config["train_cv"]["batch_size"]),), generator=gen)]
                opt.zero_grad(set_to_none=True)
                loss = _loss(model, data, pick, variant, config["loss_weights"])
                loss.backward()
                opt.step()
                final = float(loss.detach().item())
            metrics = _metrics(model, data, val_ids, variant)
            ckpt = out_root / f"fold_{fold['fold_id']}_{variant}.pt"
            torch.save({"fold_id": fold["fold_id"], "variant": variant, "input_dim": dim, "state_dict": model.state_dict(), "feature_mean": mu, "feature_std": sd}, ckpt)
            per_fold.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "metrics": metrics})
            logs.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "num_updates": int(config["train_cv"]["num_updates"]), "train_loss_initial": initial, "train_loss_final": final, "model_state_path": str(ckpt.relative_to(REPO_ROOT))})
    mean_cv = _summarize(per_fold, variants)
    decisions, classification, selected = _gate(config, mean_cv, per_fold)
    payload = {
        "experiment_name": config["experiment_name"],
        "base_candidate": config["base_candidate"],
        "locked_s5_metrics": config["locked_s5_metrics"],
        "folds": [{"fold_id": f["fold_id"], "num_train_chains": f["num_train_chains"], "num_val_chains": f["num_val_chains"]} for f in folds],
        "variants": variants,
        "dataset_metadata": data["metadata"],
        "train_logs": logs,
        "per_fold_metrics": per_fold,
        "mean_cv_metrics": mean_cv,
        "gate_decision": decisions,
        "selected_candidate_for_next_stage": selected,
        "final_classification": classification,
        "no_final_test": True,
        "s5_remains_final": True,
    }
    out_json = _resolve(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
