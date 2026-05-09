#!/usr/bin/env python3
"""Train-CV JRT1b prediction-space R/tdir refiner diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _build_eval_dataset
from jrt1_build_pair_dataset import FEATURE_NAMES, _build_s5_model, _validate_config
from jrt1_train_joint_rtdir_refiner import JointRefinerMLP, exp_so3, geodesic_rad


VARIANTS = {
    "A_s5_no_refiner_reference": {
        "description": "S5 reference, no refiner training",
        "weights": {"w_rot": 1.0, "w_tdir": 1.0, "w_couple": 0.2, "w_reg": 0.01},
        "hardcase_alpha": 0.0,
    },
    "B_rot_only_residual": {
        "description": "rotation residual only",
        "weights": {"w_rot": 1.0, "w_tdir": 1.0, "w_couple": 0.2, "w_reg": 0.01},
        "hardcase_alpha": 0.0,
    },
    "C_tdir_residual_cosine": {
        "description": "translation direction residual with cosine loss",
        "weights": {"w_rot": 0.0, "w_tdir": 1.0, "w_couple": 0.0, "w_reg": 0.01},
        "hardcase_alpha": 0.0,
    },
    "D_joint_rtdir_residual": {
        "description": "joint rotation and translation direction residual",
        "weights": {"w_rot": 1.0, "w_tdir": 1.0, "w_couple": 0.2, "w_reg": 0.01},
        "hardcase_alpha": 0.0,
    },
    "E_joint_rtdir_tdir_weighted": {
        "description": "joint residual with stronger translation direction weight",
        "weights": {"w_rot": 0.5, "w_tdir": 2.0, "w_couple": 0.2, "w_reg": 0.01},
        "hardcase_alpha": 0.0,
    },
    "F_joint_rtdir_tdir_hardcase_weighted": {
        "description": "joint residual with train-only hardcase translation direction weighting",
        "weights": {"w_rot": 0.5, "w_tdir": 2.0, "w_couple": 0.2, "w_reg": 0.01},
        "hardcase_alpha": 2.0,
    },
}
LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_indices(candidates: Sequence[int], cap: int) -> List[int]:
    vals = list(candidates)
    if cap <= 0 or len(vals) <= cap:
        return vals
    raw = np.linspace(0, len(vals) - 1, num=cap)
    out: List[int] = []
    seen = set()
    for v in raw:
        j = int(round(float(v)))
        j = max(0, min(len(vals) - 1, j))
        idx = vals[j]
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _make_folds(n: int, num_folds: int, seed: int, max_train: int, max_val: int) -> List[Dict[str, Any]]:
    if n < 2:
        raise RuntimeError(f"JRT1b needs at least 2 train-split pairs, got {n}")
    num_folds = max(2, min(int(num_folds), n))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n).tolist()
    chunks = [list(map(int, x.tolist())) for x in np.array_split(np.asarray(perm, dtype=np.int64), num_folds)]
    folds: List[Dict[str, Any]] = []
    all_set = set(range(n))
    for fold_id, val_raw in enumerate(chunks):
        val_set = set(val_raw)
        train_raw = sorted(all_set - val_set)
        val_idx = _select_indices(sorted(val_set), max_val)
        train_idx = _select_indices(train_raw, max_train)
        folds.append(
            {
                "fold_id": int(fold_id),
                "train_indices": train_idx,
                "val_indices": val_idx,
                "num_train_pairs": int(len(train_idx)),
                "num_val_pairs": int(len(val_idx)),
            }
        )
    return folds


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1.0e-12)).astype(np.float32)


def _extract_rows(model, ds, indices: Sequence[int], device: torch.device) -> List[Dict[str, Any]]:
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for offset, ds_idx in enumerate(indices):
            if offset == 0 or (offset + 1) % 128 == 0 or (offset + 1) == len(indices):
                print(f"[jrt1b-dataset] extracting {offset + 1}/{len(indices)}", flush=True)
            sample = ds[ds_idx]
            meta = manifest[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_val = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
            R_pred_t, t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_world)
            R_pred = R_pred_t.detach().float().cpu().numpy()[0].astype(np.float32)
            tdir_pred_t = aux.get("t_dir_out", t_pred_t) if isinstance(aux, dict) else t_pred_t
            tdir_pred = _unit(tdir_pred_t.detach().float().cpu().numpy()[0])
            tmag_pred = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            log_tmag_pred = float(math.log(max(tmag_pred, 1.0e-12)))
            R_gt = sample["R_gt"].detach().float().cpu().numpy().astype(np.float32)
            tdir_gt = _unit(sample["t_gt_dir"].detach().float().cpu().numpy())
            tmag_gt = float(sample["t_gt_mag"])
            k = int(meta.get("k", sample.get("meta", {}).get("k", -1)))
            features = np.concatenate(
                [
                    R_pred.reshape(-1),
                    tdir_pred.reshape(-1),
                    np.asarray([log_tmag_pred, dt_val, float(k)], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
            rows.append(
                {
                    "dataset_index": int(ds_idx),
                    "features": features,
                    "R_pred": R_pred,
                    "tdir_pred": tdir_pred,
                    "log_tmag_pred": log_tmag_pred,
                    "tmag_pred": tmag_pred,
                    "R_gt": R_gt,
                    "tdir_gt": tdir_gt,
                    "tmag_gt": tmag_gt,
                    "sequence_id": f"{meta.get('scene')}::{meta.get('seq')}",
                    "pair_id": f"{meta.get('scene')}::{meta.get('seq')}::{meta.get('tsA')}->{meta.get('tsB')}::k={k}",
                    "dt": dt_val,
                    "k": k,
                }
            )
    return rows


def _tensorize(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(np.asarray([r["features"] for r in rows], dtype=np.float32)),
        "R_pred": torch.tensor(np.asarray([r["R_pred"] for r in rows], dtype=np.float32)),
        "tdir_pred": torch.tensor(np.asarray([r["tdir_pred"] for r in rows], dtype=np.float32)),
        "log_tmag_pred": torch.tensor(np.asarray([r["log_tmag_pred"] for r in rows], dtype=np.float32)),
        "tmag_pred": torch.tensor(np.asarray([r["tmag_pred"] for r in rows], dtype=np.float32)),
        "R_gt": torch.tensor(np.asarray([r["R_gt"] for r in rows], dtype=np.float32)),
        "tdir_gt": torch.tensor(np.asarray([r["tdir_gt"] for r in rows], dtype=np.float32)),
        "tmag_gt": torch.tensor(np.asarray([r["tmag_gt"] for r in rows], dtype=np.float32)),
    }


def _standardize(features: torch.Tensor, train_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = features[train_local].mean(dim=0, keepdim=True)
    sd = features[train_local].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (features - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def _apply_variant(model: JointRefinerMLP | None, xb: torch.Tensor, R_pred: torch.Tensor, tdir_pred: torch.Tensor, variant: str):
    if model is None:
        delta = torch.zeros((xb.shape[0], 6), device=xb.device, dtype=xb.dtype)
    else:
        delta = model(xb)
    delta_rot = delta[:, :3]
    delta_tdir = delta[:, 3:6]
    if variant == "B_rot_only_residual":
        delta_tdir = torch.zeros_like(delta_tdir)
    if variant == "C_tdir_residual_cosine":
        delta_rot = torch.zeros_like(delta_rot)
    if variant == "A_s5_no_refiner_reference":
        delta_rot = torch.zeros_like(delta_rot)
        delta_tdir = torch.zeros_like(delta_tdir)
    R_ref = torch.matmul(exp_so3(delta_rot), R_pred.float())
    tdir_ref = F.normalize(tdir_pred.float() + delta_tdir.float(), dim=-1, eps=1.0e-6)
    return R_ref, tdir_ref, delta_rot, delta_tdir


def _loss(
    model: JointRefinerMLP,
    data: Dict[str, torch.Tensor],
    idx: torch.Tensor,
    variant: str,
    weights: Dict[str, float],
    hardcase_alpha: float,
) -> torch.Tensor:
    xb = data["x_std"][idx]
    R_ref, tdir_ref, delta_rot, delta_tdir = _apply_variant(model, xb, data["R_pred"][idx], data["tdir_pred"][idx], variant)
    rot_per = geodesic_rad(R_ref, data["R_gt"][idx])
    cos = (tdir_ref * F.normalize(data["tdir_gt"][idx], dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
    tdir_per = 1.0 - cos
    sample_w = torch.ones_like(tdir_per)
    if hardcase_alpha > 0.0:
        s5_cos = data["s5_tdir_cosine"][idx]
        sample_w = sample_w + float(hardcase_alpha) * (s5_cos < 0.25).float()
    l_rot = (rot_per * sample_w).sum() / sample_w.sum().clamp_min(1.0e-6)
    l_tdir = (tdir_per * sample_w).sum() / sample_w.sum().clamp_min(1.0e-6)
    l_couple = l_rot * l_tdir
    l_reg = (delta_rot.square().sum(dim=-1) + delta_tdir.square().sum(dim=-1)).mean()
    return (
        float(weights["w_rot"]) * l_rot
        + float(weights["w_tdir"]) * l_tdir
        + float(weights["w_couple"]) * l_couple
        + float(weights["w_reg"]) * l_reg
    )


def component_metrics(model, data: Dict[str, torch.Tensor], idx: torch.Tensor, variant: str) -> Dict[str, float]:
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
        "ATE_proxy": None,
        "drift_proxy": None,
        "path_ratio_proxy": None,
    }


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _summarize_cv(per_fold: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = [r for r in per_fold if r["variant"] == variant and r["status"] == "ok"]
        out[variant] = {
            key: _mean_std([row["metrics"][key] for row in rows])
            for key in [
                "rot_mean_deg",
                "rot_median_deg",
                "rot_p90_deg",
                "rot_max_deg",
                "tdir_mean_deg",
                "tdir_median_deg",
                "tdir_p90_deg",
                "tdir_max_deg",
                "tdir_mean_cosine",
                "tmag_mean_log_error",
                "tmag_median_log_error",
                "tmag_p90_log_error",
                "tmag_max_log_error",
            ]
        }
        out[variant]["num_folds_ok"] = len(rows)
    return out


def _gate_decision(config: Dict[str, Any], mean_cv: Dict[str, Dict[str, Any]], per_fold: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    gate = config["gate"]
    ref = mean_cv["A_s5_no_refiner_reference"]
    ref_tdir = float(ref["tdir_mean_deg"]["mean"])
    ref_cos = float(ref["tdir_mean_cosine"]["mean"])
    ref_rot = float(ref["rot_mean_deg"]["mean"])
    decisions: Dict[str, Any] = {}
    any_component_pass = False
    any_tdir_signal = False
    for variant, row in mean_cv.items():
        if variant == "A_s5_no_refiner_reference":
            decisions[variant] = {
                "tdir_improvement_deg": 0.0,
                "rot_worsening_deg": 0.0,
                "tdir_cos_improved": False,
                "trajectory_gate_available": False,
                "no_fold_collapse": row["num_folds_ok"] == int(config["train_cv"]["num_folds"]),
                "component_gate_pass": False,
                "gate_status": "REFERENCE",
            }
            continue
        tdir_improvement = ref_tdir - float(row["tdir_mean_deg"]["mean"])
        rot_worsening = float(row["rot_mean_deg"]["mean"]) - ref_rot
        cos_improved = float(row["tdir_mean_cosine"]["mean"]) > ref_cos
        fold_rows = [x for x in per_fold if x["variant"] == variant]
        no_fold_collapse = len(fold_rows) == int(config["train_cv"]["num_folds"]) and all(x["status"] == "ok" for x in fold_rows)
        component_pass = (
            tdir_improvement >= float(gate["min_tdir_mean_improvement_deg"])
            and cos_improved
            and rot_worsening <= float(gate["max_rot_mean_worsening_deg"])
            and no_fold_collapse
        )
        if tdir_improvement > 0.0 and cos_improved:
            any_tdir_signal = True
        if component_pass:
            any_component_pass = True
        decisions[variant] = {
            "tdir_improvement_deg": float(tdir_improvement),
            "rot_worsening_deg": float(rot_worsening),
            "tdir_cos_improved": bool(cos_improved),
            "trajectory_gate_available": False,
            "no_fold_collapse": bool(no_fold_collapse),
            "component_gate_pass": bool(component_pass),
            "gate_status": "COMPONENT-PASS-TRAJECTORY-UNAVAILABLE" if component_pass else "FAIL",
        }
    if any_component_pass:
        final = "TRAJECTORY-GATE-UNAVAILABLE"
    elif any_tdir_signal:
        final = "JRT1B-DIAGNOSTIC-SIGNAL-ONLY"
    else:
        final = "NO-STABLE-JOINT-RTDIR-REFINER-GAIN"
    return decisions, final


def _save_npz(path: Path, rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray([r["features"] for r in rows], dtype=np.float32),
        R_pred=np.asarray([r["R_pred"] for r in rows], dtype=np.float32),
        tdir_pred=np.asarray([r["tdir_pred"] for r in rows], dtype=np.float32),
        log_tmag_pred=np.asarray([r["log_tmag_pred"] for r in rows], dtype=np.float32),
        tmag_pred=np.asarray([r["tmag_pred"] for r in rows], dtype=np.float32),
        R_gt=np.asarray([r["R_gt"] for r in rows], dtype=np.float32),
        tdir_gt=np.asarray([r["tdir_gt"] for r in rows], dtype=np.float32),
        tmag_gt=np.asarray([r["tmag_gt"] for r in rows], dtype=np.float32),
        dataset_index=np.asarray([r["dataset_index"] for r in rows], dtype=np.int64),
        dt=np.asarray([r["dt"] for r in rows], dtype=np.float32),
        k=np.asarray([r["k"] for r in rows], dtype=np.int64),
        sequence_id=np.asarray([r["sequence_id"] for r in rows], dtype="U128"),
        pair_id=np.asarray([r["pair_id"] for r in rows], dtype="U256"),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, ensure_ascii=True)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run JRT1b train-CV refiner diagnostics without test GT.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    _validate_config(config)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    out_json = _resolve(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cv = config["train_cv"]
    torch.manual_seed(int(cv["seed"]))
    np.random.seed(int(cv["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, s5_policy, load_summary = _build_s5_model(config, device)
    train_ds = _build_eval_dataset(cfg, split="train")
    test_ds = _build_eval_dataset(cfg, split="test")
    folds = _make_folds(
        len(train_ds),
        int(cv["num_folds"]),
        int(cv["seed"]),
        int(cv["max_train_pairs_per_fold"]),
        int(cv["max_val_pairs_per_fold"]),
    )
    needed_indices = sorted({idx for fold in folds for idx in fold["train_indices"] + fold["val_indices"]})
    rows = _extract_rows(model, train_ds, needed_indices, device)
    index_to_local = {int(row["dataset_index"]): i for i, row in enumerate(rows)}
    dataset_summary = {
        "dataset_source": "RflyPanoPanoramaPairsEvalFixedKList train split with S5 policy predictions",
        "source_policy": str(config["base_policy_path"]),
        "source_policy_name": s5_policy["name"],
        "num_available_train_pairs": int(len(train_ds)),
        "num_available_test_pairs": int(len(test_ds)),
        "num_cached_train_split_pairs": int(len(rows)),
        "num_test_pairs_cached": 0,
        "feature_fields": FEATURE_NAMES,
        "label_fields": ["R_gt_3x3", "tdir_gt_3", "tmag_gt"],
        "feature_dim": len(FEATURE_NAMES),
        "label_dim": 13,
        "no_test_gt_used_for_training": True,
        "test_gt_cached": False,
        "folds": [
            {
                "fold_id": f["fold_id"],
                "num_train_pairs": f["num_train_pairs"],
                "num_val_pairs": f["num_val_pairs"],
            }
            for f in folds
        ],
        "load_missing": int(len(load_summary["missing"])),
        "load_unexpected": int(len(load_summary["unexpected"])),
    }
    (REPO_ROOT / "outputs" / "jrt1" / "JRT1b_train_cv_dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _save_npz(output_root / "JRT1b_train_cv_dataset_cache.npz", rows, dataset_summary)

    tensor_data = _tensorize(rows)
    s5_cos = (
        tensor_data["tdir_pred"].float() * F.normalize(tensor_data["tdir_gt"].float(), dim=-1, eps=1.0e-6)
    ).sum(dim=-1).clamp(-1.0, 1.0)
    tensor_data["s5_tdir_cosine"] = s5_cos
    input_dim = int(tensor_data["features"].shape[1])
    per_fold: List[Dict[str, Any]] = []
    train_logs: List[Dict[str, Any]] = []
    gen = torch.Generator().manual_seed(int(cv["seed"]))

    for fold in folds:
        train_local = torch.tensor([index_to_local[int(i)] for i in fold["train_indices"]], dtype=torch.long)
        val_local = torch.tensor([index_to_local[int(i)] for i in fold["val_indices"]], dtype=torch.long)
        x_std, mu, sd = _standardize(tensor_data["features"], train_local)
        data = dict(tensor_data)
        data["x_std"] = x_std
        fold_dir = output_root / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for variant, spec in VARIANTS.items():
            if variant == "A_s5_no_refiner_reference":
                metrics = component_metrics(None, data, val_local, variant)
                per_fold.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "metrics": metrics})
                train_logs.append(
                    {
                        "fold_id": fold["fold_id"],
                        "variant": variant,
                        "train_loss_initial": None,
                        "train_loss_final": None,
                        "num_updates": 0,
                        "status": "ok",
                        "error_if_any": None,
                    }
                )
                continue
            print(f"[jrt1b-train] fold={fold['fold_id']} variant={variant}", flush=True)
            model_refiner = JointRefinerMLP(input_dim)
            opt = torch.optim.AdamW(
                model_refiner.parameters(),
                lr=float(cv["learning_rate"]),
                weight_decay=float(cv["weight_decay"]),
            )
            first_idx = train_local[: min(int(cv["batch_size"]), train_local.numel())]
            loss_initial = float(
                _loss(
                    model_refiner,
                    data,
                    first_idx,
                    variant,
                    spec["weights"],
                    float(spec["hardcase_alpha"]),
                )
                .detach()
                .item()
            )
            loss_final = loss_initial
            for _step in range(int(cv["num_updates"])):
                pick = train_local[torch.randint(0, train_local.numel(), (int(cv["batch_size"]),), generator=gen)]
                opt.zero_grad(set_to_none=True)
                loss = _loss(model_refiner, data, pick, variant, spec["weights"], float(spec["hardcase_alpha"]))
                loss.backward()
                opt.step()
                loss_final = float(loss.detach().item())
            metrics = component_metrics(model_refiner, data, val_local, variant)
            ckpt_path = fold_dir / f"{variant}.pt"
            torch.save(
                {
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "input_dim": input_dim,
                    "state_dict": model_refiner.state_dict(),
                    "feature_mean": mu,
                    "feature_std": sd,
                    "num_updates": int(cv["num_updates"]),
                },
                ckpt_path,
            )
            per_fold.append({"fold_id": fold["fold_id"], "variant": variant, "status": "ok", "metrics": metrics})
            train_logs.append(
                {
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "train_loss_initial": float(loss_initial),
                    "train_loss_final": float(loss_final),
                    "num_updates": int(cv["num_updates"]),
                    "status": "ok",
                    "error_if_any": None,
                    "model_state_path": str(ckpt_path.relative_to(REPO_ROOT)),
                }
            )

    mean_cv = _summarize_cv(per_fold)
    gate_decision, final_classification = _gate_decision(config, mean_cv, per_fold)
    payload = {
        "experiment_name": config["experiment_name"],
        "base_candidate": config["base_candidate"],
        "locked_s5_metrics": config["locked_s5_metrics"],
        "folds": dataset_summary["folds"],
        "variants": [
            {"name": name, "description": spec["description"], "weights": spec["weights"], "hardcase_alpha": spec["hardcase_alpha"]}
            for name, spec in VARIANTS.items()
        ],
        "dataset_summary": dataset_summary,
        "train_logs": train_logs,
        "per_fold_metrics": per_fold,
        "mean_cv_metrics": mean_cv,
        "gate_decision": gate_decision,
        "trajectory_gate_available": False,
        "final_classification": final_classification,
        "final_test_run": False,
        "caveats": [
            "no final candidate selected",
            "no practical-ready claim",
            "S5 unchanged",
            "small sequence protocol",
            "component-only signal is insufficient for final clean claim",
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
