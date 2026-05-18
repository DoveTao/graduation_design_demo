#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from train360.core.config import Config
from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
try:
    from miniyaml import load_yaml_like
except ModuleNotFoundError:
    def load_yaml_like(path: Path) -> Dict[str, Any]:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
from models.struct360b_match_free_coarse_to_fine import (
    STRUCT360BMatchFreeCoarseToFineModel,
    count_parameters,
)
from train360.core.pose_head import matrix_geodesic_distance
from train360.core.train360d_pose_losses import train360d_pose_loss


T57B_REFERENCE = {
    "signed_tdir_mean_deg": 111.96493221327962,
    "anti_parallel_rate": 0.6746724890829694,
    "tmag_median_ratio": 0.1751560082454769,
    "path_ratio": 0.14878731297064046,
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(cmd: Sequence[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def _system_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def _subset_dataset(dataset: Dset2CCanonicalPairDataset, max_count: Optional[int], seed: int):
    if max_count is None or int(max_count) <= 0 or len(dataset) <= int(max_count):
        return dataset, {"subset_used": False, "subset_count": len(dataset), "original_count": len(dataset)}
    rng = np.random.default_rng(int(seed))
    idx = sorted(int(x) for x in rng.permutation(len(dataset))[: int(max_count)].tolist())
    return Subset(dataset, idx), {
        "subset_used": True,
        "subset_count": len(idx),
        "original_count": len(dataset),
        "subset_seed": int(seed),
    }


def _hist_from_dataset(dataset: Dset2CCanonicalPairDataset) -> Dict[str, Any]:
    ks: Dict[int, int] = {}
    adjacent = 0
    non_adjacent = 0
    tmags: List[float] = []
    for sample in dataset.samples:
        k = int(sample["k"])
        ks[k] = int(ks.get(k, 0)) + 1
        if k == 1 or str(sample.get("pair_type", "")) == "adjacent":
            adjacent += 1
        else:
            non_adjacent += 1
        tmags.append(float(sample["tmag"]))
    arr = np.asarray(tmags, dtype=np.float64)
    return {
        "pair_count": len(dataset.samples),
        "sequence_ids": sorted({str(s["sequence_id"]) for s in dataset.samples}),
        "k_histogram": {str(k): int(v) for k, v in sorted(ks.items())},
        "adjacent_pair_count": int(adjacent),
        "non_adjacent_pair_count": int(non_adjacent),
        "tmag_quantiles": {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        },
    }


def _extract_metrics_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _extract_base360_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    all_pairs = payload.get("pair_metrics", {}).get("all_pairs", {})
    return {
        "rot_mean_deg": all_pairs.get("rot_mean_deg"),
        "rot_median_deg": all_pairs.get("rot_median_deg"),
        "signed_tdir_mean_deg": all_pairs.get("signed_tdir_mean_deg"),
        "signed_tdir_median_deg": all_pairs.get("signed_tdir_median_deg"),
        "unsigned_tdir_mean_deg": all_pairs.get("unsigned_tdir_mean_deg"),
        "anti_parallel_rate": all_pairs.get("anti_parallel_rate"),
        "tmag_median_ratio": all_pairs.get("tmag_median_ratio"),
        "tmag_mean_ratio": all_pairs.get("tmag_mean_ratio"),
        "tmag_ratio_p10": all_pairs.get("tmag_p10_ratio"),
        "tmag_ratio_p90": all_pairs.get("tmag_p90_ratio"),
        "log_tmag_mae": all_pairs.get("log_tmag_mae"),
        "scale_collapse_rate": all_pairs.get("scale_collapse_rate"),
        "scale_explosion_rate": all_pairs.get("scale_explosion_rate"),
        "path_ratio": all_pairs.get("pair_component_path_ratio"),
        "pair_component_path_ratio": all_pairs.get("pair_component_path_ratio"),
        "coverage": all_pairs.get("pair_coverage"),
    }


def _compute_val_score(metrics: Mapping[str, Any], score_cfg: Optional[Mapping[str, Any]] = None, eps: float = 1.0e-6) -> float:
    score_cfg = score_cfg or {}
    signed = float(metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(metrics.get("anti_parallel_rate") or 1.0)
    tmag_ratio = float(metrics.get("tmag_median_ratio") or eps)
    path_ratio = float(metrics.get("path_ratio") or eps)
    rot = float(metrics.get("rot_mean_deg") or 0.0)
    return (
        float(score_cfg.get("signed_tdir_weight", 1.0)) * signed
        + float(score_cfg.get("anti_parallel_weight", 60.0)) * anti
        + float(score_cfg.get("tmag_ratio_weight", 20.0)) * abs(math.log(max(tmag_ratio, eps)))
        + float(score_cfg.get("path_ratio_weight", 10.0)) * abs(math.log(max(path_ratio, eps)))
        + float(score_cfg.get("rot_weight", 0.0)) * rot
    )


def _float_stats(vals: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _vector_angle_deg(a: np.ndarray, b: np.ndarray, *, absolute: bool = False) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / (na * nb))
    if absolute:
        c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _empty_pose_collector() -> Dict[str, List[float]]:
    return {
        "rot_deg": [],
        "signed_tdir": [],
        "unsigned_tdir": [],
        "anti_parallel": [],
        "tmag_ratio": [],
        "log_tmag_mae": [],
        "pred_lengths": [],
        "gt_lengths": [],
    }


def _append_pose_batch(
    collector: Dict[str, List[float]],
    *,
    R_pred: torch.Tensor,
    tdir_pred_B: torch.Tensor,
    tmag_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt_vec_B: torch.Tensor,
    tmag_epsilon: float,
) -> Tuple[int, int, int]:
    nan_count = 0
    inf_count = 0
    num_pairs = 0
    check_tensors = [R_pred, tdir_pred_B, tmag_pred, R_gt, t_gt_vec_B]
    nan_count += sum(int(torch.isnan(t).sum().item()) for t in check_tensors)
    inf_count += sum(int(torch.isinf(t).sum().item()) for t in check_tensors)

    rot_deg = (
        matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
    )
    tdir_np = tdir_pred_B.detach().float().cpu().numpy()
    tmag_np = tmag_pred.detach().float().view(-1).cpu().numpy()
    t_gt_np = t_gt_vec_B.detach().float().cpu().numpy()
    gt_mag_np = torch.linalg.norm(t_gt_vec_B.float(), dim=-1).detach().cpu().numpy()

    for i in range(R_pred.shape[0]):
        num_pairs += 1
        collector["rot_deg"].append(float(rot_deg[i]))
        signed = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=False)
        unsigned = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=True)
        if signed is not None:
            collector["signed_tdir"].append(float(signed))
            collector["anti_parallel"].append(1.0 if float(np.dot(tdir_np[i], t_gt_np[i])) < 0.0 else 0.0)
        if unsigned is not None:
            collector["unsigned_tdir"].append(float(unsigned))
        gt_mag = max(float(gt_mag_np[i]), float(tmag_epsilon))
        pred_mag = max(float(tmag_np[i]), float(tmag_epsilon))
        collector["tmag_ratio"].append(float(pred_mag / gt_mag))
        collector["log_tmag_mae"].append(abs(math.log(pred_mag) - math.log(gt_mag)))
        collector["pred_lengths"].append(pred_mag)
        collector["gt_lengths"].append(gt_mag)
    return num_pairs, nan_count, inf_count


def _summarize_pose_metrics(
    collector: Dict[str, List[float]],
    *,
    num_pairs: int,
    dataset_size: int,
    nan_count: int,
    inf_count: int,
) -> Dict[str, Any]:
    rot_stats = _float_stats(collector["rot_deg"])
    signed_stats = _float_stats(collector["signed_tdir"])
    unsigned_stats = _float_stats(collector["unsigned_tdir"])
    tmag_ratio_stats = _float_stats(collector["tmag_ratio"])
    return {
        "count": int(num_pairs),
        "coverage": float(num_pairs / max(dataset_size, 1)),
        "rot_mean_deg": rot_stats["mean"],
        "rot_median_deg": rot_stats["median"],
        "rot_p90_deg": rot_stats["p90"],
        "signed_tdir_mean_deg": signed_stats["mean"],
        "signed_tdir_median_deg": signed_stats["median"],
        "signed_tdir_p90_deg": signed_stats["p90"],
        "unsigned_tdir_mean_deg": unsigned_stats["mean"],
        "unsigned_tdir_median_deg": unsigned_stats["median"],
        "unsigned_tdir_p90_deg": unsigned_stats["p90"],
        "anti_parallel_rate": float(np.mean(np.asarray(collector["anti_parallel"], dtype=np.float64))) if collector["anti_parallel"] else None,
        "tmag_ratio_p10": tmag_ratio_stats["p10"],
        "tmag_median_ratio": tmag_ratio_stats["median"],
        "tmag_ratio_p50": tmag_ratio_stats["median"],
        "tmag_mean_ratio": tmag_ratio_stats["mean"],
        "tmag_ratio_p90": tmag_ratio_stats["p90"],
        "tmag_p90_ratio": tmag_ratio_stats["p90"],
        "log_tmag_mae": float(np.mean(np.asarray(collector["log_tmag_mae"], dtype=np.float64))) if collector["log_tmag_mae"] else None,
        "scale_collapse_rate": float(np.mean(np.asarray([1.0 if float(v) < 0.1 else 0.0 for v in collector["tmag_ratio"]], dtype=np.float64))) if collector["tmag_ratio"] else None,
        "scale_explosion_rate": float(np.mean(np.asarray([1.0 if float(v) > 10.0 else 0.0 for v in collector["tmag_ratio"]], dtype=np.float64))) if collector["tmag_ratio"] else None,
        "path_ratio": float(sum(collector["pred_lengths"]) / max(sum(collector["gt_lengths"]), 1.0e-12)) if collector["gt_lengths"] else None,
        "path_length_pred": float(sum(collector["pred_lengths"])) if collector["pred_lengths"] else None,
        "path_length_gt": float(sum(collector["gt_lengths"])) if collector["gt_lengths"] else None,
        "nan_inf_count": int(nan_count + inf_count),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "ate_none": None,
        "ate_se3": None,
        "ate_sim3": None,
    }


@torch.no_grad()
def evaluate_struct360b_pose(
    model: STRUCT360BMatchFreeCoarseToFineModel,
    loader: DataLoader,
    device: torch.device,
    *,
    tmag_epsilon: float,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    final_collector = _empty_pose_collector()
    coarse_collector = _empty_pose_collector()
    residual_rot_deg: List[float] = []
    residual_tdir_norm: List[float] = []
    residual_log_tmag_abs: List[float] = []
    residual_gate: List[float] = []
    nan_count_final = 0
    inf_count_final = 0
    nan_count_coarse = 0
    inf_count_coarse = 0
    num_pairs_final = 0
    num_pairs_coarse = 0
    alpha_val: Optional[float] = None
    beta_val: Optional[float] = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt_vec = batch["t_gt_vec"].to(device, non_blocking=True)
        dt_world = batch["meta"]["dt_world"]
        if torch.is_tensor(dt_world):
            dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
        else:
            dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)

        R_pred, _t_local, aux = model(IA, IB, dt_world=dt_world_t)
        final_added, final_nan, final_inf = _append_pose_batch(
            final_collector,
            R_pred=R_pred,
            tdir_pred_B=aux["t_dir_out"],
            tmag_pred=aux["t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt_vec,
            tmag_epsilon=tmag_epsilon,
        )
        coarse_added, coarse_nan, coarse_inf = _append_pose_batch(
            coarse_collector,
            R_pred=aux["coarse_R"],
            tdir_pred_B=aux["coarse_t_dir_out"],
            tmag_pred=aux["coarse_t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt_vec,
            tmag_epsilon=tmag_epsilon,
        )
        num_pairs_final += final_added
        nan_count_final += final_nan
        inf_count_final += final_inf
        num_pairs_coarse += coarse_added
        nan_count_coarse += coarse_nan
        inf_count_coarse += coarse_inf

        residual_rot_deg.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
        residual_tdir_norm.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
        residual_log_tmag_abs.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
        residual_gate.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
        alpha_val = float(aux["alpha"].detach().cpu().item())
        beta_val = float(aux["beta"].detach().cpu().item())

    final_metrics = _summarize_pose_metrics(
        final_collector,
        num_pairs=num_pairs_final,
        dataset_size=len(loader.dataset),
        nan_count=nan_count_final,
        inf_count=inf_count_final,
    )
    coarse_metrics = _summarize_pose_metrics(
        coarse_collector,
        num_pairs=num_pairs_coarse,
        dataset_size=len(loader.dataset),
        nan_count=nan_count_coarse,
        inf_count=inf_count_coarse,
    )
    return {
        "final_metrics": final_metrics,
        "coarse_metrics": coarse_metrics,
        "residual_stats": {
            "delta_rot_mean_deg": _float_stats(residual_rot_deg)["mean"],
            "delta_tdir_norm_mean": _float_stats(residual_tdir_norm)["mean"],
            "delta_log_tmag_abs_mean": _float_stats(residual_log_tmag_abs)["mean"],
        },
        "gate_stats": {
            "mean": _float_stats(residual_gate)["mean"],
            "median": _float_stats(residual_gate)["median"],
            "max": _float_stats(residual_gate)["max"],
        },
        "alpha": alpha_val,
        "beta": beta_val,
    }


def _inject_struct360b_cfg(base_cfg: Dict[str, Any], train_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base_cfg)
    out["H"] = int(train_cfg["data"]["image_hw"][0])
    out["W"] = int(train_cfg["data"]["image_hw"][1])
    out["use_fine_stage"] = True
    out["use_coupled_pose_residual_head"] = False
    out["use_depth_branch"] = False
    out["struct360b_fine_heads"] = int(train_cfg["model"]["struct360b"]["fine_heads"])
    out["struct360b_fine_dropout"] = float(train_cfg["model"]["struct360b"]["fine_dropout"])
    out["struct360b_fine_mlp_ratio"] = float(train_cfg["model"]["struct360b"]["fine_mlp_ratio"])
    out["struct360b_refiner_gate_bias"] = float(train_cfg["model"]["struct360b"]["refiner_gate_bias"])
    out["struct360b_residual_hidden_dim"] = int(train_cfg["model"]["struct360b"]["residual_hidden_dim"])
    out["struct360b_delta_rot_scale"] = float(train_cfg["model"]["struct360b"]["delta_rot_scale"])
    out["struct360b_delta_tdir_scale"] = float(train_cfg["model"]["struct360b"]["delta_tdir_scale"])
    out["struct360b_delta_log_tmag_scale"] = float(train_cfg["model"]["struct360b"]["delta_log_tmag_scale"])
    out["struct360b_residual_gate_bias"] = float(train_cfg["model"]["struct360b"]["residual_gate_bias"])
    out["struct360b_residual_gate_max"] = float(train_cfg["model"]["struct360b"]["residual_gate_max"])
    out["struct360b_alpha"] = float(train_cfg["model"]["struct360b"]["alpha"])
    out["struct360b_beta"] = float(train_cfg["model"]["struct360b"]["beta"])
    return out


def _load_model_with_status(
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict_attempt: bool,
    train_cfg: Mapping[str, Any],
) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Config, Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    cfg_dict = _inject_struct360b_cfg(dict(cfg_dict), train_cfg)
    cfg = _cfg_from_dict(cfg_dict)
    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    strict_ok = False
    strict_error = None
    strict_missing: List[str] = []
    strict_unexpected: List[str] = []
    if strict_attempt:
        try:
            missing, unexpected = model.load_state_dict(state, strict=True)
            strict_missing = list(missing)
            strict_unexpected = list(unexpected)
            strict_ok = True
        except Exception as exc:
            strict_error = f"{type(exc).__name__}: {exc}"
    if strict_ok:
        load_status = {
            "status": "strict",
            "strict_attempted": True,
            "strict_ok": True,
            "missing_keys": strict_missing,
            "unexpected_keys": strict_unexpected,
            "strict_error": None,
            "skipped_shape_mismatch": [],
        }
        init_status = {"newly_initialized_params": [], "checkpoint_loaded_params": list(state.keys())}
        return model, cfg, load_status, init_status

    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    model_state = model.state_dict()
    filtered = {}
    skipped_shape_mismatch: List[str] = []
    for key, value in state.items():
        if key in model_state and hasattr(value, "shape") and model_state[key].shape != value.shape:
            skipped_shape_mismatch.append(key)
            continue
        if key in model_state:
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    load_status = {
        "status": "non-strict",
        "strict_attempted": bool(strict_attempt),
        "strict_ok": False,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "strict_error": strict_error,
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }
    init_status = {
        "newly_initialized_params": list(missing),
        "checkpoint_loaded_params": sorted(filtered.keys()),
    }
    return model, cfg, load_status, init_status


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_cfg: Config,
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "cfg": dict(ckpt_cfg.__dict__),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _run_prechecks(cfg: Mapping[str, Any], cfg_path: Path) -> Dict[str, Any]:
    git_status_short = _git(["git", "status", "--short"]).splitlines()
    branch = _git(["git", "branch", "--show-current"])
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    commit = _git(["git", "rev-parse", "HEAD"])
    allowed_branches = [str(x) for x in cfg["prechecks"]["expected_branches"]]
    blockers: List[str] = []
    if branch not in allowed_branches:
        blockers.append(f"current branch mismatch: {branch} not in {allowed_branches}")

    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    if disk_free_gb <= 5.0:
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB <= 5.0 GiB")

    inputs = cfg["inputs"]
    required_paths = {name: (REPO_ROOT / rel) for name, rel in inputs.items()}
    for name, path in required_paths.items():
        if name.endswith("_checkpoint"):
            if not path.is_file():
                blockers.append(f"missing required file: {name} -> {path}")
        elif name.endswith("_metrics") or name.endswith("_report") or name == "struct360a_summary" or name in {"dataset_adapter", "train_manifest", "val_manifest", "test_manifest", "hygiene_json"}:
            if not path.is_file():
                blockers.append(f"missing required file: {name} -> {path}")

    adapter_text = required_paths["dataset_adapter"].read_text(encoding="utf-8")
    adapter_text_norm = " ".join(adapter_text.split())
    if "manifest-native" not in adapter_text_norm and "jsonl manifest and never scans raw sequence directories" not in adapter_text_norm:
        blockers.append("dataset adapter no longer advertises manifest-native / no-scan behavior")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to use raw sequence globbing")

    image_hw = tuple(int(x) for x in cfg["data"]["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(
        str(required_paths["train_manifest"]),
        expected_split="train",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    ds_val = Dset2CCanonicalPairDataset(
        str(required_paths["val_manifest"]),
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        str(required_paths["test_manifest"]),
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    split_audit = summarize_manifest_group([ds_train, ds_val, ds_test])
    if split_audit["has_overlap"]:
        blockers.append(f"sequence overlap detected: {split_audit['sequence_overlap']}")

    train360c_val = _extract_metrics_payload(_read_json(required_paths["train360c_val_metrics"]))
    train360c_test = _extract_metrics_payload(_read_json(required_paths["train360c_test_metrics"]))
    train360d_val = _extract_metrics_payload(_read_json(required_paths["train360d_val_metrics"]))
    train360d_test = _extract_metrics_payload(_read_json(required_paths["train360d_test_metrics"]))
    train360h_val = _extract_metrics_payload(_read_json(required_paths["train360h_val_metrics"]))
    train360h_test = _extract_metrics_payload(_read_json(required_paths["train360h_test_metrics"]))
    struct360a_val = _extract_metrics_payload(_read_json(required_paths["struct360a_val_metrics"]))
    struct360a_test = _extract_metrics_payload(_read_json(required_paths["struct360a_test_metrics"]))
    base360d_val = _extract_base360_metrics(_read_json(required_paths["base360d_val_metrics"]))
    base360d_test = _extract_base360_metrics(_read_json(required_paths["base360d_test_metrics"]))
    return {
        "blockers": blockers,
        "git_status_short": git_status_short,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_commit": commit,
        "disk_free_gb": disk_free_gb,
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "dataset_histograms": {
            "train": _hist_from_dataset(ds_train),
            "val": _hist_from_dataset(ds_val),
            "test": _hist_from_dataset(ds_test),
        },
        "split_audit": split_audit,
        "required_paths": {k: str(v) for k, v in required_paths.items()},
        "train360c_val_metrics": train360c_val,
        "train360c_test_metrics": train360c_test,
        "train360d_val_metrics": train360d_val,
        "train360d_test_metrics": train360d_test,
        "train360h_val_metrics": train360h_val,
        "train360h_test_metrics": train360h_test,
        "struct360a_val_metrics": struct360a_val,
        "struct360a_test_metrics": struct360a_test,
        "base360d_val_metrics": base360d_val,
        "base360d_test_metrics": base360d_test,
    }


def _write_blocker_artifacts(cfg: Mapping[str, Any], blocker_lines: List[str], precheck: Mapping[str, Any]) -> None:
    outputs = cfg["outputs"]
    report_path = REPO_ROOT / outputs["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = ["# STRUCT360B blocked before training", "", "## Blockers"]
    body.extend([f"- {line}" for line in blocker_lines])
    report_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    blocker_payload = {
        "task_name": str(cfg.get("task_name", "STRUCT360B_main_Dinit")),
        "training_executed": False,
        "checkpoint_saved": False,
        "blockers": blocker_lines,
        "precheck": {
            "git_status_short": list(precheck.get("git_status_short", [])),
            "branch": precheck.get("branch"),
            "branch_vv": list(precheck.get("branch_vv", [])),
            "git_commit": precheck.get("git_commit"),
            "disk_free_gb": precheck.get("disk_free_gb"),
            "dataset_histograms": precheck.get("dataset_histograms"),
            "split_audit": precheck.get("split_audit"),
            "required_paths": precheck.get("required_paths"),
        },
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], blocker_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], blocker_payload)


def _coarse_status(test_metrics: Mapping[str, Any], ref_metrics: Mapping[str, Any]) -> str:
    signed = float(test_metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(test_metrics.get("anti_parallel_rate") or 1.0)
    tmag = float(test_metrics.get("tmag_median_ratio") or 0.0)
    path = float(test_metrics.get("path_ratio") or 0.0)
    ref_signed = float(ref_metrics.get("signed_tdir_mean_deg") or float("inf"))
    ref_anti = float(ref_metrics.get("anti_parallel_rate") or 1.0)
    ref_tmag = float(ref_metrics.get("tmag_median_ratio") or 0.0)
    ref_path = float(ref_metrics.get("path_ratio") or 0.0)
    improved = 0
    if signed < ref_signed:
        improved += 1
    if anti < ref_anti:
        improved += 1
    if tmag > ref_tmag:
        improved += 1
    if path > ref_path:
        improved += 1
    if improved >= 3:
        return "better"
    if improved >= 1:
        return "partial"
    return "worse"


def _classify(
    test_metrics: Mapping[str, Any],
    *,
    train360c_test: Mapping[str, Any],
    train360d_test: Mapping[str, Any],
    train360h_test: Mapping[str, Any],
) -> str:
    signed = float(test_metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(test_metrics.get("anti_parallel_rate") or 1.0)
    tmag = float(test_metrics.get("tmag_median_ratio") or 0.0)
    path = float(test_metrics.get("path_ratio") or 0.0)
    nan_inf = int(test_metrics.get("nan_inf_count") or 0)
    if (
        nan_inf > 0
        or signed > float(train360c_test.get("signed_tdir_mean_deg") or float("inf"))
        or anti > float(train360c_test.get("anti_parallel_rate") or 1.0)
        or path < 0.50
    ):
        return "regression"
    if (
        signed <= float(train360d_test.get("signed_tdir_mean_deg") or float("inf"))
        and anti <= float(train360d_test.get("anti_parallel_rate") or 1.0)
        and tmag > float(train360d_test.get("tmag_median_ratio") or 0.0)
        and path > float(train360d_test.get("path_ratio") or 0.0)
    ):
        return "balanced_success"
    if signed <= 47.0 and anti <= 0.205 and tmag >= 0.80 and path >= 0.62:
        return "balanced_success"
    if signed <= float(train360d_test.get("signed_tdir_mean_deg") or float("inf")) and anti <= float(train360d_test.get("anti_parallel_rate") or 1.0):
        return "direction_best"
    if tmag >= float(train360h_test.get("tmag_median_ratio") or 0.0) and path >= float(train360h_test.get("path_ratio") or 0.0):
        return "scale_best"
    return "partial"


def _recommendation(classification: str) -> str:
    if classification == "balanced_success":
        return "proceed_to_TRAIN360I_final_retrain_with_best_hparams"
    if classification == "direction_best":
        return "proceed_to_STRUCT360C_rotation_aware_attention_bias"
    if classification == "scale_best":
        return "keep_TRAIN360H_scale_best"
    if classification == "regression":
        return "rollback_to_TRAIN360D_direction_best"
    return "proceed_to_STRUCT360C_rotation_aware_attention_bias"


def _build_optimizer(model: STRUCT360BMatchFreeCoarseToFineModel, cfg: Mapping[str, Any]) -> Tuple[torch.optim.Optimizer, List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    fine_prefixes = ("module2.patch_embed_f", "module2.enc_f", "module2.pos_enc", "struct360b_")
    coarse_prefixes = ("module2.patch_embed_c", "module2.enc_c", "module2.patch_embed_c_t", "module2.enc_c_t", "coarse")
    fine_params: List[torch.nn.Parameter] = []
    coarse_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("fine.", "direct_head", "coupled_pose_head", "depth", "depth_token_proj")):
            param.requires_grad = False
            continue
        if name.startswith(fine_prefixes):
            fine_params.append(param)
        elif name.startswith(coarse_prefixes):
            coarse_params.append(param)
        else:
            coarse_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": coarse_params, "lr": float(cfg["training"]["coarse_lr"])},
            {"params": fine_params, "lr": float(cfg["training"]["fine_lr"])},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    return optimizer, coarse_params, fine_params


def _set_stage_trainability(
    *,
    epoch: int,
    model: STRUCT360BMatchFreeCoarseToFineModel,
    coarse_params: Sequence[torch.nn.Parameter],
    fine_params: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    train_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    total_epochs = int(train_cfg["epochs"])
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    if epoch <= warmup_epochs:
        stage = "fine_only_warmup"
        for p in coarse_params:
            p.requires_grad = False
        for p in fine_params:
            p.requires_grad = True
        coarse_lr = float(train_cfg.get("warmup_coarse_lr", 0.0))
        fine_lr = float(train_cfg.get("warmup_fine_lr", train_cfg["fine_lr"]))
    else:
        stage = "partial_unfreeze"
        for p in coarse_params:
            p.requires_grad = True
        for p in fine_params:
            p.requires_grad = True
        denom = max(total_epochs - warmup_epochs - 1, 1)
        progress = float(epoch - warmup_epochs - 1) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        coarse_base = float(train_cfg["coarse_lr"])
        fine_base = float(train_cfg["fine_lr"])
        coarse_lr = min_lr + (coarse_base - min_lr) * cosine
        fine_lr = min_lr + (fine_base - min_lr) * cosine
    optimizer.param_groups[0]["lr"] = float(coarse_lr)
    optimizer.param_groups[1]["lr"] = float(fine_lr)
    return {"stage": stage, "coarse_lr": float(coarse_lr), "fine_lr": float(fine_lr)}


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "struct360b_match_free_coarse_to_fine.yaml")
    cfg = load_yaml_like(cfg_path)
    start_time = time.time()
    precheck = _run_prechecks(cfg, cfg_path)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck["blockers"], precheck)
        raise RuntimeError("STRUCT360B precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    _seed_everything(int(cfg["training"]["seed"]))
    use_cuda = bool(cfg["model"]["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    system_info = _system_info(device)

    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    run_test_during_train = bool(eval_cfg.get("run_test", True))

    ds_train = precheck["datasets"]["train"]
    ds_val = precheck["datasets"]["val"]
    ds_test = precheck["datasets"]["test"]
    train_subset, subset_info = _subset_dataset(ds_train, max_count=data_cfg.get("train_subset_max"), seed=int(train_cfg["seed"]))

    train_loader = DataLoader(train_subset, batch_size=int(data_cfg["train_batch_size"]), shuffle=bool(data_cfg["shuffle_train"]), num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=int(data_cfg["eval_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=int(data_cfg["eval_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)

    init_ckpt_path = REPO_ROOT / inputs["init_checkpoint"]
    model, ckpt_cfg, load_status, init_status = _load_model_with_status(
        init_ckpt_path,
        device,
        strict_attempt=bool(model_cfg["strict_load_attempt"]),
        train_cfg=cfg,
    )
    param_counts = count_parameters(model)
    optimizer, coarse_params, fine_params = _build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler() if bool(train_cfg["amp"]) and device.type == "cuda" else None

    checkpoint_dir = REPO_ROOT / outputs["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = checkpoint_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    (checkpoint_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    best_val_score = float("inf")
    best_epoch = -1
    best_val_eval: Optional[Dict[str, Any]] = None
    best_checkpoint_path = checkpoint_dir / "best_val.pt"

    metadata_common = {
        "task_name": str(cfg.get("task_name", "STRUCT360B_main_Dinit")),
        "init_checkpoint": str(init_ckpt_path),
        "git_branch": precheck["branch"],
        "git_commit": precheck["git_commit"],
        "parameter_count": param_counts,
        "compliance_flags": {
            "train360c_checkpoint_modified": False,
            "train360d_checkpoint_modified": False,
            "train360h_checkpoint_modified": False,
            "struct360a_checkpoint_modified": False,
            "explicit_matching_used": False,
            "confidence_matrix_used_for_matching": False,
            "mnn_match_selection_used": False,
            "match_list_output": False,
            "ransac_used": False,
            "pnp_used": False,
            "bundle_adjustment_used": False,
            "hkust_360dvo_teacher_used": False,
            "base360_outputs_used_as_training_input": False,
            "train_manifest_used": True,
            "val_manifest_used_for_selection_only": True,
            "test_manifest_used_for_final_eval_only": True,
            "direct_glob_data_360dvo_sequences": False,
            "random_pair_split_used": False,
            "s5_locked_metrics_modified": False,
            "large_checkpoints_committed_to_git": False,
        },
    }
    component_cfg = loss_cfg.get("components", {})

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        stage_info = _set_stage_trainability(
            epoch=epoch,
            model=model,
            coarse_params=coarse_params,
            fine_params=fine_params,
            optimizer=optimizer,
            train_cfg=train_cfg,
        )
        model.train()
        epoch_losses = {
            "loss_total": [],
            "loss_final": [],
            "loss_coarse_aux": [],
            "loss_residual_reg": [],
            "loss_rot": [],
            "loss_tdir": [],
            "loss_tmag": [],
            "loss_scale_stability": [],
        }
        gate_values: List[float] = []
        delta_rot_deg_values: List[float] = []
        delta_tdir_norm_values: List[float] = []
        delta_log_abs_values: List[float] = []
        obs_weights_all: List[float] = []
        k_weights_all: List[float] = []
        train_nan_inf_count = 0

        for batch in train_loader:
            IA = batch["IA"].to(device, non_blocking=True)
            IB = batch["IB"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)
            t_gt_vec = batch["t_gt_vec"].to(device, non_blocking=True)
            tmag_gt = batch["t_gt_mag"].to(device, non_blocking=True)
            k_tensor = batch["k"].to(device, non_blocking=True)
            dt_world = batch["meta"]["dt_world"]
            if torch.is_tensor(dt_world):
                dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
            else:
                dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                R_pred, _t_pred_local, aux = model(IA, IB, dt_world=dt_world_t)
                final_loss = train360d_pose_loss(
                    R_pred=R_pred,
                    tdir_pred_B=aux["t_dir_out"],
                    tmag_pred=aux["t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    k_step_config=loss_cfg["k_step_balancing"],
                    observability_config=loss_cfg["observability"],
                    scale_config=loss_cfg["scale_stability"],
                    rot_weight=float(loss_cfg["rot_weight"]),
                    tdir_weight=float(loss_cfg["tdir_weight"]),
                    tmag_weight=float(loss_cfg["tmag_weight"]),
                    scale_stability_weight=float(loss_cfg["scale_stability_weight"]),
                    tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
                    tmag_epsilon=float(data_cfg["tmag_epsilon"]),
                    enable_observability=bool(component_cfg.get("enable_observability", True)),
                    enable_k_step_balancing=bool(component_cfg.get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(component_cfg.get("enable_scale_stabilization", True)),
                )
                coarse_loss = train360d_pose_loss(
                    R_pred=aux["coarse_R"],
                    tdir_pred_B=aux["coarse_t_dir_out"],
                    tmag_pred=aux["coarse_t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    k_step_config=loss_cfg["k_step_balancing"],
                    observability_config=loss_cfg["observability"],
                    scale_config=loss_cfg["scale_stability"],
                    rot_weight=float(loss_cfg["rot_weight"]),
                    tdir_weight=float(loss_cfg["tdir_weight"]),
                    tmag_weight=float(loss_cfg["tmag_weight"]),
                    scale_stability_weight=float(loss_cfg["scale_stability_weight"]),
                    tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
                    tmag_epsilon=float(data_cfg["tmag_epsilon"]),
                    enable_observability=bool(component_cfg.get("enable_observability", True)),
                    enable_k_step_balancing=bool(component_cfg.get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(component_cfg.get("enable_scale_stabilization", True)),
                )
                residual_reg = model.residual_regularization(aux)
                loss_total = (
                    final_loss["loss_total"]
                    + float(loss_cfg["coarse_aux_weight"]) * coarse_loss["loss_total"]
                    + float(loss_cfg["residual_reg_weight"]) * residual_reg["loss"]
                )
            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                optimizer.step()

            epoch_losses["loss_total"].append(float(loss_total.detach().cpu()))
            epoch_losses["loss_final"].append(float(final_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_coarse_aux"].append(float(coarse_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_residual_reg"].append(float(residual_reg["loss"].detach().cpu()))
            epoch_losses["loss_rot"].append(float(final_loss["loss_rot"].detach().cpu()))
            epoch_losses["loss_tdir"].append(float(final_loss["loss_tdir"].detach().cpu()))
            epoch_losses["loss_tmag"].append(float(final_loss["loss_tmag"].detach().cpu()))
            epoch_losses["loss_scale_stability"].append(float(final_loss["loss_scale_stability"].detach().cpu()))
            gate_values.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
            delta_rot_deg_values.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
            delta_tdir_norm_values.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
            delta_log_abs_values.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
            obs_weights_all.extend(final_loss["obs_weights"].detach().cpu().numpy().astype(np.float64).tolist())
            k_weights_all.extend(final_loss["k_weights"].detach().cpu().numpy().astype(np.float64).tolist())
            train_nan_inf_count += int(sum((~torch.isfinite(v)).sum().item() for v in [R_pred, aux["t_dir_out"], aux["t_mag"], loss_total]))

        val_eval = evaluate_struct360b_pose(
            model,
            val_loader,
            device,
            tmag_epsilon=float(data_cfg["tmag_epsilon"]),
            max_batches=eval_cfg.get("max_val_batches"),
        )
        val_final = val_eval["final_metrics"]
        val_score = _compute_val_score(val_final, score_cfg=eval_cfg.get("selection_score"), eps=float(data_cfg["tmag_epsilon"]))
        epoch_log = {
            "epoch": int(epoch),
            "stage": stage_info["stage"],
            "lr_coarse": stage_info["coarse_lr"],
            "lr_fine": stage_info["fine_lr"],
            "train_loss_total": float(np.mean(epoch_losses["loss_total"])) if epoch_losses["loss_total"] else None,
            "train_loss_final": float(np.mean(epoch_losses["loss_final"])) if epoch_losses["loss_final"] else None,
            "train_loss_coarse_aux": float(np.mean(epoch_losses["loss_coarse_aux"])) if epoch_losses["loss_coarse_aux"] else None,
            "train_loss_residual_reg": float(np.mean(epoch_losses["loss_residual_reg"])) if epoch_losses["loss_residual_reg"] else None,
            "train_loss_rot": float(np.mean(epoch_losses["loss_rot"])) if epoch_losses["loss_rot"] else None,
            "train_loss_tdir": float(np.mean(epoch_losses["loss_tdir"])) if epoch_losses["loss_tdir"] else None,
            "train_loss_tmag": float(np.mean(epoch_losses["loss_tmag"])) if epoch_losses["loss_tmag"] else None,
            "train_loss_scale_stability": float(np.mean(epoch_losses["loss_scale_stability"])) if epoch_losses["loss_scale_stability"] else None,
            "train_nan_inf_count": int(train_nan_inf_count),
            "val_score": float(val_score),
            "val_final_metrics": val_final,
            "val_coarse_metrics": val_eval["coarse_metrics"],
            "residual_stats": val_eval["residual_stats"],
            "gate_stats": val_eval["gate_stats"],
            "obs_weight_stats": _float_stats(obs_weights_all),
            "k_weight_stats": _float_stats(k_weights_all),
            "train_gate_stats": _float_stats(gate_values),
            "train_delta_rot_deg_stats": _float_stats(delta_rot_deg_values),
            "train_delta_tdir_norm_stats": _float_stats(delta_tdir_norm_values),
            "train_delta_log_tmag_abs_stats": _float_stats(delta_log_abs_values),
        }
        _append_jsonl(train_log_path, epoch_log)
        if val_score < best_val_score and int(val_final["nan_inf_count"]) == 0:
            best_val_score = float(val_score)
            best_epoch = int(epoch)
            best_val_eval = dict(val_eval)
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                ckpt_cfg,
                {
                    **metadata_common,
                    "epoch": int(epoch),
                    "val_score": float(val_score),
                    "checkpoint_role": "best_val",
                    "checkpoint_load_status": load_status,
                    "new_parameter_init_status": init_status,
                    "subset_info": subset_info,
                    "stage": stage_info["stage"],
                },
            )

    final_checkpoint_path = checkpoint_dir / "final.pt"
    _save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        int(train_cfg["epochs"]),
        ckpt_cfg,
        {
            **metadata_common,
            "epoch": int(train_cfg["epochs"]),
            "val_score": float(best_val_score),
            "checkpoint_role": "final",
            "checkpoint_load_status": load_status,
            "new_parameter_init_status": init_status,
            "subset_info": subset_info,
        },
    )

    selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
    selected_payload = torch.load(str(selected_checkpoint_path), map_location=device)
    selected_model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    selected_model.load_state_dict(selected_payload["model"], strict=False)
    selected_model.eval()

    val_eval = evaluate_struct360b_pose(selected_model, val_loader, device, tmag_epsilon=float(data_cfg["tmag_epsilon"]), max_batches=eval_cfg.get("max_val_batches"))
    test_eval = evaluate_struct360b_pose(selected_model, test_loader, device, tmag_epsilon=float(data_cfg["tmag_epsilon"]), max_batches=eval_cfg.get("max_test_batches")) if run_test_during_train else {"final_metrics": {"skipped": True}, "coarse_metrics": {"skipped": True}, "residual_stats": {}, "gate_stats": {}}
    val_metrics = val_eval["final_metrics"]
    test_metrics = test_eval["final_metrics"]
    coarse_val_metrics = val_eval["coarse_metrics"]
    coarse_test_metrics = test_eval["coarse_metrics"]
    val_score = _compute_val_score(val_metrics, score_cfg=eval_cfg.get("selection_score"), eps=float(data_cfg["tmag_epsilon"]))
    discrepancy = abs(float(val_metrics["signed_tdir_mean_deg"]) - float(test_metrics["signed_tdir_mean_deg"]))

    val_payload = {
        "task_name": str(cfg.get("task_name")),
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "val_score": float(val_score),
        "metrics": val_metrics,
        "coarse_metrics": coarse_val_metrics,
        "residual_stats": val_eval["residual_stats"],
        "gate_stats": val_eval["gate_stats"],
        "alpha": val_eval["alpha"],
        "beta": val_eval["beta"],
    }
    test_payload = {
        "task_name": str(cfg.get("task_name")),
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "val_score_of_selected_checkpoint": float(val_score),
        "metrics": test_metrics,
        "coarse_metrics": coarse_test_metrics,
        "residual_stats": test_eval["residual_stats"],
        "gate_stats": test_eval["gate_stats"],
        "alpha": test_eval["alpha"],
        "beta": test_eval["beta"],
        "val_test_discrepancy": float(discrepancy),
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], val_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], test_payload)

    train360c_val = precheck["train360c_val_metrics"]
    train360c_test = precheck["train360c_test_metrics"]
    train360d_val = precheck["train360d_val_metrics"]
    train360d_test = precheck["train360d_test_metrics"]
    train360h_val = precheck["train360h_val_metrics"]
    train360h_test = precheck["train360h_test_metrics"]
    struct360a_val = precheck["struct360a_val_metrics"]
    struct360a_test = precheck["struct360a_test_metrics"]
    base360d_test = precheck["base360d_test_metrics"]

    classification = _classify(test_metrics, train360c_test=train360c_test, train360d_test=train360d_test, train360h_test=train360h_test)
    versus_d = _coarse_status(test_metrics, train360d_test)
    versus_h = _coarse_status(test_metrics, train360h_test)
    versus_a = _coarse_status(test_metrics, struct360a_test)
    next_recommendation = _recommendation(classification)
    runtime_sec = time.time() - start_time

    report_lines = [
        "# STRUCT360B match-free coarse-to-fine pose refinement",
        "",
        "## 1. Executive summary",
        "- training executed true/false: `true`",
        "- checkpoint saved true/false: `true`",
        f"- best checkpoint: `{selected_checkpoint_path}`",
        f"- classification: `{classification}`",
        f"- whether STRUCT360B improves over TRAIN360D/H/STRUCT360A: `vs TRAIN360D={versus_d}, vs TRAIN360H={versus_h}, vs STRUCT360A={versus_a}`",
        "",
        "## 2. LoFTR-inspired but match-free design",
        "- borrowed coarse-to-fine hierarchy from LoFTR: `true`",
        "- did not use explicit matching: `true`",
        "- did not use confidence matrix / MNN / match list: `true`",
        "- did not use RANSAC / PnP / BA: `true`",
        "- fine stage is residual pose refinement, not correspondence refinement.",
        "",
        "## 3. Architecture",
        "- coarse stage: `TRAIN360D coarse interaction predicts R0 / tdir0 / log_tmag0`.",
        "- fine pose-conditioned branch: `latent cross-image fine-token refinement conditioned on coarse pose embedding`.",
        "- pose embedding: `R0(9) + tdir0(3) + log_tmag0(1) + coarse pair context stats`.",
        "- residual pose heads: `predict ΔR / Δtdir / Δlog_tmag with conservative gate`.",
        f"- pose composition: `R_final = ΔR ∘ R0, tdir_final = normalize(tdir0 + {test_eval['alpha']} * Δtdir), log_tmag_final = log_tmag0 + {test_eval['beta']} * Δlog_tmag`.",
        "- auxiliary coarse loss: `enabled`.",
        "- residual regularization: `enabled`.",
        f"- parameter count: `{param_counts}`",
        "",
        "## 4. Initialization",
        f"- init checkpoint: `{init_ckpt_path}`",
        f"- strict/non-strict load: `{load_status['status']}`",
        f"- missing keys: `{load_status['missing_keys']}`",
        f"- unexpected keys: `{load_status['unexpected_keys']}`",
        f"- new fine branch params: `{[k for k in init_status['newly_initialized_params'] if 'struct360b' in k]}`",
        "",
        "## 5. Training setup",
        f"- freeze/unfreeze strategy: `2-epoch fine-only warmup, then partial unfreeze through epoch {train_cfg['epochs']}`",
        f"- epochs: `{train_cfg['epochs']}`",
        f"- LR groups: `warmup coarse={train_cfg['warmup_coarse_lr']}, warmup fine={train_cfg['warmup_fine_lr']}, coarse={train_cfg['coarse_lr']}, fine={train_cfg['fine_lr']}`",
        f"- loss weights: `final=1.0, coarse_aux={loss_cfg['coarse_aux_weight']}, residual_reg={loss_cfg['residual_reg_weight']}`",
        f"- alpha/beta: `alpha={test_eval['alpha']}, beta={test_eval['beta']}`",
        f"- observability/k-step/scale settings: `{loss_cfg}`",
        f"- val score: `{eval_cfg['selection_score']}`",
        f"- train subset info: `{subset_info}`",
        f"- runtime sec: `{runtime_sec:.2f}`",
        f"- system info: `{system_info}`",
        "",
        "## 6. Validation results",
        f"- coarse metrics: `{coarse_val_metrics}`",
        f"- final metrics: `{val_metrics}`",
        f"- comparison to TRAIN360D val: `signed_tdir {train360d_val.get('signed_tdir_mean_deg')} -> {val_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360d_val.get('anti_parallel_rate')} -> {val_metrics.get('anti_parallel_rate')}, tmag_ratio {train360d_val.get('tmag_median_ratio')} -> {val_metrics.get('tmag_median_ratio')}, path_ratio {train360d_val.get('path_ratio')} -> {val_metrics.get('path_ratio')}`",
        f"- comparison to TRAIN360H val: `signed_tdir {train360h_val.get('signed_tdir_mean_deg')} -> {val_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360h_val.get('anti_parallel_rate')} -> {val_metrics.get('anti_parallel_rate')}, tmag_ratio {train360h_val.get('tmag_median_ratio')} -> {val_metrics.get('tmag_median_ratio')}, path_ratio {train360h_val.get('path_ratio')} -> {val_metrics.get('path_ratio')}`",
        f"- comparison to STRUCT360A val: `signed_tdir {struct360a_val.get('signed_tdir_mean_deg')} -> {val_metrics.get('signed_tdir_mean_deg')}, anti_parallel {struct360a_val.get('anti_parallel_rate')} -> {val_metrics.get('anti_parallel_rate')}, tmag_ratio {struct360a_val.get('tmag_median_ratio')} -> {val_metrics.get('tmag_median_ratio')}, path_ratio {struct360a_val.get('path_ratio')} -> {val_metrics.get('path_ratio')}`",
        "",
        "## 7. Test results",
        f"- coarse metrics: `{coarse_test_metrics}`",
        f"- final metrics: `{test_metrics}`",
        "| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| STRUCT360B | val | {val_metrics.get('signed_tdir_mean_deg')} | {val_metrics.get('anti_parallel_rate')} | {val_metrics.get('tmag_median_ratio')} | {val_metrics.get('path_ratio')} | {val_metrics.get('coverage')} |",
        f"| STRUCT360B | test | {test_metrics.get('signed_tdir_mean_deg')} | {test_metrics.get('anti_parallel_rate')} | {test_metrics.get('tmag_median_ratio')} | {test_metrics.get('path_ratio')} | {test_metrics.get('coverage')} |",
        f"| TRAIN360C | test | {train360c_test.get('signed_tdir_mean_deg')} | {train360c_test.get('anti_parallel_rate')} | {train360c_test.get('tmag_median_ratio')} | {train360c_test.get('path_ratio')} | {train360c_test.get('coverage')} |",
        f"| TRAIN360D | test | {train360d_test.get('signed_tdir_mean_deg')} | {train360d_test.get('anti_parallel_rate')} | {train360d_test.get('tmag_median_ratio')} | {train360d_test.get('path_ratio')} | {train360d_test.get('coverage')} |",
        f"| TRAIN360H | test | {train360h_test.get('signed_tdir_mean_deg')} | {train360h_test.get('anti_parallel_rate')} | {train360h_test.get('tmag_median_ratio')} | {train360h_test.get('path_ratio')} | {train360h_test.get('coverage')} |",
        f"| STRUCT360A | test | {struct360a_test.get('signed_tdir_mean_deg')} | {struct360a_test.get('anti_parallel_rate')} | {struct360a_test.get('tmag_median_ratio')} | {struct360a_test.get('path_ratio')} | {struct360a_test.get('coverage')} |",
        f"| T57b | reference | {T57B_REFERENCE['signed_tdir_mean_deg']} | {T57B_REFERENCE['anti_parallel_rate']} | {T57B_REFERENCE['tmag_median_ratio']} | {T57B_REFERENCE['path_ratio']} | 1.0 |",
        f"| BASE360D | test component | {base360d_test.get('signed_tdir_mean_deg')} | {base360d_test.get('anti_parallel_rate')} | {base360d_test.get('tmag_median_ratio')} | {base360d_test.get('pair_component_path_ratio')} | {base360d_test.get('coverage')} |",
        "",
        "## 8. Residual behavior analysis",
        f"- residual magnitude: `{test_eval['residual_stats']}`",
        f"- fine gate stats: `{test_eval['gate_stats']}`",
        f"- whether fine branch is too aggressive: `judge from residual magnitudes plus gate stats; current gate mean={test_eval['gate_stats'].get('mean')}`",
        f"- whether coarse branch preserved direction: `coarse signed_tdir={coarse_test_metrics.get('signed_tdir_mean_deg')}, final signed_tdir={test_metrics.get('signed_tdir_mean_deg')}`",
        f"- whether scale/path improved: `coarse tmag_ratio={coarse_test_metrics.get('tmag_median_ratio')}, final tmag_ratio={test_metrics.get('tmag_median_ratio')}, coarse path_ratio={coarse_test_metrics.get('path_ratio')}, final path_ratio={test_metrics.get('path_ratio')}`",
        "",
        "## 9. Failure analysis",
        "- if anti_parallel worsens: `reduce fine residual gate or add rotation-aware bias in STRUCT360C without converting affinity into explicit matches.`",
        "- if scale worsens: `revisit beta / residual log-tmag scale and consider TRAIN360H-style scale weighting for a follow-up retrain.`",
        "- if fine branch collapses: `inspect residual gate saturation and whether warmup froze coarse long enough.`",
        f"- if val/test discrepancy remains: `current discrepancy = {discrepancy}; prefer STRUCT360C rotation-aware attention bias before any larger sweep.`",
        "",
        "## 10. Next recommendation",
        f"- `{next_recommendation}`",
        "",
        "## 11. Compliance checklist",
        "- `real_training_executed = true`",
        "- `learned_weights_saved = true`",
        "- `train360c_checkpoint_modified = false`",
        "- `train360d_checkpoint_modified = false`",
        "- `train360h_checkpoint_modified = false`",
        "- `struct360a_checkpoint_modified = false`",
        "- `explicit_matching_used = false`",
        "- `confidence_matrix_used_for_matching = false`",
        "- `mnn_match_selection_used = false`",
        "- `match_list_output = false`",
        "- `ransac_used = false`",
        "- `pnp_used = false`",
        "- `bundle_adjustment_used = false`",
        "- `hkust_360dvo_teacher_used = false`",
        "- `base360_outputs_used_as_training_input = false`",
        "- `train_manifest_used = true`",
        "- `val_manifest_used_for_selection_only = true`",
        "- `test_manifest_used_for_final_eval_only = true`",
        "- `direct_glob_data_360dvo_sequences = false`",
        "- `random_pair_split_used = false`",
        "- `s5_locked_metrics_modified = false`",
        "- `large_checkpoints_committed_to_git = false`",
    ]
    (REPO_ROOT / outputs["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# STRUCT360B vs TRAIN360C / D / H / STRUCT360A / BASE360D",
        "",
        "| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| STRUCT360B | val | {val_metrics.get('signed_tdir_mean_deg')} | {val_metrics.get('anti_parallel_rate')} | {val_metrics.get('tmag_median_ratio')} | {val_metrics.get('path_ratio')} | {val_metrics.get('coverage')} |",
        f"| STRUCT360B | test | {test_metrics.get('signed_tdir_mean_deg')} | {test_metrics.get('anti_parallel_rate')} | {test_metrics.get('tmag_median_ratio')} | {test_metrics.get('path_ratio')} | {test_metrics.get('coverage')} |",
        f"| TRAIN360C | test | {train360c_test.get('signed_tdir_mean_deg')} | {train360c_test.get('anti_parallel_rate')} | {train360c_test.get('tmag_median_ratio')} | {train360c_test.get('path_ratio')} | {train360c_test.get('coverage')} |",
        f"| TRAIN360D | test | {train360d_test.get('signed_tdir_mean_deg')} | {train360d_test.get('anti_parallel_rate')} | {train360d_test.get('tmag_median_ratio')} | {train360d_test.get('path_ratio')} | {train360d_test.get('coverage')} |",
        f"| TRAIN360H | test | {train360h_test.get('signed_tdir_mean_deg')} | {train360h_test.get('anti_parallel_rate')} | {train360h_test.get('tmag_median_ratio')} | {train360h_test.get('path_ratio')} | {train360h_test.get('coverage')} |",
        f"| STRUCT360A | test | {struct360a_test.get('signed_tdir_mean_deg')} | {struct360a_test.get('anti_parallel_rate')} | {struct360a_test.get('tmag_median_ratio')} | {struct360a_test.get('path_ratio')} | {struct360a_test.get('coverage')} |",
        f"| T57b | reference | {T57B_REFERENCE['signed_tdir_mean_deg']} | {T57B_REFERENCE['anti_parallel_rate']} | {T57B_REFERENCE['tmag_median_ratio']} | {T57B_REFERENCE['path_ratio']} | 1.0 |",
        f"| BASE360D | test component | {base360d_test.get('signed_tdir_mean_deg')} | {base360d_test.get('anti_parallel_rate')} | {base360d_test.get('tmag_median_ratio')} | {base360d_test.get('pair_component_path_ratio')} | {base360d_test.get('coverage')} |",
        "",
        f"- classification: `{classification}`",
        f"- compared to TRAIN360D: `{versus_d}`",
        f"- compared to TRAIN360H: `{versus_h}`",
        f"- compared to STRUCT360A: `{versus_a}`",
        f"- val/test discrepancy: `{discrepancy}`",
        f"- coarse test metrics: `{coarse_test_metrics}`",
        f"- residual stats: `{test_eval['residual_stats']}`",
        f"- gate stats: `{test_eval['gate_stats']}`",
    ]
    (REPO_ROOT / outputs["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("- STRUCT360B training executed: true")
    print("- checkpoint saved: true")
    print(f"- best checkpoint: {selected_checkpoint_path}")
    print(f"- init source: {model_cfg['init_source']}")
    print("- explicit matching used: false")
    print("- coarse-to-fine type: pose residual refinement")
    print(f"- test signed_tdir_mean: {test_metrics.get('signed_tdir_mean_deg')}")
    print(f"- test anti_parallel_rate: {test_metrics.get('anti_parallel_rate')}")
    print(f"- test tmag_median_ratio: {test_metrics.get('tmag_median_ratio')}")
    print(f"- test path_ratio: {test_metrics.get('path_ratio')}")
    print(f"- val/test discrepancy: {discrepancy}")
    print(f"- compared to TRAIN360D: {versus_d}")
    print(f"- compared to TRAIN360H: {versus_h}")
    print(f"- compared to STRUCT360A: {versus_a}")
    print(f"- classification: {classification}")
    print("- committed to git: false")
    print(f"- next recommended task: {next_recommendation}")


if __name__ == "__main__":
    train()
