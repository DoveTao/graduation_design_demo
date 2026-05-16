#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from train360.core.config import Config
from train360.core.losses import translation_direction_loss, translation_magnitude_loss
from train360.core.pose_head import matrix_geodesic_distance
from train360.core.train360d_pose_losses import build_k_step_weights, rotation_geodesic_loss, scale_stability_loss
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
import data360a_tdir_regime_diagnostic as data360a


DEFAULT_CONFIG = REPO_ROOT / "configs" / "final360k_conservative_observability_weighting.yaml"
TMAG_EPS = 1.0e-6
FINAL360J_BRANCH = "origin/research/final360j-tdir-loss-reweight"
FINAL360J_FALLBACK_METRICS = {
    "signed_tdir_mean_deg": 46.236052001935235,
    "anti_parallel_rate": 0.2056499407348874,
    "tmag_median_ratio": 0.7251413215270102,
    "path_ratio": 0.5542818158986929,
}


def _run(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), cwd=REPO_ROOT, text=True).strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    rot_deg = matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
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


def _load_model(checkpoint_path: Path, device: torch.device, train_cfg: Mapping[str, Any]) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Config, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = _inject_struct360b_cfg(dict(payload.get("cfg", {})), train_cfg)
    cfg = _cfg_from_dict(cfg_dict)
    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    return model, cfg, {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "epoch": int(payload.get("epoch", -1)),
    }


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, ckpt_cfg: Config, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "cfg": dict(ckpt_cfg.__dict__),
            "metadata": dict(metadata),
        },
        str(path),
    )


def _dataset(split: str, cfg: Mapping[str, Any]):
    data_cfg = cfg["data"]
    ds = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        image_hw=tuple(int(x) for x in data_cfg["image_hw"]),
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
    )
    subset_max = data_cfg.get("train_subset_max") if split == "train" else None
    if subset_max is not None and int(subset_max) > 0 and len(ds) > int(subset_max):
        rng = np.random.default_rng(int(cfg["training"]["seed"]))
        idx = sorted(int(x) for x in rng.permutation(len(ds))[: int(subset_max)].tolist())
        return Subset(ds, idx), {"subset_used": True, "subset_count": len(idx), "original_count": len(ds), "subset_indices": idx}
    return ds, {"subset_used": False, "subset_count": len(ds), "original_count": len(ds)}


def _loader(split: str, cfg: Mapping[str, Any]):
    ds, subset_info = _dataset(split, cfg)
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["train_batch_size"] if split == "train" else data_cfg["eval_batch_size"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(data_cfg["shuffle_train"]) if split == "train" else False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, subset_info


def _rotation_angle_from_gt(R_gt: torch.Tensor) -> torch.Tensor:
    trace = ((R_gt[:, 0, 0] + R_gt[:, 1, 1] + R_gt[:, 2, 2]) - 1.0) * 0.5
    trace = trace.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(trace))


def _build_weight_thresholds(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    ds = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"]["train_manifest"]),
        expected_split="train",
        image_hw=tuple(int(x) for x in cfg["data"]["image_hw"]),
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
    )
    values = np.asarray([float(s["tmag"]) for s in ds.samples], dtype=np.float64)
    quantiles = [float(q) for q in cfg["loss"]["conservative_weighting"]["tmag_quantiles"]]
    cuts = [float(np.percentile(values, q)) for q in quantiles]
    thresholds = {
        "source": str(cfg["loss"]["conservative_weighting"]["threshold_source"]),
        "tmag_quantiles": quantiles,
        "tmag_cut_values": cuts,
        "rot_buckets_deg": dict(cfg["loss"]["conservative_weighting"]["rot_buckets_deg"]),
        "tmag_weights": dict(cfg["loss"]["conservative_weighting"]["tmag_weights"]),
        "rot_weights": dict(cfg["loss"]["conservative_weighting"]["rot_weights"]),
        "clamp_min": float(cfg["loss"]["conservative_weighting"]["clamp_min"]),
        "clamp_max": float(cfg["loss"]["conservative_weighting"]["clamp_max"]),
        "train_pair_count": len(ds.samples),
    }
    counts = {"very_small": 0, "small": 0, "medium": 0, "large": 0, "very_large": 0}
    for v in values:
        counts[_tmag_bucket_name(float(v), thresholds)] += 1
    thresholds["train_bucket_counts"] = counts
    return thresholds


def _tmag_bucket_name(tmag: float, thresholds: Mapping[str, Any]) -> str:
    q10, q25, q50, q75 = [float(x) for x in thresholds["tmag_cut_values"]]
    if tmag <= q10:
        return "very_small"
    if tmag <= q25:
        return "small"
    if tmag <= q50:
        return "medium"
    if tmag <= q75:
        return "large"
    return "very_large"


def _rot_bucket_name(rot_deg: float, thresholds: Mapping[str, Any]) -> str:
    rot_cfg = thresholds["rot_buckets_deg"]
    if rot_deg < float(rot_cfg["small_max"]):
        return "small_rotation"
    if rot_deg < float(rot_cfg["medium_max"]):
        return "medium_rotation"
    if rot_deg < float(rot_cfg["large_max"]):
        return "large_rotation"
    return "extreme_rotation"


def _conservative_tdir_sample_weight(tmag_gt: torch.Tensor, R_gt: torch.Tensor, k_tensor: torch.Tensor, *, thresholds: Mapping[str, Any], k_step_cfg: Mapping[str, Any], enable_k: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tmag_np = tmag_gt.detach().float().view(-1).cpu().numpy()
    rot_np = _rotation_angle_from_gt(R_gt.detach().float()).cpu().numpy()
    tmag_weights_cfg = thresholds["tmag_weights"]
    rot_weights_cfg = thresholds["rot_weights"]
    weights = []
    tmag_bucket_counter = Counter()
    rot_bucket_counter = Counter()
    for tmag, rot in zip(tmag_np.tolist(), rot_np.tolist()):
        t_bucket = _tmag_bucket_name(float(tmag), thresholds)
        r_bucket = _rot_bucket_name(float(rot), thresholds)
        tmag_bucket_counter[t_bucket] += 1
        rot_bucket_counter[r_bucket] += 1
        w = float(tmag_weights_cfg[t_bucket]) * float(rot_weights_cfg[r_bucket])
        w = max(float(thresholds["clamp_min"]), min(float(thresholds["clamp_max"]), w))
        weights.append(w)
    weight_tensor = torch.tensor(weights, device=tmag_gt.device, dtype=torch.float32)
    if enable_k:
        k_weights = build_k_step_weights(k_tensor, **k_step_cfg).to(weight_tensor.device)
        weight_tensor = weight_tensor * k_weights
    weight_tensor = weight_tensor.clamp(min=float(thresholds["clamp_min"]), max=float(thresholds["clamp_max"]))
    stats = {
        "min": float(weight_tensor.min().detach().cpu()),
        "p10": float(torch.quantile(weight_tensor.detach().cpu(), 0.10)),
        "median": float(torch.quantile(weight_tensor.detach().cpu(), 0.50)),
        "p90": float(torch.quantile(weight_tensor.detach().cpu(), 0.90)),
        "max": float(weight_tensor.max().detach().cpu()),
        "mean": float(weight_tensor.mean().detach().cpu()),
        "tmag_bucket_counts_batch": dict(tmag_bucket_counter),
        "rot_bucket_counts_batch": dict(rot_bucket_counter),
    }
    return weight_tensor, stats


def _anti_parallel_penalty(tdir_pred_B: torch.Tensor, t_gt_vec_B: torch.Tensor, *, margin: float) -> torch.Tensor:
    pred = torch.nn.functional.normalize(tdir_pred_B.float(), dim=-1, eps=1.0e-6)
    gt = torch.nn.functional.normalize(t_gt_vec_B.float(), dim=-1, eps=1.0e-6)
    cos = (pred * gt).sum(dim=-1).clamp(-1.0, 1.0)
    penalty = torch.relu(torch.full_like(cos, float(margin)) - cos).pow(2)
    return penalty.mean()


def _pose_loss_final360k(
    *,
    R_pred: torch.Tensor,
    tdir_pred_B: torch.Tensor,
    tmag_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt_vec_B: torch.Tensor,
    tmag_gt: torch.Tensor,
    k_tensor: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    tmag_epsilon: float,
) -> Dict[str, torch.Tensor]:
    enable_k = bool(loss_cfg["components"].get("enable_k_step_balancing", True))
    pose_weight = build_k_step_weights(k_tensor, **loss_cfg["k_step_balancing"]).to(R_pred.device) if enable_k else torch.ones_like(k_tensor.float().view(-1), device=R_pred.device)
    tdir_weight, tdir_weight_stats = _conservative_tdir_sample_weight(
        tmag_gt, R_gt, k_tensor,
        thresholds=thresholds,
        k_step_cfg=loss_cfg["k_step_balancing"],
        enable_k=enable_k,
    )
    loss_rot = rotation_geodesic_loss(R_pred, R_gt, sample_weight=pose_weight)
    loss_tdir = translation_direction_loss(
        tdir_pred_B,
        t_gt_vec_B,
        R_gt,
        pred_t_frame="B",
        oriented_weight=1.0,
        axis_weight=0.0,
        sample_weight=tdir_weight,
    )
    loss_tmag = translation_magnitude_loss(
        tmag_pred,
        tmag_gt,
        loss_type=str(loss_cfg["tmag_loss_type"]),
        eps=tmag_epsilon,
        sample_weight=pose_weight,
    )
    loss_scale = scale_stability_loss(
        tmag_pred,
        tmag_gt,
        sample_weight=pose_weight,
        eps=tmag_epsilon,
        collapse_ratio=float(loss_cfg["scale_stability"]["collapse_ratio"]),
        explosion_ratio=float(loss_cfg["scale_stability"]["explosion_ratio"]),
        mean_log_bias_weight=float(loss_cfg["scale_stability"]["mean_log_bias_weight"]),
    ) if bool(loss_cfg["components"].get("enable_scale_stabilization", True)) else torch.zeros([], device=R_pred.device)
    anti_pen = _anti_parallel_penalty(
        tdir_pred_B,
        t_gt_vec_B,
        margin=float(loss_cfg["anti_parallel"]["margin"]),
    ) if float(loss_cfg["anti_parallel"]["weight"]) > 0.0 else torch.zeros([], device=R_pred.device)
    total = (
        float(loss_cfg["rot_weight"]) * loss_rot
        + float(loss_cfg["tdir_weight"]) * loss_tdir
        + float(loss_cfg["tmag_weight"]) * loss_tmag
        + float(loss_cfg["scale_stability_weight"]) * loss_scale
        + float(loss_cfg["anti_parallel"]["weight"]) * anti_pen
    )
    return {
        "loss_total": total,
        "loss_rot": loss_rot,
        "loss_tdir": loss_tdir,
        "loss_tmag": loss_tmag,
        "loss_scale_stability": loss_scale,
        "loss_antiparallel": anti_pen,
        "pose_weights": pose_weight.detach(),
        "tdir_weights": tdir_weight.detach(),
        "tdir_weight_stats": tdir_weight_stats,
    }


@torch.no_grad()
def _evaluate_pose(
    model: STRUCT360BMatchFreeCoarseToFineModel,
    loader: DataLoader,
    device: torch.device,
    *,
    tmag_epsilon: float,
    max_batches: Optional[int] = None,
    progress_label: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    collector = _empty_pose_collector()
    residual_rot_deg: List[float] = []
    residual_tdir_norm: List[float] = []
    residual_log_tmag_abs: List[float] = []
    residual_gate: List[float] = []
    num_pairs = 0
    nan_count = 0
    inf_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if progress_label and batch_idx % 50 == 0:
            print(f"[FINAL360K] {progress_label}: batch {batch_idx}/{len(loader)}", flush=True)
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
        added, nan_i, inf_i = _append_pose_batch(
            collector,
            R_pred=R_pred,
            tdir_pred_B=aux["t_dir_out"],
            tmag_pred=aux["t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt_vec,
            tmag_epsilon=tmag_epsilon,
        )
        num_pairs += added
        nan_count += nan_i
        inf_count += inf_i
        residual_rot_deg.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
        residual_tdir_norm.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
        residual_log_tmag_abs.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
        residual_gate.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
    return {
        "metrics": _summarize_pose_metrics(collector, num_pairs=num_pairs, dataset_size=len(loader.dataset), nan_count=nan_count, inf_count=inf_count),
        "diagnostics": {
            "delta_rot_mean_deg": _float_stats(residual_rot_deg)["mean"],
            "delta_tdir_norm_mean": _float_stats(residual_tdir_norm)["mean"],
            "delta_log_tmag_abs_mean": _float_stats(residual_log_tmag_abs)["mean"],
            "residual_gate_mean": _float_stats(residual_gate)["mean"],
            "residual_gate_median": _float_stats(residual_gate)["median"],
            "residual_gate_max": _float_stats(residual_gate)["max"],
        },
    }


@torch.no_grad()
def _rows_for_split(model: STRUCT360BMatchFreeCoarseToFineModel, *, split: str, cfg: Mapping[str, Any], device: torch.device, model_name: str) -> List[Dict[str, Any]]:
    data_cfg = cfg["data"]
    ds = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        image_hw=tuple(int(x) for x in data_cfg["image_hw"]),
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
    )
    loader = DataLoader(ds, batch_size=int(data_cfg["eval_batch_size"]), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    rows: List[Dict[str, Any]] = []
    print(f"[FINAL360K] bucket rows {model_name} {split}: {len(loader)} batches", flush=True)
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"[FINAL360K] bucket rows {model_name} {split}: batch {batch_idx}/{len(loader)}", flush=True)
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
        rot_err_deg = matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
        R_gt_np = R_gt.detach().cpu().numpy()
        t_gt_np = t_gt_vec.detach().cpu().numpy()
        tdir_pred_np = aux["t_dir_out"].detach().cpu().numpy()
        tmag_pred_np = aux["t_mag"].detach().view(-1).cpu().numpy()
        meta = batch["meta"]
        gt_tmag_np = torch.linalg.norm(t_gt_vec.float(), dim=-1).detach().cpu().numpy()
        for i in range(IA.shape[0]):
            pred_tdir = np.asarray(tdir_pred_np[i], dtype=np.float64)
            gt_tvec = np.asarray(t_gt_np[i], dtype=np.float64)
            gt_tmag = max(float(gt_tmag_np[i]), TMAG_EPS)
            pred_tmag = max(float(tmag_pred_np[i]), TMAG_EPS)
            gt_tdir = gt_tvec / max(float(np.linalg.norm(gt_tvec)), TMAG_EPS)
            cos_signed = float(np.dot(pred_tdir, gt_tdir))
            image_a = str(meta["image_path_a"][i])
            image_b = str(meta["image_path_b"][i])
            sequence = str(meta["sequence_id"][i])
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "scene": data360a._scene_from_sequence(sequence),
                    "sequence": sequence,
                    "sample_id": f"{split}:{sequence}:{int(meta['pair_index'][i])}",
                    "pair_id": int(meta["pair_index"][i]),
                    "frame_i": data360a._frame_id_from_path(image_a),
                    "frame_j": data360a._frame_id_from_path(image_b),
                    "k": int(meta["k"][i]),
                    "pair_type": str(meta["pair_type"][i]),
                    "timestamp_a": float(meta["timestamp_a"][i]),
                    "timestamp_b": float(meta["timestamp_b"][i]),
                    "frame_distance": abs(data360a._frame_id_from_path(image_b) - data360a._frame_id_from_path(image_a)),
                    "gt_tmag": gt_tmag,
                    "gt_rot_angle_deg": data360a._rot_angle_deg(np.asarray(R_gt_np[i], dtype=np.float64)),
                    "pred_rot_angle_error_deg": float(rot_err_deg[i]),
                    "signed_tdir_error_deg": data360a._angle_deg_from_vectors(pred_tdir, gt_tdir, absolute=False),
                    "unsigned_tdir_error_deg": data360a._angle_deg_from_vectors(pred_tdir, gt_tdir, absolute=True),
                    "cos_tdir_signed": cos_signed,
                    "anti_parallel": bool(cos_signed < 0.0),
                    "pred_tmag": pred_tmag,
                    "tmag_ratio": float(pred_tmag / gt_tmag),
                    "log_tmag_error": float(abs(math.log(pred_tmag) - math.log(gt_tmag))),
                    "finite": bool(np.isfinite(pred_tdir).all() and math.isfinite(pred_tmag)),
                    "scale_collapse": bool((pred_tmag / gt_tmag) < 0.1),
                    "scale_explosion": bool((pred_tmag / gt_tmag) > 10.0),
                    "pred_path_contribution": pred_tmag,
                    "gt_path_contribution": gt_tmag,
                }
            )
    return rows


def _selection_score(metrics: Mapping[str, Any], score_cfg: Mapping[str, Any]) -> float:
    eps = 1.0e-6
    signed = float(metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(metrics.get("anti_parallel_rate") or 1.0)
    tmag_ratio = float(metrics.get("tmag_median_ratio") or eps)
    path_ratio = float(metrics.get("path_ratio") or eps)
    rot = float(metrics.get("rot_mean_deg") or 0.0)
    return (
        float(score_cfg.get("signed_tdir_weight", 1.0)) * signed
        + float(score_cfg.get("anti_parallel_weight", 80.0)) * anti
        + float(score_cfg.get("tmag_ratio_weight", 15.0)) * abs(math.log(max(tmag_ratio, eps)))
        + float(score_cfg.get("path_ratio_weight", 8.0)) * abs(math.log(max(path_ratio, eps)))
        + float(score_cfg.get("rot_weight", 0.2)) * rot
    )


def _discrepancy(val_metrics: Mapping[str, Any], test_metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("rot_mean_deg", "signed_tdir_mean_deg", "anti_parallel_rate", "tmag_median_ratio", "path_ratio"):
        a = val_metrics.get(key)
        b = test_metrics.get(key)
        out[key] = None if a is None or b is None else float(b) - float(a)
    return out


def _read_git_json(rev_path: str) -> Dict[str, Any]:
    try:
        text = subprocess.check_output(["git", "show", rev_path], cwd=REPO_ROOT, text=True)
    except subprocess.CalledProcessError:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _final360j_reference(cfg: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    local = _read_json(REPO_ROOT / cfg["inputs"]["final360j_test_metrics"]).get("metrics")
    if isinstance(local, Mapping):
        return dict(local), "local_report"
    remote = _read_git_json(f"{FINAL360J_BRANCH}:reports/FINAL360J_metrics_test.json").get("metrics")
    if isinstance(remote, Mapping):
        return dict(remote), "git_show_remote_branch"
    return dict(FINAL360J_FALLBACK_METRICS), "task_brief_fallback"


def _compare_to_baseline(test_metrics: Mapping[str, Any], ref_metrics: Mapping[str, Any]) -> str:
    improved = 0
    worsened = 0
    fields = [
        ("signed_tdir_mean_deg", "lower"),
        ("anti_parallel_rate", "lower"),
        ("tmag_median_ratio", "higher"),
        ("path_ratio", "higher"),
    ]
    for field, mode in fields:
        a = float(test_metrics.get(field) or (999.0 if mode == "lower" else 0.0))
        b = float(ref_metrics.get(field) or (999.0 if mode == "lower" else 0.0))
        if (mode == "lower" and a < b) or (mode == "higher" and a > b):
            improved += 1
        else:
            worsened += 1
    if improved >= 3:
        return "better"
    if worsened >= 3:
        return "worse"
    return "comparable"


def _classify(test_metrics: Mapping[str, Any]) -> str:
    signed = float(test_metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(test_metrics.get("anti_parallel_rate") or 1.0)
    tmag = float(test_metrics.get("tmag_median_ratio") or 0.0)
    path = float(test_metrics.get("path_ratio") or 0.0)
    rot = float(test_metrics.get("rot_mean_deg") or float("inf"))
    if signed <= 44.5 and anti <= 0.195 and tmag >= 0.82 and path >= 0.63 and rot <= 2.6:
        return "strong_success"
    if signed <= 44.8 and anti <= 0.200 and tmag >= 0.80 and path >= 0.62 and rot <= 2.6:
        return "balanced_success"
    if (signed < 45.264702006380205 or anti < 0.20169893322797314) and tmag >= 0.78 and path >= 0.60:
        return "primary_success"
    if signed > 45.264702006380205 or anti > 0.20169893322797314 or tmag < 0.78 or path < 0.60 or rot > 3.0:
        return "regression"
    return "inconclusive"


def _bucket_diagnostic(final360k_rows: Sequence[Mapping[str, Any]], final360i_rows: Sequence[Mapping[str, Any]], final360j_bucket_diag: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    tmag_specs = data360a._tmag_named_specs([float(r["gt_tmag"]) for r in final360i_rows])
    rot_specs = data360a._rot_named_specs()
    final360k_summary = {
        "overall": data360a._summarize_rows(final360k_rows),
        "tmag_named_buckets": data360a._bucketize(final360k_rows, "gt_tmag", tmag_specs),
        "rotation_named_buckets": data360a._bucketize(final360k_rows, "gt_rot_angle_deg", rot_specs),
    }
    final360i_summary = {
        "overall": data360a._summarize_rows(final360i_rows),
        "tmag_named_buckets": data360a._bucketize(final360i_rows, "gt_tmag", tmag_specs),
        "rotation_named_buckets": data360a._bucketize(final360i_rows, "gt_rot_angle_deg", rot_specs),
    }
    final360j_summary = final360j_bucket_diag.get("final360j_test", {})

    def _find(rows: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
        return next((row for row in rows if str(row["bucket"]) == label), {})

    i_small = _find(final360i_summary["tmag_named_buckets"], "small")
    k_small = _find(final360k_summary["tmag_named_buckets"], "small")
    i_large_rot = _find(final360i_summary["rotation_named_buckets"], "large_rotation")
    k_large_rot = _find(final360k_summary["rotation_named_buckets"], "large_rotation")
    i_very_large = _find(final360i_summary["tmag_named_buckets"], "very_large")
    k_very_large = _find(final360k_summary["tmag_named_buckets"], "very_large")
    i_small_rot = _find(final360i_summary["rotation_named_buckets"], "small_rotation")
    k_small_rot = _find(final360k_summary["rotation_named_buckets"], "small_rotation")

    j_small = _find(final360j_summary.get("tmag_named_buckets", []), "small") if final360j_summary else {}
    j_large_rot = _find(final360j_summary.get("rotation_named_buckets", []), "large_rotation") if final360j_summary else {}
    j_very_large = _find(final360j_summary.get("tmag_named_buckets", []), "very_large") if final360j_summary else {}
    j_small_rot = _find(final360j_summary.get("rotation_named_buckets", []), "small_rotation") if final360j_summary else {}

    anti_comp = [
        (k_very_large.get("anti_parallel_rate") or 1.0) < (i_very_large.get("anti_parallel_rate") or 1.0),
        (k_small_rot.get("anti_parallel_rate") or 1.0) < (i_small_rot.get("anti_parallel_rate") or 1.0),
    ]
    anti_flag: Any = True if all(anti_comp) else ("partial" if any(anti_comp) else False)
    large_rot_flag: Any = (
        "partial" if (i_large_rot.get("count") or 0) == 0 or (k_large_rot.get("count") or 0) == 0
        else bool((k_large_rot.get("signed_tdir_mean_deg") or 999.0) < (i_large_rot.get("signed_tdir_mean_deg") or 999.0))
    )

    flags = {
        "small_tmag_improved": bool((k_small.get("signed_tdir_mean_deg") or 999.0) < (i_small.get("signed_tdir_mean_deg") or 999.0)),
        "large_rotation_improved": large_rot_flag,
        "anti_parallel_reduced_in_target_buckets": anti_flag,
        "scale_path_protected_vs_final360i": bool((final360k_summary["overall"].get("tmag_median_ratio") or 0.0) >= 0.78 and (final360k_summary["overall"].get("path_ratio") or 0.0) >= 0.60),
        "better_than_final360j_regression": bool((final360k_summary["overall"].get("signed_tdir_mean_deg") or 999.0) < FINAL360J_FALLBACK_METRICS["signed_tdir_mean_deg"] and (final360k_summary["overall"].get("path_ratio") or 0.0) > FINAL360J_FALLBACK_METRICS["path_ratio"]),
    }
    summary_lines = {
        "small_tmag": f"small tmag signed_tdir: FINAL360I `{i_small.get('signed_tdir_mean_deg')}` -> FINAL360K `{k_small.get('signed_tdir_mean_deg')}`" + (f" ; FINAL360J `{j_small.get('signed_tdir_mean_deg')}`" if j_small else ""),
        "large_rotation": f"large rotation signed_tdir: FINAL360I `{i_large_rot.get('signed_tdir_mean_deg')}` -> FINAL360K `{k_large_rot.get('signed_tdir_mean_deg')}`" + (f" ; FINAL360J `{j_large_rot.get('signed_tdir_mean_deg')}`" if j_large_rot else ""),
        "anti_parallel": f"very_large tmag anti_parallel `{i_very_large.get('anti_parallel_rate')}` -> `{k_very_large.get('anti_parallel_rate')}`, small_rotation anti_parallel `{i_small_rot.get('anti_parallel_rate')}` -> `{k_small_rot.get('anti_parallel_rate')}`" + (f" ; FINAL360J refs `{j_very_large.get('anti_parallel_rate')}` / `{j_small_rot.get('anti_parallel_rate')}`" if j_very_large or j_small_rot else ""),
    }
    payload = {
        "final360k_test": final360k_summary,
        "final360i_test": final360i_summary,
        "final360j_test": final360j_summary,
        "flags": flags,
        "focus_comparison": {
            "small_tmag": {"final360i": i_small, "final360j": j_small, "final360k": k_small},
            "large_rotation": {"final360i": i_large_rot, "final360j": j_large_rot, "final360k": k_large_rot},
            "very_large_tmag_anti_parallel": {"final360i": i_very_large, "final360j": j_very_large, "final360k": k_very_large},
            "small_rotation_anti_parallel": {"final360i": i_small_rot, "final360j": j_small_rot, "final360k": k_small_rot},
        },
    }
    return payload, summary_lines, flags


def _write_report(
    cfg: Mapping[str, Any],
    *,
    precheck: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    best_epoch: int,
    best_score: float,
    best_checkpoint: Path,
    val_eval: Mapping[str, Any],
    test_eval: Mapping[str, Any],
    compared_final360i: str,
    compared_final360j: str,
    classification: str,
    bucket_diag: Mapping[str, Any],
    bucket_summary: Mapping[str, Any],
    final360j_ref_source: str,
    final360j_metrics: Mapping[str, Any],
    weight_distribution: Mapping[str, Any],
    next_task: str,
) -> None:
    final360i_test = _read_json(REPO_ROOT / cfg["inputs"]["final360i_test_metrics"]).get("metrics", {})
    seq360b_test = _read_json(REPO_ROOT / cfg["inputs"]["seq360b_test_metrics"]).get("metrics", {})
    lines = [
        f"# {cfg['task_name']}",
        "",
        "## 1. Executive summary",
        "- training executed: `true`",
        f"- init checkpoint: `{REPO_ROOT / cfg['inputs']['init_checkpoint']}`",
        f"- selected checkpoint: `{best_checkpoint}`",
        f"- classification: `{classification}`",
        f"- final recommendation: `{next_task}`",
        "",
        "## 2. DATA360A motivation",
        "- DATA360A showed that FINAL360I still has a pair-level tdir bottleneck.",
        "- small tmag and large rotation are difficult regimes.",
        "- SEQ360B mainly improves scale/path rather than tdir or anti-parallel.",
        "",
        "## 3. FINAL360J negative lesson",
        "- FINAL360J used global tdir upweight plus anti-parallel penalty and regressed.",
        "- FINAL360K therefore avoids strong global tdir amplification.",
        "- FINAL360K only reweights per-sample tdir loss conservatively.",
        "",
        "## 4. Setup",
        f"- branch: `{precheck['branch']}`",
        f"- config: `configs/final360k_conservative_observability_weighting.yaml`",
        f"- checkpoint init: `{REPO_ROOT / cfg['inputs']['init_checkpoint']}`",
        f"- epochs: `{cfg['training']['epochs']}`",
        f"- seed: `{cfg['training']['seed']}`",
        "- architecture changed: `false`",
        "- translation head changed: `false`",
        "",
        "## 5. Weighting design",
        f"- threshold source: `{thresholds['source']}`",
        f"- tmag cut values: `{thresholds['tmag_cut_values']}`",
        f"- rot buckets deg: `{thresholds['rot_buckets_deg']}`",
        f"- tmag weight table: `{thresholds['tmag_weights']}`",
        f"- rot weight table: `{thresholds['rot_weights']}`",
        f"- clamp range: `[{thresholds['clamp_min']}, {thresholds['clamp_max']}]`",
        f"- train bucket counts: `{thresholds['train_bucket_counts']}`",
        f"- weight distribution: `{weight_distribution}`",
        "",
        "## 6. Model selection protocol",
        "- val used for selection, test not used for selection",
        f"- best epoch: `{best_epoch}`",
        f"- best val score: `{best_score}`",
        "",
        "## 7. Validation results",
        f"- val metrics: `{val_eval['metrics']}`",
        "",
        "## 8. Test results",
        f"- FINAL360I test metrics: `{final360i_test}`",
        f"- FINAL360J reference metrics: `{final360j_metrics}` from `{final360j_ref_source}`",
        f"- FINAL360K test metrics: `{test_eval['metrics']}`",
        f"- compared to FINAL360I: `{compared_final360i}`",
        f"- compared to FINAL360J: `{compared_final360j}`",
        "",
        "## 9. Bucket diagnostic",
        f"- {bucket_summary['small_tmag']}",
        f"- {bucket_summary['large_rotation']}",
        f"- {bucket_summary['anti_parallel']}",
        f"- bucket flags: `{bucket_diag['flags']}`",
        "",
        "## 10. Comparison to SEQ360B",
        f"- SEQ360B pair metrics: `{seq360b_test}`",
        "- FINAL360K should be interpreted as a pair-level tdir stabilization attempt rather than a scale/path correction variant.",
        "",
        "## 11. Caveats",
        f"- resource-limited single seed: `{cfg['training'].get('resource_limited_single_seed', True)}`",
        "- no sequence-level training in FINAL360K",
        "- no mature VO claim from this run",
        "",
        "## 12. Next recommendation",
        f"- `{next_task}`",
        "",
        "## 13. Compliance checklist",
        "- `training_executed = true`",
        "- `test_used_for_selection = false`",
        "- `architecture_changed = false`",
        "- `translation_head_changed = false`",
        "- `explicit_matching_used = false`",
        "- `ransac_used = false`",
        "- `pnp_used = false`",
        "- `ba_used = false`",
        "- `hkust_teacher_used = false`",
        "- `orbslam_teacher_used = false`",
        "- `metrics_modified = false`",
        "- `checkpoints_modified_original = false`",
        "- `large_checkpoints_committed = false`",
        "- `raw_data_committed = false`",
    ]
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_lines = [
        f"# {cfg['task_name']} summary",
        "",
        f"- classification: `{classification}`",
        f"- compared to FINAL360I: `{compared_final360i}`",
        f"- compared to FINAL360J: `{compared_final360j}`",
        f"- test signed_tdir_mean: `{test_eval['metrics'].get('signed_tdir_mean_deg')}`",
        f"- test anti_parallel_rate: `{test_eval['metrics'].get('anti_parallel_rate')}`",
        f"- test tmag_median_ratio: `{test_eval['metrics'].get('tmag_median_ratio')}`",
        f"- test path_ratio: `{test_eval['metrics'].get('path_ratio')}`",
        f"- next recommendation: `{next_task}`",
    ]
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    bucket_lines = [
        f"# {cfg['task_name']} bucket summary",
        "",
        f"- {bucket_summary['small_tmag']}",
        f"- {bucket_summary['large_rotation']}",
        f"- {bucket_summary['anti_parallel']}",
    ]
    (REPO_ROOT / cfg["outputs"]["bucket_summary_path"]).write_text("\n".join(bucket_lines) + "\n", encoding="utf-8")


def _precheck(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    branch = _run(["git", "branch", "--show-current"])
    tracked_status = _run(["git", "status", "--short", "--untracked-files=no"])
    if tracked_status.strip():
        blockers.append("tracked_or_staged_worktree_not_clean")
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient_disk_free_gb:{disk_free_gb:.2f}")
    if branch not in [str(x) for x in cfg["prechecks"]["expected_branches"]]:
        blockers.append(f"unexpected_branch:{branch}")
    if not torch.cuda.is_available():
        blockers.append("cuda_unavailable")
    required = [REPO_ROOT / cfg["inputs"]["init_checkpoint"]]
    required.extend(REPO_ROOT / cfg["inputs"][k] for k in ("train_manifest", "val_manifest", "test_manifest", "data360a_report", "data360a_regime_buckets", "data360a_antiparallel", "data360a_reweighting"))
    missing = [str(p) for p in required if not p.exists()]
    blockers.extend([f"missing_required:{p}" for p in missing])
    optional_missing = [str(REPO_ROOT / cfg["inputs"][k]) for k in ("final360j_report", "final360j_test_metrics", "final360j_bucket_summary") if not (REPO_ROOT / cfg["inputs"][k]).exists()]
    return {
        "blockers": blockers,
        "branch": branch,
        "git_status_short_no_untracked": tracked_status.splitlines(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "disk_free_gb": disk_free_gb,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "optional_missing": optional_missing,
    }


def main() -> int:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    precheck = _precheck(cfg)
    if precheck["blockers"]:
        blocker = {
            "task_name": cfg["task_name"],
            "training_executed": False,
            "blockers": precheck["blockers"],
            "precheck": precheck,
        }
        for key in ("val_metrics_path", "test_metrics_path", "model_selection_table_path", "bucket_diagnostic_path"):
            _write_json(REPO_ROOT / cfg["outputs"][key], blocker)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text(
            "# FINAL360K blocked\n\n" + "\n".join(f"- {b}" for b in precheck["blockers"]) + "\n",
            encoding="utf-8",
        )
        return 1

    _seed_everything(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"]["use_cuda_if_available"]) else "cpu")
    ckpt_dir = REPO_ROOT / cfg["outputs"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    model, model_cfg, load_status = _load_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], device, cfg)
    thresholds = _build_weight_thresholds(cfg)

    train_loader, train_subset = _loader("train", cfg)
    val_loader, _ = _loader("val", cfg)
    test_loader, _ = _loader("test", cfg)
    print(
        f"[FINAL360K] device={device} train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)} train_subset={train_subset}",
        flush=True,
    )

    fine_keys = ("struct360b_",)
    coarse_params: List[torch.nn.Parameter] = []
    fine_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in fine_keys):
            fine_params.append(param)
        else:
            coarse_params.append(param)
    optim_groups = []
    if coarse_params:
        optim_groups.append({"params": coarse_params, "lr": float(cfg["training"]["coarse_lr"])})
    if fine_params:
        optim_groups.append({"params": fine_params, "lr": float(cfg["training"]["fine_lr"])})
    optimizer = torch.optim.AdamW(optim_groups, lr=float(cfg["training"]["fine_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")

    best_score = float("inf")
    best_epoch = -1
    best_val_eval: Optional[Dict[str, Any]] = None
    model_selection_rows: List[Dict[str, Any]] = []
    weight_distribution_accum: List[float] = []
    epoch_bucket_counts: Dict[str, Counter] = {"tmag": Counter(), "rot": Counter()}

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        print(f"[FINAL360K] epoch {epoch}/{cfg['training']['epochs']} start", flush=True)
        epoch_losses = defaultdict(list)
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 20 == 0:
                print(f"[FINAL360K] train epoch {epoch}: batch {batch_idx}/{len(train_loader)}", flush=True)
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
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                R_pred, _t_pred, aux = model(IA, IB, dt_world=dt_world_t)
                final_loss = _pose_loss_final360k(
                    R_pred=R_pred,
                    tdir_pred_B=aux["t_dir_out"],
                    tmag_pred=aux["t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    loss_cfg=cfg["loss"],
                    thresholds=thresholds,
                    tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
                )
                coarse_loss = _pose_loss_final360k(
                    R_pred=aux["coarse_R"],
                    tdir_pred_B=aux["coarse_t_dir_out"],
                    tmag_pred=aux["coarse_t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    loss_cfg=cfg["loss"],
                    thresholds=thresholds,
                    tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
                )
                residual_reg = model.residual_regularization(aux)
                loss_total = (
                    final_loss["loss_total"]
                    + float(cfg["loss"]["coarse_aux_weight"]) * coarse_loss["loss_total"]
                    + float(cfg["loss"]["residual_reg_weight"]) * residual_reg["loss"]
                )

            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
                optimizer.step()

            epoch_losses["loss_total"].append(float(loss_total.detach().cpu()))
            epoch_losses["loss_final"].append(float(final_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_coarse_aux"].append(float(coarse_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_rot"].append(float(final_loss["loss_rot"].detach().cpu()))
            epoch_losses["loss_tdir"].append(float(final_loss["loss_tdir"].detach().cpu()))
            epoch_losses["loss_tmag"].append(float(final_loss["loss_tmag"].detach().cpu()))
            epoch_losses["loss_scale_stability"].append(float(final_loss["loss_scale_stability"].detach().cpu()))
            epoch_losses["loss_antiparallel"].append(float(final_loss["loss_antiparallel"].detach().cpu()))
            epoch_losses["loss_residual_reg"].append(float(residual_reg["loss"].detach().cpu()))
            weight_distribution_accum.extend(final_loss["tdir_weights"].detach().cpu().numpy().tolist())
            epoch_bucket_counts["tmag"].update(final_loss["tdir_weight_stats"]["tmag_bucket_counts_batch"])
            epoch_bucket_counts["rot"].update(final_loss["tdir_weight_stats"]["rot_bucket_counts_batch"])

        val_eval = _evaluate_pose(
            model,
            val_loader,
            device,
            tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
            max_batches=cfg["evaluation"]["max_val_batches"],
            progress_label=f"val epoch {epoch}",
        )
        val_score = _selection_score(val_eval["metrics"], cfg["evaluation"]["selection_score"])
        print(
            f"[FINAL360K] epoch {epoch} done: val_score={val_score:.6f} signed={val_eval['metrics'].get('signed_tdir_mean_deg')} anti={val_eval['metrics'].get('anti_parallel_rate')} tmag={val_eval['metrics'].get('tmag_median_ratio')} path={val_eval['metrics'].get('path_ratio')}",
            flush=True,
        )
        row = {
            "epoch": epoch,
            "val_score": val_score,
            "val_metrics": val_eval["metrics"],
            "val_diagnostics": val_eval["diagnostics"],
            "train_loss": {k: float(np.mean(v)) for k, v in epoch_losses.items()},
            "test_used_for_selection": False,
        }
        model_selection_rows.append(row)
        _append_jsonl(ckpt_dir / "train_log.jsonl", row)

        metadata = {
            "task_name": cfg["task_name"],
            "epoch": epoch,
            "val_score": val_score,
            "val_metrics": val_eval["metrics"],
            "load_status": load_status,
            "train_subset": train_subset,
            "weight_thresholds": thresholds,
        }
        _save_checkpoint(ckpt_dir / "final.pt", model, optimizer, epoch, model_cfg, metadata)
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_val_eval = val_eval
            _save_checkpoint(ckpt_dir / "best_val.pt", model, optimizer, epoch, model_cfg, metadata)

    if best_epoch < 0 or best_val_eval is None:
        raise RuntimeError("FINAL360K failed to select a best epoch.")

    best_model, _, _ = _load_model(ckpt_dir / "best_val.pt", device, cfg)
    test_eval = _evaluate_pose(
        best_model,
        test_loader,
        device,
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
        max_batches=cfg["evaluation"]["max_test_batches"],
        progress_label="test best",
    )

    val_metrics_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_checkpoint": str(ckpt_dir / "best_val.pt"),
        "best_epoch": best_epoch,
        "val_score": best_score,
        "metrics": best_val_eval["metrics"],
        "diagnostics": best_val_eval["diagnostics"],
        "test_used_for_selection": False,
        "weight_thresholds": thresholds,
    }
    test_metrics_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_checkpoint": str(ckpt_dir / "best_val.pt"),
        "best_epoch": best_epoch,
        "val_score": best_score,
        "metrics": test_eval["metrics"],
        "diagnostics": test_eval["diagnostics"],
        "val_test_discrepancy": _discrepancy(best_val_eval["metrics"], test_eval["metrics"]),
        "test_used_for_selection": False,
        "weight_thresholds": thresholds,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], val_metrics_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], test_metrics_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["model_selection_table_path"], {"rows": model_selection_rows, "best_epoch": best_epoch, "best_score": best_score})

    final360k_rows = _rows_for_split(best_model, split="test", cfg=cfg, device=device, model_name="FINAL360K")
    baseline_model, _, _ = _load_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], device, cfg)
    final360i_rows = _rows_for_split(baseline_model, split="test", cfg=cfg, device=device, model_name="FINAL360I")
    final360j_bucket_diag = _read_git_json(f"{FINAL360J_BRANCH}:reports/FINAL360J_bucket_diagnostic.json")
    bucket_diag, bucket_summary, bucket_flags = _bucket_diagnostic(final360k_rows, final360i_rows, final360j_bucket_diag)
    _write_json(REPO_ROOT / cfg["outputs"]["bucket_diagnostic_path"], bucket_diag)

    final360i_test = _read_json(REPO_ROOT / cfg["inputs"]["final360i_test_metrics"]).get("metrics", {})
    final360j_metrics, final360j_ref_source = _final360j_reference(cfg)
    compared_final360i = _compare_to_baseline(test_eval["metrics"], final360i_test)
    compared_final360j = _compare_to_baseline(test_eval["metrics"], final360j_metrics)
    classification = _classify(test_eval["metrics"])
    if classification in {"strong_success", "balanced_success"}:
        next_task = "run_seed_robustness_before_next_step"
    elif classification == "primary_success":
        next_task = "proceed_to_FINAL360L_translation_head_factorization"
    elif classification == "regression":
        next_task = "keep_FINAL360I_as_main_and_archive_FINAL360K"
    else:
        next_task = "tune_FINAL360K_weights_once_more_conservatively"

    weight_distribution = _float_stats(weight_distribution_accum)
    weight_distribution["tmag_bucket_counts"] = dict(epoch_bucket_counts["tmag"])
    weight_distribution["rot_bucket_counts"] = dict(epoch_bucket_counts["rot"])
    _write_report(
        cfg,
        precheck=precheck,
        thresholds=thresholds,
        best_epoch=best_epoch,
        best_score=best_score,
        best_checkpoint=ckpt_dir / "best_val.pt",
        val_eval=best_val_eval,
        test_eval=test_eval,
        compared_final360i=compared_final360i,
        compared_final360j=compared_final360j,
        classification=classification,
        bucket_diag=bucket_diag,
        bucket_summary=bucket_summary,
        final360j_ref_source=final360j_ref_source,
        final360j_metrics=final360j_metrics,
        weight_distribution=weight_distribution,
        next_task=next_task,
    )
    _write_json(
        ckpt_dir / "metadata.json",
        {
            "task_name": cfg["task_name"],
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selected_checkpoint": str(ckpt_dir / "best_val.pt"),
            "classification": classification,
            "comparison_to_final360i": compared_final360i,
            "comparison_to_final360j": compared_final360j,
            "precheck": precheck,
            "load_status": load_status,
            "train_subset": train_subset,
            "weight_thresholds": thresholds,
            "weight_distribution": weight_distribution,
            "bucket_flags": bucket_flags,
        },
    )
    print(
        json.dumps(
            {
                "task_name": cfg["task_name"],
                "best_epoch": best_epoch,
                "best_score": best_score,
                "classification": classification,
                "test_metrics": test_eval["metrics"],
                "bucket_flags": bucket_flags,
                "next_task": next_task,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
