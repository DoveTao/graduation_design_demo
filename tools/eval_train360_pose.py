#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from pose_head import matrix_geodesic_distance


def _float_stats(vals: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
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


@torch.no_grad()
def evaluate_train360_pose(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    enable_depth_fusion: bool = False,
    tmag_epsilon: float = 1.0e-6,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()

    rot_deg_vals: List[float] = []
    signed_tdir_vals: List[float] = []
    unsigned_tdir_vals: List[float] = []
    tmag_ratio_vals: List[float] = []
    log_tmag_mae_vals: List[float] = []
    anti_parallel_flags: List[float] = []
    pred_lengths: List[float] = []
    gt_lengths: List[float] = []
    nan_count = 0
    inf_count = 0
    num_pairs = 0

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

        R_pred, _t_pred_local, aux = model(
            IA,
            IB,
            enable_depth_fusion=bool(enable_depth_fusion),
            dt_world=dt_world_t,
        )
        tdir_pred = aux.get("t_dir_out", aux.get("t_dir"))
        tmag_pred = aux.get("t_mag", aux.get("final_tmag"))
        if tmag_pred is None:
            log_tmag = aux.get("log_t_mag", aux.get("final_log_tmag"))
            if log_tmag is None:
                raise KeyError("Model output missing both t_mag and log_t_mag.")
            tmag_pred = torch.exp(log_tmag.float())

        check_tensors = [R_pred, tdir_pred, tmag_pred, R_gt, t_gt_vec]
        nan_count += sum(int(torch.isnan(t).sum().item()) for t in check_tensors)
        inf_count += sum(int(torch.isinf(t).sum().item()) for t in check_tensors)

        rot_deg = (
            matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
        )
        tdir_np = tdir_pred.detach().float().cpu().numpy()
        tmag_np = tmag_pred.detach().float().view(-1).cpu().numpy()
        t_gt_np = t_gt_vec.detach().float().cpu().numpy()
        gt_mag_np = torch.linalg.norm(t_gt_vec.float(), dim=-1).detach().cpu().numpy()

        for i in range(R_pred.shape[0]):
            num_pairs += 1
            rot_deg_vals.append(float(rot_deg[i]))
            signed = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=False)
            unsigned = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=True)
            if signed is not None:
                signed_tdir_vals.append(float(signed))
                anti_parallel_flags.append(1.0 if float(np.dot(tdir_np[i], t_gt_np[i])) < 0.0 else 0.0)
            if unsigned is not None:
                unsigned_tdir_vals.append(float(unsigned))
            gt_mag = max(float(gt_mag_np[i]), float(tmag_epsilon))
            pred_mag = max(float(tmag_np[i]), float(tmag_epsilon))
            tmag_ratio_vals.append(float(pred_mag / gt_mag))
            log_tmag_mae_vals.append(abs(math.log(pred_mag) - math.log(gt_mag)))
            pred_lengths.append(pred_mag)
            gt_lengths.append(gt_mag)

    rot_stats = _float_stats(rot_deg_vals)
    signed_stats = _float_stats(signed_tdir_vals)
    unsigned_stats = _float_stats(unsigned_tdir_vals)
    tmag_ratio_stats = _float_stats(tmag_ratio_vals)
    scale_collapse_rate = (
        float(np.mean(np.asarray([1.0 if float(v) < 0.1 else 0.0 for v in tmag_ratio_vals], dtype=np.float64)))
        if tmag_ratio_vals
        else None
    )
    scale_explosion_rate = (
        float(np.mean(np.asarray([1.0 if float(v) > 10.0 else 0.0 for v in tmag_ratio_vals], dtype=np.float64)))
        if tmag_ratio_vals
        else None
    )

    return {
        "count": int(num_pairs),
        "coverage": float(num_pairs / max(len(loader.dataset), 1)),
        "rot_mean_deg": rot_stats["mean"],
        "rot_median_deg": rot_stats["median"],
        "rot_p90_deg": rot_stats["p90"],
        "signed_tdir_mean_deg": signed_stats["mean"],
        "signed_tdir_median_deg": signed_stats["median"],
        "signed_tdir_p90_deg": signed_stats["p90"],
        "unsigned_tdir_mean_deg": unsigned_stats["mean"],
        "unsigned_tdir_median_deg": unsigned_stats["median"],
        "unsigned_tdir_p90_deg": unsigned_stats["p90"],
        "anti_parallel_rate": (
            float(np.mean(np.asarray(anti_parallel_flags, dtype=np.float64)))
            if anti_parallel_flags
            else None
        ),
        "tmag_ratio_p10": tmag_ratio_stats["p10"],
        "tmag_median_ratio": tmag_ratio_stats["median"],
        "tmag_ratio_p50": tmag_ratio_stats["median"],
        "tmag_mean_ratio": tmag_ratio_stats["mean"],
        "tmag_ratio_p90": tmag_ratio_stats["p90"],
        "tmag_p90_ratio": tmag_ratio_stats["p90"],
        "log_tmag_mae": (
            float(np.mean(np.asarray(log_tmag_mae_vals, dtype=np.float64)))
            if log_tmag_mae_vals
            else None
        ),
        "scale_collapse_rate": scale_collapse_rate,
        "scale_explosion_rate": scale_explosion_rate,
        "path_ratio": (
            float(sum(pred_lengths) / max(sum(gt_lengths), 1.0e-12))
            if gt_lengths
            else None
        ),
        "path_length_pred": float(sum(pred_lengths)) if pred_lengths else None,
        "path_length_gt": float(sum(gt_lengths)) if gt_lengths else None,
        "nan_inf_count": int(nan_count + inf_count),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "ate_none": None,
        "ate_se3": None,
        "ate_sim3": None,
    }
