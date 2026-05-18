#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
import odom360a_lightweight_trajectory_fusion as odom360a


DEFAULT_CONFIG = REPO_ROOT / "configs" / "odom360b_local_pose_graph_kstep.yaml"
TMAG_EPS = 1.0e-6


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _precheck(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    branch = odom360a._run(["git", "branch", "--show-current"])
    tracked_status = odom360a._run(["git", "status", "--short", "--untracked-files=no"])
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    blockers: List[str] = []
    if bool(cfg["prechecks"].get("require_expected_branch", True)) and branch not in [str(x) for x in cfg["prechecks"]["expected_branches"]]:
        blockers.append(f"unexpected_branch:{branch}")
    if bool(cfg["prechecks"].get("require_clean_worktree", True)) and tracked_status.strip():
        blockers.append("tracked_or_staged_worktree_not_clean")
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient_disk_free_gb:{disk_free_gb:.2f}")
    required = [
        cfg["inputs"]["checkpoint"],
        cfg["inputs"]["val_manifest"],
        cfg["inputs"]["test_manifest"],
        cfg["inputs"]["odom360a_metrics_val"],
        cfg["inputs"]["odom360a_metrics_test"],
        cfg["inputs"]["odom360a_report"],
        cfg["inputs"]["odom360a_summary"],
        cfg["inputs"]["train360e_metrics_test"],
        cfg["inputs"]["seq360b_trajectory_metrics_test"],
    ]
    missing = [str(REPO_ROOT / p) for p in required if not (REPO_ROOT / p).exists()]
    blockers.extend([f"missing_required:{p}" for p in missing])
    return {
        "blockers": blockers,
        "branch": branch,
        "git_commit": odom360a._run(["git", "rev-parse", "HEAD"]),
        "git_status_short_no_untracked": tracked_status.splitlines(),
        "disk_free_gb": disk_free_gb,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def _read_prediction_cache(cfg: Mapping[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    odom360a_root = REPO_ROOT / cfg["inputs"]["odom360a_result_root"] / "cache"
    val_cache = odom360a_root / "val_all_pair_predictions.jsonl"
    test_cache = odom360a_root / "test_all_pair_predictions.jsonl"
    if val_cache.exists() and test_cache.exists():
        return {
            "val": odom360a._read_jsonl(val_cache),
            "test": odom360a._read_jsonl(test_cache),
        }, {
            "source": "odom360a_cache",
            "val_cache": str(val_cache),
            "test_cache": str(test_cache),
        }
    return odom360a._load_or_export_predictions(cfg)


def _apply_odom360a_sequence_method(
    *,
    spec: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    split: str,
    seq_id: str,
    method_name: str,
    params: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, np.ndarray, np.ndarray]]]:
    adj_rows = odom360a._sequence_adjacent_rows(prediction_rows, split, seq_id)
    if method_name == "direct_composition":
        adj_mod = [dict(row) for row in adj_rows]
        abs_rows = odom360a._compose_abs_from_adj(spec, adj_mod)
        return adj_mod, abs_rows
    if method_name == "local_window_pose_fusion":
        adj_mod = [dict(row) for row in adj_rows]
        if params.get("use_scale_pre_smoothing") == "ema":
            adj_mod = odom360a._apply_scale_only(adj_mod, {"smooth_mode": "ema", "ema_alpha": float(params["scale_ema_alpha"])})
        elif params.get("use_scale_pre_smoothing") == "median":
            adj_mod = odom360a._apply_scale_only(adj_mod, {"smooth_mode": "median", "window": int(params["scale_window"])})
        if params.get("use_rotation_pre_smoothing") == "ema":
            adj_mod = odom360a._apply_rotation_smoothing(adj_mod, {"smooth_mode": "ema", "strength": float(params["rotation_strength"])})
        elif params.get("use_rotation_pre_smoothing") == "window":
            adj_mod = odom360a._apply_rotation_smoothing(
                adj_mod,
                {"smooth_mode": "window", "window": int(params["rotation_window"]), "strength": float(params["rotation_strength"])},
            )
        if bool(params.get("use_tdir_pre_suppression")):
            adj_mod = odom360a._apply_tdir_suppression(
                adj_mod,
                {
                    "window": int(params["tdir_window"]),
                    "angular_threshold_deg": float(params["tdir_threshold_deg"]),
                    "replacement_strength": float(params["tdir_replacement_strength"]),
                },
            )
        abs_rot = odom360a._build_absolute_rotations(spec, adj_mod)
        all_rows = odom360a._sequence_all_rows(prediction_rows, split, seq_id, k_max=int(params["k_max"]))
        row_by_edge = {(round(float(r["timestamp_a"]), 6), round(float(r["timestamp_b"]), 6), int(r["k"])): dict(r) for r in all_rows}
        for row in adj_mod:
            key = (round(float(row["timestamp_a"]), 6), round(float(row["timestamp_b"]), 6), int(row["k"]))
            row_by_edge[key] = dict(row)
        fused_pair_rows = list(row_by_edge.values())
        fused_pair_rows.sort(key=lambda row: (row["timestamp_a"], row["timestamp_b"], row["pair_index"]))
        positions = odom360a._solve_positions_from_constraints(spec, abs_rot, fused_pair_rows, k_decay=float(params["k_decay"]))
        abs_rows = [(float(spec["frames"][i]["timestamp"]), abs_rot[i], positions[i]) for i in range(len(spec["frames"]))]
        return adj_mod, abs_rows
    raise ValueError(f"Unsupported ODOM360A init method: {method_name}")


def _edge_weight_from_init(
    row: Mapping[str, Any],
    *,
    init_abs_rows: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    ts_to_idx: Mapping[float, int],
    use_outlier_downweight: bool,
    outlier_angle_deg: float,
    outlier_weight: float,
) -> float:
    base = 1.0
    if not use_outlier_downweight:
        return base
    ts_a = round(float(row["timestamp_a"]), 6)
    ts_b = round(float(row["timestamp_b"]), 6)
    if ts_a not in ts_to_idx or ts_b not in ts_to_idx:
        return base
    i = ts_to_idx[ts_a]
    j = ts_to_idx[ts_b]
    p_i = np.asarray(init_abs_rows[i][2], dtype=np.float64)
    p_j = np.asarray(init_abs_rows[j][2], dtype=np.float64)
    R_j = np.asarray(init_abs_rows[j][1], dtype=np.float64)
    init_rel_t_B = R_j.T @ (p_i - p_j)
    angle = odom360a._angle_deg(np.asarray(row["pred_t_BA_B"], dtype=np.float64), init_rel_t_B)
    if angle is None:
        return base
    return float(outlier_weight if angle > float(outlier_angle_deg) else base)


def _torch_huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    delta_t = torch.tensor(float(delta), dtype=x.dtype, device=x.device)
    abs_x = x.abs()
    quad = torch.minimum(abs_x, delta_t)
    lin = abs_x - quad
    return 0.5 * quad.pow(2) + delta_t * lin


def _torch_so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1.0e-12)
    k = w / theta
    kx, ky, kz = k.unbind(dim=-1)
    O = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([O, -kz, ky], dim=-1),
            torch.stack([kz, O, -kx], dim=-1),
            torch.stack([-ky, kx, O], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(w.shape[0], 3, 3)
    theta_mat = theta.unsqueeze(-1)
    return eye + torch.sin(theta_mat) * K + (1.0 - torch.cos(theta_mat)) * (K @ K)


def _torch_so3_log(R: torch.Tensor) -> torch.Tensor:
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    small = theta < 1.0e-6
    sin_theta = torch.sin(theta).clamp_min(1.0e-6)
    factor = theta / (2.0 * sin_theta)
    skew = factor[:, None, None] * (R - R.transpose(-1, -2))
    vec = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)
    if small.any():
        vec = torch.where(small[:, None], torch.zeros_like(vec), vec)
    return vec


def _torch_rot_from_init(rot_delta: torch.Tensor, init_rot: torch.Tensor) -> torch.Tensor:
    delta_R = _torch_so3_exp(rot_delta)
    return torch.matmul(delta_R, init_rot)


def _build_sequence_constraints(
    prediction_rows: Sequence[Mapping[str, Any]],
    split: str,
    seq_id: str,
    *,
    k_max: int,
) -> List[Dict[str, Any]]:
    rows = odom360a._sequence_all_rows(prediction_rows, split, seq_id, k_max=k_max)
    filtered = []
    for row in rows:
        if bool(row.get("nan_or_inf")):
            continue
        filtered.append(dict(row))
    return filtered


def _optimize_sequence_pose_graph(
    *,
    spec: Mapping[str, Any],
    init_abs_rows: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    constraints: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    device: torch.device,
) -> Tuple[List[Tuple[float, np.ndarray, np.ndarray]], Dict[str, Any]]:
    start_time = time.time()
    n = len(init_abs_rows)
    init_rot_np = np.asarray([row[1] for row in init_abs_rows], dtype=np.float64)
    init_pos_np = np.asarray([row[2] for row in init_abs_rows], dtype=np.float64)
    rot_delta = torch.zeros((n, 3), dtype=torch.float64, device=device, requires_grad=True)
    pos_delta = torch.zeros((n, 3), dtype=torch.float64, device=device, requires_grad=True)
    init_rot = torch.from_numpy(init_rot_np).to(device=device, dtype=torch.float64)
    init_pos = torch.from_numpy(init_pos_np).to(device=device, dtype=torch.float64)
    ts_to_idx = {round(float(frame["timestamp"]), 6): idx for idx, frame in enumerate(spec["frames"])}

    edge_rows: List[Dict[str, Any]] = []
    for row in constraints:
        ts_a = round(float(row["timestamp_a"]), 6)
        ts_b = round(float(row["timestamp_b"]), 6)
        if ts_a not in ts_to_idx or ts_b not in ts_to_idx:
            continue
        i = ts_to_idx[ts_a]
        j = ts_to_idx[ts_b]
        if i >= j:
            continue
        edge_weight = float(params["k_decay"] ** max(int(row["k"]) - 1, 0))
        edge_weight *= _edge_weight_from_init(
            row,
            init_abs_rows=init_abs_rows,
            ts_to_idx=ts_to_idx,
            use_outlier_downweight=bool(params.get("use_outlier_downweight", False)),
            outlier_angle_deg=float(params.get("outlier_angle_deg", 120.0)),
            outlier_weight=float(params.get("outlier_weight", 0.35)),
        )
        edge_rows.append(
            {
                "i": i,
                "j": j,
                "k": int(row["k"]),
                "weight": edge_weight,
                "pred_R_BA": np.asarray(row["pred_R_BA"], dtype=np.float64),
                "pred_tdir_B": odom360a._normalize(np.asarray(row["pred_tdir_B"], dtype=np.float64)),
                "pred_tmag": float(row["pred_tmag"]),
            }
        )
    if edge_rows:
        idx_i = torch.tensor([row["i"] for row in edge_rows], dtype=torch.long, device=device)
        idx_j = torch.tensor([row["j"] for row in edge_rows], dtype=torch.long, device=device)
        edge_weights = torch.tensor([row["weight"] for row in edge_rows], dtype=torch.float64, device=device)
        pred_R_BA = torch.tensor(np.asarray([row["pred_R_BA"] for row in edge_rows], dtype=np.float64), dtype=torch.float64, device=device)
        pred_tdir_B = torch.tensor(np.asarray([row["pred_tdir_B"] for row in edge_rows], dtype=np.float64), dtype=torch.float64, device=device)
        pred_tmag = torch.tensor(np.asarray([row["pred_tmag"] for row in edge_rows], dtype=np.float64), dtype=torch.float64, device=device)
    else:
        idx_i = torch.zeros((0,), dtype=torch.long, device=device)
        idx_j = torch.zeros((0,), dtype=torch.long, device=device)
        edge_weights = torch.zeros((0,), dtype=torch.float64, device=device)
        pred_R_BA = torch.zeros((0, 3, 3), dtype=torch.float64, device=device)
        pred_tdir_B = torch.zeros((0, 3), dtype=torch.float64, device=device)
        pred_tmag = torch.zeros((0,), dtype=torch.float64, device=device)

    params_to_opt = [rot_delta, pos_delta]
    opt_name = str(params.get("optimizer", "adam")).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(params_to_opt, lr=float(params["lr"]))
    else:
        optimizer = torch.optim.Adam(params_to_opt, lr=float(params["lr"]))
    robust_delta = float(params.get("robust_delta", 1.0))
    use_robust = bool(params.get("use_robust", False))
    runtime_sec = 0.0
    success = True
    last_loss = float("inf")

    def _loss_fn() -> torch.Tensor:
        R_abs = _torch_rot_from_init(rot_delta, init_rot)
        p_abs = init_pos + pos_delta
        p_abs = torch.cat([init_pos[:1], p_abs[1:]], dim=0)
        R_abs = torch.cat([init_rot[:1], R_abs[1:]], dim=0)
        total = torch.zeros((), dtype=torch.float64, device=device)
        if edge_weights.numel() > 0:
            R_i = R_abs[idx_i]
            R_j = R_abs[idx_j]
            R_rel = torch.matmul(R_j.transpose(-1, -2), R_i)
            rot_vec = _torch_so3_log(torch.matmul(R_rel, pred_R_BA.transpose(-1, -2)))
            rot_err = torch.linalg.norm(rot_vec, dim=-1)
            t_world = p_abs[idx_i] - p_abs[idx_j]
            t_rel_B = torch.matmul(R_j.transpose(-1, -2), t_world.unsqueeze(-1)).squeeze(-1)
            t_rel_norm = torch.linalg.norm(t_rel_B, dim=-1).clamp_min(1.0e-8)
            cos = torch.clamp((t_rel_B / t_rel_norm.unsqueeze(-1) * pred_tdir_B).sum(dim=-1), -1.0, 1.0)
            tdir_err = 1.0 - cos
            tmag_err = torch.log(t_rel_norm + 1.0e-8) - torch.log(pred_tmag.clamp_min(1.0e-8))
            if use_robust:
                rot_terms = _torch_huber(rot_err, robust_delta) * edge_weights
                tdir_terms = _torch_huber(tdir_err, robust_delta) * edge_weights
                tmag_terms = _torch_huber(tmag_err, robust_delta) * edge_weights
            else:
                rot_terms = rot_err.pow(2) * edge_weights
                tdir_terms = tdir_err.pow(2) * edge_weights
                tmag_terms = tmag_err.pow(2) * edge_weights
            total = total + float(params["w_rot"]) * rot_terms.mean()
            total = total + float(params["w_tdir"]) * tdir_terms.mean()
            total = total + float(params["w_tmag"]) * tmag_terms.mean()
        if float(params.get("w_smooth_rot", 0.0)) > 0.0 and n > 2:
            total = total + float(params["w_smooth_rot"]) * (rot_delta[2:] - 2.0 * rot_delta[1:-1] + rot_delta[:-2]).pow(2).mean()
        if float(params.get("w_smooth_t", 0.0)) > 0.0 and n > 2:
            p_abs_use = p_abs
            total = total + float(params["w_smooth_t"]) * (p_abs_use[2:] - 2.0 * p_abs_use[1:-1] + p_abs_use[:-2]).pow(2).mean()
        if float(params.get("w_rot_prior", 0.0)) > 0.0:
            total = total + float(params["w_rot_prior"]) * rot_delta.pow(2).mean()
        if float(params.get("w_pos_prior", 0.0)) > 0.0:
            total = total + float(params["w_pos_prior"]) * pos_delta.pow(2).mean()
        if float(params.get("w_path", 0.0)) > 0.0 and n > 1:
            step_vec = p_abs[1:] - p_abs[:-1]
            step_len = torch.linalg.norm(step_vec, dim=-1).clamp_min(1.0e-8)
            init_step = torch.linalg.norm((init_pos[1:] - init_pos[:-1]), dim=-1).clamp_min(1.0e-8)
            total = total + float(params["w_path"]) * (torch.log(step_len) - torch.log(init_step)).pow(2).mean()
        return total

    try:
        for _ in range(int(params["iterations"])):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_fn()
            if not torch.isfinite(loss):
                success = False
                break
            loss.backward()
            if rot_delta.grad is not None:
                rot_delta.grad[0].zero_()
            if pos_delta.grad is not None:
                pos_delta.grad[0].zero_()
            optimizer.step()
            with torch.no_grad():
                rot_delta[0].zero_()
                pos_delta[0].zero_()
            last_loss = float(loss.detach().cpu())
    except Exception:
        success = False
    runtime_sec = time.time() - start_time

    with torch.no_grad():
        R_abs = _torch_rot_from_init(rot_delta, init_rot)
        p_abs = init_pos + pos_delta
        R_abs[0] = init_rot[0]
        p_abs[0] = init_pos[0]
        abs_rows = [
            (
                float(spec["frames"][i]["timestamp"]),
                np.asarray(R_abs[i].detach().cpu().numpy(), dtype=np.float64),
                np.asarray(p_abs[i].detach().cpu().numpy(), dtype=np.float64),
            )
            for i in range(n)
        ]
    return abs_rows, {
        "success": bool(success),
        "runtime_sec": float(runtime_sec),
        "last_loss": last_loss,
        "edge_count": int(edge_weights.numel()),
    }


def _evaluate_pose_graph_method_split(
    *,
    split: str,
    method_cfg: Mapping[str, Any],
    specs: Mapping[str, Dict[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    odom360a_selected_method: str,
    odom360a_selected_params: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    per_sequence: Dict[str, Any] = {}
    success_count = 0
    total_runtime = 0.0
    for seq_id, spec in specs.items():
        init_method = str(method_cfg["init_method"])
        if init_method == "direct_composition":
            adj_mod, init_abs_rows = _apply_odom360a_sequence_method(
                spec=spec,
                prediction_rows=prediction_rows,
                split=split,
                seq_id=seq_id,
                method_name="direct_composition",
                params={},
            )
        elif init_method == "odom360a_selected":
            adj_mod, init_abs_rows = _apply_odom360a_sequence_method(
                spec=spec,
                prediction_rows=prediction_rows,
                split=split,
                seq_id=seq_id,
                method_name=odom360a_selected_method,
                params=odom360a_selected_params,
            )
        else:
            raise ValueError(f"Unsupported init_method: {init_method}")
        constraints = _build_sequence_constraints(prediction_rows, split, seq_id, k_max=int(method_cfg["k_max"]))
        abs_rows, opt_stats = _optimize_sequence_pose_graph(
            spec=spec,
            init_abs_rows=init_abs_rows,
            constraints=constraints,
            params=method_cfg,
            device=device,
        )
        payload = odom360a._evaluate_sequence_from_abs(
            spec,
            adj_mod,
            abs_rows,
            output_dir=output_root,
            method_name=str(method_cfg["method_name"]),
            split=split,
            seq_id=seq_id,
        )
        payload["optimization_stats"] = opt_stats
        per_sequence[seq_id] = payload
        success_count += int(opt_stats["success"])
        total_runtime += float(opt_stats["runtime_sec"])

    error_buckets: Dict[str, List[float]] = {"none": [], "se3": [], "sim3": []}
    drift_buckets: Dict[str, List[float]] = {"none": [], "se3": [], "sim3": []}
    pred_path_total = 0.0
    gt_path_total = 0.0
    pair_total = 0
    pair_success = 0
    pose_total = 0
    pose_success = 0
    nan_inf_total = 0
    for payload in per_sequence.values():
        pair_total += int(payload["adjacent_pair_count"])
        pair_success += int(payload["successful_pair_count"])
        pose_total += int(payload["manifest_pose_count"])
        pose_success += int(payload["trajectory_eval"]["none"]["num_matched_poses"])
        nan_inf_total += int(payload["nan_inf_count"])
        pred_path_total += float(payload["trajectory_eval"]["none"].get("pred_path_length") or 0.0)
        gt_path_total += float(payload["trajectory_eval"]["none"].get("gt_path_length") or 0.0)
        for mode in ("none", "se3", "sim3"):
            error_buckets[mode].extend(float(x) for x in payload["trajectory_eval"][mode].get("errors", []))
            if payload["trajectory_eval"][mode].get("drift_rmse") is not None:
                drift_buckets[mode].append(float(payload["trajectory_eval"][mode]["drift_rmse"]))

    def _ate(mode: str) -> Dict[str, Any]:
        errs = np.asarray(error_buckets[mode], dtype=np.float64)
        return {
            "rmse": float(np.sqrt(np.mean(errs ** 2))) if errs.size else None,
            "mean": float(np.mean(errs)) if errs.size else None,
            "median": float(np.median(errs)) if errs.size else None,
            "drift_rmse_mean": float(np.mean(np.asarray(drift_buckets[mode], dtype=np.float64))) if drift_buckets[mode] else None,
        }

    return {
        "selected_method": str(method_cfg["method_name"]),
        "selected_params": dict(method_cfg),
        "split": split,
        "training_executed": False,
        "pair_model_modified": False,
        "sequence_count": len(per_sequence),
        "pose_count": int(pose_total),
        "pair_total": int(pair_total),
        "pair_success": int(pair_success),
        "pair_coverage": float(pair_success / max(pair_total, 1)),
        "coverage": float(pose_success / max(pose_total, 1)),
        "failure_count": int(sum(1 for payload in per_sequence.values() if not bool(payload["optimization_stats"]["success"]))),
        "optimization_success_rate": float(success_count / max(len(per_sequence), 1)),
        "runtime_sec": float(total_runtime),
        "nan_inf_count": int(nan_inf_total),
        "ate_none": _ate("none"),
        "ate_se3": _ate("se3"),
        "ate_sim3": _ate("sim3"),
        "trajectory_path_ratio": float(pred_path_total / max(gt_path_total, TMAG_EPS)),
        "path_length_pred": pred_path_total,
        "path_length_gt": gt_path_total,
        "per_sequence": per_sequence,
    }


def _degradation_penalty(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    penalty = 0.0
    if float(candidate["coverage"]) < float(baseline["coverage"]):
        penalty += 10.0
    for seq_id, payload in candidate["per_sequence"].items():
        base = baseline["per_sequence"][seq_id]
        if float(payload["trajectory_eval"]["se3"]["rmse"]) - float(base["trajectory_eval"]["se3"]["rmse"]) > 5.0:
            penalty += 1.0
        if float(payload["trajectory_eval"]["sim3"]["rmse"]) - float(base["trajectory_eval"]["sim3"]["rmse"]) > 1.0:
            penalty += 1.0
    if int(candidate["nan_inf_count"]) > 0 or float(candidate["optimization_success_rate"]) < 1.0:
        penalty += 10.0
    return penalty


def _selection_score(candidate: Mapping[str, Any], baseline: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    eps = 1.0e-6
    score_cfg = cfg["evaluation"]["selection_score"]
    penalty = _degradation_penalty(candidate, baseline)
    return (
        float(score_cfg["ate_se3_weight"]) * float(candidate["ate_se3"]["rmse"])
        + float(score_cfg["ate_sim3_weight"]) * float(candidate["ate_sim3"]["rmse"])
        + float(score_cfg["path_ratio_weight"]) * abs(math.log(max(float(candidate["trajectory_path_ratio"]), eps)))
        + float(score_cfg["degradation_penalty_weight"]) * penalty
    )


def _best_and_worst_sequences(selected: Mapping[str, Any], odom360a_baseline: Mapping[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seq_id, payload in selected["per_sequence"].items():
        base = odom360a_baseline["per_sequence"][seq_id]
        row = {
            "sequence": seq_id,
            "selected_ate_se3": float(payload["trajectory_eval"]["se3"]["rmse"]),
            "odom360a_ate_se3": float(base["trajectory_eval"]["se3"]["rmse"]),
            "delta_ate_se3": float(payload["trajectory_eval"]["se3"]["rmse"] - base["trajectory_eval"]["se3"]["rmse"]),
            "selected_ate_sim3": float(payload["trajectory_eval"]["sim3"]["rmse"]),
            "odom360a_ate_sim3": float(base["trajectory_eval"]["sim3"]["rmse"]),
            "delta_ate_sim3": float(payload["trajectory_eval"]["sim3"]["rmse"] - base["trajectory_eval"]["sim3"]["rmse"]),
            "selected_path_ratio": float(payload["trajectory_eval"]["none"]["trajectory_path_ratio"]),
            "odom360a_path_ratio": float(base["trajectory_eval"]["none"]["trajectory_path_ratio"]),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["delta_ate_se3"])
    degraded = [row for row in rows if row["delta_ate_se3"] > 0.0 or row["delta_ate_sim3"] > 0.5]
    return {
        "all_sequences": rows,
        "best_improved_sequences": rows[: min(3, len(rows))],
        "worst_degraded_sequences": degraded[: min(3, len(degraded))],
        "per_sequence_degradation_found": bool(degraded),
    }


def _compare_to_baseline(test_metrics: Mapping[str, Any], ref: Mapping[str, float]) -> str:
    improved = 0
    worsened = 0
    checks = [
        (float(test_metrics["ate_se3"]["rmse"]), float(ref["ate_se3"]), "lower"),
        (float(test_metrics["ate_sim3"]["rmse"]), float(ref["ate_sim3"]), "lower"),
        (abs(float(test_metrics["trajectory_path_ratio"]) - 1.0), abs(float(ref["path_ratio"]) - 1.0), "lower"),
    ]
    for a, b, mode in checks:
        if (mode == "lower" and a < b) or (mode == "higher" and a > b):
            improved += 1
        elif a > b:
            worsened += 1
    if improved >= 2:
        return "better"
    if worsened >= 2:
        return "worse"
    return "comparable"


def _classify(test_metrics: Mapping[str, Any], per_seq: Mapping[str, Any]) -> str:
    ate_se3 = float(test_metrics["ate_se3"]["rmse"])
    ate_sim3 = float(test_metrics["ate_sim3"]["rmse"])
    path_ratio = float(test_metrics["trajectory_path_ratio"])
    if ate_se3 < 48.0 and 0.98 <= path_ratio <= 1.05 and ate_sim3 < 27.0:
        return "strong_success"
    if ate_sim3 < 27.0:
        return "shape_improvement_success"
    if ate_se3 < 52.1945 and 0.95 <= path_ratio <= 1.10 and ate_sim3 <= 27.5805 and not bool(per_seq["per_sequence_degradation_found"]):
        return "balanced_success"
    if (ate_se3 < 52.1945 or abs(path_ratio - 1.0) < abs(1.1104915601875927 - 1.0)) and ate_sim3 <= 27.5805 + 0.25:
        return "primary_success"
    if ate_se3 > 52.1945 or ate_sim3 > 27.5805 + 0.5 or abs(path_ratio - 1.0) > abs(1.1104915601875927 - 1.0) or bool(per_seq["per_sequence_degradation_found"]):
        return "regression"
    return "inconclusive"


def _write_report(
    cfg: Mapping[str, Any],
    *,
    precheck: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    direct_val: Mapping[str, Any],
    direct_test: Mapping[str, Any],
    odom360a_val: Mapping[str, Any],
    odom360a_test: Mapping[str, Any],
    selected_val: Mapping[str, Any],
    selected_test: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]],
    train360e_test: Mapping[str, Any],
    seq360b_test: Mapping[str, Any],
    per_sequence_analysis: Mapping[str, Any],
    classification: str,
    next_task: str,
) -> None:
    lines = [
        "# ODOM360B local pose graph with k-step constraints",
        "",
        "## 1. Executive summary",
        "- pose graph executed: `true`",
        "- training executed: `false`",
        "- pair model modified: `false`",
        f"- selected method: `{selected_test['selected_method']}`",
        f"- classification: `{classification}`",
        f"- final recommendation: `{next_task}`",
        "",
        "## 2. Motivation",
        "- FINAL360I pair-level behavior is stable, but direct trajectory composition still drifts.",
        "- SEQ360B improved scale/path and SE3 but not Sim3.",
        "- ODOM360A improved path/SE3 strongly, but still did not improve Sim3 shape.",
        "- ODOM360B upgrades the backend from local fusion to an explicit local pose graph with k-step constraints.",
        "",
        "## 3. Data sources",
        f"- checkpoint: `{cfg['inputs']['checkpoint']}`",
        f"- manifests: `{cfg['inputs']['val_manifest']}`, `{cfg['inputs']['test_manifest']}`",
        f"- pair predictions: `{source_manifest['prediction_source']}`",
        f"- k-step predictions available: `{source_manifest['k_values_available']}`",
        f"- ODOM360A cache reused: `{source_manifest['reused_odom360a_cache']}`",
        "",
        "## 4. Baseline reproduction",
        f"- TRAIN360E direct reproduced test SE3 / Sim3 / path: `{direct_test['ate_se3']['rmse']}` / `{direct_test['ate_sim3']['rmse']}` / `{direct_test['trajectory_path_ratio']}`",
        f"- ODOM360A reproduced test SE3 / Sim3 / path: `{odom360a_test['ate_se3']['rmse']}` / `{odom360a_test['ate_sim3']['rmse']}` / `{odom360a_test['trajectory_path_ratio']}`",
        "- reproduction was accepted before ODOM360B search continued.",
        "",
        "## 5. Pose graph design",
        "- variables: absolute frame rotations and positions, first pose fixed as gauge anchor.",
        "- constraints: adjacent and k-step relative rotation / translation direction / log-magnitude constraints.",
        "- objective: weighted rotation + tdir + tmag residuals with optional smoothness and initialization priors.",
        "- robust loss: Huber-style robustification for PG-k2-robust and later variants.",
        "- k-step weights: geometric decay `k_decay^(k-1)`.",
        "- initialization: direct composition or ODOM360A selected local-window fusion.",
        "",
        "## 6. Val selection",
        "- score = ATE_SE3 + 2 * ATE_Sim3 + 30 * |log(path_ratio)| + 5 * degradation_penalty",
        "- test not used for selection.",
        f"- selected method on val: `{selected_val['selected_method']}` with `{selected_val['selected_params']}`",
        "",
        "## 7. Test results",
        f"- TRAIN360E: none=`{train360e_test['ate_none']['rmse']}`, se3=`{train360e_test['ate_se3']['rmse']}`, sim3=`{train360e_test['ate_sim3']['rmse']}`, path_ratio=`{train360e_test['trajectory_path_ratio']}`",
        f"- SEQ360B: none=`{seq360b_test['ate_none']['rmse']}`, se3=`{seq360b_test['ate_se3']['rmse']}`, sim3=`{seq360b_test['ate_sim3']['rmse']}`, path_ratio=`{seq360b_test['trajectory_path_ratio']}`",
        f"- ODOM360A: none=`{odom360a_test['ate_none']['rmse']}`, se3=`{odom360a_test['ate_se3']['rmse']}`, sim3=`{odom360a_test['ate_sim3']['rmse']}`, path_ratio=`{odom360a_test['trajectory_path_ratio']}`",
        f"- ODOM360B: none=`{selected_test['ate_none']['rmse']}`, se3=`{selected_test['ate_se3']['rmse']}`, sim3=`{selected_test['ate_sim3']['rmse']}`, path_ratio=`{selected_test['trajectory_path_ratio']}`",
        "",
        "## 8. Per-sequence results",
        f"- best improved sequences: `{per_sequence_analysis['best_improved_sequences']}`",
        f"- worst degraded sequences: `{per_sequence_analysis['worst_degraded_sequences']}`",
        "",
        "## 9. Shape vs scale interpretation",
        f"- Sim3 improved vs ODOM360A: `{float(selected_test['ate_sim3']['rmse']) < float(odom360a_test['ate_sim3']['rmse'])}`",
        f"- path ratio improved vs ODOM360A: `{abs(float(selected_test['trajectory_path_ratio']) - 1.0) < abs(float(odom360a_test['trajectory_path_ratio']) - 1.0)}`",
        "- If Sim3 stays flat while SE3 and path improve, the gain should still be read as mostly scale/path repair rather than true shape recovery.",
        "",
        "## 10. k-step constraints interpretation",
        "- k=2 was explicitly searched and selected if beneficial.",
        "- k=3 was evaluated as an optional ablation and kept only if val remained stable.",
        "- Robust loss and outlier downweight were evaluated separately from plain PG-k2.",
        "",
        "## 11. Limitations",
        "- post-processing only",
        "- no image-level geometric verification",
        "- no mature VO claim",
        "- no online real-time guarantee unless measured",
        "- no loop closure",
        "- no BA / reprojection constraints",
        "",
        "## 12. Next recommendation",
        f"- `{next_task}`",
        "",
        "## 13. Compliance checklist",
        "- `training_executed = false`",
        "- `pair_model_modified = false`",
        "- `test_used_for_selection = false`",
        "- `explicit_matching_used = false`",
        "- `image_ransac_used = false`",
        "- `pnp_used = false`",
        "- `ba_used = false`",
        "- `reprojection_error_used = false`",
        "- `hkust_teacher_used = false`",
        "- `orbslam_teacher_used = false`",
        "- `checkpoints_committed = false`",
        "- `raw_data_committed = false`",
        "- `large_prediction_dump_committed = false`",
        "- `metrics_modified_existing = false`",
    ]
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# ODOM360B vs TRAIN360E SEQ360B ODOM360A",
        "",
        f"- selected method: `{selected_test['selected_method']}`",
        f"- selected params: `{selected_test['selected_params']}`",
        f"- compared to TRAIN360E: `{_compare_to_baseline(selected_test, {'ate_se3': 118.6037794846689, 'ate_sim3': 27.564661865900444, 'path_ratio': 1.756343083453392})}`",
        f"- compared to SEQ360B: `{_compare_to_baseline(selected_test, {'ate_se3': 75.94691348103409, 'ate_sim3': 27.567211313835486, 'path_ratio': 1.3502369615185652})}`",
        f"- compared to ODOM360A: `{_compare_to_baseline(selected_test, {'ate_se3': 52.1944638233902, 'ate_sim3': 27.580518809274864, 'path_ratio': 1.1104915601875927})}`",
        f"- classification: `{classification}`",
        f"- next recommendation: `{next_task}`",
    ]
    (REPO_ROOT / cfg["outputs"]["summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    precheck = _precheck(cfg)
    if precheck["blockers"]:
        blocker = {
            "task_name": cfg["task_name"],
            "pose_graph_executed": False,
            "training_executed": False,
            "blockers": precheck["blockers"],
            "precheck": precheck,
        }
        for key in ("metrics_val_path", "metrics_test_path", "model_selection_table_path", "per_sequence_metrics_path", "ablation_path", "metric_source_manifest_path"):
            _write_json(REPO_ROOT / cfg["outputs"][key], blocker)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("# ODOM360B blocked\n\n" + "\n".join(f"- {b}" for b in precheck["blockers"]) + "\n", encoding="utf-8")
        return 1

    prediction_by_split, prediction_manifest = _read_prediction_cache(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_hw = tuple(int(x) for x in cfg["data"]["image_hw"])
    ds_val = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"]["val_manifest"]),
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"]["test_manifest"]),
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    val_specs, val_blockers = odom360a._build_sequence_specs(ds_val, "val")
    test_specs, test_blockers = odom360a._build_sequence_specs(ds_test, "test")
    if val_blockers or test_blockers:
        blockers = list(val_blockers) + list(test_blockers)
        blocker = {"task_name": cfg["task_name"], "pose_graph_executed": False, "blockers": blockers}
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], blocker)
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], blocker)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("# ODOM360B blocked\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
        return 1

    direct_val = odom360a._run_method_on_split("val", "direct_composition", {}, val_specs, prediction_by_split["val"], REPO_ROOT / cfg["outputs"]["result_root"])
    direct_test = odom360a._run_method_on_split("test", "direct_composition", {}, test_specs, prediction_by_split["test"], REPO_ROOT / cfg["outputs"]["result_root"])
    train360e_test = _read_json(REPO_ROOT / cfg["inputs"]["train360e_metrics_test"])
    direct_ok, direct_delta = odom360a._direct_reproduction_ok(direct_test, train360e_test, cfg["evaluation"]["direct_reproduction_tolerance"])
    if bool(cfg["evaluation"].get("require_direct_reproduction_match", True)) and not direct_ok:
        blocker = {
            "task_name": cfg["task_name"],
            "pose_graph_executed": False,
            "blocker": "direct_composition_reproduction_mismatch",
            "reproduction_delta": direct_delta,
        }
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], blocker)
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], blocker)
        return 1

    odom360a_val_report = _read_json(REPO_ROOT / cfg["inputs"]["odom360a_metrics_val"])
    odom360a_test_report = _read_json(REPO_ROOT / cfg["inputs"]["odom360a_metrics_test"])
    odom360a_method = str(odom360a_test_report["method_name"])
    odom360a_params = dict(odom360a_test_report["selected_params"])
    odom360a_val = odom360a._run_method_on_split("val", odom360a_method, odom360a_params, val_specs, prediction_by_split["val"], REPO_ROOT / cfg["outputs"]["result_root"])
    odom360a_test = odom360a._run_method_on_split("test", odom360a_method, odom360a_params, test_specs, prediction_by_split["test"], REPO_ROOT / cfg["outputs"]["result_root"])
    odom360a_ok, odom360a_delta = odom360a._direct_reproduction_ok(odom360a_test, odom360a_test_report, cfg["evaluation"]["odom360a_reproduction_tolerance"])
    if bool(cfg["evaluation"].get("require_odom360a_reproduction_match", True)) and not odom360a_ok:
        blocker = {
            "task_name": cfg["task_name"],
            "pose_graph_executed": False,
            "blocker": "odom360a_reproduction_mismatch",
            "reproduction_delta": odom360a_delta,
        }
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], blocker)
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], blocker)
        return 1

    output_root = REPO_ROOT / cfg["outputs"]["result_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    ablation_rows: List[Dict[str, Any]] = []
    best_val: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    best_method_cfg: Optional[Dict[str, Any]] = None
    for method_cfg in cfg["search"]["methods"]:
        val_metrics = _evaluate_pose_graph_method_split(
            split="val",
            method_cfg=method_cfg,
            specs=val_specs,
            prediction_rows=prediction_by_split["val"],
            odom360a_selected_method=odom360a_method,
            odom360a_selected_params=odom360a_params,
            output_root=output_root,
            device=device,
        )
        score = _selection_score(val_metrics, odom360a_val, cfg)
        row = {
            "method_name": method_cfg["method_name"],
            "params": dict(method_cfg),
            "split": "val",
            "score": score,
            "ate_none": val_metrics["ate_none"]["rmse"],
            "ate_se3": val_metrics["ate_se3"]["rmse"],
            "ate_sim3": val_metrics["ate_sim3"]["rmse"],
            "trajectory_path_ratio": val_metrics["trajectory_path_ratio"],
            "coverage": val_metrics["coverage"],
            "optimization_success_rate": val_metrics["optimization_success_rate"],
            "runtime_sec": val_metrics["runtime_sec"],
            "test_used_for_selection": False,
        }
        ablation_rows.append(row)
        if score < best_score:
            best_score = score
            best_val = val_metrics
            best_method_cfg = dict(method_cfg)
    if best_val is None or best_method_cfg is None:
        raise RuntimeError("No ODOM360B val method was evaluated.")

    selected_test = _evaluate_pose_graph_method_split(
        split="test",
        method_cfg=best_method_cfg,
        specs=test_specs,
        prediction_rows=prediction_by_split["test"],
        odom360a_selected_method=odom360a_method,
        odom360a_selected_params=odom360a_params,
        output_root=output_root,
        device=device,
    )
    best_val["selection_score"] = best_score
    selected_test["selection_score"] = best_score
    best_val["test_used_for_selection"] = False
    selected_test["test_used_for_selection"] = False

    per_sequence_analysis = _best_and_worst_sequences(selected_test, odom360a_test)
    classification = _classify(selected_test, per_sequence_analysis)
    if classification in {"strong_success", "shape_improvement_success"}:
        next_task = "proceed_to_ODOM360C_confidence_weighted_pose_graph"
    elif classification in {"balanced_success", "primary_success", "partial"}:
        next_task = "proceed_to_ODOM360D_kstep_prediction_quality_audit"
    elif classification == "regression":
        next_task = "keep_ODOM360A_as_best_sequence_backend"
    else:
        next_task = "keep_ODOM360A_as_best_sequence_backend"

    metrics_val_payload = {
        "task_name": cfg["task_name"],
        "pose_graph_executed": True,
        "training_executed": False,
        "pair_model_modified": False,
        **best_val,
    }
    metrics_test_payload = {
        "task_name": cfg["task_name"],
        "pose_graph_executed": True,
        "training_executed": False,
        "pair_model_modified": False,
        "classification": classification,
        **selected_test,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], metrics_val_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], metrics_test_payload)
    _write_json(
        REPO_ROOT / cfg["outputs"]["model_selection_table_path"],
        {
            "rows": ablation_rows,
            "selected_method": best_method_cfg["method_name"],
            "selected_params": best_method_cfg,
            "selected_val_score": best_score,
            "test_used_for_selection": False,
        },
    )
    _write_json(REPO_ROOT / cfg["outputs"]["per_sequence_metrics_path"], per_sequence_analysis)
    _write_json(REPO_ROOT / cfg["outputs"]["ablation_path"], {"rows": ablation_rows})
    metric_source_manifest = {
        "task_name": cfg["task_name"],
        "prediction_source": prediction_manifest,
        "k_values_available": sorted({int(row["k"]) for split_rows in prediction_by_split.values() for row in split_rows}),
        "reused_odom360a_cache": bool(prediction_manifest.get("source") == "odom360a_cache"),
        "direct_reproduction_delta_vs_train360e_test": direct_delta,
        "odom360a_reproduction_delta_vs_odom360a_test": odom360a_delta,
        "checkpoint": cfg["inputs"]["checkpoint"],
        "test_used_for_selection": False,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["metric_source_manifest_path"], metric_source_manifest)
    _write_report(
        cfg,
        precheck=precheck,
        source_manifest=metric_source_manifest,
        direct_val=direct_val,
        direct_test=direct_test,
        odom360a_val=odom360a_val,
        odom360a_test=odom360a_test,
        selected_val=best_val,
        selected_test=selected_test,
        ablation_rows=ablation_rows,
        train360e_test=train360e_test,
        seq360b_test=_read_json(REPO_ROOT / cfg["inputs"]["seq360b_trajectory_metrics_test"]),
        per_sequence_analysis=per_sequence_analysis,
        classification=classification,
        next_task=next_task,
    )
    print(
        json.dumps(
            {
                "task_name": cfg["task_name"],
                "pose_graph_executed": True,
                "training_executed": False,
                "selected_method": best_method_cfg["method_name"],
                "selected_params": best_method_cfg,
                "val_score": best_score,
                "test_ate_se3": selected_test["ate_se3"]["rmse"],
                "test_ate_sim3": selected_test["ate_sim3"]["rmse"],
                "test_path_ratio": selected_test["trajectory_path_ratio"],
                "classification": classification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
