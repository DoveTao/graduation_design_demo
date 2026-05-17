#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train360.core.config import Config


DEFAULT_CONFIG = REPO_ROOT / "configs" / "odom360a_lightweight_trajectory_fusion.yaml"
TMAG_EPS = 1.0e-6


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _run(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), cwd=REPO_ROOT, text=True).strip()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _mean(vals: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else None


def _so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    if theta < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    skew = (R - R.T) * (0.5 * theta / max(math.sin(theta), 1.0e-12))
    return np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def _so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1.0e-8:
        K = np.asarray(
            [
                [0.0, -w[2], w[1]],
                [w[2], 0.0, -w[0]],
                [-w[1], w[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + K
    k = w / theta
    K = np.asarray(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _normalize(v: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = _normalize(a)
    b = _normalize(b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-8 or nb <= 1.0e-8:
        return None
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _score(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    eps = 1.0e-6
    ate_se3 = float(metrics["ate_se3"]["rmse"])
    ate_sim3 = float(metrics["ate_sim3"]["rmse"])
    path_ratio = float(metrics["trajectory_path_ratio"])
    score_cfg = cfg["evaluation"]["selection_score"]
    return (
        float(score_cfg["ate_se3_weight"]) * ate_se3
        + float(score_cfg["ate_sim3_weight"]) * ate_sim3
        + float(score_cfg["path_ratio_weight"]) * abs(math.log(max(path_ratio, eps)))
    )


def _rot_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _write_tum(path: Path, rows: Sequence[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    lines: List[str] = []
    for ts, R_w, t_w in rows:
        qx, qy, qz, qw = _rot_to_quat_xyzw(R_w)
        lines.append(
            f"{float(ts):.6f} {float(t_w[0]):.9f} {float(t_w[1]):.9f} {float(t_w[2]):.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    dim, n = src.shape
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered @ src_centered.T) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(src_centered * src_centered) / float(n)
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1.0e-12))
    else:
        scale = 1.0
    t = (mean_dst - scale * R @ mean_src).reshape(dim)
    return scale, R, t


def _apply_alignment(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ points.T)).T + t.reshape(1, 3)


def _evaluate_pose_errors(
    gt_rows: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    pred_rows: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    *,
    alignment: str,
    gt_pose_count_full: int,
) -> Dict[str, Any]:
    gt_by_ts = {round(float(ts), 6): (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)) for ts, R, t in gt_rows}
    pred_by_ts = {round(float(ts), 6): (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)) for ts, R, t in pred_rows}
    matched_ts = [ts for ts in sorted(pred_by_ts.keys()) if ts in gt_by_ts]
    if len(matched_ts) < 2:
        return {
            "status": "insufficient_matches",
            "alignment_mode": alignment,
            "num_est_poses": len(pred_rows),
            "num_gt_poses": len(gt_rows),
            "num_gt_poses_full": int(gt_pose_count_full),
            "num_matched_poses": len(matched_ts),
            "pose_coverage": float(len(matched_ts) / max(len(gt_rows), 1)),
            "full_gt_pose_coverage": float(len(matched_ts) / max(int(gt_pose_count_full), 1)),
        }
    gt_pts = np.asarray([gt_by_ts[ts][1] for ts in matched_ts], dtype=np.float64)
    pred_pts_raw = np.asarray([pred_by_ts[ts][1] for ts in matched_ts], dtype=np.float64)
    if alignment == "none":
        scale = 1.0
        R_align = np.eye(3, dtype=np.float64)
        t_align = np.zeros(3, dtype=np.float64)
    elif alignment == "se3":
        scale, R_align, t_align = _umeyama(pred_pts_raw.T, gt_pts.T, with_scale=False)
    elif alignment == "sim3":
        scale, R_align, t_align = _umeyama(pred_pts_raw.T, gt_pts.T, with_scale=True)
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")
    pred_pts_aligned = _apply_alignment(pred_pts_raw, scale, R_align, t_align)
    errors = np.linalg.norm(pred_pts_aligned - gt_pts, axis=1)
    step_errors = np.linalg.norm((pred_pts_aligned[1:] - pred_pts_aligned[:-1]) - (gt_pts[1:] - gt_pts[:-1]), axis=1)
    gt_path_len = _path_length(gt_pts)
    pred_path_len_raw = _path_length(pred_pts_raw)
    return {
        "status": "ok",
        "alignment_mode": alignment,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "drift_rmse": float(np.sqrt(np.mean(step_errors ** 2))) if step_errors.size > 0 else 0.0,
        "drift_mean": float(np.mean(step_errors)) if step_errors.size > 0 else 0.0,
        "trajectory_path_ratio": float(pred_path_len_raw / max(gt_path_len, 1.0e-12)),
        "pred_path_length": pred_path_len_raw,
        "gt_path_length": gt_path_len,
        "num_est_poses": len(pred_rows),
        "num_gt_poses": len(gt_rows),
        "num_gt_poses_full": int(gt_pose_count_full),
        "num_matched_poses": len(matched_ts),
        "pose_coverage": float(len(matched_ts) / max(len(gt_rows), 1)),
        "full_gt_pose_coverage": float(len(matched_ts) / max(int(gt_pose_count_full), 1)),
        "errors": [float(x) for x in errors.tolist()],
    }


def _relative_from_absolute(R_a: np.ndarray, t_a: np.ndarray, R_b: np.ndarray, t_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R_b.T @ R_a, R_b.T @ (t_a - t_b)


def _angle_from_rot(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _relative_metric(R_pred: np.ndarray, t_pred: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> Dict[str, Any]:
    signed = _angle_deg(t_pred, t_gt)
    unsigned = None if signed is None else float(np.degrees(np.arccos(abs(np.clip(np.dot(_normalize(t_pred), _normalize(t_gt)), -1.0, 1.0)))))
    cosine = None
    if signed is not None:
        cosine = float(np.dot(_normalize(t_pred), _normalize(t_gt)))
    gt_mag = float(np.linalg.norm(t_gt))
    pred_mag = float(np.linalg.norm(t_pred))
    return {
        "rot_deg": _angle_from_rot(R_pred @ R_gt.T),
        "signed_tdir_deg": signed,
        "unsigned_tdir_deg": unsigned,
        "anti_parallel_flag": bool(cosine is not None and cosine < 0.0),
        "tmag_ratio": float(pred_mag / max(gt_mag, TMAG_EPS)),
        "pred_step_length": pred_mag,
        "gt_step_length": gt_mag,
    }


def _build_sequence_specs(dataset: Dset2CCanonicalPairDataset, split: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    sequences: Dict[str, Dict[str, Any]] = {}
    blockers: List[str] = []
    adjacent_samples = [dict(sample) for sample in dataset.samples if sample["pair_type"] == "adjacent" and int(sample["k"]) == 1]
    for sample in adjacent_samples:
        seq_id = str(sample["sequence_id"])
        bucket = sequences.setdefault(seq_id, {"sequence_id": seq_id, "split": split, "adjacent_rows": []})
        bucket["adjacent_rows"].append(sample)
    for seq_id, spec in sequences.items():
        rows = sorted(spec["adjacent_rows"], key=lambda row: (float(row["timestamp_a"]), float(row["timestamp_b"]), int(row["pair_index"])))
        if not rows:
            blockers.append(f"{split}/{seq_id}: no adjacent rows recovered from manifest")
            continue
        frames: List[Dict[str, Any]] = [{"timestamp": float(rows[0]["timestamp_a"]), "image_path": str(rows[0]["image_path_a"])}]
        gt_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
        first_ts = float(rows[0]["timestamp_a"])
        current_R = np.eye(3, dtype=np.float64)
        current_t = np.zeros(3, dtype=np.float64)
        gt_rows.append((first_ts, current_R.copy(), current_t.copy()))
        for row in rows:
            expected_prev = frames[-1]["image_path"]
            if str(row["image_path_a"]) != expected_prev:
                blockers.append(f"{split}/{seq_id}: adjacent chain discontinuity at pair_index={row['pair_index']}")
            frames.append({"timestamp": float(row["timestamp_b"]), "image_path": str(row["image_path_b"])})
            gt_R = np.asarray(row["R_BA"], dtype=np.float64)
            gt_t = np.asarray(row["t_BA_B"], dtype=np.float64)
            next_R = current_R @ gt_R.T
            next_t = current_t - next_R @ gt_t
            gt_rows.append((float(row["timestamp_b"]), next_R, next_t))
            current_R = next_R
            current_t = next_t
        spec["adjacent_rows"] = rows
        spec["frames"] = frames
        spec["frame_count"] = len(frames)
        spec["adjacent_pair_count"] = len(rows)
        spec["pair_count"] = len([sample for sample in dataset.samples if sample["sequence_id"] == seq_id])
        spec["manifest_pose_count"] = len(gt_rows)
        spec["full_gt_pose_count"] = len(gt_rows)
        spec["gt_manifest_rows"] = gt_rows
        spec["gt_full_rows"] = gt_rows
        spec["gt_manifest_coverage"] = 1.0
        spec["full_gt_coverage_from_manifest"] = 1.0
    return sequences, blockers


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg = _cfg_from_dict(dict(payload["cfg"]))
    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    load_result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, {"missing_keys": list(load_result.missing_keys), "unexpected_keys": list(load_result.unexpected_keys)}


def _classify(test_metrics: Mapping[str, Any]) -> str:
    ate_se3 = float(test_metrics["ate_se3"]["rmse"])
    ate_sim3 = float(test_metrics["ate_sim3"]["rmse"])
    path_ratio = float(test_metrics["trajectory_path_ratio"])
    train360e_se3 = 118.6037794846689
    train360e_sim3 = 27.564661865900444
    train360e_path = 1.756343083453392
    seq360b_se3 = 75.94691348103409
    seq360b_sim3 = 27.567211313835486
    seq360b_path = 1.3502369615185652
    if ate_se3 < 70.0 and 0.95 <= path_ratio <= 1.10 and ate_sim3 < 27.0:
        return "strong_success"
    if ate_se3 < seq360b_se3 and 0.9 <= path_ratio <= 1.2 and ate_sim3 <= seq360b_sim3:
        return "balanced_success"
    if ate_sim3 < 27.0:
        return "shape_improvement_success"
    if ate_se3 < seq360b_se3 or abs(path_ratio - 1.0) < abs(seq360b_path - 1.0):
        if ate_sim3 <= seq360b_sim3 + 0.25:
            return "primary_success"
        return "partial"
    if ate_se3 > train360e_se3 or ate_sim3 > train360e_sim3 + 1.0 or abs(path_ratio - 1.0) > abs(train360e_path - 1.0):
        return "regression"
    return "inconclusive"


def _compare_to_baseline(test_metrics: Mapping[str, Any], ref: Mapping[str, float]) -> str:
    improved = 0
    worsened = 0
    pairs = [
        (float(test_metrics["ate_se3"]["rmse"]), float(ref["ate_se3"]), "lower"),
        (float(test_metrics["ate_sim3"]["rmse"]), float(ref["ate_sim3"]), "lower"),
        (abs(float(test_metrics["trajectory_path_ratio"]) - 1.0), abs(float(ref["path_ratio"]) - 1.0), "lower"),
    ]
    for a, b, mode in pairs:
        if (mode == "lower" and a < b) or (mode == "higher" and a > b):
            improved += 1
        elif a > b:
            worsened += 1
    if improved >= 2:
        return "better"
    if worsened >= 2:
        return "worse"
    return "comparable"


def _precheck(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"])
    tracked_status = _run(["git", "status", "--short", "--untracked-files=no"])
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    blockers: List[str] = []
    if branch not in [str(x) for x in cfg["prechecks"]["expected_branches"]]:
        blockers.append(f"unexpected_branch:{branch}")
    if tracked_status.strip():
        blockers.append("tracked_or_staged_worktree_not_clean")
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient_disk_free_gb:{disk_free_gb:.2f}")
    required = [
        cfg["inputs"]["checkpoint"],
        cfg["inputs"]["val_manifest"],
        cfg["inputs"]["test_manifest"],
        cfg["inputs"]["final360i_metrics_test"],
        cfg["inputs"]["train360e_metrics_test"],
        cfg["inputs"]["seq360b_metrics_test"],
        cfg["inputs"]["seq360b_trajectory_metrics_test"],
        cfg["inputs"]["results_table"],
        cfg["inputs"]["results_narrative"],
        cfg["inputs"]["current_mainline"],
        cfg["inputs"]["delivery_handoff"],
        cfg["inputs"]["train360e_report"],
        cfg["inputs"]["seq360b_report"],
    ]
    missing = [str(REPO_ROOT / p) for p in required if not (REPO_ROOT / p).exists()]
    blockers.extend([f"missing_required:{p}" for p in missing])
    return {
        "blockers": blockers,
        "branch": branch,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status_short_no_untracked": tracked_status.splitlines(),
        "disk_free_gb": disk_free_gb,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def _iter_all_pair_predictions(
    model: torch.nn.Module,
    dataset: Dset2CCanonicalPairDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    allow_k: Sequence[int],
) -> List[Dict[str, Any]]:
    allow_set = {int(k) for k in allow_k}
    indices = [idx for idx, sample in enumerate(dataset.samples) if int(sample["k"]) in allow_set]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )
    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            IA = batch["IA"].to(device, non_blocking=True)
            IB = batch["IB"].to(device, non_blocking=True)
            dt_world = batch["meta"]["dt_world"]
            if torch.is_tensor(dt_world):
                dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
            else:
                dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)
            R_pred_t, _t_local, aux = model(IA, IB, dt_world=dt_world_t)
            R_pred = R_pred_t.detach().float().cpu().numpy()
            tdir_pred = aux["t_dir_out"].detach().float().cpu().numpy()
            tmag_pred = aux["t_mag"].detach().float().view(-1).cpu().numpy()
            gt_R = batch["R_gt"].detach().float().cpu().numpy()
            gt_t = batch["t_gt_vec"].detach().float().cpu().numpy()
            pair_indices = batch["pair_index"].detach().cpu().numpy()
            k_vals = batch["k"].detach().cpu().numpy()
            seq_ids = list(batch["meta"]["sequence_id"])
            splits = list(batch["meta"]["split"])
            timestamps_a = batch["meta"]["timestamp_a"].detach().cpu().numpy()
            timestamps_b = batch["meta"]["timestamp_b"].detach().cpu().numpy()
            image_paths_a = list(batch["meta"]["image_path_a"])
            image_paths_b = list(batch["meta"]["image_path_b"])
            pair_types = list(batch["meta"]["pair_type"])
            for i in range(R_pred.shape[0]):
                pred_t = np.asarray(tdir_pred[i], dtype=np.float64) * float(tmag_pred[i])
                finite = bool(np.isfinite(R_pred[i]).all() and np.isfinite(pred_t).all() and math.isfinite(float(tmag_pred[i])))
                rows.append(
                    {
                        "split": str(splits[i]),
                        "sequence": str(seq_ids[i]),
                        "pair_index": int(pair_indices[i]),
                        "k": int(k_vals[i]),
                        "pair_type": str(pair_types[i]),
                        "timestamp_a": float(timestamps_a[i]),
                        "timestamp_b": float(timestamps_b[i]),
                        "frame_i": int(Path(str(image_paths_a[i])).stem),
                        "frame_j": int(Path(str(image_paths_b[i])).stem),
                        "pred_R_BA": np.asarray(R_pred[i], dtype=np.float64).tolist(),
                        "pred_tdir_B": np.asarray(tdir_pred[i], dtype=np.float64).tolist(),
                        "pred_tmag": float(tmag_pred[i]),
                        "pred_t_BA_B": pred_t.tolist(),
                        "gt_R_BA": np.asarray(gt_R[i], dtype=np.float64).tolist(),
                        "gt_t_BA_B": np.asarray(gt_t[i], dtype=np.float64).tolist(),
                        "nan_or_inf": not finite,
                    }
                )
    rows.sort(key=lambda row: (row["split"], row["sequence"], row["timestamp_a"], row["timestamp_b"], row["pair_index"]))
    return rows


def _load_or_export_predictions(cfg: Mapping[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    root = REPO_ROOT / cfg["outputs"]["result_root"]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "reused": {},
        "generated": {},
    }
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    allow_k = [int(x) for x in cfg["data"]["export_all_pairs_for_k"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for split in ("val", "test"):
        cache_path = cache_dir / f"{split}_all_pair_predictions.jsonl"
        if bool(cfg["data"]["reuse_cache_if_present"]) and cache_path.exists():
            rows = _read_jsonl(cache_path)
            manifest["reused"][split] = str(cache_path)
            by_split[split] = rows
            continue
        if model is None:
            model, _ = _load_model_from_checkpoint(REPO_ROOT / cfg["inputs"]["checkpoint"], device)
        ds = Dset2CCanonicalPairDataset(
            str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
            expected_split=split,
            image_hw=tuple(int(x) for x in cfg["data"]["image_hw"]),
            require_paths=bool(cfg["data"]["require_paths"]),
            skip_invalid=bool(cfg["data"]["skip_invalid"]),
        )
        rows = _iter_all_pair_predictions(
            model,
            ds,
            device,
            batch_size=int(cfg["data"]["eval_batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
            allow_k=allow_k,
        )
        _write_jsonl(cache_path, rows)
        manifest["generated"][split] = str(cache_path)
        by_split[split] = rows
    return by_split, manifest


def _sequence_adjacent_rows(rows: Sequence[Mapping[str, Any]], split: str, seq_id: str) -> List[Dict[str, Any]]:
    out = [
        dict(row)
        for row in rows
        if str(row["split"]) == split and str(row["sequence"]) == seq_id and int(row["k"]) == 1 and str(row["pair_type"]) == "adjacent"
    ]
    out.sort(key=lambda row: (row["timestamp_a"], row["timestamp_b"], row["pair_index"]))
    return out


def _sequence_all_rows(rows: Sequence[Mapping[str, Any]], split: str, seq_id: str, *, k_max: int) -> List[Dict[str, Any]]:
    out = [
        dict(row)
        for row in rows
        if str(row["split"]) == split and str(row["sequence"]) == seq_id and int(row["k"]) <= int(k_max)
    ]
    out.sort(key=lambda row: (row["timestamp_a"], row["timestamp_b"], row["pair_index"]))
    return out


def _smooth_log_sequence(values: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    mode = str(params.get("smooth_mode"))
    if mode == "ema":
        alpha = float(params["ema_alpha"])
        out = np.zeros_like(values, dtype=np.float64)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out
    if mode == "median":
        window = int(params["window"])
        radius = window // 2
        out = np.zeros_like(values, dtype=np.float64)
        for i in range(len(values)):
            lo = max(0, i - radius)
            hi = min(len(values), i + radius + 1)
            out[i] = float(np.median(values[lo:hi]))
        return out
    raise ValueError(f"Unsupported scale smooth mode: {mode}")


def _apply_scale_only(adj_rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not adj_rows:
        return []
    rows = [dict(row) for row in adj_rows]
    log_mag = np.asarray([math.log(max(float(row["pred_tmag"]), TMAG_EPS)) for row in rows], dtype=np.float64)
    smooth_log = _smooth_log_sequence(log_mag, params)
    for row, log_v in zip(rows, smooth_log):
        mag = float(math.exp(float(log_v)))
        direction = _normalize(np.asarray(row["pred_tdir_B"], dtype=np.float64))
        row["pred_tmag"] = mag
        row["pred_t_BA_B"] = (direction * mag).tolist()
    return rows


def _apply_rotation_smoothing(adj_rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in adj_rows]
    if not rows:
        return rows
    mode = str(params.get("smooth_mode"))
    log_vecs = np.asarray([_so3_log(np.asarray(row["pred_R_BA"], dtype=np.float64)) for row in rows], dtype=np.float64)
    smoothed = np.zeros_like(log_vecs)
    if mode == "ema":
        strength = float(params["strength"])
        smoothed[0] = log_vecs[0]
        for i in range(1, len(rows)):
            smoothed[i] = (1.0 - strength) * smoothed[i - 1] + strength * log_vecs[i]
    elif mode == "window":
        window = int(params["window"])
        strength = float(params["strength"])
        radius = window // 2
        for i in range(len(rows)):
            lo = max(0, i - radius)
            hi = min(len(rows), i + radius + 1)
            mean_v = np.mean(log_vecs[lo:hi], axis=0)
            smoothed[i] = (1.0 - strength) * log_vecs[i] + strength * mean_v
    else:
        raise ValueError(f"Unsupported rotation smooth mode: {mode}")
    for row, v in zip(rows, smoothed):
        row["pred_R_BA"] = _so3_exp(v).tolist()
    return rows


def _apply_tdir_suppression(adj_rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in adj_rows]
    if not rows:
        return rows
    window = int(params["window"])
    threshold = float(params["angular_threshold_deg"])
    replace_strength = float(params["replacement_strength"])
    radius = window // 2
    dirs = [_normalize(np.asarray(row["pred_tdir_B"], dtype=np.float64)) for row in rows]
    mags = [float(row["pred_tmag"]) for row in rows]
    out_dirs: List[np.ndarray] = []
    for i, cur in enumerate(dirs):
        lo = max(0, i - radius)
        hi = min(len(rows), i + radius + 1)
        neigh = [dirs[j] for j in range(lo, hi) if j != i and np.linalg.norm(dirs[j]) > 0.0]
        if not neigh:
            out_dirs.append(cur)
            continue
        ref = _normalize(np.mean(np.asarray(neigh, dtype=np.float64), axis=0))
        angle = _angle_deg(cur, ref)
        if angle is None or angle <= threshold:
            out_dirs.append(cur)
            continue
        fused = _normalize((1.0 - replace_strength) * cur + replace_strength * ref)
        out_dirs.append(fused if np.linalg.norm(fused) > 0.0 else cur)
    for row, new_dir, mag in zip(rows, out_dirs, mags):
        row["pred_tdir_B"] = new_dir.tolist()
        row["pred_t_BA_B"] = (new_dir * mag).tolist()
    return rows


def _compose_abs_from_adj(spec: Mapping[str, Any], adj_rows: Sequence[Mapping[str, Any]]) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    gt_manifest_rows = list(spec["gt_manifest_rows"])
    first_ts, first_R, first_t = gt_manifest_rows[0]
    pred_abs_rows: List[Tuple[float, np.ndarray, np.ndarray]] = [
        (float(first_ts), np.asarray(first_R, dtype=np.float64), np.asarray(first_t, dtype=np.float64))
    ]
    current_R = np.asarray(first_R, dtype=np.float64)
    current_t = np.asarray(first_t, dtype=np.float64)
    for row in adj_rows:
        pred_R = np.asarray(row["pred_R_BA"], dtype=np.float64)
        pred_t = np.asarray(row["pred_t_BA_B"], dtype=np.float64)
        next_R = current_R @ pred_R.T
        next_t = current_t - next_R @ pred_t
        pred_abs_rows.append((float(row["timestamp_b"]), next_R, next_t))
        current_R = next_R
        current_t = next_t
    return pred_abs_rows


def _evaluate_sequence_from_abs(
    spec: Mapping[str, Any],
    adj_rows: Sequence[Mapping[str, Any]],
    abs_rows: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    *,
    output_dir: Path,
    method_name: str,
    split: str,
    seq_id: str,
) -> Dict[str, Any]:
    gt_manifest_rows = list(spec["gt_manifest_rows"][: len(abs_rows)])
    pred_tum = output_dir / split / method_name / seq_id / "pred_tum.txt"
    gt_tum = output_dir / split / method_name / seq_id / "gt_tum.txt"
    pred_tum.parent.mkdir(parents=True, exist_ok=True)
    _write_tum(pred_tum, abs_rows)
    _write_tum(gt_tum, gt_manifest_rows)

    trajectory_eval = {
        mode: _evaluate_pose_errors(
            gt_manifest_rows,
            abs_rows,
            alignment=mode,
            gt_pose_count_full=int(spec["full_gt_pose_count"]),
        )
        for mode in ("none", "se3", "sim3")
    }
    pair_metric_rows = [
        _relative_metric(
            np.asarray(row["pred_R_BA"], dtype=np.float64),
            np.asarray(row["pred_t_BA_B"], dtype=np.float64),
            np.asarray(row["gt_R_BA"], dtype=np.float64),
            np.asarray(row["gt_t_BA_B"], dtype=np.float64),
        )
        for row in adj_rows
    ]
    derived_rows: List[Dict[str, Any]] = []
    gt_by_ts = {round(float(ts), 6): (R, t) for ts, R, t in gt_manifest_rows}
    for idx in range(len(abs_rows) - 1):
        ts_a, R_a, t_a = abs_rows[idx]
        ts_b, R_b, t_b = abs_rows[idx + 1]
        gt_a = gt_by_ts[round(float(ts_a), 6)]
        gt_b = gt_by_ts[round(float(ts_b), 6)]
        pred_rel_R, pred_rel_t = _relative_from_absolute(R_a, t_a, R_b, t_b)
        gt_rel_R, gt_rel_t = _relative_from_absolute(gt_a[0], gt_a[1], gt_b[0], gt_b[1])
        derived_rows.append(_relative_metric(pred_rel_R, pred_rel_t, gt_rel_R, gt_rel_t))

    def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        pred = float(sum(float(np.linalg.norm(np.asarray(row["pred_t_BA_B"], dtype=np.float64))) for row in adj_rows))
        gt = float(sum(float(np.linalg.norm(np.asarray(row["gt_t_BA_B"], dtype=np.float64))) for row in adj_rows))
        signed = [float(row["signed_tdir_deg"]) for row in rows if row.get("signed_tdir_deg") is not None]
        unsigned = [float(row["unsigned_tdir_deg"]) for row in rows if row.get("unsigned_tdir_deg") is not None]
        tmag_ratio = [float(row["tmag_ratio"]) for row in rows if row.get("tmag_ratio") is not None]
        return {
            "count": int(len(rows)),
            "rot_mean_deg": _mean(row["rot_deg"] for row in rows if row.get("rot_deg") is not None),
            "rot_median_deg": float(np.median(np.asarray([row["rot_deg"] for row in rows], dtype=np.float64))) if rows else None,
            "signed_tdir_mean_deg": _mean(signed),
            "unsigned_tdir_mean_deg": _mean(unsigned),
            "anti_parallel_rate": _mean(float(bool(row["anti_parallel_flag"])) for row in rows),
            "tmag_median_ratio": float(np.median(np.asarray(tmag_ratio, dtype=np.float64))) if tmag_ratio else None,
            "tmag_mean_ratio": _mean(tmag_ratio),
            "path_ratio": float(pred / max(gt, TMAG_EPS)),
            "pred_path_length": pred,
            "gt_path_length": gt,
        }

    return {
        "available": True,
        "sequence": seq_id,
        "split": split,
        "method_name": method_name,
        "image_count": int(spec["frame_count"]),
        "adjacent_pair_count": int(spec["adjacent_pair_count"]),
        "successful_pair_count": int(len(adj_rows)),
        "skipped_pair_count": int(max(int(spec["adjacent_pair_count"]) - len(adj_rows), 0)),
        "pair_coverage": float(len(adj_rows) / max(int(spec["adjacent_pair_count"]), 1)),
        "pose_coverage": float(trajectory_eval["none"]["num_matched_poses"] / max(int(spec["manifest_pose_count"]), 1)),
        "manifest_pose_count": int(spec["manifest_pose_count"]),
        "full_gt_pose_count": int(spec["full_gt_pose_count"]),
        "nan_inf_count": int(sum(1 for row in adj_rows if bool(row.get("nan_or_inf")))),
        "trajectory_eval": trajectory_eval,
        "pair_inference_relative_metrics": _summary(pair_metric_rows),
        "trajectory_derived_relative_metrics": _summary(derived_rows),
        "pred_tum": str(pred_tum),
        "gt_tum": str(gt_tum),
    }


def _build_absolute_rotations(spec: Mapping[str, Any], adj_rows: Sequence[Mapping[str, Any]]) -> List[np.ndarray]:
    gt_manifest_rows = list(spec["gt_manifest_rows"])
    _, first_R, _ = gt_manifest_rows[0]
    out = [np.asarray(first_R, dtype=np.float64)]
    current_R = np.asarray(first_R, dtype=np.float64)
    for row in adj_rows:
        pred_R = np.asarray(row["pred_R_BA"], dtype=np.float64)
        current_R = current_R @ pred_R.T
        out.append(current_R.copy())
    return out


def _solve_positions_from_constraints(
    spec: Mapping[str, Any],
    abs_rotations: Sequence[np.ndarray],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    k_decay: float,
) -> List[np.ndarray]:
    frames = list(spec["frames"])
    ts_to_idx = {round(float(frame["timestamp"]), 6): idx for idx, frame in enumerate(frames)}
    gt_manifest_rows = list(spec["gt_manifest_rows"])
    anchor_t = np.asarray(gt_manifest_rows[0][2], dtype=np.float64)
    n = len(frames)
    if n <= 1:
        return [anchor_t.copy()]
    coeff_rows: List[np.ndarray] = []
    rhs_rows: List[np.ndarray] = []
    for row in pair_rows:
        ts_a = round(float(row["timestamp_a"]), 6)
        ts_b = round(float(row["timestamp_b"]), 6)
        if ts_a not in ts_to_idx or ts_b not in ts_to_idx:
            continue
        i = ts_to_idx[ts_a]
        j = ts_to_idx[ts_b]
        if i >= j:
            continue
        weight = float(k_decay ** max(int(row["k"]) - 1, 0))
        rel_t = np.asarray(row["pred_t_BA_B"], dtype=np.float64)
        d_ij = -abs_rotations[j] @ rel_t
        coeff = np.zeros((n - 1,), dtype=np.float64)
        if i > 0:
            coeff[i - 1] -= 1.0
        else:
            d_ij = d_ij + anchor_t
        if j > 0:
            coeff[j - 1] += 1.0
        else:
            d_ij = d_ij - anchor_t
        coeff_rows.append(coeff * math.sqrt(weight))
        rhs_rows.append(d_ij * math.sqrt(weight))
    if not coeff_rows:
        return [np.asarray(t, dtype=np.float64) for _, _, t in gt_manifest_rows[:n]]
    A = np.asarray(coeff_rows, dtype=np.float64)
    B = np.asarray(rhs_rows, dtype=np.float64)
    sol = np.zeros((n, 3), dtype=np.float64)
    sol[0] = anchor_t
    for dim in range(3):
        x, *_ = np.linalg.lstsq(A, B[:, dim], rcond=None)
        sol[1:, dim] = x
    return [sol[i].copy() for i in range(n)]


def _run_method_on_split(
    split: str,
    method_name: str,
    params: Mapping[str, Any],
    specs: Mapping[str, Dict[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    per_sequence: Dict[str, Any] = {}
    for seq_id, spec in specs.items():
        adj_rows = _sequence_adjacent_rows(prediction_rows, split, seq_id)
        if method_name == "direct_composition":
            adj_mod = adj_rows
            abs_rows = _compose_abs_from_adj(spec, adj_mod)
        elif method_name == "scale_only_smoothing":
            adj_mod = _apply_scale_only(adj_rows, params)
            abs_rows = _compose_abs_from_adj(spec, adj_mod)
        elif method_name == "rotation_smoothing":
            adj_mod = _apply_rotation_smoothing(adj_rows, params)
            abs_rows = _compose_abs_from_adj(spec, adj_mod)
        elif method_name == "tdir_outlier_suppression":
            adj_mod = _apply_tdir_suppression(adj_rows, params)
            abs_rows = _compose_abs_from_adj(spec, adj_mod)
        elif method_name == "local_window_pose_fusion":
            adj_mod = adj_rows
            if params.get("use_scale_pre_smoothing") == "ema":
                adj_mod = _apply_scale_only(adj_mod, {"smooth_mode": "ema", "ema_alpha": float(params["scale_ema_alpha"])})
            elif params.get("use_scale_pre_smoothing") == "median":
                adj_mod = _apply_scale_only(adj_mod, {"smooth_mode": "median", "window": int(params["scale_window"])})
            if params.get("use_rotation_pre_smoothing") == "ema":
                adj_mod = _apply_rotation_smoothing(adj_mod, {"smooth_mode": "ema", "strength": float(params["rotation_strength"])})
            elif params.get("use_rotation_pre_smoothing") == "window":
                adj_mod = _apply_rotation_smoothing(
                    adj_mod,
                    {"smooth_mode": "window", "window": int(params["rotation_window"]), "strength": float(params["rotation_strength"])},
                )
            if bool(params.get("use_tdir_pre_suppression")):
                adj_mod = _apply_tdir_suppression(
                    adj_mod,
                    {
                        "window": int(params["tdir_window"]),
                        "angular_threshold_deg": float(params["tdir_threshold_deg"]),
                        "replacement_strength": float(params["tdir_replacement_strength"]),
                    },
                )
            abs_rot = _build_absolute_rotations(spec, adj_mod)
            all_rows = _sequence_all_rows(prediction_rows, split, seq_id, k_max=int(params["k_max"]))
            row_by_edge = {(round(float(r["timestamp_a"]), 6), round(float(r["timestamp_b"]), 6), int(r["k"])): dict(r) for r in all_rows}
            for row in adj_mod:
                key = (round(float(row["timestamp_a"]), 6), round(float(row["timestamp_b"]), 6), int(row["k"]))
                row_by_edge[key] = dict(row)
            fused_pair_rows = list(row_by_edge.values())
            fused_pair_rows.sort(key=lambda row: (row["timestamp_a"], row["timestamp_b"], row["pair_index"]))
            positions = _solve_positions_from_constraints(spec, abs_rot, fused_pair_rows, k_decay=float(params["k_decay"]))
            abs_rows = [(float(spec["frames"][i]["timestamp"]), abs_rot[i], positions[i]) for i in range(len(spec["frames"]))]
        else:
            raise ValueError(f"Unsupported method: {method_name}")
        per_sequence[seq_id] = _evaluate_sequence_from_abs(
            spec,
            adj_mod,
            abs_rows,
            output_dir=output_dir,
            method_name=method_name,
            split=split,
            seq_id=seq_id,
        )

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
        "method_name": method_name,
        "selected_params": dict(params),
        "split": split,
        "training_executed": False,
        "pair_model_modified": False,
        "sequence_count": len(per_sequence),
        "pair_total": int(pair_total),
        "pair_success": int(pair_success),
        "pair_coverage": float(pair_success / max(pair_total, 1)),
        "pose_count": int(pose_total),
        "pose_success": int(pose_success),
        "coverage": float(pose_success / max(pose_total, 1)),
        "failure_count": int(sum(1 for payload in per_sequence.values() if payload["successful_pair_count"] < payload["adjacent_pair_count"])),
        "nan_inf_count": int(nan_inf_total),
        "ate_none": _ate("none"),
        "ate_se3": _ate("se3"),
        "ate_sim3": _ate("sim3"),
        "trajectory_path_ratio": float(pred_path_total / max(gt_path_total, TMAG_EPS)),
        "path_length_pred": pred_path_total,
        "path_length_gt": gt_path_total,
        "per_sequence": per_sequence,
    }


def _best_and_worst_sequences(selected: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seq_id, payload in selected["per_sequence"].items():
        base = baseline["per_sequence"][seq_id]
        row = {
            "sequence": seq_id,
            "selected_ate_se3": float(payload["trajectory_eval"]["se3"]["rmse"]),
            "baseline_ate_se3": float(base["trajectory_eval"]["se3"]["rmse"]),
            "delta_ate_se3": float(payload["trajectory_eval"]["se3"]["rmse"] - base["trajectory_eval"]["se3"]["rmse"]),
            "selected_ate_sim3": float(payload["trajectory_eval"]["sim3"]["rmse"]),
            "baseline_ate_sim3": float(base["trajectory_eval"]["sim3"]["rmse"]),
            "delta_ate_sim3": float(payload["trajectory_eval"]["sim3"]["rmse"] - base["trajectory_eval"]["sim3"]["rmse"]),
            "selected_path_ratio": float(payload["trajectory_eval"]["none"]["trajectory_path_ratio"]),
            "baseline_path_ratio": float(base["trajectory_eval"]["none"]["trajectory_path_ratio"]),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["delta_ate_se3"])
    degraded = [row for row in rows if row["delta_ate_se3"] > 0.0]
    return {
        "all_sequences": rows,
        "best_improved_sequences": rows[: min(3, len(rows))],
        "worst_degraded_sequences": degraded[: min(3, len(degraded))],
        "per_sequence_degradation_found": bool(degraded),
    }


def _build_method_grid(cfg: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    grid: List[Tuple[str, Dict[str, Any]]] = [("direct_composition", {})]
    for key, method_name in (
        ("scale_only", "scale_only_smoothing"),
        ("rotation_smoothing", "rotation_smoothing"),
        ("tdir_outlier_suppression", "tdir_outlier_suppression"),
        ("local_window_pose_fusion", "local_window_pose_fusion"),
    ):
        section = cfg["search"][key]
        if bool(section.get("enabled", True)):
            for params in section["methods"]:
                grid.append((method_name, dict(params)))
    return grid


def _direct_reproduction_ok(metrics: Mapping[str, Any], ref: Mapping[str, Any], tol_cfg: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    checks = {
        "ate_none_rmse": abs(float(metrics["ate_none"]["rmse"]) - float(ref["ate_none"]["rmse"])),
        "ate_se3_rmse": abs(float(metrics["ate_se3"]["rmse"]) - float(ref["ate_se3"]["rmse"])),
        "ate_sim3_rmse": abs(float(metrics["ate_sim3"]["rmse"]) - float(ref["ate_sim3"]["rmse"])),
        "trajectory_path_ratio": abs(float(metrics["trajectory_path_ratio"]) - float(ref["trajectory_path_ratio"])),
    }
    ok = all(checks[k] <= float(tol_cfg[k]) for k in checks)
    return ok, checks


def _write_report(
    cfg: Mapping[str, Any],
    *,
    precheck: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    direct_val: Mapping[str, Any],
    direct_test: Mapping[str, Any],
    selected_val: Mapping[str, Any],
    selected_test: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]],
    train360e_test: Mapping[str, Any],
    seq360b_test: Mapping[str, Any],
    per_sequence_analysis: Mapping[str, Any],
    classification: str,
    next_task: str,
) -> None:
    report_lines = [
        "# ODOM360A lightweight trajectory fusion",
        "",
        "## 1. Executive summary",
        "- fusion executed: `true`",
        "- training executed: `false`",
        "- pair model modified: `false`",
        f"- selected fusion method: `{selected_test['method_name']}`",
        f"- selected params: `{selected_test['selected_params']}`",
        f"- classification: `{classification}`",
        f"- final recommendation: `{next_task}`",
        "",
        "## 2. Motivation",
        "- FINAL360I pair-level behavior is stable but direct trajectory composition still drifts.",
        "- SEQ360B improves scale/path and SE3 ATE, but Sim3 shape barely changes.",
        "- FINAL360J/K/L show that pair-level small fixes are not the right next lever.",
        "",
        "## 3. Data sources",
        f"- checkpoint: `{cfg['inputs']['checkpoint']}`",
        f"- manifests: `{cfg['inputs']['val_manifest']}`, `{cfg['inputs']['test_manifest']}`",
        f"- pair prediction cache: `{source_manifest['prediction_cache']}`",
        f"- reports used: `{cfg['inputs']['train360e_report']}`, `{cfg['inputs']['seq360b_report']}`, `{cfg['inputs']['results_table']}`, `{cfg['inputs']['results_narrative']}`",
        "",
        "## 4. Baseline reproduction",
        f"- direct val ATE none / SE3 / Sim3: `{direct_val['ate_none']['rmse']}` / `{direct_val['ate_se3']['rmse']}` / `{direct_val['ate_sim3']['rmse']}`",
        f"- direct test ATE none / SE3 / Sim3: `{direct_test['ate_none']['rmse']}` / `{direct_test['ate_se3']['rmse']}` / `{direct_test['ate_sim3']['rmse']}`",
        f"- direct test path ratio: `{direct_test['trajectory_path_ratio']}`",
        f"- TRAIN360E reference test ATE none / SE3 / Sim3: `{train360e_test['ate_none']['rmse']}` / `{train360e_test['ate_se3']['rmse']}` / `{train360e_test['ate_sim3']['rmse']}`",
        "",
        "## 5. Fusion methods",
        "- scale-only smoothing: adjacent log-tmag EMA or median smoothing.",
        "- rotation smoothing: SO(3) EMA/window smoothing on adjacent relative rotations.",
        "- tdir outlier suppression: local angular-threshold replacement toward neighborhood mean direction.",
        "- local window pose fusion: adjacent-smoothed rotations plus k-step translation constraints solved as a lightweight anchored least-squares position graph.",
        "",
        "## 6. Val selection",
        "- score = ATE_SE3 + 2 * ATE_Sim3 + 30 * |log(path_ratio)|",
        "- test not used for selection.",
        f"- selected method on val: `{selected_val['method_name']}` with `{selected_val['selected_params']}`",
        "",
        "## 7. Test results",
        f"- TRAIN360E direct: none=`{train360e_test['ate_none']['rmse']}`, se3=`{train360e_test['ate_se3']['rmse']}`, sim3=`{train360e_test['ate_sim3']['rmse']}`, path_ratio=`{train360e_test['trajectory_path_ratio']}`",
        f"- SEQ360B: none=`{seq360b_test['ate_none']['rmse']}`, se3=`{seq360b_test['ate_se3']['rmse']}`, sim3=`{seq360b_test['ate_sim3']['rmse']}`, path_ratio=`{seq360b_test['trajectory_path_ratio']}`",
        f"- ODOM360A selected: none=`{selected_test['ate_none']['rmse']}`, se3=`{selected_test['ate_se3']['rmse']}`, sim3=`{selected_test['ate_sim3']['rmse']}`, path_ratio=`{selected_test['trajectory_path_ratio']}`",
        "",
        "## 8. Per-sequence results",
        f"- best improved sequences: `{per_sequence_analysis['best_improved_sequences']}`",
        f"- worst degraded sequences: `{per_sequence_analysis['worst_degraded_sequences']}`",
        "",
        "## 9. Shape vs scale interpretation",
        f"- Sim3 shape improved: `{float(selected_test['ate_sim3']['rmse']) < 27.567211313835486}`",
        f"- path ratio moved closer to 1.0 than SEQ360B: `{abs(float(selected_test['trajectory_path_ratio']) - 1.0) < abs(1.3502369615185652 - 1.0)}`",
        "- If SE3/path improve without Sim3 improvement, the gain should be interpreted as mostly scale/path correction rather than true shape repair.",
        "",
        "## 10. Limitations",
        "- post-processing only",
        "- no image-level geometric verification",
        "- no mature VO claim",
        "- no real-time guarantee measured here",
        "- no loop closure",
        "",
        "## 11. Next recommendation",
        f"- `{next_task}`",
        "",
        "## 12. Compliance checklist",
        "- `training_executed = false`",
        "- `pair_model_modified = false`",
        "- `test_used_for_selection = false`",
        "- `explicit_matching_used = false`",
        "- `image_ransac_used = false`",
        "- `pnp_used = false`",
        "- `ba_used = false`",
        "- `hkust_teacher_used = false`",
        "- `orbslam_teacher_used = false`",
        "- `checkpoints_committed = false`",
        "- `raw_data_committed = false`",
        "- `large_prediction_dump_committed = false`",
        "- `metrics_modified_existing = false`",
    ]
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# ODOM360A vs TRAIN360E and SEQ360B",
        "",
        f"- selected fusion method: `{selected_test['method_name']}`",
        f"- selected params: `{selected_test['selected_params']}`",
        f"- compared to TRAIN360E: `{_compare_to_baseline(selected_test, {'ate_se3': 118.6037794846689, 'ate_sim3': 27.564661865900444, 'path_ratio': 1.756343083453392})}`",
        f"- compared to SEQ360B: `{_compare_to_baseline(selected_test, {'ate_se3': 75.94691348103409, 'ate_sim3': 27.567211313835486, 'path_ratio': 1.3502369615185652})}`",
        f"- classification: `{classification}`",
        f"- next recommendation: `{next_task}`",
    ]
    (REPO_ROOT / cfg["outputs"]["summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    precheck = _precheck(cfg)
    if precheck["blockers"]:
        blocker = {
            "task_name": cfg["task_name"],
            "fusion_executed": False,
            "training_executed": False,
            "blockers": precheck["blockers"],
            "precheck": precheck,
        }
        for key in ("metrics_val_path", "metrics_test_path", "model_selection_table_path", "per_sequence_metrics_path", "ablation_path", "metric_source_manifest_path"):
            _write_json(REPO_ROOT / cfg["outputs"][key], blocker)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("# ODOM360A blocked\n\n" + "\n".join(f"- {b}" for b in precheck["blockers"]) + "\n", encoding="utf-8")
        return 1

    result_root = REPO_ROOT / cfg["outputs"]["result_root"]
    result_root.mkdir(parents=True, exist_ok=True)
    prediction_by_split, cache_manifest = _load_or_export_predictions(cfg)

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
    val_specs, val_blockers = _build_sequence_specs(ds_val, "val")
    test_specs, test_blockers = _build_sequence_specs(ds_test, "test")
    if val_blockers or test_blockers:
        blockers = list(val_blockers) + list(test_blockers)
        payload = {"task_name": cfg["task_name"], "fusion_executed": False, "blockers": blockers}
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], payload)
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], payload)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("# ODOM360A blocked\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
        return 1

    direct_val = _run_method_on_split("val", "direct_composition", {}, val_specs, prediction_by_split["val"], result_root)
    direct_test = _run_method_on_split("test", "direct_composition", {}, test_specs, prediction_by_split["test"], result_root)

    train360e_test = _read_json(REPO_ROOT / cfg["inputs"]["train360e_metrics_test"])
    seq360b_test = _read_json(REPO_ROOT / cfg["inputs"]["seq360b_trajectory_metrics_test"])
    ok, reproduction_delta = _direct_reproduction_ok(direct_test, train360e_test, cfg["evaluation"]["direct_reproduction_tolerance"])
    if not ok:
        blocker = {
            "task_name": cfg["task_name"],
            "fusion_executed": False,
            "training_executed": False,
            "blocker": "direct_composition_reproduction_mismatch",
            "reproduction_delta": reproduction_delta,
            "direct_test": direct_test,
            "train360e_reference": train360e_test,
        }
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_test_path"], blocker)
        _write_json(REPO_ROOT / cfg["outputs"]["metrics_val_path"], blocker)
        (REPO_ROOT / cfg["outputs"]["report_path"]).write_text(
            "# ODOM360A blocked\n\n- direct composition reproduction mismatch\n- deltas: "
            + json.dumps(reproduction_delta, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        return 1

    grid = _build_method_grid(cfg)
    ablation_rows: List[Dict[str, Any]] = []
    best_val: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    for method_name, params in grid:
        val_metrics = direct_val if method_name == "direct_composition" else _run_method_on_split("val", method_name, params, val_specs, prediction_by_split["val"], result_root)
        score = _score(val_metrics, cfg)
        row = {
            "method_name": method_name,
            "params": dict(params),
            "split": "val",
            "score": score,
            "ate_none": val_metrics["ate_none"]["rmse"],
            "ate_se3": val_metrics["ate_se3"]["rmse"],
            "ate_sim3": val_metrics["ate_sim3"]["rmse"],
            "trajectory_path_ratio": val_metrics["trajectory_path_ratio"],
            "coverage": val_metrics["coverage"],
            "test_used_for_selection": False,
        }
        ablation_rows.append(row)
        if score < best_score:
            best_score = score
            best_val = val_metrics
    if best_val is None:
        raise RuntimeError("No ODOM360A method was evaluated on val.")

    selected_method = str(best_val["method_name"])
    selected_params = dict(best_val["selected_params"])
    selected_test = (
        direct_test
        if selected_method == "direct_composition"
        else _run_method_on_split("test", selected_method, selected_params, test_specs, prediction_by_split["test"], result_root)
    )
    best_val["selection_score"] = best_score
    selected_test["selection_score"] = best_score
    selected_test["test_used_for_selection"] = False
    best_val["test_used_for_selection"] = False

    per_sequence_analysis = _best_and_worst_sequences(selected_test, direct_test)
    classification = _classify(selected_test)
    if classification in {"strong_success", "balanced_success", "shape_improvement_success"}:
        next_task = "proceed_to_ODOM360B_local_pose_graph_with_kstep_constraints"
    elif classification in {"primary_success", "partial"}:
        next_task = "proceed_to_ODOM360B_local_pose_graph_with_kstep_constraints"
    elif classification == "regression":
        next_task = "keep_SEQ360B_as_best_sequence_variant"
    else:
        next_task = "keep_SEQ360B_as_best_sequence_variant"

    metrics_val_payload = {
        "task_name": cfg["task_name"],
        "fusion_executed": True,
        "training_executed": False,
        "pair_model_modified": False,
        **best_val,
    }
    metrics_test_payload = {
        "task_name": cfg["task_name"],
        "fusion_executed": True,
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
            "selected_method": selected_method,
            "selected_params": selected_params,
            "selected_val_score": best_score,
            "test_used_for_selection": False,
        },
    )
    _write_json(REPO_ROOT / cfg["outputs"]["per_sequence_metrics_path"], per_sequence_analysis)
    _write_json(REPO_ROOT / cfg["outputs"]["ablation_path"], {"rows": ablation_rows})
    metric_source_manifest = {
        "task_name": cfg["task_name"],
        "prediction_cache": cache_manifest,
        "reproduction_delta_vs_train360e_test": reproduction_delta,
        "checkpoint": cfg["inputs"]["checkpoint"],
        "train360e_metrics_test": cfg["inputs"]["train360e_metrics_test"],
        "seq360b_trajectory_metrics_test": cfg["inputs"]["seq360b_trajectory_metrics_test"],
        "test_used_for_selection": False,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["metric_source_manifest_path"], metric_source_manifest)
    _write_report(
        cfg,
        precheck=precheck,
        source_manifest=metric_source_manifest,
        direct_val=direct_val,
        direct_test=direct_test,
        selected_val=best_val,
        selected_test=selected_test,
        ablation_rows=ablation_rows,
        train360e_test=train360e_test,
        seq360b_test=seq360b_test,
        per_sequence_analysis=per_sequence_analysis,
        classification=classification,
        next_task=next_task,
    )
    print(
        json.dumps(
            {
                "task_name": cfg["task_name"],
                "fusion_executed": True,
                "training_executed": False,
                "selected_method": selected_method,
                "selected_params": selected_params,
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
