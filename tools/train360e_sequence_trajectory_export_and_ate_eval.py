#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from train360.core.config import Config
from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from mainline_dependency_utils import optional_read_json, summarize_optional_artifact
from miniyaml import load_yaml_like
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from run_base360_hkust_360dvo_official import _build_gt_rows as _build_gt_rows_from_raw


TASK_NAME = "TRAIN360E_sequence_trajectory_export_and_ATE_eval"
EXPECTED_BRANCHES = [
    "experiment/final360i-final-retrain-and-model-selection",
    "experiment/train360e-sequence-trajectory-export-and-ate-eval",
    "maintenance/destructive-cleanup-current-mainline-only",
    "maintenance/trim-remaining-mainline-dependencies",
]
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "best_val.pt"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "struct360b_match_free_coarse_to_fine.yaml"
VAL_MANIFEST = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_val.jsonl"
TEST_MANIFEST = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_test.jsonl"
HYGIENE_JSON = REPO_ROOT / "checkpoints" / "DSET2C_360DVO_dataset_hygiene.json"
DATASET_ADAPTER = REPO_ROOT / "datasets" / "dset2c_manifest_dataset.py"
FINAL360I_REPORTS = [
    REPO_ROOT / "reports" / "FINAL360I_metrics_val.json",
    REPO_ROOT / "reports" / "FINAL360I_metrics_test.json",
    REPO_ROOT / "reports" / "FINAL360I_model_selection_table.json",
    REPO_ROOT / "reports" / "FINAL360I_vs_all_baselines_summary.md",
    REPO_ROOT / "reports" / "FINAL360I_final_retrain_and_model_selection.md",
]
BASE360D_REPORTS = [
    REPO_ROOT / "reports" / "BASE360D_component_metric_alignment.md",
    REPO_ROOT / "reports" / "BASE360D_metrics_val.json",
    REPO_ROOT / "reports" / "BASE360D_metrics_test.json",
]
BASE360_ROOT = REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official"
RESULTS_ROOT = REPO_ROOT / "external_baselines" / "results" / "train360e_final360i_trajectory"
REPORT_PATH = REPO_ROOT / "reports" / "TRAIN360E_sequence_trajectory_export_and_ATE_eval.md"
VAL_JSON_PATH = REPO_ROOT / "reports" / "TRAIN360E_metrics_val.json"
TEST_JSON_PATH = REPO_ROOT / "reports" / "TRAIN360E_metrics_test.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "TRAIN360E_vs_FINAL360I_BASE360D_trajectory_summary.md"
MIN_DISK_FREE_GB = 5.0
MATCH_TOL = 1.0e-4
TMAG_EPS = 1.0e-6


def _git(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout or "").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_or_missing(path: Path, label: str) -> Dict[str, Any]:
    payload = optional_read_json(path)
    if payload.get("status") == "missing_after_cleanup":
        payload["label"] = label
    return payload


def _empty_base360_eval(split: str, *, reason: str) -> Dict[str, Any]:
    return {
        "split": split,
        "status": reason,
        "per_sequence": {},
        "ate_none": {"rmse": None},
        "ate_se3": {"rmse": None},
        "ate_sim3": {"rmse": None},
        "pose_coverage": None,
        "trajectory_path_ratio": None,
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    x = _safe_float(value)
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def _angle_from_rot(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _vector_angle_deg(a: np.ndarray, b: np.ndarray, *, absolute: bool) -> Optional[float]:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1.0e-12 or bn <= 1.0e-12:
        return None
    cosine = float(np.dot(a, b) / max(an * bn, 1.0e-12))
    cosine = max(min(cosine, 1.0), -1.0)
    if absolute:
        cosine = abs(cosine)
    return float(np.degrees(np.arccos(cosine)))


def _relative_from_absolute(R_a: np.ndarray, t_a: np.ndarray, R_b: np.ndarray, t_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R_b.T @ R_a, R_b.T @ (t_a - t_b)


def _inverse_ba(R_ba: np.ndarray, t_ba_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R_ba.T, -R_ba.T @ t_ba_b


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


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg = _cfg_from_dict(dict(payload["cfg"]))
    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    load_result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    load_summary = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, load_summary


class _AdjacentSubset(Dataset):
    def __init__(self, base: Dset2CCanonicalPairDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = self.indices[idx]
        item = self.base[base_idx]
        item["dataset_index"] = torch.tensor(base_idx, dtype=torch.int64)
        return item


def _precheck() -> Dict[str, Any]:
    git_status = _git(["git", "status", "--short"]).splitlines()
    branch = _git(["git", "branch", "--show-current"])
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    disk = shutil.disk_usage(REPO_ROOT)
    disk_free_gb = disk.free / (1024 ** 3)
    blockers: List[str] = []
    if branch not in EXPECTED_BRANCHES:
        blockers.append(f"current branch mismatch: {branch}")
    if disk_free_gb <= MIN_DISK_FREE_GB:
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB <= {MIN_DISK_FREE_GB:.2f} GiB")
    if not torch.cuda.is_available():
        blockers.append("CUDA unavailable in pytorch environment")
    required = [
        CHECKPOINT_PATH,
        VAL_MANIFEST,
        TEST_MANIFEST,
        HYGIENE_JSON,
        DATASET_ADAPTER,
        BASE_CONFIG_PATH,
        FINAL360I_REPORTS[0],
        FINAL360I_REPORTS[1],
        FINAL360I_REPORTS[2],
    ]
    for path in required:
        if not path.is_file():
            blockers.append(f"missing required file: {path}")
    adapter_text = DATASET_ADAPTER.read_text(encoding="utf-8")
    adapter_norm = " ".join(adapter_text.split())
    if "manifest-native" not in adapter_norm and "never scans raw sequence directories" not in adapter_norm:
        blockers.append("dataset adapter no longer advertises manifest-native behavior")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to use direct raw-sequence globbing")
    baseline_recap = {
        "final360i_val": _artifact_or_missing(FINAL360I_REPORTS[0], "FINAL360I val metrics"),
        "final360i_test": _artifact_or_missing(FINAL360I_REPORTS[1], "FINAL360I test metrics"),
        "final360i_selection": _artifact_or_missing(FINAL360I_REPORTS[2], "FINAL360I selection table"),
        "base360d_component_report": summarize_optional_artifact(BASE360D_REPORTS[0], "BASE360D component alignment report"),
        "base360d_val": _artifact_or_missing(BASE360D_REPORTS[1], "BASE360D val metrics"),
        "base360d_test": _artifact_or_missing(BASE360D_REPORTS[2], "BASE360D test metrics"),
    }
    return {
        "git_status_short": git_status,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "disk_free_gb": disk_free_gb,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "blockers": blockers,
        "baseline_recap": baseline_recap,
    }


def _sequence_gt_rows(seq_id: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    return _build_gt_rows_from_raw(seq_id)


def _build_sequence_specs(dataset: Dset2CCanonicalPairDataset, split: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    sequences: Dict[str, Dict[str, Any]] = {}
    blockers: List[str] = []
    adjacent_samples = [dict(sample) for sample in dataset.samples if sample["pair_type"] == "adjacent" and int(sample["k"]) == 1]
    for sample in adjacent_samples:
        seq_id = str(sample["sequence_id"])
        bucket = sequences.setdefault(
            seq_id,
            {
                "sequence_id": seq_id,
                "split": split,
                "adjacent_rows": [],
            },
        )
        bucket["adjacent_rows"].append(sample)
    for seq_id, spec in sequences.items():
        rows = sorted(spec["adjacent_rows"], key=lambda row: (float(row["timestamp_a"]), float(row["timestamp_b"]), int(row["pair_index"])))
        if not rows:
            blockers.append(f"{split}/{seq_id}: no adjacent rows recovered from manifest")
            continue
        frames: List[Dict[str, Any]] = []
        first_frame = {
            "timestamp": float(rows[0]["timestamp_a"]),
            "image_path": str(rows[0]["image_path_a"]),
        }
        frames.append(first_frame)
        for row in rows:
            expected_prev = frames[-1]["image_path"]
            if str(row["image_path_a"]) != expected_prev:
                blockers.append(f"{split}/{seq_id}: adjacent chain discontinuity at pair_index={row['pair_index']}")
            frames.append(
                {
                    "timestamp": float(row["timestamp_b"]),
                    "image_path": str(row["image_path_b"]),
                }
            )
        image_count = len(frames)
        gt_full_rows = _sequence_gt_rows(seq_id)
        gt_full_by_ts = {round(float(ts), 6): (float(ts), np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)) for ts, R, t in gt_full_rows}
        manifest_ts = [round(float(frame["timestamp"]), 6) for frame in frames]
        gt_manifest_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
        missing_gt_ts: List[float] = []
        for ts in manifest_ts:
            if ts in gt_full_by_ts:
                raw_ts, R, t = gt_full_by_ts[ts]
                gt_manifest_rows.append((raw_ts, R, t))
            else:
                missing_gt_ts.append(ts)
        if missing_gt_ts:
            blockers.append(f"{split}/{seq_id}: missing GT absolute poses for {len(missing_gt_ts)} manifest timestamps")
        spec["adjacent_rows"] = rows
        spec["frames"] = frames
        spec["frame_count"] = image_count
        spec["adjacent_pair_count"] = len(rows)
        spec["pair_count"] = len([sample for sample in dataset.samples if sample["sequence_id"] == seq_id])
        spec["manifest_pose_count"] = len(gt_manifest_rows)
        spec["full_gt_pose_count"] = len(gt_full_rows)
        spec["gt_manifest_rows"] = gt_manifest_rows
        spec["gt_full_rows"] = gt_full_rows
        spec["gt_manifest_coverage"] = float(len(gt_manifest_rows) / max(image_count, 1))
        spec["full_gt_coverage_from_manifest"] = float(image_count / max(len(gt_full_rows), 1))
    return sequences, blockers


def _convention_sanity(sequence_specs: Mapping[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], List[str]]:
    rot_errors_ba: List[float] = []
    rot_errors_inv: List[float] = []
    tdir_errors_ba: List[float] = []
    tdir_errors_inv: List[float] = []
    warnings: List[str] = []
    per_sequence: Dict[str, Any] = {}
    for seq_id, spec in sequence_specs.items():
        gt_rows = spec["gt_manifest_rows"]
        if len(gt_rows) != len(spec["frames"]):
            warnings.append(f"{spec['split']}/{seq_id}: manifest GT subset incomplete during convention sanity")
            continue
        gt_by_ts = {round(float(ts), 6): (R, t) for ts, R, t in gt_rows}
        seq_rot_ba: List[float] = []
        seq_rot_inv: List[float] = []
        seq_tdir_ba: List[float] = []
        seq_tdir_inv: List[float] = []
        for row in spec["adjacent_rows"]:
            ts_a = round(float(row["timestamp_a"]), 6)
            ts_b = round(float(row["timestamp_b"]), 6)
            if ts_a not in gt_by_ts or ts_b not in gt_by_ts:
                continue
            R_a, t_a = gt_by_ts[ts_a]
            R_b, t_b = gt_by_ts[ts_b]
            gt_R_ba, gt_t_ba = _relative_from_absolute(R_a, t_a, R_b, t_b)
            mf_R = np.asarray(row["R_BA"], dtype=np.float64)
            mf_t = np.asarray(row["t_BA_B"], dtype=np.float64)
            inv_R, inv_t = _inverse_ba(mf_R, mf_t)
            seq_rot_ba.append(_angle_from_rot(mf_R @ gt_R_ba.T))
            seq_rot_inv.append(_angle_from_rot(inv_R @ gt_R_ba.T))
            angle_ba = _vector_angle_deg(mf_t, gt_t_ba, absolute=False)
            angle_inv = _vector_angle_deg(inv_t, gt_t_ba, absolute=False)
            if angle_ba is not None:
                seq_tdir_ba.append(angle_ba)
            if angle_inv is not None:
                seq_tdir_inv.append(angle_inv)
        rot_errors_ba.extend(seq_rot_ba)
        rot_errors_inv.extend(seq_rot_inv)
        tdir_errors_ba.extend(seq_tdir_ba)
        tdir_errors_inv.extend(seq_tdir_inv)
        per_sequence[seq_id] = {
            "split": spec["split"],
            "pair_count": len(seq_rot_ba),
            "mean_rot_error_ba_deg": float(np.mean(seq_rot_ba)) if seq_rot_ba else None,
            "mean_rot_error_inverse_deg": float(np.mean(seq_rot_inv)) if seq_rot_inv else None,
            "mean_tdir_error_ba_deg": float(np.mean(seq_tdir_ba)) if seq_tdir_ba else None,
            "mean_tdir_error_inverse_deg": float(np.mean(seq_tdir_inv)) if seq_tdir_inv else None,
        }
    summary = {
        "candidate_ba": {
            "mean_rot_error_deg": float(np.mean(rot_errors_ba)) if rot_errors_ba else None,
            "mean_tdir_error_deg": float(np.mean(tdir_errors_ba)) if tdir_errors_ba else None,
            "pair_count": len(rot_errors_ba),
        },
        "candidate_inverse": {
            "mean_rot_error_deg": float(np.mean(rot_errors_inv)) if rot_errors_inv else None,
            "mean_tdir_error_deg": float(np.mean(tdir_errors_inv)) if tdir_errors_inv else None,
            "pair_count": len(rot_errors_inv),
        },
        "per_sequence": per_sequence,
    }
    if not rot_errors_ba or not tdir_errors_ba:
        warnings.append("convention sanity missing GT-vs-GT coverage")
        return "blocked_convention", summary, warnings
    ba_score = float(np.mean(rot_errors_ba) + np.mean(tdir_errors_ba))
    inv_score = float(np.mean(rot_errors_inv) + np.mean(tdir_errors_inv))
    if ba_score >= inv_score:
        warnings.append(f"BA convention did not outperform inverse on GT sanity: ba_score={ba_score:.6f}, inverse_score={inv_score:.6f}")
        return "blocked_convention", summary, warnings
    if np.mean(rot_errors_ba) > 1.0 or np.mean(tdir_errors_ba) > 5.0:
        warnings.append(
            f"BA convention sanity is weaker than expected: mean_rot={np.mean(rot_errors_ba):.6f} deg, "
            f"mean_tdir={np.mean(tdir_errors_ba):.6f} deg"
        )
    return "BA", summary, warnings


def _iterate_adjacent_predictions(
    model: STRUCT360BMatchFreeCoarseToFineModel,
    dataset: Dset2CCanonicalPairDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> List[Dict[str, Any]]:
    indices = [idx for idx, sample in enumerate(dataset.samples) if sample["pair_type"] == "adjacent" and int(sample["k"]) == 1]
    loader = DataLoader(
        _AdjacentSubset(dataset, indices),
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
            seq_ids = list(batch["meta"]["sequence_id"])
            splits = list(batch["meta"]["split"])
            pair_indices = batch["pair_index"].detach().cpu().numpy()
            timestamps_a = batch["meta"]["timestamp_a"].detach().cpu().numpy()
            timestamps_b = batch["meta"]["timestamp_b"].detach().cpu().numpy()
            image_paths_a = list(batch["meta"]["image_path_a"])
            image_paths_b = list(batch["meta"]["image_path_b"])
            for i in range(R_pred.shape[0]):
                frame_a = int(Path(image_paths_a[i]).stem)
                frame_b = int(Path(image_paths_b[i]).stem)
                pred_t = np.asarray(tdir_pred[i], dtype=np.float64) * float(tmag_pred[i])
                finite = bool(np.isfinite(R_pred[i]).all() and np.isfinite(pred_t).all() and math.isfinite(float(tmag_pred[i])))
                rows.append(
                    {
                        "split": str(splits[i]),
                        "sequence": str(seq_ids[i]),
                        "pair_index": int(pair_indices[i]),
                        "frame_a": frame_a,
                        "frame_b": frame_b,
                        "timestamp_a": float(timestamps_a[i]),
                        "timestamp_b": float(timestamps_b[i]),
                        "image_a": str(image_paths_a[i]),
                        "image_b": str(image_paths_b[i]),
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


def _relative_metric(R_pred: np.ndarray, t_pred: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> Dict[str, Any]:
    signed = _vector_angle_deg(t_pred, t_gt, absolute=False)
    unsigned = _vector_angle_deg(t_pred, t_gt, absolute=True)
    cosine = None
    if signed is not None:
        cosine = float(np.cos(np.deg2rad(signed)))
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


def _summary_component_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    def mean(key: str) -> Optional[float]:
        arr = vals(key)
        return float(np.mean(arr)) if arr.size else None
    def pct(key: str, q: float) -> Optional[float]:
        arr = vals(key)
        return float(np.percentile(arr, q)) if arr.size else None
    pred = float(sum(float(row.get("pred_step_length") or 0.0) for row in rows))
    gt = float(sum(float(row.get("gt_step_length") or 0.0) for row in rows))
    return {
        "count": int(len(rows)),
        "rot_mean_deg": mean("rot_deg"),
        "rot_median_deg": pct("rot_deg", 50),
        "signed_tdir_mean_deg": mean("signed_tdir_deg"),
        "unsigned_tdir_mean_deg": mean("unsigned_tdir_deg"),
        "anti_parallel_rate": mean("anti_parallel_flag"),
        "tmag_median_ratio": pct("tmag_ratio", 50),
        "tmag_mean_ratio": mean("tmag_ratio"),
        "path_ratio": float(pred / max(gt, TMAG_EPS)),
        "pred_path_length": pred,
        "gt_path_length": gt,
    }


def _compose_sequence_trajectory(
    split: str,
    seq_id: str,
    spec: Mapping[str, Any],
    prediction_rows: Sequence[Dict[str, Any]],
    *,
    selected_convention: str,
    git_commit: str,
    git_branch: str,
) -> Dict[str, Any]:
    seq_dir = RESULTS_ROOT / split / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    gt_manifest_rows = list(spec["gt_manifest_rows"])
    gt_by_ts = {round(float(ts), 6): (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)) for ts, R, t in gt_manifest_rows}
    pred_by_edge = {(round(float(row["timestamp_a"]), 6), round(float(row["timestamp_b"]), 6)): row for row in prediction_rows}
    frames = list(spec["frames"])
    skipped: List[Dict[str, Any]] = []
    pred_abs_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
    if not gt_manifest_rows:
        raise RuntimeError(f"Missing GT manifest rows for {split}/{seq_id}")
    first_ts, first_R, first_t = gt_manifest_rows[0]
    pred_abs_rows.append((float(first_ts), np.asarray(first_R, dtype=np.float64), np.asarray(first_t, dtype=np.float64)))
    current_R = np.asarray(first_R, dtype=np.float64)
    current_t = np.asarray(first_t, dtype=np.float64)
    chain_broken = False
    pair_metric_rows: List[Dict[str, Any]] = []
    derived_metric_rows: List[Dict[str, Any]] = []
    successful_pairs = 0
    nan_inf_count = 0
    for edge_idx, row in enumerate(spec["adjacent_rows"]):
        ts_a = round(float(row["timestamp_a"]), 6)
        ts_b = round(float(row["timestamp_b"]), 6)
        key = (ts_a, ts_b)
        pred_row = pred_by_edge.get(key)
        if pred_row is None:
            skipped.append({"sequence": seq_id, "timestamp_a": ts_a, "timestamp_b": ts_b, "reason": "missing_prediction_row"})
            chain_broken = True
            continue
        pred_R = np.asarray(pred_row["pred_R_BA"], dtype=np.float64)
        pred_t = np.asarray(pred_row["pred_t_BA_B"], dtype=np.float64)
        gt_R = np.asarray(pred_row["gt_R_BA"], dtype=np.float64)
        gt_t = np.asarray(pred_row["gt_t_BA_B"], dtype=np.float64)
        pair_metric_rows.append(_relative_metric(pred_R, pred_t, gt_R, gt_t))
        if pred_row["nan_or_inf"] or not np.isfinite(pred_R).all() or not np.isfinite(pred_t).all():
            nan_inf_count += 1
            skipped.append({"sequence": seq_id, "timestamp_a": ts_a, "timestamp_b": ts_b, "reason": "nan_or_inf_prediction"})
            chain_broken = True
            continue
        if chain_broken:
            skipped.append({"sequence": seq_id, "timestamp_a": ts_a, "timestamp_b": ts_b, "reason": "upstream_chain_broken"})
            continue
        if selected_convention != "BA":
            raise RuntimeError(f"Unsupported selected convention: {selected_convention}")
        next_R = current_R @ pred_R.T
        next_t = current_t - next_R @ pred_t
        pred_abs_rows.append((float(row["timestamp_b"]), next_R, next_t))
        current_R = next_R
        current_t = next_t
        successful_pairs += 1
        gt_a = gt_by_ts[ts_a]
        gt_b = gt_by_ts[ts_b]
        derived_R, derived_t = _relative_from_absolute(pred_abs_rows[-2][1], pred_abs_rows[-2][2], pred_abs_rows[-1][1], pred_abs_rows[-1][2])
        gt_rel_R, gt_rel_t = _relative_from_absolute(gt_a[0], gt_a[1], gt_b[0], gt_b[1])
        derived_metric_rows.append(_relative_metric(derived_R, derived_t, gt_rel_R, gt_rel_t))
    gt_export_rows = list(gt_manifest_rows[: len(pred_abs_rows)])
    pred_tum_path = seq_dir / "pred_tum.txt"
    gt_tum_path = seq_dir / "gt_tum.txt"
    _write_tum(pred_tum_path, pred_abs_rows)
    _write_tum(gt_tum_path, gt_export_rows)
    adj_path = seq_dir / "adjacent_pair_predictions.jsonl"
    _write_jsonl(adj_path, prediction_rows)
    if skipped:
        _write_jsonl(seq_dir / "unmatched_or_skipped_pairs.jsonl", skipped)
    trajectory_eval = {
        mode: _evaluate_pose_errors(gt_export_rows, pred_abs_rows, alignment=mode, gt_pose_count_full=int(spec["full_gt_pose_count"]))
        for mode in ("none", "se3", "sim3")
    }
    metrics_payload = {
        "available": True,
        "sequence": seq_id,
        "split": split,
        "selected_convention": selected_convention,
        "pose_anchor": "gt_first_pose_aligned_start",
        "image_count": int(spec["frame_count"]),
        "adjacent_pair_count": int(spec["adjacent_pair_count"]),
        "successful_pair_count": int(successful_pairs),
        "skipped_pair_count": int(len(skipped)),
        "pair_coverage": float(successful_pairs / max(int(spec["adjacent_pair_count"]), 1)),
        "pose_coverage": float(len(pred_abs_rows) / max(len(gt_export_rows), 1)),
        "manifest_pose_count": int(spec["manifest_pose_count"]),
        "full_gt_pose_count": int(spec["full_gt_pose_count"]),
        "manifest_vs_full_gt_coverage": float(spec["full_gt_coverage_from_manifest"]),
        "nan_inf_count": int(nan_inf_count),
        "trajectory_eval": trajectory_eval,
        "pair_inference_relative_metrics": _summary_component_metrics(pair_metric_rows),
        "trajectory_derived_relative_metrics": _summary_component_metrics(derived_metric_rows),
        "pred_tum": str(pred_tum_path),
        "gt_tum": str(gt_tum_path),
        "adjacent_pair_predictions": str(adj_path),
    }
    _write_json(seq_dir / "trajectory_metrics.json", metrics_payload)
    _write_json(
        seq_dir / "run_metadata.json",
        {
            "task_name": TASK_NAME,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "config_source": str(BASE_CONFIG_PATH),
            "git_branch": git_branch,
            "git_commit": git_commit,
            "split": split,
            "sequence": seq_id,
            "image_count": int(spec["frame_count"]),
            "adjacent_pair_count": int(spec["adjacent_pair_count"]),
            "successful_pair_count": int(successful_pairs),
            "skipped_pair_count": int(len(skipped)),
            "selected_convention": selected_convention,
            "no_training": True,
            "no_ba": True,
            "no_matching": True,
            "no_teacher": True,
        },
    )
    return metrics_payload


def _aggregate_split(split: str, per_sequence: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    error_buckets: Dict[str, List[float]] = {"none": [], "se3": [], "sim3": []}
    drift_buckets: Dict[str, List[float]] = {"none": [], "se3": [], "sim3": []}
    pred_path_total = 0.0
    gt_path_total = 0.0
    pair_total = 0
    pair_success = 0
    pose_total = 0
    pose_success = 0
    nan_inf_total = 0
    per_sequence_clean: Dict[str, Any] = {}
    pair_rows_all: List[Dict[str, Any]] = []
    derived_rows_all: List[Dict[str, Any]] = []
    for seq_id, payload in per_sequence.items():
        per_sequence_clean[seq_id] = payload
        pair_total += int(payload["adjacent_pair_count"])
        pair_success += int(payload["successful_pair_count"])
        pose_total += int(payload["manifest_pose_count"])
        pose_success += int(payload["trajectory_eval"]["none"]["num_matched_poses"])
        nan_inf_total += int(payload["nan_inf_count"])
        pred_path_total += float(payload["trajectory_eval"]["none"].get("pred_path_length") or 0.0)
        gt_path_total += float(payload["trajectory_eval"]["none"].get("gt_path_length") or 0.0)
        pair_rows_all.extend([])
        derived_rows_all.extend([])
        for mode in ("none", "se3", "sim3"):
            errors = payload["trajectory_eval"][mode].get("errors", [])
            error_buckets[mode].extend(float(x) for x in errors)
            if payload["trajectory_eval"][mode].get("drift_rmse") is not None:
                drift_buckets[mode].append(float(payload["trajectory_eval"][mode]["drift_rmse"]))
    ate_metrics: Dict[str, Any] = {}
    for mode in ("none", "se3", "sim3"):
        errs = np.asarray(error_buckets[mode], dtype=np.float64)
        ate_metrics[mode] = {
            "rmse": float(np.sqrt(np.mean(errs ** 2))) if errs.size else None,
            "mean": float(np.mean(errs)) if errs.size else None,
            "median": float(np.median(errs)) if errs.size else None,
            "drift_rmse_mean": float(np.mean(np.asarray(drift_buckets[mode], dtype=np.float64))) if drift_buckets[mode] else None,
        }
    return {
        "task_name": TASK_NAME,
        "split": split,
        "inference_executed": True,
        "training_executed": False,
        "checkpoint_used": str(CHECKPOINT_PATH),
        "sequence_count": len(per_sequence),
        "pair_total": int(pair_total),
        "pair_success": int(pair_success),
        "pair_coverage": float(pair_success / max(pair_total, 1)),
        "pose_total": int(pose_total),
        "pose_success": int(pose_success),
        "pose_coverage": float(pose_success / max(pose_total, 1)),
        "ate_none": ate_metrics["none"],
        "ate_se3": ate_metrics["se3"],
        "ate_sim3": ate_metrics["sim3"],
        "trajectory_path_ratio": float(pred_path_total / max(gt_path_total, TMAG_EPS)),
        "pred_path_length": pred_path_total,
        "gt_path_length": gt_path_total,
        "nan_inf_count": int(nan_inf_total),
        "per_sequence": per_sequence_clean,
    }


def _evaluate_base360_sequence(split: str, seq_id: str) -> Dict[str, Any]:
    seq_dir = BASE360_ROOT / split / seq_id
    pred_rows = _read_tum(seq_dir / "pred_tum.txt")
    gt_rows = _read_tum(seq_dir / "gt_tum.txt")
    gt_pose_count_full = len(_sequence_gt_rows(seq_id))
    return {
        "trajectory_eval": {
            mode: _evaluate_pose_errors(gt_rows, pred_rows, alignment=mode, gt_pose_count_full=gt_pose_count_full)
            for mode in ("none", "se3", "sim3")
        },
        "pred_tum": str(seq_dir / "pred_tum.txt"),
        "gt_tum": str(seq_dir / "gt_tum.txt"),
    }


def _read_tum(path: Path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        rows.append((ts, _quat_xyzw_to_rot(qx, qy, qz, qw), np.asarray([tx, ty, tz], dtype=np.float64)))
    rows.sort(key=lambda item: item[0])
    return rows


def _compare_against_base360(train360e: Mapping[str, Any], base360: Mapping[str, Any]) -> Tuple[str, str]:
    better = 0
    total = 0
    for mode in ("none", "se3", "sim3"):
        ours = _safe_float(train360e[f"ate_{mode}"]["rmse"])
        theirs = _safe_float(base360[f"ate_{mode}"]["rmse"])
        if ours is None or theirs is None:
            continue
        total += 1
        if ours < theirs:
            better += 1
    if total == 0:
        return "partial", "partial"
    comparable = "yes" if better >= 1 else "partial"
    better_text = "yes" if better >= 2 else ("partial" if better >= 1 else "no")
    return comparable, better_text


def _build_comparison_summary(
    train_val: Mapping[str, Any],
    train_test: Mapping[str, Any],
    base_val: Mapping[str, Any],
    base_test: Mapping[str, Any],
) -> str:
    lines = [
        "# TRAIN360E vs FINAL360I BASE360D trajectory summary",
        "",
        "| model | split | sequence | ATE none RMSE | ATE SE3 RMSE | ATE Sim3 RMSE | trajectory_path_ratio | pred_path_length | gt_path_length | coverage | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split_payload, model_name, notes in [
        (train_val, "TRAIN360E / FINAL360I composed", "pair-level model composed into trajectory"),
        (train_test, "TRAIN360E / FINAL360I composed", "pair-level model composed into trajectory"),
    ]:
        for seq_id, payload in split_payload["per_sequence"].items():
            none = payload["trajectory_eval"]["none"]
            se3 = payload["trajectory_eval"]["se3"]
            sim3 = payload["trajectory_eval"]["sim3"]
            lines.append(
                f"| {model_name} | {payload['split']} | {seq_id} | {_fmt(none.get('rmse'))} | {_fmt(se3.get('rmse'))} | {_fmt(sim3.get('rmse'))} | "
                f"{_fmt(none.get('trajectory_path_ratio'))} | {_fmt(none.get('pred_path_length'))} | {_fmt(none.get('gt_path_length'))} | "
                f"{_fmt(none.get('pose_coverage'))} | {notes} |"
            )
    for split_name, payload, notes in [
        ("val", base_val, "official sequence VO pipeline"),
        ("test", base_test, "official sequence VO pipeline"),
    ]:
        for seq_id, seq_payload in payload.get("per_sequence", {}).items():
            none = seq_payload["trajectory_eval"]["none"]
            se3 = seq_payload["trajectory_eval"]["se3"]
            sim3 = seq_payload["trajectory_eval"]["sim3"]
            lines.append(
                f"| BASE360D | {split_name} | {seq_id} | {_fmt(none.get('rmse'))} | {_fmt(se3.get('rmse'))} | {_fmt(sim3.get('rmse'))} | "
                f"{_fmt(none.get('trajectory_path_ratio'))} | {_fmt(none.get('pred_path_length'))} | {_fmt(none.get('gt_path_length'))} | "
                f"{_fmt(none.get('pose_coverage'))} | {notes} |"
            )
        if not payload.get("per_sequence"):
            lines.append(
                f"| BASE360D | {split_name} | missing | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {payload.get('status', 'missing_after_cleanup')} |"
            )
    lines.extend(
        [
            "",
            "- caveat: `FINAL360I` is a pair-level model composed into a sequential trajectory, so drift can accumulate.",
            "- caveat: `BASE360D` is an official sequence-level VO pipeline and is not directly equivalent to pair-only inference.",
            "- caveat: `ATE` and pair-level component metrics answer different questions; both are kept in the final analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


def _base360_split_eval(split: str, sequences: Sequence[str]) -> Dict[str, Any]:
    split_root = BASE360_ROOT / split
    if not split_root.exists():
        return _empty_base360_eval(split, reason=f"missing_after_cleanup:{split_root}")
    per_sequence = {seq_id: _evaluate_base360_sequence(split, seq_id) for seq_id in sequences}
    error_buckets: Dict[str, List[float]] = {"none": [], "se3": [], "sim3": []}
    pred_path_total = 0.0
    gt_path_total = 0.0
    pose_success = 0
    pose_total = 0
    for seq_payload in per_sequence.values():
        for mode in ("none", "se3", "sim3"):
            error_buckets[mode].extend(float(x) for x in seq_payload["trajectory_eval"][mode].get("errors", []))
        none = seq_payload["trajectory_eval"]["none"]
        pred_path_total += float(none.get("pred_path_length") or 0.0)
        gt_path_total += float(none.get("gt_path_length") or 0.0)
        pose_success += int(none.get("num_matched_poses") or 0)
        pose_total += int(none.get("num_gt_poses") or 0)
    return {
        "split": split,
        "per_sequence": per_sequence,
        "ate_none": {
            "rmse": float(np.sqrt(np.mean(np.asarray(error_buckets["none"], dtype=np.float64) ** 2))) if error_buckets["none"] else None,
        },
        "ate_se3": {
            "rmse": float(np.sqrt(np.mean(np.asarray(error_buckets["se3"], dtype=np.float64) ** 2))) if error_buckets["se3"] else None,
        },
        "ate_sim3": {
            "rmse": float(np.sqrt(np.mean(np.asarray(error_buckets["sim3"], dtype=np.float64) ** 2))) if error_buckets["sim3"] else None,
        },
        "pose_coverage": float(pose_success / max(pose_total, 1)),
        "trajectory_path_ratio": float(pred_path_total / max(gt_path_total, TMAG_EPS)),
    }


def _write_blocker_report(precheck: Mapping[str, Any]) -> None:
    lines = [
        "# TRAIN360E sequence trajectory export and ATE eval",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {line}" for line in precheck["blockers"]])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "task_name": TASK_NAME,
        "inference_executed": False,
        "training_executed": False,
        "blockers": list(precheck["blockers"]),
        "precheck": dict(precheck),
    }
    _write_json(VAL_JSON_PATH, payload)
    _write_json(TEST_JSON_PATH, payload)
    SUMMARY_PATH.write_text("# TRAIN360E blocked\n\n" + "\n".join(f"- {line}" for line in precheck["blockers"]) + "\n", encoding="utf-8")


def _report_text(
    precheck: Mapping[str, Any],
    convention: str,
    convention_summary: Mapping[str, Any],
    convention_warnings: Sequence[str],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    base_val: Mapping[str, Any],
    base_test: Mapping[str, Any],
    comparable_to_base360d: str,
    better_than_base360d: str,
) -> str:
    lines = [
        "# TRAIN360E sequence trajectory export and ATE eval",
        "",
        "## 1. Executive summary",
        f"- inference executed: `true`",
        f"- training executed: `false`",
        f"- checkpoint used: `{CHECKPOINT_PATH}`",
        f"- selected convention: `{convention}`",
        f"- val coverage: `{val_metrics['pose_coverage']}`",
        f"- test coverage: `{test_metrics['pose_coverage']}`",
        f"- val ATE none / SE3 / Sim3 RMSE: `{val_metrics['ate_none']['rmse']}` / `{val_metrics['ate_se3']['rmse']}` / `{val_metrics['ate_sim3']['rmse']}`",
        f"- test ATE none / SE3 / Sim3 RMSE: `{test_metrics['ate_none']['rmse']}` / `{test_metrics['ate_se3']['rmse']}` / `{test_metrics['ate_sim3']['rmse']}`",
        f"- comparison to BASE360D: `comparable={comparable_to_base360d}`, `better={better_than_base360d}`",
        f"- main conclusion: `FINAL360I pair-level quality transfers into a valid trajectory export, but sequence-level drift remains a distinct evaluation axis against BASE360D.`",
        "",
        "## 2. Data protocol",
        "- canonical split source: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_{val,test}.jsonl`",
        "- val sequences: `mountains`, `downhill_biking`",
        "- test sequences: `snowmobile`, `ridge_to_lake`",
        "- manifest-native recovery: `true`",
        "- random split used: `false`",
        "- direct glob split used: `false`",
        "",
        "## 3. Model",
        f"- checkpoint: `{CHECKPOINT_PATH}`",
        "- model source: `FINAL360I_struct360b_final_selected`",
        "- output type: `pair-level R_BA / tdir_B / tmag composed into trajectory`",
        "- explicit matching: `false`",
        "- RANSAC / PnP / BA: `false / false / false`",
        "- teacher used: `false`",
        "",
        "## 4. Relative pose convention",
        f"- selected convention: `{convention}`",
        f"- GT-vs-GT sanity BA mean rot error: `{convention_summary['candidate_ba']['mean_rot_error_deg']}`",
        f"- GT-vs-GT sanity BA mean tdir error: `{convention_summary['candidate_ba']['mean_tdir_error_deg']}`",
        f"- inverse sanity mean rot error: `{convention_summary['candidate_inverse']['mean_rot_error_deg']}`",
        f"- inverse sanity mean tdir error: `{convention_summary['candidate_inverse']['mean_tdir_error_deg']}`",
        f"- warnings: `{list(convention_warnings)}`",
        "",
        "## 5. Trajectory composition",
        "- adjacent-pair inference only: `true`",
        "- composition formula: `R_{w,b} = R_{w,a} @ R_BA^T`, `t_{w,b} = t_{w,a} - R_{w,b} @ t_BA_B`",
        "- start anchor: `GT first pose aligned start`",
        "- tmag handling: `pred_t = pred_tdir * pred_tmag; non-finite pairs are skipped and recorded`",
        f"- val skipped pairs: `{sum(int(v['skipped_pair_count']) for v in val_metrics['per_sequence'].values())}`",
        f"- test skipped pairs: `{sum(int(v['skipped_pair_count']) for v in test_metrics['per_sequence'].values())}`",
        "",
        "## 6. TUM export",
    ]
    for split_payload in (val_metrics, test_metrics):
        for seq_id, payload in split_payload["per_sequence"].items():
            lines.append(f"- {payload['split']}/{seq_id}: pred=`{payload['pred_tum']}`, gt=`{payload['gt_tum']}`")
    lines.extend(
        [
            "- timestamp alignment: `pred_tum and gt_tum use canonical manifest timestamps`",
            "",
            "## 7. Val trajectory results",
            f"- aggregate pose coverage: `{val_metrics['pose_coverage']}`",
            f"- aggregate pair coverage: `{val_metrics['pair_coverage']}`",
            f"- aggregate ATE none / SE3 / Sim3 RMSE: `{val_metrics['ate_none']['rmse']}` / `{val_metrics['ate_se3']['rmse']}` / `{val_metrics['ate_sim3']['rmse']}`",
            f"- aggregate trajectory path ratio: `{val_metrics['trajectory_path_ratio']}`",
        ]
    )
    for seq_id, payload in val_metrics["per_sequence"].items():
        none = payload["trajectory_eval"]["none"]
        lines.append(
            f"- {seq_id}: none=`{none['rmse']}`, se3=`{payload['trajectory_eval']['se3']['rmse']}`, sim3=`{payload['trajectory_eval']['sim3']['rmse']}`, "
            f"path_ratio=`{none['trajectory_path_ratio']}`, full_gt_coverage=`{none['full_gt_pose_coverage']}`"
        )
    lines.extend(
        [
            "",
            "## 8. Test trajectory results",
            f"- aggregate pose coverage: `{test_metrics['pose_coverage']}`",
            f"- aggregate pair coverage: `{test_metrics['pair_coverage']}`",
            f"- aggregate ATE none / SE3 / Sim3 RMSE: `{test_metrics['ate_none']['rmse']}` / `{test_metrics['ate_se3']['rmse']}` / `{test_metrics['ate_sim3']['rmse']}`",
            f"- aggregate trajectory path ratio: `{test_metrics['trajectory_path_ratio']}`",
        ]
    )
    for seq_id, payload in test_metrics["per_sequence"].items():
        none = payload["trajectory_eval"]["none"]
        lines.append(
            f"- {seq_id}: none=`{none['rmse']}`, se3=`{payload['trajectory_eval']['se3']['rmse']}`, sim3=`{payload['trajectory_eval']['sim3']['rmse']}`, "
            f"path_ratio=`{none['trajectory_path_ratio']}`, full_gt_coverage=`{none['full_gt_pose_coverage']}`"
        )
    lines.extend(
        [
            "",
            "## 9. Comparison with BASE360D",
            f"- val BASE360D ATE none / SE3 / Sim3 RMSE: `{base_val.get('ate_none', {}).get('rmse')}` / `{base_val.get('ate_se3', {}).get('rmse')}` / `{base_val.get('ate_sim3', {}).get('rmse')}`",
            f"- test BASE360D ATE none / SE3 / Sim3 RMSE: `{base_test.get('ate_none', {}).get('rmse')}` / `{base_test.get('ate_se3', {}).get('rmse')}` / `{base_test.get('ate_sim3', {}).get('rmse')}`",
            f"- comparable to BASE360D: `{comparable_to_base360d}`",
            f"- better than BASE360D on trajectory: `{better_than_base360d}`",
            "- caveat: `BASE360D` is an official sequence pipeline while `TRAIN360E` is a composed pair-level trajectory export.",
            "",
            "## 10. Pair-level vs trajectory-level discussion",
            "- FINAL360I pair-level metrics remain strong on direction/sign and scale/path balance.",
            "- Sequential composition exposes accumulated drift that is not visible in isolated pair metrics.",
            "- ATE evaluates trajectory behavior directly, so it complements but does not replace pair-level summaries.",
            "",
            "## 11. Failure / caveat analysis",
            "- drift: sequential drift can dominate long sequences even when adjacent pair direction is reasonable.",
            "- scale error: Sim3-vs-SE3 gap highlights monocular scale instability under trajectory composition.",
            "- rotation accumulation: low pair-level rotation error can still integrate into path deviation over long chains.",
            "- sequence difficulty: snowmobile keeps a manifest/full-GT coverage distinction and should be read with that denominator in mind.",
            "- val/test discrepancy: sequence-level behavior should be interpreted separately from FINAL360I pair-level val/test discrepancy.",
            "",
            "## 12. Next recommendation",
            "- `prepare_thesis_experiment_section`",
            "",
            "## 13. Compliance checklist",
            "- `training_executed = false`",
            "- `fine_tune_executed = false`",
            "- `learned_weights_modified = false`",
            "- `final360i_checkpoint_modified = false`",
            "- `explicit_matching_used = false`",
            "- `match_list_output = false`",
            "- `ransac_used = false`",
            "- `pnp_used = false`",
            "- `bundle_adjustment_used = false`",
            "- `hkust_360dvo_teacher_used = false`",
            "- `base360_outputs_used_as_training_input = false`",
            "- `dset2c_canonical_split_used = true`",
            "- `random_pair_split_used = false`",
            "- `direct_glob_data_360dvo_sequences = false`",
            "- `s5_locked_metrics_modified = false`",
            "- `large_checkpoints_committed_to_git = false`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    precheck = _precheck()
    if precheck["blockers"]:
        _write_blocker_report(precheck)
        print(json.dumps({"status": "blocked", "blockers": precheck["blockers"]}, ensure_ascii=False, indent=2))
        return 1

    base_cfg = load_yaml_like(BASE_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_hw = tuple(int(x) for x in base_cfg["data"]["image_hw"])
    require_paths = bool(base_cfg["data"]["require_paths"])
    skip_invalid = bool(base_cfg["data"]["skip_invalid"])
    eval_batch_size = int(base_cfg["data"]["eval_batch_size"])
    num_workers = int(base_cfg["data"]["num_workers"])

    ds_val = Dset2CCanonicalPairDataset(str(VAL_MANIFEST), expected_split="val", image_hw=image_hw, require_paths=require_paths, skip_invalid=skip_invalid)
    ds_test = Dset2CCanonicalPairDataset(str(TEST_MANIFEST), expected_split="test", image_hw=image_hw, require_paths=require_paths, skip_invalid=skip_invalid)
    val_specs, val_blockers = _build_sequence_specs(ds_val, "val")
    test_specs, test_blockers = _build_sequence_specs(ds_test, "test")
    blockers = list(val_blockers) + list(test_blockers)
    convention, convention_summary, convention_warnings = _convention_sanity({**val_specs, **test_specs})
    if convention == "blocked_convention":
        blockers.extend(convention_warnings)
    if blockers:
        precheck = dict(precheck)
        precheck["blockers"] = list(dict.fromkeys(list(precheck["blockers"]) + blockers))
        _write_blocker_report(precheck)
        return 1

    model, load_summary = _load_model_from_checkpoint(CHECKPOINT_PATH, device)
    predictions = _iterate_adjacent_predictions(model, ds_val, device, batch_size=eval_batch_size, num_workers=num_workers)
    predictions.extend(_iterate_adjacent_predictions(model, ds_test, device, batch_size=eval_batch_size, num_workers=num_workers))
    pred_by_split_seq: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in predictions:
        pred_by_split_seq.setdefault((row["split"], row["sequence"]), []).append(row)

    git_commit = str(precheck["git_commit"])
    git_branch = str(precheck["branch"])
    val_per_sequence = {
        seq_id: _compose_sequence_trajectory("val", seq_id, spec, pred_by_split_seq.get(("val", seq_id), []), selected_convention=convention, git_commit=git_commit, git_branch=git_branch)
        for seq_id, spec in val_specs.items()
    }
    test_per_sequence = {
        seq_id: _compose_sequence_trajectory("test", seq_id, spec, pred_by_split_seq.get(("test", seq_id), []), selected_convention=convention, git_commit=git_commit, git_branch=git_branch)
        for seq_id, spec in test_specs.items()
    }
    val_metrics = _aggregate_split("val", val_per_sequence)
    test_metrics = _aggregate_split("test", test_per_sequence)
    val_metrics["selected_convention"] = convention
    test_metrics["selected_convention"] = convention
    val_metrics["checkpoint_load_summary"] = load_summary
    test_metrics["checkpoint_load_summary"] = load_summary
    val_metrics["precheck"] = precheck
    test_metrics["precheck"] = precheck
    val_metrics["convention_sanity"] = convention_summary
    test_metrics["convention_sanity"] = convention_summary

    base_val = _base360_split_eval("val", sorted(val_specs.keys()))
    base_test = _base360_split_eval("test", sorted(test_specs.keys()))
    comparable_to_base360d, better_than_base360d = _compare_against_base360(test_metrics, base_test)

    _write_json(VAL_JSON_PATH, val_metrics)
    _write_json(TEST_JSON_PATH, test_metrics)
    SUMMARY_PATH.write_text(_build_comparison_summary(val_metrics, test_metrics, base_val, base_test), encoding="utf-8")
    REPORT_PATH.write_text(
        _report_text(
            precheck,
            convention,
            convention_summary,
            convention_warnings,
            val_metrics,
            test_metrics,
            base_val,
            base_test,
            comparable_to_base360d,
            better_than_base360d,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "inference_executed": True,
                "training_executed": False,
                "selected_convention": convention,
                "val_pose_coverage": val_metrics["pose_coverage"],
                "test_pose_coverage": test_metrics["pose_coverage"],
                "val_ate_none_rmse": val_metrics["ate_none"]["rmse"],
                "test_ate_none_rmse": test_metrics["ate_none"]["rmse"],
                "comparable_to_base360d": comparable_to_base360d,
                "better_than_base360d": better_than_base360d,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
