#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from tools.eval_clean_policy import DtBucketScaledMagnitudeModel, _load_fine_model
from train_mvp import _camera_center_from_T_c0_np, _compose_rel_pose_np, _resolve_clean_policy_payload


ALLOWED_CLASSIFICATIONS = [
    "S5D_DENSE_EXPORT_READY",
    "S5D_SPARSE_EXPORT_ONLY",
    "S5D_DENSE_EXPORT_UNAVAILABLE",
    "S5D_EXPORT_REJECTED_GT_LEAKAGE_RISK",
    "S5D_EXPORT_ERROR",
]

DEFAULT_POLICY = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
DEFAULT_TIMESTAMPS = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "timestamps.txt"
DEFAULT_GT = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "groundtruth_tum.txt"
DEFAULT_OUT = REPO_ROOT / "external_baselines" / "results" / "s5_dense" / "scene01_seq03_s5_dense_est_tum.txt"
DEFAULT_JSON = REPO_ROOT / "checkpoints" / "S5D_dense_external_export.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "s5_dense_external_export_report.md"


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _read_timestamps(path: Path) -> List[str]:
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            vals.append(line.split()[0])
    return vals


def _read_tum(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append([float(x) for x in parts[:8]])
    return rows


def _quat_xyzw_from_rot(R: np.ndarray) -> List[float]:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _load_s5_model(policy_path: Path):
    policy = _resolve_clean_policy_payload(str(policy_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        max_eval_batches=0,
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    factors = {k: float(v) for k, v in policy["effective_bucket_factors"].items()}
    return DtBucketScaledMagnitudeModel(model, factors), cfg, device, policy, load_summary


def _make_dense_dataset(cfg: Any, scene: str, seq: str) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=None,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=(1,),
        pair_step=1,
        min_dt=0.0,
        max_dt=None,
        scenes=[scene],
        seqs=[seq],
    )


def _export_dense(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[str]]:
    policy_path = _resolve(args.policy)
    timestamps_path = _resolve(args.timestamps)
    out_path = _resolve(args.out)
    scene = str(args.scene)
    seq = str(args.seq)
    timestamps = _read_timestamps(timestamps_path)
    wrapped, cfg, device, policy, load_summary = _load_s5_model(policy_path)
    ds = _make_dense_dataset(cfg, scene, seq)
    manifest = sorted(ds.manifest(), key=lambda m: (int(m["i"]), int(m["j"])))
    seq_ts = list(ds.seqs_data[0]["ts"]) if getattr(ds, "seqs_data", None) else []

    notes: List[str] = []
    timestamp_match_status = "aligned_exact" if seq_ts == timestamps else "mismatch"
    if timestamp_match_status != "aligned_exact":
        notes.append("Dataset sequence timestamps do not exactly match the external timestamp file.")

    contiguous = (
        len(manifest) == max(len(seq_ts) - 1, 0)
        and all(int(m["i"]) == idx and int(m["j"]) == idx + 1 for idx, m in enumerate(manifest))
    )
    if not contiguous:
        notes.append("S5 dataset manifest did not expose a complete adjacent-pair chain.")

    rows: List[str] = []
    if seq_ts:
        rows.append(
            f"{float(seq_ts[0]):.6f} 0.000000000 0.000000000 0.000000000 "
            "0.000000000 0.000000000 0.000000000 1.000000000"
        )

    world_R = np.eye(3, dtype=np.float64)
    world_t = np.zeros(3, dtype=np.float64)
    if timestamp_match_status == "aligned_exact" and contiguous:
        wrapped.eval()
        with torch.no_grad():
            for item in manifest:
                sample = ds[int(item["_ds_idx"])] if "_ds_idx" in item else ds[manifest.index(item)]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                meta = sample.get("meta", {})
                dt_val = float(meta.get("dt_world", item.get("dt_world", 0.01)))
                dt_tensor = torch.tensor([dt_val], device=device, dtype=torch.float32)
                R_pred, t_pred, aux = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                R_np = R_pred.detach().float().cpu().numpy()[0]
                t_vec_metric = None
                if isinstance(aux, dict) and aux.get("t_vec_out", None) is not None:
                    t_vec_metric = aux["t_vec_out"].detach().float().cpu().numpy()[0]
                elif isinstance(aux, dict) and aux.get("t_mag", None) is not None:
                    t_dir_t = aux.get("t_dir_out", t_pred)
                    t_dir = torch.nn.functional.normalize(t_dir_t.detach().float(), dim=-1, eps=1e-6).cpu().numpy()[0]
                    t_mag = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
                    t_vec_metric = t_dir * t_mag
                if t_vec_metric is None:
                    raise RuntimeError("S5 dense export failed because metric translation vector was unavailable.")
                world_R, world_t = _compose_rel_pose_np(R_np, np.asarray(t_vec_metric, dtype=np.float64), world_R, world_t)
                camera_center = _camera_center_from_T_c0_np(world_R, world_t)
                qx, qy, qz, qw = _quat_xyzw_from_rot(world_R)
                ts_b = seq_ts[int(item["j"])]
                rows.append(
                    f"{float(ts_b):.6f} {camera_center[0]:.9f} {camera_center[1]:.9f} {camera_center[2]:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "policy": policy,
        "cfg": cfg,
        "load_summary": load_summary,
        "trajectory_path": out_path,
        "timestamps": timestamps,
        "sequence_timestamps": seq_ts,
        "manifest_pairs": manifest,
        "timestamp_match_status": timestamp_match_status,
        "contiguous_adjacent_pairs": bool(contiguous),
        "num_rows_written": len(rows),
    }, notes


def _safety_checks(out_path: Path, gt_path: Path, num_timestamps: int) -> Dict[str, Any]:
    est = _read_tum(out_path)
    gt = _read_tum(gt_path)
    ts = [r[0] for r in est]
    duplicate_timestamps = len(ts) - len(set(ts))
    monotonic = all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))
    line_count = len(est)
    gt_by_ts = {round(r[0], 6): r for r in gt}
    matched_est = []
    matched_gt = []
    for row in est:
        g = gt_by_ts.get(round(row[0], 6))
        if g is not None:
            matched_est.append(row)
            matched_gt.append(g)
    leakage_passed = True
    leakage_reason = "not_identical_to_groundtruth"
    mean_translation_delta = None
    if matched_est and len(matched_est) == len(est):
        est_arr = np.asarray(matched_est, dtype=np.float64)
        gt_arr = np.asarray(matched_gt, dtype=np.float64)
        trans_delta = np.linalg.norm(est_arr[:, 1:4] - gt_arr[:, 1:4], axis=1)
        mean_translation_delta = float(trans_delta.mean())
        if np.allclose(est_arr[:, 1:8], gt_arr[:, 1:8], atol=1.0e-7, rtol=0.0):
            leakage_passed = False
            leakage_reason = "estimated_tum_rows_match_groundtruth_pose_columns"
        elif mean_translation_delta < 1.0e-9:
            leakage_passed = False
            leakage_reason = "estimated_translations_are_numerically_indistinguishable_from_groundtruth"
    sparse_only = line_count < int(0.99 * max(num_timestamps, 1))
    return {
        "gt_used_to_generate_prediction": False,
        "gt_leakage_check_passed": bool(leakage_passed),
        "gt_leakage_check_reason": leakage_reason,
        "matched_gt_rows_for_leakage_check": int(len(matched_est)),
        "mean_translation_delta_vs_gt": mean_translation_delta,
        "duplicate_timestamps": int(duplicate_timestamps),
        "monotonic_timestamps": bool(monotonic),
        "trajectory_file_line_count": int(line_count),
        "sparse_only": bool(sparse_only),
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    export = payload["export"]
    safety = payload["safety_checks"]
    lines = [
        "# S5 Dense External Trajectory Export Report",
        "",
        "## Executive summary",
        "",
        f"- Experiment: `{payload['experiment']}`",
        f"- Final classification: `{payload['final_classification']}`",
        "- Checkpoint: `checkpoints/S5D_dense_external_export.json`",
        f"- Trajectory: `{export['trajectory_path']}`",
        f"- Estimated poses: `{export['num_est_poses']}` / timestamps `{payload['timestamps']['num_timestamps']}`",
        f"- Coverage: `{export['coverage']}`",
        "",
        "This export does not modify S5 locked metrics or official evaluator.",
        "S5 official locked metrics are unchanged and remain separate from this external diagnostic comparison.",
        "",
        "## Why dense export is needed",
        "",
        "The previous S5 external TUM export was sparse diagnostic only. A dense diagnostic export is needed before comparing S5 and ORB-SLAM3 with the same GT, same external evaluator, and same alignment modes.",
        "",
        "## S5 pose source",
        "",
        f"- Policy: `{payload['pose_source']['policy_path']}`",
        f"- Base checkpoint: `{payload['pose_source']['base_checkpoint_path']}`",
        f"- Pose source: `{export['pose_source']}`",
        f"- Composition convention: `{export['composition_convention']}`",
        f"- Frame convention: `{export['frame_convention']}`",
        "",
        "## Timestamp alignment",
        "",
        f"- Timestamp file: `{payload['timestamps']['path']}`",
        f"- Timestamp match status: `{export['timestamp_match_status']}`",
        f"- Duplicate timestamps: `{safety['duplicate_timestamps']}`",
        f"- Monotonic timestamps: `{safety['monotonic_timestamps']}`",
        "",
        "## TUM format validation",
        "",
        f"- Format: `{export['format']}`",
        f"- Trajectory file line count: `{safety['trajectory_file_line_count']}`",
        "",
        "## Coverage",
        "",
        f"- num_s5_poses: `{export['num_est_poses']}`",
        f"- num_timestamps: `{payload['timestamps']['num_timestamps']}`",
        f"- coverage: `{export['coverage']}`",
        f"- dense_or_sparse: `{'sparse' if safety['sparse_only'] else 'dense'}`",
        "",
        "## GT leakage prevention",
        "",
        "- GT poses are not copied into the exported estimate.",
        "- The exporter uses S5 model predictions and the locked policy wrapper.",
        "- GT is read only after export for leakage checks and validation.",
        f"- GT leakage check passed: `{safety['gt_leakage_check_passed']}`",
        f"- Leakage check reason: `{safety['gt_leakage_check_reason']}`",
        "",
        "## Final classification",
        "",
        f"`{payload['final_classification']}`",
        "",
        "Allowed classifications: " + ", ".join(f"`{x}`" for x in ALLOWED_CLASSIFICATIONS) + ".",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a dense diagnostic S5 TUM trajectory without modifying the official evaluator.")
    p.add_argument("--scene", default="scene01")
    p.add_argument("--seq", default="seq03")
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--timestamps", default=str(DEFAULT_TIMESTAMPS))
    p.add_argument("--groundtruth", default=str(DEFAULT_GT))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--out-json", default=str(DEFAULT_JSON))
    p.add_argument("--out-report", default=str(DEFAULT_REPORT))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_path = _resolve(args.out)
    gt_path = _resolve(args.groundtruth)
    timestamps_path = _resolve(args.timestamps)
    notes: List[str] = []
    try:
        meta, export_notes = _export_dense(args)
        notes.extend(export_notes)
        num_timestamps = len(meta["timestamps"])
        num_est = int(meta["num_rows_written"])
        coverage = float(num_est / max(num_timestamps, 1))
        safety = _safety_checks(out_path, gt_path, num_timestamps)
        if not safety["gt_leakage_check_passed"]:
            classification = "S5D_EXPORT_REJECTED_GT_LEAKAGE_RISK"
        elif not meta["contiguous_adjacent_pairs"] or meta["timestamp_match_status"] != "aligned_exact":
            classification = "S5D_DENSE_EXPORT_UNAVAILABLE"
        elif safety["sparse_only"]:
            classification = "S5D_SPARSE_EXPORT_ONLY"
        else:
            classification = "S5D_DENSE_EXPORT_READY"
        policy = meta["policy"]
        payload: Dict[str, Any] = {
            "experiment": "S5D_dense_external_trajectory_export",
            "s5_locked_metrics_reference": {
                "ate": 7.352288,
                "drift": 1.327343,
                "path_ratio": 0.932379,
                "unchanged": True,
            },
            "sequence": f"{args.scene}/{args.seq}",
            "timestamps": {
                "path": str(timestamps_path.relative_to(REPO_ROOT) if timestamps_path.is_relative_to(REPO_ROOT) else timestamps_path),
                "num_timestamps": int(num_timestamps),
            },
            "pose_source": {
                "policy_path": str(_resolve(args.policy).relative_to(REPO_ROOT)),
                "base_checkpoint_path": str(policy.get("base_checkpoint_path", "")),
                "load_missing": int(len(meta["load_summary"]["missing"])),
                "load_unexpected": int(len(meta["load_summary"]["unexpected"])),
                "dense_helper": "tools/export_s5_dense_trajectory.py",
            },
            "export": {
                "trajectory_path": str(out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path),
                "format": "TUM",
                "num_est_poses": int(num_est),
                "coverage": float(coverage),
                "timestamp_match_status": str(meta["timestamp_match_status"]),
                "pose_source": "S5 locked-policy model predictions on adjacent panorama pairs; no GT pose copy",
                "composition_convention": "adjacent relative transforms are composed with train_mvp._compose_rel_pose_np; TUM translation is the camera center from train_mvp._camera_center_from_T_c0_np",
                "frame_convention": "camera-center translation is written in TUM timestamp tx ty tz qx qy qz qw order",
            },
            "safety_checks": safety,
            "allowed_classifications": ALLOWED_CLASSIFICATIONS,
            "final_classification": classification,
            "notes": notes,
        }
    except Exception as exc:
        payload = {
            "experiment": "S5D_dense_external_trajectory_export",
            "s5_locked_metrics_reference": {
                "ate": 7.352288,
                "drift": 1.327343,
                "path_ratio": 0.932379,
                "unchanged": True,
            },
            "sequence": f"{args.scene}/{args.seq}",
            "timestamps": {"path": str(timestamps_path), "num_timestamps": len(_read_timestamps(timestamps_path)) if timestamps_path.exists() else 0},
            "export": {
                "trajectory_path": str(out_path),
                "format": "TUM",
                "num_est_poses": None,
                "coverage": None,
                "timestamp_match_status": "error",
                "pose_source": "unavailable",
                "composition_convention": "unavailable",
                "frame_convention": "unavailable",
            },
            "safety_checks": {
                "gt_used_to_generate_prediction": False,
                "gt_leakage_check_passed": False,
                "duplicate_timestamps": 0,
                "monotonic_timestamps": False,
                "sparse_only": True,
            },
            "allowed_classifications": ALLOWED_CLASSIFICATIONS,
            "final_classification": "S5D_EXPORT_ERROR",
            "notes": [f"{type(exc).__name__}: {exc}"],
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["final_classification"] != "S5D_EXPORT_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
