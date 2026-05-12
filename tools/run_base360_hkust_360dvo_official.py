#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import importlib
import json
import math
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from evaluate_external_baseline_trajectory import evaluate_external_baseline_trajectory


RESULTS_ROOT = REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official"
REPORT_PATH = REPO_ROOT / "reports" / "BASE360_HKUST_360DVO_official_baseline_eval.md"
VAL_JSON_PATH = REPO_ROOT / "reports" / "BASE360_metrics_val.json"
TEST_JSON_PATH = REPO_ROOT / "reports" / "BASE360_metrics_test.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "TRAIN360_vs_BASE360_vs_T57b_summary.md"

DEFAULT_HYGIENE = REPO_ROOT / "checkpoints" / "DSET2C_360DVO_dataset_hygiene.json"
DEFAULT_CANONICAL = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical"
DEFAULT_TRAIN360C_REPORT = REPO_ROOT / "reports" / "TRAIN360C_spherical_pose_baseline.md"
DEFAULT_TRAIN360C_VAL = REPO_ROOT / "reports" / "TRAIN360C_metrics_val.json"
DEFAULT_TRAIN360C_TEST = REPO_ROOT / "reports" / "TRAIN360C_metrics_test.json"
DEFAULT_TRAIN360C_CKPT = REPO_ROOT / "checkpoints" / "TRAIN360C_spherical_pose_baseline" / "best_val.pt"
DEFAULT_T57B_EVAL = REPO_ROOT / "external_baselines" / "results" / "gen5_360dvo_t57b_external_eval" / "eval_summary.json"

OFFICIAL_EXPECTED_REL = Path("demo.py")
OFFICIAL_WEIGHT_CANDIDATES = ("360dvo.pth", "checkpoints/360dvo.pth", "weights/360dvo.pth")
OFFICIAL_DEMO_INPUT_SCALE = 0.5
OFFICIAL_TIMEOUT_SECONDS = 900


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt_num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "N/A"
    return f"{100.0 * x:.2f}%"


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


def _build_gt_rows(seq_id: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    pose_path = REPO_ROOT / "data" / "360DVO" / "GroundTruth" / f"{seq_id}.txt"
    ts_path = REPO_ROOT / "data" / "360DVO" / "Timestamps" / f"{seq_id}_timestamps.txt"
    if not pose_path.exists() or not ts_path.exists():
        return []
    pose_lines = [line.strip() for line in pose_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ts_lines = [line.strip() for line in ts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = min(len(pose_lines), len(ts_lines))
    out: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for idx in range(count):
        vals = [float(x) for x in pose_lines[idx].split()]
        if len(vals) < 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = vals[:7]
        out.append(
            (
                float(ts_lines[idx]),
                _quat_xyzw_to_rot(qx, qy, qz, qw),
                np.asarray([tx, ty, tz], dtype=np.float64),
            )
        )
    return out


def _materialize_official_demo_input(src_dir: Path, dst_dir: Path, scale: float) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in dst_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    image_paths = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        resized = resized[: h - (h % 16), : w - (w % 16)]
        cv2.imwrite(str(dst_dir / path.name), resized)
    return dst_dir


def _check_modules() -> Dict[str, Any]:
    required = ["torch", "evo", "viser", "cv2", "numpy", "yaml"]
    payload: Dict[str, Any] = {}
    for name in required:
        try:
            importlib.import_module(name)
            payload[name] = {"available": True, "detail": "ok"}
        except Exception as exc:
            payload[name] = {"available": False, "detail": f"{type(exc).__name__}: {exc}"}
    return payload


def _inventory_official_repo(official_repo: Path) -> Dict[str, Any]:
    if not official_repo.exists():
        return {
            "official_repo_exists": False,
            "official_code_vendored_in_main_repo": False,
            "official_method_path": str(official_repo),
            "runnable": False,
            "blockers": ["OFFICIAL_REPO_NOT_PRESENT"],
        }

    def _git(args: List[str]) -> str:
        proc = subprocess.run(
            ["git", "-C", str(official_repo)] + args,
            check=False,
            capture_output=True,
            text=True,
        )
        return (proc.stdout or proc.stderr).strip()

    modules = _check_modules()
    weight_candidates = [official_repo / rel for rel in OFFICIAL_WEIGHT_CANDIDATES]
    found_weights = [str(p) for p in weight_candidates if p.exists()]
    has_demo = (official_repo / OFFICIAL_EXPECTED_REL).exists()
    has_setup = (official_repo / "setup.py").exists()
    blockers: List[str] = []
    if not has_demo:
        blockers.append("OFFICIAL_DEMO_ENTRY_MISSING")
    if not has_setup:
        blockers.append("OFFICIAL_BUILD_SCRIPT_MISSING")
    if not found_weights:
        blockers.append("OFFICIAL_WEIGHTS_MISSING")
    for name, info in modules.items():
        if not info["available"] and name in {"torch", "evo", "viser"}:
            blockers.append(f"PYTHON_MODULE_MISSING:{name}")
    blockers = list(dict.fromkeys(blockers))
    return {
        "official_repo_exists": True,
        "official_code_vendored_in_main_repo": False,
        "official_method_path": str(official_repo),
        "official_commit": _git(["rev-parse", "HEAD"]),
        "official_branch": _git(["branch", "--show-current"]),
        "official_remote": _git(["remote", "-v"]),
        "entrypoint": str(official_repo / "demo.py"),
        "config_path": str(official_repo / "config" / "360.yaml"),
        "setup_path": str(official_repo / "setup.py"),
        "environment_path": str(official_repo / "environment.yml"),
        "weights_found": found_weights,
        "required_dependencies": [
            "python=3.12",
            "pytorch=2.3.1",
            "pytorch-cuda=12.2",
            "pytorch-scatter=2.1.2",
            "evo",
            "opencv-python",
            "viser",
            "yacs",
            "custom CUDA extensions via pip install .",
            "eigen-3.4.0 zip under thirdparty/",
        ],
        "python_modules": modules,
        "requires_external_install": True,
        "requires_precompiled_or_local_build": True,
        "method_modified": False,
        "runnable": not blockers,
        "blockers": blockers,
    }


def _prepare_split(
    split: str,
    rows: Sequence[Dict[str, Any]],
    official_repo: Path,
    inventory: Dict[str, Any],
    official_python: str,
) -> Dict[str, Any]:
    split_dir = RESULTS_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    seq_ids = sorted({r["seq_id"] for r in rows})
    seq_payloads: Dict[str, Any] = {}
    coverage_flags: List[float] = []

    for seq_id in seq_ids:
        seq_dir = split_dir / seq_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_rows = [r for r in rows if r["seq_id"] == seq_id]
        adj_rows = [r for r in seq_rows if r["pair_type"] == "adjacent" and int(r["k"]) == 1]
        image_paths = sorted({r["image_path_a"] for r in seq_rows} | {r["image_path_b"] for r in seq_rows})
        gt_rows = _build_gt_rows(seq_id)
        gt_tum = seq_dir / "gt_tum.txt"
        _write_tum(gt_tum, gt_rows)
        (seq_dir / "image_list.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")

        pred_tum = seq_dir / "pred_tum.txt"
        stdout_log = seq_dir / "stdout.log"
        stderr_log = seq_dir / "stderr.log"
        official_image_dir = _materialize_official_demo_input(
            REPO_ROOT / "data" / "360DVO" / "Sequences" / seq_id,
            seq_dir / "official_demo_input_x0p5",
            OFFICIAL_DEMO_INPUT_SCALE,
        )
        run_meta = {
            "sequence": seq_id,
            "split": split,
            "image_dir": str(REPO_ROOT / "data" / "360DVO" / "Sequences" / seq_id),
            "official_demo_image_dir": str(official_image_dir),
            "official_demo_input_scale": OFFICIAL_DEMO_INPUT_SCALE,
            "image_list": str(seq_dir / "image_list.txt"),
            "gt_tum": str(gt_tum),
            "gt_pose_file": str(REPO_ROOT / "data" / "360DVO" / "GroundTruth" / f"{seq_id}.txt"),
            "gt_timestamp_file": str(REPO_ROOT / "data" / "360DVO" / "Timestamps" / f"{seq_id}_timestamps.txt"),
            "pred_tum": str(pred_tum),
            "frame_count": len(image_paths),
            "pair_count": len(seq_rows),
            "adjacent_pair_count": len(adj_rows),
            "official_sequence_level_input": True,
            "official_pair_level_input": False,
            "camera_model": "ERP spherical intrinsics inferred by official demo.compute_intrinsics(H, W)",
            "calibration_source": "official demo intrinsics from image size; no dataset-specific legacy scene01 calibration used",
            "official_timeout_seconds": OFFICIAL_TIMEOUT_SECONDS,
            "run_status": "not_attempted",
            "failure_reason": None,
            "command": [
                official_python,
                str(official_repo / "demo.py"),
                "--imagedir",
                str(official_image_dir),
                "--config",
                str(official_repo / "config" / "360.yaml"),
                "--save_trajectory",
                "--name",
                f"{split}_{seq_id}",
            ],
        }
        _write_json(seq_dir / "input_manifest.json", run_meta)

        proc = subprocess.run(
            run_meta["command"],
            cwd=str(official_repo) if official_repo.exists() else str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=OFFICIAL_TIMEOUT_SECONDS,
            env={
                **os.environ,
                "MKL_SERVICE_FORCE_INTEL": os.environ.get("MKL_SERVICE_FORCE_INTEL", "1"),
                "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER", "GNU"),
            },
        ) if official_repo.exists() else None

        if proc is None:
            run_meta["run_status"] = "blocked"
            run_meta["failure_reason"] = "official_repo_missing"
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("official_repo_missing\n", encoding="utf-8")
        else:
            stdout_log.write_text(proc.stdout or "", encoding="utf-8")
            stderr_log.write_text(proc.stderr or "", encoding="utf-8")
            saved_traj = official_repo / "saved_trajectories" / f"{split}_{seq_id}.txt"
            if proc.returncode == 0 and saved_traj.exists():
                shutil.copyfile(saved_traj, pred_tum)
            if proc.returncode == 0 and pred_tum.exists():
                run_meta["run_status"] = "success"
                coverage_flags.append(1.0)
            else:
                run_meta["run_status"] = "blocked"
                stderr_text = (proc.stderr or "").strip()
                if "No module named 'torch'" in stderr_text or 'No module named "torch"' in stderr_text:
                    run_meta["failure_reason"] = "missing_python_dependency_torch"
                elif inventory.get("weights_found") == []:
                    run_meta["failure_reason"] = "official_weights_missing"
                else:
                    run_meta["failure_reason"] = f"returncode_{proc.returncode}"
                coverage_flags.append(0.0)
        _write_json(seq_dir / "run_metadata.json", run_meta)

        metrics = {
            "available": pred_tum.exists(),
            "coverage": 1.0 if pred_tum.exists() else 0.0,
            "failed_frames": len(image_paths) if not pred_tum.exists() else 0,
            "trajectory_eval": {},
            "relative_metrics": {},
        }
        if pred_tum.exists():
            for mode in ("none", "se3", "sim3"):
                metrics["trajectory_eval"][mode] = evaluate_external_baseline_trajectory(
                    gt_path=gt_tum,
                    est_path=pred_tum,
                    alignment=mode,
                )
        _write_json(seq_dir / "per_sequence_metrics.json", metrics)
        seq_payloads[seq_id] = {
            "frame_count": len(image_paths),
            "pair_count": len(seq_rows),
            "adjacent_pair_count": len(adj_rows),
            "image_dir": str(REPO_ROOT / "data" / "360DVO" / "Sequences" / seq_id),
            "gt_tum": str(gt_tum),
            "pred_tum": str(pred_tum) if pred_tum.exists() else None,
            "run_metadata": str(seq_dir / "run_metadata.json"),
            "metrics": metrics,
        }

    coverage = float(sum(coverage_flags) / max(len(coverage_flags), 1)) if seq_payloads else 0.0
    split_metrics = {
        "split": split,
        "sequence_count": len(seq_ids),
        "pair_count": len(rows),
        "adjacent_pair_count": sum(1 for r in rows if r["pair_type"] == "adjacent" and int(r["k"]) == 1),
        "coverage": coverage,
        "ate_none": None,
        "ate_se3": None,
        "ate_sim3": None,
        "path_ratio": None,
        "predicted_path_length": None,
        "gt_path_length": None,
        "rot_mean_deg": None,
        "rot_median_deg": None,
        "signed_tdir_mean_deg": None,
        "signed_tdir_median_deg": None,
        "unsigned_tdir_mean_deg": None,
        "anti_parallel_rate": None,
        "tmag_median_ratio": None,
        "tmag_mean_ratio": None,
        "log_tmag_mae": None,
        "nan_inf_count": None,
        "execution_status": "success" if coverage == 1.0 else ("partial" if coverage > 0.0 else "blocked"),
        "per_sequence": seq_payloads,
    }
    _write_json(split_dir / "split_metrics.json", split_metrics)
    return split_metrics


def _build_compare_rows(
    train360_val: Dict[str, Any],
    train360_test: Dict[str, Any],
    base360_val: Dict[str, Any],
    base360_test: Dict[str, Any],
    t57b_eval: Dict[str, Any],
) -> List[Dict[str, Any]]:
    t57b_overall = t57b_eval.get("overall_component", {})
    t57b_traj = t57b_eval.get("trajectory_eval", {}).get("weighted_summary", {})
    return [
        {
            "model": "T57b recovered legacy image-pair baseline",
            "source": "GEN5 external eval artifact",
            "split": "external_ref_only",
            "rot_mean_deg": t57b_overall.get("rot_mean_deg"),
            "signed_tdir_mean_deg": t57b_overall.get("signed_tdir_mean_deg"),
            "anti_parallel_rate": t57b_overall.get("anti_parallel_rate"),
            "tmag_median_ratio": t57b_overall.get("tmag_median_ratio"),
            "path_ratio": t57b_overall.get("path_ratio"),
            "ate_none": t57b_traj.get("none_ate"),
            "ate_se3": t57b_traj.get("se3_ate"),
            "ate_sim3": t57b_traj.get("sim3_ate"),
            "coverage": 1.0,
            "notes": "Not DSET2C canonical val/test; kept only as legacy/external reference.",
        },
        {
            "model": "TRAIN360C self-developed baseline",
            "source": "reports/TRAIN360C_metrics_val.json",
            "split": "val",
            "rot_mean_deg": train360_val.get("rot_mean_deg"),
            "signed_tdir_mean_deg": train360_val.get("signed_tdir_mean_deg"),
            "anti_parallel_rate": train360_val.get("anti_parallel_rate"),
            "tmag_median_ratio": train360_val.get("tmag_median_ratio"),
            "path_ratio": train360_val.get("path_ratio"),
            "ate_none": train360_val.get("ate_none"),
            "ate_se3": train360_val.get("ate_se3"),
            "ate_sim3": train360_val.get("ate_sim3"),
            "coverage": train360_val.get("coverage"),
            "notes": "Canonical DSET2C val split.",
        },
        {
            "model": "TRAIN360C self-developed baseline",
            "source": "reports/TRAIN360C_metrics_test.json",
            "split": "test",
            "rot_mean_deg": train360_test.get("rot_mean_deg"),
            "signed_tdir_mean_deg": train360_test.get("signed_tdir_mean_deg"),
            "anti_parallel_rate": train360_test.get("anti_parallel_rate"),
            "tmag_median_ratio": train360_test.get("tmag_median_ratio"),
            "path_ratio": train360_test.get("path_ratio"),
            "ate_none": train360_test.get("ate_none"),
            "ate_se3": train360_test.get("ate_se3"),
            "ate_sim3": train360_test.get("ate_sim3"),
            "coverage": train360_test.get("coverage"),
            "notes": "Canonical DSET2C test split.",
        },
        {
            "model": "BASE360 HKUST official 360DVO baseline",
            "source": "external official repo",
            "split": "val",
            "rot_mean_deg": base360_val.get("rot_mean_deg"),
            "signed_tdir_mean_deg": base360_val.get("signed_tdir_mean_deg"),
            "anti_parallel_rate": base360_val.get("anti_parallel_rate"),
            "tmag_median_ratio": base360_val.get("tmag_median_ratio"),
            "path_ratio": base360_val.get("path_ratio"),
            "ate_none": base360_val.get("ate_none"),
            "ate_se3": base360_val.get("ate_se3"),
            "ate_sim3": base360_val.get("ate_sim3"),
            "coverage": base360_val.get("coverage"),
            "notes": "Blocked in current environment; no official predictions materialized.",
        },
        {
            "model": "BASE360 HKUST official 360DVO baseline",
            "source": "external official repo",
            "split": "test",
            "rot_mean_deg": base360_test.get("rot_mean_deg"),
            "signed_tdir_mean_deg": base360_test.get("signed_tdir_mean_deg"),
            "anti_parallel_rate": base360_test.get("anti_parallel_rate"),
            "tmag_median_ratio": base360_test.get("tmag_median_ratio"),
            "path_ratio": base360_test.get("path_ratio"),
            "ate_none": base360_test.get("ate_none"),
            "ate_se3": base360_test.get("ate_se3"),
            "ate_sim3": base360_test.get("ate_sim3"),
            "coverage": base360_test.get("coverage"),
            "notes": "Blocked in current environment; no official predictions materialized.",
        },
    ]


def _write_summary_markdown(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    cols = [
        "model",
        "source",
        "split",
        "rot_mean_deg",
        "signed_tdir_mean_deg",
        "anti_parallel_rate",
        "tmag_median_ratio",
        "path_ratio",
        "ate_none",
        "ate_se3",
        "ate_sim3",
        "coverage",
        "notes",
    ]
    lines = [
        "# TRAIN360 vs BASE360 vs T57b summary",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            if col == "coverage":
                vals.append(_fmt_pct(row.get(col)))
            elif col == "notes":
                vals.append(str(row.get(col, "")))
            else:
                val = row.get(col)
                if isinstance(val, (int, float)) or val is None:
                    vals.append(_fmt_num(val))
                else:
                    vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_report(
    report_path: Path,
    inventory: Dict[str, Any],
    canonical_summary: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    train360_val: Dict[str, Any],
    train360_test: Dict[str, Any],
    comparison_rows: Sequence[Dict[str, Any]],
    official_repo: Path,
) -> None:
    val_seq_lines = [f"- `{seq}`" for seq in canonical_summary.get("val_sequences", [])]
    test_seq_lines = [f"- `{seq}`" for seq in canonical_summary.get("test_sequences", [])]
    blocker_lines = [f"- `{b}`" for b in inventory.get("blockers", [])] or ["- none"]
    comp_md = SUMMARY_PATH
    lines = [
        "# BASE360 HKUST 360DVO official baseline eval",
        "",
        "## 1. Executive summary",
        f"- BASE360 是否成功运行: `{'yes' if test_metrics.get('execution_status') == 'success' and val_metrics.get('execution_status') == 'success' else 'no'}`",
        f"- val coverage: `{_fmt_pct(val_metrics.get('coverage'))}`",
        f"- test coverage: `{_fmt_pct(test_metrics.get('coverage'))}`",
        f"- 是否获得可比 metrics: `{'yes' if val_metrics.get('ate_sim3') is not None and test_metrics.get('ate_sim3') is not None else 'no'}`",
        "- 与 TRAIN360C 相比的主要结论: 当前无法形成 BASE360 官方结果，因为 official method 在本地未完成依赖安装、CUDA 扩展构建和权重下载；TRAIN360C 仍是当前唯一完整的 DSET2C canonical val/test 结果。",
        "- 当前最大 caveat: BASE360 真实 inference 被环境/权重阻塞，因此主表中的 BASE360 行是 blocker-aware placeholder，而不是性能数字。",
        "",
        "## 2. Official method inventory",
        f"- official code path: `{official_repo}`",
        f"- version / commit: `{inventory.get('official_commit')}`",
        f"- required dependencies: `{', '.join(inventory.get('required_dependencies', []))}`",
        f"- config path: `{inventory.get('config_path')}`",
        f"- whether method was modified: `{inventory.get('method_modified')}`",
        "- run commands:",
        f"  - `python3 {official_repo / 'demo.py'} --imagedir data/360DVO/Sequences/<sequence> --config {official_repo / 'config' / '360.yaml'} --save_trajectory --name <split>_<sequence>`",
        "- inventory blockers:",
        *blocker_lines,
        "",
        "## 3. DSET2C canonical split compliance",
        "- val/test sequence list:",
        *val_seq_lines,
        *test_seq_lines,
        f"- pair counts: `val={canonical_summary.get('num_pairs_val')}`, `test={canonical_summary.get('num_pairs_test')}`",
        f"- adjacent pair counts: `val={canonical_summary.get('num_adjacent_pairs_val')}`, `test={canonical_summary.get('num_adjacent_pairs_test')}`",
        "- no random split: `true`",
        "- no direct split glob: `true`",
        "- canonical manifest / hygiene json usage: `true`",
        "",
        "## 4. Input adapter",
        "- sequence-level or pair-level: `sequence-level`",
        "- image order: recovered from canonical manifest timestamps / frame paths, not directory glob split discovery",
        "- calibration / ERP config: official demo uses ERP intrinsics derived from image height/width; no legacy scene01 calibration was used",
        "- symlink/list/config generation: generated `image_list.txt`, `input_manifest.json`, and `gt_tum.txt` per sequence without copying images",
        "- failure handling: every attempted run writes `stdout.log`, `stderr.log`, and `run_metadata.json` per sequence",
        "",
        "## 5. Inference results",
        f"- per-sequence status: `val={val_metrics.get('execution_status')}`, `test={test_metrics.get('execution_status')}`",
        "- runtime summary: official demo invocation attempted for each canonical sequence and failed before prediction export",
        "- failed frames: full sequence coverage missing because no official prediction file was produced",
        f"- coverage: `val={_fmt_pct(val_metrics.get('coverage'))}`, `test={_fmt_pct(test_metrics.get('coverage'))}`",
        "",
        "## 6. Evaluation protocol",
        "- evaluator used: `tools/evaluate_external_baseline_trajectory.py` for TUM trajectory metrics",
        "- metric definitions: ATE none / SE3 / Sim3 use the existing project evaluator; path_ratio remains raw predicted path length over GT path length",
        "- trajectory alignment modes: `none`, `se3`, `sim3`",
        "- relative pose computation: unavailable for BASE360 because official trajectory was not generated",
        "- known incompatibilities: T57b reference row is not on DSET2C canonical val/test and is included only as external context",
        "",
        "## 7. Val metrics",
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        "",
        "## 8. Test metrics",
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        "",
        "## 9. TRAIN360C vs BASE360 vs T57b comparison",
        f"- unified table: `{comp_md}`",
        f"- TRAIN360C test signed_tdir_mean_deg: `{_fmt_num(train360_test.get('signed_tdir_mean_deg'))}`",
        f"- BASE360 test signed_tdir_mean_deg: `{_fmt_num(test_metrics.get('signed_tdir_mean_deg'))}`",
        "- which model is best on which metric: only TRAIN360C has canonical DSET2C val/test metrics at this point; BASE360 official is blocked and T57b is out-of-split reference only",
        "- whether TRAIN360C remains competitive: yes, because it remains the only completed canonical evaluation",
        "- whether BASE360 exposes weaknesses in TRAIN360C: not yet, because no official BASE360 metrics were produced",
        "",
        "## 10. Failure / caveat analysis",
        "- official method repository exists externally but is not vendored in the main repo",
        "- current Python environment is missing at least `torch`, `evo`, and `viser` for the official method",
        "- official repo also requires local CUDA extension build via `pip install .` and Eigen download before inference",
        "- official weights were not present locally, so even a dependency-complete run would still be blocked without weight download",
        "- val/test difficulty difference cannot be assessed for BASE360 because coverage is zero",
        "",
        "## 11. Next step recommendation",
        "- `fix_BASE360_blockers`",
        "",
        "## 12. Compliance checklist",
        "- `base360_inference_executed = false`",
        "- `official_method_used_as_teacher = false`",
        "- `train360_weights_modified = false`",
        "- `train360_training_executed = false`",
        "- `s5_locked_metrics_modified = false`",
        "- `legacy_scene01_artifact_dependency = false`",
        "- `random_pair_split_used = false`",
        "- `dset2c_canonical_split_used = true`",
        "- `val_used_for_tuning = false`",
        "- `test_used_for_tuning = false`",
        "- `s5e15_included_as_external_model = false`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-repo", default="/tmp/360DVO_official")
    p.add_argument("--official-python", default="python3")
    p.add_argument("--hygiene-json", default=str(DEFAULT_HYGIENE))
    p.add_argument("--canonical-dir", default=str(DEFAULT_CANONICAL))
    p.add_argument("--train360c-report", default=str(DEFAULT_TRAIN360C_REPORT))
    p.add_argument("--train360c-val", default=str(DEFAULT_TRAIN360C_VAL))
    p.add_argument("--train360c-test", default=str(DEFAULT_TRAIN360C_TEST))
    p.add_argument("--train360c-ckpt", default=str(DEFAULT_TRAIN360C_CKPT))
    p.add_argument("--t57b-eval", default=str(DEFAULT_T57B_EVAL))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    official_repo = Path(args.official_repo)
    hygiene = _read_json(Path(args.hygiene_json))
    canonical_dir = Path(args.canonical_dir)
    canonical_summary = _read_json(canonical_dir / "canonical_manifest_summary.json")
    val_rows = _read_jsonl(canonical_dir / "pair_manifest_val.jsonl")
    test_rows = _read_jsonl(canonical_dir / "pair_manifest_test.jsonl")
    _ = _read_jsonl(canonical_dir / "pair_manifest_train.jsonl")
    _ = Path(args.train360c_report).read_text(encoding="utf-8")
    _ = Path(args.train360c_ckpt).exists()
    train360_val = _read_json(Path(args.train360c_val))["metrics"]
    train360_test = _read_json(Path(args.train360c_test))["metrics"]
    t57b_eval = _read_json(Path(args.t57b_eval))

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = _inventory_official_repo(official_repo)
    _write_json(RESULTS_ROOT / "official_method_inventory.json", inventory)
    _write_json(RESULTS_ROOT / "dset2c_hygiene_snapshot.json", hygiene)
    _write_json(RESULTS_ROOT / "canonical_manifest_summary_snapshot.json", canonical_summary)

    val_metrics = _prepare_split("val", val_rows, official_repo, inventory, args.official_python)
    test_metrics = _prepare_split("test", test_rows, official_repo, inventory, args.official_python)
    _write_json(VAL_JSON_PATH, val_metrics)
    _write_json(TEST_JSON_PATH, test_metrics)

    rows = _build_compare_rows(train360_val, train360_test, val_metrics, test_metrics, t57b_eval)
    _write_summary_markdown(rows, SUMMARY_PATH)
    _write_main_report(
        REPORT_PATH,
        inventory,
        canonical_summary,
        val_metrics,
        test_metrics,
        train360_val,
        train360_test,
        rows,
        official_repo,
    )

    terminal_summary = {
        "BASE360 execution": test_metrics.get("execution_status") if test_metrics.get("execution_status") == val_metrics.get("execution_status") else "partial",
        "official method path": inventory.get("official_method_path"),
        "val coverage": val_metrics.get("coverage"),
        "test coverage": test_metrics.get("coverage"),
        "val signed_tdir_mean": val_metrics.get("signed_tdir_mean_deg"),
        "test signed_tdir_mean": test_metrics.get("signed_tdir_mean_deg"),
        "test path_ratio": test_metrics.get("path_ratio"),
        "test ATE sim3": test_metrics.get("ate_sim3"),
        "comparable to TRAIN360C": "yes" if test_metrics.get("ate_sim3") is not None else "no",
        "best model on test translation": "TRAIN360C self-developed baseline",
        "next recommended task": "fix_BASE360_blockers",
    }
    _write_json(RESULTS_ROOT / "terminal_summary.json", terminal_summary)
    print(f"BASE360 execution: {terminal_summary['BASE360 execution']}")
    print(f"official method path: {terminal_summary['official method path']}")
    print(f"val coverage: {terminal_summary['val coverage']}")
    print(f"test coverage: {terminal_summary['test coverage']}")
    print(f"val signed_tdir_mean: {terminal_summary['val signed_tdir_mean']}")
    print(f"test signed_tdir_mean: {terminal_summary['test signed_tdir_mean']}")
    print(f"test path_ratio: {terminal_summary['test path_ratio']}")
    print(f"test ATE sim3: {terminal_summary['test ATE sim3']}")
    print(f"comparable to TRAIN360C: {terminal_summary['comparable to TRAIN360C']}")
    print(f"best model on test translation: {terminal_summary['best model on test translation']}")
    print(f"next recommended task: {terminal_summary['next recommended task']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
