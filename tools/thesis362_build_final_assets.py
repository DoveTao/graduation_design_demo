#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from thesis362_export_matching_geometry_visuals import generate_matching_geometry_assets


REPORTS_DIR = REPO_ROOT / "reports"
THESIS_DIR = REPO_ROOT / "thesis" / "final_assets"
FIG_DIR = THESIS_DIR / "figures"
TABLE_DIR = THESIS_DIR / "tables"
CAPTION_DIR = THESIS_DIR / "captions"
MANIFEST_DIR = THESIS_DIR / "manifests"


def _ensure_dirs() -> None:
    for path in [
        FIG_DIR / "main_results",
        FIG_DIR / "ablations",
        FIG_DIR / "trajectories",
        FIG_DIR / "diagnostics",
        FIG_DIR / "training_curves",
        FIG_DIR / "matching_geometry",
        TABLE_DIR,
        CAPTION_DIR,
        MANIFEST_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _read_jsonl_maybe(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path) if path.exists() else []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _fmt(value: Any, digits: int = 6) -> str:
    x = _safe_float(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def _maybe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_md(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_best_train_log_row(path: Path, score_key: str) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    return min(rows, key=lambda row: float(row[score_key]))


def _parse_train_log_rows(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


def _find_named_metrics_anywhere(payload: Any, target_name: str) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        if payload.get("name") == target_name and isinstance(payload.get("metrics"), dict):
            return payload["metrics"]
        for value in payload.values():
            found = _find_named_metrics_anywhere(value, target_name)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_named_metrics_anywhere(item, target_name)
            if found is not None:
                return found
    return None


def _build_pair_sources() -> Dict[str, Dict[str, Any]]:
    main_results = _read_json(REPORTS_DIR / "RESULTS360_main_results_table.json")
    final360i = _read_json(REPORTS_DIR / "FINAL360I_metrics_test.json")["metrics"]
    train360e = _read_json(REPORTS_DIR / "TRAIN360E_metrics_test.json")
    seq360b_pair = _read_json(REPORTS_DIR / "SEQ360B_metrics_test.json")["metrics"]
    base_component_report = _read_json(REPORTS_DIR / "BASE360D_metrics_test.json")

    component_table = main_results["metrics"]["component_table"]
    trajectory_table = main_results["metrics"]["trajectory_table"]

    def _find_component(model_substr: str, split: str = "test") -> Dict[str, Any]:
        for row in component_table:
            if model_substr in row["model"] and row["split"] == split:
                return row
        raise KeyError(model_substr)

    def _find_traj(model_substr: str, split: str = "test") -> Dict[str, Any]:
        for row in trajectory_table:
            if model_substr in row["model"] and row["split"] == split:
                return row
        raise KeyError(model_substr)

    train360c_test = _find_component("TRAIN360C")
    recovered_train360c = _find_named_metrics_anywhere(train360e, "TRAIN360C")
    if recovered_train360c is not None:
        train360c_test = dict(recovered_train360c)
    t57b_component = _find_component("T57b", split="external_ref_only")
    base360d_component = _find_component("BASE360D")
    base360d_traj = _find_traj("BASE360D")

    return {
        "FINAL360I": {
            "metrics": final360i,
            "split": "test",
            "source_files": [_maybe_rel(REPORTS_DIR / "FINAL360I_metrics_test.json")],
            "notes": "Current thesis main pair-level model.",
        },
        "TRAIN360C": {
            "metrics": train360c_test,
            "split": "test",
            "source_files": [_maybe_rel(REPORTS_DIR / "RESULTS360_main_results_table.json"), _maybe_rel(REPORTS_DIR / "TRAIN360E_metrics_test.json")],
            "notes": "Retained self-developed baseline recovered from aggregate reports because standalone TRAIN360C json is absent in the current worktree.",
        },
        "T57b": {
            "metrics": dict(t57b_component),
            "split": "reference",
            "source_files": [_maybe_rel(REPORTS_DIR / "RESULTS360_main_results_table.json"), _maybe_rel(REPORTS_DIR / "TRAIN360E_metrics_test.json")],
            "notes": "Recovered legacy external ERP baseline reference; rot_mean_deg unavailable in retained sources.",
        },
        "BASE360D": {
            "metrics": {
                "rot_mean_deg": base_component_report["pair_metrics"]["all_pairs"].get("rot_mean_deg"),
                "signed_tdir_mean_deg": base_component_report["pair_metrics"]["all_pairs"].get("signed_tdir_mean_deg"),
                "anti_parallel_rate": base_component_report["pair_metrics"]["all_pairs"].get("anti_parallel_rate"),
                "tmag_median_ratio": base_component_report["pair_metrics"]["all_pairs"].get("tmag_median_ratio"),
                "path_ratio": base_component_report["pair_metrics"]["all_pairs"].get("pair_component_path_ratio"),
                "coverage": base_component_report["pair_metrics"]["all_pairs"].get("pair_coverage"),
            },
            "trajectory_metrics": base360d_traj,
            "split": "test",
            "source_files": [_maybe_rel(REPORTS_DIR / "BASE360D_metrics_test.json"), _maybe_rel(REPORTS_DIR / "RESULTS360_main_results_table.json")],
            "notes": "Official 360DVO external baseline; pair-level component metrics are trajectory-derived proxies, not native pair-forward outputs.",
        },
        "TRAIN360E": {
            "metrics": train360e,
            "split": "test",
            "source_files": [_maybe_rel(REPORTS_DIR / "TRAIN360E_metrics_test.json")],
            "notes": "Adjacent-pair composition of FINAL360I into full trajectories.",
        },
        "SEQ360B": {
            "pair_metrics": seq360b_pair,
            "trajectory_metrics": _read_json(REPORTS_DIR / "SEQ360B_trajectory_metrics_test.json"),
            "split": "test",
            "source_files": [_maybe_rel(REPORTS_DIR / "SEQ360B_metrics_test.json"), _maybe_rel(REPORTS_DIR / "SEQ360B_trajectory_metrics_test.json")],
            "notes": "Sequence-scale variant; pair metrics keep FINAL360I rotation and direction while correcting magnitude.",
        },
    }


def _build_ablation_sources() -> Dict[str, Dict[str, Any]]:
    abldvo2 = _read_json(REPORTS_DIR / "ABLDVO2_metrics_test.json")
    abldvo3 = _read_json(REPORTS_DIR / "ABLDVO3_metrics_test.json")
    return {
        "PlainPairVO": {
            "metrics": abldvo2["models"]["ABLDVO2_PlainPairVO"]["metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json")],
        },
        "NoSphericalGeometry": {
            "metrics": abldvo2["models"]["ABLDVO2_NoSphericalGeometry"]["metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json")],
        },
        "NoCrossImageInteraction": {
            "metrics": abldvo2["models"]["ABLDVO2_NoCrossImageInteraction"]["metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json")],
        },
        "SingleStagePoseRegression": {
            "metrics": abldvo3["models"]["ABLDVO3_SingleStagePoseRegression"]["metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO3_metrics_test.json")],
        },
        "FINAL360I": {
            "metrics": abldvo2["final360i_reference"],
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json"), _maybe_rel(REPORTS_DIR / "ABLDVO3_metrics_test.json")],
        },
    }


def _load_tum(path: Path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        ts, tx, ty, tz, qx, qy, qz, qw = [float(x) for x in text.split()]
        q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1.0e-12)
        x, y, z, w = q
        R = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        rows.append((ts, R, np.asarray([tx, ty, tz], dtype=np.float64)))
    return rows


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


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
    alignment: str,
) -> Dict[str, Any]:
    gt_pts = np.asarray([row[2] for row in gt_rows], dtype=np.float64)
    pred_pts_raw = np.asarray([row[2] for row in pred_rows], dtype=np.float64)
    if alignment == "none":
        scale, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
    elif alignment == "se3":
        scale, R_align, t_align = _umeyama(pred_pts_raw.T, gt_pts.T, with_scale=False)
    elif alignment == "sim3":
        scale, R_align, t_align = _umeyama(pred_pts_raw.T, gt_pts.T, with_scale=True)
    else:
        raise ValueError(alignment)
    pred_pts = _apply_alignment(pred_pts_raw, scale, R_align, t_align)
    errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "trajectory_path_ratio": float(_path_length(pred_pts_raw) / max(_path_length(gt_pts), 1.0e-12)),
        "pred_points": pred_pts_raw,
        "gt_points": gt_pts,
    }


def _aggregate_tum_variant_dir(variant_dir: Path) -> Dict[str, Any]:
    per_sequence: Dict[str, Dict[str, Any]] = {}
    pred_total = 0.0
    gt_total = 0.0
    none_sq: List[float] = []
    se3_sq: List[float] = []
    sim3_sq: List[float] = []
    for seq_dir in sorted(variant_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        pred_path = seq_dir / "pred_tum.txt"
        gt_path = seq_dir / "gt_tum.txt"
        if not pred_path.exists() or not gt_path.exists():
            continue
        gt_rows = _load_tum(gt_path)
        pred_rows = _load_tum(pred_path)
        none = _evaluate_pose_errors(gt_rows, pred_rows, "none")
        se3 = _evaluate_pose_errors(gt_rows, pred_rows, "se3")
        sim3 = _evaluate_pose_errors(gt_rows, pred_rows, "sim3")
        per_sequence[seq_dir.name] = {"none": none, "se3": se3, "sim3": sim3}
        pred_total += _path_length(np.asarray([row[2] for row in pred_rows], dtype=np.float64))
        gt_total += _path_length(np.asarray([row[2] for row in gt_rows], dtype=np.float64))
        none_sq.extend(np.linalg.norm(none["pred_points"] - none["gt_points"], axis=1).tolist())
        se3_pred = _apply_alignment(none["pred_points"], *_umeyama(none["pred_points"].T, none["gt_points"].T, with_scale=False))
        sim3_pred = _apply_alignment(none["pred_points"], *_umeyama(none["pred_points"].T, none["gt_points"].T, with_scale=True))
        se3_sq.extend(np.linalg.norm(se3_pred - none["gt_points"], axis=1).tolist())
        sim3_sq.extend(np.linalg.norm(sim3_pred - none["gt_points"], axis=1).tolist())
    def _rmse(values: Sequence[float]) -> Optional[float]:
        return float(np.sqrt(np.mean(np.square(values)))) if values else None
    return {
        "per_sequence": per_sequence,
        "ate_none": {"rmse": _rmse(none_sq)},
        "ate_se3": {"rmse": _rmse(se3_sq)},
        "ate_sim3": {"rmse": _rmse(sim3_sq)},
        "trajectory_path_ratio": float(pred_total / max(gt_total, 1.0e-12)) if gt_total > 0 else None,
        "pred_path_length": pred_total,
        "gt_path_length": gt_total,
        "coverage": 1.0 if per_sequence else None,
    }


def _build_trajectory_sources(pair_sources: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    odom360a_root = REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "local_window_pose_fusion"
    odom360b_root = REPO_ROOT / "external_baselines" / "results" / "odom360b_local_pose_graph_kstep" / "test" / "pg_k3"
    base360d_report = pair_sources["BASE360D"]["trajectory_metrics"]
    base360d_metrics = {
        "ate_none": {"rmse": base360d_report.get("ate_none")},
        "ate_se3": {"rmse": base360d_report.get("ate_se3")},
        "ate_sim3": {"rmse": base360d_report.get("ate_sim3")},
        "trajectory_path_ratio": base360d_report.get("trajectory_path_ratio") or base360d_report.get("path_ratio"),
        "pred_path_length": base360d_report.get("pred_path_length"),
        "gt_path_length": base360d_report.get("gt_path_length"),
        "coverage": base360d_report.get("pose_coverage") or base360d_report.get("coverage"),
    }
    return {
        "TRAIN360E": {
            "metrics": pair_sources["TRAIN360E"]["metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "TRAIN360E_metrics_test.json")],
            "notes": "Direct adjacent-pair composition of FINAL360I.",
        },
        "SEQ360B": {
            "metrics": pair_sources["SEQ360B"]["trajectory_metrics"],
            "source_files": [_maybe_rel(REPORTS_DIR / "SEQ360B_trajectory_metrics_test.json")],
            "notes": "Scale/log-scale smoothing improves path ratio and SE3 ATE, but not Sim3.",
        },
        "ODOM360A": {
            "metrics": _aggregate_tum_variant_dir(odom360a_root),
            "source_files": [_maybe_rel(odom360a_root.parent)],
            "notes": "Computed from existing local_window_pose_fusion TUM exports; no model re-evaluation executed.",
        },
        "ODOM360B": {
            "metrics": _aggregate_tum_variant_dir(odom360b_root),
            "source_files": [_maybe_rel(odom360b_root.parent)],
            "notes": "Computed from retained pg_k3 pose-graph TUM exports because standalone ODOM360B metrics json is unavailable in the current worktree.",
        },
        "BASE360D": {
            "metrics": base360d_metrics,
            "source_files": [_maybe_rel(REPORTS_DIR / "BASE360D_metrics_test.json")],
            "notes": "Official 360DVO trajectory baseline.",
        },
    }


def _build_negative_sources() -> Dict[str, Dict[str, Any]]:
    final360i_log = _parse_train_log_rows(REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "train_log.jsonl")
    final360i_best = min(final360i_log, key=lambda row: float(row["val_score"]))
    j_best = _load_best_train_log_row(REPO_ROOT / "checkpoints" / "FINAL360J_tdir_loss_reweight" / "train_log.jsonl", "val_score")
    k_best = _load_best_train_log_row(REPO_ROOT / "checkpoints" / "FINAL360K_conservative_observability_weighting" / "train_log.jsonl", "val_score")
    l_best = _load_best_train_log_row(REPO_ROOT / "checkpoints" / "FINAL360L_translation_head_factorization" / "train_log.jsonl", "val_score")
    return {
        "FINAL360I": {"split": "val", "metrics": final360i_best["val_final_metrics"], "status": "reference"},
        "FINAL360J": {"split": "val", "metrics": j_best["val_metrics"], "status": "regression"},
        "FINAL360K": {"split": "val", "metrics": k_best["val_metrics"], "status": "regression"},
        "FINAL360L": {"split": "val", "metrics": l_best["val_metrics"], "status": "inconclusive"},
        "SEQ360A": {"split": None, "metrics": {}, "status": "sequence consistency no_improvement"},
        "STRUCT360C": {"split": None, "metrics": {}, "status": "trained but evaluation_failed"},
        "DATA360A": {"split": "test/diagnostic", "metrics": {}, "status": "bucket diagnostic derived from retained jsonl rows"},
    }


def _plot_metric_bars(
    labels: Sequence[str],
    metric_names: Sequence[str],
    value_lookup: Mapping[str, Mapping[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_names):
        vals = []
        xs = []
        for label in labels:
            v = _safe_float(value_lookup[label].get(metric))
            if v is not None:
                xs.append(label)
                vals.append(v)
        ax.bar(xs, vals, color=["#184e77", "#1f7a8c", "#76c893", "#f4a261", "#e76f51"][: len(xs)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_training_curves(log_rows: Sequence[Dict[str, Any]], out_path: Path, title: str, is_final360i: bool) -> None:
    epochs = [int(row["epoch"]) for row in log_rows]
    if is_final360i:
        metric_rows = [row["val_final_metrics"] for row in log_rows]
        joint = [float(row["val_score"]) for row in log_rows]
        train_loss = [float(row["train_loss_total"]) for row in log_rows]
    else:
        metric_rows = [row.get("val_metrics", row.get("minival_metrics", {})) for row in log_rows]
        joint = [float(row.get("val_score", row.get("minival_score", float("nan")))) for row in log_rows]
        train_loss = [float(row.get("train_loss_total", float("nan"))) for row in log_rows]
    rot = [_safe_float(m.get("rot_mean_deg")) for m in metric_rows]
    tdir = [_safe_float(m.get("signed_tdir_mean_deg")) for m in metric_rows]
    path = [_safe_float(m.get("path_ratio")) for m in metric_rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    series = [
        (axes[0, 0], rot, "val rot_mean_deg"),
        (axes[0, 1], tdir, "val signed_tdir_mean_deg"),
        (axes[1, 0], joint, "joint/minival score"),
        (axes[1, 1], train_loss, "train loss"),
    ]
    for ax, values, subtitle in series:
        ax.plot(epochs, values, marker="o", linewidth=2.0, color="#1d3557")
        ax.set_title(subtitle)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[1, 0].plot(epochs, path, marker="s", linewidth=1.4, color="#e76f51", label="path_ratio")
    axes[1, 0].legend()
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_joint_vs_single(log_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in log_rows]
    tdir = [float(row["val_final_metrics"]["signed_tdir_mean_deg"]) for row in log_rows]
    joint = [float(row["val_score"]) for row in log_rows]
    best_joint_epoch = epochs[int(np.argmin(joint))]
    best_tdir_epoch = epochs[int(np.argmin(tdir))]
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax1.plot(epochs, joint, marker="o", color="#1d3557", label="joint score")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("joint score")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, tdir, marker="s", color="#d62828", label="signed_tdir_mean_deg")
    ax2.set_ylabel("signed_tdir_mean_deg")
    ax1.axvline(best_joint_epoch, color="#1d3557", linestyle="--", alpha=0.7)
    ax1.axvline(best_tdir_epoch, color="#d62828", linestyle="--", alpha=0.7)
    ax1.set_title("FINAL360I: Joint-score Best vs Single-metric Best")
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _bucketize(rows: Sequence[Dict[str, Any]], bucket_edges: Sequence[float], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        bucket_rows = [row for row in rows if lo <= float(row[key]) < hi]
        if not bucket_rows:
            out.append({"label": f"[{lo:.2f},{hi:.2f})", "mean": None, "count": 0, "anti_parallel_rate": None})
            continue
        out.append(
            {
                "label": f"[{lo:.2f},{hi:.2f})",
                "mean": float(np.mean([float(r["signed_tdir_error_deg"]) for r in bucket_rows])),
                "count": len(bucket_rows),
                "anti_parallel_rate": float(np.mean([1.0 if r.get("anti_parallel") else 0.0 for r in bucket_rows])),
            }
        )
    return out


def _plot_bucket_compare(rows_a: Sequence[Dict[str, Any]], rows_b: Optional[Sequence[Dict[str, Any]]], key: str, out_path: Path, title: str) -> List[Dict[str, Any]]:
    finite_a = [row for row in rows_a if row.get("finite", True)]
    values = [float(row[key]) for row in finite_a]
    q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    q[-1] = float(q[-1]) + 1.0e-9
    buckets_a = _bucketize(finite_a, q, key)
    buckets_b = _bucketize(rows_b or [], q, key) if rows_b else []
    labels = [b["label"] for b in buckets_a]
    xa = np.arange(len(labels))
    width = 0.36 if rows_b else 0.6
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.bar(xa - (width / 2 if rows_b else 0.0), [b["mean"] or 0.0 for b in buckets_a], width=width, label="FINAL360I", color="#1d3557")
    if rows_b:
        ax.bar(xa + width / 2, [b["mean"] or 0.0 for b in buckets_b], width=width, label="SEQ360B", color="#e76f51")
    ax.set_xticks(xa)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("mean signed_tdir_error_deg")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return [{"final360i": a, "seq360b": b if rows_b else None} for a, b in zip(buckets_a, buckets_b or buckets_a)]


def _plot_negative_compare(negative_sources: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    labels = ["FINAL360I", "FINAL360J", "FINAL360K", "FINAL360L"]
    metrics = ["signed_tdir_mean_deg", "anti_parallel_rate", "path_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        vals = [float(negative_sources[label]["metrics"].get(metric)) for label in labels]
        ax.bar(labels, vals, color=["#1d3557", "#f4a261", "#e76f51", "#2a9d8f"])
        ax.set_title(f"val {metric}")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Validation-only comparison of FINAL360I / J / K / L", fontsize=13)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _trajectory_overlay(per_model_sequence: Mapping[str, np.ndarray], out_path: Path, title: str) -> None:
    gt = per_model_sequence["GT"]
    centered = gt - gt.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    colors = {"GT": "#111111", "TRAIN360E": "#1d3557", "SEQ360B": "#e76f51", "ODOM360A": "#2a9d8f", "BASE360D": "#8d99ae"}
    for name, pts in per_model_sequence.items():
        pts2 = (pts - gt.mean(axis=0, keepdims=True)) @ basis
        ax.plot(pts2[:, 0], pts2[:, 1], label=name, linewidth=2.0 if name == "GT" else 1.8, color=colors.get(name))
    ax.set_title(title)
    ax.set_xlabel("PCA axis 1")
    ax.set_ylabel("PCA axis 2")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _trajectory_metric_bar(trajectory_sources: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    labels = ["TRAIN360E", "SEQ360B", "ODOM360A", "ODOM360B", "BASE360D"]
    metrics = ["ate_none", "ate_se3", "ate_sim3", "trajectory_path_ratio"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        vals = []
        for label in labels:
            payload = trajectory_sources[label]["metrics"]
            if metric == "trajectory_path_ratio":
                vals.append(float(payload.get(metric)))
            else:
                vals.append(float(payload[metric]["rmse"]))
        ax.bar(labels, vals, color=["#1d3557", "#e76f51", "#2a9d8f", "#f4a261", "#8d99ae"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Trajectory-level metric comparison", fontsize=13)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _build_tables(
    pair_sources: Mapping[str, Dict[str, Any]],
    ablation_sources: Mapping[str, Dict[str, Any]],
    trajectory_sources: Mapping[str, Dict[str, Any]],
    negative_sources: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    table_manifest: List[Dict[str, Any]] = []

    pair_path = TABLE_DIR / "final_main_pair_results.md"
    pair_lines = [
        "# Final Pair-level Main Results",
        "",
        "| Model | split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in ["T57b", "TRAIN360C", "FINAL360I", "BASE360D"]:
        payload = pair_sources[model]
        m = payload["metrics"]
        pair_lines.append(
            f"| {model} | {payload['split']} | {_fmt(m.get('rot_mean_deg'))} | {_fmt(m.get('signed_tdir_mean_deg'))} | "
            f"{_fmt(m.get('anti_parallel_rate'))} | {_fmt(m.get('tmag_median_ratio'))} | {_fmt(m.get('path_ratio'))} | "
            f"{_fmt(m.get('coverage'))} | {payload['notes']} |"
        )
    pair_lines.extend(
        [
            "",
            "Table note: `BASE360D` pair-level component metrics are trajectory-derived proxies from the official 360DVO pipeline and therefore only partially comparable with native pair-forward outputs.",
            "Table note: `T57b` rot_mean_deg is unavailable in the retained cleaned source set and is marked as `N/A` rather than reconstructed.",
        ]
    )
    _write_md(pair_path, pair_lines)
    table_manifest.append(
        {
            "file_path": _maybe_rel(pair_path),
            "title_zh": "最终 pair-level 主结果表",
            "title_en": "Final pair-level main results",
            "source_files": sum([pair_sources[m]["source_files"] for m in ["T57b", "TRAIN360C", "FINAL360I", "BASE360D"]], []),
            "recommended_location": "thesis_main",
            "notes": "Includes BASE360D caveat and retained-source missing-value disclosure.",
        }
    )

    ablation_path = TABLE_DIR / "final_ablation_results.md"
    ablation_lines = [
        "# Final Structural Ablation Results",
        "",
        "| Model | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in ["PlainPairVO", "NoSphericalGeometry", "NoCrossImageInteraction", "SingleStagePoseRegression", "FINAL360I"]:
        m = ablation_sources[model]["metrics"]
        ablation_lines.append(
            f"| {model} | {_fmt(m.get('rot_mean_deg'))} | {_fmt(m.get('signed_tdir_mean_deg'))} | {_fmt(m.get('anti_parallel_rate'))} | "
            f"{_fmt(m.get('tmag_median_ratio'))} | {_fmt(m.get('path_ratio'))} | {_fmt(m.get('coverage'))} | "
            f"{', '.join(ablation_sources[model]['source_files'])} |"
        )
    ablation_lines.append("")
    ablation_lines.append("Table note: This core thesis ablation table merges the retained `ABLDVO2` module ablations and the `ABLDVO3` single-stage regression ablation.")
    _write_md(ablation_path, ablation_lines)
    table_manifest.append(
        {
            "file_path": _maybe_rel(ablation_path),
            "title_zh": "大模块结构消融表",
            "title_en": "Structural ablation table",
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json"), _maybe_rel(REPORTS_DIR / "ABLDVO3_metrics_test.json")],
            "recommended_location": "thesis_main",
            "notes": "Merged retained ABLDVO2 + ABLDVO3 results.",
        }
    )

    traj_path = TABLE_DIR / "final_trajectory_results.md"
    traj_lines = [
        "# Final Trajectory-level Results",
        "",
        "| Model | ATE none | ATE SE3 | ATE Sim3 | trajectory_path_ratio | coverage | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in ["TRAIN360E", "SEQ360B", "ODOM360A", "ODOM360B", "BASE360D"]:
        m = trajectory_sources[model]["metrics"]
        traj_lines.append(
            f"| {model} | {_fmt(m['ate_none']['rmse'])} | {_fmt(m['ate_se3']['rmse'])} | {_fmt(m['ate_sim3']['rmse'])} | "
            f"{_fmt(m.get('trajectory_path_ratio'))} | {_fmt(m.get('coverage') or m.get('pose_coverage') or m.get('pair_coverage'))} | {trajectory_sources[model]['notes']} |"
        )
    traj_lines.extend(
        [
            "",
            "Table note: `ODOM360A` is the strongest retained sequence backend among locally available export-only artifacts.",
            "Table note: `ODOM360B` is reported from retained pose-graph export files because no standalone `reports/ODOM360B_metrics_test.json` is present in the current worktree.",
            "Table note: `SEQ360B` improves path ratio and SE3 ATE but not Sim3 ATE, which suggests trajectory-shape error remains unresolved.",
        ]
    )
    _write_md(traj_path, traj_lines)
    table_manifest.append(
        {
            "file_path": _maybe_rel(traj_path),
            "title_zh": "trajectory-level 结果表",
            "title_en": "Trajectory-level results",
            "source_files": sum([trajectory_sources[m]["source_files"] for m in ["TRAIN360E", "SEQ360B", "ODOM360A", "ODOM360B", "BASE360D"]], []),
            "recommended_location": "thesis_main",
            "notes": "Includes export-derived ODOM360A/B metrics with provenance notes.",
        }
    )

    neg_path = TABLE_DIR / "final_negative_results_appendix.md"
    neg_lines = [
        "# Final Negative / Appendix Results",
        "",
        "| Model | available split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | status | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    notes = {
        "FINAL360J": "Loss reweighting and anti-parallel stabilization did not surpass FINAL360I on retained validation logs.",
        "FINAL360K": "Conservative observability weighting remained a regression relative to FINAL360I on retained validation logs.",
        "FINAL360L": "Translation head factorization stayed inconclusive and was stopped early for stability reasons.",
        "SEQ360A": "Sequence consistency direction did not show clear improvement and was not kept as active codepath.",
        "STRUCT360C": "Rotation-aware refinement was trained but evaluation failed, so it was not promoted.",
        "DATA360A": "Diagnostic-only bucket analysis indicates translation direction remains the dominant bottleneck.",
    }
    for model in ["FINAL360J", "FINAL360K", "FINAL360L", "SEQ360A", "STRUCT360C", "DATA360A"]:
        payload = negative_sources[model]
        m = payload["metrics"]
        neg_lines.append(
            f"| {model} | {payload.get('split') or 'N/A'} | {_fmt(m.get('rot_mean_deg'))} | {_fmt(m.get('signed_tdir_mean_deg'))} | "
            f"{_fmt(m.get('anti_parallel_rate'))} | {_fmt(m.get('tmag_median_ratio'))} | {_fmt(m.get('path_ratio'))} | "
            f"{payload['status']} | {notes[model]} |"
        )
    neg_lines.append("")
    neg_lines.append("Table note: `FINAL360J/K/L` are validation-only summaries extracted from retained train logs because standalone test metric json files are unavailable in the cleaned current worktree.")
    _write_md(neg_path, neg_lines)
    table_manifest.append(
        {
            "file_path": _maybe_rel(neg_path),
            "title_zh": "负结果与附录表",
            "title_en": "Negative and appendix results",
            "source_files": [
                _maybe_rel(REPO_ROOT / "checkpoints" / "FINAL360J_tdir_loss_reweight" / "train_log.jsonl"),
                _maybe_rel(REPO_ROOT / "checkpoints" / "FINAL360K_conservative_observability_weighting" / "train_log.jsonl"),
                _maybe_rel(REPO_ROOT / "checkpoints" / "FINAL360L_translation_head_factorization" / "train_log.jsonl"),
                _maybe_rel(REPORTS_DIR / "SEQ360A_status_summary.md"),
                _maybe_rel(REPORTS_DIR / "STRUCT360C_status_summary.md"),
                _maybe_rel(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic"),
            ],
            "recommended_location": "thesis_appendix",
            "notes": "Validation-only negative branch summary plus diagnostic statuses.",
        }
    )
    return table_manifest


def _build_figures(
    pair_sources: Mapping[str, Dict[str, Any]],
    ablation_sources: Mapping[str, Dict[str, Any]],
    trajectory_sources: Mapping[str, Dict[str, Any]],
    negative_sources: Mapping[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fig_manifest: List[Dict[str, Any]] = []
    debug_report: Dict[str, Any] = {}

    pair_fig = FIG_DIR / "main_results" / "pair_level_main_results_bar.png"
    pair_vals = {k: v["metrics"] for k, v in pair_sources.items() if "metrics" in v}
    _plot_metric_bars(["T57b", "TRAIN360C", "FINAL360I", "BASE360D"], ["rot_mean_deg", "signed_tdir_mean_deg", "anti_parallel_rate", "tmag_median_ratio", "path_ratio"], pair_vals, pair_fig, "Pair-level main result comparison")
    fig_manifest.append(
        {
            "file_path": _maybe_rel(pair_fig),
            "title_zh": "pair-level 主结果条形图",
            "title_en": "Pair-level main result bar chart",
            "source_files": sum([pair_sources[m]["source_files"] for m in ["T57b", "TRAIN360C", "FINAL360I", "BASE360D"]], []),
            "recommended_location": "thesis_main",
            "notes": "T57b rot_mean bar is omitted because the cleaned source set does not retain that value.",
        }
    )

    ablation_fig = FIG_DIR / "ablations" / "structural_ablation_multimetric.png"
    _plot_metric_bars(["PlainPairVO", "NoSphericalGeometry", "NoCrossImageInteraction", "SingleStagePoseRegression", "FINAL360I"], ["rot_mean_deg", "signed_tdir_mean_deg", "anti_parallel_rate", "tmag_median_ratio", "path_ratio"], {k: v["metrics"] for k, v in ablation_sources.items()}, ablation_fig, "Structural ablation multi-metric comparison")
    fig_manifest.append(
        {
            "file_path": _maybe_rel(ablation_fig),
            "title_zh": "结构消融多指标图",
            "title_en": "Structural ablation multi-metric chart",
            "source_files": [_maybe_rel(REPORTS_DIR / "ABLDVO2_metrics_test.json"), _maybe_rel(REPORTS_DIR / "ABLDVO3_metrics_test.json")],
            "recommended_location": "thesis_main",
            "notes": "Covers rotation, translation direction, anti-parallel rate, scale ratio, and path ratio together.",
        }
    )

    traj_metric_fig = FIG_DIR / "trajectories" / "trajectory_metric_comparison.png"
    _trajectory_metric_bar(trajectory_sources, traj_metric_fig)
    fig_manifest.append(
        {
            "file_path": _maybe_rel(traj_metric_fig),
            "title_zh": "trajectory ATE 与 path_ratio 对比图",
            "title_en": "Trajectory ATE and path-ratio comparison",
            "source_files": sum([trajectory_sources[m]["source_files"] for m in trajectory_sources], []),
            "recommended_location": "thesis_main",
            "notes": "ODOM360A/B values are computed from existing TUM exports only.",
        }
    )

    seq_name = "ridge_to_lake"
    odom360a_local_dir = REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "local_window_pose_fusion" / seq_name
    odom360a_direct_dir = REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "direct_composition" / seq_name
    odom360b_pg_dir = REPO_ROOT / "external_baselines" / "results" / "odom360b_local_pose_graph_kstep" / "test" / "pg_k3" / seq_name
    base360_seq_dir = REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official" / "test" / seq_name
    overlay_inputs = {
        "GT": np.asarray([row[2] for row in _load_tum(odom360a_local_dir / "gt_tum.txt")], dtype=np.float64),
        "ODOM360A-direct": np.asarray([row[2] for row in _load_tum(odom360a_direct_dir / "pred_tum.txt")], dtype=np.float64),
        "ODOM360A": np.asarray([row[2] for row in _load_tum(odom360a_local_dir / "pred_tum.txt")], dtype=np.float64),
        "ODOM360B": np.asarray([row[2] for row in _load_tum(odom360b_pg_dir / "pred_tum.txt")], dtype=np.float64),
        "BASE360D": np.asarray([row[2] for row in _load_tum(base360_seq_dir / "pred_tum.txt")], dtype=np.float64),
    }
    overlay_fig = FIG_DIR / "trajectories" / f"trajectory_overlay_{seq_name}.png"
    _trajectory_overlay(overlay_inputs, overlay_fig, f"Representative trajectory overlay from retained exports: {seq_name}")
    fig_manifest.append(
        {
            "file_path": _maybe_rel(overlay_fig),
            "title_zh": "代表性序列轨迹叠加图",
            "title_en": "Representative trajectory overlay",
            "source_files": [
                _maybe_rel(REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion"),
                _maybe_rel(REPO_ROOT / "external_baselines" / "results" / "odom360b_local_pose_graph_kstep"),
                _maybe_rel(REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official"),
            ],
            "recommended_location": "defense_ppt",
            "notes": f"Uses `{seq_name}` and only retained TUM exports that still exist in the cleaned worktree; direct TRAIN360E/SEQ360B TUM files are unavailable, so the overlay emphasizes export-available sequence backends plus a direct-composition reference.",
        }
    )

    final360i_log = _parse_train_log_rows(REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "train_log.jsonl")
    no_cross_log = _parse_train_log_rows(REPO_ROOT / "checkpoints" / "ABLDVO2_NoCrossImageInteraction" / "train_log.jsonl")
    single_stage_log = _parse_train_log_rows(REPO_ROOT / "checkpoints" / "ABLDVO3_SingleStagePoseRegression" / "train_log.jsonl")
    final_curve_fig = FIG_DIR / "training_curves" / "final360i_validation_curves.png"
    no_cross_curve_fig = FIG_DIR / "training_curves" / "no_cross_interaction_validation_curves.png"
    single_stage_curve_fig = FIG_DIR / "training_curves" / "single_stage_validation_curves.png"
    joint_vs_single_fig = FIG_DIR / "training_curves" / "final360i_joint_vs_single_best.png"
    _plot_training_curves(final360i_log, final_curve_fig, "FINAL360I validation curves", True)
    _plot_training_curves(no_cross_log, no_cross_curve_fig, "NoCrossImageInteraction minival curves", False)
    _plot_training_curves(single_stage_log, single_stage_curve_fig, "SingleStagePoseRegression minival curves", False)
    _plot_joint_vs_single(final360i_log, joint_vs_single_fig)
    for file_path, title_zh, title_en, notes in [
        (final_curve_fig, "FINAL360I 验证曲线", "FINAL360I validation curves", "Validation curves extracted from retained train_log.jsonl."),
        (no_cross_curve_fig, "NoCrossInteraction 验证曲线", "NoCrossInteraction validation curves", "Representative ablation curve from retained minival logs."),
        (single_stage_curve_fig, "SingleStage 验证曲线", "SingleStage validation curves", "Representative ablation curve from retained minival logs."),
        (joint_vs_single_fig, "joint score 与单指标 best 对比图", "Joint-score vs single-metric best", "Compares checkpoint selection by joint score against minimum signed_tdir epoch."),
    ]:
        fig_manifest.append(
            {
                "file_path": _maybe_rel(file_path),
                "title_zh": title_zh,
                "title_en": title_en,
                "source_files": [_maybe_rel(REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "train_log.jsonl")],
                "recommended_location": "thesis_appendix" if "joint" not in file_path.name else "defense_ppt",
                "notes": notes,
            }
        )

    final360i_rows = _read_jsonl(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic" / "final360i_test_pair_rows.jsonl")
    seq360b_rows = _read_jsonl(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic" / "seq360b_test_adjacent_rows.jsonl")
    tmag_bucket_fig = FIG_DIR / "diagnostics" / "data360a_tmag_bucket_signed_tdir.png"
    rot_bucket_fig = FIG_DIR / "diagnostics" / "data360a_rotation_bucket_signed_tdir.png"
    seq_compare_fig = FIG_DIR / "diagnostics" / "final360i_vs_seq360b_bucket_compare.png"
    neg_compare_fig = FIG_DIR / "diagnostics" / "final360i_j_k_l_validation_compare.png"
    _plot_bucket_compare(final360i_rows, None, "gt_tmag", tmag_bucket_fig, "FINAL360I signed_tdir by gt_tmag bucket")
    _plot_bucket_compare(final360i_rows, None, "gt_rot_angle_deg", rot_bucket_fig, "FINAL360I signed_tdir by gt rotation bucket")
    _plot_bucket_compare([r for r in final360i_rows if r["pair_type"] == "adjacent"], seq360b_rows, "gt_tmag", seq_compare_fig, "FINAL360I vs SEQ360B by gt_tmag bucket")
    _plot_negative_compare(negative_sources, neg_compare_fig)
    anti_parallel_fig = FIG_DIR / "diagnostics" / "anti_parallel_regime_distribution.png"
    anti_vals = [float(row["signed_tdir_error_deg"]) for row in final360i_rows if row.get("anti_parallel")]
    non_anti_vals = [float(row["signed_tdir_error_deg"]) for row in final360i_rows if not row.get("anti_parallel")]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.hist(non_anti_vals, bins=30, alpha=0.7, label="not anti-parallel", color="#1d3557")
    ax.hist(anti_vals, bins=30, alpha=0.7, label="anti-parallel", color="#e76f51")
    ax.set_title("Anti-parallel regime distribution on FINAL360I test rows")
    ax.set_xlabel("signed_tdir_error_deg")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(anti_parallel_fig, dpi=220)
    fig.savefig(anti_parallel_fig.with_suffix(".pdf"))
    plt.close(fig)
    for file_path, zh, en, notes in [
        (tmag_bucket_fig, "DATA360A tmag bucket signed_tdir 图", "DATA360A tmag-bucket signed_tdir plot", "Derived from retained final360i test pair rows."),
        (rot_bucket_fig, "DATA360A rotation bucket signed_tdir 图", "DATA360A rotation-bucket signed_tdir plot", "Derived from retained final360i test pair rows."),
        (anti_parallel_fig, "anti_parallel regime 分布图", "Anti-parallel regime distribution", "Shows anti-parallel remains a broad failure regime rather than a single-bucket issue."),
        (neg_compare_fig, "FINAL360I / J / K / L 对比图", "FINAL360I / J / K / L comparison", "Validation-only comparison because standalone J/K/L test json files are unavailable."),
        (seq_compare_fig, "FINAL360I vs SEQ360B bucket 对比图", "FINAL360I vs SEQ360B bucket comparison", "Shows SEQ360B mainly changes scale/path behavior rather than direction error."),
    ]:
        fig_manifest.append(
            {
                "file_path": _maybe_rel(file_path),
                "title_zh": zh,
                "title_en": en,
                "source_files": [_maybe_rel(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic")],
                "recommended_location": "thesis_appendix",
                "notes": notes,
            }
        )

    matching_entries = generate_matching_geometry_assets()
    debug_report = {
        "matching_geometry_entries": matching_entries,
        "default_model_behavior_changed": False,
        "debug_hooks_added": False,
        "training_executed": False,
        "metrics_modified": False,
    }
    for idx, entry in enumerate(matching_entries, start=1):
        fig_manifest.append(
            {
                "file_path": entry["file_path_png"],
                "title_zh": f"matching / geometry 可视化样本 {idx}",
                "title_en": f"Matching / geometry visualization sample {idx}",
                "source_files": [entry["source_manifest"], entry["checkpoint"]],
                "recommended_location": "thesis_appendix",
                "notes": "Generated by eval-only forward using existing auxiliary outputs; no training, no checkpoint update, no metric file modification.",
            }
        )

    return fig_manifest, debug_report


def _write_caption_packs(fig_manifest: Sequence[Dict[str, Any]]) -> None:
    zh_lines = ["# Final Figure Captions (ZH)", ""]
    en_lines = ["# Final Figure Captions (EN)", ""]
    for item in fig_manifest:
        zh_lines.extend(
            [
                f"## {_maybe_rel(REPO_ROOT / item['file_path']) if item['file_path'].startswith('thesis/') else item['file_path']}",
                f"- 标题：{item['title_zh']}",
                f"- 图注：本图基于 `{', '.join(item['source_files'])}` 整理，用于展示 {item['title_zh']}。{item['notes']}",
                "",
            ]
        )
        en_lines.extend(
            [
                f"## {item['file_path']}",
                f"- Title: {item['title_en']}",
                f"- Caption: This figure is prepared from `{', '.join(item['source_files'])}` and visualizes {item['title_en']}. {item['notes']}",
                "",
            ]
        )
    _write_md(CAPTION_DIR / "final_figure_captions_zh.md", zh_lines)
    _write_md(CAPTION_DIR / "final_figure_captions_en.md", en_lines)


def _build_visual_source_manifest(fig_manifest: Sequence[Dict[str, Any]], table_manifest: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined = []
    for item in list(fig_manifest) + list(table_manifest):
        combined.append(
            {
                "output_file": item["file_path"],
                "source_files": item["source_files"],
                "notes": item["notes"],
            }
        )
    return combined


def main() -> None:
    _ensure_dirs()
    pair_sources = _build_pair_sources()
    ablation_sources = _build_ablation_sources()
    trajectory_sources = _build_trajectory_sources(pair_sources)
    negative_sources = _build_negative_sources()

    table_manifest = _build_tables(pair_sources, ablation_sources, trajectory_sources, negative_sources)
    fig_manifest, debug_report = _build_figures(pair_sources, ablation_sources, trajectory_sources, negative_sources)
    _write_caption_packs(fig_manifest)
    visual_source_manifest = _build_visual_source_manifest(fig_manifest, table_manifest)

    fig_manifest_path = REPORTS_DIR / "THESIS362_figure_manifest.json"
    table_manifest_path = REPORTS_DIR / "THESIS362_table_manifest.json"
    visual_manifest_path = REPORTS_DIR / "THESIS362_visual_source_manifest.json"
    _write_json(fig_manifest_path, fig_manifest)
    _write_json(table_manifest_path, table_manifest)
    _write_json(visual_manifest_path, visual_source_manifest)

    report_path = REPORTS_DIR / "THESIS362_final_experiment_master_table_and_visual_pack.md"
    report_lines = [
        "# THESIS362 Final Experiment Master Table and Visual Pack",
        "",
        "## Scope",
        "- this task reorganized retained experiment outputs into thesis-ready tables, figures, captions, and manifests",
        "- no training was executed",
        "- no full-test model re-evaluation was executed",
        "- no metrics json file was modified",
        "- no checkpoint was modified",
        "",
        "## Matching / Geometry Visuals",
        "- matching / geometry figure export was attempted in eval-only mode against the retained FINAL360I mainline",
        "- no training was executed",
        "- no checkpoint or metric file was modified",
        f"- default model behavior changed: `{str(debug_report['default_model_behavior_changed']).lower()}`",
        f"- debug hooks added: `{str(debug_report['debug_hooks_added']).lower()}`",
        f"- matching / geometry figures written: `{'true' if debug_report['matching_geometry_entries'] else 'false'}`",
        "- status: `unavailable` for the retained FINAL360I STRUCT360B mainline because the requested `Wf_ab_raw / Wf_ab / allowed_mask / epi_bias` tensors are not emitted without intrusive model changes",
        "",
        "## Missing / Unavailable Items",
        "- `reports/ODOM360A_metrics_test.json`: unavailable in current worktree; trajectory metrics were derived from retained TUM exports instead",
        "- `reports/ODOM360B_metrics_test.json`: unavailable in current worktree; trajectory metrics were derived from retained TUM exports instead",
        "- `reports/DATA360A_regime_buckets.json`: unavailable in current worktree; bucket diagnostics were derived from retained jsonl row exports instead",
        "- `reports/DATA360A_antiparallel_analysis.json`: unavailable in current worktree; anti-parallel analysis was derived from retained jsonl row exports instead",
        "- `reports/FINAL360J_metrics_test.json`, `reports/FINAL360K_metrics_test.json`, `reports/FINAL360L_metrics_test.json`: unavailable in current worktree; appendix summaries use retained validation train logs instead",
        "- `reports/TRAIN360C_metrics_test.json`: unavailable in current worktree; retained TRAIN360C numbers were recovered from aggregate results reports",
        "",
        "## Outputs",
        f"- tables: `{_maybe_rel(TABLE_DIR)}`",
        f"- figures: `{_maybe_rel(FIG_DIR)}`",
        f"- captions: `{_maybe_rel(CAPTION_DIR)}`",
        f"- manifests: `{_maybe_rel(MANIFEST_DIR)}`",
    ]
    _write_md(report_path, report_lines)
    print(json.dumps({"written": True, "report": _maybe_rel(report_path), "figures": len(fig_manifest), "tables": len(table_manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
