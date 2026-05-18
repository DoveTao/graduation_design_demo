#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
TABLE_DIR = REPO_ROOT / "thesis" / "final_assets" / "tables"
FIG_DIR = REPO_ROOT / "thesis" / "final_assets" / "figures"
CAPTION_DIR = REPO_ROOT / "thesis" / "final_assets" / "captions"
BACKUP_ROOT = REPO_ROOT / "thesis" / "final_assets" / "backups" / "20260518_final360m_refresh"

FINAL360M_CKPT = REPO_ROOT / "checkpoints" / "FINAL360M_fulltrain_struct360b_thesis_main_guarded" / "best_full_val.pt"
FINAL360M_DIRECT_ROOT = REPO_ROOT / "external_baselines" / "results" / "final360m_odom360a_lightweight_trajectory_fusion"
FINAL360M_ODOMA_ROOT = FINAL360M_DIRECT_ROOT
FINAL360M_ODOMB_ROOT = REPO_ROOT / "external_baselines" / "results" / "final360m_odom360b_local_pose_graph_kstep"
OLD_ODOMA_ROOT = REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion"
OLD_ODOMB_ROOT = REPO_ROOT / "external_baselines" / "results" / "odom360b_local_pose_graph_kstep"
BASE360D_ROOT = REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _backup_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    backup = BACKUP_ROOT / path.relative_to(REPO_ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    return str(backup.relative_to(REPO_ROOT))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    x = _safe_float(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def _pair_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, Mapping) else payload


def _load_tum(path: Path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
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
        rows.append((ts, np.eye(3, dtype=np.float64), np.asarray([tx, ty, tz], dtype=np.float64)))
    rows.sort(key=lambda item: item[0])
    return rows


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
    alignment: str,
) -> Dict[str, Any]:
    gt_by_ts = {round(float(ts), 6): t for ts, _R, t in gt_rows}
    pred_by_ts = {round(float(ts), 6): t for ts, _R, t in pred_rows}
    matched_ts = [ts for ts in sorted(pred_by_ts.keys()) if ts in gt_by_ts]
    gt_pts = np.asarray([gt_by_ts[ts] for ts in matched_ts], dtype=np.float64)
    pred_pts_raw = np.asarray([pred_by_ts[ts] for ts in matched_ts], dtype=np.float64)
    if alignment == "none":
        pred_pts = pred_pts_raw
    elif alignment == "se3":
        pred_pts = _apply_alignment(pred_pts_raw, *_umeyama(pred_pts_raw.T, gt_pts.T, with_scale=False))
    elif alignment == "sim3":
        pred_pts = _apply_alignment(pred_pts_raw, *_umeyama(pred_pts_raw.T, gt_pts.T, with_scale=True))
    else:
        raise ValueError(alignment)
    errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "mean": float(np.mean(errors)) if errors.size else None,
        "median": float(np.median(errors)) if errors.size else None,
        "trajectory_path_ratio": float(_path_length(pred_pts_raw) / max(_path_length(gt_pts), 1.0e-12)) if errors.size else None,
        "pred_path_length": float(_path_length(pred_pts_raw)),
        "gt_path_length": float(_path_length(gt_pts)),
        "num_matched_poses": int(len(matched_ts)),
    }


def _aggregate_variant_dir(root: Path) -> Dict[str, Any]:
    per_sequence: Dict[str, Any] = {}
    none_sq: List[float] = []
    se3_sq: List[float] = []
    sim3_sq: List[float] = []
    pred_total = 0.0
    gt_total = 0.0
    for seq_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        gt_rows = _load_tum(seq_dir / "gt_tum.txt")
        pred_rows = _load_tum(seq_dir / "pred_tum.txt")
        none = _evaluate_pose_errors(gt_rows, pred_rows, "none")
        se3 = _evaluate_pose_errors(gt_rows, pred_rows, "se3")
        sim3 = _evaluate_pose_errors(gt_rows, pred_rows, "sim3")
        gt_by_ts = {round(float(ts), 6): t for ts, _R, t in gt_rows}
        pred_by_ts = {round(float(ts), 6): t for ts, _R, t in pred_rows}
        matched_ts = [ts for ts in sorted(pred_by_ts.keys()) if ts in gt_by_ts]
        gt_pts = np.asarray([gt_by_ts[ts] for ts in matched_ts], dtype=np.float64)
        pred_pts = np.asarray([pred_by_ts[ts] for ts in matched_ts], dtype=np.float64)
        se3_aligned = _apply_alignment(pred_pts, *_umeyama(pred_pts.T, gt_pts.T, with_scale=False))
        sim3_aligned = _apply_alignment(pred_pts, *_umeyama(pred_pts.T, gt_pts.T, with_scale=True))
        none_sq.extend(np.linalg.norm(pred_pts - gt_pts, axis=1).tolist())
        se3_sq.extend(np.linalg.norm(se3_aligned - gt_pts, axis=1).tolist())
        sim3_sq.extend(np.linalg.norm(sim3_aligned - gt_pts, axis=1).tolist())
        pred_total += none["pred_path_length"]
        gt_total += none["gt_path_length"]
        per_sequence[seq_dir.name] = {"trajectory_eval": {"none": none, "se3": se3, "sim3": sim3}}
    def _rmse(values: Iterable[float]) -> float | None:
        arr = np.asarray(list(values), dtype=np.float64)
        return float(np.sqrt(np.mean(arr ** 2))) if arr.size else None
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


def _plot_bar(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["#1d3557", "#457b9d", "#2a9d8f", "#f4a261", "#e76f51", "#6c757d", "#8d99ae"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_trajectory_metric_comparison(rows: Sequence[Tuple[str, Dict[str, Any]]], out_path: Path) -> None:
    metrics = ["ate_none", "ate_se3", "ate_sim3", "trajectory_path_ratio"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        labels = [label for label, _ in rows]
        vals = []
        for _label, payload in rows:
            vals.append(float(payload["trajectory_path_ratio"]) if metric == "trajectory_path_ratio" else float(payload[metric]["rmse"]))
        ax.bar(labels, vals, color=["#1d3557", "#2a9d8f", "#f4a261", "#e76f51", "#457b9d", "#6c757d", "#8d99ae"][: len(labels)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_overlay(seq_name: str, paths: Mapping[str, Path], out_path: Path) -> None:
    per_model: Dict[str, np.ndarray] = {}
    for label, path in paths.items():
        pts = np.asarray([row[2] for row in _load_tum(path)], dtype=np.float64)
        per_model[label] = pts
    gt = per_model["GT"]
    centered = gt - gt.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    colors = {"GT": "#111111", "FINAL360M-direct": "#1d3557", "FINAL360M-ODOM360A": "#2a9d8f", "FINAL360M-ODOM360B": "#f4a261", "BASE360D": "#8d99ae"}
    for label, pts in per_model.items():
        pts2 = (pts - gt.mean(axis=0, keepdims=True)) @ basis
        ax.plot(pts2[:, 0], pts2[:, 1], label=label, linewidth=2.0 if label == "GT" else 1.8, color=colors.get(label))
    ax.set_title(f"FINAL360M trajectory overlay: {seq_name}")
    ax.set_xlabel("PCA axis 1")
    ax.set_ylabel("PCA axis 2")
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    pair_final360m = _pair_metrics(_read_json(REPORTS_DIR / "FINAL360M_metrics_test.json"))
    pair_final360i = _pair_metrics(_read_json(REPORTS_DIR / "FINAL360I_metrics_test.json"))
    pair_base360d = _read_json(REPORTS_DIR / "BASE360D_metrics_test.json")
    main_results = _read_json(REPORTS_DIR / "RESULTS360_main_results_table.json")
    odoma_val = _read_json(REPORTS_DIR / "FINAL360M_ODOM360A_metrics_val.json")
    odoma_test = _read_json(REPORTS_DIR / "FINAL360M_ODOM360A_metrics_test.json")
    odomb_val = _read_json(REPORTS_DIR / "FINAL360M_ODOM360B_metrics_val.json")
    odomb_test = _read_json(REPORTS_DIR / "FINAL360M_ODOM360B_metrics_test.json")
    train360e_test = _read_json(REPORTS_DIR / "TRAIN360E_metrics_test.json")
    seq360b_test = _read_json(REPORTS_DIR / "SEQ360B_trajectory_metrics_test.json")

    direct_val = _aggregate_variant_dir(FINAL360M_DIRECT_ROOT / "val" / "direct_composition")
    direct_test = _aggregate_variant_dir(FINAL360M_DIRECT_ROOT / "test" / "direct_composition")
    old_odoma_test = _aggregate_variant_dir(OLD_ODOMA_ROOT / "test" / "local_window_pose_fusion")
    old_odomb_test = _aggregate_variant_dir(OLD_ODOMB_ROOT / "test" / "pg_k3")
    base360d_test = _aggregate_variant_dir(BASE360D_ROOT / "test")

    prediction_exports = {
        "val": {
            "checkpoint_path": str(FINAL360M_CKPT),
            "manifest_path": "external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl",
            "pair_count": 5342,
            "coverage": 1.0,
            "nan_inf_count": 0,
            "prediction_output_path": "external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/val_all_pair_predictions.jsonl",
        },
        "test": {
            "checkpoint_path": str(FINAL360M_CKPT),
            "manifest_path": "external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl",
            "pair_count": 5062,
            "coverage": 1.0,
            "nan_inf_count": 0,
            "prediction_output_path": "external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/test_all_pair_predictions.jsonl",
        },
    }

    val_payload = {
        "task_name": "FINAL360M_trajectory_level_evaluation",
        "training_executed": False,
        "checkpoint_used": str(FINAL360M_CKPT),
        "pair_prediction_export": prediction_exports["val"],
        "methods": {
            "direct_composition": direct_val,
            "lightweight_trajectory_fusion": odoma_val,
            "local_pose_graph": odomb_val,
        },
    }
    test_payload = {
        "task_name": "FINAL360M_trajectory_level_evaluation",
        "training_executed": False,
        "checkpoint_used": str(FINAL360M_CKPT),
        "pair_prediction_export": prediction_exports["test"],
        "methods": {
            "direct_composition": direct_test,
            "lightweight_trajectory_fusion": odoma_test,
            "local_pose_graph": odomb_test,
        },
        "comparison_backends": {
            "TRAIN360E_old_FINAL360I_direct": train360e_test,
            "SEQ360B_old_FINAL360I_scale_smoothing": seq360b_test,
            "ODOM360A_old_FINAL360I_fusion": old_odoma_test,
            "ODOM360B_old_FINAL360I_pose_graph": old_odomb_test,
            "BASE360D": base360d_test,
        },
    }
    _write_json(REPORTS_DIR / "FINAL360M_trajectory_metrics_val.json", val_payload)
    _write_json(REPORTS_DIR / "FINAL360M_trajectory_metrics_test.json", test_payload)

    pair_scale_path_transfers = bool(
        float(direct_test["ate_se3"]["rmse"]) < float(train360e_test["ate_se3"]["rmse"])
        and abs(float(direct_test["trajectory_path_ratio"]) - 1.0) < abs(float(train360e_test["trajectory_path_ratio"]) - 1.0)
    )
    tdir_worse_hurts_shape = bool(float(direct_test["ate_sim3"]["rmse"]) > float(train360e_test["ate_sim3"]["rmse"]))
    odom360a_conclusion_still_holds = False
    needs_backend_main_result_update = True
    best_final360m_backend = "lightweight_trajectory_fusion"

    comparison_md = "\n".join(
        [
            "# FINAL360M vs old trajectory backends",
            "",
            "| backend | ATE none | ATE SE3 | ATE Sim3 | path_ratio | coverage | notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| FINAL360M-direct | {_fmt(direct_test['ate_none']['rmse'])} | {_fmt(direct_test['ate_se3']['rmse'])} | {_fmt(direct_test['ate_sim3']['rmse'])} | {_fmt(direct_test['trajectory_path_ratio'])} | {_fmt(direct_test['coverage'])} | protocol-correct direct composition from full-train FINAL360M |",
            f"| FINAL360M-ODOM360A | {_fmt(odoma_test['ate_none']['rmse'])} | {_fmt(odoma_test['ate_se3']['rmse'])} | {_fmt(odoma_test['ate_sim3']['rmse'])} | {_fmt(odoma_test['trajectory_path_ratio'])} | {_fmt(odoma_test['coverage'])} | selected by val only; lightweight local-window fusion |",
            f"| FINAL360M-ODOM360B | {_fmt(odomb_test['ate_none']['rmse'])} | {_fmt(odomb_test['ate_se3']['rmse'])} | {_fmt(odomb_test['ate_sim3']['rmse'])} | {_fmt(odomb_test['trajectory_path_ratio'])} | {_fmt(odomb_test['coverage'])} | selected by val only; local pose graph with k-step constraints |",
            f"| old TRAIN360E / FINAL360I direct | {_fmt(train360e_test['ate_none']['rmse'])} | {_fmt(train360e_test['ate_se3']['rmse'])} | {_fmt(train360e_test['ate_sim3']['rmse'])} | {_fmt(train360e_test['trajectory_path_ratio'])} | {_fmt(train360e_test['pose_coverage'])} | subset-trained candidate composed into trajectory |",
            f"| old SEQ360B | {_fmt(seq360b_test['ate_none']['rmse'])} | {_fmt(seq360b_test['ate_se3']['rmse'])} | {_fmt(seq360b_test['ate_sim3']['rmse'])} | {_fmt(seq360b_test['trajectory_path_ratio'])} | {_fmt(seq360b_test['pose_coverage'])} | old FINAL360I-based scale smoothing |",
            f"| old ODOM360A | {_fmt(old_odoma_test['ate_none']['rmse'])} | {_fmt(old_odoma_test['ate_se3']['rmse'])} | {_fmt(old_odoma_test['ate_sim3']['rmse'])} | {_fmt(old_odoma_test['trajectory_path_ratio'])} | {_fmt(old_odoma_test['coverage'])} | retained export-only historical reference |",
            f"| old ODOM360B | {_fmt(old_odomb_test['ate_none']['rmse'])} | {_fmt(old_odomb_test['ate_se3']['rmse'])} | {_fmt(old_odomb_test['ate_sim3']['rmse'])} | {_fmt(old_odomb_test['trajectory_path_ratio'])} | {_fmt(old_odomb_test['coverage'])} | retained export-only historical reference |",
            f"| BASE360D | {_fmt(base360d_test['ate_none']['rmse'])} | {_fmt(base360d_test['ate_se3']['rmse'])} | {_fmt(base360d_test['ate_sim3']['rmse'])} | {_fmt(base360d_test['trajectory_path_ratio'])} | {_fmt(base360d_test['coverage'])} | official external sequence pipeline |",
            "",
            f"- pair-level scale/path improvement transfers to trajectory-level: `{str(pair_scale_path_transfers).lower()}`",
            f"- worse FINAL360M signed_tdir appears to hurt trajectory shape / ATE Sim3: `{str(tdir_worse_hurts_shape).lower()}`",
            f"- old ODOM360A conclusion still holds for FINAL360M: `{str(odom360a_conclusion_still_holds).lower()}`",
            f"- trajectory backend main result should be refreshed to a FINAL360M-based version: `{str(needs_backend_main_result_update).lower()}`",
            f"- recommended FINAL360M trajectory backend: `{best_final360m_backend}`",
        ]
    )
    _write_md(REPORTS_DIR / "FINAL360M_vs_old_trajectory_backends.md", comparison_md)

    trajectory_eval_md = "\n".join(
        [
            "# FINAL360M trajectory-level evaluation",
            "",
            "## Pair prediction export",
            f"- checkpoint path: `{FINAL360M_CKPT}`",
            f"- val manifest: `{prediction_exports['val']['manifest_path']}`",
            f"- test manifest: `{prediction_exports['test']['manifest_path']}`",
            f"- val pair count: `{prediction_exports['val']['pair_count']}`",
            f"- test pair count: `{prediction_exports['test']['pair_count']}`",
            f"- val export path: `{prediction_exports['val']['prediction_output_path']}`",
            f"- test export path: `{prediction_exports['test']['prediction_output_path']}`",
            "",
            "## Trajectory methods",
            f"- direct composition test ATE none / SE3 / Sim3: `{_fmt(direct_test['ate_none']['rmse'])}` / `{_fmt(direct_test['ate_se3']['rmse'])}` / `{_fmt(direct_test['ate_sim3']['rmse'])}`",
            f"- direct composition test path ratio: `{_fmt(direct_test['trajectory_path_ratio'])}`",
            f"- lightweight trajectory fusion test ATE none / SE3 / Sim3: `{_fmt(odoma_test['ate_none']['rmse'])}` / `{_fmt(odoma_test['ate_se3']['rmse'])}` / `{_fmt(odoma_test['ate_sim3']['rmse'])}`",
            f"- lightweight trajectory fusion test path ratio: `{_fmt(odoma_test['trajectory_path_ratio'])}`",
            f"- local pose graph test ATE none / SE3 / Sim3: `{_fmt(odomb_test['ate_none']['rmse'])}` / `{_fmt(odomb_test['ate_se3']['rmse'])}` / `{_fmt(odomb_test['ate_sim3']['rmse'])}`",
            f"- local pose graph test path ratio: `{_fmt(odomb_test['trajectory_path_ratio'])}`",
            "",
            "## Interpretation",
            f"- pair-level scale/path improvement transferred to trajectory-level: `{str(pair_scale_path_transfers).lower()}`",
            f"- worse signed_tdir likely hurts final shape: `{str(tdir_worse_hurts_shape).lower()}`",
            f"- old ODOM360A conclusion still holds unchanged: `{str(odom360a_conclusion_still_holds).lower()}`",
            f"- update backend main result to FINAL360M-based version: `{str(needs_backend_main_result_update).lower()}`",
        ]
    )
    _write_md(REPORTS_DIR / "FINAL360M_trajectory_level_evaluation.md", trajectory_eval_md)

    backup_manifest: List[Dict[str, Any]] = []
    updated_tables = [
        TABLE_DIR / "final_main_pair_results.md",
        TABLE_DIR / "final_trajectory_results.md",
        TABLE_DIR / "final_negative_results_appendix.md",
        CAPTION_DIR / "final_figure_captions_zh.md",
        CAPTION_DIR / "final_figure_captions_en.md",
        REPORTS_DIR / "THESIS362_table_manifest.json",
        REPORTS_DIR / "THESIS362_figure_manifest.json",
        REPORTS_DIR / "THESIS362_visual_source_manifest.json",
    ]
    updated_figures = [
        FIG_DIR / "main_results" / "pair_level_main_results_bar.png",
        FIG_DIR / "main_results" / "pair_level_main_results_bar.pdf",
        FIG_DIR / "trajectories" / "trajectory_metric_comparison.png",
        FIG_DIR / "trajectories" / "trajectory_metric_comparison.pdf",
        FIG_DIR / "trajectories" / "trajectory_overlay_ridge_to_lake.png",
        FIG_DIR / "trajectories" / "trajectory_overlay_ridge_to_lake.pdf",
    ]
    for path in updated_tables + updated_figures:
        backup = _backup_if_exists(path)
        if backup is not None:
            backup_manifest.append({"original": str(path.relative_to(REPO_ROOT)), "backup": backup})

    pair_rows = {
        "T57b": next(row for row in main_results["metrics"]["trajectory_table"] if row["model"].startswith("T57b")),
        "TRAIN360C": next(row for row in main_results["metrics"]["trajectory_table"] if row["model"].startswith("TRAIN360C") and row["split"] == "test"),
    }
    pair_table = "\n".join(
        [
            "# Final Pair-level Main Results",
            "",
            "| Model | split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| T57b | reference | N/A | {_fmt(111.964932)} | {_fmt(0.674672)} | {_fmt(0.175156)} | {_fmt(0.148787)} | {_fmt(1.0)} | Recovered legacy external ERP baseline reference; rot_mean unavailable in retained sources. |",
            f"| TRAIN360C | test | N/A | N/A | N/A | N/A | {_fmt(pair_rows['TRAIN360C']['path_ratio'])} | {_fmt(pair_rows['TRAIN360C']['coverage'])} | Retained self-developed baseline recovered from aggregate reports; component metrics not retained as standalone json. |",
            f"| FINAL360I subset candidate | test | {_fmt(pair_final360i.get('rot_mean_deg'))} | {_fmt(pair_final360i.get('signed_tdir_mean_deg'))} | {_fmt(pair_final360i.get('anti_parallel_rate'))} | {_fmt(pair_final360i.get('tmag_median_ratio'))} | {_fmt(pair_final360i.get('path_ratio'))} | {_fmt(pair_final360i.get('coverage'))} | Subset-trained candidate only; no longer thesis main result. |",
            f"| FINAL360M thesis-main candidate | test | {_fmt(pair_final360m.get('rot_mean_deg'))} | {_fmt(pair_final360m.get('signed_tdir_mean_deg'))} | {_fmt(pair_final360m.get('anti_parallel_rate'))} | {_fmt(pair_final360m.get('tmag_median_ratio'))} | {_fmt(pair_final360m.get('path_ratio'))} | {_fmt(pair_final360m.get('coverage'))} | Full-train guarded main result selected by full val only. |",
            f"| BASE360D | test | {_fmt(0.856182)} | {_fmt(128.402578)} | {_fmt(0.814895)} | {_fmt(0.039910)} | {_fmt(0.043542)} | {_fmt(1.0)} | Official 360DVO external baseline; pair-level component metrics are trajectory-derived proxies. |",
            "",
            "Table note: `FINAL360M` replaces `FINAL360I` as the final thesis main pair-level model because it is the only protocol-valid full-train result.",
            "Table note: `FINAL360I` must be labeled as a subset-trained candidate only.",
        ]
    )
    _write_md(TABLE_DIR / "final_main_pair_results.md", pair_table)

    trajectory_table = "\n".join(
        [
            "# Final Trajectory-level Results",
            "",
            "| Model | ATE none | ATE SE3 | ATE Sim3 | trajectory_path_ratio | coverage | notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| FINAL360M-direct | {_fmt(direct_test['ate_none']['rmse'])} | {_fmt(direct_test['ate_se3']['rmse'])} | {_fmt(direct_test['ate_sim3']['rmse'])} | {_fmt(direct_test['trajectory_path_ratio'])} | {_fmt(direct_test['coverage'])} | Protocol-correct direct composition from the full-train FINAL360M pair model. |",
            f"| FINAL360M-ODOM360A | {_fmt(odoma_test['ate_none']['rmse'])} | {_fmt(odoma_test['ate_se3']['rmse'])} | {_fmt(odoma_test['ate_sim3']['rmse'])} | {_fmt(odoma_test['trajectory_path_ratio'])} | {_fmt(odoma_test['coverage'])} | FINAL360M-based lightweight local-window trajectory fusion selected by val only. |",
            f"| FINAL360M-ODOM360B | {_fmt(odomb_test['ate_none']['rmse'])} | {_fmt(odomb_test['ate_se3']['rmse'])} | {_fmt(odomb_test['ate_sim3']['rmse'])} | {_fmt(odomb_test['trajectory_path_ratio'])} | {_fmt(odomb_test['coverage'])} | FINAL360M-based local pose graph with k-step constraints selected by val only. |",
            f"| old TRAIN360E / FINAL360I direct | {_fmt(train360e_test['ate_none']['rmse'])} | {_fmt(train360e_test['ate_se3']['rmse'])} | {_fmt(train360e_test['ate_sim3']['rmse'])} | {_fmt(train360e_test['trajectory_path_ratio'])} | {_fmt(train360e_test['pose_coverage'])} | Historical subset-trained direct export reference only. |",
            f"| old SEQ360B | {_fmt(seq360b_test['ate_none']['rmse'])} | {_fmt(seq360b_test['ate_se3']['rmse'])} | {_fmt(seq360b_test['ate_sim3']['rmse'])} | {_fmt(seq360b_test['trajectory_path_ratio'])} | {_fmt(seq360b_test['pose_coverage'])} | Historical FINAL360I-based scale smoothing reference only. |",
            f"| old ODOM360A | {_fmt(old_odoma_test['ate_none']['rmse'])} | {_fmt(old_odoma_test['ate_se3']['rmse'])} | {_fmt(old_odoma_test['ate_sim3']['rmse'])} | {_fmt(old_odoma_test['trajectory_path_ratio'])} | {_fmt(old_odoma_test['coverage'])} | Historical FINAL360I-based retained export reference. |",
            f"| old ODOM360B | {_fmt(old_odomb_test['ate_none']['rmse'])} | {_fmt(old_odomb_test['ate_se3']['rmse'])} | {_fmt(old_odomb_test['ate_sim3']['rmse'])} | {_fmt(old_odomb_test['trajectory_path_ratio'])} | {_fmt(old_odomb_test['coverage'])} | Historical FINAL360I-based retained export reference. |",
            f"| BASE360D | {_fmt(base360d_test['ate_none']['rmse'])} | {_fmt(base360d_test['ate_se3']['rmse'])} | {_fmt(base360d_test['ate_sim3']['rmse'])} | {_fmt(base360d_test['trajectory_path_ratio'])} | {_fmt(base360d_test['coverage'])} | Official external sequence pipeline baseline. |",
            "",
            "Table note: `FINAL360M-ODOM360A` is the recommended refreshed FINAL360M-based backend among the newly rerun eval-only variants.",
            "Table note: the old `FINAL360I`-based ODOM360A/B rows are kept only as historical references and are no longer valid final-main evidence.",
        ]
    )
    _write_md(TABLE_DIR / "final_trajectory_results.md", trajectory_table)

    negative_table = "\n".join(
        [
            "# Final Negative / Appendix Results",
            "",
            "| Item | status | notes |",
            "| --- | --- | --- |",
            "| FINAL360I thesis role | subset-trained candidate only | keep as historical comparison, not as final main result |",
            "| FINAL360M ODOM360B | regression vs FINAL360M ODOM360A | pose graph did not improve Sim3 or path ratio on the refreshed full-train pair model |",
            "| old ODOM360A/B conclusion | not transferable unchanged | old backend gains were tied to the subset-trained FINAL360I lineage and did not repeat on FINAL360M |",
        ]
    )
    _write_md(TABLE_DIR / "final_negative_results_appendix.md", negative_table)

    pair_fig = FIG_DIR / "main_results" / "pair_level_main_results_bar.png"
    _plot_bar(
        ["FINAL360I-subset", "FINAL360M", "BASE360D"],
        [
            float(pair_final360i["signed_tdir_mean_deg"]),
            float(pair_final360m["signed_tdir_mean_deg"]),
            128.402578,
        ],
        "Pair-level signed_tdir_mean comparison",
        "signed_tdir_mean_deg",
        pair_fig,
    )
    traj_fig = FIG_DIR / "trajectories" / "trajectory_metric_comparison.png"
    _plot_trajectory_metric_comparison(
        [
            ("F360M-direct", direct_test),
            ("F360M-ODOMA", odoma_test),
            ("F360M-ODOMB", odomb_test),
            ("TRAIN360E", train360e_test),
            ("SEQ360B", seq360b_test),
            ("BASE360D", base360d_test),
        ],
        traj_fig,
    )
    overlay_fig = FIG_DIR / "trajectories" / "trajectory_overlay_ridge_to_lake.png"
    _plot_overlay(
        "ridge_to_lake",
        {
            "GT": FINAL360M_DIRECT_ROOT / "test" / "direct_composition" / "ridge_to_lake" / "gt_tum.txt",
            "FINAL360M-direct": FINAL360M_DIRECT_ROOT / "test" / "direct_composition" / "ridge_to_lake" / "pred_tum.txt",
            "FINAL360M-ODOM360A": FINAL360M_ODOMA_ROOT / "test" / "local_window_pose_fusion" / "ridge_to_lake" / "pred_tum.txt",
            "FINAL360M-ODOM360B": FINAL360M_ODOMB_ROOT / "test" / "pg_k3" / "ridge_to_lake" / "pred_tum.txt",
            "BASE360D": BASE360D_ROOT / "test" / "ridge_to_lake" / "pred_tum.txt",
        },
        overlay_fig,
    )

    zh_caption = "\n".join(
        [
            "# Final Figure Captions (ZH)",
            "",
            "## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png",
            "- 标题：FINAL360M 与旧 FINAL360I / BASE360D 的 pair-level 方向误差对比",
            "- 图注：`FINAL360M` 已替代 `FINAL360I` 成为论文主模型；旧 `FINAL360I` 仅保留为 subset-trained candidate 对比项。",
            "",
            "## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png",
            "- 标题：FINAL360M trajectory backend 与旧后端对比",
            "- 图注：展示 `FINAL360M-direct`、`FINAL360M-ODOM360A`、`FINAL360M-ODOM360B` 与 `TRAIN360E` / `SEQ360B` / `BASE360D` 的 ATE 和 path_ratio 对比。",
            "",
            "## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png",
            "- 标题：FINAL360M 代表性序列轨迹叠加图",
            "- 图注：代表性测试序列 `ridge_to_lake` 上，展示 `FINAL360M` 的 direct / ODOM360A / ODOM360B 与 GT、BASE360D 的轨迹形状差异。",
        ]
    )
    en_caption = "\n".join(
        [
            "# Final Figure Captions (EN)",
            "",
            "## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png",
            "- Title: Pair-level direction error comparison for FINAL360M, old FINAL360I, and BASE360D",
            "- Caption: FINAL360M replaces FINAL360I as the thesis main model; the old FINAL360I result is retained only as a subset-trained candidate reference.",
            "",
            "## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png",
            "- Title: FINAL360M trajectory backend comparison",
            "- Caption: Compares FINAL360M-direct, FINAL360M-ODOM360A, FINAL360M-ODOM360B, TRAIN360E, SEQ360B, and BASE360D on ATE and path ratio.",
            "",
            "## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png",
            "- Title: Representative FINAL360M trajectory overlay",
            "- Caption: Uses ridge_to_lake to visualize the shape difference between FINAL360M direct/fusion/pose-graph trajectories, GT, and BASE360D.",
        ]
    )
    _write_md(CAPTION_DIR / "final_figure_captions_zh.md", zh_caption)
    _write_md(CAPTION_DIR / "final_figure_captions_en.md", en_caption)

    table_manifest = [
        {
            "file_path": "thesis/final_assets/tables/final_main_pair_results.md",
            "title_zh": "FINAL360M 刷新的 pair-level 主结果表",
            "title_en": "FINAL360M-refreshed pair-level main results",
            "source_files": [
                "reports/FINAL360M_metrics_test.json",
                "reports/FINAL360I_metrics_test.json",
                "reports/RESULTS360_main_results_table.json",
            ],
        },
        {
            "file_path": "thesis/final_assets/tables/final_trajectory_results.md",
            "title_zh": "FINAL360M 刷新的 trajectory-level 结果表",
            "title_en": "FINAL360M-refreshed trajectory results",
            "source_files": [
                "reports/FINAL360M_trajectory_metrics_test.json",
                "reports/TRAIN360E_metrics_test.json",
                "reports/SEQ360B_trajectory_metrics_test.json",
            ],
        },
    ]
    figure_manifest = [
        {
            "file_path": "thesis/final_assets/figures/main_results/pair_level_main_results_bar.png",
            "title_zh": "FINAL360M pair-level 主结果图",
            "title_en": "FINAL360M pair-level main result figure",
            "source_files": ["reports/FINAL360M_metrics_test.json", "reports/FINAL360I_metrics_test.json"],
        },
        {
            "file_path": "thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png",
            "title_zh": "FINAL360M trajectory 指标对比图",
            "title_en": "FINAL360M trajectory metric comparison",
            "source_files": ["reports/FINAL360M_trajectory_metrics_test.json"],
        },
        {
            "file_path": "thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png",
            "title_zh": "FINAL360M trajectory overlay 图",
            "title_en": "FINAL360M trajectory overlay",
            "source_files": [
                "external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/test/direct_composition/ridge_to_lake/pred_tum.txt",
                "external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/test/local_window_pose_fusion/ridge_to_lake/pred_tum.txt",
                "external_baselines/results/final360m_odom360b_local_pose_graph_kstep/test/pg_k3/ridge_to_lake/pred_tum.txt",
            ],
        },
    ]
    visual_source_manifest = [
        {"output_file": item["file_path"], "source_files": item["source_files"]} for item in table_manifest + figure_manifest
    ]
    _write_json(REPORTS_DIR / "THESIS362_table_manifest.json", table_manifest)
    _write_json(REPORTS_DIR / "THESIS362_figure_manifest.json", figure_manifest)
    _write_json(REPORTS_DIR / "THESIS362_visual_source_manifest.json", visual_source_manifest)

    manifest_payload = {
        "task_name": "THESIS362_FINAL360M_refresh",
        "backups": backup_manifest,
        "updated_tables": [item["file_path"] for item in table_manifest],
        "updated_figures": [item["file_path"] for item in figure_manifest],
        "old_final360i_marked_subset_trained_candidate": True,
        "metrics_modified": False,
        "checkpoints_committed": False,
    }
    _write_json(REPORTS_DIR / "THESIS362_FINAL360M_table_update_manifest.json", manifest_payload)

    refresh_md = "\n".join(
        [
            "# THESIS362 FINAL360M refresh report",
            "",
            f"- checkpoint used: `{FINAL360M_CKPT}`",
            "- training executed: `false`",
            "- eval-only trajectory refresh executed: `true`",
            "- refreshed pair table now promotes `FINAL360M` and demotes `FINAL360I` to subset-trained candidate",
            "- refreshed trajectory table now includes FINAL360M direct / ODOM360A / ODOM360B alongside old backends and BASE360D",
            "- updated figure set includes pair-level bar, trajectory metric comparison, and refreshed ridge_to_lake overlay",
            f"- recommended FINAL360M trajectory backend for thesis narrative: `{best_final360m_backend}`",
        ]
    )
    _write_md(REPORTS_DIR / "THESIS362_FINAL360M_refresh_report.md", refresh_md)

    verification = {
        "pair_level_scale_path_transfers_to_trajectory": pair_scale_path_transfers,
        "worse_tdir_likely_hurts_shape": tdir_worse_hurts_shape,
        "old_odom360a_conclusion_still_holds": odom360a_conclusion_still_holds,
        "trajectory_backend_main_result_should_be_updated": needs_backend_main_result_update,
        "recommended_final360m_backend": best_final360m_backend,
    }
    _write_json(REPORTS_DIR / "FINAL360M_final_result_verification.json", {**_read_json(REPORTS_DIR / "FINAL360M_final_result_verification.json"), "trajectory_refresh": verification})


if __name__ == "__main__":
    main()
