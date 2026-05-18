#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import thesis362_build_final_assets as thesis362


PLOT_DATA_DIR = REPO_ROOT / "thesis" / "final_assets" / "plot_data"
FIG_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS362_figure_manifest.json"
VISUAL_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS362_visual_source_manifest.json"
THESIS360_DEBUG_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS360_matching_geometry_debug_manifest.json"
THESIS364_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS364_plot_data_manifest.json"
UNAVAILABLE_PATH = PLOT_DATA_DIR / "unavailable_plot_data.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _safe_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _tidy_row(
    *,
    figure_name: str,
    group: str,
    model: str,
    metric: str,
    value: Any,
    split: str,
    source_file: str,
    note: str,
    **extra: Any,
) -> Dict[str, Any]:
    row = {
        "figure_name": figure_name,
        "group": group,
        "model": model,
        "metric": metric,
        "value": value,
        "split": split,
        "source_file": source_file,
        "note": note,
    }
    row.update(extra)
    return row


def _trajectory_row(
    *,
    trajectory_name: str,
    index: int,
    point: np.ndarray,
    source_file: str,
) -> Dict[str, Any]:
    return {
        "trajectory_name": trajectory_name,
        "index": index,
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
        "source_file": source_file,
    }


def _bucket_row(
    *,
    bucket_type: str,
    bucket_name: str,
    metric: str,
    value: Any,
    count: int,
    source_file: str,
    figure_name: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "figure_name": figure_name,
        "bucket_type": bucket_type,
        "bucket_name": bucket_name,
        "metric": metric,
        "value": value,
        "count": count,
        "source_file": source_file,
        "note": note,
    }


def _figure_stem(figure_path: str) -> str:
    return Path(figure_path).stem


def _collect_visual_manifest_lookup() -> Dict[str, Dict[str, Any]]:
    visual_manifest = _read_json(VISUAL_MANIFEST_PATH)
    return {item["output_file"]: item for item in visual_manifest}


def _export_pair_main(pair_sources: Mapping[str, Dict[str, Any]], fig_meta: Dict[str, Any]) -> Path:
    figure_path = fig_meta["file_path"]
    stem = _figure_stem(figure_path)
    out_path = PLOT_DATA_DIR / f"{stem}.csv"
    models = ["T57b", "TRAIN360C", "FINAL360I", "BASE360D"]
    metrics = ["rot_mean_deg", "signed_tdir_mean_deg", "anti_parallel_rate", "tmag_median_ratio", "path_ratio"]
    rows: List[Dict[str, Any]] = []
    for model in models:
        payload = pair_sources[model]
        for metric in metrics:
            rows.append(
                _tidy_row(
                    figure_name=stem,
                    group="pair_main_results",
                    model=model,
                    metric=metric,
                    value=payload["metrics"].get(metric),
                    split=str(payload["split"]),
                    source_file=";".join(payload["source_files"]),
                    note=payload["notes"],
                )
            )
    _write_csv(out_path, ["figure_name", "group", "model", "metric", "value", "split", "source_file", "note"], rows)
    return out_path


def _export_ablation(ablation_sources: Mapping[str, Dict[str, Any]], fig_meta: Dict[str, Any]) -> Path:
    stem = _figure_stem(fig_meta["file_path"])
    out_path = PLOT_DATA_DIR / f"{stem}.csv"
    models = ["PlainPairVO", "NoSphericalGeometry", "NoCrossImageInteraction", "SingleStagePoseRegression", "FINAL360I"]
    metrics = ["rot_mean_deg", "signed_tdir_mean_deg", "anti_parallel_rate", "tmag_median_ratio", "path_ratio"]
    rows: List[Dict[str, Any]] = []
    for model in models:
        payload = ablation_sources[model]
        for metric in metrics:
            rows.append(
                _tidy_row(
                    figure_name=stem,
                    group="ablation_results",
                    model=model,
                    metric=metric,
                    value=payload["metrics"].get(metric),
                    split="test",
                    source_file=";".join(payload["source_files"]),
                    note="Structural ablation result exported for manual redrawing.",
                )
            )
    _write_csv(out_path, ["figure_name", "group", "model", "metric", "value", "split", "source_file", "note"], rows)
    return out_path


def _export_trajectory_metric(trajectory_sources: Mapping[str, Dict[str, Any]], fig_meta: Dict[str, Any]) -> Path:
    stem = _figure_stem(fig_meta["file_path"])
    out_path = PLOT_DATA_DIR / f"{stem}.csv"
    models = ["TRAIN360E", "SEQ360B", "ODOM360A", "ODOM360B", "BASE360D"]
    rows: List[Dict[str, Any]] = []
    for model in models:
        payload = trajectory_sources[model]["metrics"]
        for metric in ["ate_none", "ate_se3", "ate_sim3"]:
            rows.append(
                _tidy_row(
                    figure_name=stem,
                    group="trajectory_metrics",
                    model=model,
                    metric=metric,
                    value=payload[metric]["rmse"],
                    split="test",
                    source_file=";".join(trajectory_sources[model]["source_files"]),
                    note=trajectory_sources[model]["notes"],
                )
            )
        rows.append(
            _tidy_row(
                figure_name=stem,
                group="trajectory_metrics",
                model=model,
                metric="trajectory_path_ratio",
                value=payload.get("trajectory_path_ratio"),
                split="test",
                source_file=";".join(trajectory_sources[model]["source_files"]),
                note=trajectory_sources[model]["notes"],
            )
        )
    _write_csv(out_path, ["figure_name", "group", "model", "metric", "value", "split", "source_file", "note"], rows)
    return out_path


def _load_tum_points(path: Path) -> np.ndarray:
    rows = thesis362._load_tum(path)
    return np.asarray([row[2] for row in rows], dtype=np.float64)


def _export_trajectory_overlay(fig_meta: Dict[str, Any]) -> List[Path]:
    stem = _figure_stem(fig_meta["file_path"])
    seq_name = "ridge_to_lake"
    sources = {
        "GT": REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "local_window_pose_fusion" / seq_name / "gt_tum.txt",
        "ODOM360A-direct": REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "direct_composition" / seq_name / "pred_tum.txt",
        "ODOM360A": REPO_ROOT / "external_baselines" / "results" / "odom360a_lightweight_trajectory_fusion" / "test" / "local_window_pose_fusion" / seq_name / "pred_tum.txt",
        "ODOM360B": REPO_ROOT / "external_baselines" / "results" / "odom360b_local_pose_graph_kstep" / "test" / "pg_k3" / seq_name / "pred_tum.txt",
        "BASE360D": REPO_ROOT / "external_baselines" / "results" / "base360_hkust_360dvo_official" / "test" / seq_name / "pred_tum.txt",
    }
    written: List[Path] = []
    for trajectory_name, src in sources.items():
        points = _load_tum_points(src)
        rows = [_trajectory_row(trajectory_name=trajectory_name, index=i, point=point, source_file=_safe_rel(src)) for i, point in enumerate(points)]
        out_path = PLOT_DATA_DIR / f"{stem}__{trajectory_name.replace('/', '_')}.csv"
        _write_csv(out_path, ["trajectory_name", "index", "x", "y", "z", "source_file"], rows)
        written.append(out_path)
    summary_path = PLOT_DATA_DIR / f"{stem}.json"
    _write_json(
        summary_path,
        {
            "figure_name": stem,
            "sequence": seq_name,
            "trajectory_files": [_safe_rel(path) for path in written],
            "note": "Per-trajectory XYZ data exported for the representative overlay figure.",
        },
    )
    written.append(summary_path)
    return written


def _export_training_curve(train_log_path: Path, figure_name: str, model: str, is_final360i: bool) -> Path:
    rows = thesis362._parse_train_log_rows(train_log_path)
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        update = int(row["epoch"])
        if is_final360i:
            metrics = row["val_final_metrics"]
            score = row["val_score"]
            train_loss = row["train_loss_total"]
        else:
            metrics = row.get("val_metrics", row.get("minival_metrics", {}))
            score = row.get("val_score", row.get("minival_score"))
            train_loss = row.get("train_loss_total")
        metric_items = {
            "rot_mean_deg": metrics.get("rot_mean_deg"),
            "signed_tdir_mean_deg": metrics.get("signed_tdir_mean_deg"),
            "path_ratio": metrics.get("path_ratio"),
            "joint_or_minival_score": score,
            "train_loss_total": train_loss,
        }
        for metric, value in metric_items.items():
            out_rows.append(
                {
                    "model": model,
                    "update": update,
                    "epoch": update,
                    "metric": metric,
                    "value": value,
                    "source_file": _safe_rel(train_log_path),
                }
            )
    out_path = PLOT_DATA_DIR / f"{figure_name}.csv"
    _write_csv(out_path, ["model", "update", "epoch", "metric", "value", "source_file"], out_rows)
    return out_path


def _export_joint_vs_single(train_log_path: Path, figure_name: str) -> Path:
    rows = thesis362._parse_train_log_rows(train_log_path)
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        epoch = int(row["epoch"])
        out_rows.append(
            {
                "model": "FINAL360I",
                "update": epoch,
                "epoch": epoch,
                "metric": "joint_score",
                "value": row["val_score"],
                "source_file": _safe_rel(train_log_path),
            }
        )
        out_rows.append(
            {
                "model": "FINAL360I",
                "update": epoch,
                "epoch": epoch,
                "metric": "signed_tdir_mean_deg",
                "value": row["val_final_metrics"]["signed_tdir_mean_deg"],
                "source_file": _safe_rel(train_log_path),
            }
        )
    out_path = PLOT_DATA_DIR / f"{figure_name}.csv"
    _write_csv(out_path, ["model", "update", "epoch", "metric", "value", "source_file"], out_rows)
    return out_path


def _bucketize(rows: Sequence[Dict[str, Any]], bucket_edges: Sequence[float], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        bucket_rows = [row for row in rows if lo <= float(row[key]) < hi]
        if not bucket_rows:
            out.append({"label": f"[{lo:.4f},{hi:.4f})", "mean": None, "count": 0, "anti_parallel_rate": None})
            continue
        out.append(
            {
                "label": f"[{lo:.4f},{hi:.4f})",
                "mean": float(np.mean([float(r["signed_tdir_error_deg"]) for r in bucket_rows])),
                "count": len(bucket_rows),
                "anti_parallel_rate": float(np.mean([1.0 if r.get("anti_parallel") else 0.0 for r in bucket_rows])),
            }
        )
    return out


def _export_bucket_figure(
    *,
    figure_name: str,
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Optional[Sequence[Dict[str, Any]]],
    key: str,
    bucket_type: str,
    source_file: str,
    compare_label_b: Optional[str] = None,
) -> Path:
    values = [float(row[key]) for row in rows_a if row.get("finite", True)]
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
    quantiles[-1] += 1.0e-9
    buckets_a = _bucketize(rows_a, quantiles, key)
    buckets_b = _bucketize(rows_b or [], quantiles, key) if rows_b else []
    out_rows: List[Dict[str, Any]] = []
    for idx, bucket in enumerate(buckets_a):
        out_rows.append(
            _bucket_row(
                figure_name=figure_name,
                bucket_type=bucket_type,
                bucket_name=bucket["label"],
                metric="signed_tdir_mean_deg",
                value=bucket["mean"],
                count=int(bucket["count"]),
                source_file=source_file,
                note="Primary bucket series for FINAL360I.",
            )
        )
        out_rows.append(
            _bucket_row(
                figure_name=figure_name,
                bucket_type=bucket_type,
                bucket_name=bucket["label"],
                metric="anti_parallel_rate",
                value=bucket["anti_parallel_rate"],
                count=int(bucket["count"]),
                source_file=source_file,
                note="Primary bucket series for FINAL360I.",
            )
        )
        if rows_b:
            bucket_b = buckets_b[idx]
            out_rows.append(
                _bucket_row(
                    figure_name=figure_name,
                    bucket_type=f"{bucket_type}_{compare_label_b or 'compare'}",
                    bucket_name=bucket_b["label"],
                    metric="signed_tdir_mean_deg",
                    value=bucket_b["mean"],
                    count=int(bucket_b["count"]),
                    source_file=source_file,
                    note=f"Comparison bucket series for {compare_label_b}.",
                )
            )
            out_rows.append(
                _bucket_row(
                    figure_name=figure_name,
                    bucket_type=f"{bucket_type}_{compare_label_b or 'compare'}",
                    bucket_name=bucket_b["label"],
                    metric="anti_parallel_rate",
                    value=bucket_b["anti_parallel_rate"],
                    count=int(bucket_b["count"]),
                    source_file=source_file,
                    note=f"Comparison bucket series for {compare_label_b}.",
                )
            )
    out_path = PLOT_DATA_DIR / f"{figure_name}.csv"
    _write_csv(out_path, ["figure_name", "bucket_type", "bucket_name", "metric", "value", "count", "source_file", "note"], out_rows)
    return out_path


def _export_antiparallel_distribution(rows: Sequence[Dict[str, Any]], figure_name: str, source_file: str) -> Path:
    anti_vals = [float(row["signed_tdir_error_deg"]) for row in rows if row.get("anti_parallel")]
    non_vals = [float(row["signed_tdir_error_deg"]) for row in rows if not row.get("anti_parallel")]
    all_vals = anti_vals + non_vals
    bins = np.histogram_bin_edges(all_vals, bins=30)
    anti_hist, _ = np.histogram(anti_vals, bins=bins)
    non_hist, _ = np.histogram(non_vals, bins=bins)
    out_rows: List[Dict[str, Any]] = []
    for idx in range(len(bins) - 1):
        bucket_name = f"[{bins[idx]:.4f},{bins[idx + 1]:.4f})"
        out_rows.append(
            _bucket_row(
                figure_name=figure_name,
                bucket_type="anti_parallel_hist_not_anti",
                bucket_name=bucket_name,
                metric="count",
                value=int(non_hist[idx]),
                count=int(non_hist[idx]),
                source_file=source_file,
                note="Histogram bin count for non anti-parallel rows.",
            )
        )
        out_rows.append(
            _bucket_row(
                figure_name=figure_name,
                bucket_type="anti_parallel_hist_anti",
                bucket_name=bucket_name,
                metric="count",
                value=int(anti_hist[idx]),
                count=int(anti_hist[idx]),
                source_file=source_file,
                note="Histogram bin count for anti-parallel rows.",
            )
        )
    out_path = PLOT_DATA_DIR / f"{figure_name}.csv"
    _write_csv(out_path, ["figure_name", "bucket_type", "bucket_name", "metric", "value", "count", "source_file", "note"], out_rows)
    return out_path


def _export_negative_compare(negative_sources: Mapping[str, Dict[str, Any]], figure_name: str) -> Path:
    out_rows: List[Dict[str, Any]] = []
    for model in ["FINAL360I", "FINAL360J", "FINAL360K", "FINAL360L"]:
        payload = negative_sources[model]
        for metric in ["signed_tdir_mean_deg", "anti_parallel_rate", "path_ratio"]:
            out_rows.append(
                _tidy_row(
                    figure_name=figure_name,
                    group="negative_validation_compare",
                    model=model,
                    metric=metric,
                    value=payload["metrics"].get(metric),
                    split=str(payload.get("split") or "val"),
                    source_file="validation_train_logs",
                    note=f"{model} validation-only retained negative-branch summary.",
                )
            )
    out_path = PLOT_DATA_DIR / f"{figure_name}.csv"
    _write_csv(out_path, ["figure_name", "group", "model", "metric", "value", "split", "source_file", "note"], out_rows)
    return out_path


def _export_matching_geometry_unavailable(figures: Sequence[Dict[str, Any]]) -> Tuple[List[Path], List[Dict[str, Any]]]:
    unavailable = _read_json(THESIS360_DEBUG_MANIFEST_PATH) if THESIS360_DEBUG_MANIFEST_PATH.exists() else {}
    per_figure_paths: List[Path] = []
    unavailable_entries: List[Dict[str, Any]] = []
    for figure in figures:
        stem = _figure_stem(figure["file_path"])
        payload = {
            "figure_name": stem,
            "status": "unavailable_plot_data",
            "reason": "Current THESIS360 assets retain figure images and metadata, but not the raw token-level matrices or debug tensors required for plot-data reconstruction without rerunning eval-only forward.",
            "source_files": figure["source_files"],
            "task_manifest": unavailable,
        }
        out_path = PLOT_DATA_DIR / f"{stem}.json"
        _write_json(out_path, payload)
        per_figure_paths.append(out_path)
        unavailable_entries.append(payload)
    return per_figure_paths, unavailable_entries


def main() -> None:
    _ensure_dir(PLOT_DATA_DIR)
    figure_manifest = _read_json(FIG_MANIFEST_PATH)
    visual_lookup = _collect_visual_manifest_lookup()
    pair_sources = thesis362._build_pair_sources()
    ablation_sources = thesis362._build_ablation_sources()
    trajectory_sources = thesis362._build_trajectory_sources(pair_sources)
    negative_sources = thesis362._build_negative_sources()

    exported_manifest: List[Dict[str, Any]] = []
    unavailable_manifest: List[Dict[str, Any]] = []

    final360i_rows = _read_jsonl(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic" / "final360i_test_pair_rows.jsonl")
    seq360b_rows = _read_jsonl(REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic" / "seq360b_test_adjacent_rows.jsonl")
    diag_source = "external_baselines/results/data360a_tdir_regime_diagnostic"

    figure_groups = {item["file_path"]: item for item in figure_manifest}

    for figure_path, fig_meta in figure_groups.items():
        stem = _figure_stem(figure_path)
        written_paths: List[Path] = []
        status = "exported"
        reason = ""
        if figure_path.endswith("pair_level_main_results_bar.png"):
            written_paths = [_export_pair_main(pair_sources, fig_meta)]
        elif figure_path.endswith("structural_ablation_multimetric.png"):
            written_paths = [_export_ablation(ablation_sources, fig_meta)]
        elif figure_path.endswith("trajectory_metric_comparison.png"):
            written_paths = [_export_trajectory_metric(trajectory_sources, fig_meta)]
        elif figure_path.endswith("trajectory_overlay_ridge_to_lake.png"):
            written_paths = _export_trajectory_overlay(fig_meta)
        elif figure_path.endswith("final360i_validation_curves.png"):
            written_paths = [
                _export_training_curve(
                    REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "train_log.jsonl",
                    stem,
                    "FINAL360I",
                    True,
                )
            ]
        elif figure_path.endswith("no_cross_interaction_validation_curves.png"):
            written_paths = [
                _export_training_curve(
                    REPO_ROOT / "checkpoints" / "ABLDVO2_NoCrossImageInteraction" / "train_log.jsonl",
                    stem,
                    "ABLDVO2_NoCrossImageInteraction",
                    False,
                )
            ]
        elif figure_path.endswith("single_stage_validation_curves.png"):
            written_paths = [
                _export_training_curve(
                    REPO_ROOT / "checkpoints" / "ABLDVO3_SingleStagePoseRegression" / "train_log.jsonl",
                    stem,
                    "ABLDVO3_SingleStagePoseRegression",
                    False,
                )
            ]
        elif figure_path.endswith("final360i_joint_vs_single_best.png"):
            written_paths = [
                _export_joint_vs_single(
                    REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "train_log.jsonl",
                    stem,
                )
            ]
        elif figure_path.endswith("data360a_tmag_bucket_signed_tdir.png"):
            written_paths = [
                _export_bucket_figure(
                    figure_name=stem,
                    rows_a=final360i_rows,
                    rows_b=None,
                    key="gt_tmag",
                    bucket_type="gt_tmag",
                    source_file=diag_source,
                )
            ]
        elif figure_path.endswith("data360a_rotation_bucket_signed_tdir.png"):
            written_paths = [
                _export_bucket_figure(
                    figure_name=stem,
                    rows_a=final360i_rows,
                    rows_b=None,
                    key="gt_rot_angle_deg",
                    bucket_type="gt_rot_angle_deg",
                    source_file=diag_source,
                )
            ]
        elif figure_path.endswith("anti_parallel_regime_distribution.png"):
            written_paths = [_export_antiparallel_distribution(final360i_rows, stem, diag_source)]
        elif figure_path.endswith("final360i_j_k_l_validation_compare.png"):
            written_paths = [_export_negative_compare(negative_sources, stem)]
        elif figure_path.endswith("final360i_vs_seq360b_bucket_compare.png"):
            written_paths = [
                _export_bucket_figure(
                    figure_name=stem,
                    rows_a=[row for row in final360i_rows if row["pair_type"] == "adjacent"],
                    rows_b=seq360b_rows,
                    key="gt_tmag",
                    bucket_type="gt_tmag",
                    source_file=diag_source,
                    compare_label_b="SEQ360B",
                )
            ]
        elif "/matching_geometry/" in figure_path:
            written_paths, unavailable_entries = _export_matching_geometry_unavailable([fig_meta])
            unavailable_manifest.extend(unavailable_entries)
            status = "unavailable"
            reason = unavailable_entries[0]["reason"]
        else:
            status = "unavailable"
            reason = "No exporter implemented for this figure path."
            payload = {"figure_name": stem, "status": status, "reason": reason, "source_files": fig_meta["source_files"]}
            out_path = PLOT_DATA_DIR / f"{stem}.json"
            _write_json(out_path, payload)
            written_paths = [out_path]
            unavailable_manifest.append(payload)

        exported_manifest.append(
            {
                "figure_path": figure_path,
                "figure_name": stem,
                "status": status,
                "plot_data_files": [_safe_rel(path) for path in written_paths],
                "source_files": fig_meta["source_files"],
                "reason": reason,
                "notes": visual_lookup.get(figure_path, {}).get("notes", fig_meta.get("notes", "")),
            }
        )

    _write_json(UNAVAILABLE_PATH, unavailable_manifest)
    summary = {
        "task_name": "THESIS364_export_plot_data_for_manual_redrawing",
        "plot_data_root": _safe_rel(PLOT_DATA_DIR),
        "figures_total": len(figure_manifest),
        "figures_covered": [item["figure_name"] for item in exported_manifest if item["status"] == "exported"],
        "figures_unavailable": [item["figure_name"] for item in exported_manifest if item["status"] == "unavailable"],
        "trajectory_plot_data_exported": True,
        "training_curve_plot_data_exported": True,
        "metrics_modified": False,
        "training_executed": False,
        "checkpoints_modified": False,
        "entries": exported_manifest,
        "unavailable_plot_data_file": _safe_rel(UNAVAILABLE_PATH),
    }
    _write_json(THESIS364_MANIFEST_PATH, summary)
    print(
        json.dumps(
            {
                "written": True,
                "manifest": _safe_rel(THESIS364_MANIFEST_PATH),
                "covered": len(summary["figures_covered"]),
                "unavailable": len(summary["figures_unavailable"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
