#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from evaluate_external_baseline_trajectory import evaluate_external_baseline_trajectory
from s5e2_adjacent_dense_lib import rot_to_quat_xyzw, write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compose_world_pose(R_w_cur: np.ndarray, t_w_cur: np.ndarray, R_BA: np.ndarray, t_BA_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_w_next = R_w_cur @ R_BA.T
    t_w_next = t_w_cur - R_w_next @ t_BA_B
    return R_w_next, t_w_next


def _write_tum(path: Path, rows: Sequence[Tuple[float, np.ndarray, np.ndarray]]) -> None:
    lines: List[str] = []
    for ts, R_w, t_w in rows:
        qx, qy, qz, qw = rot_to_quat_xyzw(np.asarray(R_w, dtype=np.float64))
        t = np.asarray(t_w, dtype=np.float64)
        lines.append(f"{float(ts):.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trajectory_for_sequence(manifest_rows: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], seq_id: str) -> Dict[str, Any]:
    gt_adj = sorted(
        [r for r in manifest_rows if r["seq_id"] == seq_id and r["pair_type"] == "adjacent" and int(r["k"]) == 1],
        key=lambda r: (float(r["timestamp_a"]), int(r["pair_index"])),
    )
    pred_map = {int(r["pair_index"]): r for r in predictions if r["seq_id"] == seq_id and r["pair_type"] == "adjacent" and int(r["k"]) == 1}
    usable = [r for r in gt_adj if int(r["pair_index"]) in pred_map]
    if not usable:
        return {"available": False, "reason": "no_adjacent_predictions"}

    first = usable[0]
    gt_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []
    pred_rows: List[Tuple[float, np.ndarray, np.ndarray]] = []

    R_w_gt = np.asarray(first["T_w_a"]["R"], dtype=np.float64)
    t_w_gt = np.asarray(first["T_w_a"]["t"], dtype=np.float64)
    R_w_pr = np.eye(3, dtype=np.float64)
    t_w_pr = np.zeros(3, dtype=np.float64)
    gt_rows.append((float(first["timestamp_a"]), R_w_gt, t_w_gt))
    pred_rows.append((float(first["timestamp_a"]), R_w_pr.copy(), t_w_pr.copy()))

    for row in usable:
        pred = pred_map[int(row["pair_index"])]
        R_gt_rel = np.asarray(row["R_BA"], dtype=np.float64)
        t_gt_rel = np.asarray(row["t_BA_B"], dtype=np.float64)
        R_pr_rel = np.asarray(pred["R_pred_BA"], dtype=np.float64)
        t_pr_rel = np.asarray(pred["tvec_pred_B"], dtype=np.float64)
        R_w_gt, t_w_gt = _compose_world_pose(R_w_gt, t_w_gt, R_gt_rel, t_gt_rel)
        R_w_pr, t_w_pr = _compose_world_pose(R_w_pr, t_w_pr, R_pr_rel, t_pr_rel)
        gt_rows.append((float(row["timestamp_b"]), R_w_gt.copy(), t_w_gt.copy()))
        pred_rows.append((float(row["timestamp_b"]), R_w_pr.copy(), t_w_pr.copy()))

    return {
        "available": True,
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "coverage": float(len(usable) / max(len(gt_adj), 1)),
        "num_adjacent_pairs_total": len(gt_adj),
        "num_adjacent_pairs_predicted": len(usable),
    }


def _weighted_mean(items: Sequence[Tuple[float, float]]) -> float | None:
    vals = [(v, w) for v, w in items if v is not None and math.isfinite(float(v)) and w > 0]
    if not vals:
        return None
    num = sum(float(v) * float(w) for v, w in vals)
    den = sum(float(w) for _v, w in vals)
    return float(num / max(den, 1.0e-12))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_json = Path(args.out_json)
    out_report = Path(args.out_report)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_json(results_dir / "edge_component_metrics.json")
    export_summary = _read_json(results_dir / "export_summary.json")
    predictions = _read_jsonl(results_dir / "predictions.jsonl")
    manifest_rows = _read_jsonl(Path(args.manifest_val)) + _read_jsonl(Path(args.manifest_test))
    dset1 = _read_json(Path(args.dset1_checkpoint))
    data3 = _read_json(Path(args.data3_checkpoint))
    s5e15 = _read_json(Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json"))

    by_seq_traj: Dict[str, Any] = {}
    ate_weighted: Dict[str, List[Tuple[float, float]]] = {"none": [], "se3": [], "sim3": []}
    for seq_id in sorted({r["seq_id"] for r in manifest_rows}):
        traj = _trajectory_for_sequence(manifest_rows, predictions, seq_id)
        if not traj.get("available"):
            by_seq_traj[seq_id] = traj
            continue
        gt_path = results_dir / f"{seq_id}_gt_tum.txt"
        est_path = results_dir / f"{seq_id}_pred_tum.txt"
        _write_tum(gt_path, traj["gt_rows"])
        _write_tum(est_path, traj["pred_rows"])
        evals = {}
        for mode in ("none", "se3", "sim3"):
            ev = evaluate_external_baseline_trajectory(gt_path=gt_path, est_path=est_path, alignment=mode)
            evals[mode] = ev
            if ev.get("status") == "ok" and ev.get("ATE") is not None:
                ate_weighted[mode].append((float(ev["ATE"]), float(ev.get("num_matched_poses", 0))))
        by_seq_traj[seq_id] = {
            "available": True,
            "coverage": traj["coverage"],
            "num_adjacent_pairs_total": traj["num_adjacent_pairs_total"],
            "num_adjacent_pairs_predicted": traj["num_adjacent_pairs_predicted"],
            "tum_local_only": {"gt": str(gt_path), "est": str(est_path)},
            "trajectory_eval": evals,
        }

    overall_component = metrics.get("component_metrics", {})
    weighted_traj = {
        "none_ate": _weighted_mean(ate_weighted["none"]),
        "se3_ate": _weighted_mean(ate_weighted["se3"]),
        "sim3_ate": _weighted_mean(ate_weighted["sim3"]),
    }
    final = "GEN5_360DVO_T57b_true_external_eval_ready" if export_summary.get("model_eval_available") else "GEN5_360DVO_T57b_true_external_eval_blocked"

    payload = {
        "experiment": "GEN5_360DVO_T57b_true_external_eval",
        "dataset": "360DVO",
        "source_model": "T57b_no_dt_multiscale_tmag_head_400/final.pt",
        "training": {
            "train_new_model": False,
            "fine_tune": False,
            "uses_360dvo_gt_for_calibration": False,
            "uses_orbslam3_teacher": False,
        },
        "export": export_summary,
        "component_metrics": {
            "overall": overall_component,
            "by_sequence": metrics.get("metrics_by_sequence", {}),
            "by_k": metrics.get("metrics_by_k", {}),
            "by_split": metrics.get("metrics_by_split", {}),
        },
        "trajectory_eval": {
            "by_sequence": by_seq_traj,
            "weighted_summary": weighted_traj,
            "path_ratio_adjacent_component": overall_component.get("path_ratio"),
        },
        "comparison_to_legacy": {
            "data2_distribution_shift": dset1.get("comparison", {}).get("data2_train_eval_shift", {}),
            "legacy_quality_risk": data3.get("final_classification"),
            "s5e15_legacy_overall": s5e15.get("component_metrics", {}).get("overall", {}),
            "caveat": "GEN5 uses 360DVO external multi-sequence data and is not directly substitutable for scene01/seq03 official legacy metrics.",
        },
        "recommendation": {
            "keep_s5e15_as_legacy_best_candidate": True,
            "use_gen5_as_external_generalization_evidence": bool(export_summary.get("model_eval_available")),
            "do_train360_baseline": True,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(out_json, payload)

    report_lines = [
        "# GEN5 360DVO T57b 真实外部评估",
        "",
        "## 1. 执行摘要",
        f"- final_classification = `{final}`",
        f"- model_eval_available = `{bool(export_summary.get('model_eval_available'))}`",
        f"- num_pairs_predicted = `{export_summary.get('num_pairs_predicted')}` / `{export_summary.get('num_pairs_total')}`",
        "",
        "## 2. Overall component metrics",
        json.dumps(overall_component, ensure_ascii=False, indent=2),
        "",
        "## 3. Metrics by sequence",
        json.dumps(metrics.get("metrics_by_sequence", {}), ensure_ascii=False, indent=2),
        "",
        "## 4. Metrics by k",
        json.dumps(metrics.get("metrics_by_k", {}), ensure_ascii=False, indent=2),
        "",
        "## 5. Trajectory / path_ratio / ATE",
        json.dumps(payload["trajectory_eval"], ensure_ascii=False, indent=2),
        "",
        "## 6. 与 legacy DATA2 / S5E15 的 caveat",
        json.dumps(payload["comparison_to_legacy"], ensure_ascii=False, indent=2),
        "",
        "## 7. 结论",
        "- 本轮不训练、不 fine-tune、不使用 360DVO GT 做 calibration。",
        "- 结果可作为外部泛化证据，但不能替代 official S5 locked legacy result。",
    ]
    out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    eval_json = {
        "overall_component": overall_component,
        "trajectory_eval": payload["trajectory_eval"],
        "comparison_to_legacy": payload["comparison_to_legacy"],
    }
    write_json(results_dir / "eval_summary.json", eval_json)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--manifest-val", required=True)
    parser.add_argument("--manifest-test", required=True)
    parser.add_argument("--dset1-checkpoint", required=True)
    parser.add_argument("--data3-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
