#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from s5e2_adjacent_dense_lib import write_json

DEFAULT_REPORT_PATH = "reports/s5e19b_noop_export_path_audit.md"
DEFAULT_CHECKPOINT_PATH = "checkpoints/S5E19B_noop_export_path_audit.json"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path, ref: Optional[Path] = None) -> Dict[str, Any]:
    stat = path.stat()
    payload = {
        "path": str(path),
        "sha256": _sha256(path),
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime,
    }
    if ref is not None and ref.exists():
        rstat = ref.stat()
        payload["identical"] = payload["sha256"] == _sha256(ref)
        payload["near_identical"] = (payload["file_size"] == rstat.st_size) and not payload["identical"]
    return payload


def _quat_angle_deg(q1: Iterable[float], q2: Iterable[float]) -> float:
    a = list(q1)
    b = list(q2)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _load_tum(path: Path) -> Dict[float, Dict[str, List[float]]]:
    out: Dict[float, Dict[str, List[float]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        out[float(p[0])] = {
            "t": [float(p[1]), float(p[2]), float(p[3])],
            "q": [float(p[4]), float(p[5]), float(p[6]), float(p[7])],
        }
    return out


def _vec_diff(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv)))


def _trajectory_diff(s5e15_tum: Path, s5e19_tum: Path) -> Dict[str, Any]:
    a = _load_tum(s5e15_tum)
    b = _load_tum(s5e19_tum)
    common = sorted(set(a) & set(b))
    tdiffs: List[float] = []
    qdiffs: List[float] = []
    identical = 0
    near_identical = 0
    for ts in common:
        td = _vec_diff(a[ts]["t"], b[ts]["t"])
        qd = _quat_angle_deg(a[ts]["q"], b[ts]["q"])
        tdiffs.append(td)
        qdiffs.append(qd)
        if td == 0.0 and qd == 0.0:
            identical += 1
        if td <= 1.0e-9 and qd <= 1.0e-6:
            near_identical += 1
    exact_copy = bool(common) and identical == len(common)
    near_noop = bool(common) and near_identical == len(common)
    return {
        "num_common_timestamps": len(common),
        "max_translation_difference": max(tdiffs) if tdiffs else None,
        "mean_translation_difference": sum(tdiffs) / len(tdiffs) if tdiffs else None,
        "max_quaternion_angular_difference_deg": max(qdiffs) if qdiffs else None,
        "mean_quaternion_angular_difference_deg": sum(qdiffs) / len(qdiffs) if qdiffs else None,
        "identical_pose_count": identical,
        "near_identical_pose_count": near_identical,
        "trajectory_exact_copy": exact_copy,
        "trajectory_near_noop": near_noop,
    }


def _edge_metrics_diff(s5e15_metrics: Dict[str, Any], s5e19_metrics: Dict[str, Any], s5e15_prov: List[Dict[str, Any]], s5e19_prov: List[Dict[str, Any]]) -> Dict[str, Any]:
    del s5e15_metrics, s5e19_metrics
    diffs_rot: List[float] = []
    diffs_tdir: List[float] = []
    diffs_tmag: List[float] = []
    anti_changed = 0
    changed_edges = 0
    unchanged_edges = 0
    for p15, p19 in zip(s5e15_prov, s5e19_prov):
        m15 = p15.get("metric_preview", {})
        m19 = p19.get("metric_preview", {})
        drot = abs(float(m15.get("rot_deg", 0.0)) - float(m19.get("rot_deg", 0.0)))
        dtdir = abs(float(m15.get("tdir_deg", 0.0)) - float(m19.get("tdir_deg", 0.0)))
        dtmag = abs(float(m15.get("tmag_ratio", 0.0)) - float(m19.get("tmag_ratio", 0.0)))
        diffs_rot.append(drot)
        diffs_tdir.append(dtdir)
        diffs_tmag.append(dtmag)
        if bool(m15.get("anti_parallel_flag")) != bool(m19.get("anti_parallel_flag")):
            anti_changed += 1
        if drot == 0.0 and dtdir == 0.0 and dtmag == 0.0 and p15.get("translation_direction") == p19.get("final_translation_direction", p19.get("translation_direction")):
            unchanged_edges += 1
        else:
            changed_edges += 1
    return {
        "mean_abs_diff_rot": sum(diffs_rot) / len(diffs_rot) if diffs_rot else None,
        "mean_abs_diff_tdir": sum(diffs_tdir) / len(diffs_tdir) if diffs_tdir else None,
        "anti_parallel_flag_changed_count": anti_changed,
        "tmag_ratio_mean_abs_diff": sum(diffs_tmag) / len(diffs_tmag) if diffs_tmag else None,
        "changed_edge_count": changed_edges,
        "unchanged_edge_count": unchanged_edges,
        "edge_metrics_noop": changed_edges == 0,
    }


def _provenance_delta_audit(s5e15_prov: List[Dict[str, Any]], s5e19_prov: List[Dict[str, Any]]) -> Dict[str, Any]:
    zero_tdir = 0
    tdir_norms: List[float] = []
    zero_tmag = 0
    tmag_abs: List[float] = []
    sign_scores = set()
    final_equals_coarse = 0
    final_equals_s5e15 = 0
    for p15, p19 in zip(s5e15_prov, s5e19_prov):
        delta_tdir = p19.get("delta_tdir", [0.0, 0.0, 0.0])
        delta_norm = _vec_diff(delta_tdir, [0.0, 0.0, 0.0])
        tdir_norms.append(delta_norm)
        if delta_norm <= 1.0e-12:
            zero_tdir += 1
        delta_log_tmag = abs(float(p19.get("delta_log_tmag", 0.0)))
        tmag_abs.append(delta_log_tmag)
        if delta_log_tmag <= 1.0e-12:
            zero_tmag += 1
        sign_scores.add(round(float(p19.get("sign_score", 0.0)), 12))
        final_dir = p19.get("final_translation_direction", p19.get("translation_direction"))
        coarse_dir = p19.get("tdir_coarse", p15.get("translation_direction"))
        if final_dir == coarse_dir:
            final_equals_coarse += 1
        if final_dir == p15.get("translation_direction") and abs(float(p19.get("final_translation_magnitude", p19.get("translation_magnitude", 0.0))) - float(p15.get("translation_magnitude", 0.0))) <= 1.0e-12:
            final_equals_s5e15 += 1
    return {
        "delta_tdir_zero_count": zero_tdir,
        "delta_tdir_norm_mean": sum(tdir_norms) / len(tdir_norms) if tdir_norms else None,
        "delta_log_tmag_zero_count": zero_tmag,
        "delta_log_tmag_abs_mean": sum(tmag_abs) / len(tmag_abs) if tmag_abs else None,
        "sign_score_unique_count": len(sign_scores),
        "final_equals_coarse_count": final_equals_coarse,
        "final_equals_s5e15_count": final_equals_s5e15,
        "refinement_head_effective": zero_tdir < len(s5e19_prov),
        "scale_head_effective": zero_tmag < len(s5e19_prov),
        "sign_head_effective": len(sign_scores) > 1,
    }


def _bool_text(text: str, needle: str) -> bool:
    return needle in text


def _training_path_audit(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "real_training_executed": all(tok in text for tok in ["optimizer", "losses", "attempted"]),
        "optimizer_step_present": _bool_text(text, "optimizer.step") or _bool_text(text, "opt.step"),
        "learned_weights_saved": _bool_text(text, "torch.save") and _bool_text(text, "state_dict"),
        "smoke_policy_only": _bool_text(text, "s5e19_smoke_policy.json"),
        "losses_actually_computed": _bool_text(text, "loss") and _bool_text(text, "classification"),
    }


def _export_path_audit(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "loads_s5e19_checkpoint": "args.candidate" in text or "candidate" in text,
        "loads_s5e15_checkpoint": "s5e15_checkpoint" in text or "S5E15_PROV" in text,
        "fallback_to_s5e15": "S5E15_PROV" in text or "s5e15_prov" in text,
        "applies_delta_tdir": "delta_tdir" in text,
        "applies_delta_log_tmag": "delta_log_tmag" in text,
        "applies_sign_score": "sign_score" in text,
        "likely_noop_export": "final_local_dir = tdir_coarse.copy()" in text or "final_local_dir = tdir_coarse" in text,
    }


def _evaluator_path_audit(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "reads_s5e19_trajectory": "args.trajectory" in text,
        "reads_s5e15_trajectory": "S5E15_REF" in text and "path_ratio" in text,
        "hardcoded_s5e15_metrics": "S5E15_REF" in text,
        "likely_old_metrics_reuse": "S5E15_REF" in text and "signed_tdir_mean_deg" in text,
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E19B no-op export path audit",
        "",
        "## 执行摘要",
        f"最终分类：`{payload['final_classification']}`。",
        "",
        "## 为什么 S5E19 指标几乎等于 S5E15",
        "这次审计主要区分：真实无提升，还是 training smoke-only、export fallback、evaluator 复用旧路径导致的表面不变。",
        "",
        "## trajectory diff",
        str(payload["trajectory_diff"]),
        "",
        "## edge metrics diff",
        str(payload["edge_metrics_diff"]),
        "",
        "## provenance delta audit",
        str(payload["provenance_delta_audit"]),
        "",
        "## training path audit",
        str(payload["training_path"]),
        "",
        "## export path audit",
        str(payload["export_path"]),
        "",
        "## evaluator path audit",
        str(payload["evaluator_path"]),
        "",
        "## final diagnosis",
        str(payload["diagnosis"]),
        "",
        "## 是否应该继续 S5E20",
        "如果这里确认是 smoke/no-op/fallback，先修训练或导出路径；不要把 S5E19 当真实性能实验继续往后推。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root)
    s5e15_results = Path(args.s5e15_results)
    s5e19_results = Path(args.s5e19_results)

    s5e15_tum = s5e15_results / "scene01_seq03_s5e15_traceable_dense_tum.txt"
    s5e19_tum = s5e19_results / "scene01_seq03_s5e19_traceable_dense_tum.txt"
    s5e15_metrics_path = s5e15_results / "edge_component_metrics.json"
    s5e19_metrics_path = s5e19_results / "edge_component_metrics.json"
    s5e15_ckpt = Path(args.s5e15_checkpoint)
    s5e19_ckpt = Path(args.s5e19_checkpoint)
    s5e19_train = project_root / "checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate/training_status.json"
    s5e19_smoke = project_root / "checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate/s5e19_smoke_policy.json"
    s5e19_prov_path = s5e19_results / "edge_provenance.jsonl"
    s5e15_prov_path = s5e15_results / "edge_provenance.jsonl"

    file_audit = {
        "s5e15_trajectory": _file_meta(s5e15_tum, s5e19_tum),
        "s5e15_metrics": _file_meta(s5e15_metrics_path, s5e19_metrics_path),
        "s5e15_checkpoint": _file_meta(s5e15_ckpt, s5e19_ckpt),
        "s5e19_trajectory": _file_meta(s5e19_tum, s5e15_tum),
        "s5e19_metrics": _file_meta(s5e19_metrics_path, s5e15_metrics_path),
        "s5e19_checkpoint": _file_meta(s5e19_ckpt, s5e15_ckpt),
        "s5e19_training_status": _file_meta(s5e19_train),
        "s5e19_smoke_policy": _file_meta(s5e19_smoke),
    }
    trajectory_diff = _trajectory_diff(s5e15_tum, s5e19_tum)
    s5e15_metrics = _read_json(s5e15_metrics_path)
    s5e19_metrics = _read_json(s5e19_metrics_path)
    s5e15_prov = _read_jsonl(s5e15_prov_path)
    s5e19_prov = _read_jsonl(s5e19_prov_path)
    edge_metrics_diff = _edge_metrics_diff(s5e15_metrics, s5e19_metrics, s5e15_prov, s5e19_prov)
    provenance_delta_audit = _provenance_delta_audit(s5e15_prov, s5e19_prov)
    training_path = _training_path_audit(project_root / "tools/train_s5e19_rotation_compensated_multiframe.py")
    export_path = _export_path_audit(project_root / "tools/export_s5e19_adjacent_dense_predictions.py")
    evaluator_path = _evaluator_path_audit(project_root / "tools/evaluate_s5e19_traceable_dense.py")

    diagnosis = {
        "s5e19_effectively_noop": bool(trajectory_diff["trajectory_near_noop"] and edge_metrics_diff["unchanged_edge_count"] == len(s5e15_prov)),
        "smoke_policy_only": bool(training_path["smoke_policy_only"] and not training_path["optimizer_step_present"] and not training_path["learned_weights_saved"]),
        "export_fallback_to_s5e15": bool(export_path["fallback_to_s5e15"]),
        "evaluator_reused_old_metrics": bool(evaluator_path["hardcoded_s5e15_metrics"] and evaluator_path["likely_old_metrics_reuse"]),
        "true_model_no_improvement": False,
    }
    diagnosis["true_model_no_improvement"] = not diagnosis["s5e19_effectively_noop"] and not diagnosis["smoke_policy_only"] and not diagnosis["export_fallback_to_s5e15"] and not diagnosis["evaluator_reused_old_metrics"]

    if diagnosis["smoke_policy_only"] and diagnosis["export_fallback_to_s5e15"]:
        final = "S5E19B_MIXED_NOOP_AND_SMOKE"
    elif diagnosis["export_fallback_to_s5e15"] and diagnosis["s5e19_effectively_noop"]:
        final = "S5E19B_NOOP_EXPORT_CONFIRMED"
    elif diagnosis["smoke_policy_only"]:
        final = "S5E19B_SMOKE_POLICY_NO_REAL_TRAINING"
    elif diagnosis["evaluator_reused_old_metrics"]:
        final = "S5E19B_EVALUATOR_REUSED_OLD_METRICS"
    elif diagnosis["true_model_no_improvement"]:
        final = "S5E19B_TRUE_NO_IMPROVEMENT"
    else:
        final = "S5E19B_AUDIT_ERROR"

    payload = {
        "experiment": "S5E19B_noop_and_export_path_audit",
        "file_audit": file_audit,
        "trajectory_diff": trajectory_diff,
        "edge_metrics_diff": edge_metrics_diff,
        "provenance_delta_audit": provenance_delta_audit,
        "training_path": training_path,
        "export_path": export_path,
        "evaluator_path": evaluator_path,
        "diagnosis": diagnosis,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--s5e19-checkpoint", required=True)
    p.add_argument("--s5e15-results", required=True)
    p.add_argument("--s5e19-results", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
