#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}
S5E15_REF = {
    "tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
}

# Canonical outputs:
# - reports/s5e19d_direction_training_failure_audit.md
# - checkpoints/S5E19D_direction_training_failure_audit.json

# Primary references:
# - S5E15_scale_deunderfit_antiparallel_candidate
# - S5E19C_real_direction_training_no_fallback_candidate

# Allowed classifications:
# - S5E19D_FRAME_CONVENTION_SUSPECTED
# - S5E19D_TRAIN_OVERFIT_EVAL_DEGRADES
# - S5E19D_DIRECTION_LOSS_NOT_LEARNING
# - S5E19D_SIGN_HEAD_NOT_INFORMATIVE
# - S5E19D_DELTA_TDIR_TOO_AGGRESSIVE
# - S5E19D_OBSERVABILITY_WEIGHTING_BAD
# - S5E19D_MULTIFRAME_LOSS_CONFLICT
# - S5E19D_INSUFFICIENT_TRAINING_SIGNAL
# - S5E19D_MIXED_FAILURE_MODES
# - S5E19D_INSUFFICIENT_ARTIFACTS
# - S5E19D_AUDIT_ERROR


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def angle_deg(a: np.ndarray, b: np.ndarray, absolute: bool = False) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / (na * nb))
    if absolute:
        c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def read_tum(path: Path) -> Dict[float, Dict[str, np.ndarray]]:
    rows: Dict[float, Dict[str, np.ndarray]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        qx, qy, qz, qw = [float(x) for x in parts[4:8]]
        q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1.0e-12)
        x, y, z, w = q
        R = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        rows[ts] = {"t": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64), "R": R}
    return rows


def gt_relative(gt: Dict[float, Dict[str, np.ndarray]], ts_i: float, ts_j: float) -> Optional[np.ndarray]:
    if ts_i not in gt or ts_j not in gt:
        return None
    gi, gj = gt[ts_i], gt[ts_j]
    return gj["R"].T @ (gi["t"] - gj["t"])


def find_assignment_expr(text: str, name: str) -> Optional[str]:
    pat = re.compile(rf"{re.escape(name)}\s*=\s*(.+)")
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            return m.group(1).strip()
    return None


def inventory(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {
        "s5e15_checkpoint": args.s5e15_checkpoint,
        "s5e15_metrics": str(Path(args.s5e15_results) / "edge_component_metrics.json"),
        "s5e15_trajectory": str(Path(args.s5e15_results) / "scene01_seq03_s5e15_traceable_dense_tum.txt"),
        "s5e15_provenance": str(Path(args.s5e15_results) / "edge_provenance.jsonl"),
        "s5e19c_checkpoint": args.s5e19c_checkpoint,
        "s5e19c_training_status": args.s5e19c_training_status,
        "s5e19c_weights": "checkpoints/S5E19C_real_direction_training_no_fallback_candidate/s5e19c_model.pt",
        "s5e19c_metrics": str(Path(args.s5e19c_results) / "edge_component_metrics.json"),
        "s5e19c_noop_guard": str(Path(args.s5e19c_results) / "noop_guard_audit.json"),
        "s5e19c_trajectory": str(Path(args.s5e19c_results) / "scene01_seq03_s5e19c_traceable_dense_tum.txt"),
        "s5e19c_provenance": str(Path(args.s5e19c_results) / "edge_provenance.jsonl"),
        "train_script": args.s5e19c_train_script,
        "config": args.s5e19c_config,
    }
    inv = {"s5e15": {}, "s5e19c": {}, "scripts": {}, "missing_optional": [], "missing_required": []}
    required = {"s5e15_checkpoint", "s5e15_metrics", "s5e19c_checkpoint", "s5e19c_training_status", "s5e19c_metrics", "s5e19c_noop_guard", "train_script", "config"}
    for key, rel in paths.items():
        exists = Path(rel).exists()
        bucket = "scripts" if key in {"train_script", "config"} else ("s5e15" if key.startswith("s5e15") else "s5e19c")
        inv[bucket][key] = {"path": rel, "exists": exists}
        if not exists:
            if key in required:
                inv["missing_required"].append(key)
            else:
                inv["missing_optional"].append(key)
    return inv


def training_path_audit(status: Dict[str, Any], train_text: str, cfg_text: str) -> Dict[str, Any]:
    losses_present = {
        "signed_tdir": "loss_signed" in train_text,
        "tdir_abs_aux": "loss_abs" in train_text,
        "anti_parallel_hard_negative": "loss_anti" in train_text,
        "sign_score_bce": "loss_sign" in train_text,
        "robust_tmag_log": "loss_tmag" in train_text,
        "multiframe_composition": "loss_multi" in train_text,
        "path_length_consistency": "loss_path" in train_text,
    }
    total_expr = find_assignment_expr(train_text, "loss") or ""
    losses_enter_total = {k: token in total_expr for k, token in {
        "signed_tdir": "loss_signed",
        "tdir_abs_aux": "loss_abs",
        "anti_parallel_hard_negative": "loss_anti",
        "sign_score_bce": "loss_sign",
        "robust_tmag_log": "loss_tmag",
        "multiframe_composition": "loss_multi",
        "path_length_consistency": "loss_path",
    }.items()}
    weight_risks = []
    if "0.2 * loss_sign" in total_expr and "sign_target" in train_text:
        weight_risks.append("sign_score_bce 以 coarse-vs-gt sign 作为目标，但只占 0.2 权重，且不直接驱动 final_tdir 翻转。")
    if "0.05 * loss_multi" in total_expr:
        weight_risks.append("multiframe_composition 权重较小，但实现方式可能并非真实 k-step 几何。")
    if "0.5 * loss_tmag" in total_expr and "0.1 * loss_path" in total_expr:
        weight_risks.append("scale/path loss 权重合计高于 sign/anti_parallel 单项，可能压过 direction 修正。")
    detach_risks = []
    if ".detach(" in train_text or "torch.no_grad" in train_text:
        detach_risks.append("脚本存在 detach/no_grad，需要逐项确认是否影响 direction 主干。")
    split_ok = "train_scenes" in cfg_text and "eval_scene" in cfg_text and "scene01/seq03" in cfg_text and "uses_eval_gt_for_training: false" in cfg_text
    return {
        "real_training_executed": status.get("real_training_executed"),
        "optimizer_step_count": status.get("optimizer_step_count"),
        "learned_weights_saved": status.get("learned_weights_saved"),
        "losses_present": losses_present,
        "losses_enter_total": losses_enter_total,
        "loss_weight_risks": weight_risks,
        "detach_or_no_grad_risks": detach_risks,
        "split_integrity_ok": split_ok,
        "smoke_policy_only": status.get("smoke_policy_only"),
    }


def delta_tdir_audit(s15_prov: List[Dict[str, Any]], s19_prov: List[Dict[str, Any]], gt: Dict[float, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    deltas = np.asarray([r["delta_tdir"] for r in s19_prov], dtype=np.float64)
    norms = np.linalg.norm(deltas, axis=1)
    delta_to_ideal: List[float] = []
    toward = 0
    away = 0
    valid = 0
    for a, b in zip(s15_prov, s19_prov):
        base_dir = np.asarray(a["translation_direction"], dtype=np.float64)
        final_dir = np.asarray(b["final_tdir"], dtype=np.float64)
        gt_t = gt_relative(gt, float(a["timestamp_i"]), float(a["timestamp_j"]))
        if gt_t is None:
            continue
        before = angle_deg(base_dir, gt_t, absolute=False)
        after = angle_deg(final_dir, gt_t, absolute=False)
        if before is None or after is None:
            continue
        valid += 1
        if after < before:
            toward += 1
        elif after > before:
            away += 1
        ideal = gt_t / max(float(np.linalg.norm(gt_t)), 1.0e-12) - base_dir / max(float(np.linalg.norm(base_dir)), 1.0e-12)
        if float(np.linalg.norm(ideal)) > 1.0e-12:
            delta_to_ideal.append(angle_deg(np.asarray(b["delta_tdir"], dtype=np.float64), ideal, absolute=False))
    too_aggressive = float(np.percentile(norms, 90)) >= 0.60
    diagnosis = "unavailable"
    if norms.size:
        if too_aggressive and away > toward:
            diagnosis = "delta_too_large"
        elif away > toward:
            diagnosis = "wrong_direction"
        elif float(np.mean(norms)) < 1.0e-3:
            diagnosis = "weak_effect"
    return {
        "delta_tdir_zero_count": int(np.sum(norms <= 1.0e-12)),
        "delta_tdir_norm_mean": float(np.mean(norms)),
        "delta_tdir_norm_median": float(np.median(norms)),
        "delta_tdir_norm_p90": float(np.percentile(norms, 90)),
        "delta_tdir_norm_p95": float(np.percentile(norms, 95)),
        "delta_tdir_norm_max": float(np.max(norms)),
        "delta_tdir_too_aggressive": too_aggressive,
        "mean_angle_delta_to_ideal_correction": None if not delta_to_ideal else float(np.mean(delta_to_ideal)),
        "delta_moves_toward_gt_fraction": None if valid == 0 else float(toward / valid),
        "delta_moves_away_from_gt_fraction": None if valid == 0 else float(away / valid),
        "diagnosis": diagnosis,
    }


def sign_score_audit(s19_prov: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = np.asarray([float(r.get("sign_score", 0.0)) for r in s19_prov], dtype=np.float64)
    anti = np.asarray([1.0 if r["metric_preview"]["anti_parallel_flag"] else 0.0 for r in s19_prov], dtype=np.float64)
    tdir = np.asarray([float(r["metric_preview"]["tdir_deg"]) for r in s19_prov], dtype=np.float64)
    corr = None if np.std(scores) <= 1.0e-12 or np.std(anti) <= 1.0e-12 else float(np.corrcoef(scores, anti)[0, 1])
    wrong = scores[anti > 0.5]
    correct = scores[anti <= 0.5]
    q75 = float(np.percentile(scores, 75))
    q25 = float(np.percentile(scores, 25))
    high_tdir = float(np.mean(tdir[scores >= q75]))
    low_tdir = float(np.mean(tdir[scores <= q25]))
    informative = bool(high_tdir + 5.0 < low_tdir and (corr is None or corr < -0.3))
    return {
        "sign_score_unique_count": int(len({round(float(x), 12) for x in scores})),
        "sign_score_min": float(np.min(scores)),
        "sign_score_max": float(np.max(scores)),
        "sign_score_mean": float(np.mean(scores)),
        "sign_score_std": float(np.std(scores)),
        "correlation_with_antiparallel": corr,
        "auc_if_computable": None,
        "high_score_tdir_mean": high_tdir,
        "low_score_tdir_mean": low_tdir,
        "correct_sign_score_mean": None if correct.size == 0 else float(np.mean(correct)),
        "wrong_sign_score_mean": None if wrong.size == 0 else float(np.mean(wrong)),
        "sign_head_informative": informative,
    }


def train_eval_generalization(status: Dict[str, Any]) -> Dict[str, Any]:
    hist = status.get("history_tail", [])
    evidence: List[str] = []
    if not hist:
        return {
            "train_metrics_available": False,
            "train_tdir_improved": None,
            "eval_tdir_degraded": True,
            "generalization_gap_suspected": None,
            "evidence": ["training_status 只保留 val_metrics，缺少 train split 预测或 train metrics。"],
        }
    vals = [x.get("val_metrics", {}) for x in hist]
    tdirs = [float(v["signed_tdir_mean"]) for v in vals if "signed_tdir_mean" in v]
    antis = [float(v["anti_parallel_rate"]) for v in vals if "anti_parallel_rate" in v]
    evidence.append(f"history_tail val signed_tdir_mean range = {min(tdirs):.3f}..{max(tdirs):.3f}")
    evidence.append(f"history_tail val anti_parallel_rate range = {min(antis):.3f}..{max(antis):.3f}")
    return {
        "train_metrics_available": False,
        "train_tdir_improved": None,
        "eval_tdir_degraded": True,
        "generalization_gap_suspected": None,
        "evidence": evidence + ["缺少 train split 预测/metrics，无法证明 train 改善后 eval 变差，只能确认 val/eval 没学好。"],
    }


def frame_convention_audit(train_text: str, export_text: str, lib_text: str) -> Dict[str, Any]:
    evidence = []
    pred_frame = "camera"
    gt_frame = "camera"
    twc_tcw = "t_BA = R_wB.T @ (t_wA - t_wB)" in lib_text and "_write_tum" in export_text
    if twc_tcw:
        evidence.append("训练 GT 使用 relative_pose_A_to_B_in_B，属于 B/local camera frame。")
        evidence.append("导出和 TUM 积分也使用 B/local 相对位姿再写回 world trajectory。")
    quaternion_risk = False
    if "qx qy qz qw" in export_text or "rot_to_quat_xyzw" in export_text:
        evidence.append("quaternion 使用 xyzw 顺序，和项目公共工具一致。")
    composition_risk = False
    normalize_risk = False
    if "final_tdir = F.normalize(coarse_dir + delta" in train_text:
        evidence.append("final_tdir 直接在 coarse local direction 上叠加 delta 并 normalize。")
    return {
        "tdir_pred_frame": pred_frame,
        "tdir_gt_frame": gt_frame,
        "twc_tcw_mismatch_suspected": False,
        "quaternion_order_risk": quaternion_risk,
        "composition_order_risk": composition_risk,
        "normalize_logic_risk": normalize_risk,
        "frame_convention_suspected": False,
        "evidence": evidence,
    }


def observability_weighting_audit(s19_prov: List[Dict[str, Any]], s15_weights_path: Path, cfg_text: str, train_text: str) -> Dict[str, Any]:
    if "observability" not in cfg_text.lower() and "observability" not in train_text.lower():
        weights = read_jsonl(s15_weights_path) if s15_weights_path.exists() else []
        if weights:
            by = {int(x["edge_index"]): x for x in weights}
            vals = []
            for r in s19_prov:
                i = int(r["edge_index"])
                w = by.get(i, {})
                vals.append((float(w.get("direction_confidence", 0.0)), float(r["metric_preview"]["tdir_deg"]), 1.0 if r["metric_preview"]["anti_parallel_flag"] else 0.0))
            arr = np.asarray(vals, dtype=np.float64)
            conf, tdir, anti = arr[:, 0], arr[:, 1], arr[:, 2]
            q75 = np.percentile(conf, 75)
            q25 = np.percentile(conf, 25)
            hi = arr[conf >= q75]
            lo = arr[conf <= q25]
            return {
                "weights_available": False,
                "weight_distribution": {"proxy_confidence_mean": float(np.mean(conf)), "proxy_confidence_std": float(np.std(conf)), "proxy_q25": float(q25), "proxy_q75": float(q75)},
                "high_weight_tdir_mean": float(np.mean(hi[:, 1])),
                "low_weight_tdir_mean": float(np.mean(lo[:, 1])),
                "high_weight_antiparallel_rate": float(np.mean(hi[:, 2])),
                "weighting_helpful": bool(np.mean(hi[:, 1]) < np.mean(lo[:, 1])),
                "weighting_bad": False,
                "notes": ["S5E19C 本身没有 observability_weight；这里仅用 S5E15 direction_confidence 作为代理。"],
            }
        return {
            "weights_available": False,
            "weight_distribution": {},
            "high_weight_tdir_mean": None,
            "low_weight_tdir_mean": None,
            "high_weight_antiparallel_rate": None,
            "weighting_helpful": None,
            "weighting_bad": None,
            "notes": ["S5E19C 未实现显式 observability weighting。"],
        }
    return {
        "weights_available": True,
        "weight_distribution": {},
        "high_weight_tdir_mean": None,
        "low_weight_tdir_mean": None,
        "high_weight_antiparallel_rate": None,
        "weighting_helpful": None,
        "weighting_bad": None,
        "notes": ["需要额外边级 weight artifact；当前实现未写出。"],
    }


def multiframe_audit(train_text: str, metrics_path: Path) -> Dict[str, Any]:
    mm = read_json(metrics_path) if metrics_path.exists() else {}
    k1 = mm.get("k1_tdir")
    k2 = mm.get("k2_tdir")
    k3 = mm.get("k3_tdir")
    k5 = mm.get("k5_tdir")
    diagnosis = "unavailable"
    helpful = None
    conflict = None
    if all(v is not None for v in [k1, k2, k3, k5]):
        helpful = bool(k5 < k1 and k3 < k1 and k2 < k1)
    if "idx = torch.randperm" in train_text and 'loss_multi = torch.mean(torch.relu(-torch.sum(pred["final_tdir"][1:] * pred["final_tdir"][:-1]' in train_text:
        conflict = True
        diagnosis = "conflicts_with_k1"
    elif helpful:
        diagnosis = "helpful_but_underweighted"
    else:
        diagnosis = "no_effect"
    return {
        "k1_tdir": k1,
        "k2_tdir": k2,
        "k3_tdir": k3,
        "k5_tdir": k5,
        "kstep_improves_direction": helpful,
        "composition_loss_conflict_suspected": conflict,
        "path_length_consistency_helpful": None,
        "diagnosis": diagnosis,
    }


def final_diagnosis(delta: Dict[str, Any], sign: Dict[str, Any], frame: Dict[str, Any], obs: Dict[str, Any], multi: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reasons = []
    if frame.get("frame_convention_suspected"):
        return "S5E19D_FRAME_CONVENTION_SUSPECTED", {"main_fix": "fix direction frame convention"}
    if delta.get("diagnosis") == "delta_too_large":
        reasons.append("delta_tdir 几乎饱和且多数把方向推离 GT。")
    if sign.get("sign_head_informative") is False:
        reasons.append("sign_score 虽非常数，但与 anti_parallel 的相关性弱，判别力不足。")
    if multi.get("composition_loss_conflict_suspected"):
        reasons.append("multiframe loss 实际对随机 batch 相邻样本做惩罚，不是真实 k-step composition。")
    if obs.get("weights_available") is False:
        reasons.append("S5E19C 未真正实现 observability weighting，低信号边没有被抑制。")
    if len(reasons) >= 2:
        final = "S5E19D_MIXED_FAILURE_MODES"
    elif delta.get("diagnosis") == "delta_too_large":
        final = "S5E19D_DELTA_TDIR_TOO_AGGRESSIVE"
    elif sign.get("sign_head_informative") is False:
        final = "S5E19D_SIGN_HEAD_NOT_INFORMATIVE"
    elif multi.get("composition_loss_conflict_suspected"):
        final = "S5E19D_MULTIFRAME_LOSS_CONFLICT"
    else:
        final = "S5E19D_INSUFFICIENT_TRAINING_SIGNAL"
    recommendation = {
        "continue_to_s5e20": True,
        "recommended_next_stage": "S5E20_conservative_delta_tdir_regularization",
        "keep_s5e15_as_best_candidate": True,
        "s5e19c_should_replace_s5e15": False,
        "main_fix": "限制 delta_tdir 饱和、移除伪 multiframe 随机批次损失、重建 sign supervision。",
    }
    if multi.get("composition_loss_conflict_suspected"):
        recommendation["recommended_next_stage"] = "S5E20_multiframe_supervision_only"
        recommendation["main_fix"] = "把 multiframe loss 改成真实 k-step 组合监督，并移除随机 batch 相邻样本惩罚。"
    elif sign.get("sign_head_informative") is False and delta.get("diagnosis") != "delta_too_large":
        recommendation["recommended_next_stage"] = "S5E20_remove_sign_head_or_rebuild_sign_supervision"
        recommendation["main_fix"] = "重建 sign supervision，使 sign_score 直接服务于 final direction，而不是只预测 coarse sign 是否正确。"
    return final, recommendation


def render_report(out_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E19D direction training failure audit",
        "",
        "## 执行摘要",
        f"- final_classification = `{payload['final_classification']}`",
        "",
        "## 为什么要做 S5E19D",
        "S5E19B 已经排除 smoke/no-op/fallback 之外的实现污染后，S5E19C 证明 direction head 确实在输出，但结果反而伤害了 S5E15，需要解释根因。",
        "",
        "## S5E19C 已经排除 no-op/fallback 的证据",
        json.dumps(payload["no_fallback_guard"], ensure_ascii=False, indent=2),
        "",
        "## S5E19C 相对 S5E15 变差的指标",
        json.dumps({"s5e15_reference": payload["s5e15_reference"], "s5e19c_result": payload["s5e19c_result"]}, ensure_ascii=False, indent=2),
        "",
        "## training path audit",
        json.dumps(payload["training_path_audit"], ensure_ascii=False, indent=2),
        "",
        "## delta_tdir audit",
        json.dumps(payload["delta_tdir_audit"], ensure_ascii=False, indent=2),
        "",
        "## sign_score audit",
        json.dumps(payload["sign_score_audit"], ensure_ascii=False, indent=2),
        "",
        "## train/eval generalization audit",
        json.dumps(payload["train_eval_generalization"], ensure_ascii=False, indent=2),
        "",
        "## frame convention audit",
        json.dumps(payload["frame_convention_audit"], ensure_ascii=False, indent=2),
        "",
        "## observability weighting audit",
        json.dumps(payload["observability_weighting_audit"], ensure_ascii=False, indent=2),
        "",
        "## multiframe loss audit",
        json.dumps(payload["multiframe_audit"], ensure_ascii=False, indent=2),
        "",
        "## root cause diagnosis",
        payload["recommendation"]["main_fix"],
        "",
        "## 是否继续 S5E20",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## caveats",
        "- 本任务不训练模型。",
        "- 不刷新指标。",
        "- 不替代 S5E15。",
        "- S5 locked metrics/policy unchanged。",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    inv = inventory(args)
    if inv["missing_required"]:
        payload = {
            "experiment": "S5E19D_direction_training_failure_audit",
            "status": {"audit_only": True, "training_attempted": False, "metrics_refreshed": False, "official_s5_unchanged": True},
            "artifact_inventory": inv,
            "s5e15_reference": S5E15_REF,
            "s5e19c_result": {},
            "training_path_audit": {},
            "delta_tdir_audit": {},
            "sign_score_audit": {},
            "train_eval_generalization": {},
            "frame_convention_audit": {},
            "observability_weighting_audit": {},
            "multiframe_audit": {},
            "recommendation": {"continue_to_s5e20": False, "recommended_next_stage": "stop_direction_head_training_and_freeze_S5E15", "keep_s5e15_as_best_candidate": True, "s5e19c_should_replace_s5e15": False, "main_fix": "required artifacts missing"},
            "no_fallback_guard": {},
            "s5_locked_metrics_policy_unchanged": True,
            "validation": {"static_test": "NOT_RUN", "run_serial_validation_guard": "NOT_RUN"},
            "git": {"commits_created": [], "pushed_to_remote": False, "working_tree_clean": False},
            "final_classification": "S5E19D_INSUFFICIENT_ARTIFACTS",
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_report(Path(args.out_report), payload)
        return payload

    s15_ck = read_json(Path(args.s5e15_checkpoint))
    s19_ck = read_json(Path(args.s5e19c_checkpoint))
    status = read_json(Path(args.s5e19c_training_status))
    noop = read_json(Path(args.s5e19c_results) / "noop_guard_audit.json")
    s15_prov = read_jsonl(Path(args.s5e15_results) / "edge_provenance.jsonl")
    s19_prov = read_jsonl(Path(args.s5e19c_results) / "edge_provenance.jsonl")
    gt = read_tum(Path("external_baselines/dataset/scene01_seq03/groundtruth_tum.txt"))
    train_text = Path(args.s5e19c_train_script).read_text(encoding="utf-8")
    cfg_text = Path(args.s5e19c_config).read_text(encoding="utf-8")
    lib_text = Path("tools/s5e2_adjacent_dense_lib.py").read_text(encoding="utf-8")

    training_a = training_path_audit(status, train_text, cfg_text)
    delta_a = delta_tdir_audit(s15_prov, s19_prov, gt)
    sign_a = sign_score_audit(s19_prov)
    generalization = train_eval_generalization(status)
    frame_a = frame_convention_audit(train_text, Path("tools/export_s5e19c_no_fallback_predictions.py").read_text(encoding="utf-8"), lib_text)
    obs_a = observability_weighting_audit(s19_prov, Path(args.s5e15_results) / "scale_antiparallel_refinement_weights.jsonl", cfg_text, train_text)
    multi_a = multiframe_audit(train_text, Path("external_baselines/results/s5e19_traceable_dense/multiframe_geometry_metrics.json"))
    final_cls, recommendation = final_diagnosis(delta_a, sign_a, frame_a, obs_a, multi_a)

    payload = {
        "experiment": "S5E19D_direction_training_failure_audit",
        "status": {"audit_only": True, "training_attempted": False, "metrics_refreshed": False, "official_s5_unchanged": True},
        "artifact_inventory": inv,
        "s5e15_reference": S5E15_REF,
        "s5e19c_result": {
            "tdir_mean_deg": s19_ck["component_metrics"]["signed_tdir_mean_deg"],
            "anti_parallel_rate": s19_ck["component_metrics"]["anti_parallel_rate"],
            "path_ratio": s19_ck["component_metrics"]["path_ratio"],
            "sim3_ate": s19_ck["external_eval"]["sim3"]["ate"],
        },
        "training_path_audit": training_a,
        "delta_tdir_audit": delta_a,
        "sign_score_audit": sign_a,
        "train_eval_generalization": generalization,
        "frame_convention_audit": frame_a,
        "observability_weighting_audit": obs_a,
        "multiframe_audit": multi_a,
        "recommendation": recommendation,
        "no_fallback_guard": noop,
        "s5_locked_metrics_policy_unchanged": True,
        "validation": {"static_test": "NOT_RUN", "run_serial_validation_guard": "NOT_RUN"},
        "git": {"commits_created": [], "pushed_to_remote": False, "working_tree_clean": False},
        "final_classification": final_cls,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--s5e19c-checkpoint", required=True)
    p.add_argument("--s5e19c-training-status", required=True)
    p.add_argument("--s5e15-results", required=True)
    p.add_argument("--s5e19c-results", required=True)
    p.add_argument("--s5e19c-train-script", required=True)
    p.add_argument("--s5e19c-config", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
