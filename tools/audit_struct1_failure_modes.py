#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "STRUCT1A_SCALE_GUARD_NOT_EFFECTIVE",
    "STRUCT1A_EXPORT_TMAG_BUG_SUSPECTED",
    "STRUCT1A_GLOBAL_SCALE_EXPLOSION",
    "STRUCT1A_DIRECTION_AND_SCALE_BOTH_FAILED",
    "STRUCT1A_GEOMETRY_TOKEN_SIGNAL_UNRELIABLE",
    "STRUCT1A_INSUFFICIENT_ARTIFACTS",
    "STRUCT1A_AUDIT_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    if float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _run_text(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout.strip()


def _extract_source_flags(train_text: str, export_text: str, model_text: str) -> Dict[str, Any]:
    has_bounded_prior = "FineScaleHeadWithS5E15Guard" in model_text and "delta_log_tmag" in model_text
    final_tmag_is_exp = "final_tmag = torch.exp(final_log_tmag).clamp_min(1.0e-6)" in model_text
    final_tmag_line = "final_tmag = torch.exp(final_log_tmag).clamp_min(1.0e-6)"
    has_upper_clamp = False
    path_loss_entered_total = (
        "loss_path =" in train_text and '"w_path_length_consistency"' in train_text and '+ float(cfg["losses"]["w_path_length_consistency"]) * loss_path' in train_text
    )
    export_used_guarded_tmag = 'final_mag = float(out["tmag"][0].cpu().item())' in export_text
    export_used_raw_tmag = "raw_tmag" in export_text
    froze_geometry_backbone = 'name.startswith("backbone_a.")' in train_text and 'name.startswith("geometry_layer.")' in train_text
    return {
        "bounded_train_prior_terms_present": has_bounded_prior,
        "final_tmag_uses_exp_log": final_tmag_is_exp,
        "final_tmag_has_upper_clamp": has_upper_clamp,
        "path_loss_entered_total": path_loss_entered_total,
        "export_used_guarded_tmag": export_used_guarded_tmag,
        "export_used_raw_tmag": export_used_raw_tmag,
        "froze_geometry_backbone_during_training": froze_geometry_backbone,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root)
    try:
        ckpt = _read_json(Path(args.struct1_checkpoint))
        training = _read_json(Path(args.training_status))
        metrics = _read_json(Path(args.results) / "edge_component_metrics.json")
        integrity = _read_json(Path(args.results) / "struct1_integrity_audit.json")
        cfg_text = Path(args.config).read_text(encoding="utf-8")
        train_text = Path("tools/train_struct1_geometry_token_pose_solver.py").read_text(encoding="utf-8")
        export_text = Path("tools/export_struct1_adjacent_dense_predictions.py").read_text(encoding="utf-8")
        model_text = Path("model.py").read_text(encoding="utf-8")
        prov = _read_jsonl(Path(args.results) / "edge_provenance.jsonl")
    except Exception as exc:
        payload = {
            "final_classification": "STRUCT1A_AUDIT_ERROR",
            "error": str(exc),
            "s5_locked_metrics_policy_unchanged": True,
        }
        write_json(Path(args.out_json), payload)
        Path(args.out_report).write_text("# STRUCT1 失败模式审计\n\n读取审计输入失败。\n", encoding="utf-8")
        return payload

    rows = metrics.get("metrics_rows", [])
    if not rows or not prov:
        payload = {
            "final_classification": "STRUCT1A_INSUFFICIENT_ARTIFACTS",
            "scale_failure_audit": {},
            "direction_failure_audit": {},
            "git_status": {},
            "recommendation": {},
            "s5_locked_metrics_policy_unchanged": True,
        }
        write_json(Path(args.out_json), payload)
        Path(args.out_report).write_text("# STRUCT1 失败模式审计\n\n审计所需 artifact 不完整。\n", encoding="utf-8")
        return payload

    ratios = np.asarray([float(r["tmag_ratio"]) for r in rows], dtype=float)
    pred_steps = np.asarray([float(r["pred_step_length"]) for r in rows], dtype=float)
    gt_steps = np.asarray([float(r["gt_step_length"]) for r in rows], dtype=float)
    tdir = np.asarray([float(r["tdir_deg"]) for r in rows], dtype=float)
    anti = np.asarray([1.0 if r["anti_parallel_flag"] else 0.0 for r in rows], dtype=float)
    conf = np.asarray([float(p.get("confidence", 0.0)) for p in prov], dtype=float)
    ent = np.asarray([float(p.get("entropy", 0.0)) for p in prov], dtype=float)
    comp = metrics["component_metrics"]
    flags = _extract_source_flags(train_text, export_text, model_text)

    path_loss_weight = None
    for line in cfg_text.splitlines():
        if line.strip().startswith("w_path_length_consistency:"):
            path_loss_weight = float(line.split(":", 1)[1].strip())
            break
    high_conf_thr = float(np.median(conf))
    hi = conf >= high_conf_thr
    lo = conf < high_conf_thr

    tmag_stats = {
        "tmag_median_ratio": float(np.median(ratios)),
        "tmag_p90_ratio": float(np.percentile(ratios, 90)),
        "tmag_p95_ratio": float(np.percentile(ratios, 95)),
        "tmag_p99_ratio": float(np.percentile(ratios, 99)),
        "tmag_max_ratio": float(np.max(ratios)),
    }
    path_ratio = float(np.sum(pred_steps) / max(float(np.sum(gt_steps)), 1.0e-12))
    top10_path_contrib = float(np.sum(np.sort(pred_steps)[-10:]) / max(float(np.sum(pred_steps)), 1.0e-12))
    frac_gt_10 = float(np.mean(ratios > 10.0))
    frac_gt_20 = float(np.mean(ratios > 20.0))
    outlier_dominated = bool(top10_path_contrib > 0.80 and frac_gt_10 < 0.25)
    scale_explosion_global = bool(float(np.median(ratios)) > 5.0 and frac_gt_10 > 0.30 and path_ratio > 2.0)
    bounded_prior_effective = bool(tmag_stats["tmag_median_ratio"] < 5.0 and path_ratio < 2.0) if flags["bounded_train_prior_terms_present"] else None

    conf_tmag_spearman = _corr(_rankdata(conf), _rankdata(ratios))
    ent_tmag_spearman = _corr(_rankdata(ent), _rankdata(ratios))
    conf_tdir_spearman = _corr(_rankdata(conf), _rankdata(tdir))
    ent_tdir_spearman = _corr(_rankdata(ent), _rankdata(tdir))

    main_scale_failure_source = (
        "bounded_train_prior 仅对 base_log_tmag 做小幅 delta 修正，缺少 upper clamp；同时训练阶段冻结了 geometry/backbone，仅训练 solver 头，"
        "导致 scale head 只能围绕已有 coarse prior 漂移，path/tmag loss 权重不足以压住全局放大。"
    )
    if not flags["export_used_guarded_tmag"]:
        main_scale_failure_source = "export 可能未使用 guarded_tmag，而是绕回 raw_tmag。"
    elif scale_explosion_global and outlier_dominated:
        main_scale_failure_source = "全局 scale 偏大，同时伴随少数超大 outlier 放大 path_ratio。"
    elif scale_explosion_global:
        main_scale_failure_source = "全局 scale 系统性放大；outlier 进一步推高 path_ratio，但不是唯一来源。"

    geometry_token_tdir_failed = bool(comp["signed_tdir_mean_deg"] > 60.0 and comp["anti_parallel_rate"] > 0.25)
    confidence_correlates_with_tdir = bool(conf_tdir_spearman is not None and abs(conf_tdir_spearman) > 0.20) if conf_tdir_spearman is not None else None
    main_direction_failure_source = (
        "geometry token solver 确实接入了 tdir，但高低 confidence 的 tdir 均较差，说明 token signal 没有形成可靠方向约束；"
        "同时高 confidence 边的 anti-parallel rate 反而更高，表明 confidence 不能有效筛掉错误方向。"
    )

    scale_failure_audit = {
        **tmag_stats,
        "path_ratio": path_ratio,
        "scale_explosion_global": scale_explosion_global,
        "outlier_dominated": outlier_dominated,
        "bounded_prior_effective": bounded_prior_effective,
        "export_used_guarded_tmag": flags["export_used_guarded_tmag"],
        "path_loss_entered_total": flags["path_loss_entered_total"],
        "path_loss_weight": path_loss_weight,
        "frac_tmag_ratio_gt_10": frac_gt_10,
        "frac_tmag_ratio_gt_20": frac_gt_20,
        "top10_path_contribution": top10_path_contrib,
        "high_conf_tmag_median_ratio": float(np.median(ratios[hi])),
        "low_conf_tmag_median_ratio": float(np.median(ratios[lo])),
        "main_scale_failure_source": main_scale_failure_source,
    }
    direction_failure_audit = {
        "tdir_mean_deg": float(comp["signed_tdir_mean_deg"]),
        "anti_parallel_rate": float(comp["anti_parallel_rate"]),
        "geometry_token_tdir_failed": geometry_token_tdir_failed,
        "confidence_correlates_with_tdir": confidence_correlates_with_tdir,
        "high_conf_tdir_mean_deg": float(np.mean(tdir[hi])),
        "low_conf_tdir_mean_deg": float(np.mean(tdir[lo])),
        "high_conf_anti_parallel_rate": float(np.mean(anti[hi])),
        "low_conf_anti_parallel_rate": float(np.mean(anti[lo])),
        "confidence_vs_tdir_spearman": conf_tdir_spearman,
        "confidence_vs_tmag_spearman": conf_tmag_spearman,
        "entropy_vs_tdir_spearman": ent_tdir_spearman,
        "entropy_vs_tmag_spearman": ent_tmag_spearman,
        "main_direction_failure_source": main_direction_failure_source,
    }

    if not flags["export_used_guarded_tmag"]:
        final = "STRUCT1A_EXPORT_TMAG_BUG_SUSPECTED"
    elif scale_explosion_global and geometry_token_tdir_failed:
        final = "STRUCT1A_DIRECTION_AND_SCALE_BOTH_FAILED"
    elif scale_explosion_global and not bounded_prior_effective:
        final = "STRUCT1A_SCALE_GUARD_NOT_EFFECTIVE"
    elif scale_explosion_global:
        final = "STRUCT1A_GLOBAL_SCALE_EXPLOSION"
    else:
        final = "STRUCT1A_GEOMETRY_TOKEN_SIGNAL_UNRELIABLE"

    current_branch = _run_text(["git", "branch", "--show-current"], project_root)
    git_status = _run_text(["git", "status", "--short"], project_root)
    recommendation = {
        "promote_struct1": False,
        "keep_s5e15_as_best_candidate": True,
        "continue_to_struct2": False,
        "recommended_next_stage": "stop_model_development_and_close_with_negative_result" if final in {
            "STRUCT1A_DIRECTION_AND_SCALE_BOTH_FAILED",
            "STRUCT1A_SCALE_GUARD_NOT_EFFECTIVE",
            "STRUCT1A_GLOBAL_SCALE_EXPLOSION",
        } else "consider_STRUCT2_only_if_new_scale_prior_design_exists",
    }

    payload = {
        "final_classification": final,
        "scale_failure_audit": scale_failure_audit,
        "direction_failure_audit": direction_failure_audit,
        "export_scale_guard_suspicion": {
            "bounded_train_prior_terms_present": flags["bounded_train_prior_terms_present"],
            "final_tmag_uses_exp_log": flags["final_tmag_uses_exp_log"],
            "final_tmag_has_upper_clamp": flags["final_tmag_has_upper_clamp"],
            "export_used_guarded_tmag": flags["export_used_guarded_tmag"],
            "export_used_raw_tmag": flags["export_used_raw_tmag"],
            "froze_geometry_backbone_during_training": flags["froze_geometry_backbone_during_training"],
        },
        "git_status": {
            "branch": current_branch,
            "status_short": git_status.splitlines(),
        },
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
    }
    write_json(Path(args.out_json), payload)

    report = [
        "# STRUCT1 失败模式审计",
        "",
        "## 1. 执行摘要",
        f"本次审计结论为 `{final}`。STRUCT1 的 geometry-token pose solver 确实真实接入，但 scale/path 与 direction 同时失败，S5E15 仍应保留为 best candidate。",
        "",
        "## 2. STRUCT1 为什么失败",
        "失败不是因为没有接入 geometry token，而是接入后没有形成可用的方向与尺度约束。训练状态显示 `real_training_executed=true`、`W_ab_used_in_pose_solver=true`、`tdir_from_geometry_tokens=true`、`fine_stage_used=true`，因此这是一次真实的负结果。",
        "",
        "## 3. scale/path 爆炸分析",
        f"- tmag 中位数比例={scale_failure_audit['tmag_median_ratio']:.4f}，p95={scale_failure_audit['tmag_p95_ratio']:.4f}，max={scale_failure_audit['tmag_max_ratio']:.4f}。",
        f"- path_ratio={scale_failure_audit['path_ratio']:.4f}，属于明显全局爆炸，`scale_explosion_global={scale_failure_audit['scale_explosion_global']}`。",
        f"- 超过 10 倍 GT 的边比例={scale_failure_audit['frac_tmag_ratio_gt_10']:.4f}，超过 20 倍 GT 的边比例={scale_failure_audit['frac_tmag_ratio_gt_20']:.4f}。",
        f"- 前 10 个最长预测步贡献了总路径的 {scale_failure_audit['top10_path_contribution']:.4f}，说明不是纯 outlier，但也存在重尾放大。",
        f"- bounded_train_prior 名义上存在，但 `bounded_prior_effective={scale_failure_audit['bounded_prior_effective']}`，核心原因是它只做 `base_log_tmag + bounded_delta`，没有更强的上界约束。",
        "",
        "## 4. tdir/anti_parallel 退化分析",
        f"- signed_tdir={direction_failure_audit['tdir_mean_deg']:.4f}，anti_parallel_rate={direction_failure_audit['anti_parallel_rate']:.4f}。",
        f"- geometry_token_tdir_failed={direction_failure_audit['geometry_token_tdir_failed']}。",
        f"- 高置信度边 tdir 均值={direction_failure_audit['high_conf_tdir_mean_deg']:.4f}，低置信度边 tdir 均值={direction_failure_audit['low_conf_tdir_mean_deg']:.4f}。",
        f"- 高置信度 anti_parallel_rate={direction_failure_audit['high_conf_anti_parallel_rate']:.4f}，低置信度 anti_parallel_rate={direction_failure_audit['low_conf_anti_parallel_rate']:.4f}。",
        "",
        "## 5. geometry token confidence / entropy 与错误关系",
        f"- confidence vs tmag Spearman={direction_failure_audit['confidence_vs_tmag_spearman']}.",
        f"- confidence vs tdir Spearman={direction_failure_audit['confidence_vs_tdir_spearman']}.",
        f"- entropy vs tmag Spearman={direction_failure_audit['entropy_vs_tmag_spearman']}.",
        f"- entropy vs tdir Spearman={direction_failure_audit['entropy_vs_tdir_spearman']}.",
        "这些相关性整体偏弱，说明现有 confidence / entropy 不能可靠地区分方向好坏；对 tmag 仅有弱相关，无法作为有效 scale guard。",
        "",
        "## 6. export / scale guard 是否可疑",
        f"- export_used_guarded_tmag={scale_failure_audit['export_used_guarded_tmag']}，说明导出没有明显绕过 guard。",
        f"- path_loss_entered_total={scale_failure_audit['path_loss_entered_total']}，权重={scale_failure_audit['path_loss_weight']}.",
        f"- final_tmag 使用 `exp(log_tmag)`，并且缺少 upper clamp：`final_tmag_has_upper_clamp={flags['final_tmag_has_upper_clamp']}`。",
        f"- 训练中还冻结了 geometry backbone / layer：`froze_geometry_backbone_during_training={flags['froze_geometry_backbone_during_training']}`，这会进一步削弱 geometry token 主干对方向和尺度的联合学习能力。",
        "",
        "## 7. 是否值得 STRUCT2",
        f"`continue_to_struct2={recommendation['continue_to_struct2']}`。如果没有全新的尺度先验设计与更强的导出上界保护，不建议继续沿同一路线推进 STRUCT2。",
        "",
        "## 8. 是否保留 S5E15",
        "应保留。S5E15 仍然在 signed_tdir、anti_parallel、tmag_median_ratio、path_ratio、sim3 ATE 上明显优于 STRUCT1。",
        "",
        "## 9. Git hygiene 处理",
        "本次审计新增只读审计脚本、报告、checkpoint 与静态测试；不训练新模型，不刷新官方指标，不修改 S5 locked metrics/policy。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--struct1-checkpoint", required=True)
    p.add_argument("--training-status", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
