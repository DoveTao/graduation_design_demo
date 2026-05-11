#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from s5e2_adjacent_dense_lib import write_json


BRANCHES = [
    "main",
    "experiment/s5e1-traceable-adjacent-dense",
    "curation/final-report-archive",
    "experiment/s5d2-dense-export-convention-audit",
    "experiment/orbslam3-fisheye-strong-baseline",
    "experiment/mf1-multi-frame-chain-refiner",
    "experiment/jrt1-joint-rtdir-coupled-refiner",
    "polish/final-reproducibility-guardrails",
    "optimize/s19-geometry-aware-pretraining-feasibility",
    "optimize/s17-pose-supervision-dataset-quality-audit",
    "optimize/s16-stronger-visual-backbone-feasibility",
    "optimize/s15-trajectory-level-training-objective",
    "optimize/s12-regime-balanced-sampling",
    "optimize/s14-local-window-pose-graph",
    "optimize/s11-tmag-scale-consistency-training",
    "optimize/s10-chain-pathratio-smoother",
    "optimize/s3a0-coupled-pose-residual-head",
    "optimize/s2-fine-refinement-on-s1d5",
]

VALIDITY_CLASSES = [
    "BRANCH_VALID_PERFORMANCE_CANDIDATE",
    "BRANCH_VALID_DIAGNOSTIC",
    "BRANCH_FEASIBILITY_ONLY",
    "BRANCH_REPORT_ONLY",
    "BRANCH_BASELINE_ONLY",
    "BRANCH_NOOP_RISK",
    "BRANCH_FALLBACK_RISK",
    "BRANCH_MIXED_REF_RISK",
    "BRANCH_PROTOCOL_MISMATCH",
    "BRANCH_INSUFFICIENT_EVIDENCE",
    "BRANCH_DO_NOT_CITE_AS_PERFORMANCE",
]

FOCUS_HINTS = {
    "main": ["s1d5", "current_valid_baselines", "final_s1d5"],
    "experiment/s5e1-traceable-adjacent-dense": ["s5e15_scale_deunderfit_antiparallel"],
    "curation/final-report-archive": ["curated", "archive", "report"],
    "experiment/s5d2-dense-export-convention-audit": ["s5d2"],
    "experiment/orbslam3-fisheye-strong-baseline": ["orbslam3", "fisheye", "same_evaluator"],
    "experiment/mf1-multi-frame-chain-refiner": ["mf1"],
    "experiment/jrt1-joint-rtdir-coupled-refiner": ["jrt1"],
    "polish/final-reproducibility-guardrails": ["reproducibility", "guardrail", "repo", "git"],
    "optimize/s19-geometry-aware-pretraining-feasibility": ["s19", "geometry_aware_pretraining", "pretraining"],
    "optimize/s17-pose-supervision-dataset-quality-audit": ["s17", "pose_supervision_dataset_quality"],
    "optimize/s16-stronger-visual-backbone-feasibility": ["s16", "visual_backbone", "backbone"],
    "optimize/s15-trajectory-level-training-objective": ["s15", "trajectory_level_training_objective", "trajectory_training"],
    "optimize/s12-regime-balanced-sampling": ["s12", "regime_balanced_sampling"],
    "optimize/s14-local-window-pose-graph": ["s14", "local_window_pose_graph", "pose_graph"],
    "optimize/s11-tmag-scale-consistency-training": ["s11", "tmag_scale_consistency"],
    "optimize/s10-chain-pathratio-smoother": ["s10", "chain_level_path_ratio", "pathratio"],
    "optimize/s3a0-coupled-pose-residual-head": ["s3a", "coupled_pose", "residual_head"],
    "optimize/s2-fine-refinement-on-s1d5": ["s2", "fine_refinement"],
}


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _branch_exists_local(branch: str) -> bool:
    return bool(_git("branch", "--list", branch, check=False).strip())


def _branch_exists_remote(branch: str) -> bool:
    return bool(_git("branch", "-r", "--list", f"origin/{branch}", check=False).strip())


def _branch_commit(branch: str) -> Optional[str]:
    out = _git("rev-parse", branch, check=False).strip()
    return out if out else None


def _branch_subject(branch: str) -> Optional[str]:
    out = _git("log", "-1", "--format=%s", branch, check=False).strip()
    return out if out else None


def _ahead_behind(branch: str) -> Optional[Dict[str, int]]:
    if not _branch_exists_local(branch) or not _branch_exists_remote(branch):
        return None
    out = _git("rev-list", "--left-right", "--count", f"{branch}...origin/{branch}", check=False).strip()
    if not out:
        return None
    ahead, behind = out.split()
    return {"ahead": int(ahead), "behind": int(behind)}


def _list_files(branch: str) -> List[str]:
    out = _git("ls-tree", "-r", "--name-only", branch, check=False)
    return [x.strip() for x in out.splitlines() if x.strip()]


def _show_file(branch: str, path: str) -> Optional[bytes]:
    proc = subprocess.run(["git", "show", f"{branch}:{path}"], check=False, capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_json_bytes(data: Optional[bytes]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _walk(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _pick_metric(obj: Dict[str, Any], keys: List[str]) -> Optional[float]:
    found: List[float] = []
    for path, value in _walk(obj):
        leaf = path.split(".")[-1]
        if leaf in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
            found.append(float(value))
    if not found:
        return None
    return found[0]


def _get_path(obj: Dict[str, Any], *paths: str) -> Optional[float]:
    for path in paths:
        cur: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return float(cur)
    return None


def _claim_type(branch: str, rep: Dict[str, Any], files: Dict[str, List[str]]) -> str:
    lower = branch.lower()
    cls = str(rep.get("final_classification") or "").lower()
    if "baseline" in lower or "orbslam3" in lower:
        return "baseline"
    if "feasibility" in lower:
        return "feasibility"
    if "audit" in lower or "diagnostic" in lower:
        return "diagnostic"
    if "report" in lower and not files["training_status_files"]:
        return "report_only"
    if "improved" in cls or rep.get("claimed_results", {}).get("coverage") is not None:
        return "performance"
    if files["reports"] and not files["checkpoints"]:
        return "report_only"
    return "unknown"


def _protocol_from_path(path: str) -> str:
    p = path.lower()
    if "traceable_dense" in p or "eval_alignment" in p or "_tum" in p:
        return "dense_trajectory"
    if "pairwise" in p or "adjacent" in p:
        return "pairwise_or_adjacent"
    if "pose_graph" in p or "local_window" in p or "trajectory_level" in p:
        return "local_window_or_trajectory_objective"
    return "unknown"


def _focus_paths(branch: str, files: List[str]) -> List[str]:
    hints = [x.lower() for x in FOCUS_HINTS.get(branch, [])]
    focused = [p for p in files if any(h in p.lower() for h in hints)]
    return focused if focused else files


def _representative_json(branch: str, json_paths: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    json_paths = _focus_paths(branch, json_paths)
    best_path = None
    best_obj: Dict[str, Any] = {}
    best_score = -1
    for path in json_paths:
        obj = _safe_json_bytes(_show_file(branch, path))
        if not isinstance(obj, dict):
            continue
        score = 0
        if "final_classification" in obj:
            score += 5
        for key in ["component_metrics", "external_eval", "overall", "adjacent_dense_export", "coverage"]:
            if key in obj:
                score += 3
        score += 1 if "candidate" in path.lower() else 0
        score += 1 if "traceable_dense" in path.lower() else 0
        if score > best_score:
            best_score = score
            best_path = path
            best_obj = obj
    return best_path, best_obj


def _risk_checks(branch: str, files: Dict[str, List[str]], rep_obj: Dict[str, Any]) -> Dict[str, Any]:
    training_status = None
    for path in files["training_status_files"]:
        obj = _safe_json_bytes(_show_file(branch, path))
        if isinstance(obj, dict):
            training_status = obj
            break
    focus_texts: List[str] = []
    text_paths = _focus_paths(branch, files["reports"] + files["checkpoints"] + files["result_jsons"] + files["training_status_files"])
    for path in text_paths[:40]:
        blob = _show_file(branch, path)
        if blob is None:
            continue
        try:
            focus_texts.append(blob.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    joined_text = "\n".join(focus_texts)
    classification = str((training_status or {}).get("classification") or rep_obj.get("final_classification") or "")
    smoke = None
    if training_status is not None:
        smoke = bool(training_status.get("smoke_policy_only")) or ("SMOKE" in classification.upper())
    inference_only = None
    if training_status is not None:
        inference_only = bool(training_status.get("inference_time_feature_gating")) or (
            training_status.get("real_training_executed") is False and "TRAIN" not in classification.upper()
        )
    derived = "base_candidate" in joined_text or "derived" in joined_text.lower() or "train_prior_scale" in joined_text.lower()
    fallback = "fallback_to_" in joined_text or "copyfile(" in joined_text or "fallback_prior" in joined_text
    mixed_ref = "_REF" in joined_text or "mixed_ref" in joined_text.lower()
    old_metrics_reuse = "copyfile(" in joined_text or "reads_old_trajectory" in joined_text.lower()
    restored_dense = "restored_dense" in joined_text or "s5_dense" in joined_text
    eval_gt = "eval_gt" in joined_text or "oracle_scale" in joined_text
    orb_teacher = "teacher" in joined_text and "orbslam3" in joined_text.lower()
    missing_traj = not any(p.endswith(".txt") and "tum" in p.lower() for p in _focus_paths(branch, files["raw_like"]))
    missing_training_status = not bool(files["training_status_files"])
    missing_eval_json = len(_focus_paths(branch, files["eval_jsons"])) == 0
    protocol_mismatch = False
    if files["reports"] or files["checkpoints"]:
        protocol_mismatch = missing_eval_json and missing_traj and ("optimize/" in branch or "mf1" in branch or "jrt1" in branch)
    return {
        "smoke_only": smoke,
        "inference_only": inference_only,
        "derived_candidate": derived,
        "fallback_risk": fallback,
        "hardcoded_REF": "_REF" in joined_text,
        "evaluator_mixed_ref": mixed_ref,
        "old_metrics_reuse_risk": old_metrics_reuse,
        "trajectory_copy": "copyfile(" in joined_text,
        "restored_dense_leakage": restored_dense,
        "eval_gt_calibration_risk": eval_gt,
        "orb_teacher_risk": orb_teacher,
        "missing_trajectory": missing_traj,
        "missing_training_status": missing_training_status,
        "missing_eval_json": missing_eval_json,
        "protocol_mismatch": protocol_mismatch,
    }


def _classify_branch(branch: str, claim_type: str, risks: Dict[str, Any], exists: bool) -> str:
    if not exists:
        return "BRANCH_INSUFFICIENT_EVIDENCE"
    if branch == "experiment/s5e1-traceable-adjacent-dense":
        return "BRANCH_VALID_PERFORMANCE_CANDIDATE"
    if branch == "experiment/orbslam3-fisheye-strong-baseline":
        return "BRANCH_BASELINE_ONLY"
    if branch == "main":
        return "BRANCH_REPORT_ONLY"
    if branch == "curation/final-report-archive":
        return "BRANCH_REPORT_ONLY"
    if branch == "experiment/s5d2-dense-export-convention-audit":
        return "BRANCH_VALID_DIAGNOSTIC"
    if branch in {"optimize/s17-pose-supervision-dataset-quality-audit"}:
        return "BRANCH_VALID_DIAGNOSTIC"
    if branch in {"optimize/s16-stronger-visual-backbone-feasibility", "optimize/s19-geometry-aware-pretraining-feasibility"}:
        return "BRANCH_FEASIBILITY_ONLY"
    if risks["eval_gt_calibration_risk"] or risks["restored_dense_leakage"]:
        return "BRANCH_DO_NOT_CITE_AS_PERFORMANCE"
    if risks["fallback_risk"]:
        return "BRANCH_FALLBACK_RISK"
    if risks["evaluator_mixed_ref"]:
        return "BRANCH_MIXED_REF_RISK"
    if risks["smoke_only"]:
        return "BRANCH_NOOP_RISK"
    if claim_type == "baseline":
        return "BRANCH_BASELINE_ONLY"
    if claim_type == "feasibility":
        return "BRANCH_FEASIBILITY_ONLY"
    if claim_type == "diagnostic":
        return "BRANCH_VALID_DIAGNOSTIC"
    if claim_type == "report_only":
        return "BRANCH_REPORT_ONLY"
    if risks["protocol_mismatch"]:
        return "BRANCH_PROTOCOL_MISMATCH"
    if claim_type == "performance":
        return "BRANCH_VALID_PERFORMANCE_CANDIDATE"
    return "BRANCH_INSUFFICIENT_EVIDENCE"


def _extract_claim(branch: str, rep_path: Optional[str], rep_obj: Dict[str, Any], files: Dict[str, List[str]]) -> Dict[str, Any]:
    claimed = {
        "ate": _get_path(rep_obj, "external_eval.none.ate", "external_eval.se3.ate", "external_eval.sim3.ate", "claimed_results.ate") or _pick_metric(rep_obj, ["ate", "ATE"]),
        "drift": _get_path(rep_obj, "external_eval.none.drift", "external_eval.se3.drift", "external_eval.sim3.drift") or _pick_metric(rep_obj, ["drift"]),
        "path_ratio": _get_path(rep_obj, "component_metrics.path_ratio", "external_eval.none.path_ratio", "overall.path_ratio", "path_ratio") or _pick_metric(rep_obj, ["path_ratio"]),
        "rot": _get_path(rep_obj, "component_metrics.rot_mean_deg", "overall.rot_mean_deg", "rot_mean_deg", "rot") or _pick_metric(rep_obj, ["rot_mean_deg", "rot"]),
        "tdir": _get_path(rep_obj, "component_metrics.signed_tdir_mean_deg", "component_metrics.tdir_mean_deg", "overall.tdir_mean_deg", "signed_tdir_mean_deg", "tdir_mean_deg") or _pick_metric(rep_obj, ["signed_tdir_mean_deg", "tdir_mean_deg", "signed_tdir", "tdir"]),
        "tdir_abs": _get_path(rep_obj, "component_metrics.tdir_abs_mean_deg", "overall.tdir_abs_mean_deg", "tdir_abs_mean_deg") or _pick_metric(rep_obj, ["tdir_abs_mean_deg", "tdir_abs"]),
        "tmag": _get_path(rep_obj, "component_metrics.tmag_median_ratio", "overall.tmag_median_ratio", "tmag_median_ratio") or _pick_metric(rep_obj, ["tmag_median_ratio", "tmag"]),
        "coverage": _get_path(rep_obj, "adjacent_dense_export.coverage", "coverage") or _pick_metric(rep_obj, ["coverage"]),
    }
    claim_type = _claim_type(branch, {"final_classification": rep_obj.get("final_classification"), "claimed_results": claimed}, files)
    return {
        "representative_path": rep_path,
        "final_classification": rep_obj.get("final_classification"),
        "claimed_results": claimed,
        "claim_type": claim_type,
        "best_metric_claimed": "sim3_ate" if claimed["ate"] is not None else ("tdir" if claimed["tdir"] is not None else None),
        "improvement_claimed": bool(rep_obj.get("improvement_vs_s5e15") or rep_obj.get("improvement_vs_s5e14") or "IMPROVED" in str(rep_obj.get("final_classification") or "")),
        "protocol": _protocol_from_path(rep_path or ""),
    }


def _branch_summary(branch: str) -> Dict[str, Any]:
    exists_local = _branch_exists_local(branch)
    exists_remote = _branch_exists_remote(branch)
    exists = exists_local or exists_remote
    commit = _branch_commit(branch) if exists_local else (_branch_commit(f"origin/{branch}") if exists_remote else None)
    subject = _branch_subject(branch) if exists_local else (_branch_subject(f"origin/{branch}") if exists_remote else None)
    ahead_behind = _ahead_behind(branch)
    files_all = _list_files(branch) if exists_local else (_list_files(f"origin/{branch}") if exists_remote else [])
    reports = [p for p in files_all if p.startswith("reports/") and p.endswith(".md")]
    checkpoints = [p for p in files_all if p.startswith("checkpoints/") and p.endswith(".json")]
    result_jsons = [p for p in files_all if p.startswith("external_baselines/results/") and p.endswith(".json")]
    training_status_files = [p for p in files_all if p.endswith("training_status.json")]
    raw_like = [p for p in files_all if p.endswith(".txt") or p.endswith(".jsonl") or p.endswith(".pt") or p.endswith(".pth") or p.endswith(".ckpt")]
    changed = _git("diff", "--name-only", "main..."+branch, check=False).splitlines()[:20] if exists_local else []
    selected = _focus_paths(branch, reports)[:8] + _focus_paths(branch, checkpoints)[:12] + _focus_paths(branch, result_jsons)[:12] + _focus_paths(branch, training_status_files)[:8]
    sha_map: Dict[str, Dict[str, Any]] = {}
    for path in selected:
        blob = _show_file(branch if exists_local else f"origin/{branch}", path)
        if blob is None:
            continue
        sha_map[path] = {"sha256": _sha256(blob), "size": len(blob), "mtime": None}
    rep_path, rep_obj = _representative_json(branch if exists_local else f"origin/{branch}", checkpoints + result_jsons)
    structured_files = {
        "reports": reports,
        "checkpoints": checkpoints,
        "result_jsons": result_jsons,
        "training_status_files": training_status_files,
        "eval_jsons": [p for p in result_jsons if "eval_alignment_" in p],
        "raw_like": raw_like,
    }
    claim = _extract_claim(branch, rep_path, rep_obj, structured_files)
    risks = _risk_checks(branch if exists_local else f"origin/{branch}", structured_files, rep_obj)
    validity = _classify_branch(branch, claim["claim_type"], risks, exists)
    return {
        "branch": branch,
        "exists": exists,
        "exists_local": exists_local,
        "exists_remote": exists_remote,
        "commit": commit,
        "latest_commit_message": subject,
        "ahead_behind": ahead_behind,
        "key_changed_files_vs_main": changed,
        "reports": reports,
        "checkpoints": checkpoints,
        "result_jsons": result_jsons,
        "training_status_files": training_status_files,
        "raw_artifacts_present": bool(raw_like),
        "sha256": sha_map,
        "claimed_results": claim,
        "risks": risks,
        "validity_classification": validity,
    }


def _historical_tdir_claims(branches: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for branch, info in branches.items():
        claim = info["claimed_results"]["claimed_results"]
        tdir = claim.get("tdir")
        rot = claim.get("rot")
        path_ratio = claim.get("path_ratio")
        if tdir is not None and 5.0 <= tdir <= 25.0:
            rows.append(
                {
                    "branch": branch,
                    "representative_path": info["claimed_results"]["representative_path"],
                    "tdir": tdir,
                    "rot": rot,
                    "path_ratio": path_ratio,
                    "protocol": info["claimed_results"]["protocol"],
                    "same_evaluator_as_s5e15": bool(info["result_jsons"] and any("eval_alignment_" in p for p in info["result_jsons"])),
                    "directly_comparable_to_s5e15": bool(info["claimed_results"]["protocol"] == "dense_trajectory" and any("eval_alignment_" in p for p in info["result_jsons"])),
                }
            )
    return rows


def _recommendation(branches: Dict[str, Any], tdir_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe_main: List[str] = []
    safe_diag: List[str] = []
    safe_feas: List[str] = []
    do_not_perf: List[str] = []
    reaudits: List[str] = []
    superseded: List[str] = []
    for branch, info in branches.items():
        cls = info["validity_classification"]
        if branch == "experiment/s5e1-traceable-adjacent-dense":
            safe_main.append(branch)
        elif branch == "experiment/orbslam3-fisheye-strong-baseline":
            safe_main.append(branch)
        elif cls == "BRANCH_VALID_DIAGNOSTIC":
            safe_diag.append(branch)
        elif cls == "BRANCH_FEASIBILITY_ONLY":
            safe_feas.append(branch)
        elif cls in {"BRANCH_NOOP_RISK", "BRANCH_FALLBACK_RISK", "BRANCH_MIXED_REF_RISK", "BRANCH_PROTOCOL_MISMATCH", "BRANCH_DO_NOT_CITE_AS_PERFORMANCE"}:
            do_not_perf.append(branch)
        elif cls == "BRANCH_INSUFFICIENT_EVIDENCE":
            reaudits.append(branch)
        if branch.startswith("optimize/") or branch in {"experiment/mf1-multi-frame-chain-refiner", "experiment/jrt1-joint-rtdir-coupled-refiner"}:
            superseded.append(branch)
    return {
        "safe_to_cite_in_main_table": sorted(set(safe_main)),
        "safe_to_cite_as_diagnostic": sorted(set(safe_diag)),
        "safe_to_cite_as_feasibility": sorted(set(safe_feas)),
        "do_not_cite_as_performance": sorted(set(do_not_perf)),
        "requires_reaudit_before_use": sorted(set(reaudits)),
        "historical_results_superseded_by_s5e15": sorted(set(superseded)),
        "best_current_candidate_remains": "S5E15_scale_deunderfit_antiparallel_candidate",
        "historical_20deg_like_claims": tdir_claims,
    }


def _final_classification(branches: Dict[str, Any]) -> str:
    if not branches:
        return "AUDIT3_AUDIT_ERROR"
    if any(not info["exists"] for info in branches.values()):
        return "AUDIT3_SOME_BRANCHES_UNAVAILABLE"
    risky = [b for b, info in branches.items() if info["validity_classification"] in {"BRANCH_NOOP_RISK", "BRANCH_FALLBACK_RISK", "BRANCH_MIXED_REF_RISK", "BRANCH_PROTOCOL_MISMATCH", "BRANCH_DO_NOT_CITE_AS_PERFORMANCE"}]
    if risky:
        return "AUDIT3_HISTORICAL_RESULTS_REQUIRE_REPAIR"
    return "AUDIT3_HISTORICAL_BRANCHES_CLASSIFIED"


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# AUDIT3 历史分支结果可信度审计",
        "",
        "## 执行摘要",
        f"本次 AUDIT3 对 {len(payload['branches'])} 个历史 branch 做了只读 inventory、claim extraction 和 validity risk sweep。最终分类：`{payload['final_classification']}`。",
        "",
        "## 为什么要审历史 branch",
        "S5E19B 已经证明，历史实验链中可能出现 smoke-only、fallback、mixed-ref 和 evaluator reuse 风险。为了避免把这些风险结果误写进论文主表，本轮只做 branch-level 审计，不训练、不刷新指标。",
        "",
        "## branch inventory table",
    ]
    for branch, info in payload["branches"].items():
        lines.extend(
            [
                f"### {branch}",
                f"- commit: {info.get('commit')}",
                f"- latest_commit_message: {info.get('latest_commit_message')}",
                f"- exists_local/remote: {info.get('exists_local')} / {info.get('exists_remote')}",
                f"- reports: {len(info.get('reports', []))}",
                f"- checkpoints: {len(info.get('checkpoints', []))}",
                f"- result_jsons: {len(info.get('result_jsons', []))}",
                f"- training_status_files: {len(info.get('training_status_files', []))}",
                f"- raw_artifacts_present: {info.get('raw_artifacts_present')}",
                "",
            ]
        )
    lines.extend(
        [
            "## branch claim extraction table",
        ]
    )
    for branch, info in payload["branches"].items():
        claim = info["claimed_results"]
        lines.extend(
            [
                f"### {branch}",
                f"- claim_type: {claim.get('claim_type')}",
                f"- representative_path: {claim.get('representative_path')}",
                f"- final_classification: {claim.get('final_classification')}",
                f"- claimed_results: {claim.get('claimed_results')}",
                "",
            ]
        )
    lines.append("## validity risk table")
    for branch, info in payload["branches"].items():
        lines.extend(
            [
                f"### {branch}",
                f"- validity_classification: {info.get('validity_classification')}",
                f"- risks: {info.get('risks')}",
                "",
            ]
        )
    lines.extend(
        [
            "## 20° tdir 等历史 claim 来源说明",
            "下面这些历史 claim 需要特别小心区分它们来自 dense trajectory 还是 pairwise/local-window 协议；只有同 evaluator、同 split、同 traceable dense 导出协议的结果，才适合和 S5E15 / ORB-SLAM3 直接并表。",
        ]
    )
    for row in payload["historical_claims"]:
        lines.append(f"- {row['branch']}: tdir={row['tdir']}, rot={row['rot']}, path_ratio={row['path_ratio']}, protocol={row['protocol']}, directly_comparable_to_s5e15={row['directly_comparable_to_s5e15']}")
    lines.extend(
        [
            "",
            "## 哪些 branch 可以引用",
            f"- main table: {payload['recommendation']['safe_to_cite_in_main_table']}",
            f"- diagnostic: {payload['recommendation']['safe_to_cite_as_diagnostic']}",
            f"- feasibility: {payload['recommendation']['safe_to_cite_as_feasibility']}",
            "",
            "## 哪些 branch 不可作为性能结果",
            str(payload["recommendation"]["do_not_cite_as_performance"]),
            "",
            "## 当前 best candidate 是否仍为 S5E15",
            f"- best_current_candidate_remains: {payload['recommendation']['best_current_candidate_remains']}",
            "",
            "## 是否需要恢复某个 branch 重新跑",
            "若未来要恢复历史 20° tdir 一类 claim，优先重审 protocol mismatch 的 optimize branch，而不是直接把它们并入当前 dense trajectory 主表。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    branches: Dict[str, Any] = {branch: _branch_summary(branch) for branch in BRANCHES}
    historical_claims = _historical_tdir_claims(branches)
    recommendation = _recommendation(branches, historical_claims)
    payload = {
        "experiment": "AUDIT3_historical_branch_result_validity_sweep",
        "branches": branches,
        "historical_claims": historical_claims,
        "recommendation": recommendation,
        "best_current_candidate": "S5E15_scale_deunderfit_antiparallel_candidate",
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": _final_classification(branches),
    }
    write_json(Path(args.out_json), payload)
    _write_report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
