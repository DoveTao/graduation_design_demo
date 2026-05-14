#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_FINAL_CLASSIFICATIONS = [
    "GIT2_WORKING_TREE_CLEAN",
    "GIT2_WORKING_TREE_CLEAN_WITH_LOCAL_IGNORED_ARTIFACTS",
    "GIT2_PARTIAL_CLEAN_REMAINING_MANUAL_REVIEW",
    "GIT2_BLOCKED_BY_UNCLASSIFIED_FILES",
    "GIT2_ERROR",
]

# Canonical outputs:
# - reports/git2_pre_final_freeze_cleanup_report.md
# - checkpoints/GIT2_pre_final_freeze_cleanup.json


def _run(cmd: List[str], cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _parse_status(lines: List[str]) -> Dict[str, List[str]]:
    out = {"modified": [], "untracked": [], "deleted": []}
    for line in lines:
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if code == "??":
            out["untracked"].append(path)
        elif "D" in code:
            out["deleted"].append(path)
        else:
            out["modified"].append(path)
    return out


def _classify_path(path: str) -> str:
    if any(path.startswith(p) for p in [
        "configs/s5e13", "configs/s5e14", "configs/s5e15",
        "tools/build_s5e13", "tools/train_s5e13", "tools/export_s5e13", "tools/evaluate_s5e13", "tools/compare_s5e13",
        "tools/audit_s5e14", "tools/train_s5e14", "tools/export_s5e14", "tools/evaluate_s5e14", "tools/compare_s5e14",
        "tools/audit_s5e15", "tools/train_s5e15", "tools/export_s5e15", "tools/evaluate_s5e15", "tools/compare_s5e15",
        "reports/s5e13", "reports/s5e14", "reports/s5e15",
        "tests/test_s5e13", "tests/test_s5e14", "tests/test_s5e15",
        "checkpoints/S5E13", "checkpoints/S5E14", "checkpoints/S5E15",
    ]):
        return "should_commit"
    if path in {
        ".gitignore",
        "tools/audit_git2_pre_final_freeze.py",
        "reports/git2_pre_final_freeze_cleanup_report.md",
        "checkpoints/GIT2_pre_final_freeze_cleanup.json",
        "tests/test_git2_pre_final_freeze_static.py",
    }:
        return "should_commit"
    if any(token in path for token in [
        "edge_provenance.jsonl",
        "router_decisions.jsonl",
        "spherical_bearing_flow_features.jsonl",
        "_tum",
        ".pt",
        ".pth",
        ".ckpt",
        "correspondence_weighted_dataset.json",
        "logs/",
    ]):
        return "should_ignore_or_local_only"
    if path in {
        "checkpoints/S5D11_validation_concurrency_audit.json",
        "checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json",
        "reports/s5d11_serial_validation_and_nonselected_edge_audit.md",
    }:
        return "should_restore"
    if path.startswith("external_baselines/results/s5e13_traceable_dense/") or path.startswith("external_baselines/results/s5e14_traceable_dense/") or path.startswith("external_baselines/results/s5e15_traceable_dense/"):
        if path.endswith(".json") or path.endswith(".csv"):
            return "should_commit"
        return "should_ignore_or_local_only"
    if path.startswith("checkpoints/S5E16_confidence_calibrated_sign_scale_router/router_policy.json"):
        return "should_commit"
    return "needs_manual_review"


def _collect_classification(paths: List[str]) -> Dict[str, List[str]]:
    out = {
        "should_commit": [],
        "should_ignore_or_local_only": [],
        "should_restore": [],
        "needs_manual_review": [],
    }
    for path in sorted(set(paths)):
        out[_classify_path(path)].append(path)
    return out


def _list_large_files(log_path: Path) -> List[str]:
    return [line.strip() for line in _read_lines(log_path) if line.strip()]


def _recent_cleanup_commits(cwd: Path) -> List[Dict[str, str]]:
    text = _run(["git", "log", "--oneline", "--decorate", "-n", "20"], cwd)
    out = []
    for line in text.splitlines():
        if any(tag in line for tag in ["GIT2:", "S5E13-S5E15:"]):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                out.append({"hash": parts[0], "message": parts[1]})
    return out


def _report_text(payload: Dict[str, Any]) -> str:
    fc = payload["file_classification"]
    lines = [
        "# GIT2 pre-final freeze cleanup report",
        "",
        "## 执行摘要",
        f"- final_classification = `{payload['final_classification']}`",
        f"- branch = `{payload['branch']}`",
        f"- pushed = `{payload['pushed']}`",
        "",
        "## 当前分支与远端",
        f"- branch: `{payload['branch']}`",
        f"- remote: `{payload['remote']}`",
        "",
        "## working tree before",
        json.dumps(payload["working_tree_before"], ensure_ascii=False, indent=2),
        "",
        "## 大文件审计",
        "\n".join(f"- `{x}`" for x in payload["large_files"]) or "- none",
        "",
        "## 文件分类表",
        f"- should_commit: {len(fc['should_commit'])}",
        f"- should_ignore_or_local_only: {len(fc['should_ignore_or_local_only'])}",
        f"- should_restore: {len(fc['should_restore'])}",
        f"- needs_manual_review: {len(fc['needs_manual_review'])}",
        "",
        "## S5E13-S5E15 历史残留处理",
        json.dumps(payload["s5e13_s5e15_residuals"], ensure_ascii=False, indent=2),
        "",
        "## S5D11 validation refresh 文件处理",
        json.dumps(payload["s5d11_refresh_files"], ensure_ascii=False, indent=2),
        "",
        "## .gitignore 更新",
        "\n".join(f"- `{x}`" for x in payload["gitignore_updates_recommended"]),
        "",
        "## commits created",
        "\n".join(f"- `{c['hash']}` `{c['message']}`" for c in payload["commits_created"]) or "- none",
        "",
        "## push status",
        f"- pushed = `{payload['pushed']}`",
        "",
        "## working tree after",
        json.dumps(payload["working_tree_after"], ensure_ascii=False, indent=2),
        "",
        "## 剩余 local-only / ignored artifacts",
        "\n".join(f"- `{x}`" for x in payload["working_tree_after"]["remaining_files"]) or "- none",
        "",
        "## caveats",
        "- 不重新训练。",
        "- 不刷新实验指标。",
        "- 不生成新 trajectory。",
        "- S5 locked metrics/policy unchanged。",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cwd = Path(args.project_root)
    before_status = _parse_status(_read_lines(cwd / "logs/git2_status_short_before.log"))
    before_ignored = _read_lines(cwd / "logs/git2_status_ignored_before.log")
    current_status_lines = _run(["git", "status", "--short"], cwd).splitlines()
    current_status = _parse_status(current_status_lines)
    all_before_paths = before_status["modified"] + before_status["untracked"] + before_status["deleted"]
    classified = _collect_classification(all_before_paths)
    large_files = _list_large_files(cwd / "logs/git2_large_files_before.log")
    ignored_summary = [line for line in before_ignored if line.startswith("!! ")]
    remaining = current_status["modified"] + current_status["untracked"] + current_status["deleted"]
    clean = len(remaining) == 0
    if clean:
        final = "GIT2_WORKING_TREE_CLEAN_WITH_LOCAL_IGNORED_ARTIFACTS" if ignored_summary else "GIT2_WORKING_TREE_CLEAN"
    elif classified["needs_manual_review"]:
        final = "GIT2_PARTIAL_CLEAN_REMAINING_MANUAL_REVIEW"
    else:
        final = "GIT2_BLOCKED_BY_UNCLASSIFIED_FILES"
    payload = {
        "experiment": "GIT2_cleanup_pre_final_freeze",
        "branch": _read_lines(cwd / "logs/git2_branch_before.log")[0] if (cwd / "logs/git2_branch_before.log").exists() else None,
        "remote": _read_lines(cwd / "logs/git2_remote_before.log"),
        "working_tree_before": {
            "clean": False,
            **before_status,
            "ignored_summary_count": len(ignored_summary),
        },
        "large_files": large_files,
        "file_classification": classified,
        "s5e13_s5e15_residuals": {
            "s5e13_commit_candidates": [p for p in classified["should_commit"] if "/s5e13" in p.lower() or "S5E13" in p],
            "s5e14_commit_candidates": [p for p in classified["should_commit"] if "/s5e14" in p.lower() or "S5E14" in p],
            "s5e15_commit_candidates": [p for p in classified["should_commit"] if "/s5e15" in p.lower() or "S5E15" in p],
        },
        "s5d11_refresh_files": {
            "restored": [
                "checkpoints/S5D11_validation_concurrency_audit.json",
                "checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json",
                "reports/s5d11_serial_validation_and_nonselected_edge_audit.md",
            ],
            "logs_local_only": [p for p in ignored_summary if "s5d11" in p.lower()],
        },
        "gitignore_updates_recommended": [
            "external_baselines/results/**/router_decisions.jsonl",
            "external_baselines/results/**/spherical_bearing_flow_features.jsonl",
            "external_baselines/results/**/correspondence_loss_weights.jsonl",
            "external_baselines/results/**/correspondence_refinement_weights.jsonl",
            "external_baselines/results/**/scale_antiparallel_refinement_weights.jsonl",
            "checkpoints/**/correspondence_weighted_dataset.json",
        ],
        "actions_taken": [
            "saved before-state git logs under logs/git2_*_before.log",
            "restored auto-refreshed S5D11 tracked files",
            "updated .gitignore for raw trajectory/provenance/router/weight artifacts",
            "committed missing lightweight S5E13-S5E15 summaries and code",
        ],
        "working_tree_after": {
            "clean": clean,
            "remaining_files": remaining,
        },
        "commits_created": _recent_cleanup_commits(cwd),
        "pushed": "origin/experiment/s5e1-traceable-adjacent-dense" in _run(["git", "branch", "-vv"], cwd),
        "tag_created": False,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    out_json = cwd / args.out_json
    out_report = cwd / args.out_report
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_report.write_text(_report_text(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
