#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED_S5 = {
    "ATE": 7.352288,
    "drift": 1.327343,
    "path_ratio": 0.932379,
}


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _git_show(branch: str, repo_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{repo_path}"], cwd=REPO_ROOT)


def _blob_hash(branch: str, repo_path: str) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", f"{branch}:{repo_path}"], cwd=REPO_ROOT, text=True)
        .strip()
    )


def _infer_role(target_path: str) -> str:
    parts = Path(target_path).parts
    if len(parts) < 2:
        return "unknown"
    if parts[1] == "final":
        return "final"
    if parts[1] == "ablations":
        return "ablation"
    if parts[1] == "baselines":
        return "baseline"
    if parts[1] == "diagnostics":
        return "diagnostic"
    if parts[1] == "audits":
        return "audit"
    if parts[1] == "environment":
        return "environment"
    if parts[1] == "negative_experiments":
        return "negative_experiment"
    return "unknown"


def _build_readme(manifest: Dict[str, Any]) -> str:
    copied = sum(1 for row in manifest["entries"] if row["status"] == "copied")
    missing = sum(1 for row in manifest["entries"] if row["status"] == "missing")
    skipped = sum(1 for row in manifest["entries"] if row["status"] == "skipped_needs_review")
    branches = sorted({row["source_branch"] for row in manifest["entries"]})
    lines: List[str] = [
        "# Curated Reports Archive",
        "",
        "This directory contains copied canonical report snapshots from multiple branches.",
        "Original reports remain in their source branches.",
        "No source reports were deleted or moved.",
        "This archive is for thesis / final review convenience.",
        "",
        "## Source Branches",
        "",
    ]
    for branch in branches:
        lines.append(f"- `{branch}`")
    lines += [
        "",
        "## Directory Structure",
        "",
        "- `reports_curated/final/`",
        "- `reports_curated/ablations/`",
        "- `reports_curated/baselines/`",
        "- `reports_curated/diagnostics/`",
        "- `reports_curated/negative_experiments/jrt1/`",
        "- `reports_curated/negative_experiments/mf1/`",
        "- `reports_curated/audits/`",
        "- `reports_curated/environment/`",
        "",
        "## Provenance",
        "",
        "Use `reports_curated/MANIFEST.json` to verify the source branch, source path, source commit, and source blob hash for each copied file.",
        "",
        "## Materialization Summary",
        "",
        f"- copied canonical reports: `{copied}`",
        f"- missing sources: `{missing}`",
        f"- skipped for review: `{skipped}`",
        "",
        "## S5 Locked Metrics",
        "",
        f"- ATE = `{LOCKED_S5['ATE']:.6f}`",
        f"- drift = `{LOCKED_S5['drift']:.6f}`",
        f"- path_ratio = `{LOCKED_S5['path_ratio']:.6f}`",
        "",
        "S5 remains final clean candidate.",
        "",
        "## Caveats",
        "",
        "- S5 official locked metrics and dense external / same-evaluator metrics have scope differences.",
        "- ORB-SLAM3 is an external baseline and is not a replacement for S5.",
        "- JRT1 and MF1 are negative experiments and remain archived as such.",
        "- This archive is a copied snapshot set; branch-local reports still live on their source branches.",
        "",
    ]
    if missing or skipped:
        lines += ["## Missing or Skipped Sources", ""]
        for row in manifest["entries"]:
            if row["status"] != "copied":
                lines.append(
                    f"- `{row['target_path']}` <- `{row['source_branch']}:{row['source_path']}` (`{row['status']}`)"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize curated reports from REPORT3 plan.")
    ap.add_argument(
        "--plan-json",
        default="checkpoints/REPORT3_curated_report_archive_plan.json",
    )
    ap.add_argument(
        "--output-root",
        default="reports_curated",
    )
    args = ap.parse_args()

    _run_guard()
    plan_path = _resolve(args.plan_json)
    output_root = _resolve(args.output_root)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []

    output_root.mkdir(parents=True, exist_ok=True)
    for rel in (
        "final",
        "ablations",
        "baselines",
        "diagnostics",
        "negative_experiments/jrt1",
        "negative_experiments/mf1",
        "audits",
        "environment",
    ):
        (output_root / rel).mkdir(parents=True, exist_ok=True)

    for row in plan["canonical_report_selection"]:
        target_rel = row["curated_target_path"]
        target = _resolve(target_rel)
        source_branch = row["source_branch"]
        source_path = row["source_path"]
        status = "copied"
        notes = row["reason"]
        blob_hash = None
        if row["status"] == "needs_user_review":
            status = "skipped_needs_review"
            notes = f"Skipped during materialization because REPORT3 marked it needs review: {row['reason']}"
        else:
            try:
                content = _git_show(source_branch, source_path)
                blob_hash = _blob_hash(source_branch, source_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            except subprocess.CalledProcessError:
                status = "missing"
                notes = f"Source file missing at materialization time: {source_branch}:{source_path}"
        entries.append(
            {
                "target_path": target_rel,
                "source_branch": source_branch,
                "source_path": source_path,
                "source_commit": row["source_commit"],
                "source_blob_hash": blob_hash,
                "status": status,
                "role": _infer_role(target_rel),
                "notes": notes,
            }
        )

    manifest = {
        "materialized_at_git_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "no_original_reports_deleted": True,
        "no_original_reports_moved": True,
        "no_branches_merged": True,
        "no_cherry_picks": True,
        "s5_locked_metrics": LOCKED_S5,
        "entries": entries,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(_build_readme(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"copied": sum(1 for x in entries if x["status"] == "copied"),
                      "missing": sum(1 for x in entries if x["status"] == "missing"),
                      "skipped": sum(1 for x in entries if x["status"] == "skipped_needs_review")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
