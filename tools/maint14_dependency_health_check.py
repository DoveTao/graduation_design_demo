#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON = REPO_ROOT / "reports" / "MAINT15_dependency_health_after_label_trim.json"
REPORT_MD = REPO_ROOT / "reports" / "MAINT15_dependency_health_after_label_trim.md"
DELETED_MANIFEST = REPO_ROOT / "reports" / "MAINT13_deleted_files_manifest.json"

SCAN_TARGETS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CURRENT_MAINLINE.md",
    REPO_ROOT / "PROJECT_STRUCTURE.md",
    REPO_ROOT / "tools" / "current",
    REPO_ROOT / "tools" / "final360i_retrain_and_select.py",
    REPO_ROOT / "tools" / "train360e_sequence_trajectory_export_and_ate_eval.py",
    REPO_ROOT / "tools" / "train_seq360b_lightweight_scale_smoothing.py",
    REPO_ROOT / "models" / "struct360b_match_free_coarse_to_fine.py",
    REPO_ROOT / "models" / "seq360b_scale_smoothing_head.py",
    REPO_ROOT / "configs" / "final360i_struct360b_final.yaml",
    REPO_ROOT / "configs" / "seq360b_lightweight_scale_smoothing.yaml",
    REPO_ROOT / "train360" / "core",
    REPO_ROOT / "datasets" / "dset2c_manifest_dataset.py",
    REPO_ROOT / "datasets" / "dset2c_sequence_clip_dataset.py",
]

TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".txt"}

OPTIONAL_REPORTS = [
    REPO_ROOT / "reports" / "FINAL360I_final_retrain_and_model_selection.md",
    REPO_ROOT / "reports" / "FINAL360I_metrics_val.json",
    REPO_ROOT / "reports" / "FINAL360I_metrics_test.json",
    REPO_ROOT / "reports" / "SEQ360B_train_lightweight_scale_smoothing_head.md",
    REPO_ROOT / "reports" / "SEQ360B_metrics_test.json",
    REPO_ROOT / "reports" / "TRAIN360E_sequence_trajectory_export_and_ATE_eval.md",
    REPO_ROOT / "reports" / "TRAIN360E_metrics_test.json",
    REPO_ROOT / "reports" / "BASE360D_component_metric_alignment.md",
    REPO_ROOT / "reports" / "BASE360D_metrics_test.json",
    REPO_ROOT / "reports" / "RESULTS360_main_results_table.md",
]

INTENTIONALLY_OMITTED_PATHS = {
    "reports/BASE360D_metrics_val.json",
    "reports/FINAL360I_vs_all_baselines_summary.md",
    "external_baselines/results/seq360b_scale_smoothing_trajectory",
    "reports/SEQ360B_metrics_val.json",
    "reports/SEQ360B_trajectory_metrics_val.json",
    "reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md",
}

KEYWORDS = [
    "TRAIN360A",
    "TRAIN360B",
    "TRAIN360C",
    "TRAIN360D",
    "TRAIN360H",
    "STRUCT360A",
    "STRUCT360C",
    "SEQ360A",
    "DEV360",
    "MAINT",
    "HANDOFF",
    "RUNHANDOFF",
    "S5E15",
    "scene01",
    "C31",
    "dataset_pano_only",
    "train_mvp",
]

PATH_RE = re.compile(r"(?:reports|configs|checkpoints|tools|external_baselines|data)/[A-Za-z0-9_./-]+")
ABSOLUTE_PATH_RE = re.compile(r"(/home/[^\s\"'`]+)")

CURRENT_REQUIRED_PATTERNS = (
    "configs/final360i_struct360b_final.yaml",
    "configs/seq360b_lightweight_scale_smoothing.yaml",
    "configs/struct360b_match_free_coarse_to_fine.yaml",
    "tools/final360i_retrain_and_select.py",
    "tools/train360e_sequence_trajectory_export_and_ate_eval.py",
    "tools/train_seq360b_lightweight_scale_smoothing.py",
    "tools/current/",
    "models/struct360b_match_free_coarse_to_fine.py",
    "models/seq360b_scale_smoothing_head.py",
    "datasets/dset2c_manifest_dataset.py",
    "datasets/dset2c_sequence_clip_dataset.py",
    "train360/core/",
    "external_baselines/results/dset2c_360dvo_canonical/",
)


def _iter_files(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path
        elif path.is_dir():
            for child in sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and "__pycache__" not in p.parts
            ):
                yield child


def _load_deleted_paths() -> set[str]:
    if not DELETED_MANIFEST.exists():
        return set()
    payload = json.loads(DELETED_MANIFEST.read_text(encoding="utf-8"))
    deleted: set[str] = set()
    for key in ("tracked_deleted", "untracked_deleted"):
        rows = payload.get(key, [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("path"):
                    deleted.add(str(row["path"]))
                elif isinstance(row, str):
                    deleted.add(row)
    return deleted


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _guess_exists(reference: str) -> bool:
    if reference.startswith("/home/"):
        return Path(reference).exists()
    return (REPO_ROOT / reference).exists()


def _is_current_required(reference: str) -> bool:
    return any(pattern in reference for pattern in CURRENT_REQUIRED_PATTERNS)


def _categorize(file_rel: str, reference: str, exists_now: bool, deleted_paths: set[str], line_text: str) -> Tuple[str, str]:
    ref_lower = reference.lower()
    line_lower = line_text.lower()
    has_legacy_keyword = any(keyword.lower() in reference.lower() or keyword.lower() in line_lower for keyword in KEYWORDS)
    is_keyword_only = "/" not in reference and not reference.startswith("/home/")
    is_output_path = (
        reference.startswith("reports/FINAL360I_")
        or reference.startswith("reports/SEQ360B_")
        or reference.startswith("external_baselines/results/seq360b_scale_smoothing_trajectory")
        or reference.startswith("checkpoints/FINAL360I_struct360b_final")
        or reference.startswith("checkpoints/SEQ360B_lightweight_scale_smoothing")
    )
    if exists_now and _is_current_required(reference):
        return "current_required", "keep"
    if exists_now and ("status_summary" in reference or "status summaries" in line_lower or "no longer active codepaths" in line_lower):
        return "current_optional", "keep"
    if is_keyword_only and has_legacy_keyword:
        if "status summary" in line_lower or "cleanup summaries" in line_lower or "no longer active codepaths" in line_lower:
            return "current_optional", "keep"
        return "legacy_reference", "keep"
    if exists_now and (reference.startswith("reports/") or file_rel.startswith("README") or file_rel.endswith(".md")):
        return ("legacy_reference", "keep") if has_legacy_keyword else ("current_optional", "keep")
    if exists_now and has_legacy_keyword:
        return "legacy_reference", "keep"
    if exists_now:
        return "current_optional", "keep"
    if reference in INTENTIONALLY_OMITTED_PATHS:
        return "current_optional", "mark_intentionally_omitted"
    if is_output_path:
        return "current_optional", "mark_intentionally_omitted"
    if reference in deleted_paths or has_legacy_keyword:
        if "status_summary" in line_lower or "status summary" in line_lower:
            return "current_optional", "keep"
        if "base360d_metrics_val" in ref_lower or "final360i_vs_all_baselines_summary" in ref_lower:
            return "current_optional", "make_optional"
        if "seq360a" in ref_lower or "struct360c" in ref_lower:
            return "deleted_reference", "replace_path"
        return "deleted_reference", "remove_reference"
    if reference.startswith("/home/"):
        return "unknown", "replace_path"
    return "unknown", "replace_path"


def _collect_records() -> List[Dict[str, Any]]:
    deleted_paths = _load_deleted_paths()
    records: List[Dict[str, Any]] = []
    seen = set()
    for file_path in _iter_files(SCAN_TARGETS):
        file_rel = _repo_rel(file_path)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            refs = PATH_RE.findall(line) + ABSOLUTE_PATH_RE.findall(line)
            keyword_hits = [keyword for keyword in KEYWORDS if keyword in line]
            for reference in refs + keyword_hits:
                key = (file_rel, line_no, reference)
                if key in seen:
                    continue
                seen.add(key)
                exists_now = _guess_exists(reference) if "/" in reference or reference.startswith("/home/") else False
                category, action = _categorize(file_rel, reference, exists_now, deleted_paths, line)
                records.append(
                    {
                        "file": file_rel,
                        "line": line_no,
                        "reference": reference,
                        "exists_now": exists_now,
                        "category": category,
                        "recommended_action": action,
                    }
                )
    records.sort(key=lambda row: (row["file"], row["line"], row["reference"]))
    return records


def _render_markdown(records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = [
        "# MAINT15 dependency health after label trim",
        "",
        "## Summary",
        f"- scanned files: `{summary['scanned_files']}`",
        f"- total references: `{summary['total_references']}`",
        f"- current_required missing: `{summary['current_required_missing']}`",
        f"- current_optional missing: `{summary['current_optional_missing']}`",
        f"- intentional omissions: `{summary['intentional_omission_count']}`",
        f"- deleted references: `{summary['deleted_reference_count']}`",
        f"- health status: `{summary['health_status']}`",
        "",
        "## Reference table",
        "| file | line | reference | exists_now | category | recommended_action |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in records:
        lines.append(
            f"| {row['file']} | {row['line']} | `{row['reference']}` | `{str(row['exists_now']).lower()}` | `{row['category']}` | `{row['recommended_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Optional report snapshot",
        ]
    )
    for path in OPTIONAL_REPORTS:
        lines.append(f"- `{_repo_rel(path)}`: `{'present' if path.exists() else 'missing_after_cleanup'}`")
    lines.extend(
        [
            "",
            "## Intentional omissions",
        ]
    )
    for path in sorted(INTENTIONALLY_OMITTED_PATHS):
        lines.append(f"- `{path}`: `{'present' if (REPO_ROOT / path).exists() else 'intentionally_omitted_after_cleanup'}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    records = _collect_records()
    scanned_files = list(_iter_files(SCAN_TARGETS))
    summary = {
        "scanned_files": len(scanned_files),
        "total_references": len(records),
        "current_required_missing": sum(
            1 for row in records if row["category"] == "current_required" and not row["exists_now"] and "/" in row["reference"]
        ),
        "current_optional_missing": sum(
            1 for row in records if row["category"] == "current_optional" and not row["exists_now"] and "/" in row["reference"]
        ),
        "intentional_omission_count": sum(
            1 for row in records if row["recommended_action"] == "mark_intentionally_omitted" and not row["exists_now"]
        ),
        "deleted_reference_count": sum(1 for row in records if row["category"] == "deleted_reference"),
        "health_status": "pass",
    }
    if summary["current_required_missing"] > 0:
        summary["health_status"] = "fail"
    elif summary["deleted_reference_count"] > 0:
        summary["health_status"] = "partial"
    elif summary["current_optional_missing"] > 0:
        summary["health_status"] = "pass_with_intentional_omissions"
    payload = {
        "task_name": "MAINT15_reduce_legacy_comparison_labels_inside_final360i_reports_and_configs",
        "summary": summary,
        "records": records,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(records, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
