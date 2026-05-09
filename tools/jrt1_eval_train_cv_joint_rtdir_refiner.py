#!/usr/bin/env python3
"""Write the JRT1b train-CV report from train-CV diagnostics JSON."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.6f}"


def _ms(row: Dict[str, Any], key: str) -> str:
    obj = row[key]
    return f"{_fmt(obj['mean'])} / {_fmt(obj['std'])}"


def _write_report(path: Path, config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    folds = payload["folds"]
    variants = payload["variants"]
    mean_cv = payload["mean_cv_metrics"]
    decisions = payload["gate_decision"]
    fold_table = [
        "| fold | train_pairs | val_pairs |",
        "| --- | ---: | ---: |",
    ]
    for f in folds:
        fold_table.append(f"| {f['fold_id']} | {f['num_train_pairs']} | {f['num_val_pairs']} |")
    variant_table = [
        "| variant | description | weights | hardcase_alpha |",
        "| --- | --- | --- | ---: |",
    ]
    for v in variants:
        variant_table.append(f"| {v['name']} | {v['description']} | `{v['weights']}` | {v['hardcase_alpha']} |")
    metrics_table = [
        "| variant | rot_mean mean/std | tdir_mean mean/std | tdir_cos mean/std | tmag_log_err mean/std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in mean_cv.items():
        metrics_table.append(
            f"| {name} | {_ms(row, 'rot_mean_deg')} | {_ms(row, 'tdir_mean_deg')} | {_ms(row, 'tdir_mean_cosine')} | {_ms(row, 'tmag_mean_log_error')} |"
        )
    gate_table = [
        "| variant | tdir_improvement | rot_worsening | tdir_cos_improved | trajectory_gate_available | gate_status |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for name, row in decisions.items():
        gate_table.append(
            f"| {name} | {_fmt(row['tdir_improvement_deg'])} | {_fmt(row['rot_worsening_deg'])} | {row['tdir_cos_improved']} | {row['trajectory_gate_available']} | {row['gate_status']} |"
        )
    any_tdir = any(
        name != "A_s5_no_refiner_reference" and float(row["tdir_improvement_deg"]) > 0.0 and row["tdir_cos_improved"]
        for name, row in decisions.items()
    )
    joint_names = [
        "D_joint_rtdir_residual",
        "E_joint_rtdir_tdir_weighted",
        "F_joint_rtdir_tdir_hardcase_weighted",
    ]
    best_variant = min(mean_cv.items(), key=lambda kv: float(kv[1]["tdir_mean_deg"]["mean"]))[0]
    lines = [
        "# JRT1b Train-CV Joint R/tdir Refiner",
        "",
        "## Scope",
        "",
        "JRT1b is train-CV only. It does not run final test, and S5 remains the final candidate.",
        "",
        "## JRT1a Recap",
        "",
        "JRT1a showed that rotation residual learning works in smoke, while translation direction did not improve. JRT1b therefore adds translation-direction weighted and train-only hardcase-weighted variants.",
        "",
        "## Dataset and Folds",
        "",
        f"- dataset source: `{payload['dataset_summary']['dataset_source']}`",
        f"- cached train-split pairs: `{payload['dataset_summary']['num_cached_train_split_pairs']}`",
        f"- cached test pairs: `{payload['dataset_summary']['num_test_pairs_cached']}`",
        f"- no test GT cached/used: `{payload['dataset_summary']['no_test_gt_used_for_training']}`",
        "",
        *fold_table,
        "",
        "## Variants",
        "",
        *variant_table,
        "",
        "## Train-CV Component Metrics",
        "",
        *metrics_table,
        "",
        "## Gate Decision",
        "",
        *gate_table,
        "",
        "## Interpretation",
        "",
        f"- translation-direction signal present: `{any_tdir}`",
        f"- best variant by mean CV tdir error: `{best_variant}`",
        f"- joint variants evaluated: `{joint_names}`",
        f"- trajectory gate available: `{payload['trajectory_gate_available']}`",
        "Component-only train-CV diagnostics are insufficient for a final clean claim without trajectory gate support.",
        "",
        "## Final Classification",
        "",
        f"`{payload['final_classification']}`",
        "",
        "## Next Step",
        "",
        "Do not run final test unless a later run has both a component gate pass and an available trajectory gate.",
        "",
        "## Caveats",
        "",
        "- no final candidate selected",
        "- no practical-ready claim",
        "- S5 unchanged",
        "- small sequence protocol",
        "- component-only signal is insufficient for final clean claim",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write JRT1b train-CV report.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    payload = _read_json(_resolve(args.results))
    out_md = _resolve(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_report(out_md, config, payload)
    print(out_md.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
