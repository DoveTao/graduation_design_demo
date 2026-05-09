#!/usr/bin/env python3
"""Write MF1b train-CV chain refiner report."""

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
    return "nan" if not math.isfinite(x) else f"{x:.6f}"


def _write_report(path: Path, config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    fold_rows = ["| fold | train_chains | val_chains |", "| --- | ---: | ---: |"]
    for f in payload["folds"]:
        fold_rows.append(f"| {f['fold_id']} | {f['num_train_chains']} | {f['num_val_chains']} |")
    var_rows = ["| variant | role |", "| --- | --- |"]
    roles = {
        "A_s5_chain_reference": "no refiner",
        "B_pairwise_jrt_style_reference": "independent pairwise residual across chain",
        "C_temporal_conv_chain_refiner": "temporal convolution over chain",
        "D_gru_chain_refiner": "GRU over chain",
        "E_gru_chain_refiner_pathratio_loss": "GRU with stronger path-ratio penalty",
        "F_gru_chain_refiner_tmag_head_diagnostic": "GRU with diagnostic tmag residual head",
    }
    for v in payload["variants"]:
        var_rows.append(f"| {v} | {roles.get(v, '')} |")
    mean_rows = [
        "| variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean | tdir_mean | tdir_cos | gate_status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for v, row in payload["mean_cv_metrics"].items():
        gate = payload["gate_decision"].get(v, {}).get("gate_status", "")
        mean_rows.append(
            f"| {v} | {_fmt(row['ATE_proxy']['mean'])} | {_fmt(row['drift_proxy']['mean'])} | {_fmt(row['path_ratio_proxy']['mean'])} | {_fmt(row['rot_mean_deg']['mean'])} | {_fmt(row['tdir_mean_deg']['mean'])} | {_fmt(row['tdir_mean_cosine']['mean'])} | {gate} |"
        )
    decision = payload["final_classification"]
    lines = [
        "# MF1b Train-CV Chain Refiner",
        "",
        "## Scope",
        "",
        "MF1b is train-CV only. It runs no final test, selects no final candidate, and S5 remains final clean candidate.",
        "",
        "## MF1a Recap",
        "",
        "MF1a smoke passed. GRU and JRT-style variants improved proxy ATE/drift, while path_ratio stayed around 1.445 because scale behavior was unchanged.",
        "",
        "## Dataset and Folds",
        "",
        f"- dataset source: `{payload['dataset_metadata'].get('dataset_source')}`",
        f"- no test GT used: `{payload['dataset_metadata'].get('no_test_gt_used_for_training')}`",
        "",
        *fold_rows,
        "",
        "## Variants",
        "",
        *var_rows,
        "",
        "## Mean CV Results",
        "",
        *mean_rows,
        "",
        "## Gate Decision",
        "",
        f"Final classification: `{decision}`",
        f"Selected candidate for next stage: `{payload.get('selected_candidate_for_next_stage')}`",
        "",
        "## Interpretation",
        "",
        "The report compares all variants against A_s5_chain_reference on train-CV folds. It must not be read as a final clean result. The key blocker is whether path_ratio enters the [0.90, 1.05] gate while ATE/drift and component metrics remain stable.",
        "",
        "## Final Classification",
        "",
        f"`{decision}`",
        "",
        "## Next Step",
        "",
        "If gate pass, prepare a separate MF1c final-test plan. Otherwise close out or revise the model objective.",
        "",
        "## Caveats",
        "",
        "- train-CV only",
        "- no final test",
        "- no candidate replacement",
        "- S5 unchanged",
        "- no deployment-readiness claim",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write MF1b train-CV report.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    _run_guard()
    config = _read_json(_resolve(args.config))
    payload = _read_json(_resolve(args.results))
    out = _resolve(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(out, config, payload)
    print(out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
