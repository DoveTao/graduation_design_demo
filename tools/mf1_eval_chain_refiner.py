#!/usr/bin/env python3
"""Evaluate MF1a smoke results and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List


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


def _classification(train_log: Dict[str, Any]) -> str:
    if train_log.get("status") != "ok":
        return "MF1A-TRAINING-HARNESS-ISSUE"
    vals = {row["variant"]: row["val_metrics_after"] for row in train_log["variants"]}
    ref = vals.get("A_s5_chain_reference")
    if not ref:
        return "MF1A-CHAIN-DATASET-ISSUE"
    has_signal = False
    for name, m in vals.items():
        if name == "A_s5_chain_reference":
            continue
        if float(m["ATE_proxy"]) < float(ref["ATE_proxy"]) or float(m["drift_proxy"]) < float(ref["drift_proxy"]) or float(m["tdir_mean_deg"]) < float(ref["tdir_mean_deg"]):
            has_signal = True
    return "MF1A-FRAMEWORK-SMOKE-PASS" if has_signal else "MF1A-NO-CHAIN-SIGNAL"


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    rows = [
        "| variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean_deg | tdir_mean_deg | tdir_mean_cosine | tmag_mean_log_error | num_chains | num_pairs | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["variant_metrics"]:
        m = row["metrics"]
        rows.append(
            f"| {row['variant']} | {_fmt(m['ATE_proxy'])} | {_fmt(m['drift_proxy'])} | {_fmt(m['path_ratio_proxy'])} | {_fmt(m['rot_mean_deg'])} | {_fmt(m['tdir_mean_deg'])} | {_fmt(m['tdir_mean_cosine'])} | {_fmt(m['tmag_mean_log_error'])} | {m['num_chains']} | {m['num_pairs']} | {row['status']} |"
        )
    meta = payload["dataset_metadata"]
    lines = [
        "# MF1a Chain Refiner Smoke Report",
        "",
        "## Scope",
        "",
        "MF1a is a multi-frame / chain-level refiner smoke experiment. It is not a final candidate, does not run final test, and does not replace S5.",
        "",
        "## Motivation",
        "",
        "JRT1 closed as `NO_STABLE_JRT1_TRAJECTORY_GAIN`, suggesting pair-level residual correction is not enough for trajectory accumulation. MF1a tests whether a lightweight chain-context refiner can provide a better smoke signal.",
        "",
        "## Dataset",
        "",
        f"- source: `{meta.get('dataset_source')}`",
        f"- window_size: `{meta.get('window_size')}`",
        f"- train/val chains: `{meta.get('num_train_chains')}` / `{meta.get('num_val_chains')}`",
        f"- no test GT used for training: `{meta.get('no_test_gt_used_for_training')}`",
        "",
        "## Metrics",
        "",
        *rows,
        "",
        "## Classification",
        "",
        f"`{payload['classification']}`",
        "",
        "## Caveats",
        "",
        "- smoke only",
        "- no final test",
        "- no candidate replacement",
        "- S5 locked metrics unchanged",
        "- no practical-ready claim",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MF1a smoke train log.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    train_log = _read_json(_resolve(args.train_log))
    classification = _classification(train_log)
    variant_metrics = [{"variant": row["variant"], "status": row["status"], "metrics": row["val_metrics_after"], "train_loss_initial": row.get("train_loss_initial"), "train_loss_final": row.get("train_loss_final")} for row in train_log["variants"]]
    payload = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "base_candidate": config["base_candidate"],
        "locked_s5_metrics": config["locked_s5_metrics"],
        "dataset_metadata": train_log["dataset_metadata"],
        "variant_metrics": variant_metrics,
        "classification": classification,
        "no_final_test": True,
        "s5_remains_final": True,
    }
    out_json = _resolve(args.output_json)
    out_md = _resolve(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(out_md, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
