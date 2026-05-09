#!/usr/bin/env python3
"""Evaluate JRT1a smoke variants with pair-level component diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from jrt1_train_joint_rtdir_refiner import JointRefinerMLP, _load_dataset, component_metrics


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


def _load_model(path: Path) -> JointRefinerMLP:
    ckpt = torch.load(str(path), map_location="cpu")
    model = JointRefinerMLP(int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    meta = payload["dataset_metadata"]
    train_rows = payload["training_log"]["variants"]
    metrics = payload["variant_metrics"]
    train_table = [
        "| variant | status | train_loss_initial | train_loss_final | num_updates | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in train_rows:
        notes = row.get("error_if_any") or "smoke harness row"
        train_table.append(
            "| {variant} | {status} | {li} | {lf} | {updates} | {notes} |".format(
                variant=row["variant"],
                status=row["status"],
                li="" if row.get("train_loss_initial") is None else _fmt(row["train_loss_initial"]),
                lf="" if row.get("train_loss_final") is None else _fmt(row["train_loss_final"]),
                updates=row["num_updates"],
                notes=notes,
            )
        )
    diag_table = [
        "| variant | rot_mean_deg | rot_p90_deg | tdir_mean_deg | tdir_p90_deg | tdir_mean_cosine | tmag_mean_log_error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, row in metrics.items():
        diag_table.append(
            "| {variant} | {rot_mean} | {rot_p90} | {tdir_mean} | {tdir_p90} | {cos} | {tmag} |".format(
                variant=variant,
                rot_mean=_fmt(row["rot_mean_deg"]),
                rot_p90=_fmt(row["rot_p90_deg"]),
                tdir_mean=_fmt(row["tdir_mean_deg"]),
                tdir_p90=_fmt(row["tdir_p90_deg"]),
                cos=_fmt(row["tdir_mean_cosine"]),
                tmag=_fmt(row["tmag_mean_log_error"]),
            )
        )
    lines = [
        "# JRT1a Joint R/tdir Refiner Framework Smoke",
        "",
        "## Scope",
        "",
        "JRT1a is a framework and smoke-test harness only. It does not select a final candidate, does not run final test, and leaves S5 as the final clean candidate.",
        "",
        "## Motivation",
        "",
        "S13 oracle evidence isolates a coupled R/tdir bottleneck: oracle_R alone worsened ATE to 10.590301 and drift to 1.516962; oracle_tdir alone remained limited at ATE 7.658319 and drift 1.375470; oracle_R_tdir improved sharply to ATE 0.911024 and drift 0.304305.",
        "",
        "## Dataset Builder",
        "",
        f"- dataset source: `{meta.get('dataset_source')}`",
        f"- num pairs: `{meta.get('num_total_pairs')}`",
        f"- train/val/test cached counts: `{meta.get('num_train_pairs')}` / `{meta.get('num_val_pairs')}` / `{meta.get('num_test_pairs')}`",
        f"- available test pairs counted but not cached for training: `{meta.get('num_available_test_pairs')}`",
        f"- feature fields: `{meta.get('feature_fields')}`",
        f"- label fields: `{meta.get('label_fields')}`",
        f"- test GT excluded from training: `{meta.get('no_test_gt_used_for_training')}`",
        "",
        "## Model",
        "",
        "Prediction-space 2-layer MLP residual refiner. Variants are A_s5_no_refiner_reference, B_rot_only_residual_smoke, C_tdir_only_residual_smoke, and D_joint_rtdir_residual_smoke. Loss terms are geodesic rotation loss, translation-direction cosine loss, a multiplicative coupling term, and residual regularization.",
        "",
        "## Smoke Training",
        "",
        *train_table,
        "",
        "## Component Diagnostics",
        "",
        *diag_table,
        "",
        "## Smoke Classification",
        "",
        f"`{payload['smoke_classification']}`",
        "",
        "## Next Step",
        "",
        "Proceed to JRT1b train-CV variants only if this smoke classification remains pass; otherwise fix the dataset or harness before algorithm claims.",
        "",
        "## Caveats",
        "",
        "- no final test",
        "- no candidate replacement",
        "- no practical-ready claim",
        "- S5 locked metrics unchanged",
        "- smoke results are not clean final results",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate JRT1a smoke refiner variants.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    data = _load_dataset(_resolve(args.dataset))
    train_log = _read_json(_resolve(args.train_log))
    train_idx = torch.where(data["split"] == 0)[0]
    x = data["features"]
    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    data["x_std"] = (x - mu) / sd
    val_idx = torch.where(data["split"] == 1)[0]
    if val_idx.numel() == 0:
        raise RuntimeError("JRT1a eval smoke needs a non-empty val split.")

    variant_metrics: Dict[str, Dict[str, float]] = {}
    status_issue = None
    for row in train_log["variants"]:
        variant = row["variant"]
        if row.get("status") != "ok":
            status_issue = "JRT1A-TRAINING-HARNESS-ISSUE"
        if variant == "A_s5_no_refiner_reference":
            model = None
        else:
            rel = row.get("model_state_path")
            if not rel:
                status_issue = "JRT1A-TRAINING-HARNESS-ISSUE"
                continue
            model = _load_model(_resolve(rel))
        variant_metrics[variant] = component_metrics(model, data, val_idx, variant)

    classification = status_issue or "JRT1A-FRAMEWORK-SMOKE-PASS"
    payload = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "base_candidate": config["base_candidate"],
        "locked_s5_metrics": config["locked_s5_metrics"],
        "dataset_metadata": data["metadata"],
        "training_log": train_log,
        "variant_metrics": variant_metrics,
        "smoke_classification": classification,
        "gate": {
            "no_final_test": True,
            "no_candidate_reselection": True,
            "no_test_gt_used_for_training": True,
            "s5_locked_metrics_unchanged": True,
        },
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
