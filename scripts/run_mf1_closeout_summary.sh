#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/verify_final_candidate.sh

/home/dovetao/miniconda3/envs/pytorch/bin/python - <<'PY'
import json
from pathlib import Path

ROOT = Path.cwd()
MF1A_PATH = ROOT / "checkpoints" / "MF1a_chain_refiner_smoke_results.json"
MF1B_PATH = ROOT / "checkpoints" / "MF1b_train_cv_results.json"
OUT_JSON = ROOT / "checkpoints" / "MF1_final_closeout_summary.json"
OUT_MD = ROOT / "reports" / "MF1_final_closeout_summary.md"

S5_LOCKED = {
    "ATE": 7.352288,
    "drift": 1.327343,
    "path_ratio": 0.932379,
}


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics, name):
    value = metrics[name]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def fmt(value):
    return f"{float(value):.6f}"


def mf1a_lookup(payload):
    return {row["variant"]: row["metrics"] for row in payload["variant_metrics"]}


def mf1b_mean_rows(payload):
    rows = []
    for variant in payload["variants"]:
        metrics = payload["mean_cv_metrics"][variant]
        gate = payload["gate_decision"][variant]["gate_status"]
        rows.append(
            {
                "variant": variant,
                "ATE_proxy": metric(metrics, "ATE_proxy"),
                "drift_proxy": metric(metrics, "drift_proxy"),
                "path_ratio": metric(metrics, "path_ratio_proxy"),
                "rot_mean": metric(metrics, "rot_mean_deg"),
                "tdir_mean": metric(metrics, "tdir_mean_deg"),
                "tdir_cos": metric(metrics, "tdir_mean_cosine"),
                "gate_status": gate,
            }
        )
    return rows


mf1a = read_json(MF1A_PATH)
mf1b = read_json(MF1B_PATH)
mf1a_metrics = mf1a_lookup(mf1a)
mf1b_rows = mf1b_mean_rows(mf1b)

summary = {
    "experiment_name": "MF1_multi_frame_chain_refiner_closeout",
    "final_classification": "NO_STABLE_MF1_CHAIN_GAIN",
    "s5_locked_metrics": S5_LOCKED,
    "mf1a_summary": {
        "classification": mf1a["classification"],
        "chain_dataset_and_train_harness_worked": True,
        "key_metrics": {
            "A_s5_chain_reference": mf1a_metrics["A_s5_chain_reference"],
            "D_gru_chain_refiner": mf1a_metrics["D_gru_chain_refiner"],
        },
        "interpretation": (
            "MF1a smoke passed. GRU and JRT-style smoke variants improved "
            "ATE_proxy, drift_proxy, rotation, and tdir, while path_ratio "
            "remained around 1.445 because scale was unchanged."
        ),
    },
    "mf1b_summary": {
        "classification": mf1b["final_classification"],
        "num_folds": len(mf1b["folds"]),
        "folds": mf1b["folds"],
        "train_split_only": True,
        "no_test_gt_used": True,
        "no_final_test": bool(mf1b["no_final_test"]),
        "mean_cv_metrics": mf1b_rows,
        "selected_candidate_for_next_stage": mf1b["selected_candidate_for_next_stage"],
        "interpretation": (
            "Train-CV retained trajectory-shape signal, but the stability gate "
            "did not pass. B/C/D/E improved ATE_proxy and drift_proxy without "
            "fixing path_ratio. F improved mean path_ratio diagnostically, but "
            "did not pass the 2/3 fold stability requirement."
        ),
    },
    "final_decision": {
        "no_mf1_variant_proceeds_to_final_test": True,
        "selected_candidate_for_next_stage": None,
        "s5_remains_final_clean_candidate": True,
        "do_not_continue_mf1c": True,
    },
    "selected_candidate": None,
    "no_final_test": True,
    "s5_remains_final": True,
    "caveats": [
        "train-CV only",
        "no final test",
        "proxy trajectory is not official final test",
        "small sequence protocol",
        "S5 unchanged",
        "no practical-ready claim",
        "F tmag-head is diagnostic only and did not pass stability gate",
    ],
}

if summary["s5_locked_metrics"] != S5_LOCKED:
    raise AssertionError("S5 locked metrics changed")
if summary["selected_candidate"] is not None:
    raise AssertionError("MF1 closeout must not select a candidate")
if not summary["no_final_test"]:
    raise AssertionError("MF1 closeout must record no final test")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

mf1a_a = mf1a_metrics["A_s5_chain_reference"]
mf1a_d = mf1a_metrics["D_gru_chain_refiner"]

table_lines = [
    "| variant | ATE_proxy | drift_proxy | path_ratio | rot_mean | tdir_mean | gate_status |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for row in mf1b_rows:
    table_lines.append(
        "| {variant} | {ATE_proxy} | {drift_proxy} | {path_ratio} | {rot_mean} | {tdir_mean} | {gate_status} |".format(
            variant=row["variant"],
            ATE_proxy=fmt(row["ATE_proxy"]),
            drift_proxy=fmt(row["drift_proxy"]),
            path_ratio=fmt(row["path_ratio"]),
            rot_mean=fmt(row["rot_mean"]),
            tdir_mean=fmt(row["tdir_mean"]),
            gate_status=row["gate_status"],
        )
    )

md = [
    "# MF1 Final Closeout Summary",
    "",
    "## Scope",
    "",
    "MF1 explored multi-frame / chain-level refinement after JRT1 failed to transfer pair-level gains to trajectory-level gains. MF1 is a new algorithm experiment, and it did not produce a final clean candidate. S5 remains final clean candidate.",
    "",
    "## Motivation",
    "",
    "JRT1 showed component gains but no trajectory proxy gains. MF1 therefore moved from pair-level residual correction to chain-level temporal refinement, with losses over pair components and local chain accumulation.",
    "",
    "## MF1a Summary",
    "",
    f"classification = `{mf1a['classification']}`",
    "",
    "The chain dataset and train harness worked. GRU/JRT-style smoke variants improved ATE_proxy, drift_proxy, rotation, and tdir relative to the S5 chain reference, while path_ratio remained around 1.445.",
    "",
    "| variant | ATE_proxy | drift_proxy | path_ratio | rot_mean | tdir_mean |",
    "|---|---:|---:|---:|---:|---:|",
    f"| A_s5_chain_reference | {fmt(mf1a_a['ATE_proxy'])} | {fmt(mf1a_a['drift_proxy'])} | {fmt(mf1a_a['path_ratio_proxy'])} | {fmt(mf1a_a['rot_mean_deg'])} | {fmt(mf1a_a['tdir_mean_deg'])} |",
    f"| D_gru_chain_refiner | {fmt(mf1a_d['ATE_proxy'])} | {fmt(mf1a_d['drift_proxy'])} | {fmt(mf1a_d['path_ratio_proxy'])} | {fmt(mf1a_d['rot_mean_deg'])} | {fmt(mf1a_d['tdir_mean_deg'])} |",
    "",
    "## MF1b Summary",
    "",
    f"classification = `{mf1b['final_classification']}`",
    "",
    "MF1b used 3-fold train-CV on the train split only. No test GT was used, and no final test was run.",
    "",
    *table_lines,
    "",
    "## Final Decision",
    "",
    "No MF1 variant proceeds to final test. `selected_candidate_for_next_stage = null`. S5 remains final clean candidate.",
    "",
    "## Final Classification",
    "",
    "`NO_STABLE_MF1_CHAIN_GAIN`",
    "",
    "## Thesis / Report Usage",
    "",
    "MF1 is a negative but informative result. It shows that chain-level temporal refinement provides stronger trajectory-shape signal than JRT1. However, scale/path-ratio stability remains unresolved and train-CV stability is insufficient. This supports the final limitation that practical improvement likely requires a more principled multi-frame geometric architecture and scale-aware design, not just a lightweight chain refiner.",
    "",
    "## Caveats",
    "",
    "- train-CV only",
    "- no final test",
    "- proxy trajectory is not official final test",
    "- small sequence protocol",
    "- S5 unchanged",
    "- no practical-ready claim",
    "- F tmag-head is diagnostic only and did not pass stability gate",
]
OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

print(OUT_JSON.relative_to(ROOT))
print(OUT_MD.relative_to(ROOT))
PY

echo "[mf1-closeout-summary] PASS"
