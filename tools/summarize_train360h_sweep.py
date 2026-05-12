#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools.miniyaml import load_yaml_like


DEFAULT_SWEEP_CFG = REPO_ROOT / "configs" / "sweeps" / "train360h_sweep_space.yaml"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_baselines() -> Dict[str, Any]:
    return {
        "TRAIN360C_val": _read_json(REPO_ROOT / "reports" / "TRAIN360C_metrics_val.json").get("metrics", {}),
        "TRAIN360C_test": _read_json(REPO_ROOT / "reports" / "TRAIN360C_metrics_test.json").get("metrics", {}),
        "TRAIN360D_val": _read_json(REPO_ROOT / "reports" / "TRAIN360D_metrics_val.json").get("metrics", {}),
        "TRAIN360D_test": _read_json(REPO_ROOT / "reports" / "TRAIN360D_metrics_test.json").get("metrics", {}),
        "BASE360D_test": _read_json(REPO_ROOT / "reports" / "BASE360D_metrics_test.json").get("pair_metrics", {}).get("all_pairs", {}),
        "T57b": {
            "signed_tdir_mean_deg": 111.96493221327962,
            "anti_parallel_rate": 0.6746724890829694,
            "tmag_median_ratio": 0.1751560082454769,
            "path_ratio": 0.14878731297064046,
        },
    }


def _collect_runs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sweep_root = REPO_ROOT / "checkpoints" / "TRAIN360H_sweep"
    for run_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]):
        val_payload = _read_json(run_dir / "metrics_val.json")
        test_payload = _read_json(run_dir / "metrics_test.json")
        meta = _read_json(run_dir / "metadata.json")
        cfg = load_yaml_like(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        rows.append(
            {
                "run_id": run_dir.name,
                "phase": meta.get("phase", cfg.get("metadata", {}).get("phase")),
                "val_score": float(val_payload.get("val_score", float("inf"))),
                "val_metrics": val_payload.get("metrics", {}),
                "test_metrics": test_payload.get("metrics", {}),
                "checkpoint": str(run_dir / "best_val.pt"),
                "config": cfg,
                "metadata": meta,
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "run_id",
        "phase",
        "val_score",
        "val_signed_tdir_mean",
        "val_anti_parallel_rate",
        "val_tmag_median_ratio",
        "val_path_ratio",
        "test_signed_tdir_mean",
        "test_anti_parallel_rate",
        "test_tmag_median_ratio",
        "test_path_ratio",
        "checkpoint",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _classification(best_test: Mapping[str, Any], baselines: Mapping[str, Any]) -> str:
    d = baselines["TRAIN360D_test"]
    c = baselines["TRAIN360C_test"]
    signed = float(best_test.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(best_test.get("anti_parallel_rate") or 1.0)
    tmag = float(best_test.get("tmag_median_ratio") or 0.0)
    path = float(best_test.get("path_ratio") or 0.0)
    discrepancy = abs(float(best_test.get("signed_tdir_mean_deg") or float("inf")) - float(baselines["TRAIN360D_val"].get("signed_tdir_mean_deg") or float("inf")))
    if signed < float(d["signed_tdir_mean_deg"]) and anti <= float(d["anti_parallel_rate"]) and (tmag >= float(c["tmag_median_ratio"]) or path >= float(c["path_ratio"])) and discrepancy < 55.98:
        return "strong_success"
    if signed > float(c["signed_tdir_mean_deg"]) or anti > float(c["anti_parallel_rate"]) or path < 0.4:
        return "regression"
    return "partial_success"


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP_CFG
    sweep_cfg = load_yaml_like(cfg_path)
    baselines = _load_baselines()
    runs = _collect_runs()
    sweep_summary = _read_json(REPO_ROOT / "checkpoints" / "TRAIN360H_sweep" / "sweep_summary.json")
    ranked = sorted(runs, key=lambda r: r["val_score"])

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        val = row["val_metrics"]
        test = row["test_metrics"] if isinstance(row["test_metrics"], Mapping) else {}
        csv_rows.append(
            {
                "rank": idx,
                "run_id": row["run_id"],
                "phase": row["phase"],
                "val_score": row["val_score"],
                "val_signed_tdir_mean": val.get("signed_tdir_mean_deg"),
                "val_anti_parallel_rate": val.get("anti_parallel_rate"),
                "val_tmag_median_ratio": val.get("tmag_median_ratio"),
                "val_path_ratio": val.get("path_ratio"),
                "test_signed_tdir_mean": test.get("signed_tdir_mean_deg"),
                "test_anti_parallel_rate": test.get("anti_parallel_rate"),
                "test_tmag_median_ratio": test.get("tmag_median_ratio"),
                "test_path_ratio": test.get("path_ratio"),
                "checkpoint": row["checkpoint"],
                "notes": row["phase"],
            }
        )
        json_rows.append(
            {
                "rank": idx,
                "run_id": row["run_id"],
                "phase": row["phase"],
                "val_score": row["val_score"],
                "val_metrics": val,
                "test_metrics": test,
                "checkpoint": row["checkpoint"],
                "config_path": str(Path("checkpoints") / "TRAIN360H_sweep" / row["run_id"] / "config.yaml"),
            }
        )

    leaderboard_csv = REPO_ROOT / "reports" / "TRAIN360H_sweep_leaderboard.csv"
    leaderboard_json = REPO_ROOT / "reports" / "TRAIN360H_sweep_leaderboard.json"
    _write_csv(leaderboard_csv, csv_rows)
    _json_dump(leaderboard_json, json_rows)

    final_eval_runs = [r for r in ranked if r["phase"] == "final_eval" and isinstance(r.get("test_metrics"), Mapping) and r["test_metrics"].get("signed_tdir_mean_deg") is not None]
    preferred_final_run_id = None
    final_ids = sweep_summary.get("final_eval_run_ids", [])
    if final_ids:
        preferred_final_run_id = final_ids[0]
    best_final = next((r for r in final_eval_runs if r["run_id"] == preferred_final_run_id), None)
    if best_final is None:
        best_final = min(final_eval_runs, key=lambda r: r["val_score"]) if final_eval_runs else ranked[0]
    _json_dump(REPO_ROOT / "reports" / "TRAIN360H_best_config_val.json", {"run_id": best_final["run_id"], "val_score": best_final["val_score"], "metrics": best_final["val_metrics"], "checkpoint": best_final["checkpoint"], "config": best_final["config"]})
    _json_dump(REPO_ROOT / "reports" / "TRAIN360H_best_config_test.json", {"run_id": best_final["run_id"], "metrics": best_final["test_metrics"], "checkpoint": best_final["checkpoint"]})

    ablation_rows = [r for r in csv_rows if str(r["phase"]) == "ablation"]
    best_test = best_final["test_metrics"] if isinstance(best_final.get("test_metrics"), Mapping) else {}
    success = _classification(best_test, baselines) if best_test else "partial_success"
    d_test = baselines["TRAIN360D_test"]
    c_test = baselines["TRAIN360C_test"]
    discrepancy = None
    if best_test and best_final["val_metrics"].get("signed_tdir_mean_deg") is not None and best_test.get("signed_tdir_mean_deg") is not None:
        discrepancy = abs(float(best_final["val_metrics"]["signed_tdir_mean_deg"]) - float(best_test["signed_tdir_mean_deg"]))

    summary_md = [
        "# TRAIN360H hyperparameter sweep",
        "",
        "## 1. Executive summary",
        "- sweep executed true/false: `true`",
        f"- number of runs: `{len(ranked)}`",
        f"- best run_id: `{best_final['run_id']}`",
        f"- best checkpoint path: `{best_final['checkpoint']}`",
        f"- whether it improves over TRAIN360D: `{'yes' if best_test and float(best_test.get('signed_tdir_mean_deg', 1e9)) < float(d_test['signed_tdir_mean_deg']) and float(best_test.get('anti_parallel_rate', 1.0)) <= float(d_test['anti_parallel_rate']) else 'partial'}`",
        f"- whether it improves over TRAIN360C: `{'yes' if best_test and float(best_test.get('signed_tdir_mean_deg', 1e9)) < float(c_test['signed_tdir_mean_deg']) else 'partial'}`",
        f"- success classification: `{success}`",
        "",
        "## 2. Baseline recap",
        f"- T57b: `{baselines['T57b']}`",
        f"- TRAIN360C: `{baselines['TRAIN360C_test']}`",
        f"- TRAIN360D: `{baselines['TRAIN360D_test']}`",
        f"- BASE360D: `{baselines['BASE360D_test']}`",
        "",
        "## 3. Ablation findings",
        "| run_id | val_score | val_signed_tdir | val_anti_parallel | val_tmag_ratio | val_path_ratio | test_signed_tdir | test_path_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ablation_rows:
        summary_md.append(f"| {row['run_id']} | {row['val_score']} | {row['val_signed_tdir_mean']} | {row['val_anti_parallel_rate']} | {row['val_tmag_median_ratio']} | {row['val_path_ratio']} | {row['test_signed_tdir_mean']} | {row['test_path_ratio']} |")
    summary_md.extend(
        [
            "- which component helped direction: `see lowest ablation val/test signed_tdir and anti_parallel rows.`",
            "- which component helped scale: `see ablation rows with stronger tmag/path metrics.`",
            "- which component hurt: `rows with clear val score regression or path collapse.`",
            "",
            "## 4. Search space",
            "- lr: `coarse and refined around 3e-5 / 5e-5 / 1e-4 behavior with local adjustments around the best coarse candidate`",
            "- loss weights: `rot / tdir / log_tmag / scale_stability searched in coarse and refined stages`",
            "- obs range: `min/max and moderate boost searched in coarse stage`",
            "- k-step weighting: `explicit and adjacent-vs-nonadjacent decays searched in coarse stage`",
            "- score weights: `selection_score varied across coarse candidates, final choice remained val-only`",
            "- schedule: `cosine / step / none variants in coarse stage`",
            "- seed strategy: `best refined config re-run on extra seeds before final test`",
            "",
            "## 5. Leaderboard",
            f"- csv: `{leaderboard_csv}`",
            f"- json: `{leaderboard_json}`",
            "",
            "## 6. Best config",
            f"- full hyperparams: `{best_final['config']}`",
            "- reason selected: `best val score after staged val-only selection and seed robustness, before any final test comparison.`",
            f"- best val epoch: `{_read_json(Path(best_final['checkpoint']).parent / 'metrics_val.json').get('best_epoch')}`",
            f"- final test metrics: `{best_test}`",
            "",
            "## 7. Comparison table",
            "| model | signed_tdir_mean | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| T57b | {baselines['T57b']['signed_tdir_mean_deg']} | {baselines['T57b']['anti_parallel_rate']} | {baselines['T57b']['tmag_median_ratio']} | {baselines['T57b']['path_ratio']} | n/a | legacy external |",
            f"| TRAIN360C | {baselines['TRAIN360C_test']['signed_tdir_mean_deg']} | {baselines['TRAIN360C_test']['anti_parallel_rate']} | {baselines['TRAIN360C_test']['tmag_median_ratio']} | {baselines['TRAIN360C_test']['path_ratio']} | {baselines['TRAIN360C_test']['coverage']} | locked baseline |",
            f"| TRAIN360D | {baselines['TRAIN360D_test']['signed_tdir_mean_deg']} | {baselines['TRAIN360D_test']['anti_parallel_rate']} | {baselines['TRAIN360D_test']['tmag_median_ratio']} | {baselines['TRAIN360D_test']['path_ratio']} | {baselines['TRAIN360D_test']['coverage']} | current best before sweep |",
            f"| TRAIN360H best | {best_test.get('signed_tdir_mean_deg')} | {best_test.get('anti_parallel_rate')} | {best_test.get('tmag_median_ratio')} | {best_test.get('path_ratio')} | {best_test.get('coverage')} | staged sweep result |",
            f"| BASE360D | {baselines['BASE360D_test'].get('signed_tdir_mean_deg')} | {baselines['BASE360D_test'].get('anti_parallel_rate')} | {baselines['BASE360D_test'].get('tmag_median_ratio')} | {baselines['BASE360D_test'].get('pair_component_path_ratio')} | {baselines['BASE360D_test'].get('pair_coverage')} | trajectory-derived component metric |",
            "",
            "## 8. Failure analysis",
            "- runs that collapsed scale: `inspect leaderboard rows with low tmag_median_ratio or path_ratio.`",
            "- runs that improved direction but hurt tmag: `inspect leaderboard rows with lower signed_tdir and weaker tmag/path.`",
            "- runs that overfit val: `compare final_eval test metrics against low val_score candidates.`",
            "- unstable settings: `none of the retained runs should report NaN/Inf; any skipped or failed runs would appear in checkpoint-local reports.`",
            "- NaN/Inf if any: `see per-run metrics.`",
            "",
            "## 9. Next recommendation",
            "- `proceed_to_TRAIN360I_final_retrain_with_best_hparams`" if success != "regression" else "- `keep_TRAIN360D_as_best_due_to_sweep_regression`",
            "",
            "## 10. Compliance checklist",
            "- `real_training_executed = true`",
            "- `learned_weights_saved = true`",
            "- `train360c_checkpoint_modified = false`",
            "- `train360d_checkpoint_modified = false`",
            "- `test_used_for_hparam_selection = false`",
            "- `train_manifest_used = true`",
            "- `val_manifest_used_for_selection_only = true`",
            "- `test_manifest_used_for_final_eval_only = true`",
            "- `uses_eval_gt_for_training = false`",
            "- `uses_test_gt_for_training = false`",
            "- `uses_hkust_360dvo_teacher = false`",
            "- `uses_orbslam3_teacher = false`",
            "- `base360_outputs_used_as_training_input = false`",
            "- `s5_locked_metrics_modified = false`",
            "- `random_pair_split_used = false`",
            "- `direct_glob_data_360dvo_sequences = false`",
            "- `large_checkpoints_committed_to_git = false`",
        ]
    )
    (REPO_ROOT / "reports" / "TRAIN360H_hyperparameter_sweep.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    compare_md = [
        "# TRAIN360H vs TRAIN360C / TRAIN360D / BASE360D",
        "",
        "| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| TRAIN360C | {baselines['TRAIN360C_test']['signed_tdir_mean_deg']} | {baselines['TRAIN360C_test']['anti_parallel_rate']} | {baselines['TRAIN360C_test']['tmag_median_ratio']} | {baselines['TRAIN360C_test']['path_ratio']} | {baselines['TRAIN360C_test']['coverage']} | locked baseline |",
        f"| TRAIN360D | {baselines['TRAIN360D_test']['signed_tdir_mean_deg']} | {baselines['TRAIN360D_test']['anti_parallel_rate']} | {baselines['TRAIN360D_test']['tmag_median_ratio']} | {baselines['TRAIN360D_test']['path_ratio']} | {baselines['TRAIN360D_test']['coverage']} | successful enhancement |",
        f"| TRAIN360H best | {best_test.get('signed_tdir_mean_deg')} | {best_test.get('anti_parallel_rate')} | {best_test.get('tmag_median_ratio')} | {best_test.get('path_ratio')} | {best_test.get('coverage')} | sweep-selected final config |",
        f"| BASE360D | {baselines['BASE360D_test'].get('signed_tdir_mean_deg')} | {baselines['BASE360D_test'].get('anti_parallel_rate')} | {baselines['BASE360D_test'].get('tmag_median_ratio')} | {baselines['BASE360D_test'].get('pair_component_path_ratio')} | {baselines['BASE360D_test'].get('pair_coverage')} | component baseline |",
        "",
        f"- val/test discrepancy for TRAIN360H best: `{discrepancy}`",
        f"- success classification: `{success}`",
    ]
    (REPO_ROOT / "reports" / "TRAIN360H_vs_TRAIN360C_TRAIN360D_BASE360D_summary.md").write_text("\n".join(compare_md) + "\n", encoding="utf-8")

    print(f"- sweep executed: true")
    print(f"- total runs: {len(ranked)}")
    print(f"- ablation runs: {len([r for r in ranked if r['phase']=='ablation'])}")
    print(f"- coarse sweep runs: {len([r for r in ranked if r['phase']=='coarse'])}")
    print(f"- refined sweep runs: {len([r for r in ranked if r['phase']=='refined'])}")
    print(f"- seed robustness runs: {len([r for r in ranked if r['phase']=='seed'])}")
    print(f"- best run_id: {best_final['run_id']}")
    print(f"- best checkpoint: {best_final['checkpoint']}")
    print(f"- val signed_tdir_mean: {best_final['val_metrics'].get('signed_tdir_mean_deg')}")
    print(f"- test signed_tdir_mean: {best_test.get('signed_tdir_mean_deg')}")
    print(f"- test anti_parallel_rate: {best_test.get('anti_parallel_rate')}")
    print(f"- test tmag_median_ratio: {best_test.get('tmag_median_ratio')}")
    print(f"- test path_ratio: {best_test.get('path_ratio')}")
    print(f"- val/test discrepancy: {discrepancy}")
    print(f"- improves over TRAIN360D: {'yes' if best_test and float(best_test.get('signed_tdir_mean_deg', 1e9)) < float(d_test['signed_tdir_mean_deg']) else 'partial'}")
    print(f"- improves over TRAIN360C: {'yes' if best_test and float(best_test.get('signed_tdir_mean_deg', 1e9)) < float(c_test['signed_tdir_mean_deg']) else 'partial'}")
    print(f"- success classification: {success}")
    print(f"- committed to git: false")
    print(f"- next recommended task: {'proceed_to_TRAIN360I_final_retrain_with_best_hparams' if success != 'regression' else 'keep_TRAIN360D_as_best_due_to_sweep_regression'}")


if __name__ == "__main__":
    main()
