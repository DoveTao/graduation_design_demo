#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/dovetao/miniconda3/envs/pytorch/bin/python")
BASE_CKPT = "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}

REPORT_PATH = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s11_tmag_scale_consistency_summary.md"
TMP_DATA_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s11_train_cv_data"
RUN_ROOT = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_runs"
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
FOLD_SEEDS: Tuple[Tuple[str, int], ...] = (("heldout_scene01_seq01", 0), ("heldout_scene01_seq02", 3))
SAFE_PATH_RANGE = (0.90, 0.97)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    family: str
    overrides: Tuple[str, ...]
    simplicity_rank: Tuple[float, float, float]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _mean(items: Sequence[float]) -> float:
    vals = [float(x) for x in items if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    s5_ok = all(abs(float(s5["metrics"][k]) - float(S5_LOCKED[k])) <= 1.0e-9 for k in ("drift", "ATE", "path_ratio"))
    passed = bool(contract["current_architecture_14_key_path_allowed_for_s8_baseline_gate"]) and s5_ok
    return {
        "passed": passed,
        "contract_path": str(S8B_CONTRACT_PATH),
        "s5_locked_metrics_ok": s5_ok,
        "s5_metrics": s5["metrics"],
    }


def _candidate_configs() -> List[CandidateConfig]:
    common = (
        "train_tmag_head_only=True",
        "use_coupled_pose_residual_head=False",
        "fine_rot_fuse_strength=0.45",
        "fine_tdir_fuse_strength=0.0",
        "fine_tmag_fuse_strength=0.0",
        "use_geometry_refine=False",
        "strict_load_checkpoint=False",
        f"init_checkpoint={BASE_CKPT}",
        "use_translation_magnitude_head=True",
        "tmag_loss_type=log_smooth_l1",
        "w_tmag=0.10",
        "tmag_start_updates=0",
        "tmag_ramp_updates=0",
        "save_best_odom_checkpoint=False",
        "save_best_smallk_odom_checkpoint=False",
        "save_metric_checkpoints=False",
        "save_best_joint_checkpoint=False",
        "save_best_local_joint_checkpoint=False",
        "save_last_eval_checkpoint=False",
        "save_last_train_state=False",
        "save_vis_examples=False",
        "save_vis_payload_npz=False",
        "save_vis_diag_json=False",
        "num_workers=0",
        "pin_memory=False",
        "persistent_workers=False",
        "amp=False",
        "batch_size=1",
        "grad_accum=1",
        "log_every=10",
    )
    return [
        CandidateConfig(
            name="A_tmag_head_only_baseline",
            family="baseline_tmag_head_only",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_affine_calib=False",
                "use_tmag_speed_loss=False",
                "use_tmag_ratio_loss=False",
                "use_tmag_chain_sum_loss=False",
                "use_tmag_regime_reweight=False",
            ),
            simplicity_rank=(1.0, 0.0, 0.0),
        ),
        CandidateConfig(
            name="B_dt_aware_affine_calib",
            family="dt_aware_shallow_calibrator",
            overrides=common + (
                "tmag_condition_on_dt=True",
                "use_tmag_affine_calib=True",
                "tmag_affine_init_scale=1.0",
                "tmag_affine_init_bias=0.0",
                "use_tmag_speed_loss=False",
                "use_tmag_ratio_loss=False",
                "use_tmag_chain_sum_loss=False",
                "use_tmag_regime_reweight=False",
            ),
            simplicity_rank=(2.0, 1.0, 0.0),
        ),
        CandidateConfig(
            name="C_consistency_losses",
            family="loss_only_consistency",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_affine_calib=False",
                "use_tmag_speed_loss=True",
                "w_tmag_speed=0.05",
                "use_tmag_ratio_loss=True",
                "w_tmag_ratio=0.05",
                "use_tmag_chain_sum_loss=True",
                "w_tmag_chain_sum=0.05",
                "use_tmag_regime_reweight=False",
            ),
            simplicity_rank=(3.0, 1.0, 1.0),
        ),
        CandidateConfig(
            name="D_consistency_high_regime_weighted",
            family="high_regime_weighted_consistency",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_affine_calib=False",
                "use_tmag_speed_loss=True",
                "w_tmag_speed=0.05",
                "use_tmag_ratio_loss=True",
                "w_tmag_ratio=0.05",
                "use_tmag_chain_sum_loss=True",
                "w_tmag_chain_sum=0.05",
                "use_tmag_regime_reweight=True",
                "tmag_regime_weight_pred_high=1.35",
                "tmag_regime_weight_gt_high=1.35",
                "tmag_regime_weight_dt_ge_1=1.20",
                "tmag_regime_weight_k20=1.20",
                "tmag_loss_gt_weight_alpha=0.35",
            ),
            simplicity_rank=(4.0, 1.0, 1.0),
        ),
    ]


def _ensure_subset_data_root() -> Path:
    pano_root = REPO_ROOT / "data" / "PanoramaView"
    if not pano_root.is_dir():
        raise FileNotFoundError(f"PanoramaView root not found: {pano_root}")
    if TMP_DATA_ROOT.exists():
        shutil.rmtree(TMP_DATA_ROOT)
    for scene, seq in TRAIN_GROUPS:
        src = pano_root / scene / seq
        if not src.exists():
            raise FileNotFoundError(f"Missing source seq dir: {src}")
        dst = TMP_DATA_ROOT / "PanoramaView" / scene / seq
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)
    return TMP_DATA_ROOT


def _parse_init_counts(stdout: str) -> Tuple[int, int]:
    pat = re.compile(r"\[InitCkpt\].*missing=(\d+)\s+\|\s+unexpected=(\d+)")
    hits = pat.findall(stdout)
    if not hits:
        return -1, -1
    a, b = hits[-1]
    return int(a), int(b)


def _latest_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    return dict(summary.get("last_eval", {}))


def _run_train(candidate: CandidateConfig, fold_name: str, split_seed: int, data_root: Path, max_steps: int, *, smoke: bool) -> Dict[str, Any]:
    exp_name = f"S11_{candidate.name}_{fold_name}"
    run_dir = RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", "ckpt_dir=checkpoints/S11_tmag_scale_consistency_runs",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", f"max_steps={max_steps}",
        "--set", f"eval_every={max_steps}",
        "--set", f"max_eval_batches={8 if smoke else 0}",
        "--set", f"max_train_eval_batches={8 if smoke else 0}",
        "--set", f"use_odometry_eval={'False' if smoke else 'True'}",
    ]
    for item in candidate.overrides:
        cmd.extend(["--set", item])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"S11 training failed for {exp_name}; see {run_dir / 'train_stdout.log'}")
    summary_path = run_dir / "final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final_summary.json for {exp_name}")
    summary = _read_json(summary_path)
    init_missing, init_unexpected = _parse_init_counts(proc.stdout)
    metrics = _latest_eval(summary)
    return {
        "candidate": candidate.name,
        "fold": fold_name,
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "init_missing": init_missing,
        "init_unexpected": init_unexpected,
        "trainable_names": summary.get("trainable_names", []),
        "metrics": metrics,
        "path_ratio": float(metrics.get("odom_shape_metric_mean_path_length_ratio", float("nan"))),
        "ATE": float(metrics.get("odom_metric_ATE", float("nan"))),
        "drift": float(metrics.get("odom_metric_drift", float("nan"))),
        "RPE_rot": float(metrics.get("odom_metric_RPE_rot", float("nan"))),
        "RPE_trans_dir": float(metrics.get("odom_metric_RPE_trans_dir", float("nan"))),
        "RPE_trans_mag": float(metrics.get("odom_metric_RPE_trans_mag", float("nan"))),
        "tmag_rel_err": float(metrics.get("tmag_rel_err", float("nan"))),
    }


def _candidate_gate(candidate_rows: Sequence[Dict[str, Any]], baseline_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    cand_ate = _mean([float(r["ATE"]) for r in candidate_rows])
    cand_drift = _mean([float(r["drift"]) for r in candidate_rows])
    cand_path = _mean([float(r["path_ratio"]) for r in candidate_rows])
    base_ate = _mean([float(r["ATE"]) for r in baseline_rows])
    base_drift = _mean([float(r["drift"]) for r in baseline_rows])
    init_ok = all(int(r["init_unexpected"]) in (-1, 0) for r in candidate_rows)
    path_ok = SAFE_PATH_RANGE[0] <= cand_path <= SAFE_PATH_RANGE[1]
    ok = (
        math.isfinite(cand_ate)
        and math.isfinite(cand_drift)
        and math.isfinite(cand_path)
        and path_ok
        and cand_ate <= base_ate + 1e-9
        and cand_drift <= base_drift + 1.0e-3
        and init_ok
    )
    return ok, {
        "cv_mean_ATE": cand_ate,
        "cv_mean_drift": cand_drift,
        "cv_mean_path_ratio": cand_path,
        "baseline_cv_mean_ATE": base_ate,
        "baseline_cv_mean_drift": base_drift,
        "path_ok": path_ok,
        "init_ok": init_ok,
    }


def _select_candidate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    gated = [r for r in records if bool(r["satisfies_clean_gate"])]
    if gated:
        gated.sort(key=lambda r: (float(r["cv_mean_ATE"]), float(r["cv_mean_drift"]), abs(float(r["cv_mean_path_ratio"]) - S5_LOCKED["path_ratio"]), r["simplicity_rank"]))
        return gated[0]
    diagnostic = [r for r in records if float(r["cv_mean_ATE"]) < float(r["baseline_cv_mean_ATE"]) and SAFE_PATH_RANGE[0] <= float(r["cv_mean_path_ratio"]) <= SAFE_PATH_RANGE[1]]
    if diagnostic:
        diagnostic.sort(key=lambda r: (float(r["cv_mean_ATE"]), float(r["cv_mean_drift"]), r["simplicity_rank"]))
        return diagnostic[0]
    return None


def _final_classification(baseline_gate_ok: bool, selected: Dict[str, Any] | None, final_test: Dict[str, Any] | None) -> Tuple[str, bool]:
    if not baseline_gate_ok:
        return "REPRODUCTION-MISMATCH", False
    if selected is None:
        return "NO-STABLE-TMAG-CONSISTENCY-GAIN", False
    if final_test is None:
        if bool(selected["satisfies_clean_gate"]):
            return "INCONCLUSIVE", False
        if float(selected["cv_mean_ATE"]) < float(selected["baseline_cv_mean_ATE"]):
            return "TMAG-CONSISTENCY-DIAGNOSTIC-GAIN", False
        return "NO-STABLE-TMAG-CONSISTENCY-GAIN", False
    replacement = (
        float(final_test["ATE"]) < S5_LOCKED["ATE"]
        and float(final_test["drift"]) <= S5_LOCKED["drift"] + 1e-3
        and SAFE_PATH_RANGE[0] <= float(final_test["path_ratio"]) <= SAFE_PATH_RANGE[1]
    )
    if replacement:
        return "TMAG-CONSISTENCY-CLEAN-GAIN", True
    return "TMAG-CONSISTENCY-CLEAN-BUT-MARGINAL", False


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S11 Tmag Scale Consistency Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S11 replaces S5: `{payload['s11_replaces_s5']}`\n")
    lines.append(f"- selected candidate: `{payload['selected']['name'] if payload.get('selected') else 'None'}`\n")
    lines.append(f"- final test run: `{payload.get('final_test') is not None}`\n\n")
    lines.append("## Motivation from S4/S5/S8/S9/S10\n\n")
    lines.append("- S4/S5 showed that magnitude regime dominates much of the current error structure.\n")
    lines.append("- S8/S9/S10 indicated that post-processing, routing, and chain smoothing provide limited clean improvement space.\n")
    lines.append("- S11 therefore shifts the optimization target back into training-time tmag scale learning.\n\n")
    lines.append("## Why training-time tmag consistency instead of post-processing\n\n")
    lines.append("- S11 keeps the R / tdir path unchanged and only adjusts tmag-related supervision or calibration inside the model.\n")
    lines.append("- gt_tmag is used only as a training label, never as an inference-time feature.\n\n")
    lines.append("## Candidate loss definitions\n\n")
    lines.append("- `loss_tmag_log`: existing log-space Huber supervision.\n")
    lines.append("- `loss_speed`: Huber on `log(pred_tmag/dt) - log(gt_tmag/dt)`.\n")
    lines.append("- `loss_ratio`: Huber on adjacent-pair log-ratio consistency inside train triplets.\n")
    lines.append("- `loss_chain_sum`: Huber on short-chain log-sum consistency proxy.\n")
    lines.append("- high-regime reweighting is train-only and uses pred/gt/dt/k labels without leaking them into inference.\n\n")
    lines.append("## Training protocol\n\n")
    lines.append(f"- mode: `{payload['mode']}`\n")
    lines.append(f"- max_steps per fold: `{payload['max_steps']}`\n")
    lines.append("- train-CV split uses `scene01/seq01` and `scene01/seq02` only for launch-stage selection.\n")
    lines.append("- all candidates initialize from the coarse-to-fine base checkpoint and freeze non-tmag parameters.\n\n")
    if payload["mode"] == "smoke":
        lines.append("Smoke note: this launch-stage run disables odometry evaluation to validate training/loss plumbing first, so ATE/drift/path_ratio are expected to remain unavailable (`nan`).\n\n")
    lines.append("## Train-CV selection\n\n")
    lines.append("| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path_ratio | clean_gate |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |\n")
    for row in payload["candidates"]:
        lines.append(
            f"| {row['name']} | {row['family']} | {_fmt(row['cv_mean_ATE'])} | {_fmt(row['cv_mean_drift'])} | {_fmt(row['cv_mean_path_ratio'])} | {row['satisfies_clean_gate']} |\n"
        )
    lines.append("\n")
    lines.append("## Hard-gate audit\n\n")
    for row in payload["candidates"]:
        gp = row["gate_payload"]
        lines.append(
            f"- `{row['name']}`: mean_ATE `{_fmt(gp['cv_mean_ATE'])}` vs baseline `{_fmt(gp['baseline_cv_mean_ATE'])}`, "
            f"mean_drift `{_fmt(gp['cv_mean_drift'])}` vs baseline `{_fmt(gp['baseline_cv_mean_drift'])}`, "
            f"path_ratio `{_fmt(gp['cv_mean_path_ratio'])}`, path_ok=`{gp['path_ok']}`, init_ok=`{gp['init_ok']}`, gate=`{row['satisfies_clean_gate']}`\n"
        )
    lines.append("\n")
    lines.append("## Final test result, if selected\n\n")
    if payload.get("final_test") is None:
        lines.append("- not run in this launch stage.\n\n")
    else:
        ft = payload["final_test"]
        lines.append(f"- ATE = `{_fmt(ft['ATE'])}`\n")
        lines.append(f"- drift = `{_fmt(ft['drift'])}`\n")
        lines.append(f"- path_ratio = `{_fmt(ft['path_ratio'])}`\n\n")
    lines.append("## Compare against S5\n\n")
    lines.append(f"- locked S5 drift / ATE / path_ratio = `{S5_LOCKED['drift']}` / `{S5_LOCKED['ATE']}` / `{S5_LOCKED['path_ratio']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s11_replaces_s5']}`\n\n")
    lines.append("## Leakage audit\n\n")
    for k, v in payload["leakage_audit"].items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")
    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Final S11 Tmag Scale Consistency Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S11 replaces S5: `{payload['s11_replaces_s5']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s11_replaces_s5']}`\n")
    lines.append(f"- Selected candidate: `{payload['selected']['name'] if payload.get('selected') else 'None'}`\n\n")
    lines.append("S11 moves optimization back into training-time tmag scale learning. ")
    lines.append("It freezes non-tmag parameters and compares baseline tmag-head tuning against consistency-loss and high-regime weighted variants.\n\n")
    if payload["mode"] == "smoke":
        lines.append("This first launch only validates the S11 training path and report generation. Odometry clean-CV metrics are intentionally deferred to the next heavier run, so the current result is strictly a smoke-stage `INCONCLUSIVE`.\n\n")
    if payload.get("final_test") is None:
        lines.append("This launch did not promote a new final clean candidate yet, so S5 remains the locked final candidate while S11 continues as an internal training-direction probe.\n")
    else:
        ft = payload["final_test"]
        lines.append(
            f"The selected S11 candidate achieved drift/ATE/path_ratio = `{_fmt(ft['drift'])}` / `{_fmt(ft['ATE'])}` / `{_fmt(ft['path_ratio'])}`.\n"
        )
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run(mode: str, max_steps: int, run_final_test: bool) -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "mode": mode,
        "max_steps": int(max_steps),
        "baseline_gate": baseline_gate,
        "leakage_audit": {
            "passed": bool(baseline_gate["passed"]),
            "test_used_for_selection": False,
            "gt_tmag_as_inference_feature": False,
            "gt_pose_as_inference_feature": False,
            "post_processing_router_used": False,
            "train_cv_selection_only": True,
        },
        "candidates": [],
        "selected": None,
        "final_test": None,
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "s11_replaces_s5": False,
    }
    if not baseline_gate["passed"]:
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload

    data_root = _ensure_subset_data_root()
    smoke = mode == "smoke"
    candidate_list = _candidate_configs() if not smoke else _candidate_configs()[:2]
    fold_list = FOLD_SEEDS if not smoke else FOLD_SEEDS[:1]
    per_candidate_rows: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidate_list:
        per_candidate_rows[cand.name] = []
        for fold_name, seed in fold_list:
            per_candidate_rows[cand.name].append(_run_train(cand, fold_name, seed, data_root, max_steps, smoke=smoke))

    baseline_rows = per_candidate_rows["A_tmag_head_only_baseline"]
    candidates = []
    for cand in candidate_list:
        rows = per_candidate_rows[cand.name]
        gate_ok, gate_payload = _candidate_gate(rows, baseline_rows)
        candidates.append(
            {
                "name": cand.name,
                "family": cand.family,
                "simplicity_rank": cand.simplicity_rank,
                "overrides": list(cand.overrides),
                "fold_rows": rows,
                "cv_mean_ATE": _mean([float(r["ATE"]) for r in rows]),
                "cv_mean_drift": _mean([float(r["drift"]) for r in rows]),
                "cv_mean_path_ratio": _mean([float(r["path_ratio"]) for r in rows]),
                "baseline_cv_mean_ATE": _mean([float(r["ATE"]) for r in baseline_rows]),
                "baseline_cv_mean_drift": _mean([float(r["drift"]) for r in baseline_rows]),
                "satisfies_clean_gate": gate_ok,
                "gate_payload": gate_payload,
            }
        )
    candidates.sort(key=lambda r: (float(r["cv_mean_ATE"]), float(r["cv_mean_drift"]), r["simplicity_rank"]))
    selected = _select_candidate(candidates)
    payload["candidates"] = candidates
    payload["selected"] = selected

    if run_final_test and (not smoke) and selected is not None and bool(selected["satisfies_clean_gate"]):
        payload["final_test"] = {"skipped_in_launch_impl": True}

    final_classification, replaces = _final_classification(bool(baseline_gate["passed"]), selected, payload.get("final_test"))
    if smoke:
        final_classification = "INCONCLUSIVE"
        replaces = False
    payload["final_classification"] = final_classification
    payload["s11_replaces_s5"] = replaces
    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    _write_summary(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "cv"], default="smoke")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--run-final-test", action="store_true")
    args = ap.parse_args()
    payload = run(args.mode, args.max_steps, args.run_final_test)
    print(json.dumps({
        "final_classification": payload["final_classification"],
        "s11_replaces_s5": payload["s11_replaces_s5"],
        "selected_candidate": None if payload["selected"] is None else payload["selected"]["name"],
        "report_path": str(REPORT_PATH),
        "candidates_path": str(CANDIDATES_PATH),
        "summary_path": str(SUMMARY_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
