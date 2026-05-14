#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from model import PanoramaRelPoseModel
from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg
from tools.s15_trajectory_level_training_objective import (
    BASE_CKPT,
    FOLDS,
    MANIFEST_ROOT,
    PYTHON_BIN,
    RUN_ROOT,
    S5_POLICY_PATH,
    _baseline_gate,
    _build_eval_dataset,
    _build_window_manifest,
    _ensure_subset_data_root,
    _parse_init_counts,
    _proxy_metrics,
    _read_json,
    _safe_float,
    _write_json,
)
from tools.s15d_train_eval_forward_parity_fix import PredTmagShrinkModel


REPORT_PATH = REPO_ROOT / "checkpoints" / "S15e_tiny_trajectory_retest_after_harness_fix_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S15e_tiny_trajectory_retest_after_harness_fix_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15e_tiny_trajectory_retest_after_harness_fix_summary.md"
PRE_FIX_SMOKE = {
    "candidate": "D_full_light_traj",
    "val_ate_proxy": 15.708141,
    "val_path_proxy": 2.515539,
    "odometry_ATE": 19.286121,
    "odometry_drift": 24.674397,
}
UPDATES = 20
BATCH_SIZE = 1
WINDOW_SIZE = 3
FOLD_NAME, SPLIT_SEED = FOLDS[0]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    trainable_mode: str
    overrides: Tuple[str, ...]


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return f"{x:.{digits}f}" if math.isfinite(x) else "nan"


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _build_training_cfg() -> Tuple[Config, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    s5 = _read_json(S5_POLICY_PATH)
    cfg = Config()
    overrides = [
        f"init_checkpoint={REPO_ROOT / s5['base_checkpoint_path']}",
        f"dt_bucket_scale_anchor_policy_json={S5_POLICY_PATH}",
        "use_fine_stage=True",
        f"fine_rot_fuse_strength={float(s5['fine_rot_fuse_strength'])}",
        f"fine_tdir_fuse_strength={float(s5['fine_tdir_fuse_strength'])}",
        f"fine_tmag_fuse_strength={float(s5['fine_tmag_fuse_strength'])}",
        "use_geometry_refine=False",
        "use_translation_magnitude_head=True",
        "train_tmag_head_only=True",
        "train_fine_only=False",
        "use_coupled_pose_residual_head=False",
        "train_forward_eval_mode=False",
    ]
    from train_mvp import _apply_cfg_overrides, _apply_dt_bucket_scale_anchor_policy, _restore_cfg_from_policy_base_checkpoint

    _apply_cfg_overrides(cfg, overrides)
    policy_summary_before = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, overrides)
    policy_summary_after = _apply_dt_bucket_scale_anchor_policy(cfg)
    return cfg, policy_summary_before, restore_summary, policy_summary_after


def _build_wrapped_model(device: torch.device, ckpt_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    cfg, _before, restore_summary, policy_after = _build_training_cfg()
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    s5 = _read_json(S5_POLICY_PATH)
    wrapped = PredTmagShrinkModel(
        model,
        q90=float(s5["thresholds"]["q90_value"]),
        q95=float(s5["thresholds"]["q95_upper_tail_value"]),
        mid_scale=float(s5["scales"]["mid_scale"]),
        high_scale=float(s5["scales"]["high_scale"]),
    ).to(device)
    return wrapped, {
        "missing": len(msg.missing_keys),
        "unexpected": len(msg.unexpected_keys),
        "policy_wrapper_enabled": bool(policy_after.get("enabled", False)),
        "restore_enabled": bool(restore_summary.get("enabled", False)),
        "policy_lineage": list(policy_after.get("policy_lineage", [])),
        "dt_anchor_apply": bool(getattr(cfg, "dt_bucket_scale_anchor_apply", False)),
    }


def _predict_batch(model: torch.nn.Module, sample: Dict[str, Any], meta: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
    with torch.no_grad():
        R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    return {
        "R": R_pred[0].detach().cpu().numpy(),
        "tvec": aux["t_vec_out"][0].detach().cpu().numpy(),
        "tmag": float(aux["t_mag"].detach().float().view(-1)[0].cpu()),
    }


def _delta_audit(initial_ckpt: Path, final_ckpt: Path, ds_val, device: torch.device) -> Dict[str, Any]:
    init_model, init_meta = _build_wrapped_model(device, initial_ckpt)
    final_model, final_meta = _build_wrapped_model(device, final_ckpt)
    sample, meta = ds_val[0], ds_val.manifest()[0]
    before = _predict_batch(init_model, sample, meta, device)
    after = _predict_batch(final_model, sample, meta, device)
    return {
        "R_delta": float(np.linalg.norm(after["R"] - before["R"])),
        "tvec_delta": float(np.linalg.norm(after["tvec"] - before["tvec"])),
        "tmag_delta": abs(after["tmag"] - before["tmag"]),
        "initial_missing": int(init_meta["missing"]),
        "initial_unexpected": int(init_meta["unexpected"]),
        "final_missing": int(final_meta["missing"]),
        "final_unexpected": int(final_meta["unexpected"]),
        "policy_wrapper_enabled": bool(final_meta["policy_wrapper_enabled"]),
        "restore_enabled": bool(final_meta["restore_enabled"]),
        "dt_anchor_apply": bool(final_meta["dt_anchor_apply"]),
        "policy_lineage": list(final_meta["policy_lineage"]),
    }


def _candidate_configs() -> List[CandidateConfig]:
    ckpt_cfg = _load_ckpt_cfg(BASE_CKPT)
    s5_policy = _read_json(S5_POLICY_PATH)
    common = (
        "use_coupled_pose_residual_head=False",
        "strict_load_checkpoint=False",
        f"init_checkpoint={BASE_CKPT}",
        f"dt_bucket_scale_anchor_policy_json={S5_POLICY_PATH}",
        "use_fine_stage=True",
        f"fine_rot_fuse_strength={float(s5_policy['fine_rot_fuse_strength'])}",
        f"fine_tdir_fuse_strength={float(s5_policy['fine_tdir_fuse_strength'])}",
        f"fine_tmag_fuse_strength={float(s5_policy['fine_tmag_fuse_strength'])}",
        "use_geometry_refine=False",
        "use_translation_magnitude_head=True",
        f"tmag_head_mode={ckpt_cfg.get('tmag_head_mode', 'multiscale')}",
        f"tmag_multiscale_num_bins={ckpt_cfg.get('tmag_multiscale_num_bins', 4)}",
        f"tmag_multiscale_log_centers={ckpt_cfg.get('tmag_multiscale_log_centers', '-3.5,-1.7,-0.9,-0.3')}",
        f"tmag_multiscale_residual_scale={ckpt_cfg.get('tmag_multiscale_residual_scale', 1.0)}",
        f"use_tmag_global_bias={ckpt_cfg.get('use_tmag_global_bias', True)}",
        "use_tmag_affine_calib=False",
        "w_tmag=0.10",
        "tmag_loss_type=log_smooth_l1",
        "tmag_start_updates=0",
        "tmag_ramp_updates=0",
        "train_tmag_head_only=True",
        "train_fine_only=False",
        "train_forward_eval_mode=False",
        "train_fixed_pairs_return_triplet=True",
        "train_fixed_pairs_seq_turn_only_k=1",
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
        f"batch_size={BATCH_SIZE}",
        "grad_accum=1",
        "log_every=20",
        "eval_use_fixed_pairs=True",
        "use_odometry_eval=True",
        "eval_k_list=(1,2,3,5,10,20)",
        "odom_eval_prefer_k=1",
        "odom_eval_fallback_to_min_k=False",
        "traj_window_only_k=1",
        "traj_window_min_gt=0.02",
        "traj_scale_normalize=True",
        "traj_start_updates=0",
        "traj_ramp_updates=0",
        "use_tmag_speed_loss=False",
        "use_tmag_ratio_loss=False",
        "use_tmag_chain_sum_loss=False",
        "use_tmag_regime_reweight=False",
        "use_tdir_anchor_loss=False",
        "w_tdir_anchor=0.0",
        "use_seq_turn_loss=False",
        "use_seq_turn_chain_loss=False",
        "use_odom_chain_len_loss=False",
        "use_odom_chain_vec_loss=False",
    )
    return [
        CandidateConfig(
            name="A_pair_only_baseline_fixed_harness",
            trainable_mode="tmag_head_only",
            overrides=common + (
                "use_traj_ate_loss=False",
                "use_traj_drift_loss=False",
                "use_traj_path_loss=False",
                "use_traj_rot_loss=False",
                "use_traj_tdir_loss=False",
                "use_traj_tmag_step_loss=False",
            ),
        ),
        CandidateConfig(
            name="D_full_light_traj_fixed_harness",
            trainable_mode="tmag_head_only",
            overrides=common + (
                "use_traj_ate_loss=True",
                "w_traj_ate=0.05",
                "use_traj_drift_loss=True",
                "w_traj_drift=0.05",
                "use_traj_path_loss=True",
                "w_traj_path=0.05",
                "use_traj_rot_loss=True",
                "w_traj_rot=0.05",
                "use_traj_tdir_loss=True",
                "w_traj_tdir=0.05",
                "use_traj_tmag_step_loss=True",
                "w_traj_tmag_step=0.05",
            ),
        ),
    ]


def _run_train(candidate: CandidateConfig, data_root: Path, manifest_path: Path, max_steps: int) -> Dict[str, Any]:
    exp_name = f"S15e_{candidate.name}_{FOLD_NAME}_u{max_steps}"
    run_dir = RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", f"ckpt_dir={RUN_ROOT}",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={SPLIT_SEED}",
        "--set", f"max_steps={max_steps}",
        "--set", f"eval_every={max_steps}",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
        "--set", f"train_fixed_pairs_manifest_json={manifest_path}",
        "--set", "min_dt=0.0",
        "--set", "eval_min_dt=0.0",
    ]
    for item in candidate.overrides:
        cmd.extend(["--set", item])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"S15e training failed for {exp_name}; see {run_dir / 'train_stdout.log'}")
    summary = _read_json(run_dir / "final_summary.json")
    init_missing, init_unexpected = _parse_init_counts(proc.stdout)
    return {
        "candidate": candidate.name,
        "trainable_mode": candidate.trainable_mode,
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "final.pt"),
        "init_missing": int(init_missing),
        "init_unexpected": int(init_unexpected),
        "train_summary": summary,
    }


def _extract_result(candidate: CandidateConfig, run_payload: Dict[str, Any], data_root: Path, ds_val, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(run_payload["checkpoint_path"])
    summary = run_payload["train_summary"]
    last_eval = dict(summary.get("last_eval", {}))
    proxy = _proxy_metrics(ckpt_path, data_root, SPLIT_SEED)
    delta = _delta_audit(BASE_CKPT, ckpt_path, ds_val, device)
    first_train = dict(summary.get("first_train", {}))
    last_train = dict(summary.get("last_train", {}))
    return {
        "name": candidate.name,
        "trainable_mode": candidate.trainable_mode,
        "run_dir": run_payload["run_dir"],
        "checkpoint_path": run_payload["checkpoint_path"],
        "train_loss_snapshot": {
            "first_train": first_train,
            "last_train": last_train,
        },
        "val_ate_proxy": _safe_float(proxy.get("val_ate_proxy")),
        "val_drift_proxy": _safe_float(proxy.get("val_drift_proxy")),
        "val_path_proxy": _safe_float(proxy.get("val_path_proxy")),
        "val_rot_traj": _safe_float(proxy.get("val_rot_traj_deg")),
        "val_tdir_traj": _safe_float(proxy.get("val_tdir_traj_deg")),
        "val_tmag_step": _safe_float(proxy.get("val_tmag_step_log")),
        "odometry_ATE": _safe_float(last_eval.get("odom_metric_ATE")),
        "odometry_drift": _safe_float(last_eval.get("odom_metric_drift")),
        "odometry_path_ratio": _safe_float(last_eval.get("odom_shape_metric_mean_path_length_ratio")),
        "trainable_param_audit": {
            "trainable_param_count": int(summary.get("trainable_param_count", 0)),
            "frozen_param_count": int(summary.get("frozen_param_count", 0)),
            "optimizer_param_count": int(summary.get("optimizer_param_count", 0)),
            "trainable_groups": dict(summary.get("trainable_groups", {})),
            "trainable_names": list(summary.get("trainable_names", [])),
            "forbidden_trainable_params": list(summary.get("forbidden_trainable_params", [])),
        },
        "R_delta": _safe_float(delta["R_delta"]),
        "tvec_delta": _safe_float(delta["tvec_delta"]),
        "tmag_delta": _safe_float(delta["tmag_delta"]),
        "missing_unexpected": {
            "init_missing": int(run_payload["init_missing"]),
            "init_unexpected": int(run_payload["init_unexpected"]),
            "final_missing": int(delta["final_missing"]),
            "final_unexpected": int(delta["final_unexpected"]),
        },
        "policy_wrapper_enabled": bool(delta["policy_wrapper_enabled"]),
        "restore_enabled": bool(delta["restore_enabled"]),
        "dt_anchor_apply": bool(delta["dt_anchor_apply"]),
        "policy_lineage": list(delta["policy_lineage"]),
        "odometry_eval_ran": math.isfinite(_safe_float(last_eval.get("odom_metric_ATE"))),
    }


def _candidate_stable(row: Dict[str, Any]) -> bool:
    return (
        math.isfinite(float(row["val_ate_proxy"]))
        and math.isfinite(float(row["val_path_proxy"]))
        and math.isfinite(float(row["odometry_ATE"]))
        and math.isfinite(float(row["odometry_drift"]))
        and float(row["val_ate_proxy"]) < 10.0
        and float(row["val_path_proxy"]) < 1.5
        and float(row["odometry_ATE"]) < 12.0
        and float(row["odometry_drift"]) < 12.0
    )


def _trajectory_improved_vs_prefix(row: Dict[str, Any]) -> bool:
    return (
        math.isfinite(float(row["val_ate_proxy"]))
        and math.isfinite(float(row["val_path_proxy"]))
        and math.isfinite(float(row["odometry_ATE"]))
        and math.isfinite(float(row["odometry_drift"]))
        and float(row["val_ate_proxy"]) < float(PRE_FIX_SMOKE["val_ate_proxy"]) * 0.75
        and float(row["val_path_proxy"]) < float(PRE_FIX_SMOKE["val_path_proxy"]) * 0.75
        and float(row["odometry_ATE"]) < float(PRE_FIX_SMOKE["odometry_ATE"]) * 0.75
        and float(row["odometry_drift"]) < float(PRE_FIX_SMOKE["odometry_drift"]) * 0.75
    )


def _classify(rows: Sequence[Dict[str, Any]], baseline_gate: Dict[str, Any]) -> Tuple[str, str, bool]:
    if not bool(baseline_gate["passed"]):
        return "REPRODUCTION-MISMATCH", "baseline gate failed", False
    row_map = {row["name"]: row for row in rows}
    a_row = row_map["A_pair_only_baseline_fixed_harness"]
    d_row = row_map["D_full_light_traj_fixed_harness"]
    a_stable = _candidate_stable(a_row)
    d_stable = _candidate_stable(d_row)
    d_better = _trajectory_improved_vs_prefix(d_row)
    if not a_stable:
        return "TRAINING-STILL-UNSTABLE", "pair-only baseline still collapses after harness parity fix", False
    if d_stable and d_better:
        return "TRAJ-RETEST-STABILIZED", "pair-only is stable and trajectory retest no longer collapses versus pre-fix smoke", True
    if a_stable and not d_stable:
        return "TRAJ-OBJECTIVE-STILL-UNSTABLE", "pair-only is stable but trajectory objective still collapses", False
    return "INCONCLUSIVE", "pair-only stabilized but trajectory result is mixed without a strong clean improvement", False


def _write_report(payload: Dict[str, Any]) -> None:
    a_row = payload["results"][0]
    d_row = payload["results"][1]
    lines = [
        "# S15e Tiny Trajectory Retest After Harness Fix Report",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- main finding: `{payload['main_finding']}`",
        f"- candidates run: `{', '.join(payload['candidates_run'])}`",
        f"- training budget: `W={WINDOW_SIZE}, updates={UPDATES}, batch_size={BATCH_SIZE}`",
        f"- trainable mode used: `{payload['trainable_mode']}`",
        f"- S15f lightweight CV recommended: `{payload['recommend_s15f']}`",
        "- S5 remains final clean candidate: `yes`",
        "",
        "## Baseline context",
        f"- branch: `optimize/s15-trajectory-level-training-objective`",
        f"- S15d commit: `8aeeb84`",
        f"- S15d classification: `FORWARD-PARITY-FIX-PASS`",
        f"- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`",
        "",
        "## Tiny retest setup",
        f"- fold: `{FOLD_NAME}`",
        f"- split_seed: `{SPLIT_SEED}`",
        f"- manifest root: `{MANIFEST_ROOT}`",
        f"- policy wrapper expected: `{a_row['policy_wrapper_enabled'] and d_row['policy_wrapper_enabled']}`",
        f"- dt anchor apply: `{a_row['dt_anchor_apply'] and d_row['dt_anchor_apply']}`",
        "",
        "## Candidate results",
        "| candidate | val_ate_proxy | val_drift_proxy | val_path_proxy | val_rot_traj | val_tdir_traj | val_tmag_step | odom_ATE | odom_drift | odom_path_ratio | R_delta | tvec_delta | tmag_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {a_row['name']} | {_fmt(a_row['val_ate_proxy'])} | {_fmt(a_row['val_drift_proxy'])} | {_fmt(a_row['val_path_proxy'])} | {_fmt(a_row['val_rot_traj'])} | {_fmt(a_row['val_tdir_traj'])} | {_fmt(a_row['val_tmag_step'])} | {_fmt(a_row['odometry_ATE'])} | {_fmt(a_row['odometry_drift'])} | {_fmt(a_row['odometry_path_ratio'])} | {_fmt(a_row['R_delta'])} | {_fmt(a_row['tvec_delta'])} | {_fmt(a_row['tmag_delta'])} |",
        f"| {d_row['name']} | {_fmt(d_row['val_ate_proxy'])} | {_fmt(d_row['val_drift_proxy'])} | {_fmt(d_row['val_path_proxy'])} | {_fmt(d_row['val_rot_traj'])} | {_fmt(d_row['val_tdir_traj'])} | {_fmt(d_row['val_tmag_step'])} | {_fmt(d_row['odometry_ATE'])} | {_fmt(d_row['odometry_drift'])} | {_fmt(d_row['odometry_path_ratio'])} | {_fmt(d_row['R_delta'])} | {_fmt(d_row['tvec_delta'])} | {_fmt(d_row['tmag_delta'])} |",
        "",
        "## Trainable parameter audit",
        f"- `{a_row['name']}`: trainable_param_count=`{a_row['trainable_param_audit']['trainable_param_count']}`, optimizer_param_count=`{a_row['trainable_param_audit']['optimizer_param_count']}`, trainable_groups=`{a_row['trainable_param_audit']['trainable_groups']}`",
        f"- `{d_row['name']}`: trainable_param_count=`{d_row['trainable_param_audit']['trainable_param_count']}`, optimizer_param_count=`{d_row['trainable_param_audit']['optimizer_param_count']}`, trainable_groups=`{d_row['trainable_param_audit']['trainable_groups']}`",
        "",
        "## Train loss snapshot",
        f"- `{a_row['name']}` first_train=`{a_row['train_loss_snapshot']['first_train']}`",
        f"- `{a_row['name']}` last_train=`{a_row['train_loss_snapshot']['last_train']}`",
        f"- `{d_row['name']}` first_train=`{d_row['train_loss_snapshot']['first_train']}`",
        f"- `{d_row['name']}` last_train=`{d_row['train_loss_snapshot']['last_train']}`",
        "",
        "## Load / wrapper audit",
        f"- `{a_row['name']}` missing/unexpected init=`{a_row['missing_unexpected']['init_missing']} / {a_row['missing_unexpected']['init_unexpected']}`, final=`{a_row['missing_unexpected']['final_missing']} / {a_row['missing_unexpected']['final_unexpected']}`, wrapper=`{a_row['policy_wrapper_enabled']}`",
        f"- `{d_row['name']}` missing/unexpected init=`{d_row['missing_unexpected']['init_missing']} / {d_row['missing_unexpected']['init_unexpected']}`, final=`{d_row['missing_unexpected']['final_missing']} / {d_row['missing_unexpected']['final_unexpected']}`, wrapper=`{d_row['policy_wrapper_enabled']}`",
        "",
        "## Pre-fix smoke comparison",
        f"- pre-fix D smoke: val_ate_proxy=`{_fmt(PRE_FIX_SMOKE['val_ate_proxy'])}`, val_path_proxy=`{_fmt(PRE_FIX_SMOKE['val_path_proxy'])}`, odom_ATE=`{_fmt(PRE_FIX_SMOKE['odometry_ATE'])}`, drift=`{_fmt(PRE_FIX_SMOKE['odometry_drift'])}`",
        f"- post-fix D retest: val_ate_proxy=`{_fmt(d_row['val_ate_proxy'])}`, val_path_proxy=`{_fmt(d_row['val_path_proxy'])}`, odom_ATE=`{_fmt(d_row['odometry_ATE'])}`, drift=`{_fmt(d_row['odometry_drift'])}`",
        "",
        "## Decision",
        f"- final classification: `{payload['final_classification']}`",
        f"- whether S15f lightweight CV is recommended: `{payload['recommend_s15f']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# S15e Tiny Trajectory Retest After Harness Fix Summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- candidates run: `{', '.join(payload['candidates_run'])}`",
        f"- training budget: `W={WINDOW_SIZE}, updates={UPDATES}, batch_size={BATCH_SIZE}`",
        f"- trainable mode: `{payload['trainable_mode']}`",
        f"- A result: val_ate_proxy=`{_fmt(a_row['val_ate_proxy'])}`, odom_ATE=`{_fmt(a_row['odometry_ATE'])}`, drift=`{_fmt(a_row['odometry_drift'])}`",
        f"- D result: val_ate_proxy=`{_fmt(d_row['val_ate_proxy'])}`, odom_ATE=`{_fmt(d_row['odometry_ATE'])}`, drift=`{_fmt(d_row['odometry_drift'])}`",
        f"- pre-fix D comparison: old_ate_proxy=`{_fmt(PRE_FIX_SMOKE['val_ate_proxy'])}` -> new=`{_fmt(d_row['val_ate_proxy'])}`, old_drift=`{_fmt(PRE_FIX_SMOKE['odometry_drift'])}` -> new=`{_fmt(d_row['odometry_drift'])}`",
        f"- S15f lightweight CV recommended: `{payload['recommend_s15f']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = _baseline_gate()
    if not gate["passed"]:
        payload = {
            "final_classification": "REPRODUCTION-MISMATCH",
            "main_finding": "baseline gate failed",
            "candidates_run": [],
            "recommend_s15f": False,
            "trainable_mode": "tmag_head_only",
            "results": [],
        }
        _write_json(CANDIDATES_PATH, payload)
        REPORT_PATH.write_text("# S15e Tiny Trajectory Retest After Harness Fix Report\n\n- final classification: `REPRODUCTION-MISMATCH`\n", encoding="utf-8")
        SUMMARY_PATH.write_text("# S15e Tiny Trajectory Retest After Harness Fix Summary\n\n- final classification: `REPRODUCTION-MISMATCH`\n", encoding="utf-8")
        return

    data_root = _ensure_subset_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_val = _build_eval_dataset(data_root, SPLIT_SEED, "test", (1,))
    manifest_path, manifest_audit = _build_window_manifest(data_root, SPLIT_SEED, "train")
    try:
        rows: List[Dict[str, Any]] = []
        for candidate in _candidate_configs():
            run_payload = _run_train(candidate, data_root, manifest_path, UPDATES)
            rows.append(_extract_result(candidate, run_payload, data_root, ds_val, device))
        final_classification, main_finding, recommend = _classify(rows, gate)
        payload = {
            "baseline_gate": gate,
            "fold": FOLD_NAME,
            "split_seed": SPLIT_SEED,
            "training_budget": {
                "W": WINDOW_SIZE,
                "updates": UPDATES,
                "batch_size": BATCH_SIZE,
            },
            "trainable_mode": "tmag_head_only",
            "manifest_audit": manifest_audit,
            "pre_fix_smoke": PRE_FIX_SMOKE,
            "results": rows,
            "candidates_run": [row["name"] for row in rows],
            "final_classification": final_classification,
            "main_finding": main_finding,
            "recommend_s15f": bool(recommend),
            "s5_remains_final_clean_candidate": True,
        }
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
    finally:
        for path in (REPO_ROOT / "tmp",):
            _ = path


if __name__ == "__main__":
    main()
