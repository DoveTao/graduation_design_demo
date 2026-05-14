#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from losses import pose_loss, translation_magnitude_loss
from model import PanoramaRelPoseModel
from tools.s15_trajectory_level_training_objective import (
    FOLDS,
    S5_POLICY_PATH,
    _baseline_gate,
    _build_eval_dataset,
    _ensure_subset_data_root,
)
from tools.s15c_clean_policy_wrapped_training_harness_fix import (
    PredTmagShrinkModel,
    _build_direct_s5_wrapped_model,
    _fmt,
    _read_json,
    _safe_float,
)
from train_mvp import (
    _apply_cfg_overrides,
    _apply_dt_bucket_scale_anchor_policy,
    _configure_model_train_mode,
    _restore_cfg_from_policy_base_checkpoint,
    _seed_everything,
    _set_train_tmag_head_only,
)


REPORT_PATH = REPO_ROOT / "checkpoints" / "S15d_train_eval_forward_parity_fix_report.md"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15d_train_eval_forward_parity_fix_summary.md"


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _build_training_cfg(*, train_forward_eval_mode: bool = False) -> Tuple[Config, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
        f"train_forward_eval_mode={str(bool(train_forward_eval_mode))}",
    ]
    _apply_cfg_overrides(cfg, overrides)
    policy_summary_before = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, overrides)
    policy_summary_after = _apply_dt_bucket_scale_anchor_policy(cfg)
    return cfg, policy_summary_before, restore_summary, policy_summary_after


def _build_training_path_model(device: torch.device, *, train_forward_eval_mode: bool = False) -> Tuple[torch.nn.Module, torch.nn.Module, Config, Dict[str, Any], Dict[str, Any]]:
    cfg, _policy_before, restore_summary, policy_after = _build_training_cfg(train_forward_eval_mode=train_forward_eval_mode)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    ckpt_path = Path(str(getattr(cfg, "init_checkpoint")))
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    freeze_summary = _set_train_tmag_head_only(model, cfg)
    s5 = _read_json(S5_POLICY_PATH)
    wrapped = PredTmagShrinkModel(
        model,
        q90=float(s5["thresholds"]["q90_value"]),
        q95=float(s5["thresholds"]["q95_upper_tail_value"]),
        mid_scale=float(s5["scales"]["mid_scale"]),
        high_scale=float(s5["scales"]["high_scale"]),
    ).to(device)
    return wrapped, model, cfg, {
        "missing": len(msg.missing_keys),
        "unexpected": len(msg.unexpected_keys),
        "restore_enabled": bool(restore_summary.get("enabled", False)),
        "policy_enabled": bool(policy_after.get("enabled", False)),
        "policy_lineage": list(policy_after.get("policy_lineage", [])),
    }, freeze_summary


def _sample_tensors(sample: Dict[str, Any], meta: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
    return IA, IB, dt


def _tensor_maxdiff(a: Any, b: Any) -> float:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return float("nan")
    aa = a.detach().float()
    bb = b.detach().float()
    if aa.shape != bb.shape:
        return float("inf")
    return float((aa - bb).abs().max().cpu())


def _collect_common_aux_diffs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    common = sorted(set(a.keys()) & set(b.keys()))
    out: Dict[str, float] = {}
    for key in common:
        if torch.is_tensor(a[key]) and torch.is_tensor(b[key]):
            out[key] = _tensor_maxdiff(a[key], b[key])
    return out


def _forward_capture(
    wrapped: torch.nn.Module,
    base: torch.nn.Module,
    IA: torch.Tensor,
    IB: torch.Tensor,
    dt: torch.Tensor,
    *,
    mode: str,
    use_grad: bool,
    force_all_dropout_eval: bool = False,
) -> Dict[str, Any]:
    if mode == "eval":
        base.eval()
    elif mode == "train":
        base.train()
    elif mode == "configured":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dropout_states: List[Tuple[torch.nn.Module, bool]] = []
    if force_all_dropout_eval:
        for mod in base.modules():
            if "Dropout" in mod.__class__.__name__:
                dropout_states.append((mod, bool(mod.training)))
                mod.training = False
    try:
        if use_grad:
            R_wrap, _t_wrap, aux_wrap = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt)
            R_raw, _t_raw, aux_raw = base(IA, IB, enable_depth_fusion=True, dt_world=dt)
        else:
            with torch.no_grad():
                R_wrap, _t_wrap, aux_wrap = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt)
                R_raw, _t_raw, aux_raw = base(IA, IB, enable_depth_fusion=True, dt_world=dt)
    finally:
        for mod, state in dropout_states:
            mod.training = state

    return {
        "wrapped_R": R_wrap.detach().float().cpu(),
        "wrapped_tvec": aux_wrap["t_vec_out"].detach().float().cpu(),
        "wrapped_tmag": aux_wrap["t_mag"].detach().float().view(-1).cpu(),
        "raw_R": R_raw.detach().float().cpu(),
        "raw_tdir": aux_raw["t_dir_out"].detach().float().cpu(),
        "raw_tmag": aux_raw["t_mag"].detach().float().view(-1).cpu(),
        "raw_tvec": aux_raw["t_vec_out"].detach().float().cpu(),
        "aux_wrap": {k: v.detach().float().cpu() if torch.is_tensor(v) else v for k, v in aux_wrap.items()},
        "aux_raw": {k: v.detach().float().cpu() if torch.is_tensor(v) else v for k, v in aux_raw.items()},
        "base_training": bool(base.training),
    }


def _delta_summary(a: Dict[str, Any], b: Dict[str, Any], *, aux_source: str = "aux_raw") -> Dict[str, Any]:
    aux_diffs = _collect_common_aux_diffs(a[aux_source], b[aux_source])
    large_aux = sorted(((k, v) for k, v in aux_diffs.items() if math.isfinite(v)), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "R_diff": float((a["wrapped_R"] - b["wrapped_R"]).abs().max().cpu()),
        "tvec_diff": float((a["wrapped_tvec"] - b["wrapped_tvec"]).abs().max().cpu()),
        "tmag_diff": float((a["wrapped_tmag"] - b["wrapped_tmag"]).abs().max().cpu()),
        "raw_R_diff": float((a["raw_R"] - b["raw_R"]).abs().max().cpu()),
        "raw_tdir_diff": float((a["raw_tdir"] - b["raw_tdir"]).abs().max().cpu()),
        "raw_tmag_diff": float((a["raw_tmag"] - b["raw_tmag"]).abs().max().cpu()),
        "raw_tvec_diff": float((a["raw_tvec"] - b["raw_tvec"]).abs().max().cpu()),
        "top_aux_diffs": [{"key": k, "diff": float(v)} for k, v in large_aux],
    }


def _deterministic_forward_audit(ds_train, device: torch.device) -> Dict[str, Any]:
    wrapped, base, cfg, load_summary, _freeze = _build_training_path_model(device, train_forward_eval_mode=False)
    sample, meta = ds_train[0], ds_train.manifest()[0]
    IA, IB, dt = _sample_tensors(sample, meta, device)
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))
    eval_no_grad = _forward_capture(wrapped, base, IA, IB, dt, mode="eval", use_grad=False)
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))
    eval_grad = _forward_capture(wrapped, base, IA, IB, dt, mode="eval", use_grad=True)
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))
    _configure_model_train_mode(base, cfg)
    train_config_no_grad = _forward_capture(wrapped, base, IA, IB, dt, mode="configured", use_grad=False)
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))
    _configure_model_train_mode(base, cfg)
    train_config_grad = _forward_capture(wrapped, base, IA, IB, dt, mode="configured", use_grad=True)
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))
    _configure_model_train_mode(base, cfg)
    train_force_dropout_eval = _forward_capture(
        wrapped,
        base,
        IA,
        IB,
        dt,
        mode="configured",
        use_grad=False,
        force_all_dropout_eval=True,
    )
    grad_eval = _delta_summary(eval_no_grad, eval_grad)
    grad_train = _delta_summary(train_config_no_grad, train_config_grad)
    eval_vs_train = _delta_summary(eval_no_grad, train_config_no_grad)
    eval_vs_force_dropout_eval = _delta_summary(eval_no_grad, train_force_dropout_eval)
    return {
        "load_summary": load_summary,
        "grad_enabled_vs_no_grad_eval": grad_eval,
        "grad_enabled_vs_no_grad_train": grad_train,
        "eval_vs_train_configured": eval_vs_train,
        "eval_vs_train_force_all_dropout_eval": eval_vs_force_dropout_eval,
        "configured_mode": _configure_model_train_mode(base, cfg),
        "grad_forward_mismatch": max(
            _safe_float(grad_eval["R_diff"]),
            _safe_float(grad_eval["tvec_diff"]),
            _safe_float(grad_eval["tmag_diff"]),
            _safe_float(grad_train["R_diff"]),
            _safe_float(grad_train["tvec_diff"]),
            _safe_float(grad_train["tmag_diff"]),
        ) > 1.0e-8,
    }


def _zero_update_audit(ds_val, device: torch.device) -> Dict[str, Any]:
    direct_model, _direct_cfg, direct_meta = _build_direct_s5_wrapped_model(device)
    train_model, base, _cfg, load_summary, _freeze = _build_training_path_model(device, train_forward_eval_mode=False)
    base.eval()
    sample, meta = ds_val[0], ds_val.manifest()[0]
    IA, IB, dt = _sample_tensors(sample, meta, device)
    with torch.no_grad():
        R_a, _t_a, aux_a = direct_model(IA, IB, enable_depth_fusion=True, dt_world=dt)
        R_b, _t_b, aux_b = train_model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    return {
        "direct_missing": int(direct_meta["missing"]),
        "direct_unexpected": int(direct_meta["unexpected"]),
        "training_missing": int(load_summary["missing"]),
        "training_unexpected": int(load_summary["unexpected"]),
        "R_diff": float((R_a - R_b).abs().max().cpu()),
        "tvec_diff": float((aux_a["t_vec_out"] - aux_b["t_vec_out"]).abs().max().cpu()),
        "tmag_diff": float((aux_a["t_mag"].view(-1) - aux_b["t_mag"].view(-1)).abs().max().cpu()),
    }


def _source_audit() -> Dict[str, Any]:
    source_targets = ["train_mvp.py", "model.py", "interaction.py", "transformer_encoder.py"]
    patterns = {
        "functional_dropout": "F\\.dropout|functional\\.dropout",
        "train_calls": "\\.train\\(",
        "training_true_dropout": "dropout\\([^\\n]*training\\s*=\\s*True",
    }
    matches: Dict[str, List[str]] = {}
    for key, pattern in patterns.items():
        proc = subprocess.run(
            ["rg", "-n", pattern, *source_targets],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        matches[key] = rows
    hidden_train_call = any(
        "model.train()" in row or "model.train(True)" in row
        for row in matches["train_calls"]
        if "tools/s15" not in row
    )
    return {
        "matches": matches,
        "functional_dropout_found": len(matches["functional_dropout"]) > 0,
        "training_true_found": len(matches["training_true_dropout"]) > 0,
        "hidden_train_call_found": bool(hidden_train_call),
    }


def _buffer_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: buf.detach().clone().cpu() for name, buf in model.named_buffers()}


def _param_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone().cpu() for name, param in model.named_parameters()}


def _changed_names(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor], *, thresh: float = 0.0) -> List[str]:
    out = []
    for name, old in before.items():
        new = after[name]
        diff = float((new.float() - old.float()).abs().max().cpu())
        if diff > thresh:
            out.append(name)
    return out


def _one_update_audit(ds_train, ds_val, device: torch.device, *, train_forward_eval_mode: bool, lr: float = 5.0e-6) -> Dict[str, Any]:
    wrapped, base, cfg, load_summary, freeze_summary = _build_training_path_model(
        device,
        train_forward_eval_mode=train_forward_eval_mode,
    )
    train_sample, train_meta = ds_train[0], ds_train.manifest()[0]
    val_sample, val_meta = ds_val[0], ds_val.manifest()[0]
    IA_train, IB_train, dt_train = _sample_tensors(train_sample, train_meta, device)
    IA_val, IB_val, dt_val = _sample_tensors(val_sample, val_meta, device)
    R_gt = train_sample["R_gt"].unsqueeze(0).to(device)
    t_gt_dir = train_sample["t_gt_dir"].unsqueeze(0).to(device)
    t_gt_mag = train_sample["t_gt_mag"].view(1).to(device)

    mode_summary = _configure_model_train_mode(base, cfg)
    before = _forward_capture(wrapped, base, IA_val, IB_val, dt_val, mode="configured", use_grad=False)
    params_before = _param_snapshot(base)
    buffers_before = _buffer_snapshot(base)
    trainable_names = {name for name, p in base.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.wd)
    optimizer_names = {
        name
        for name, p in base.named_parameters()
        if any(p is ref for group in optimizer.param_groups for ref in group["params"])
    }
    optimizer.zero_grad(set_to_none=True)
    R_pred, t_pred, aux = base(IA_train, IB_train, enable_depth_fusion=True, dt_world=dt_train)
    loss_pose = pose_loss(R_pred, t_pred, R_gt, t_gt_dir, pred_t_frame=aux.get("t_local_frame", "A"))
    loss_tmag = translation_magnitude_loss(aux["t_mag"], t_gt_mag, loss_type="log_smooth_l1", eps=1.0e-3)
    loss_total = loss_pose + 0.1 * loss_tmag
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = _forward_capture(wrapped, base, IA_val, IB_val, dt_val, mode="configured", use_grad=False)
    params_after = _param_snapshot(base)
    buffers_after = _buffer_snapshot(base)

    changed_params = _changed_names(params_before, params_after)
    changed_buffers = _changed_names(buffers_before, buffers_after)
    frozen_changed = [name for name in changed_params if name not in trainable_names]
    forbidden_optimizer = sorted(set(optimizer_names) - trainable_names)
    return {
        "train_forward_eval_mode": bool(train_forward_eval_mode),
        "mode_summary": mode_summary,
        "load_summary": load_summary,
        "freeze_summary": freeze_summary,
        "lr": float(lr),
        "loss_pose": float(loss_pose.detach().cpu()),
        "loss_tmag": float(loss_tmag.detach().cpu()),
        "loss_total": float(loss_total.detach().cpu()),
        "wrapped": _delta_summary(before, after),
        "raw": {
            "R_delta": float((after["raw_R"] - before["raw_R"]).norm().cpu()),
            "tdir_delta": float((after["raw_tdir"] - before["raw_tdir"]).norm().cpu()),
            "tvec_delta": float((after["raw_tvec"] - before["raw_tvec"]).norm().cpu()),
            "tmag_delta": float((after["raw_tmag"] - before["raw_tmag"]).abs().max().cpu()),
        },
        "changed_params": changed_params,
        "changed_buffers": changed_buffers,
        "frozen_changed_params": frozen_changed,
        "forbidden_optimizer_params": forbidden_optimizer,
    }


def _classify(
    zero_update: Dict[str, Any],
    deterministic: Dict[str, Any],
    source_audit: Dict[str, Any],
    standard_update: Dict[str, Any],
    eval_forward_update: Dict[str, Any],
) -> Tuple[str, str, bool]:
    if max(_safe_float(zero_update["R_diff"]), _safe_float(zero_update["tvec_diff"]), _safe_float(zero_update["tmag_diff"])) > 1.0e-8:
        return "PARITY-UNRESOLVED", "zero-update wrapped mismatch remains", False
    if bool(deterministic["grad_forward_mismatch"]):
        return "GRAD-FORWARD-MISMATCH", "grad-enabled forward differs from no_grad forward", False
    if source_audit["functional_dropout_found"] or source_audit["training_true_found"] or source_audit["hidden_train_call_found"]:
        return "HIDDEN-DROPOUT-OR-TRAIN-CALL", "source audit found functional dropout or hidden train-mode call", False
    if standard_update["frozen_changed_params"] or standard_update["changed_buffers"] or standard_update["forbidden_optimizer_params"]:
        return "MUTABLE-STATE-DRIFT", "frozen params, buffers, or optimizer containment drifted", False

    eval_train = deterministic["eval_vs_train_configured"]
    standard_ok = (
        _safe_float(eval_train["R_diff"]) < 0.05
        and _safe_float(eval_train["tvec_diff"]) < 0.01
        and _safe_float(standard_update["wrapped"]["R_diff"]) < 0.01
        and _safe_float(standard_update["wrapped"]["tvec_diff"]) < 0.005
    )
    eval_forward_ok = (
        _safe_float(eval_forward_update["wrapped"]["R_diff"]) < 0.01
        and _safe_float(eval_forward_update["wrapped"]["tvec_diff"]) < 0.005
        and not eval_forward_update["frozen_changed_params"]
        and not eval_forward_update["changed_buffers"]
        and not eval_forward_update["forbidden_optimizer_params"]
    )
    if standard_ok:
        return "FORWARD-PARITY-FIX-PASS", "fixed train-mode preservation removed unintended train/eval parity drift", True
    if eval_forward_ok:
        return "EVAL-FORWARD-TRAINING-REQUIRED", "training must keep eval-mode forward to preserve parity", True
    return "PARITY-UNRESOLVED", "forward parity is still outside acceptance after the harness fix", False


def _write_report(payload: Dict[str, Any]) -> None:
    det = payload["deterministic_forward_audit"]
    src = payload["dropout_hidden_train_audit"]
    std = payload["one_update_standard"]
    fev = payload["one_update_eval_forward"]
    lines = [
        "# S15d Train/Eval Forward Parity Fix Report",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- main root cause: `{payload['main_root_cause']}`",
        f"- S15 can continue: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
        "",
        "## Zero-update wrapped audit",
        f"- wrapped R diff: `{_fmt(payload['zero_update']['R_diff'])}`",
        f"- wrapped tvec diff: `{_fmt(payload['zero_update']['tvec_diff'])}`",
        f"- wrapped tmag diff: `{_fmt(payload['zero_update']['tmag_diff'])}`",
        f"- missing/unexpected direct: `{payload['zero_update']['direct_missing']} / {payload['zero_update']['direct_unexpected']}`",
        f"- missing/unexpected training: `{payload['zero_update']['training_missing']} / {payload['zero_update']['training_unexpected']}`",
        "",
        "## Deterministic forward audit",
        f"- configured mode: `{det['configured_mode'].get('mode')}` | root_train=`{det['configured_mode'].get('root_train')}`",
        f"- eval vs configured train: R_diff=`{_fmt(det['eval_vs_train_configured']['R_diff'])}`, tvec_diff=`{_fmt(det['eval_vs_train_configured']['tvec_diff'])}`, tmag_diff=`{_fmt(det['eval_vs_train_configured']['tmag_diff'])}`",
        f"- eval vs force-all-dropout-eval: R_diff=`{_fmt(det['eval_vs_train_force_all_dropout_eval']['R_diff'])}`, tvec_diff=`{_fmt(det['eval_vs_train_force_all_dropout_eval']['tvec_diff'])}`, tmag_diff=`{_fmt(det['eval_vs_train_force_all_dropout_eval']['tmag_diff'])}`",
        f"- grad/no_grad eval: R_diff=`{_fmt(det['grad_enabled_vs_no_grad_eval']['R_diff'])}`, tvec_diff=`{_fmt(det['grad_enabled_vs_no_grad_eval']['tvec_diff'])}`, tmag_diff=`{_fmt(det['grad_enabled_vs_no_grad_eval']['tmag_diff'])}`",
        f"- grad/no_grad configured train: R_diff=`{_fmt(det['grad_enabled_vs_no_grad_train']['R_diff'])}`, tvec_diff=`{_fmt(det['grad_enabled_vs_no_grad_train']['tvec_diff'])}`, tmag_diff=`{_fmt(det['grad_enabled_vs_no_grad_train']['tmag_diff'])}`",
        f"- grad forward mismatch: `{det['grad_forward_mismatch']}`",
        "",
        "## Dropout / hidden train-call audit",
        f"- functional dropout found: `{src['functional_dropout_found']}`",
        f"- explicit training=True found: `{src['training_true_found']}`",
        f"- hidden train call found: `{src['hidden_train_call_found']}`",
        f"- train() matches: `{len(src['matches']['train_calls'])}`",
        "",
        "## One-update tiny audit",
        f"- standard mode wrapped: R_delta=`{_fmt(std['wrapped']['R_diff'])}`, tvec_delta=`{_fmt(std['wrapped']['tvec_diff'])}`, tmag_delta=`{_fmt(std['wrapped']['tmag_diff'])}`",
        f"- standard mode raw: R_delta=`{_fmt(std['raw']['R_delta'])}`, tdir_delta=`{_fmt(std['raw']['tdir_delta'])}`, tvec_delta=`{_fmt(std['raw']['tvec_delta'])}`, tmag_delta=`{_fmt(std['raw']['tmag_delta'])}`",
        f"- eval-forward mode wrapped: R_delta=`{_fmt(fev['wrapped']['R_diff'])}`, tvec_delta=`{_fmt(fev['wrapped']['tvec_diff'])}`, tmag_delta=`{_fmt(fev['wrapped']['tmag_diff'])}`",
        f"- eval-forward mode raw: R_delta=`{_fmt(fev['raw']['R_delta'])}`, tdir_delta=`{_fmt(fev['raw']['tdir_delta'])}`, tvec_delta=`{_fmt(fev['raw']['tvec_delta'])}`, tmag_delta=`{_fmt(fev['raw']['tmag_delta'])}`",
        "",
        "## Mutable state audit",
        f"- standard frozen params changed: `{len(std['frozen_changed_params'])}`",
        f"- standard buffers changed: `{len(std['changed_buffers'])}`",
        f"- standard forbidden optimizer params: `{len(std['forbidden_optimizer_params'])}`",
        f"- eval-forward frozen params changed: `{len(fev['frozen_changed_params'])}`",
        f"- eval-forward buffers changed: `{len(fev['changed_buffers'])}`",
        f"- eval-forward forbidden optimizer params: `{len(fev['forbidden_optimizer_params'])}`",
        "",
        "## Decision",
        f"- final classification: `{payload['final_classification']}`",
        f"- whether S15 can continue: `{payload['s15_can_continue']}`",
        f"- next step: `{payload['next_step']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# S15d Train/Eval Forward Parity Fix Summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- main root cause: `{payload['main_root_cause']}`",
        f"- deterministic forward audit: eval_vs_train_R_diff=`{_fmt(det['eval_vs_train_configured']['R_diff'])}`, eval_vs_train_tvec_diff=`{_fmt(det['eval_vs_train_configured']['tvec_diff'])}`",
        f"- dropout/hidden train-call audit: functional_dropout=`{src['functional_dropout_found']}`, hidden_train_call=`{src['hidden_train_call_found']}`",
        f"- standard one-update: R_delta=`{_fmt(std['wrapped']['R_diff'])}`, tvec_delta=`{_fmt(std['wrapped']['tvec_diff'])}`, tmag_delta=`{_fmt(std['wrapped']['tmag_diff'])}`",
        f"- eval-forward one-update: R_delta=`{_fmt(fev['wrapped']['R_diff'])}`, tvec_delta=`{_fmt(fev['wrapped']['tvec_diff'])}`, tmag_delta=`{_fmt(fev['wrapped']['tmag_diff'])}`",
        f"- mutable state audit: standard_buffers=`{len(std['changed_buffers'])}`, eval_forward_buffers=`{len(fev['changed_buffers'])}`",
        f"- S15 can continue: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = _baseline_gate()
    if not gate["passed"]:
        REPORT_PATH.write_text("# S15d Train/Eval Forward Parity Fix Report\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        SUMMARY_PATH.write_text("# S15d Train/Eval Forward Parity Fix Summary\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        return

    data_root = _ensure_subset_data_root()
    _fold_name, split_seed = FOLDS[0]
    ds_train = _build_eval_dataset(data_root, split_seed, "train", (1,))
    ds_val = _build_eval_dataset(data_root, split_seed, "test", (1,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zero_update = _zero_update_audit(ds_val, device)
    deterministic = _deterministic_forward_audit(ds_train, device)
    source_audit = _source_audit()
    standard_update = _one_update_audit(ds_train, ds_val, device, train_forward_eval_mode=False)
    eval_forward_update = _one_update_audit(ds_train, ds_val, device, train_forward_eval_mode=True)
    final_classification, root_cause, can_continue = _classify(
        zero_update,
        deterministic,
        source_audit,
        standard_update,
        eval_forward_update,
    )
    payload = {
        "zero_update": zero_update,
        "deterministic_forward_audit": deterministic,
        "dropout_hidden_train_audit": source_audit,
        "one_update_standard": standard_update,
        "one_update_eval_forward": eval_forward_update,
        "final_classification": final_classification,
        "main_root_cause": root_cause,
        "s15_can_continue": can_continue,
        "next_step": "S15e tiny trajectory retest" if can_continue else "stop S15 line; keep S5 as final clean candidate",
    }
    _write_report(payload)


if __name__ == "__main__":
    main()
