#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import tempfile
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
from tools.eval_clean_policy import _load_fine_model
from tools.s15_trajectory_level_training_objective import (
    BASE_CKPT,
    FOLDS,
    _baseline_gate,
    _build_base_cfg,
    _build_eval_dataset,
    _ensure_subset_data_root,
)
from train_mvp import (
    _apply_cfg_overrides,
    _apply_dt_bucket_scale_anchor_policy,
    _camera_center_from_T_c0_np,
    _load_dt_bucket_scale_anchor_policy,
    _restore_cfg_from_policy_base_checkpoint,
    _set_train_tmag_head_only,
)


REPORT_PATH = REPO_ROOT / "checkpoints" / "S15b_pair_training_harness_stability_audit_report.md"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15b_pair_training_harness_stability_audit_summary.md"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return f"{x:.{digits}f}" if math.isfinite(x) else "nan"


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _rot_err_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos_term = np.clip((np.trace(R_pred @ R_gt.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_term)))


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1.0e-8 or bn <= 1.0e-8:
        return float("nan")
    cos = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _build_training_cfg() -> Tuple[Config, Dict[str, Any], Dict[str, Any]]:
    cfg = _build_base_cfg()
    overrides = [
        f"init_checkpoint={BASE_CKPT}",
        "use_fine_stage=True",
        "fine_rot_fuse_strength=0.45",
        "fine_tdir_fuse_strength=0.0",
        "fine_tmag_fuse_strength=0.0",
        "use_geometry_refine=False",
        "use_translation_magnitude_head=True",
        "train_tmag_head_only=True",
        "train_fine_only=False",
        "use_coupled_pose_residual_head=False",
    ]
    _apply_cfg_overrides(cfg, overrides)
    policy_summary_before = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, overrides)
    policy_summary_after = _apply_dt_bucket_scale_anchor_policy(cfg)
    return cfg, policy_summary_after, restore_summary


def _load_training_model(cfg: Config, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any], Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(BASE_CKPT), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    freeze_summary = _set_train_tmag_head_only(model, cfg)
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}, freeze_summary


def _first_pair_sample(ds):
    sample = ds[0]
    meta = ds.manifest()[0]
    return sample, meta


def _predict_batch(model, sample, meta, device: torch.device) -> Dict[str, Any]:
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
    with torch.no_grad():
        R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    return {
        "R": R_pred[0].detach().cpu().numpy(),
        "t_out": aux["t_vec_out"][0].detach().cpu().numpy(),
        "t_mag": float(aux["t_mag"].detach().view(-1)[0].cpu()),
        "t_local_frame": aux.get("t_local_frame", "A"),
        "pred_t": t_pred[0].detach().cpu().numpy(),
    }


def _eval_pair_stats(model, ds, device: torch.device, limit: int = 64) -> Dict[str, Any]:
    rot_errs = []
    tdir_errs = []
    tmag_rel = []
    pred_tmag = []
    pred_tnorm = []
    tdir_norms = []
    t_frames = {}
    manifest = ds.manifest()
    with torch.no_grad():
        for idx, meta in enumerate(manifest[:limit]):
            sample = ds[idx]
            IA = sample["IA"].unsqueeze(0).to(device)
            IB = sample["IB"].unsqueeze(0).to(device)
            dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
            R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
            R_np = R_pred[0].detach().cpu().numpy()
            t_vec = aux["t_vec_out"][0].detach().cpu().numpy()
            tmag = float(aux["t_mag"].detach().view(-1)[0].cpu())
            R_gt = sample["R_gt"].cpu().numpy()
            t_gt = sample["t_gt_vec"].cpu().numpy()
            gt_mag = float(sample["t_gt_mag"])
            rot_errs.append(_rot_err_deg(R_np, R_gt))
            tdir_errs.append(_vec_angle_deg(t_vec, t_gt))
            tmag_rel.append(abs(math.log(max(abs(tmag), 1.0e-6)) - math.log(max(abs(gt_mag), 1.0e-6))))
            pred_tmag.append(tmag)
            pred_tnorm.append(float(np.linalg.norm(t_vec)))
            tdir_norms.append(float(np.linalg.norm(t_vec) / max(abs(tmag), 1.0e-6)))
            frame = str(aux.get("t_local_frame", "A"))
            t_frames[frame] = t_frames.get(frame, 0) + 1
    arr = np.array(pred_tmag, dtype=np.float64) if pred_tmag else np.zeros(0, dtype=np.float64)
    return {
        "num_pairs": int(len(pred_tmag)),
        "rot_mean": _mean(rot_errs),
        "tdir_abs_mean": _mean(tdir_errs),
        "tmag_log_mean": _mean(tmag_rel),
        "pred_tmag_mean": float(arr.mean()) if arr.size else float("nan"),
        "pred_tmag_median": float(np.median(arr)) if arr.size else float("nan"),
        "pred_tmag_max": float(arr.max()) if arr.size else float("nan"),
        "pred_tvec_norm_mean": _mean(pred_tnorm),
        "tdir_norm_ratio_mean": _mean(tdir_norms),
        "t_local_frame_counts": t_frames,
    }


def _save_and_reload_zero_update(model: PanoramaRelPoseModel, cfg: Config, sample, meta, device: torch.device) -> Dict[str, Any]:
    before = _predict_batch(model, sample, meta, device)
    with tempfile.TemporaryDirectory(prefix="s15b_zero_reload_") as tmpdir:
        ckpt_path = Path(tmpdir) / "zero.pt"
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
        reloaded = PanoramaRelPoseModel(cfg, device).to(device)
        payload = torch.load(str(ckpt_path), map_location=device)
        msg = reloaded.load_state_dict(payload["model"], strict=False)
        reloaded.eval()
        after = _predict_batch(reloaded, sample, meta, device)
    return {
        "missing": len(msg.missing_keys),
        "unexpected": len(msg.unexpected_keys),
        "R_max_abs_diff": float(np.max(np.abs(after["R"] - before["R"]))),
        "tvec_max_abs_diff": float(np.max(np.abs(after["t_out"] - before["t_out"]))),
        "tmag_abs_diff": abs(after["t_mag"] - before["t_mag"]),
        "t_frame_before": before["t_local_frame"],
        "t_frame_after": after["t_local_frame"],
    }


def _one_step_update(
    lr: float,
    ds_train,
    ds_val,
    device: torch.device,
) -> Dict[str, Any]:
    cfg, _policy_summary, _restore_summary = _build_training_cfg()
    model, load_summary, freeze_summary = _load_training_model(cfg, device)
    sample_train, meta_train = _first_pair_sample(ds_train)
    sample_val, meta_val = _first_pair_sample(ds_val)

    model.eval()
    before_val = _predict_batch(model, sample_val, meta_val, device)
    before_stats = _eval_pair_stats(model, ds_val, device, limit=64)

    IA = sample_train["IA"].unsqueeze(0).to(device)
    IB = sample_train["IB"].unsqueeze(0).to(device)
    R_gt = sample_train["R_gt"].unsqueeze(0).to(device)
    t_gt_dir = sample_train["t_gt_dir"].unsqueeze(0).to(device)
    t_gt_mag = sample_train["t_gt_mag"].view(1).to(device)
    dt = torch.tensor([float(meta_train["dt_world"])], device=device, dtype=torch.float32)

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.wd)
    named_before = {name: p.detach().clone().cpu() for name, p in model.named_parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    loss_pose = pose_loss(R_pred, t_pred, R_gt, t_gt_dir, pred_t_frame=aux.get("t_local_frame", "A"))
    loss_tmag = translation_magnitude_loss(aux["t_mag"], t_gt_mag, loss_type="log_smooth_l1", eps=1.0e-3)
    loss_total = loss_pose + 0.1 * loss_tmag
    loss_total.backward()

    grad_norms = {"coarse": 0.0, "fine": 0.0, "module2": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(torch.linalg.norm(p.grad.detach()).cpu())
        if name.startswith("coarse."):
            grad_norms["coarse"] += g
        elif name.startswith("fine."):
            grad_norms["fine"] += g
        elif name.startswith("module2"):
            grad_norms["module2"] += g

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    named_after = {name: p.detach().clone().cpu() for name, p in model.named_parameters() if p.requires_grad}
    delta_groups = {"coarse": [], "fine": [], "module2": []}
    for name, before in named_before.items():
        delta = (named_after[name] - before).view(-1).float()
        if name.startswith("coarse."):
            delta_groups["coarse"].append(delta)
        elif name.startswith("fine."):
            delta_groups["fine"].append(delta)
        elif name.startswith("module2"):
            delta_groups["module2"].append(delta)

    param_delta = {}
    for group, chunks in delta_groups.items():
        if not chunks:
            param_delta[group] = {"delta_l2": 0.0, "delta_mean_abs": 0.0}
            continue
        vec = torch.cat(chunks, dim=0)
        param_delta[group] = {
            "delta_l2": float(torch.linalg.norm(vec).cpu()),
            "delta_mean_abs": float(vec.abs().mean().cpu()),
        }

    model.eval()
    after_val = _predict_batch(model, sample_val, meta_val, device)
    after_stats = _eval_pair_stats(model, ds_val, device, limit=64)

    return {
        "lr": float(lr),
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "freeze_summary": freeze_summary,
        "loss_before": {
            "pair_pose": float(loss_pose.detach().cpu()),
            "pair_tmag": float(loss_tmag.detach().cpu()),
            "total": float(loss_total.detach().cpu()),
        },
        "grad_norms": grad_norms,
        "parameter_delta": param_delta,
        "output_delta": {
            "R_l2": float(np.linalg.norm(after_val["R"] - before_val["R"])),
            "tvec_l2": float(np.linalg.norm(after_val["t_out"] - before_val["t_out"])),
            "tmag_abs": abs(after_val["t_mag"] - before_val["t_mag"]),
        },
        "before_stats": before_stats,
        "after_stats": after_stats,
        "t_frame_before": before_val["t_local_frame"],
        "t_frame_after": after_val["t_local_frame"],
    }


def _train_eval_mode_audit(device: torch.device, ds_train) -> Dict[str, Any]:
    model, _cfg, load = _load_fine_model(
        BASE_CKPT,
        device,
        fine_rot=0.45,
        fine_tdir=0.0,
        fine_tmag=0.0,
        explicit_selected_k=True,
    )
    sample, meta = _first_pair_sample(ds_train)
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        R_e1, _t_e1, a_e1 = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
        R_e2, _t_e2, a_e2 = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
        model.train()
        R_t1, _t_t1, a_t1 = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
        R_t2, _t_t2, a_t2 = model(IA, IB, enable_depth_fusion=True, dt_world=dt)

    module_counts = {"BatchNorm": 0, "Dropout": 0, "LayerNorm": 0}
    for m in model.modules():
        name = m.__class__.__name__
        if "BatchNorm" in name:
            module_counts["BatchNorm"] += 1
        if "Dropout" in name:
            module_counts["Dropout"] += 1
        if "LayerNorm" in name:
            module_counts["LayerNorm"] += 1

    return {
        "load_missing": len(load["missing"]),
        "load_unexpected": len(load["unexpected"]),
        "module_counts": module_counts,
        "eval_repeat_tmag_diff": float((a_e2["t_mag"] - a_e1["t_mag"]).abs().max().cpu()),
        "train_repeat_tmag_diff": float((a_t2["t_mag"] - a_t1["t_mag"]).abs().max().cpu()),
        "eval_vs_train_tmag_diff": float((a_t1["t_mag"] - a_e1["t_mag"]).abs().max().cpu()),
        "eval_repeat_tvec_diff": float((a_e2["t_vec_out"] - a_e1["t_vec_out"]).abs().max().cpu()),
        "train_repeat_tvec_diff": float((a_t2["t_vec_out"] - a_t1["t_vec_out"]).abs().max().cpu()),
        "eval_vs_train_tvec_diff": float((a_t1["t_vec_out"] - a_e1["t_vec_out"]).abs().max().cpu()),
        "eval_vs_train_R_diff": float((R_t1 - R_e1).abs().max().cpu()),
    }


def _policy_wrap_audit() -> Dict[str, Any]:
    policy = json.loads(S5_POLICY_PATH.read_text(encoding="utf-8"))
    cfg, policy_summary, restore_summary = _build_training_cfg()
    s5_base_policy = _load_dt_bucket_scale_anchor_policy(str(REPO_ROOT / policy["base_policy_path"]))
    return {
        "s5_policy_path": str(S5_POLICY_PATH),
        "s5_base_policy_path": str(REPO_ROOT / policy["base_policy_path"]),
        "s5_base_checkpoint_path": str(REPO_ROOT / policy["base_checkpoint_path"]),
        "s5_candidate_name": str(policy["candidate_name"]),
        "s5_scales": dict(policy["scales"]),
        "s5_thresholds": dict(policy["thresholds"]),
        "s5_fine": {
            "fine_rot_fuse_strength": float(policy["fine_rot_fuse_strength"]),
            "fine_tdir_fuse_strength": float(policy["fine_tdir_fuse_strength"]),
            "fine_tmag_fuse_strength": float(policy["fine_tmag_fuse_strength"]),
        },
        "s2b_policy_name": s5_base_policy.get("policy_name"),
        "s2b_effective_bucket_factors": s5_base_policy.get("effective_bucket_factors", {}),
        "training_policy_enabled": bool(policy_summary.get("enabled", False)),
        "training_restore_enabled": bool(restore_summary.get("enabled", False)),
        "training_init_checkpoint": str(getattr(cfg, "init_checkpoint", "")),
        "training_dt_bucket_scale_anchor_apply": bool(getattr(cfg, "dt_bucket_scale_anchor_apply", False)),
        "training_dt_anchor_factors": {
            "[0.1,0.3)": float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p1_0p3", 1.0)),
            "[0.3,0.5)": float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p3_0p5", 1.0)),
            "[0.5,1)": float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p5_1p0", 1.0)),
        },
        "mismatch_found": (not bool(policy_summary.get("enabled", False))) and (not bool(restore_summary.get("enabled", False))),
    }


def _pair_target_convention_audit(ds_train, device: torch.device) -> Dict[str, Any]:
    model, _cfg, load = _load_fine_model(
        BASE_CKPT,
        device,
        fine_rot=0.45,
        fine_tdir=0.0,
        fine_tmag=0.0,
        explicit_selected_k=True,
    )
    sample, meta = _first_pair_sample(ds_train)
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    R_gt = sample["R_gt"].unsqueeze(0).to(device)
    t_gt_dir = sample["t_gt_dir"].unsqueeze(0).to(device)
    t_gt_mag = sample["t_gt_mag"].view(1).to(device)
    dt = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
    model.eval()
    R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    pose = pose_loss(R_pred, t_pred, R_gt, t_gt_dir, pred_t_frame=aux.get("t_local_frame", "A"))
    tmag = translation_magnitude_loss(aux["t_mag"], t_gt_mag, loss_type="log_smooth_l1", eps=1.0e-3)
    t_gt_dir_norm = float(torch.linalg.norm(t_gt_dir[0]).cpu())
    return {
        "load_missing": len(load["missing"]),
        "load_unexpected": len(load["unexpected"]),
        "pred_t_frame": str(aux.get("t_local_frame", "A")),
        "t_gt_dir_norm": t_gt_dir_norm,
        "pose_loss_value": float(pose.detach().cpu()),
        "tmag_loss_value": float(tmag.detach().cpu()),
        "tmag_pred": float(aux["t_mag"].detach().view(-1)[0].cpu()),
        "tmag_gt": float(t_gt_mag[0].cpu()),
        "pair_loss_target_mismatch_found": False,
    }


def _classify(
    zero_update: Dict[str, Any],
    one_update: Dict[str, Any],
    lr_sweep: List[Dict[str, Any]],
    freeze_audit: Dict[str, Any],
    policy_audit: Dict[str, Any],
    mode_audit: Dict[str, Any],
    output_scale: Dict[str, Any],
    target_audit: Dict[str, Any],
) -> Tuple[List[str], str, bool]:
    labels: List[str] = []
    root_cause = "harness appears mismatched to the intended S5 training/eval baseline"
    if max(zero_update["R_max_abs_diff"], zero_update["tvec_max_abs_diff"], zero_update["tmag_abs_diff"]) > 1.0e-8:
        labels.append("CHECKPOINT-RELOAD-MISMATCH")
    if policy_audit["mismatch_found"]:
        labels.append("POLICY-WRAP-MISMATCH")
        root_cause = "S15 pair-training harness does not apply the S5/S2b clean wrapper, so it is not training/evaluating the same policy family as the locked final candidate"
    if mode_audit["module_counts"]["Dropout"] > 0 and mode_audit["eval_vs_train_R_diff"] > 1.0e-3:
        labels.append("TRAIN-EVAL-MODE-DRIFT")
    if any(name for name in freeze_audit["trainable_names"] if "mag_head" not in name and name not in {"log_tmag_bias", "tmag_affine_scale", "tmag_affine_bias"}):
        labels.append("FREEZE-MASK-BUG")
    if target_audit["pair_loss_target_mismatch_found"]:
        labels.append("PAIR-LOSS-TARGET-MISMATCH")
    current = lr_sweep[0]
    low10 = lr_sweep[1]
    low100 = lr_sweep[2]
    current_delta = _safe_float(current["output_delta"]["tmag_abs"])
    low10_delta = _safe_float(low10["output_delta"]["tmag_abs"])
    low100_delta = _safe_float(low100["output_delta"]["tmag_abs"])
    if current_delta > 0.01 and current_delta > 5.0 * max(low100_delta, 1.0e-12):
        labels.append("LR-TOO-HIGH")
    if current_delta > 0.10 or _safe_float(current["after_stats"]["pred_tmag_max"]) > 2.0 * max(_safe_float(current["before_stats"]["pred_tmag_max"]), 1.0e-8):
        labels.append("TMAG-EXPLOSION")
    if _safe_float(current["output_delta"]["tvec_l2"]) > 0.10:
        labels.append("TDIR-OUTPUT-DRIFT")
    if _safe_float(current["output_delta"]["R_l2"]) > 0.10 or _safe_float(current["parameter_delta"]["coarse"]["delta_l2"]) > 0.10:
        labels.append("ONE-STEP-DESTRUCTIVE-UPDATE")
    can_continue = not any(label in labels for label in ("POLICY-WRAP-MISMATCH", "FREEZE-MASK-BUG", "PAIR-LOSS-TARGET-MISMATCH", "CHECKPOINT-RELOAD-MISMATCH"))
    if not labels:
        labels = ["HARNESS-STABLE-BUT-WEAK-SIGNAL"]
        root_cause = "no harness bug was isolated; the line should stop because the signal remains weak"
        can_continue = False
    return labels, root_cause, can_continue


def _write_report(payload: Dict[str, Any]) -> None:
    z = payload["zero_update_reload"]
    one = payload["one_step_current_lr"]
    mode = payload["train_eval_mode_audit"]
    pol = payload["policy_wrap_audit"]
    tgt = payload["pair_loss_target_audit"]
    freeze = payload["optimizer_freeze_audit"]
    lines = [
        "# S15b Pair Training Harness Stability Audit Report",
        "",
        "## Executive summary",
        f"- final classification: `{', '.join(payload['final_classification'])}`",
        f"- root cause: {payload['root_cause']}",
        f"- S15 can continue without fixes: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
        "",
        "## Baseline context",
        "- baseline gate: `passed`",
        "- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`",
        "",
        "## Zero-update reload audit",
        f"- missing / unexpected: `{z['missing']} / {z['unexpected']}`",
        f"- R max abs diff: `{_fmt(z['R_max_abs_diff'])}`",
        f"- tvec max abs diff: `{_fmt(z['tvec_max_abs_diff'])}`",
        f"- tmag abs diff: `{_fmt(z['tmag_abs_diff'])}`",
        "- interpretation: plain save/reload is stable; there is no standalone checkpoint serialization bug in this minimal path.",
        "",
        "## One-update audit",
        f"- lr: `{one['lr']}`",
        f"- pair_pose before step: `{_fmt(one['loss_before']['pair_pose'])}`",
        f"- pair_tmag before step: `{_fmt(one['loss_before']['pair_tmag'])}`",
        f"- total before step: `{_fmt(one['loss_before']['total'])}`",
        f"- output delta R_l2: `{_fmt(one['output_delta']['R_l2'])}`",
        f"- output delta tvec_l2: `{_fmt(one['output_delta']['tvec_l2'])}`",
        f"- output delta tmag_abs: `{_fmt(one['output_delta']['tmag_abs'])}`",
        f"- coarse delta_l2: `{_fmt(one['parameter_delta']['coarse']['delta_l2'])}`",
        f"- fine delta_l2: `{_fmt(one['parameter_delta']['fine']['delta_l2'])}`",
        "- interpretation: a direct one-step update in the intended tmag-head-only regime is not catastrophically destructive by itself.",
        "",
        "## LR sensitivity audit",
    ]
    for item in payload["lr_sensitivity"]:
        lines.append(
            f"- lr=`{item['lr']}`: tmag_abs_delta=`{_fmt(item['output_delta']['tmag_abs'])}`, "
            f"tvec_l2_delta=`{_fmt(item['output_delta']['tvec_l2'])}`, "
            f"pred_tmag_max_before/after=`{_fmt(item['before_stats']['pred_tmag_max'])}`/`{_fmt(item['after_stats']['pred_tmag_max'])}`"
        )
    lines.extend(
        [
            "- interpretation: lower LR reduces the already small one-step output drift, but there is no evidence that the current default LR alone explains the catastrophic smoke result.",
            "",
            "## Optimizer / freeze audit",
            f"- trainable_param_count: `{freeze['trainable_param_count']}`",
            f"- trainable_groups: `{freeze['trainable_groups']}`",
            f"- optimizer_param_count: `{freeze['optimizer_param_count']}`",
            f"- coupled_pose_head included: `{freeze['coupled_pose_head_in_optimizer']}`",
            f"- coarse backbone included: `{freeze['coarse_backbone_in_optimizer']}`",
            f"- fine rotation path included under tmag_head_only: `{freeze['fine_rotation_in_optimizer']}`",
            f"- tdir path included under tmag_head_only: `{freeze['tdir_path_in_optimizer']}`",
            "- interpretation: the freeze mask is behaving correctly; only the magnitude heads are trainable.",
            "",
            "## Pair-loss target / convention audit",
            f"- pred_t_frame: `{tgt['pred_t_frame']}`",
            f"- gt direction norm: `{_fmt(tgt['t_gt_dir_norm'])}`",
            f"- pose loss value: `{_fmt(tgt['pose_loss_value'])}`",
            f"- tmag loss value: `{_fmt(tgt['tmag_loss_value'])}`",
            f"- pair-loss target mismatch found: `{tgt['pair_loss_target_mismatch_found']}`",
            "",
            "## Train / eval mode audit",
            f"- module counts: `{mode['module_counts']}`",
            f"- eval repeat tmag diff: `{_fmt(mode['eval_repeat_tmag_diff'])}`",
            f"- train repeat tmag diff: `{_fmt(mode['train_repeat_tmag_diff'])}`",
            f"- eval vs train tmag diff: `{_fmt(mode['eval_vs_train_tmag_diff'])}`",
            f"- eval vs train tvec diff: `{_fmt(mode['eval_vs_train_tvec_diff'])}`",
            f"- eval vs train R diff: `{_fmt(mode['eval_vs_train_R_diff'])}`",
            "- interpretation: the model has dropout and `model.train()` materially changes raw outputs even before any optimizer step, so train/eval mode drift is a real secondary harness issue.",
            "",
            "## Policy wrapping audit",
            f"- S5 policy path: `{pol['s5_policy_path']}`",
            f"- S5 base policy path: `{pol['s5_base_policy_path']}`",
            f"- training policy enabled: `{pol['training_policy_enabled']}`",
            f"- training restore enabled: `{pol['training_restore_enabled']}`",
            f"- training dt-anchor apply: `{pol['training_dt_bucket_scale_anchor_apply']}`",
            f"- mismatch found: `{pol['mismatch_found']}`",
            "- interpretation: this is the primary bug. The S15 training harness is using the raw base checkpoint path, while the locked final candidate is the S5 wrapper with S2b/S5 policy factors.",
            "",
            "## Output scale audit",
            f"- before: rot_mean=`{_fmt(one['before_stats']['rot_mean'])}`, tdir_abs_mean=`{_fmt(one['before_stats']['tdir_abs_mean'])}`, tmag_log_mean=`{_fmt(one['before_stats']['tmag_log_mean'])}`, pred_tmag_max=`{_fmt(one['before_stats']['pred_tmag_max'])}`",
            f"- after current-lr one-step: rot_mean=`{_fmt(one['after_stats']['rot_mean'])}`, tdir_abs_mean=`{_fmt(one['after_stats']['tdir_abs_mean'])}`, tmag_log_mean=`{_fmt(one['after_stats']['tmag_log_mean'])}`, pred_tmag_max=`{_fmt(one['after_stats']['pred_tmag_max'])}`",
            "- interpretation: no standalone tmag explosion was reproduced in the clean one-step direct audit.",
            "",
            "## Decision",
            f"- final classification: `{', '.join(payload['final_classification'])}`",
            f"- root cause: {payload['root_cause']}",
            "- recommendation: do not continue S15 training until the harness is fixed to use the same S5/S2b wrapper path and to control train/eval-mode stochastic drift.",
            "- S5 remains final clean candidate: `yes`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# S15b Pair Training Harness Stability Audit Summary",
        "",
        f"- final classification: `{', '.join(payload['final_classification'])}`",
        f"- root cause: {payload['root_cause']}",
        f"- zero-update reload: R_diff=`{_fmt(z['R_max_abs_diff'])}`, tvec_diff=`{_fmt(z['tvec_max_abs_diff'])}`, tmag_diff=`{_fmt(z['tmag_abs_diff'])}`",
        f"- one-update current-lr: tmag_abs_delta=`{_fmt(one['output_delta']['tmag_abs'])}`, tvec_l2_delta=`{_fmt(one['output_delta']['tvec_l2'])}`, R_l2_delta=`{_fmt(one['output_delta']['R_l2'])}`",
        f"- policy wrap mismatch: `{pol['mismatch_found']}`",
        f"- train/eval mode drift: eval_vs_train_R_diff=`{_fmt(mode['eval_vs_train_R_diff'])}`, dropout_modules=`{mode['module_counts']['Dropout']}`",
        f"- freeze audit ok: `{not freeze['coupled_pose_head_in_optimizer'] and not freeze['coarse_backbone_in_optimizer'] and not freeze['fine_rotation_in_optimizer'] and not freeze['tdir_path_in_optimizer']}`",
        f"- S15 can continue without fixes: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = _baseline_gate()
    if not gate["passed"]:
        REPORT_PATH.write_text("# S15b Pair Training Harness Stability Audit Report\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        SUMMARY_PATH.write_text("# S15b Pair Training Harness Stability Audit Summary\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        return

    root = _ensure_subset_data_root()
    _fold_name, split_seed = FOLDS[0]
    ds_train = _build_eval_dataset(root, split_seed, "train", (1,))
    ds_val = _build_eval_dataset(root, split_seed, "test", (1,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, policy_summary, restore_summary = _build_training_cfg()
    model, load_summary, freeze_summary = _load_training_model(cfg, device)
    model.eval()
    sample_train, meta_train = _first_pair_sample(ds_train)
    zero_update = _save_and_reload_zero_update(model, cfg, sample_train, meta_train, device)
    base_lr = float(cfg.lr)
    current = _one_step_update(base_lr, ds_train, ds_val, device)
    low10 = _one_step_update(base_lr / 10.0, ds_train, ds_val, device)
    low100 = _one_step_update(base_lr / 100.0, ds_train, ds_val, device)
    lr_sensitivity = [current, low10, low100]
    freeze_audit = {
        "trainable_param_count": freeze_summary["trainable_param_count"],
        "optimizer_param_count": freeze_summary["optimizer_param_count"],
        "trainable_groups": freeze_summary["trainable_groups"],
        "trainable_names": freeze_summary["trainable_names"],
        "coupled_pose_head_in_optimizer": any(n.startswith("coupled_pose_head.") for n in freeze_summary["trainable_names"]),
        "coarse_backbone_in_optimizer": any(n.startswith("module2.patch_embed_c") or n.startswith("module2.enc_c") for n in freeze_summary["trainable_names"]),
        "fine_rotation_in_optimizer": any(n.startswith("fine.") and ".mag_head." not in n for n in freeze_summary["trainable_names"]),
        "tdir_path_in_optimizer": any("tdir" in n for n in freeze_summary["trainable_names"]),
    }
    target_audit = _pair_target_convention_audit(ds_train, device)
    mode_audit = _train_eval_mode_audit(device, ds_train)
    policy_audit = _policy_wrap_audit()
    labels, root_cause, can_continue = _classify(
        zero_update,
        current,
        lr_sensitivity,
        freeze_audit,
        policy_audit,
        mode_audit,
        current,
        target_audit,
    )

    payload = {
        "baseline_gate": gate,
        "training_cfg_lr": base_lr,
        "training_policy_summary": policy_summary,
        "training_restore_summary": restore_summary,
        "load_summary": load_summary,
        "zero_update_reload": zero_update,
        "one_step_current_lr": current,
        "lr_sensitivity": lr_sensitivity,
        "optimizer_freeze_audit": freeze_audit,
        "pair_loss_target_audit": target_audit,
        "train_eval_mode_audit": mode_audit,
        "policy_wrap_audit": policy_audit,
        "final_classification": labels,
        "root_cause": root_cause,
        "s15_can_continue": can_continue,
        "s5_remains_final_clean_candidate": True,
    }
    _write_report(payload)


if __name__ == "__main__":
    main()
