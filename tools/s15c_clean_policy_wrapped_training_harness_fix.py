#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
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
from tools.eval_clean_policy import DtBucketScaledMagnitudeModel, _cfg_from_dict, _load_ckpt_cfg
from tools.s15_trajectory_level_training_objective import (
    FOLDS,
    PYTHON_BIN,
    RUN_ROOT,
    S5_POLICY_PATH,
    _baseline_gate,
    _build_eval_dataset,
    _build_window_manifest,
    _ensure_subset_data_root,
)
from train_mvp import (
    _apply_cfg_overrides,
    _apply_dt_bucket_scale_anchor_policy,
    _apply_train_mode_preserving_frozen_subtrees,
    _restore_cfg_from_policy_base_checkpoint,
    _set_train_tmag_head_only,
    eval_odometry_sequence,
)


REPORT_PATH = REPO_ROOT / "checkpoints" / "S15c_clean_policy_wrapped_training_harness_fix_report.md"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15c_clean_policy_wrapped_training_harness_fix_summary.md"


class PredTmagShrinkModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, q90: float, q95: float, mid_scale: float, high_scale: float) -> None:
        super().__init__()
        self.base = base
        self.q90 = float(q90)
        self.q95 = float(q95)
        self.mid_scale = float(mid_scale)
        self.high_scale = float(high_scale)
        self.cfg = getattr(base, "cfg", None)

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu().item())
        scale = 1.0
        if pred_mag >= self.q95:
            scale = self.high_scale
        elif pred_mag >= self.q90:
            scale = self.mid_scale
        if abs(scale - 1.0) < 1.0e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(scale, device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(1.0e-3))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["s5_tmag_scale_factor"] = fac
        return R_pred, t_pred, aux


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    ]
    _apply_cfg_overrides(cfg, overrides)
    policy_summary_before = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, overrides)
    policy_summary_after = _apply_dt_bucket_scale_anchor_policy(cfg)
    return cfg, policy_summary_before, restore_summary, policy_summary_after


def _build_direct_s5_wrapped_model(device: torch.device) -> Tuple[torch.nn.Module, Config, Dict[str, Any]]:
    s5 = _read_json(S5_POLICY_PATH)
    s2b = _read_json(REPO_ROOT / s5["base_policy_path"])
    ckpt_path = REPO_ROOT / s5["base_checkpoint_path"]
    cfg_dict = _load_ckpt_cfg(ckpt_path)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(s2b["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(s2b["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(s2b["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(s2b.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    cfg.max_eval_batches = 0
    cfg.odom_eval_prefer_k = 1
    cfg.odom_eval_fallback_to_min_k = False
    cfg.eval_k_list = (1, 2, 3, 5, 10, 20)
    base = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = base.load_state_dict(state, strict=False)
    base.eval()
    factors = {str(k): float(v) for k, v in s2b.get("effective_bucket_factors", {}).items()}
    s2b_wrapped = DtBucketScaledMagnitudeModel(base, factors).to(device)
    s5_wrapped = PredTmagShrinkModel(
        s2b_wrapped,
        q90=float(s5["thresholds"]["q90_value"]),
        q95=float(s5["thresholds"]["q95_upper_tail_value"]),
        mid_scale=float(s5["scales"]["mid_scale"]),
        high_scale=float(s5["scales"]["high_scale"]),
    ).to(device)
    return s5_wrapped, cfg, {"missing": len(msg.missing_keys), "unexpected": len(msg.unexpected_keys), "s5_policy": s5, "s2b_policy": s2b}


def _build_training_path_model(device: torch.device, ckpt_path: Path | None = None) -> Tuple[torch.nn.Module, Config, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg, policy_summary_before, restore_summary, policy_summary_after = _build_training_cfg()
    model = PanoramaRelPoseModel(cfg, device).to(device)
    use_ckpt = ckpt_path or Path(str(getattr(cfg, "init_checkpoint")))
    payload = torch.load(str(use_ckpt), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    freeze_summary = _set_train_tmag_head_only(model, cfg)
    model.eval()
    s5 = _read_json(S5_POLICY_PATH)
    wrapped = PredTmagShrinkModel(
        model,
        q90=float(s5["thresholds"]["q90_value"]),
        q95=float(s5["thresholds"]["q95_upper_tail_value"]),
        mid_scale=float(s5["scales"]["mid_scale"]),
        high_scale=float(s5["scales"]["high_scale"]),
    ).to(device)
    return wrapped, cfg, {"missing": len(msg.missing_keys), "unexpected": len(msg.unexpected_keys)}, freeze_summary, {
        "before": policy_summary_before,
        "restore": restore_summary,
        "after": policy_summary_after,
    }


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
        "t_frame": str(aux.get("t_local_frame", "A")),
    }


def _proxy_metrics(model, ds, device: torch.device, max_windows: int = 128) -> Dict[str, Any]:
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    ate_terms = []
    path_terms = []
    windows = 0
    with torch.no_grad():
        for idx, meta in enumerate(manifest):
            if windows >= int(max_windows):
                break
            nxt = by_seq_i.get((str(meta["scene"]), str(meta["seq"]), int(meta["j"])), None)
            if nxt is None:
                continue
            sample_ab = ds[idx]
            sample_bc = ds[nxt[0]]
            p_ab = _predict_batch(model, sample_ab, meta, device)
            p_bc = _predict_batch(model, sample_bc, nxt[1], device)
            R_gt_ab = sample_ab["R_gt"].cpu().numpy()
            t_gt_ab = sample_ab["t_gt_vec"].cpu().numpy()
            R_gt_bc = sample_bc["R_gt"].cpu().numpy()
            t_gt_bc = sample_bc["t_gt_vec"].cpu().numpy()
            from train_mvp import _compose_rel_pose_np, _camera_center_from_T_c0_np
            R_gt_BA, t_gt_BA = _compose_rel_pose_np(R_gt_ab, t_gt_ab, np.eye(3), np.zeros(3))
            R_gt_CA, t_gt_CA = _compose_rel_pose_np(R_gt_bc, t_gt_bc, R_gt_BA, t_gt_BA)
            R_pr_BA, t_pr_BA = _compose_rel_pose_np(p_ab["R"], p_ab["t_out"], np.eye(3), np.zeros(3))
            R_pr_CA, t_pr_CA = _compose_rel_pose_np(p_bc["R"], p_bc["t_out"], R_pr_BA, t_pr_BA)
            p1_gt = _camera_center_from_T_c0_np(R_gt_BA, t_gt_BA)
            p2_gt = _camera_center_from_T_c0_np(R_gt_CA, t_gt_CA)
            p1_pr = _camera_center_from_T_c0_np(R_pr_BA, t_pr_BA)
            p2_pr = _camera_center_from_T_c0_np(R_pr_CA, t_pr_CA)
            gt_path = float(np.linalg.norm(p1_gt) + np.linalg.norm(p2_gt - p1_gt))
            pred_path = float(np.linalg.norm(p1_pr) + np.linalg.norm(p2_pr - p1_pr))
            if gt_path <= 1.0e-6:
                continue
            windows += 1
            ate_terms.extend([
                float(np.linalg.norm(p1_pr - p1_gt)) / gt_path,
                float(np.linalg.norm(p2_pr - p2_gt)) / gt_path,
            ])
            path_terms.append(abs(math.log(max(pred_path, 1.0e-6)) - math.log(max(gt_path, 1.0e-6))))
    return {"num_windows": int(windows), "val_ate_proxy": _mean(ate_terms), "val_path_proxy": _mean(path_terms)}


def _zero_update_audit(ds_val, device: torch.device) -> Dict[str, Any]:
    direct_model, _direct_cfg, direct_meta = _build_direct_s5_wrapped_model(device)
    train_model, _cfg, load_summary, _freeze, policy_meta = _build_training_path_model(device)
    sample, meta = ds_val[0], ds_val.manifest()[0]
    a = _predict_batch(direct_model, sample, meta, device)
    b = _predict_batch(train_model, sample, meta, device)
    return {
        "direct_missing": int(direct_meta["missing"]),
        "direct_unexpected": int(direct_meta["unexpected"]),
        "training_missing": int(load_summary["missing"]),
        "training_unexpected": int(load_summary["unexpected"]),
        "R_diff": float(np.max(np.abs(a["R"] - b["R"]))),
        "tvec_diff": float(np.max(np.abs(a["t_out"] - b["t_out"]))),
        "tmag_diff": abs(a["t_mag"] - b["t_mag"]),
        "policy_meta": policy_meta,
        "wrapper_still_mismatch": max(float(np.max(np.abs(a["R"] - b["R"]))), float(np.max(np.abs(a["t_out"] - b["t_out"]))), abs(a["t_mag"] - b["t_mag"])) > 1.0e-4,
    }


def _dropout_drift_audit(ds_train, device: torch.device) -> Dict[str, Any]:
    wrapped, _cfg, _load, _freeze, _policy = _build_training_path_model(device)
    base = wrapped.base
    sample, meta = ds_train[0], ds_train.manifest()[0]
    base.eval()
    eval_out = _predict_batch(wrapped, sample, meta, device)
    mode_summary = _apply_train_mode_preserving_frozen_subtrees(base)
    train_out = _predict_batch(wrapped, sample, meta, device)
    base.eval()
    return {
        "dropout_modules": int(mode_summary["dropout_modules"]),
        "dropout_trainable_train": int(mode_summary["dropout_trainable_train"]),
        "dropout_frozen_eval": int(mode_summary["dropout_frozen_eval"]),
        "eval_vs_train_R_diff": float(np.max(np.abs(train_out["R"] - eval_out["R"]))),
        "eval_vs_train_tvec_diff": float(np.max(np.abs(train_out["t_out"] - eval_out["t_out"]))),
        "eval_vs_train_tmag_diff": abs(train_out["t_mag"] - eval_out["t_mag"]),
    }


def _one_update_audit(ds_train, ds_val, device: torch.device, lr: float) -> Dict[str, Any]:
    wrapped, cfg, load_summary, freeze_summary, _policy = _build_training_path_model(device)
    model = wrapped.base
    _apply_train_mode_preserving_frozen_subtrees(model)
    sample_train, meta_train = ds_train[0], ds_train.manifest()[0]
    sample_val, meta_val = ds_val[0], ds_val.manifest()[0]
    before = _predict_batch(wrapped, sample_val, meta_val, device)
    IA = sample_train["IA"].unsqueeze(0).to(device)
    IB = sample_train["IB"].unsqueeze(0).to(device)
    R_gt = sample_train["R_gt"].unsqueeze(0).to(device)
    t_gt_dir = sample_train["t_gt_dir"].unsqueeze(0).to(device)
    t_gt_mag = sample_train["t_gt_mag"].view(1).to(device)
    dt = torch.tensor([float(meta_train["dt_world"])], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.wd)
    named_before = {n: p.detach().clone().cpu() for n, p in model.named_parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)
    loss_pose = pose_loss(R_pred, t_pred, R_gt, t_gt_dir, pred_t_frame=aux.get("t_local_frame", "A"))
    loss_tmag = translation_magnitude_loss(aux["t_mag"], t_gt_mag, loss_type="log_smooth_l1", eps=1.0e-3)
    loss_total = loss_pose + 0.1 * loss_tmag
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    after = _predict_batch(wrapped, sample_val, meta_val, device)
    named_after = {n: p.detach().clone().cpu() for n, p in model.named_parameters() if p.requires_grad}
    delta_l2 = 0.0
    for n in named_before:
        delta_l2 += float(torch.linalg.norm((named_after[n] - named_before[n]).view(-1).float()).cpu())
    return {
        "lr": float(lr),
        "load_missing": int(load_summary["missing"]),
        "load_unexpected": int(load_summary["unexpected"]),
        "trainable_param_count": int(freeze_summary["trainable_param_count"]),
        "loss_pair_pose": float(loss_pose.detach().cpu()),
        "loss_pair_tmag": float(loss_tmag.detach().cpu()),
        "loss_total": float(loss_total.detach().cpu()),
        "R_delta": float(np.linalg.norm(after["R"] - before["R"])),
        "tvec_delta": float(np.linalg.norm(after["t_out"] - before["t_out"])),
        "tmag_delta": abs(after["t_mag"] - before["t_mag"]),
        "parameter_delta_l2": delta_l2,
    }


def _tiny_smoke(ds_val, data_root: Path, device: torch.device) -> Dict[str, Any]:
    fold_name, split_seed = FOLDS[0]
    manifest_path, _audit = _build_window_manifest(data_root, split_seed, "train")
    exp_name = "S15c_B_pair_plus_ate_path_wrapped_smoke_u10"
    run_dir = RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", f"ckpt_dir={RUN_ROOT}",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", "max_steps=10",
        "--set", "eval_every=0",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
        "--set", f"train_fixed_pairs_manifest_json={manifest_path}",
        "--set", f"dt_bucket_scale_anchor_policy_json={S5_POLICY_PATH}",
        "--set", "min_dt=0.0",
        "--set", "eval_min_dt=0.0",
        "--set", "use_coupled_pose_residual_head=False",
        "--set", "strict_load_checkpoint=False",
        "--set", "use_fine_stage=True",
        "--set", "fine_rot_fuse_strength=0.45",
        "--set", "fine_tdir_fuse_strength=0.0",
        "--set", "fine_tmag_fuse_strength=0.0",
        "--set", "use_geometry_refine=False",
        "--set", "use_translation_magnitude_head=True",
        "--set", "train_tmag_head_only=True",
        "--set", "train_fine_only=False",
        "--set", "use_traj_ate_loss=True",
        "--set", "w_traj_ate=0.05",
        "--set", "use_traj_path_loss=True",
        "--set", "w_traj_path=0.05",
        "--set", "use_traj_drift_loss=False",
        "--set", "use_traj_rot_loss=False",
        "--set", "use_traj_tdir_loss=False",
        "--set", "use_traj_tmag_step_loss=False",
        "--set", "use_traj_speed_loss=False",
        "--set", "use_tdir_anchor_loss=False",
        "--set", "w_tdir_anchor=0.0",
        "--set", "use_seq_turn_loss=False",
        "--set", "use_seq_turn_chain_loss=False",
        "--set", "use_odom_chain_len_loss=False",
        "--set", "odom_chain_len_loss_w=0.0",
        "--set", "use_odom_chain_vec_loss=False",
        "--set", "odom_chain_vec_loss_w=0.0",
        "--set", "num_workers=0",
        "--set", "pin_memory=False",
        "--set", "persistent_workers=False",
        "--set", "amp=False",
        "--set", "batch_size=1",
        "--set", "grad_accum=1",
        "--set", "log_every=20",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {"ran": False, "error": f"train_mvp failed, see {run_dir / 'train_stdout.log'}"}
    final_ckpt = run_dir / "final.pt"
    wrapped, cfg, load_summary, _freeze, _policy = _build_training_path_model(device, ckpt_path=final_ckpt)
    proxy = _proxy_metrics(wrapped, ds_val, device, max_windows=128)
    out_dir = run_dir / "wrapped_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    odom = eval_odometry_sequence(wrapped, ds_val, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    return {
        "ran": True,
        "run_dir": str(run_dir),
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "train_summary_last_eval": dict(summary.get("last_eval", {})),
        "proxy": proxy,
        "wrapped_odom_ATE": _safe_float(odom.get("odom_metric_ATE")),
        "wrapped_odom_drift": _safe_float(odom.get("odom_metric_drift")),
        "wrapped_odom_path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
    }


def _classify(zero_update: Dict[str, Any], drift: Dict[str, Any], one_a: Dict[str, Any], one_b: Dict[str, Any]) -> Tuple[str, bool]:
    if bool(zero_update["wrapper_still_mismatch"]):
        return "WRAPPER-STILL-MISMATCH", False
    drift_bad = _safe_float(drift["eval_vs_train_R_diff"]) >= 0.05 or _safe_float(drift["eval_vs_train_tvec_diff"]) >= 0.01
    one_step_bad = max(_safe_float(one_a["R_delta"]), _safe_float(one_a["tvec_delta"]), _safe_float(one_b["R_delta"]), _safe_float(one_b["tvec_delta"])) > 0.05
    if drift_bad and one_step_bad:
        return "HARNESS-FIX-PARTIAL", False
    if drift_bad:
        return "DROPOUT-DRIFT-UNRESOLVED", False
    if one_step_bad:
        return "ONE-STEP-STILL-DESTRUCTIVE", False
    return "HARNESS-FIX-PASS", True


def _write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# S15c Clean Policy Wrapped Training Harness Fix Report",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- S15 can continue: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
        "",
        "## Policy wrapper audit",
        f"- policy path: `{payload['policy_wrapper_audit']['policy_path']}`",
        f"- base policy path: `{payload['policy_wrapper_audit']['base_policy_path']}`",
        f"- base checkpoint path: `{payload['policy_wrapper_audit']['base_checkpoint_path']}`",
        f"- fine rot/tdir/tmag: `{payload['policy_wrapper_audit']['fine_rot']}` / `{payload['policy_wrapper_audit']['fine_tdir']}` / `{payload['policy_wrapper_audit']['fine_tmag']}`",
        f"- S5 thresholds: `{payload['policy_wrapper_audit']['thresholds']}`",
        f"- S5 scales: `{payload['policy_wrapper_audit']['scales']}`",
        f"- dt-anchor apply after fix: `{payload['policy_wrapper_audit']['training_dt_anchor_apply']}`",
        f"- policy wrapping enabled after fix: `{payload['policy_wrapper_audit']['training_policy_enabled']}`",
        f"- restore enabled after fix: `{payload['policy_wrapper_audit']['training_restore_enabled']}`",
        "",
        "## Dropout train/eval drift after fix",
        f"- dropout modules: `{payload['dropout_drift']['dropout_modules']}`",
        f"- dropout frozen eval: `{payload['dropout_drift']['dropout_frozen_eval']}`",
        f"- dropout trainable train: `{payload['dropout_drift']['dropout_trainable_train']}`",
        f"- eval_vs_train_R_diff: `{_fmt(payload['dropout_drift']['eval_vs_train_R_diff'])}`",
        f"- eval_vs_train_tvec_diff: `{_fmt(payload['dropout_drift']['eval_vs_train_tvec_diff'])}`",
        f"- eval_vs_train_tmag_diff: `{_fmt(payload['dropout_drift']['eval_vs_train_tmag_diff'])}`",
        "",
        "## Zero-update wrapped audit",
        f"- direct missing/unexpected: `{payload['zero_update']['direct_missing']} / {payload['zero_update']['direct_unexpected']}`",
        f"- training missing/unexpected: `{payload['zero_update']['training_missing']} / {payload['zero_update']['training_unexpected']}`",
        f"- wrapped R diff: `{_fmt(payload['zero_update']['R_diff'])}`",
        f"- wrapped tvec diff: `{_fmt(payload['zero_update']['tvec_diff'])}`",
        f"- wrapped tmag diff: `{_fmt(payload['zero_update']['tmag_diff'])}`",
        "",
        "## One-update tiny audit",
        f"- lr=`5e-06`: R_delta=`{_fmt(payload['one_update_lr']['R_delta'])}`, tvec_delta=`{_fmt(payload['one_update_lr']['tvec_delta'])}`, tmag_delta=`{_fmt(payload['one_update_lr']['tmag_delta'])}`, param_delta_l2=`{_fmt(payload['one_update_lr']['parameter_delta_l2'])}`",
        f"- lr=`5e-07`: R_delta=`{_fmt(payload['one_update_low_lr']['R_delta'])}`, tvec_delta=`{_fmt(payload['one_update_low_lr']['tvec_delta'])}`, tmag_delta=`{_fmt(payload['one_update_low_lr']['tmag_delta'])}`, param_delta_l2=`{_fmt(payload['one_update_low_lr']['parameter_delta_l2'])}`",
        "",
        "## Tiny smoke after fix",
    ]
    smoke = payload["tiny_smoke"]
    if smoke.get("ran"):
        lines.extend(
            [
                f"- val_ate_proxy: `{_fmt(smoke['proxy']['val_ate_proxy'])}`",
                f"- val_path_proxy: `{_fmt(smoke['proxy']['val_path_proxy'])}`",
                f"- wrapped odom ATE: `{_fmt(smoke['wrapped_odom_ATE'])}`",
                f"- wrapped odom drift: `{_fmt(smoke['wrapped_odom_drift'])}`",
                f"- wrapped odom path_ratio: `{_fmt(smoke['wrapped_odom_path_ratio'])}`",
                f"- run dir: `{smoke['run_dir']}`",
            ]
        )
    else:
        lines.append(f"- tiny smoke failed to run: `{smoke.get('error', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Decision",
            f"- final classification: `{payload['final_classification']}`",
            f"- whether S15 can continue: `{payload['s15_can_continue']}`",
            "- next step: `S15d tiny trajectory retest` only if this harness-fix pass is accepted; otherwise stop S15.",
            "- S5 remains final clean candidate: `yes`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# S15c Clean Policy Wrapped Training Harness Fix Summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- policy wrapper audit: training_policy_enabled=`{payload['policy_wrapper_audit']['training_policy_enabled']}`, dt_anchor_apply=`{payload['policy_wrapper_audit']['training_dt_anchor_apply']}`",
        f"- dropout drift after fix: R_diff=`{_fmt(payload['dropout_drift']['eval_vs_train_R_diff'])}`, tvec_diff=`{_fmt(payload['dropout_drift']['eval_vs_train_tvec_diff'])}`",
        f"- zero-update wrapped result: R_diff=`{_fmt(payload['zero_update']['R_diff'])}`, tvec_diff=`{_fmt(payload['zero_update']['tvec_diff'])}`, tmag_diff=`{_fmt(payload['zero_update']['tmag_diff'])}`",
        f"- one-update result current-lr: R_delta=`{_fmt(payload['one_update_lr']['R_delta'])}`, tvec_delta=`{_fmt(payload['one_update_lr']['tvec_delta'])}`, tmag_delta=`{_fmt(payload['one_update_lr']['tmag_delta'])}`",
        f"- tiny smoke result: val_ate_proxy=`{_fmt(payload['tiny_smoke'].get('proxy', {}).get('val_ate_proxy'))}`, val_path_proxy=`{_fmt(payload['tiny_smoke'].get('proxy', {}).get('val_path_proxy'))}`",
        f"- S15 can continue: `{payload['s15_can_continue']}`",
        "- S5 remains final clean candidate: `yes`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = _baseline_gate()
    if not gate["passed"]:
        REPORT_PATH.write_text("# S15c Clean Policy Wrapped Training Harness Fix Report\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        SUMMARY_PATH.write_text("# S15c Clean Policy Wrapped Training Harness Fix Summary\n\n- final classification: `INCONCLUSIVE`\n", encoding="utf-8")
        return

    data_root = _ensure_subset_data_root()
    _fold_name, split_seed = FOLDS[0]
    ds_train = _build_eval_dataset(data_root, split_seed, "train", (1,))
    ds_val = _build_eval_dataset(data_root, split_seed, "test", (1,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s5 = _read_json(S5_POLICY_PATH)
    s2b = _read_json(REPO_ROOT / s5["base_policy_path"])
    cfg, policy_before, restore_summary, policy_after = _build_training_cfg()
    zero_update = _zero_update_audit(ds_val, device)
    dropout_drift = _dropout_drift_audit(ds_train, device)
    one_lr = _one_update_audit(ds_train, ds_val, device, 5.0e-6)
    one_low = _one_update_audit(ds_train, ds_val, device, 5.0e-7)
    smoke = _tiny_smoke(ds_val, data_root, device)
    final_classification, can_continue = _classify(zero_update, dropout_drift, one_lr, one_low)
    payload = {
        "baseline_gate": gate,
        "policy_wrapper_audit": {
            "policy_path": str(S5_POLICY_PATH),
            "base_policy_path": str(REPO_ROOT / s5["base_policy_path"]),
            "base_checkpoint_path": str(REPO_ROOT / s5["base_checkpoint_path"]),
            "fine_rot": float(s2b["fine_rot_fuse_strength"]),
            "fine_tdir": float(s2b["fine_tdir_fuse_strength"]),
            "fine_tmag": float(s2b["fine_tmag_fuse_strength"]),
            "thresholds": dict(s5["thresholds"]),
            "scales": dict(s5["scales"]),
            "training_policy_enabled": bool(policy_after.get("enabled", False)),
            "training_restore_enabled": bool(restore_summary.get("enabled", False)),
            "training_dt_anchor_apply": bool(getattr(cfg, "dt_bucket_scale_anchor_apply", False)),
            "policy_lineage": list(policy_after.get("policy_lineage", [])),
        },
        "dropout_drift": dropout_drift,
        "zero_update": zero_update,
        "one_update_lr": one_lr,
        "one_update_low_lr": one_low,
        "tiny_smoke": smoke,
        "final_classification": final_classification,
        "s15_can_continue": can_continue,
    }
    _write_report(payload)


if __name__ == "__main__":
    main()
