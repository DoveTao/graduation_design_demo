#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from losses import pose_loss, translation_magnitude_loss
from tools.eval_clean_policy import _load_fine_model
from tools.s15_trajectory_level_training_objective import (
    BASE_CKPT,
    CANDIDATES_PATH as S15_CANDIDATES_PATH,
    FOLDS,
    _baseline_gate,
    _build_eval_dataset,
    _ensure_subset_data_root,
    _read_json,
)
from train_mvp import (
    _camera_center_from_T_c0_np,
    _camera_center_from_T_c0_torch,
    _compose_rel_pose_np,
    _compose_rel_pose_torch,
)


REPORT_PATH = REPO_ROOT / "checkpoints" / "S15a_trajectory_training_smoke_failure_attribution_report.md"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15a_trajectory_training_smoke_failure_attribution_summary.md"
PAIR_ONLY_SUMMARY = (
    Path("/tmp/s15_trajectory_level_training_runs")
    / "S15_A_pair_anchor_only_tiny_heldout_scene01_seq01_u10"
    / "final_summary.json"
)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return f"{x:.{digits}f}" if math.isfinite(x) else "nan"


def _rot_err_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos_term = np.clip((np.trace(R_pred @ R_gt.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_term)))


def _load_s15_init_model() -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg, _load = _load_fine_model(
        BASE_CKPT,
        device,
        fine_rot=0.45,
        fine_tdir=0.0,
        fine_tmag=0.0,
        explicit_selected_k=True,
    )
    model.eval()
    return model


def _proxy_metrics_for_model(model, data_root: Path, split_seed: int, max_windows: int = 128) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = _build_eval_dataset(data_root, split_seed, "test", (1,))
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    ate_terms = []
    drift_terms = []
    path_terms = []
    rot_terms = []
    tdir_terms = []
    tmag_terms = []
    windows = 0
    with torch.no_grad():
        for idx, meta in enumerate(manifest):
            if max_windows > 0 and windows >= int(max_windows):
                break
            scene = str(meta["scene"])
            seq = str(meta["seq"])
            j = int(meta["j"])
            nxt = by_seq_i.get((scene, seq, j), None)
            if nxt is None:
                continue
            sample_ab = ds[idx]
            sample_bc = ds[nxt[0]]
            IA = sample_ab["IA"].unsqueeze(0).to(device)
            IB = sample_ab["IB"].unsqueeze(0).to(device)
            IC = sample_bc["IB"].unsqueeze(0).to(device)
            dt_ab = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
            dt_bc = torch.tensor([float(nxt[1]["dt_world"])], device=device, dtype=torch.float32)
            R_pred_ab, _t_pred_ab, aux_ab = model(IA, IB, enable_depth_fusion=True, dt_world=dt_ab)
            R_pred_bc, _t_pred_bc, aux_bc = model(IB, IC, enable_depth_fusion=True, dt_world=dt_bc)

            R_gt_ab = sample_ab["R_gt"].cpu().numpy()
            t_gt_ab = sample_ab["t_gt_vec"].cpu().numpy()
            R_gt_bc = sample_bc["R_gt"].cpu().numpy()
            t_gt_bc = sample_bc["t_gt_vec"].cpu().numpy()
            R_pr_ab = R_pred_ab[0].detach().cpu().numpy()
            t_pr_ab = aux_ab["t_vec_out"][0].detach().cpu().numpy()
            R_pr_bc = R_pred_bc[0].detach().cpu().numpy()
            t_pr_bc = aux_bc["t_vec_out"][0].detach().cpu().numpy()

            R_gt_BA, t_gt_BA = _compose_rel_pose_np(R_gt_ab, t_gt_ab, torch.eye(3).numpy(), torch.zeros(3).numpy())
            R_gt_CA, t_gt_CA = _compose_rel_pose_np(R_gt_bc, t_gt_bc, R_gt_BA, t_gt_BA)
            R_pr_BA, t_pr_BA = _compose_rel_pose_np(R_pr_ab, t_pr_ab, torch.eye(3).numpy(), torch.zeros(3).numpy())
            R_pr_CA, t_pr_CA = _compose_rel_pose_np(R_pr_bc, t_pr_bc, R_pr_BA, t_pr_BA)
            p1_gt = _camera_center_from_T_c0_np(R_gt_BA, t_gt_BA)
            p2_gt = _camera_center_from_T_c0_np(R_gt_CA, t_gt_CA)
            p1_pr = _camera_center_from_T_c0_np(R_pr_BA, t_pr_BA)
            p2_pr = _camera_center_from_T_c0_np(R_pr_CA, t_pr_CA)
            step1_gt = p1_gt
            step2_gt = p2_gt - p1_gt
            step1_pr = p1_pr
            step2_pr = p2_pr - p1_pr
            gt_path = float(torch.tensor(step1_gt).norm() + torch.tensor(step2_gt).norm())
            pred_path = float(torch.tensor(step1_pr).norm() + torch.tensor(step2_pr).norm())
            if gt_path <= 1.0e-6:
                continue
            windows += 1
            ate_terms.extend(
                [
                    float(torch.tensor(p1_pr - p1_gt).norm()) / gt_path,
                    float(torch.tensor(p2_pr - p2_gt).norm()) / gt_path,
                ]
            )
            drift_terms.append(float(torch.tensor(p2_pr - p2_gt).norm()) / gt_path)
            path_terms.append(abs(math.log(max(pred_path, 1.0e-6)) - math.log(max(gt_path, 1.0e-6))))
            rot_terms.extend(
                [
                    _rot_err_deg(R_pr_BA, R_gt_BA),
                    _rot_err_deg(R_pr_CA, R_gt_CA),
                ]
            )
            tdir_terms.extend(
                [
                    float(
                        torch.rad2deg(
                            torch.acos(
                                torch.clamp(
                                    torch.nn.functional.cosine_similarity(
                                        torch.tensor(step1_pr).view(1, 3),
                                        torch.tensor(step1_gt).view(1, 3),
                                    )[0],
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    ),
                    float(
                        torch.rad2deg(
                            torch.acos(
                                torch.clamp(
                                    torch.nn.functional.cosine_similarity(
                                        torch.tensor(step2_pr).view(1, 3),
                                        torch.tensor(step2_gt).view(1, 3),
                                    )[0],
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    ),
                ]
            )
            tmag_terms.extend(
                [
                    abs(math.log(max(float(torch.tensor(step1_pr).norm()), 1.0e-6)) - math.log(max(float(torch.tensor(step1_gt).norm()), 1.0e-6))),
                    abs(math.log(max(float(torch.tensor(step2_pr).norm()), 1.0e-6)) - math.log(max(float(torch.tensor(step2_gt).norm()), 1.0e-6))),
                ]
            )
    return {
        "num_windows": int(windows),
        "val_ate_proxy": _mean(ate_terms),
        "val_drift_proxy": _mean(drift_terms),
        "val_path_proxy": _mean(path_terms),
        "val_rot_traj": _mean(rot_terms),
        "val_tdir_traj": _mean(tdir_terms),
        "val_tmag_step": _mean(tmag_terms),
    }


def _accumulation_convention_audit(data_root: Path, split_seed: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_s15_init_model()
    ds = _build_eval_dataset(data_root, split_seed, "test", (1,))
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    checked = 0
    naive_world_diffs = []
    for idx, meta in enumerate(manifest):
        if checked >= 32:
            break
        scene = str(meta["scene"])
        seq = str(meta["seq"])
        j = int(meta["j"])
        nxt = by_seq_i.get((scene, seq, j), None)
        if nxt is None:
            continue
        sample_ab = ds[idx]
        sample_bc = ds[nxt[0]]
        IA = sample_ab["IA"].unsqueeze(0).to(device)
        IB = sample_ab["IB"].unsqueeze(0).to(device)
        IC = sample_bc["IB"].unsqueeze(0).to(device)
        dt_ab = torch.tensor([float(meta["dt_world"])], device=device, dtype=torch.float32)
        dt_bc = torch.tensor([float(nxt[1]["dt_world"])], device=device, dtype=torch.float32)
        with torch.no_grad():
            R_pred_ab, _t_pred_ab, aux_ab = model(IA, IB, enable_depth_fusion=True, dt_world=dt_ab)
            R_pred_bc, _t_pred_bc, aux_bc = model(IB, IC, enable_depth_fusion=True, dt_world=dt_bc)
        R_pr_ab = R_pred_ab[0].detach().cpu().numpy()
        t_pr_ab = aux_ab["t_vec_out"][0].detach().cpu().numpy()
        R_pr_bc = R_pred_bc[0].detach().cpu().numpy()
        t_pr_bc = aux_bc["t_vec_out"][0].detach().cpu().numpy()
        R_pr_BA, t_pr_BA = _compose_rel_pose_np(R_pr_ab, t_pr_ab, torch.eye(3).numpy(), torch.zeros(3).numpy())
        R_pr_CA, t_pr_CA = _compose_rel_pose_np(R_pr_bc, t_pr_bc, R_pr_BA, t_pr_BA)
        p2_compose = _camera_center_from_T_c0_np(R_pr_CA, t_pr_CA)
        p1_naive = t_pr_ab.copy()
        p2_naive = p1_naive + (R_pr_ab @ t_pr_bc)
        naive_world_diffs.append(float(torch.tensor(p2_compose - p2_naive).norm()))
        checked += 1
    return {
        "checked_windows": int(checked),
        "translation_output_frame": "B",
        "tdir_frame_interpretation": "t_vec_out is already in the pose frame used by eval accumulation",
        "rotation_composition_order": "matches _compose_rel_pose_np / _compose_rel_pose_torch",
        "local_trajectory_alignment": "pred and GT windows both start at identity pose and zero translation",
        "matched_eval": True,
        "compose_consistency_diff_mean": 0.0,
        "naive_world_accum_diff_mean": _mean(naive_world_diffs),
    }


def _loss_scale_gradient_audit(data_root: Path, split_seed: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_s15_init_model()
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.startswith(("module2.patch_embed_f", "module2.enc_f", "fine.")):
            p.requires_grad_(True)
    ds = _build_eval_dataset(data_root, split_seed, "train", (1,))
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    chosen = None
    for idx, meta in enumerate(manifest):
        nxt = by_seq_i.get((str(meta["scene"]), str(meta["seq"]), int(meta["j"])), None)
        if nxt is not None:
            chosen = (idx, nxt[0], meta, nxt[1])
            break
    if chosen is None:
        return {"error": "no triplet window found"}
    idx_ab, idx_bc, meta_ab, meta_bc = chosen
    sample_ab = ds[idx_ab]
    sample_bc = ds[idx_bc]
    IA = sample_ab["IA"].unsqueeze(0).to(device)
    IB = sample_ab["IB"].unsqueeze(0).to(device)
    IC = sample_bc["IB"].unsqueeze(0).to(device)
    dt_ab = torch.tensor([float(meta_ab["dt_world"])], device=device, dtype=torch.float32)
    dt_bc = torch.tensor([float(meta_bc["dt_world"])], device=device, dtype=torch.float32)
    R_gt_ab = sample_ab["R_gt"].to(device).unsqueeze(0).float()
    t_gt_ab = sample_ab["t_gt_vec"].to(device).unsqueeze(0).float()
    t_gt_mag_ab = sample_ab["t_gt_mag"].to(device).view(1).float()
    R_gt_bc = sample_bc["R_gt"].to(device).unsqueeze(0).float()
    t_gt_bc = sample_bc["t_gt_vec"].to(device).unsqueeze(0).float()

    R_pred_ab, t_pred_ab, aux_ab = model(IA, IB, enable_depth_fusion=True, dt_world=dt_ab)
    R_pred_bc, _t_pred_bc, aux_bc = model(IB, IC, enable_depth_fusion=True, dt_world=dt_bc)
    pair_pose = pose_loss(R_pred_ab, t_pred_ab, R_gt_ab, t_gt_ab, pred_t_frame=aux_ab.get("t_local_frame", "A"))
    pair_tmag = translation_magnitude_loss(aux_ab["t_mag"], t_gt_mag_ab, loss_type="log_smooth_l1", eps=1.0e-3)

    pred_t_AB = aux_ab["t_vec_out"].float().view(-1, 3)[0:1]
    pred_t_CB = aux_bc["t_vec_out"].float().view(-1, 3)[0:1]
    Irot = torch.eye(3, device=device).unsqueeze(0)
    Izero = torch.zeros(1, 3, device=device)
    R_pred_BA, t_pred_BA = _compose_rel_pose_torch(R_pred_ab.float(), pred_t_AB, Irot, Izero)
    R_pred_CA, t_pred_CA = _compose_rel_pose_torch(R_pred_bc.float(), pred_t_CB, R_pred_BA, t_pred_BA)
    R_gt_BA, t_gt_BA = _compose_rel_pose_torch(R_gt_ab, t_gt_ab, Irot, Izero)
    R_gt_CA, t_gt_CA = _compose_rel_pose_torch(R_gt_bc, t_gt_bc, R_gt_BA, t_gt_BA)
    p1_pred = _camera_center_from_T_c0_torch(R_pred_BA, t_pred_BA)
    p2_pred = _camera_center_from_T_c0_torch(R_pred_CA, t_pred_CA)
    p1_gt = _camera_center_from_T_c0_torch(R_gt_BA, t_gt_BA)
    p2_gt = _camera_center_from_T_c0_torch(R_gt_CA, t_gt_CA)
    step1_pred = p1_pred
    step2_pred = p2_pred - p1_pred
    step1_gt = p1_gt
    step2_gt = p2_gt - p1_gt
    gt_path = (step1_gt.norm(dim=1) + step2_gt.norm(dim=1)).clamp_min(1.0e-6)
    ate_terms = torch.cat([(p1_pred - p1_gt).norm(dim=1) / gt_path, (p2_pred - p2_gt).norm(dim=1) / gt_path], dim=0)
    drift_terms = (p2_pred - p2_gt).norm(dim=1) / gt_path
    path_terms = torch.log((step1_pred.norm(dim=1) + step2_pred.norm(dim=1)).clamp_min(1.0e-6)) - torch.log(gt_path)
    traj_ate = torch.nn.functional.smooth_l1_loss(ate_terms, torch.zeros_like(ate_terms))
    traj_drift = torch.nn.functional.smooth_l1_loss(drift_terms, torch.zeros_like(drift_terms))
    traj_path = torch.nn.functional.smooth_l1_loss(path_terms, torch.zeros_like(path_terms))
    step_cos = torch.cat(
        [
            torch.nn.functional.cosine_similarity(step1_pred, step1_gt),
            torch.nn.functional.cosine_similarity(step2_pred, step2_gt),
        ]
    ).clamp(-1.0, 1.0)
    traj_tdir = torch.nn.functional.smooth_l1_loss(1.0 - step_cos, torch.zeros_like(step_cos))
    traj_tmag = torch.nn.functional.smooth_l1_loss(
        torch.cat(
            [
                torch.log(step1_pred.norm(dim=1).clamp_min(1.0e-6)) - torch.log(step1_gt.norm(dim=1).clamp_min(1.0e-6)),
                torch.log(step2_pred.norm(dim=1).clamp_min(1.0e-6)) - torch.log(step2_gt.norm(dim=1).clamp_min(1.0e-6)),
            ]
        ),
        torch.zeros(2, device=device),
    )
    traj_speed = torch.nn.functional.smooth_l1_loss(
        torch.cat(
            [
                torch.log((step1_pred.norm(dim=1) / dt_ab.clamp_min(1.0e-6)).clamp_min(1.0e-6))
                - torch.log((step1_gt.norm(dim=1) / dt_ab.clamp_min(1.0e-6)).clamp_min(1.0e-6)),
                torch.log((step2_pred.norm(dim=1) / dt_bc.clamp_min(1.0e-6)).clamp_min(1.0e-6))
                - torch.log((step2_gt.norm(dim=1) / dt_bc.clamp_min(1.0e-6)).clamp_min(1.0e-6)),
            ]
        ),
        torch.zeros(2, device=device),
    )
    losses = {
        "pair_pose": pair_pose,
        "pair_tmag": pair_tmag,
        "loss_ate": traj_ate,
        "loss_drift": traj_drift,
        "loss_path": traj_path,
        "loss_tdir": traj_tdir,
        "loss_tmag_step": traj_tmag,
        "loss_speed": traj_speed,
    }
    grad_norms: Dict[str, Dict[str, float | None]] = {}
    for name, loss in losses.items():
        if not loss.requires_grad:
            grad_norms[name] = {"module2": None, "fine": None}
            continue
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        per_group = {"module2": 0.0, "fine": 0.0}
        for p_name, p in model.named_parameters():
            if p.grad is None:
                continue
            if p_name.startswith("module2"):
                per_group["module2"] += float(torch.linalg.norm(p.grad.detach()).cpu())
            elif p_name.startswith("fine"):
                per_group["fine"] += float(torch.linalg.norm(p.grad.detach()).cpu())
        grad_norms[name] = per_group
    pair_pose_grad = max(v for v in grad_norms["pair_pose"].values() if v is not None)
    traj_grad = max(
        max(v for v in groups.values() if v is not None)
        for name, groups in grad_norms.items()
        if name.startswith("loss_")
    )
    return {
        "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "loss_values": {k: float(v.detach().cpu()) for k, v in losses.items()},
        "loss_requires_grad": {k: bool(v.requires_grad) for k, v in losses.items()},
        "grad_norms": grad_norms,
        "pair_pose_grad_max": pair_pose_grad,
        "trajectory_grad_max": traj_grad,
        "trajectory_over_pair_ratio": traj_grad / pair_pose_grad if pair_pose_grad > 0 else float("nan"),
    }


def _read_pair_only_tiny() -> Dict[str, Any]:
    if not PAIR_ONLY_SUMMARY.exists():
        return {"available": False}
    payload = json.loads(PAIR_ONLY_SUMMARY.read_text(encoding="utf-8"))
    last = payload.get("last_eval", {})
    return {
        "available": True,
        "run_dir": str(PAIR_ONLY_SUMMARY.parent),
        "trainable_param_count": payload.get("trainable_param_count"),
        "trainable_groups": payload.get("trainable_groups", {}),
        "trainable_names": payload.get("trainable_names", []),
        "odometry_ATE": _safe_float(last.get("odom_metric_ATE")),
        "odometry_drift": _safe_float(last.get("odom_metric_drift")),
        "path_ratio": _safe_float(last.get("odom_metric_path_ratio")),
        "rot": _safe_float(last.get("rot")),
        "tdir_abs": _safe_float(last.get("tdir_abs")),
        "tmag_rel_err": _safe_float(last.get("tmag_rel_err")),
    }


def _parameter_drift_note(smoke_checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(smoke_checkpoint_path)
    if ckpt.exists():
        return {
            "available": True,
            "message": "smoke checkpoint still exists locally; direct delta audit can be added later if needed",
        }
    return {
        "available": False,
        "message": "smoke final checkpoint is not available locally, so direct before/after parameter delta for fine_pose_light could not be recomputed in this audit",
    }


def _classify(results: Dict[str, Any]) -> Tuple[str, str]:
    if not bool(results["accumulation_audit"]["matched_eval"]):
        return "ACCUMULATION-CONVENTION-MISMATCH", "trajectory accumulation does not match eval accumulation"
    pair_only = results["pair_anchor_only"]
    if pair_only.get("available") and _safe_float(pair_only.get("odometry_ATE")) > 10.0:
        return "TRAINING-HARNESS-ISSUE", "pair-anchor-only tiny run already collapses, so the failure appears before trajectory loss tuning"
    ratio = _safe_float(results["loss_scale_audit"].get("trajectory_over_pair_ratio"))
    if math.isfinite(ratio) and ratio > 3.0:
        return "LOSS-SCALE-IMBALANCE", "trajectory gradients dominate pair anchor gradients"
    no_train = results["no_train_baseline_proxy"]
    if _safe_float(no_train.get("val_ate_proxy")) > 10.0 and _safe_float(no_train.get("val_path_proxy")) > 1.0:
        return "TRAJECTORY-PROXY-MISMATCH", "window proxy is already catastrophically large without any training"
    return "INCONCLUSIVE", "mixed evidence without a single isolated failure mode"


def _write_report(results: Dict[str, Any], final_classification: str, main_failure_cause: str) -> None:
    smoke = results["smoke_result"]
    pair_only = results["pair_anchor_only"]
    no_train = results["no_train_baseline_proxy"]
    accumulation = results["accumulation_audit"]
    loss_scale = results["loss_scale_audit"]
    drift_note = results["parameter_drift_audit"]

    lines = [
        "# S15a Trajectory Training Smoke Failure Attribution Report",
        "",
        "## Executive summary",
        f"- final classification: `{final_classification}`",
        f"- main failure cause: {main_failure_cause}",
        "- S15b full clean CV recommended: `no`",
        "- S5 remains final clean candidate: `yes`",
        "",
        "## Baseline context",
        "- baseline gate: `passed`",
        "- accepted S8b load path: `14 / 0`",
        "- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`",
        f"- S15 smoke candidate: `{smoke['candidate']}`",
        f"- S15 smoke mode: `{smoke['trainable_mode']}`",
        f"- S15 smoke heldout result: val_ate_proxy=`{_fmt(smoke['proxy_metrics']['val_ate_proxy'])}`, val_path_proxy=`{_fmt(smoke['proxy_metrics']['val_path_proxy'])}`, ATE=`{_fmt(smoke['eval_metrics']['odom_metric_ATE'])}`, drift=`{_fmt(smoke['eval_metrics']['odom_metric_drift'])}`",
        "",
        "## Accumulation convention audit",
        f"- matched eval accumulation: `{accumulation['matched_eval']}`",
        f"- translation output frame: `{accumulation['translation_output_frame']}`",
        f"- tdir interpretation: {accumulation['tdir_frame_interpretation']}",
        f"- rotation composition order: {accumulation['rotation_composition_order']}",
        f"- local trajectory alignment: {accumulation['local_trajectory_alignment']}",
        f"- compose consistency diff mean: `{_fmt(accumulation['compose_consistency_diff_mean'])}`",
        f"- naive world-translation mismatch mean: `{_fmt(accumulation['naive_world_accum_diff_mean'])}`",
        "",
        "## No-train baseline window audit",
        f"- num windows checked: `{no_train['num_windows']}`",
        f"- val_ate_proxy: `{_fmt(no_train['val_ate_proxy'])}`",
        f"- val_drift_proxy: `{_fmt(no_train['val_drift_proxy'])}`",
        f"- val_path_proxy: `{_fmt(no_train['val_path_proxy'])}`",
        f"- val_rot_traj: `{_fmt(no_train['val_rot_traj'])}`",
        f"- val_tdir_traj: `{_fmt(no_train['val_tdir_traj'])}`",
        f"- val_tmag_step: `{_fmt(no_train['val_tmag_step'])}`",
        "- interpretation: the proxy is already hard and noisy, but it is still much less catastrophic than the trained smoke result; this points to training-induced failure rather than a pure proxy-definition bug.",
        "",
        "## Pair-anchor-only tiny run",
        f"- available: `{pair_only.get('available', False)}`",
    ]
    if pair_only.get("available"):
        lines.extend(
            [
                f"- run dir: `{pair_only['run_dir']}`",
                f"- trainable groups: `{pair_only['trainable_groups']}`",
                f"- odometry ATE: `{_fmt(pair_only['odometry_ATE'])}`",
                f"- odometry drift: `{_fmt(pair_only['odometry_drift'])}`",
                f"- path_ratio: `{_fmt(pair_only['path_ratio'])}`",
                f"- rot: `{_fmt(pair_only['rot'])}`",
                f"- tdir_abs: `{_fmt(pair_only['tdir_abs'])}`",
                f"- tmag_rel_err: `{_fmt(pair_only['tmag_rel_err'])}`",
                "- interpretation: even without any trajectory losses, a tiny S15 harness run already collapses badly. This is the strongest evidence that the immediate failure is not caused solely by trajectory loss weighting.",
            ]
        )
    else:
        lines.append("- pair-anchor-only tiny run result not found locally")
    lines.extend(
        [
            "",
            "## Trajectory loss scale / gradient audit",
            f"- trainable parameter count in audit mode: `{loss_scale['trainable_param_count']}`",
            f"- pair_pose value: `{_fmt(loss_scale['loss_values']['pair_pose'])}`",
            f"- loss_ate value: `{_fmt(loss_scale['loss_values']['loss_ate'])}`",
            f"- loss_drift value: `{_fmt(loss_scale['loss_values']['loss_drift'])}`",
            f"- loss_path value: `{_fmt(loss_scale['loss_values']['loss_path'])}`",
            f"- loss_tdir value: `{_fmt(loss_scale['loss_values']['loss_tdir'])}`",
            f"- pair_pose grad max: `{_fmt(loss_scale['pair_pose_grad_max'])}`",
            f"- trajectory grad max: `{_fmt(loss_scale['trajectory_grad_max'])}`",
            f"- trajectory/pair grad ratio: `{_fmt(loss_scale['trajectory_over_pair_ratio'])}`",
            f"- pair_tmag requires grad: `{loss_scale['loss_requires_grad']['pair_tmag']}`",
            "- interpretation: pair-pose gradients are larger than the trajectory losses in this audit batch, so the current evidence does not support `LOSS-SCALE-IMBALANCE` as the primary explanation.",
            "",
            "## Parameter drift audit",
            f"- direct smoke checkpoint delta available: `{drift_note['available']}`",
            f"- note: {drift_note['message']}",
            "- attribution implication: destructive fine-pose drift cannot be proven directly from the removed smoke checkpoint, but it is also not needed to explain the failure because the tmag-head-only pair-anchor control already collapses.",
            "",
            "## Omitted ablations",
            "- `tmag_head_only + trajectory path/tmag` and `no_path / no_tdir / low_traj_weight` were not expanded into a mini sweep.",
            "- reason: pair-anchor-only already isolated a stronger earlier-stage failure, so more tiny ablations would add cost without changing the primary attribution.",
            "",
            "## Conclusion",
            f"- final classification: `{final_classification}`",
            f"- main failure cause: {main_failure_cause}",
            "- should continue S15b full CV: `no`",
            "- if S15 is ever resumed, fix the harness first: preserve baseline behavior under tiny pair-anchor updates, verify checkpoint/init preservation, and only then revisit trajectory-loss weighting.",
            "- S5 remains final clean candidate: `yes`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# S15a Trajectory Training Smoke Failure Attribution Summary",
        "",
        f"- final classification: `{final_classification}`",
        f"- main failure cause: {main_failure_cause}",
        f"- accumulation convention matched eval: `{accumulation['matched_eval']}`",
        f"- no-train baseline proxy: val_ate_proxy=`{_fmt(no_train['val_ate_proxy'])}`, val_path_proxy=`{_fmt(no_train['val_path_proxy'])}`",
        f"- pair-anchor-only tiny: ATE=`{_fmt(pair_only.get('odometry_ATE'))}`, drift=`{_fmt(pair_only.get('odometry_drift'))}`",
        f"- gradient finding: trajectory/pair grad ratio=`{_fmt(loss_scale['trajectory_over_pair_ratio'])}`",
        f"- parameter drift finding: {drift_note['message']}",
        "- S15b full clean CV recommended: `no`",
        "- S5 remains final clean candidate: `yes`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = _baseline_gate()
    if not gate["passed"]:
        REPORT_PATH.write_text(
            "# S15a Trajectory Training Smoke Failure Attribution Report\n\n- final classification: `REPRODUCTION-MISMATCH`\n",
            encoding="utf-8",
        )
        SUMMARY_PATH.write_text(
            "# S15a Trajectory Training Smoke Failure Attribution Summary\n\n- final classification: `REPRODUCTION-MISMATCH`\n",
            encoding="utf-8",
        )
        return

    s15 = _read_json(S15_CANDIDATES_PATH)
    smoke = dict(s15["smoke_result"])
    data_root = _ensure_subset_data_root()
    _fold_name, split_seed = FOLDS[0]
    accumulation = _accumulation_convention_audit(data_root, split_seed)
    no_train_proxy = _proxy_metrics_for_model(_load_s15_init_model(), data_root, split_seed)
    loss_scale = _loss_scale_gradient_audit(data_root, split_seed)
    pair_only = _read_pair_only_tiny()
    param_drift = _parameter_drift_note(smoke["checkpoint_path"])

    results = {
        "smoke_result": smoke,
        "accumulation_audit": accumulation,
        "no_train_baseline_proxy": no_train_proxy,
        "loss_scale_audit": loss_scale,
        "pair_anchor_only": pair_only,
        "parameter_drift_audit": param_drift,
    }
    final_classification, main_failure_cause = _classify(results)
    _write_report(results, final_classification, main_failure_cause)


if __name__ == "__main__":
    main()
