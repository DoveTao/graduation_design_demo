#!/usr/bin/env python3
"""Audit S3a1 rot-only residual head wiring on top of the S2b clean policy."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from train_mvp import _apply_dt_bucket_scale_anchor_policy, _set_train_coupled_pose_residual_only, eval_model, eval_odometry_sequence


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
REFERENCE_SUMMARY_PATH = REPO_ROOT / "checkpoints" / "S2b_final_repro" / "s1d5_policy_eval_summary.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S3a1_rot_only_residual_audit_report.md"
TMP_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s3a1_rot_only_audit"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _cfg_from_ckpt(ckpt_path: Path) -> Config:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _load_model(model: PanoramaRelPoseModel, ckpt_path: Path, device: torch.device) -> Dict[str, List[str]]:
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected), "skipped": skipped}


def _build_dataset(cfg: Config) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split="test",
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=float(cfg.eval_max_dt),
    )


def _build_loader(cfg: Config, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        drop_last=False,
    )


def _sample_batch(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split="train",
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=float(cfg.eval_max_dt),
    )
    sample = ds[0]
    dt_val = float(sample.get("dt_world", sample.get("t_gt_mag", 0.01)))
    return {
        "IA": sample["IA"].unsqueeze(0).to(device),
        "IB": sample["IB"].unsqueeze(0).to(device),
        "dt_world": torch.tensor([dt_val], device=device, dtype=torch.float32),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def _eval_variant(name: str, model: torch.nn.Module, cfg: Config, ds, loader, device: torch.device, load_summary: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = TMP_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        model, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    sample = _sample_batch(cfg, device)
    with torch.no_grad():
        _R, _t, aux = model(sample["IA"], sample["IB"], enable_depth_fusion=True, dt_world=sample["dt_world"])
    payload = {
        "variant": name,
        "rot": float(rot),
        "tdir_abs": float(tdir_abs),
        "tdir_local_A_abs": float(tdir_local_A_abs),
        "drift": _safe_float(odom.get("odom_metric_drift")),
        "ATE": _safe_float(odom.get("odom_metric_ATE")),
        "path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        "load_missing": len(load_summary.get("missing", [])),
        "load_unexpected": len(load_summary.get("unexpected", [])),
        "tmag_before_after_max_diff": float(aux.get("tmag_before_after_max_diff", torch.zeros((), device=device)).detach().float().cpu()),
        "tdir_before_after_max_diff": float(aux.get("tdir_before_after_max_diff", torch.zeros((), device=device)).detach().float().cpu()),
        "delta_rot_norm_mean": float(aux.get("delta_rot_norm", torch.zeros((1,), device=device)).detach().float().mean().cpu()),
        "delta_tdir_norm_mean": float(aux.get("delta_tdir_norm", torch.zeros((1,), device=device)).detach().float().mean().cpu()),
        "gate_mean": float(aux.get("gate_mean", torch.zeros((), device=device)).detach().float().cpu()),
        "selected_k": int(odom.get("odom_selected_k", -1)),
        "num_pairs": int(odom.get("odom_num_pairs", 0)),
        "num_chains": int(odom.get("odom_num_chains", 0)),
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    return payload


def main() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = _cfg_from_ckpt(ckpt_path)
    base_cfg.dt_bucket_scale_anchor_policy_json = str(POLICY_PATH)
    _apply_dt_bucket_scale_anchor_policy(base_cfg)
    base_cfg.use_coupled_pose_residual_head = False

    rot_only_cfg = Config(**vars(base_cfg))
    rot_only_cfg.use_coupled_pose_residual_head = True
    rot_only_cfg.coupled_pose_residual_trainable = True
    rot_only_cfg.coupled_pose_residual_enable_rot = True
    rot_only_cfg.coupled_pose_residual_enable_tdir = False
    rot_only_cfg.coupled_pose_residual_force_tdir_zero = True
    rot_only_cfg.coupled_pose_residual_rot_scale = 0.02
    rot_only_cfg.coupled_pose_residual_tdir_scale = 0.0
    rot_only_cfg.coupled_pose_residual_gate_max = 0.0
    rot_only_cfg.train_coupled_pose_residual_only = True

    ds = _build_dataset(base_cfg)
    loader = _build_loader(base_cfg, ds)

    baseline_model = PanoramaRelPoseModel(base_cfg, device).to(device)
    baseline_load = _load_model(baseline_model, ckpt_path, device)
    baseline_model.eval()

    rot_only_model = PanoramaRelPoseModel(rot_only_cfg, device).to(device)
    rot_only_load = _load_model(rot_only_model, ckpt_path, device)
    rot_only_model.eval()

    rot_only_train_model = PanoramaRelPoseModel(rot_only_cfg, device).to(device)
    _load_model(rot_only_train_model, ckpt_path, device)
    freeze_summary = _set_train_coupled_pose_residual_only(rot_only_train_model, rot_only_cfg)
    trainable_names = list(freeze_summary.get("trainable_names", []))
    optimizer_only_rot_gate = all(
        name.startswith(("coupled_pose_head.backbone.", "coupled_pose_head.rot_head.", "coupled_pose_head.gate_head."))
        for name in trainable_names
    )
    no_tdir_trainable = all("tdir_head" not in name for name in trainable_names)

    baseline = _eval_variant("policy_aligned_baseline", baseline_model, base_cfg, ds, loader, device, baseline_load)
    rot_gate_zero = _eval_variant("rot_only_gate_zero", rot_only_model, rot_only_cfg, ds, loader, device, rot_only_load)

    ref_drift = float(ref["drift"])
    ref_ate = float(ref["ATE"])
    ref_path = float(ref["metric_path_ratio"])
    audit_pass = (
        abs(baseline["drift"] - ref_drift) <= 1.0e-6
        and abs(baseline["ATE"] - ref_ate) <= 1.0e-6
        and abs(baseline["path_ratio"] - ref_path) <= 1.0e-6
        and abs(rot_gate_zero["drift"] - ref_drift) <= 1.0e-6
        and abs(rot_gate_zero["ATE"] - ref_ate) <= 1.0e-6
        and abs(rot_gate_zero["path_ratio"] - ref_path) <= 1.0e-6
        and rot_gate_zero["tdir_before_after_max_diff"] <= 1.0e-9
        and rot_gate_zero["tmag_before_after_max_diff"] <= 1.0e-9
        and optimizer_only_rot_gate
        and no_tdir_trainable
        and int(freeze_summary.get("optimizer_param_count", 0)) == int(freeze_summary.get("trainable_param_count", 0))
        and int(rot_only_load["unexpected"].__len__()) == 0
    )

    lines = [
        "# S3a1 Rot-Only Residual Audit Report",
        "",
        f"- policy: `{POLICY_PATH}`",
        f"- base checkpoint: `{ckpt_path}`",
        f"- audit_pass: `{audit_pass}`",
        "",
        "## Baseline",
        "",
        f"- drift: `{baseline['drift']:.6f}`",
        f"- ATE: `{baseline['ATE']:.6f}`",
        f"- path_ratio: `{baseline['path_ratio']:.6f}`",
        "",
        "## Rot-Only Gate-Zero",
        "",
        f"- drift: `{rot_gate_zero['drift']:.6f}`",
        f"- ATE: `{rot_gate_zero['ATE']:.6f}`",
        f"- path_ratio: `{rot_gate_zero['path_ratio']:.6f}`",
        f"- delta_rot_norm_mean: `{rot_gate_zero['delta_rot_norm_mean']:.12f}`",
        f"- delta_tdir_norm_mean: `{rot_gate_zero['delta_tdir_norm_mean']:.12f}`",
        f"- gate_mean: `{rot_gate_zero['gate_mean']:.12f}`",
        f"- tdir_before_after_max_diff: `{rot_gate_zero['tdir_before_after_max_diff']:.12f}`",
        f"- tmag_before_after_max_diff: `{rot_gate_zero['tmag_before_after_max_diff']:.12f}`",
        "",
        "## Optimizer Audit",
        "",
        f"- trainable_param_count: `{freeze_summary.get('trainable_param_count', 0)}`",
        f"- frozen_param_count: `{freeze_summary.get('frozen_param_count', 0)}`",
        f"- optimizer_param_count: `{freeze_summary.get('optimizer_param_count', 0)}`",
        f"- optimizer only rot residual/gate: `{optimizer_only_rot_gate}`",
        f"- no tdir residual params trainable: `{no_tdir_trainable}`",
        f"- trainable_names: `{trainable_names}`",
        "",
        "## Load Summary",
        "",
        f"- baseline missing/unexpected: `{len(baseline_load['missing'])}/{len(baseline_load['unexpected'])}`",
        f"- rot-only missing/unexpected: `{len(rot_only_load['missing'])}/{len(rot_only_load['unexpected'])}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
