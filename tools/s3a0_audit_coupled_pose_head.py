#!/usr/bin/env python3
"""Smoke-audit the S3a0 coupled pose residual head on top of the S2b base policy."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from train_mvp import _set_train_coupled_pose_residual_only


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S3a0_coupled_pose_head_smoke_audit.md"
ALLOWED_BASE_MISSING = {
    "coarse.mag_head.ridge_calib_raw_center",
    "fine.mag_head.ridge_calib_raw_center",
}


def _cfg_from_ckpt(ckpt_path: Path) -> Config:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg payload type: {type(cfg_dict)}")
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _load_model(model: PanoramaRelPoseModel, ckpt_path: Path, device: torch.device) -> Dict[str, List[str]]:
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no state_dict: {ckpt_path}")
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def _build_sample(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
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
    return {
        "IA": sample["IA"].unsqueeze(0).to(device),
        "IB": sample["IB"].unsqueeze(0).to(device),
        "dt_world": torch.tensor([float(sample.get("t_gt_mag", 0.01))], device=device, dtype=torch.float32),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_base = _cfg_from_ckpt(ckpt_path)
    cfg_base.use_fine_stage = True
    cfg_base.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg_base.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg_base.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg_base.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    cfg_base.tmag_condition_on_dt = False

    sample = _build_sample(cfg_base, device)

    cfg_plain = Config(**vars(cfg_base))
    cfg_plain.use_coupled_pose_residual_head = False
    plain_model = PanoramaRelPoseModel(cfg_plain, device).to(device)
    plain_load = _load_model(plain_model, ckpt_path, device)
    plain_model.eval()

    cfg_coupled = Config(**vars(cfg_base))
    cfg_coupled.use_coupled_pose_residual_head = True
    cfg_coupled.coupled_pose_residual_trainable = False
    cfg_coupled.coupled_pose_residual_gate_init = -4.0
    coupled_model = PanoramaRelPoseModel(cfg_coupled, device).to(device)
    coupled_load = _load_model(coupled_model, ckpt_path, device)
    coupled_model.eval()

    with torch.no_grad():
        R_plain, t_plain, aux_plain = plain_model(
            sample["IA"], sample["IB"], enable_depth_fusion=True, dt_world=sample["dt_world"]
        )
        R_coupled, t_coupled, aux_coupled = coupled_model(
            sample["IA"], sample["IB"], enable_depth_fusion=True, dt_world=sample["dt_world"]
        )

    coupled_param_names = [name for name, _ in coupled_model.named_parameters() if name.startswith("coupled_pose_head.")]
    tmag_diff = _max_abs_diff(aux_coupled["tmag_before_coupled"], aux_coupled["tmag_after_coupled"])
    rot_diff = _max_abs_diff(R_plain, R_coupled)
    tdir_diff = _max_abs_diff(aux_plain["t_dir_out"], aux_coupled["t_dir_out"])
    tmag_baseline_diff = _max_abs_diff(aux_plain["t_mag"], aux_coupled["t_mag"])
    residual_rot_mean = float(aux_coupled["coupled_delta_rot_norm"].detach().float().mean().cpu())
    residual_tdir_mean = float(aux_coupled["coupled_delta_tdir_norm"].detach().float().mean().cpu())
    gate_mean = float(aux_coupled["coupled_gate_mean"].detach().float().cpu())

    cfg_train = Config(**vars(cfg_coupled))
    cfg_train.coupled_pose_residual_trainable = True
    cfg_train.train_coupled_pose_residual_only = True
    model_train = PanoramaRelPoseModel(cfg_train, device).to(device)
    _load_model(model_train, ckpt_path, device)
    freeze_summary = _set_train_coupled_pose_residual_only(model_train, cfg_train)
    trainable_names = list(freeze_summary.get("trainable_names", []))
    optimizer_only_coupled = all(name.startswith("coupled_pose_head.") for name in trainable_names)

    normalized_missing = [name for name in coupled_load["missing"] if name not in ALLOWED_BASE_MISSING]
    missing_coupled_only = all(name.startswith("coupled_pose_head.") for name in normalized_missing)
    unexpected_empty = len(coupled_load["unexpected"]) == 0
    exact_baseline_match = rot_diff <= 1.0e-9 and tdir_diff <= 1.0e-9 and tmag_baseline_diff <= 1.0e-9
    audit_pass = (
        len(coupled_param_names) > 0
        and missing_coupled_only
        and unexpected_empty
        and tmag_diff <= 1.0e-9
        and exact_baseline_match
        and optimizer_only_coupled
        and len(freeze_summary.get("forbidden_trainable_params", [])) == 0
    )

    lines = [
        "# S3a0 Coupled Pose Head Smoke Audit",
        "",
        f"- policy: `{POLICY_PATH}`",
        f"- base checkpoint: `{ckpt_path}`",
        f"- device: `{device}`",
        f"- audit_pass: `{audit_pass}`",
        "",
        "## Load Summary",
        "",
        f"- plain missing={len(plain_load['missing'])} unexpected={len(plain_load['unexpected'])} skipped={len(plain_load['skipped'])}",
        f"- coupled missing={len(coupled_load['missing'])} unexpected={len(coupled_load['unexpected'])} skipped={len(coupled_load['skipped'])}",
        f"- coupled missing preview: `{coupled_load['missing'][:8]}`",
        f"- allowed base missing: `{sorted(ALLOWED_BASE_MISSING)}`",
        f"- coupled unexpected preview: `{coupled_load['unexpected'][:8]}`",
        "",
        "## Coupled Head Presence",
        "",
        f"- coupled param count: `{len(coupled_param_names)}`",
        f"- coupled param preview: `{coupled_param_names[:8]}`",
        "",
        "## Forward Sanity",
        "",
        f"- baseline vs coupled `R` max abs diff: `{rot_diff:.12f}`",
        f"- baseline vs coupled `t_dir_out` max abs diff: `{tdir_diff:.12f}`",
        f"- baseline vs coupled `t_mag` max abs diff: `{tmag_baseline_diff:.12f}`",
        f"- `tmag_before_coupled` vs `tmag_after_coupled` max abs diff: `{tmag_diff:.12f}`",
        f"- residual rot norm mean: `{residual_rot_mean:.12f}`",
        f"- residual tdir norm mean: `{residual_tdir_mean:.12f}`",
        f"- gate mean: `{gate_mean:.12f}`",
        "",
        "## Freeze / Optimizer Audit",
        "",
        f"- trainable_param_count: `{freeze_summary.get('trainable_param_count', 0)}`",
        f"- frozen_param_count: `{freeze_summary.get('frozen_param_count', 0)}`",
        f"- optimizer_param_count: `{freeze_summary.get('optimizer_param_count', 0)}`",
        f"- optimizer only coupled head: `{optimizer_only_coupled}`",
        f"- forbidden trainable params: `{freeze_summary.get('forbidden_trainable_params', [])}`",
        f"- trainable names: `{trainable_names}`",
        "",
        "## Verdict",
        "",
        f"- missing keys limited to new coupled head params after allowed-base filtering: `{missing_coupled_only}`",
        f"- unexpected keys empty: `{unexpected_empty}`",
        f"- baseline output unchanged when coupled head is enabled at zero residual init: `{exact_baseline_match}`",
        f"- `tmag` preserved before/after coupled residual: `{tmag_diff <= 1.0e-9}`",
        f"- optimizer isolated to coupled head params: `{optimizer_only_coupled}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
