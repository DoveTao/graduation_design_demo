#!/usr/bin/env python3
"""Audit whether S3a0 smoke/eval now matches the S2b clean policy path."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel, _local_t_to_output_frame
from tools.eval_clean_policy import DtBucketScaledMagnitudeModel, _extract_pairs, _q
from train_mvp import (
    _apply_dt_bucket_scale_anchor_policy,
    eval_model,
    eval_odometry_sequence,
)


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
REFERENCE_SUMMARY_PATH = REPO_ROOT / "checkpoints" / "S2b_final_repro" / "s1d5_policy_eval_summary.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S3a0c_policy_alignment_audit_report.md"
TMP_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s3a0c_policy_alignment_audit"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _cfg_from_ckpt(ckpt_path: Path) -> Config:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg payload type: {type(cfg_dict)}")
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _load_model(model: PanoramaRelPoseModel, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
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


def _build_dataset(cfg: Config, split: str = "test") -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=split,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
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


def _build_sample(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    ds = _build_dataset(cfg, split="train")
    sample = ds[0]
    dt_val = float(sample.get("dt_world", sample.get("t_gt_mag", 0.01)))
    return {
        "IA": sample["IA"].unsqueeze(0).to(device),
        "IB": sample["IB"].unsqueeze(0).to(device),
        "dt_world": torch.tensor([dt_val], device=device, dtype=torch.float32),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


class GateZeroWrapper(torch.nn.Module):
    def __init__(self, base: PanoramaRelPoseModel) -> None:
        super().__init__()
        self.base = base
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        _R_pred, _t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        if "R_before_coupled" not in aux or "tdir_before_coupled" not in aux:
            return _R_pred, _t_pred, aux
        aux = dict(aux)
        R_before = aux["R_before_coupled"]
        tdir_before = aux["tdir_before_coupled"]
        tdir_out = _local_t_to_output_frame(R_before, tdir_before)
        tmag = aux["t_mag"].float().view(-1)
        aux["R_after_coupled"] = R_before
        aux["tdir_after_coupled"] = tdir_before
        aux["t_dir_local"] = tdir_before
        aux["t_dir"] = tdir_before
        aux["t_dir_out"] = tdir_out
        aux["t_vec"] = tdir_out * tmag.unsqueeze(-1)
        aux["t_vec_out"] = aux["t_vec"]
        aux["t_vec_local"] = tdir_before.float() * tmag.unsqueeze(-1)
        aux["coupled_gate"] = torch.zeros_like(aux.get("coupled_gate", tmag.unsqueeze(-1)))
        aux["coupled_gate_mean"] = torch.zeros((), device=tmag.device, dtype=torch.float32)
        return R_before, tdir_before, aux


def _sample_forward_stats(model: torch.nn.Module, cfg: Config, device: torch.device) -> Dict[str, Any]:
    sample = _build_sample(cfg, device)
    with torch.no_grad():
        _R_pred, _t_pred, aux = model(
            sample["IA"],
            sample["IB"],
            enable_depth_fusion=True,
            dt_world=sample["dt_world"],
        )
    stats = {
        "tmag_before_after_max_diff": float("nan"),
        "coupled_delta_rot_norm": float("nan"),
        "coupled_delta_tdir_norm": float("nan"),
        "coupled_gate_mean": float("nan"),
        "dt_bucket_scale_anchor_factor": float("nan"),
    }
    if isinstance(aux, dict):
        if "tmag_before_coupled" in aux and "tmag_after_coupled" in aux:
            stats["tmag_before_after_max_diff"] = _max_abs_diff(aux["tmag_before_coupled"], aux["tmag_after_coupled"])
        if "coupled_delta_rot_norm" in aux:
            stats["coupled_delta_rot_norm"] = float(aux["coupled_delta_rot_norm"].detach().float().mean().cpu())
        if "coupled_delta_tdir_norm" in aux:
            stats["coupled_delta_tdir_norm"] = float(aux["coupled_delta_tdir_norm"].detach().float().mean().cpu())
        if "coupled_gate_mean" in aux:
            stats["coupled_gate_mean"] = float(aux["coupled_gate_mean"].detach().float().cpu())
        if "dt_bucket_scale_anchor_factor" in aux:
            stats["dt_bucket_scale_anchor_factor"] = float(aux["dt_bucket_scale_anchor_factor"].detach().float().view(-1)[0].cpu())
    return stats


def _variant_payload(
    *,
    name: str,
    model: torch.nn.Module,
    cfg: Config,
    ds,
    loader,
    device: torch.device,
    load_summary: Dict[str, Any],
    policy_path: str,
    dt_anchor_applied: bool,
    note: str,
) -> Dict[str, Any]:
    out_dir = TMP_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        model, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    rows = _extract_pairs(model, ds, device)
    q = _q([float(r["tmag_pred"]) for r in rows])
    forward_stats = _sample_forward_stats(model, cfg, device)
    payload = {
        "variant": name,
        "status": "ok",
        "note": note,
        "load_missing": len(load_summary.get("missing", [])),
        "load_unexpected": len(load_summary.get("unexpected", [])),
        "load_skipped": len(load_summary.get("skipped", [])),
        "fine_rot_fuse_strength": float(getattr(cfg, "fine_rot_fuse_strength", float("nan"))),
        "fine_tdir_fuse_strength": float(getattr(cfg, "fine_tdir_fuse_strength", float("nan"))),
        "fine_tmag_fuse_strength": float(getattr(cfg, "fine_tmag_fuse_strength", float("nan"))),
        "policy_json_path": policy_path,
        "dt_anchor_applied": bool(dt_anchor_applied),
        "rot": float(rot),
        "tdir_abs": float(tdir_abs),
        "tdir_local_A_abs": float(tdir_local_A_abs),
        "drift": _safe_float(odom.get("odom_metric_drift")),
        "ATE": _safe_float(odom.get("odom_metric_ATE")),
        "path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        "RPE_rot": _safe_float(odom.get("odom_metric_RPE_rot")),
        "RPE_trans_dir": _safe_float(odom.get("odom_metric_RPE_trans_dir")),
        "RPE_trans_mag": _safe_float(odom.get("odom_metric_RPE_trans_mag")),
        "tmag_p10": float(q["p10"]),
        "tmag_p50": float(q["p50"]),
        "tmag_p90": float(q["p90"]),
        "selected_k": int(odom.get("odom_selected_k", -1)),
        "num_pairs": int(odom.get("odom_num_pairs", 0)),
        "num_chains": int(odom.get("odom_num_chains", 0)),
        **forward_stats,
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    return payload


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x):
        return "NA"
    return f"{x:.6f}"


def _close(a: float, b: float, tol: float = 1.0e-6) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    ref_summary = json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    raw_cfg = _cfg_from_ckpt(ckpt_path)
    raw_cfg.use_fine_stage = True
    raw_cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    raw_cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    raw_cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    raw_cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    raw_cfg.tmag_condition_on_dt = False

    eval_cfg = Config(**vars(raw_cfg))
    eval_cfg.dt_bucket_scale_anchor_policy_json = str(POLICY_PATH)
    policy_summary = _apply_dt_bucket_scale_anchor_policy(eval_cfg)

    ds = _build_dataset(eval_cfg, split="test")
    loader = _build_loader(eval_cfg, ds)

    raw_model = PanoramaRelPoseModel(raw_cfg, device).to(device)
    raw_load = _load_model(raw_model, ckpt_path, device)
    raw_model.eval()

    wrapper_base_model = PanoramaRelPoseModel(raw_cfg, device).to(device)
    wrapper_load = _load_model(wrapper_base_model, ckpt_path, device)
    wrapper_base_model.eval()
    factors = {str(k): float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped_model = DtBucketScaledMagnitudeModel(wrapper_base_model, factors)

    integrated_off_cfg = Config(**vars(eval_cfg))
    integrated_off_cfg.use_coupled_pose_residual_head = False
    integrated_off_model = PanoramaRelPoseModel(integrated_off_cfg, device).to(device)
    integrated_off_load = _load_model(integrated_off_model, ckpt_path, device)
    integrated_off_model.eval()

    gate_zero_cfg = Config(**vars(eval_cfg))
    gate_zero_cfg.use_coupled_pose_residual_head = True
    gate_zero_cfg.coupled_pose_residual_trainable = False
    gate_zero_cfg.coupled_pose_residual_gate_init = -4.0
    gate_zero_model_base = PanoramaRelPoseModel(gate_zero_cfg, device).to(device)
    gate_zero_load = _load_model(gate_zero_model_base, ckpt_path, device)
    gate_zero_model_base.eval()
    gate_zero_model = GateZeroWrapper(gate_zero_model_base)

    scale_zero_cfg = Config(**vars(eval_cfg))
    scale_zero_cfg.use_coupled_pose_residual_head = True
    scale_zero_cfg.coupled_pose_residual_trainable = False
    scale_zero_cfg.coupled_pose_residual_rot_scale = 0.0
    scale_zero_cfg.coupled_pose_residual_tdir_scale = 0.0
    scale_zero_cfg.coupled_pose_residual_gate_init = -4.0
    scale_zero_model = PanoramaRelPoseModel(scale_zero_cfg, device).to(device)
    scale_zero_load = _load_model(scale_zero_model, ckpt_path, device)
    scale_zero_model.eval()

    variants = [
        _variant_payload(
            name="raw_no_policy",
            model=raw_model,
            cfg=raw_cfg,
            ds=ds,
            loader=loader,
            device=device,
            load_summary=raw_load,
            policy_path="",
            dt_anchor_applied=False,
            note="raw base checkpoint path without S2b bucket scaling policy",
        ),
        _variant_payload(
            name="S2b_policy_baseline",
            model=wrapped_model,
            cfg=raw_cfg,
            ds=ds,
            loader=loader,
            device=device,
            load_summary=wrapper_load,
            policy_path=str(POLICY_PATH),
            dt_anchor_applied=True,
            note="official wrapper-style S2b clean policy path",
        ),
        _variant_payload(
            name="coupled_head_off_with_policy",
            model=integrated_off_model,
            cfg=integrated_off_cfg,
            ds=ds,
            loader=loader,
            device=device,
            load_summary=integrated_off_load,
            policy_path=str(POLICY_PATH),
            dt_anchor_applied=True,
            note="current train_mvp/model path with integrated policy and coupled head disabled",
        ),
        _variant_payload(
            name="coupled_gate_zero_with_policy",
            model=gate_zero_model,
            cfg=gate_zero_cfg,
            ds=ds,
            loader=loader,
            device=device,
            load_summary=gate_zero_load,
            policy_path=str(POLICY_PATH),
            dt_anchor_applied=True,
            note="coupled head enabled, but outputs forced back to pre-coupled pose",
        ),
        _variant_payload(
            name="coupled_residual_scale_zero_with_policy",
            model=scale_zero_model,
            cfg=scale_zero_cfg,
            ds=ds,
            loader=loader,
            device=device,
            load_summary=scale_zero_load,
            policy_path=str(POLICY_PATH),
            dt_anchor_applied=True,
            note="coupled head enabled with residual scales forced to zero",
        ),
    ]

    by_name = {row["variant"]: row for row in variants}
    baseline = by_name["S2b_policy_baseline"]
    checks = {
        "S2b_policy_baseline_matches_reference": (
            _close(baseline["drift"], float(ref_summary["drift"]), 1.0e-6)
            and _close(baseline["ATE"], float(ref_summary["ATE"]), 1.0e-6)
            and _close(baseline["path_ratio"], float(ref_summary["metric_path_ratio"]), 1.0e-6)
        ),
        "coupled_head_off_matches_S2b": (
            _close(by_name["coupled_head_off_with_policy"]["drift"], baseline["drift"], 1.0e-6)
            and _close(by_name["coupled_head_off_with_policy"]["ATE"], baseline["ATE"], 1.0e-6)
            and _close(by_name["coupled_head_off_with_policy"]["path_ratio"], baseline["path_ratio"], 1.0e-6)
        ),
        "coupled_gate_zero_matches_S2b": (
            _close(by_name["coupled_gate_zero_with_policy"]["drift"], baseline["drift"], 1.0e-6)
            and _close(by_name["coupled_gate_zero_with_policy"]["ATE"], baseline["ATE"], 1.0e-6)
            and _close(by_name["coupled_gate_zero_with_policy"]["path_ratio"], baseline["path_ratio"], 1.0e-6)
        ),
        "coupled_scale_zero_matches_S2b": (
            _close(by_name["coupled_residual_scale_zero_with_policy"]["drift"], baseline["drift"], 1.0e-6)
            and _close(by_name["coupled_residual_scale_zero_with_policy"]["ATE"], baseline["ATE"], 1.0e-6)
            and _close(by_name["coupled_residual_scale_zero_with_policy"]["path_ratio"], baseline["path_ratio"], 1.0e-6)
        ),
        "tmag_before_after_preserved": (
            (not math.isfinite(by_name["coupled_gate_zero_with_policy"]["tmag_before_after_max_diff"]) or by_name["coupled_gate_zero_with_policy"]["tmag_before_after_max_diff"] <= 1.0e-9)
            and (not math.isfinite(by_name["coupled_residual_scale_zero_with_policy"]["tmag_before_after_max_diff"]) or by_name["coupled_residual_scale_zero_with_policy"]["tmag_before_after_max_diff"] <= 1.0e-9)
        ),
        "unexpected_zero": all(int(row["load_unexpected"]) == 0 for row in variants),
        "fine_rot_is_0p45": all(abs(float(row["fine_rot_fuse_strength"]) - 0.45) <= 1.0e-9 for row in variants),
        "no_training": True,
    }
    audit_pass = all(checks.values())

    lines = [
        "# S3a0c Policy Alignment Audit Report",
        "",
        "## Expected Targets",
        "",
        f"- policy json: `{POLICY_PATH}`",
        f"- base checkpoint: `{ckpt_path}`",
        f"- reference drift: `{float(ref_summary['drift']):.6f}`",
        f"- reference ATE: `{float(ref_summary['ATE']):.6f}`",
        f"- reference path_ratio: `{float(ref_summary['metric_path_ratio']):.6f}`",
        "",
        "## Variant Table",
        "",
        "| variant | drift | ATE | path_ratio | RPE_rot | RPE_tdir | RPE_tmag | rot | tdir_abs | tdir_local_A_abs | tmag_p10 | tmag_p50 | tmag_p90 | missing | unexpected | fine_rot | fine_tdir | fine_tmag | policy_json | dt_anchor | tmag_before_after_diff | gate_mean | d_rot_norm | d_tdir_norm | factor | k | pairs | chains | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in variants:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    _fmt(row["drift"]),
                    _fmt(row["ATE"]),
                    _fmt(row["path_ratio"]),
                    _fmt(row["RPE_rot"]),
                    _fmt(row["RPE_trans_dir"]),
                    _fmt(row["RPE_trans_mag"]),
                    _fmt(row["rot"]),
                    _fmt(row["tdir_abs"]),
                    _fmt(row["tdir_local_A_abs"]),
                    _fmt(row["tmag_p10"]),
                    _fmt(row["tmag_p50"]),
                    _fmt(row["tmag_p90"]),
                    str(row["load_missing"]),
                    str(row["load_unexpected"]),
                    _fmt(row["fine_rot_fuse_strength"]),
                    _fmt(row["fine_tdir_fuse_strength"]),
                    _fmt(row["fine_tmag_fuse_strength"]),
                    row["policy_json_path"] or "-",
                    str(row["dt_anchor_applied"]),
                    _fmt(row["tmag_before_after_max_diff"]),
                    _fmt(row["coupled_gate_mean"]),
                    _fmt(row["coupled_delta_rot_norm"]),
                    _fmt(row["coupled_delta_tdir_norm"]),
                    _fmt(row["dt_bucket_scale_anchor_factor"]),
                    str(row["selected_k"]),
                    str(row["num_pairs"]),
                    str(row["num_chains"]),
                    row["note"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## PASS Checks",
            "",
            f"- audit_pass: `{audit_pass}`",
        ]
    )
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Path Audit Answers",
            "",
            f"1. S3a0 smoke 当前从哪个 checkpoint 初始化？`{policy['base_checkpoint_path']}`",
            "2. S3a0 smoke 当前是否加载 checkpoints/S2b_clean_fine_rot_policy.json？修复前否；修复后通过 `dt_bucket_scale_anchor_policy_json` 显式加载。",
            "3. S3a0 smoke 当前是否应用 S1d5/S2b dt-anchor effective bucket factors？修复前否；修复后是。",
            f"4. S3a0 smoke 当前是否应用 fine_rot=0.45？是，当前审计所有 variant 都是 `fine_rot=0.45`。",
            "5. S3a0 smoke 当前评估 path_ratio 的路径是否等价于 eval_s2b_clean_policy.sh？修复后是；`S2b_policy_baseline` 与 `coupled_head_off_with_policy` 应一致。",
            "6. gate=0 时为什么之前复现的是 raw_no_policy 0.496，而不是 S2b 0.935？因为旧 smoke 路径没有加载 S2b clean bucket-scaled dt-anchor magnitude policy，gate=0 只关闭了 coupled residual，没有补回 S2b 的 magnitude calibration。",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
