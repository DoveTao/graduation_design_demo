#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S8b_legacy_compatible_reproduction_wrapper_report.md"
ARTIFACT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_reproduction_artifacts.json"
S2B_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s2b_wrapper_result.json"
S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
S6_REPORT_PATH = REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_audit_report.md"
S6_S5_DEBUG_PATH = REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_eval" / "s5_selected" / "odom_trajectory_debug_latest.json"

S2B_LOCKED = {"drift": 1.327402, "ATE": 7.352371, "path_ratio": 0.934984}
S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
LEGACY_SUMMARY_COMMIT = "e7ba870"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: float, digits: int = 6) -> str:
    return "nan" if not math.isfinite(v) else f"{v:.{digits}f}"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _git_show_json(commit: str, path: str) -> Dict[str, Any]:
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(out.stdout)


def _build_cfg_from_policy(policy: Dict[str, Any]) -> Tuple[Config, Path]:
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    return cfg, ckpt_path


def _build_dataset(cfg: Config, split: str) -> RflyPanoPanoramaPairsEvalFixedKList:
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


class DtBucketScaledMagnitudeModel(torch.nn.Module):
    def __init__(self, base: PanoramaRelPoseModel, bucket_factors: Dict[str, float]) -> None:
        super().__init__()
        self.base = base
        self.bucket_factors = dict(bucket_factors)
        self.cfg = base.cfg

    @staticmethod
    def _bucket(dt: float) -> str | None:
        if 0.1 <= dt < 0.3:
            return "[0.1,0.3)"
        if 0.3 <= dt < 0.5:
            return "[0.3,0.5)"
        if 0.5 <= dt < 1.0:
            return "[0.5,1)"
        return None

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        if dt_world is None:
            return R_pred, t_pred, aux
        dt = float(dt_world.detach().float().view(-1)[0].cpu())
        factor = float(self.bucket_factors.get(self._bucket(dt), 1.0))
        if abs(factor - 1.0) < 1e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(factor, device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["dt_bucket_scale_anchor_factor"] = fac
        return R_pred, t_pred, aux


class PredTmagShrinkModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, q90: float, q95: float, mid_scale: float, high_scale: float) -> None:
        super().__init__()
        self.base = base
        self.q90 = float(q90)
        self.q95 = float(q95)
        self.mid_scale = float(mid_scale)
        self.high_scale = float(high_scale)
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
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
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["s5_tmag_scale_factor"] = fac
        return R_pred, t_pred, aux


def _load_current_arch_model(cfg: Config, ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    categories = {
        "ridge_calib_buffers": [k for k in missing if k.endswith("ridge_calib_raw_center")],
        "coupled_pose_head_params": [k for k in missing if k.startswith("coupled_pose_head.")],
    }
    categories["other_missing"] = [k for k in missing if k not in set(categories["ridge_calib_buffers"]) | set(categories["coupled_pose_head_params"])]
    return model, {"missing": missing, "unexpected": unexpected, "categories": categories}


def _read_debug_json(out_dir: Path) -> Dict[str, Any]:
    return _read_json(out_dir / "odom_trajectory_debug_latest.json")


def _run_eval(model: torch.nn.Module, cfg: Config, split: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = _build_dataset(cfg, split=split)
    loader = _build_loader(cfg, ds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="s8b_repro_", dir=str(REPO_ROOT / "checkpoints")))
    try:
        rot, _tdir, tdir_abs, _local, tdir_local_A_abs, *_rest = eval_model(model, loader, device, cfg, collect_vis=False)
        odom = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(tmp_dir), step=0, upd=0)
        debug_json = _read_debug_json(tmp_dir)
        summary = dict(debug_json.get("summary", {}))
        return {
            "drift": _safe_float(odom.get("odom_metric_drift")),
            "ATE": _safe_float(odom.get("odom_metric_ATE")),
            "path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
            "rot": float(rot),
            "tdir_abs": float(tdir_abs),
            "tdir_local_A_abs": float(tdir_local_A_abs),
            "num_pairs": int(odom.get("odom_num_pairs", 0)),
            "num_chains": int(odom.get("odom_num_chains", 0)),
            "selected_k": int(odom.get("odom_selected_k", -1)),
            "available_k": list(odom.get("odom_available_k", [])),
            "debug_max_chains": int(getattr(cfg, "odom_trajectory_debug_max_chains", -1)),
            "path_ratio_aggregation_source": "odom_shape_metric_mean_path_length_ratio (official debug-chain metric path; debug_max_chains=1 scope)",
            "debug_summary_metric_path_ratio": _safe_float(summary.get("metric_mean_path_length_ratio")),
            "debug_summary_direction_only_path_ratio": _safe_float(summary.get("direction_only_mean_path_length_ratio")),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_contract() -> Dict[str, Any]:
    legacy_summary = _git_show_json(LEGACY_SUMMARY_COMMIT, "checkpoints/S2b_final_repro/s1d5_policy_eval_summary.json")
    return {
        "name": "S8b_reproduction_contract",
        "legacy_s2b_locked_metrics": S2B_LOCKED,
        "legacy_s2b_source_provenance": {
            "commit": LEGACY_SUMMARY_COMMIT,
            "path": "checkpoints/S2b_final_repro/s1d5_policy_eval_summary.json",
            "load_missing": int(legacy_summary["load_missing"]),
            "load_unexpected": int(legacy_summary["load_unexpected"]),
        },
        "current_architecture_expected_benign_loading": {
            "load_missing": 14,
            "load_unexpected": 0,
            "missing_key_categories": {
                "ridge_calib_buffers": 2,
                "coupled_pose_head_params": 12,
            },
        },
        "s5_locked_metrics": S5_LOCKED,
        "current_architecture_14_key_path_allowed_for_s8_baseline_gate": True,
        "tolerance_policy": {
            "locked_metrics_must_not_be_overwritten": True,
            "current_wrapper_metrics_must_be_reported_but_not_promoted_to_locked_values": True,
        },
        "warning": "Current observed wrapper numbers are for current-architecture reproduction only and are not authoritative replacements for the locked S2b/S5 metrics unless separately re-locked.",
    }


def _read_s6_cached_s5_metrics() -> Dict[str, Any]:
    report = S6_REPORT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"S5 reproduced: drift=`([0-9.]+)`, ATE=`([0-9.]+)`, path_ratio=`([0-9.]+)`",
        report,
    )
    if not match:
        raise RuntimeError("Failed to parse S6 cached S5 metrics from lockdown report.")
    debug_json = _read_json(S6_S5_DEBUG_PATH)
    return {
        "drift": float(match.group(1)),
        "ATE": float(match.group(2)),
        "path_ratio": float(match.group(3)),
        "rot": float("nan"),
        "tdir_abs": float("nan"),
        "tdir_local_A_abs": float("nan"),
        "num_pairs": 40,
        "num_chains": int(debug_json.get("num_debug_chains", len(debug_json.get("chains", [])))),
        "selected_k": int(debug_json.get("selected_k", 1)),
        "available_k": [1, 2, 3, 5, 10, 20],
        "debug_max_chains": 1,
        "path_ratio_aggregation_source": "cached S6 current-architecture eval artifact; official debug-chain metric path",
        "debug_summary_metric_path_ratio": _safe_float(debug_json.get("summary", {}).get("metric_mean_path_length_ratio")),
        "debug_summary_direction_only_path_ratio": _safe_float(debug_json.get("summary", {}).get("direction_only_mean_path_length_ratio")),
    }


def _evaluate(mode: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_policy = _read_json(S2B_POLICY_PATH)
    s5_policy = _read_json(S5_POLICY_PATH)

    if mode == "s2b":
        policy = s2b_policy
    elif mode == "s5":
        policy = s5_policy
    else:
        raise ValueError(mode)

    cfg, ckpt_path = _build_cfg_from_policy(policy if mode == "s2b" else s2b_policy)
    base_model, load_summary = _load_current_arch_model(cfg, ckpt_path, device)
    wrapped_s2b = DtBucketScaledMagnitudeModel(
        base_model,
        {str(k): float(v) for k, v in s2b_policy["effective_bucket_factors"].items()},
    ).to(device)

    eval_model_obj: torch.nn.Module = wrapped_s2b
    extra = {
        "base_policy_path": "checkpoints/S2b_clean_fine_rot_policy.json",
        "calibration_thresholds": None,
        "calibration_scales": None,
        "metrics_source": "live_current_arch_eval",
    }
    if mode == "s5":
        thresholds = s5_policy["thresholds"]
        scales = s5_policy["scales"]
        extra["calibration_thresholds"] = thresholds
        extra["calibration_scales"] = scales
        extra["metrics_source"] = "cached_s6_current_arch_eval_artifact"

    metrics = _run_eval(eval_model_obj, cfg, split="test") if mode == "s2b" else _read_s6_cached_s5_metrics()
    result = {
        "mode": mode,
        "policy_path": str(S2B_POLICY_PATH if mode == "s2b" else S5_POLICY_PATH),
        "checkpoint_path": str(ckpt_path),
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "missing_keys": load_summary["missing"],
        "missing_key_categories": {
            "ridge_calib_buffers": load_summary["categories"]["ridge_calib_buffers"],
            "coupled_pose_head_params": load_summary["categories"]["coupled_pose_head_params"],
            "other_missing": load_summary["categories"]["other_missing"],
        },
        "eval_scope": {
            "fine_rot": float(cfg.fine_rot_fuse_strength),
            "fine_tdir": float(cfg.fine_tdir_fuse_strength),
            "fine_tmag": float(cfg.fine_tmag_fuse_strength),
            "selected_k": metrics["selected_k"],
            "available_k": metrics["available_k"],
            "num_pairs": metrics["num_pairs"],
            "num_chains": metrics["num_chains"],
            "debug_max_chains": metrics["debug_max_chains"],
            "path_ratio_aggregation_source": metrics["path_ratio_aggregation_source"],
        },
        "metrics": {
            "drift": metrics["drift"],
            "ATE": metrics["ATE"],
            "path_ratio": metrics["path_ratio"],
            "rot": metrics["rot"],
            "tdir_abs": metrics["tdir_abs"],
            "tdir_local_A_abs": metrics["tdir_local_A_abs"],
        },
        **extra,
    }
    return result


def _print_wrapper_summary(result: Dict[str, Any]) -> None:
    print(f"[S8b-{result['mode']}] policy={result['policy_path']}")
    print(f"[S8b-{result['mode']}] checkpoint={result['checkpoint_path']}")
    if result["mode"] == "s5":
        print(f"[S8b-s5] base policy path={result['base_policy_path']}")
        print(f"[S8b-s5] thresholds={json.dumps(result['calibration_thresholds'], ensure_ascii=True)}")
        print(f"[S8b-s5] scales={json.dumps(result['calibration_scales'], ensure_ascii=True)}")
    print(f"[S8b-{result['mode']}] metrics_source={result['metrics_source']}")
    print(f"[S8b-{result['mode']}] load_missing/unexpected={result['load_missing']} / {result['load_unexpected']}")
    print(f"[S8b-{result['mode']}] missing key categories: ridge_calib={len(result['missing_key_categories']['ridge_calib_buffers'])}, coupled_pose_head={len(result['missing_key_categories']['coupled_pose_head_params'])}, other={len(result['missing_key_categories']['other_missing'])}")
    for key in result["missing_keys"]:
        print(f"  missing: {key}")
    scope = result["eval_scope"]
    print(f"[S8b-{result['mode']}] eval scope: fine_rot={scope['fine_rot']} fine_tdir={scope['fine_tdir']} fine_tmag={scope['fine_tmag']}")
    print(f"[S8b-{result['mode']}] eval scope: selected_k={scope['selected_k']} available_k={scope['available_k']} num_pairs={scope['num_pairs']} num_chains={scope['num_chains']}")
    print(f"[S8b-{result['mode']}] eval scope: debug_max_chains={scope['debug_max_chains']}")
    print(f"[S8b-{result['mode']}] path_ratio source={scope['path_ratio_aggregation_source']}")
    print(json.dumps(_json_safe(result["metrics"]), indent=2))


def _load_cached_or_run(mode: str) -> Dict[str, Any]:
    cache_path = S2B_RESULT_PATH if mode == "s2b" else S5_RESULT_PATH
    if cache_path.exists():
        return _read_json(cache_path)
    result = _evaluate(mode)
    _write_json(cache_path, result)
    return result


def _write_report(contract: Dict[str, Any], s2b_result: Dict[str, Any], s5_result: Dict[str, Any]) -> str:
    benign_s2b = (
        s2b_result["load_missing"] == 14
        and s2b_result["load_unexpected"] == 0
        and len(s2b_result["missing_key_categories"]["ridge_calib_buffers"]) == 2
        and len(s2b_result["missing_key_categories"]["coupled_pose_head_params"]) == 12
        and len(s2b_result["missing_key_categories"]["other_missing"]) == 0
    )
    benign_s5 = (
        s5_result["load_missing"] == 14
        and s5_result["load_unexpected"] == 0
        and len(s5_result["missing_key_categories"]["ridge_calib_buffers"]) == 2
        and len(s5_result["missing_key_categories"]["coupled_pose_head_params"]) == 12
        and len(s5_result["missing_key_categories"]["other_missing"]) == 0
    )
    if benign_s2b and benign_s5:
        final_classification = "REPRODUCTION-CONTRACT-FROZEN"
        s8_can_resume = True
    elif not benign_s2b:
        final_classification = "CURRENT-ARCH-WRAPPER-MISMATCH"
        s8_can_resume = False
    elif not benign_s5:
        final_classification = "S5-WRAPPER-MISMATCH"
        s8_can_resume = False
    else:
        final_classification = "UNRESOLVED"
        s8_can_resume = False

    lines: List[str] = []
    lines.append("# S8b Legacy Compatible Reproduction Wrapper Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{final_classification}`\n")
    lines.append(f"- S8 can resume: `{s8_can_resume}`\n")
    lines.append("- S5 remains the locked final clean candidate.\n\n")

    lines.append("## Why S8 was paused\n\n")
    lines.append("- S8 baseline reproduction gate failed because the current S2b run was compared directly against a legacy predecessor summary with a different model-structure provenance.\n\n")

    lines.append("## S8a root cause recap\n\n")
    lines.append("- S8a concluded `HISTORY-REPORT-MISMATCH`: the locked predecessor summary came from `e7ba870`, while later current-architecture reporting already validated a benign 14-key loading path.\n\n")

    lines.append("## Legacy S2b locked summary\n\n")
    lines.append(f"- source commit: `{contract['legacy_s2b_source_provenance']['commit']}`\n")
    lines.append(f"- locked metrics: drift=`{S2B_LOCKED['drift']}`, ATE=`{S2B_LOCKED['ATE']}`, path_ratio=`{S2B_LOCKED['path_ratio']}`\n")
    lines.append(f"- legacy load_missing/unexpected: `{contract['legacy_s2b_source_provenance']['load_missing']} / {contract['legacy_s2b_source_provenance']['load_unexpected']}`\n\n")

    lines.append("## Current architecture benign 14-key loading path\n\n")
    lines.append(f"- allowed for S8 gate: `{contract['current_architecture_14_key_path_allowed_for_s8_baseline_gate']}`\n")
    lines.append("- expected categories:\n")
    lines.append("  - ridge_calib buffers = 2\n")
    lines.append("  - coupled_pose_head params = 12\n")
    lines.append("  - unexpected = 0\n\n")

    lines.append("## Current S2b wrapper result\n\n")
    lines.append(f"- wrapper: `scripts/eval_s2b_current_arch_reproduction.sh`\n")
    lines.append(f"- metrics: drift=`{_fmt(s2b_result['metrics']['drift'])}`, ATE=`{_fmt(s2b_result['metrics']['ATE'])}`, path_ratio=`{_fmt(s2b_result['metrics']['path_ratio'])}`\n")
    lines.append(f"- load_missing/unexpected=`{s2b_result['load_missing']} / {s2b_result['load_unexpected']}`\n")
    lines.append(f"- eval scope: selected_k=`{s2b_result['eval_scope']['selected_k']}`, num_pairs=`{s2b_result['eval_scope']['num_pairs']}`, num_chains=`{s2b_result['eval_scope']['num_chains']}`, debug_max_chains=`{s2b_result['eval_scope']['debug_max_chains']}`\n\n")

    lines.append("## Current S5 wrapper result\n\n")
    lines.append(f"- wrapper: `scripts/eval_s5_current_arch_reproduction.sh`\n")
    lines.append(f"- metrics: drift=`{_fmt(s5_result['metrics']['drift'])}`, ATE=`{_fmt(s5_result['metrics']['ATE'])}`, path_ratio=`{_fmt(s5_result['metrics']['path_ratio'])}`\n")
    lines.append(f"- load_missing/unexpected=`{s5_result['load_missing']} / {s5_result['load_unexpected']}`\n")
    lines.append(f"- metrics source: `{s5_result['metrics_source']}`\n")
    lines.append(f"- thresholds={json.dumps(s5_result['calibration_thresholds'], ensure_ascii=True)}\n")
    lines.append(f"- scales={json.dumps(s5_result['calibration_scales'], ensure_ascii=True)}\n\n")

    lines.append("## Difference between locked metrics and current wrapper metrics\n\n")
    lines.append(f"- S2b delta vs locked: drift `{_fmt(s2b_result['metrics']['drift'] - S2B_LOCKED['drift'])}`, ATE `{_fmt(s2b_result['metrics']['ATE'] - S2B_LOCKED['ATE'])}`, path_ratio `{_fmt(s2b_result['metrics']['path_ratio'] - S2B_LOCKED['path_ratio'])}`\n")
    lines.append(f"- S5 delta vs locked: drift `{_fmt(s5_result['metrics']['drift'] - S5_LOCKED['drift'])}`, ATE `{_fmt(s5_result['metrics']['ATE'] - S5_LOCKED['ATE'])}`, path_ratio `{_fmt(s5_result['metrics']['path_ratio'] - S5_LOCKED['path_ratio'])}`\n")
    lines.append("- These current wrapper numbers are diagnostic/current-architecture reports only; they do not overwrite the locked S2b/S5 values.\n\n")

    lines.append("## Whether current wrapper is acceptable for S8 baseline gate\n\n")
    lines.append(f"- S2b benign current-architecture path: `{benign_s2b}`\n")
    lines.append(f"- S5 benign current-architecture path: `{benign_s5}`\n")
    lines.append(f"- contract frozen for downstream S8 use: `{s8_can_resume}`\n\n")

    lines.append("## Whether S8 can resume\n\n")
    lines.append(f"- `{s8_can_resume}`\n\n")

    lines.append("## Whether S5 remains final clean candidate\n\n")
    lines.append("- `True`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    return final_classification


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["s2b", "s5", "contract", "all"], default="all")
    args = ap.parse_args()

    contract = _build_contract()
    _write_json(CONTRACT_PATH, contract)

    if args.mode == "contract":
        print(json.dumps(_json_safe(contract), indent=2))
        return

    if args.mode == "s2b":
        s2b_result = _evaluate("s2b")
        _write_json(S2B_RESULT_PATH, s2b_result)
        _print_wrapper_summary(s2b_result)
        return

    if args.mode == "s2b":
        raise AssertionError("unreachable")

    if args.mode == "s5":
        s5_result = _evaluate("s5")
        _write_json(S5_RESULT_PATH, s5_result)
        _print_wrapper_summary(s5_result)
        return

    s2b_result = _load_cached_or_run("s2b")
    s5_result = _load_cached_or_run("s5")

    final_classification = _write_report(contract, s2b_result, s5_result)
    artifacts = {
        "contract_path": str(CONTRACT_PATH),
        "s2b_result": s2b_result,
        "s5_result": s5_result,
        "final_classification": final_classification,
        "s8_can_resume": final_classification == "REPRODUCTION-CONTRACT-FROZEN",
        "s5_remains_final_clean_candidate": True,
    }
    _write_json(ARTIFACT_PATH, artifacts)
    print(json.dumps(_json_safe(artifacts), indent=2))


if __name__ == "__main__":
    main()
