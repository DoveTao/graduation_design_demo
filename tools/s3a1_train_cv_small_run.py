#!/usr/bin/env python3
from __future__ import annotations

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

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from eval_clean_policy import _extract_pairs
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/dovetao/miniconda3/envs/pytorch/bin/python")
BASE_CKPT = "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
POLICY_JSON = "checkpoints/S2b_clean_fine_rot_policy.json"
EXP_NAME = "S3a1_train_cv_small_run"
OUT_ROOT = REPO_ROOT / "checkpoints" / EXP_NAME
REPORT_PATH = REPO_ROOT / "checkpoints" / f"{EXP_NAME}_report.md"
TMP_DATA_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s3a1_train_cv_small_data"
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
SEQ01_TRAIN_SEED = 0
SEQ02_TRAIN_SEED = 3
S2B_DRIFT = 1.327402
S2B_ATE = 7.352371
S2B_PATH_RATIO = 0.934984


@dataclass(frozen=True)
class RunConfig:
    name: str
    lr: float
    max_updates: int
    rot_scale: float
    gate_max: float
    residual_reg_w: float
    simplicity_rank: Tuple[float, float, float, int]


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_cfg_from_ckpt(ckpt_path: Path) -> Config:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _build_eval_dataset(cfg: Config, split: str):
    from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList

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


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Config, Dict[str, int]]:
    cfg = _load_cfg_from_ckpt(ckpt_path)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, {"missing": len(msg.missing_keys), "unexpected": len(msg.unexpected_keys)}


def _eval_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, load_summary = _load_model_from_ckpt(ckpt_path, device)
    ds = _build_eval_dataset(cfg, split="test")
    loader = _build_loader(cfg, ds)
    out_dir = ckpt_path.parent / "_final_eval_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        model, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    rows = _extract_pairs(model, ds, device)
    vals = np.asarray([float(r["tmag_pred"]) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    q = {
        "p10": float(np.percentile(vals, 10)) if vals.size else float("nan"),
        "p50": float(np.percentile(vals, 50)) if vals.size else float("nan"),
        "p90": float(np.percentile(vals, 90)) if vals.size else float("nan"),
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    return {
        "drift": _safe_float(odom.get("odom_metric_drift")),
        "ATE": _safe_float(odom.get("odom_metric_ATE")),
        "path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        "RPE_rot": _safe_float(odom.get("odom_metric_RPE_rot")),
        "RPE_trans_dir": _safe_float(odom.get("odom_metric_RPE_trans_dir")),
        "RPE_trans_mag": _safe_float(odom.get("odom_metric_RPE_trans_mag")),
        "rot": float(rot),
        "tdir_abs": float(tdir_abs),
        "tdir_local_A_abs": float(tdir_local_A_abs),
        "selected_k": int(odom.get("odom_selected_k", -1)),
        "num_pairs": int(odom.get("odom_num_pairs", 0)),
        "num_chains": int(odom.get("odom_num_chains", 0)),
        "missing": int(load_summary["missing"]),
        "unexpected": int(load_summary["unexpected"]),
        "tmag_p10": q["p10"],
        "tmag_p50": q["p50"],
        "tmag_p90": q["p90"],
    }


def _configs() -> List[RunConfig]:
    return [
        RunConfig("A", 5.0e-5, 100, 0.01, 0.02, 0.01, (0.01, 0.02, 5.0e-5, 100)),
        RunConfig("B", 5.0e-5, 200, 0.01, 0.02, 0.01, (0.01, 0.02, 5.0e-5, 200)),
        RunConfig("C", 1.0e-4, 100, 0.02, 0.05, 0.01, (0.02, 0.05, 1.0e-4, 100)),
        RunConfig("D", 1.0e-4, 200, 0.02, 0.05, 0.01, (0.02, 0.05, 1.0e-4, 200)),
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
    matches = pat.findall(stdout)
    if not matches:
        return -1, -1
    miss, unexp = matches[-1]
    return int(miss), int(unexp)


def _run_train(exp_name: str, data_root: Path, split_seed: int, cfg: RunConfig, output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"[S3a1-cv] start {exp_name} | lr={cfg.lr} updates={cfg.max_updates} rot_scale={cfg.rot_scale} gate_max={cfg.gate_max}", flush=True)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", "ckpt_dir=checkpoints",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", f"dt_bucket_scale_anchor_policy_json={POLICY_JSON}",
        "--set", f"init_checkpoint={BASE_CKPT}",
        "--set", "strict_load_checkpoint=False",
        "--set", "use_fine_stage=True",
        "--set", "fine_rot_fuse_strength=0.45",
        "--set", "fine_tdir_fuse_strength=0.0",
        "--set", "fine_tmag_fuse_strength=0.0",
        "--set", "use_geometry_refine=False",
        "--set", "use_coupled_pose_residual_head=True",
        "--set", "coupled_pose_residual_trainable=True",
        "--set", "coupled_pose_residual_enable_rot=True",
        "--set", "coupled_pose_residual_enable_tdir=False",
        "--set", "coupled_pose_residual_force_tdir_zero=True",
        "--set", f"coupled_pose_residual_rot_scale={cfg.rot_scale}",
        "--set", "coupled_pose_residual_tdir_scale=0.0",
        "--set", f"coupled_pose_residual_gate_max={cfg.gate_max}",
        "--set", "train_coupled_pose_residual_only=True",
        "--set", "freeze_backbone_for_coupled_pose=True",
        "--set", "freeze_tmag_for_coupled_pose=True",
        "--set", "freeze_dt_anchor_for_coupled_pose=True",
        "--set", "coupled_pose_tdir_loss_w=0.0",
        "--set", "coupled_pose_joint_loss_w=0.0",
        "--set", "coupled_pose_chain_loss_w=0.0",
        "--set", f"coupled_pose_residual_reg_w={cfg.residual_reg_w}",
        "--set", "use_tdir_anchor_loss=False",
        "--set", "w_tdir_anchor=0.0",
        "--set", "use_seq_turn_loss=False",
        "--set", "seq_turn_loss_w=0.0",
        "--set", "use_seq_turn_chain_loss=False",
        "--set", "seq_turn_chain_loss_w=0.0",
        "--set", "use_odom_chain_len_loss=False",
        "--set", "odom_chain_len_loss_w=0.0",
        "--set", "use_odom_chain_vec_loss=False",
        "--set", "odom_chain_vec_loss_w=0.0",
        "--set", f"lr={cfg.lr}",
        "--set", "batch_size=1",
        "--set", "grad_accum=1",
        "--set", f"max_steps={cfg.max_updates}",
        "--set", f"eval_every={cfg.max_updates}",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
        "--set", "num_workers=0",
        "--set", "pin_memory=False",
        "--set", "persistent_workers=False",
        "--set", "prefetch_factor=2",
        "--set", "log_every=10",
        "--set", "save_last_train_state=False",
        "--set", "save_last_eval_checkpoint=False",
        "--set", "save_metric_checkpoints=False",
        "--set", "save_best_joint_checkpoint=False",
        "--set", "save_best_local_joint_checkpoint=False",
        "--set", "save_best_odom_checkpoint=False",
        "--set", "save_best_smallk_odom_checkpoint=False",
        "--set", "save_odom_trajectory_debug=True",
        "--set", "save_vis_examples=False",
        "--set", "save_vis_payload_npz=False",
        "--set", "save_vis_diag_json=False",
        "--set", "amp=False",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = output_dir / "train_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for {exp_name}. See {log_path}")
    summary_path = output_dir / "final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final_summary.json: {summary_path}")
    summary = _read_json(summary_path)
    missing, unexpected = _parse_init_counts(proc.stdout)
    summary["init_missing"] = missing
    summary["init_unexpected"] = unexpected
    _write_json(output_dir / "run_summary_augmented.json", summary)
    last_eval = dict(summary.get("last_eval", {}))
    print(
        "[S3a1-cv] done "
        f"{exp_name} | drift={_safe_float(last_eval.get('odom_metric_drift')):.6f} "
        f"ATE={_safe_float(last_eval.get('odom_metric_ATE')):.6f} "
        f"path_ratio={_safe_float(last_eval.get('odom_shape_metric_mean_path_length_ratio')):.6f} "
        f"unexpected={unexpected}",
        flush=True,
    )
    return summary


def _fold_name(heldout_seq: str) -> str:
    return f"heldout_scene01_{heldout_seq}"


def _candidate_key(cfg: RunConfig) -> str:
    return f"{cfg.name}_lr{cfg.lr:g}_upd{cfg.max_updates}_rot{cfg.rot_scale:g}_gate{cfg.gate_max:g}"


def _trainable_audit(summary: Dict[str, Any]) -> Dict[str, Any]:
    names = list(summary.get("trainable_names", []))
    allowed = (
        "coupled_pose_head.backbone.",
        "coupled_pose_head.rot_head.",
        "coupled_pose_head.gate_head.",
    )
    forbidden_names = [n for n in names if not n.startswith(allowed)]
    optimizer_only = all(n.startswith(allowed) for n in names)
    excludes = all(
        all(token not in n for token in ("tdir_head", "encoder", "coarse", "fine", "direct_head", "mag_head"))
        for n in names
    )
    return {
        "optimizer_only_rot_gate": bool(optimizer_only and excludes),
        "forbidden_trainable_params_count": int(len(summary.get("forbidden_trainable_params", [])) + len(forbidden_names)),
    }


def _row_from_summary(cfg: RunConfig, heldout_seq: str, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    last_eval = dict(summary.get("last_eval", {}))
    first_train = dict(summary.get("first_train", {}))
    last_train = dict(summary.get("last_train", {}))
    audit = _trainable_audit(summary)
    return {
        "candidate": _candidate_key(cfg),
        "candidate_name": cfg.name,
        "heldout_seq": heldout_seq,
        "lr": cfg.lr,
        "max_updates": cfg.max_updates,
        "rot_scale": cfg.rot_scale,
        "gate_max": cfg.gate_max,
        "residual_reg_w": cfg.residual_reg_w,
        "trainable_param_names": list(summary.get("trainable_names", [])),
        "trainable_param_count": int(summary.get("trainable_param_count", 0)),
        "optimizer_param_count": int(summary.get("optimizer_param_count", 0)),
        "forbidden_trainable_params_count": int(audit["forbidden_trainable_params_count"]),
        "bad_forward": int(summary.get("bad_forward", 0)),
        "skip_updates": int(summary.get("skip_updates", 0)),
        "total_updates": int(summary.get("total_updates", 0)),
        "loss_before": _safe_float(first_train.get("loss_total")),
        "loss_after": _safe_float(last_train.get("loss_total")),
        "delta_rot_norm": _safe_float(last_train.get("delta_rot_norm_mean")),
        "delta_tdir_norm": _safe_float(last_train.get("delta_tdir_norm_mean")),
        "gate_mean": _safe_float(last_train.get("coupled_gate_mean")),
        "tdir_diff": _safe_float(last_train.get("tdir_before_after_max_diff")),
        "tmag_diff": _safe_float(last_train.get("tmag_before_after_max_diff")),
        "drift": _safe_float(last_eval.get("odom_metric_drift")),
        "ATE": _safe_float(last_eval.get("odom_metric_ATE")),
        "path_ratio": _safe_float(last_eval.get("odom_shape_metric_mean_path_length_ratio")),
        "RPE_rot": _safe_float(last_eval.get("odom_metric_RPE_rot")),
        "RPE_trans_dir": _safe_float(last_eval.get("odom_metric_RPE_trans_dir")),
        "RPE_trans_mag": _safe_float(last_eval.get("odom_metric_RPE_trans_mag")),
        "rot": _safe_float(last_eval.get("rot")),
        "tdir_abs": _safe_float(last_eval.get("tdir_abs")),
        "tdir_local_A_abs": _safe_float(last_eval.get("tdir_local_A_abs")),
        "selected_k": int(last_eval.get("odom_selected_k", -1)),
        "num_pairs": int(last_eval.get("odom_num_pairs", 0)),
        "num_chains": int(last_eval.get("odom_num_chains", 0)),
        "missing": int(summary.get("init_missing", -1)),
        "unexpected": int(summary.get("init_unexpected", -1)),
        "optimizer_only_rot_gate": bool(audit["optimizer_only_rot_gate"]),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
    }


def _candidate_is_valid(row: Dict[str, Any]) -> bool:
    return (
        row["path_ratio"] >= 0.90
        and row["drift"] <= 1.45
        and abs(row["tdir_diff"]) <= 1.0e-9
        and abs(row["tmag_diff"]) <= 1.0e-9
        and row["unexpected"] == 0
        and row["forbidden_trainable_params_count"] == 0
        and row["bad_forward"] == 0
        and row["skip_updates"] == 0
        and row["optimizer_only_rot_gate"]
        and not math.isnan(row["ATE"])
    )


def _aggregate_candidate(cfg: RunConfig, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mean_ate = float(np.mean([float(r["ATE"]) for r in rows]))
    mean_drift = float(np.mean([float(r["drift"]) for r in rows]))
    mean_path = float(np.mean([float(r["path_ratio"]) for r in rows]))
    is_valid = all(_candidate_is_valid(r) for r in rows)
    return {
        "candidate": _candidate_key(cfg),
        "candidate_name": cfg.name,
        "lr": cfg.lr,
        "max_updates": cfg.max_updates,
        "rot_scale": cfg.rot_scale,
        "gate_max": cfg.gate_max,
        "residual_reg_w": cfg.residual_reg_w,
        "mean_ATE": mean_ate,
        "mean_drift": mean_drift,
        "mean_path_ratio": mean_path,
        "valid": bool(is_valid),
        "simplicity_rank": cfg.simplicity_rank,
        "rows": list(rows),
    }


def _select_candidate(aggregates: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    eligible = [a for a in aggregates if a["valid"]]
    if not eligible:
        return None
    eligible.sort(key=lambda a: a["mean_ATE"])
    best_ate = eligible[0]["mean_ATE"]
    close_ate = [a for a in eligible if abs(a["mean_ATE"] - best_ate) < 0.03]
    if len(close_ate) == 1:
        return close_ate[0]
    close_ate.sort(key=lambda a: a["mean_drift"])
    best_drift = close_ate[0]["mean_drift"]
    close_drift = [a for a in close_ate if abs(a["mean_drift"] - best_drift) < 0.01]
    if len(close_drift) == 1:
        return close_drift[0]
    close_drift.sort(key=lambda a: a["simplicity_rank"])
    return close_drift[0]


def _verdict(final_metrics: Dict[str, Any], selected: Dict[str, Any] | None, optimizer_ok: bool, tdir_ok: bool, tmag_ok: bool) -> str:
    if selected is None:
        return "FAIL"
    if (
        final_metrics["ATE"] < S2B_ATE
        and final_metrics["drift"] <= 1.35
        and final_metrics["path_ratio"] >= 0.90
        and optimizer_ok
        and tdir_ok
        and tmag_ok
        and final_metrics["unexpected"] == 0
    ):
        return "SUCCESS"
    if final_metrics["path_ratio"] >= 0.90 and final_metrics["drift"] <= 1.45 and optimizer_ok and tdir_ok and tmag_ok:
        return "PARTIAL"
    return "FAIL"


def _write_report(
    rows: Sequence[Dict[str, Any]],
    aggregates: Sequence[Dict[str, Any]],
    selected: Dict[str, Any] | None,
    final_run: Dict[str, Any] | None,
    final_metrics: Dict[str, Any] | None,
    verdict: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# {EXP_NAME} Report\n\n")
    lines.append("## 1. S2b baseline\n\n")
    lines.append(f"- drift = `{S2B_DRIFT}`\n")
    lines.append(f"- ATE = `{S2B_ATE}`\n")
    lines.append(f"- path_ratio = `{S2B_PATH_RATIO}`\n\n")
    lines.append("## 2. Smoke result recap\n\n")
    lines.append("- source commit: `15dc1a0754c7698754cccc4ba2b0ba7b5ac62e19`\n")
    lines.append("- drift = `1.3284205512539493`\n")
    lines.append("- ATE = `7.356587799281789`\n")
    lines.append("- path_ratio = `0.9350349269488233`\n")
    lines.append("- tdir_before_after_max_diff = `0.0`\n")
    lines.append("- tmag_before_after_max_diff = `0.0`\n")
    lines.append("- optimizer only rot residual/gate = `True`\n\n")

    lines.append("## 3. Train-CV candidate table\n\n")
    lines.append("| candidate | heldout | lr | updates | rot_scale | gate_max | reg_w | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | selected_k | num_pairs | num_chains | missing | unexpected | bad_forward | skip_updates | tdir_diff | tmag_diff | valid |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['heldout_seq']} | {_fmt(row['lr'])} | {row['max_updates']} | {_fmt(row['rot_scale'])} | {_fmt(row['gate_max'])} | {_fmt(row['residual_reg_w'])} | {_fmt(row['drift'])} | {_fmt(row['ATE'])} | {_fmt(row['path_ratio'])} | {_fmt(row['RPE_rot'])} | {_fmt(row['RPE_trans_dir'])} | {_fmt(row['RPE_trans_mag'])} | {_fmt(row['rot'])} | {_fmt(row['tdir_abs'])} | {_fmt(row['tdir_local_A_abs'])} | {row['selected_k']} | {row['num_pairs']} | {row['num_chains']} | {row['missing']} | {row['unexpected']} | {row['bad_forward']} | {row['skip_updates']} | {_fmt(row['tdir_diff'])} | {_fmt(row['tmag_diff'])} | {str(_candidate_is_valid(row))} |\n"
        )
    lines.append("\nMean ranking:\n\n")
    lines.append("| candidate | mean_ATE | mean_drift | mean_path_ratio | valid | simplicity |\n")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |\n")
    for agg in sorted(aggregates, key=lambda a: (not a["valid"], a["mean_ATE"], a["mean_drift"], a["simplicity_rank"])):
        lines.append(
            f"| {agg['candidate']} | {_fmt(agg['mean_ATE'])} | {_fmt(agg['mean_drift'])} | {_fmt(agg['mean_path_ratio'])} | {agg['valid']} | `{agg['simplicity_rank']}` |\n"
        )

    lines.append("\n## 4. Selected config\n\n")
    if selected is None:
        lines.append("- no train-CV candidate satisfied the hard gates\n\n")
    else:
        lines.append(f"- selected config = `{selected['candidate']}`\n")
        lines.append(f"- lr = `{selected['lr']}`\n")
        lines.append(f"- max_updates = `{selected['max_updates']}`\n")
        lines.append(f"- rot_scale = `{selected['rot_scale']}`\n")
        lines.append(f"- gate_max = `{selected['gate_max']}`\n")
        lines.append(f"- residual_reg_w = `{selected['residual_reg_w']}`\n")
        lines.append(f"- mean CV ATE = `{selected['mean_ATE']}`\n")
        lines.append(f"- mean CV drift = `{selected['mean_drift']}`\n")
        lines.append(f"- mean CV path_ratio = `{selected['mean_path_ratio']}`\n\n")

    lines.append("## 5. Final test result\n\n")
    if final_metrics is None:
        lines.append("- final test eval not run because no valid train-CV selection was available\n\n")
    else:
        lines.append(f"- drift = `{final_metrics['drift']}`\n")
        lines.append(f"- ATE = `{final_metrics['ATE']}`\n")
        lines.append(f"- path_ratio = `{final_metrics['path_ratio']}`\n")
        lines.append(f"- RPE_rot = `{final_metrics['RPE_rot']}`\n")
        lines.append(f"- RPE_trans_dir = `{final_metrics['RPE_trans_dir']}`\n")
        lines.append(f"- RPE_trans_mag = `{final_metrics['RPE_trans_mag']}`\n")
        lines.append(f"- rot = `{final_metrics['rot']}`\n")
        lines.append(f"- tdir_abs = `{final_metrics['tdir_abs']}`\n")
        lines.append(f"- tdir_local_A_abs = `{final_metrics['tdir_local_A_abs']}`\n")
        lines.append(
            f"- tmag P10/P50/P90 = `{final_metrics['tmag_p10']}` / `{final_metrics['tmag_p50']}` / `{final_metrics['tmag_p90']}`\n"
        )
        lines.append(f"- selected_k = `{final_metrics['selected_k']}`\n")
        lines.append(f"- num_pairs = `{final_metrics['num_pairs']}`\n")
        lines.append(f"- num_chains = `{final_metrics['num_chains']}`\n")
        lines.append(f"- missing/unexpected = `{final_metrics['missing']}` / `{final_metrics['unexpected']}`\n")
        lines.append(f"- tdir before/after diff = `{final_run['tdir_diff']}`\n")
        lines.append(f"- tmag before/after diff = `{final_run['tmag_diff']}`\n\n")

    lines.append("## 6. Optimizer/freeze audit\n\n")
    if final_run is None:
        lines.append("- no final run audit available\n\n")
    else:
        lines.append(f"- trainable parameter count = `{final_run['trainable_param_count']}`\n")
        lines.append(f"- optimizer parameter count = `{final_run['optimizer_param_count']}`\n")
        lines.append(f"- forbidden trainable params count = `{final_run['forbidden_trainable_params_count']}`\n")
        lines.append(f"- optimizer only rot residual/gate = `{final_run['optimizer_only_rot_gate']}`\n")
        lines.append(f"- trainable parameter names = `{final_run['trainable_param_names']}`\n\n")

    lines.append("## 7. Tdir/Tmag invariance audit\n\n")
    if final_run is None:
        lines.append("- no final run invariance audit available\n\n")
    else:
        lines.append(f"- tdir after == before = `{abs(final_run['tdir_diff']) <= 1.0e-9}`\n")
        lines.append(f"- tmag after == before = `{abs(final_run['tmag_diff']) <= 1.0e-9}`\n")
        lines.append(f"- delta_tdir_norm = `{final_run['delta_tdir_norm']}`\n")
        lines.append(f"- delta_rot_norm = `{final_run['delta_rot_norm']}`\n")
        lines.append(f"- gate_mean = `{final_run['gate_mean']}`\n\n")

    lines.append("## 8. Verdict\n\n")
    lines.append(f"- verdict = `{verdict}`\n")
    if final_metrics is not None:
        lines.append(f"- final ATE < S2b = `{final_metrics['ATE'] < S2B_ATE}`\n")
        lines.append(f"- final drift <= 1.35 = `{final_metrics['drift'] <= 1.35}`\n")
        lines.append(f"- final path_ratio >= 0.90 = `{final_metrics['path_ratio'] >= 0.90}`\n\n")
    else:
        lines.append("- final ATE < S2b = `False`\n")
        lines.append("- final drift <= 1.35 = `False`\n")
        lines.append("- final path_ratio >= 0.90 = `False`\n\n")

    lines.append("## 9. Whether to enter larger S3a1 run\n\n")
    lines.append(f"- recommendation = `{verdict == 'SUCCESS'}`\n\n")

    lines.append("## 10. Whether to replace S2b\n\n")
    replace_s2b = bool(final_metrics is not None and verdict == "SUCCESS")
    lines.append(f"- replace S2b = `{replace_s2b}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    subset_root = _ensure_subset_data_root()
    rows: List[Dict[str, Any]] = []
    configs = _configs()

    fold_plan = [
        ("seq01", SEQ02_TRAIN_SEED),
        ("seq02", SEQ01_TRAIN_SEED),
    ]

    for cfg in configs:
        cand_dir = OUT_ROOT / _candidate_key(cfg)
        for heldout_seq, split_seed in fold_plan:
            fold_dir = cand_dir / _fold_name(heldout_seq)
            exp_name = f"{EXP_NAME}/{_candidate_key(cfg)}/{_fold_name(heldout_seq)}"
            summary = _run_train(exp_name, subset_root, split_seed, cfg, fold_dir)
            rows.append(_row_from_summary(cfg, heldout_seq, summary, fold_dir))

    aggregates = [_aggregate_candidate(cfg, [r for r in rows if r["candidate"] == _candidate_key(cfg)]) for cfg in configs]
    selected = _select_candidate(aggregates)

    final_run = None
    final_metrics = None
    verdict = "FAIL"
    if selected is not None:
        cfg = next(c for c in configs if _candidate_key(c) == selected["candidate"])
        final_dir = OUT_ROOT / "final_test" / selected["candidate"]
        summary = _run_train(f"{EXP_NAME}/final_test/{selected['candidate']}", REPO_ROOT / "data", 3407, cfg, final_dir)
        final_run = _row_from_summary(cfg, "test", summary, final_dir)
        final_ckpt = final_dir / "final.pt"
        final_metrics = _eval_checkpoint(final_ckpt)
        final_metrics["tdir_diff"] = final_run["tdir_diff"]
        final_metrics["tmag_diff"] = final_run["tmag_diff"]
        verdict = _verdict(
            final_metrics,
            selected,
            final_run["optimizer_only_rot_gate"],
            abs(final_run["tdir_diff"]) <= 1.0e-9,
            abs(final_run["tmag_diff"]) <= 1.0e-9,
        )

    _write_report(rows, aggregates, selected, final_run, final_metrics, verdict)
    _write_json(
        OUT_ROOT / "summary.json",
        {
            "selected": selected,
            "final_run": final_run,
            "final_metrics": final_metrics,
            "verdict": verdict,
            "rows": rows,
            "aggregates": aggregates,
        },
    )
    shutil.rmtree(TMP_DATA_ROOT, ignore_errors=True)
    print(
        json.dumps(
            {
                "selected_config": None if selected is None else selected["candidate"],
                "final_metrics": final_metrics,
                "verdict": verdict,
                "report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
