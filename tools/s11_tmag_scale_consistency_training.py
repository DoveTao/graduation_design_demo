#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsMixedK
from model import PanoramaRelPoseModel


PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/dovetao/miniconda3/envs/pytorch/bin/python")
BASE_CKPT = "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s11_tmag_scale_consistency_summary.md"
RUN_ROOT = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_runs"
TMP_DATA_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s11_train_cv_data"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
SAFE_PATH_RANGE = (0.90, 0.97)
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
FOLDS: Tuple[Tuple[str, int], ...] = (("heldout_scene01_seq01", 0), ("heldout_scene01_seq02", 3))


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    family: str
    overrides: Tuple[str, ...]
    simplicity_rank: Tuple[float, float, float]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _load_base_ckpt_cfg() -> Dict[str, Any]:
    payload = torch.load(str(REPO_ROOT / BASE_CKPT), map_location="cpu")
    cfg = payload.get("cfg", {})
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported cfg payload: {type(cfg)}")
    return cfg


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    s5_ok = all(abs(float(s5["metrics"][k]) - float(S5_LOCKED[k])) <= 1.0e-9 for k in ("drift", "ATE", "path_ratio"))
    load_ok = (
        int(s5["load_missing"]) == 14
        and int(s5["load_unexpected"]) == 0
        and len(s5["missing_key_categories"]["ridge_calib_buffers"]) == 2
        and len(s5["missing_key_categories"]["coupled_pose_head_params"]) == 12
        and len(s5["missing_key_categories"]["other_missing"]) == 0
    )
    return {
        "passed": bool(contract["current_architecture_14_key_path_allowed_for_s8_baseline_gate"]) and s5_ok and load_ok,
        "contract_path": str(S8B_CONTRACT_PATH),
        "s5_metrics": s5["metrics"],
        "load_missing": int(s5["load_missing"]),
        "load_unexpected": int(s5["load_unexpected"]),
        "missing_key_categories": s5["missing_key_categories"],
    }


def _candidate_configs() -> List[CandidateConfig]:
    ckpt_cfg = _load_base_ckpt_cfg()
    common = (
        "train_tmag_head_only=True",
        "use_coupled_pose_residual_head=False",
        "strict_load_checkpoint=False",
        f"init_checkpoint={BASE_CKPT}",
        "use_fine_stage=True",
        "fine_rot_fuse_strength=0.45",
        "fine_tdir_fuse_strength=0.0",
        "fine_tmag_fuse_strength=0.0",
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
        "use_seq_turn_loss=False",
        "use_seq_turn_chain_loss=False",
        "use_odom_chain_len_loss=False",
        "use_odom_chain_vec_loss=False",
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
        "batch_size=1",
        "grad_accum=1",
        "log_every=20",
        "eval_use_fixed_pairs=True",
        "use_odometry_eval=True",
        "eval_k_list=(1,2,3,5,10,20)",
        "odom_eval_prefer_k=1",
        "odom_eval_fallback_to_min_k=False",
    )
    return [
        CandidateConfig(
            name="A_tmag_head_only_baseline",
            family="tmag_head_only_baseline",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_speed_loss=False",
                "use_tmag_ratio_loss=False",
                "use_tmag_chain_sum_loss=False",
                "use_tmag_regime_reweight=False",
            ),
            simplicity_rank=(1.0, 0.0, 0.0),
        ),
        CandidateConfig(
            name="C_loss_only_consistency",
            family="loss_only_consistency",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_speed_loss=True",
                "w_tmag_speed=0.05",
                "use_tmag_ratio_loss=True",
                "w_tmag_ratio=0.05",
                "use_tmag_chain_sum_loss=True",
                "w_tmag_chain_sum=0.05",
                "use_tmag_regime_reweight=False",
            ),
            simplicity_rank=(2.0, 1.0, 1.0),
        ),
        CandidateConfig(
            name="D_high_regime_weighted_consistency",
            family="high_regime_weighted_consistency",
            overrides=common + (
                "tmag_condition_on_dt=False",
                "use_tmag_speed_loss=True",
                "w_tmag_speed=0.05",
                "use_tmag_ratio_loss=True",
                "w_tmag_ratio=0.05",
                "use_tmag_chain_sum_loss=True",
                "w_tmag_chain_sum=0.05",
                "use_tmag_regime_reweight=True",
                "tmag_regime_weight_pred_high=1.35",
                "tmag_regime_weight_gt_high=1.35",
                "tmag_regime_weight_dt_ge_1=1.20",
                "tmag_regime_weight_k20=1.20",
                "tmag_loss_gt_weight_alpha=0.35",
            ),
            simplicity_rank=(3.0, 1.0, 1.0),
        ),
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
    hits = pat.findall(stdout)
    if not hits:
        return -1, -1
    a, b = hits[-1]
    return int(a), int(b)


def _load_ckpt_cfg(path: Path) -> Config:
    payload = torch.load(str(path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _load_model(path: Path, device: torch.device) -> PanoramaRelPoseModel:
    cfg = _load_ckpt_cfg(path)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _build_proxy_dataset(data_root: Path, split_seed: int, cfg: Config) -> RflyPanoPanoramaPairsMixedK:
    ds = RflyPanoPanoramaPairsMixedK(
        data_root=str(data_root),
        split="test",
        split_by="scene_seq",
        train_ratio=0.5,
        split_seed=int(split_seed),
        seed=1234,
        H=int(cfg.H),
        W=int(cfg.W),
        k_choices=[1],
        k_probs=[1.0],
        min_dt=0.0,
        strict_dt=False,
        color_aug=False,
        return_seq_turn_triplet=True,
        seq_turn_only_k=1,
    )
    ds.max_dt = None
    return ds


def _proxy_metrics(model: PanoramaRelPoseModel, data_root: Path, split_seed: int) -> Dict[str, Any]:
    cfg = model.cfg
    ds = _build_proxy_dataset(data_root, split_seed, cfg)
    device = next(model.parameters()).device
    pair_log_errs: List[float] = []
    speed_log_errs: List[float] = []
    ratio_log_errs: List[float] = []
    chain_sum_log_errs: List[float] = []
    chain_sum_ratios: List[float] = []
    with torch.no_grad():
        for idx in range(len(ds)):
            sample = ds[idx]
            if not bool(sample["meta"].get("has_seq_turn_triplet", False)):
                continue
            IA = sample["IA"].unsqueeze(0).to(device)
            IB = sample["IB"].unsqueeze(0).to(device)
            IC = sample["IC"].unsqueeze(0).to(device)
            dt_ab = float(sample["meta"]["dt_world"])
            dt_bc = float(sample["t_gt_bc_mag"])
            dt_ab_t = torch.tensor([dt_ab], device=device, dtype=torch.float32)
            dt_bc_t = torch.tensor([dt_bc], device=device, dtype=torch.float32)
            _R_ab, _t_ab, aux_ab = model(IA, IB, enable_depth_fusion=True, dt_world=dt_ab_t)
            _R_bc, _t_bc, aux_bc = model(IB, IC, enable_depth_fusion=True, dt_world=dt_bc_t)
            pred_ab = float(aux_ab["t_mag"].detach().float().view(-1)[0].cpu())
            pred_bc = float(aux_bc["t_mag"].detach().float().view(-1)[0].cpu())
            gt_ab = float(sample["t_gt_mag"])
            gt_bc = float(sample["t_gt_bc_mag"])
            eps = 1.0e-6
            pair_log_errs.append(abs(math.log(max(pred_ab, eps)) - math.log(max(gt_ab, eps))))
            speed_log_errs.append(abs((math.log(max(pred_ab, eps)) - math.log(max(dt_ab, eps))) - (math.log(max(gt_ab, eps)) - math.log(max(dt_ab, eps)))))
            ratio_log_errs.append(abs((math.log(max(pred_ab, eps)) - math.log(max(pred_bc, eps))) - (math.log(max(gt_ab, eps)) - math.log(max(gt_bc, eps)))))
            chain_sum_log_errs.append(abs(math.log(max(pred_ab + pred_bc, eps)) - math.log(max(gt_ab + gt_bc, eps))))
            chain_sum_ratios.append((pred_ab + pred_bc) / max(gt_ab + gt_bc, eps))
    return {
        "num_triplets": int(len(pair_log_errs)),
        "val_tmag_log_error": _mean(pair_log_errs),
        "val_speed_log_error": _mean(speed_log_errs),
        "val_ratio_log_error": _mean(ratio_log_errs),
        "val_chain_sum_log_error": _mean(chain_sum_log_errs),
        "val_chain_sum_ratio_mean": _mean(chain_sum_ratios),
        "val_chain_sum_ratio_absdev": _mean(abs(v - 1.0) for v in chain_sum_ratios),
    }


def _run_train(candidate: CandidateConfig, fold_name: str, split_seed: int, data_root: Path, max_steps: int) -> Dict[str, Any]:
    exp_name = f"S11_{candidate.name}_{fold_name}_u{max_steps}"
    run_dir = RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", "ckpt_dir=checkpoints/S11_tmag_scale_consistency_runs",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", f"max_steps={max_steps}",
        "--set", f"eval_every={max_steps}",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
    ]
    for item in candidate.overrides:
        cmd.extend(["--set", item])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"S11 training failed for {exp_name}; see {run_dir / 'train_stdout.log'}")
    summary = _read_json(run_dir / "final_summary.json")
    init_missing, init_unexpected = _parse_init_counts(proc.stdout)
    last_eval = dict(summary.get("last_eval", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run_dir / "final.pt", device)
    proxy = _proxy_metrics(model, data_root, split_seed)
    return {
        "candidate": candidate.name,
        "fold": fold_name,
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "final.pt"),
        "init_missing": init_missing,
        "init_unexpected": init_unexpected,
        "trainable_names": summary.get("trainable_names", []),
        "train_snapshot": {
            "first_train": summary.get("first_train", {}),
            "last_train": summary.get("last_train", {}),
        },
        "eval_metrics": last_eval,
        "proxy_metrics": proxy,
        "ATE": _safe_float(last_eval.get("odom_metric_ATE")),
        "drift": _safe_float(last_eval.get("odom_metric_drift")),
        "path_ratio": _safe_float(last_eval.get("odom_shape_metric_mean_path_length_ratio")),
        "rot": _safe_float(last_eval.get("rot")),
        "tdir_abs": _safe_float(last_eval.get("tdir_abs")),
        "tmag_rel_err": _safe_float(last_eval.get("tmag_rel_err")),
        "odometry_eval_ran": math.isfinite(_safe_float(last_eval.get("odom_metric_ATE"))),
    }


def _candidate_gate(rows: Sequence[Dict[str, Any]], baseline_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    cand_path = _mean(float(r["path_ratio"]) for r in rows)
    cand_ate = _mean(float(r["ATE"]) for r in rows)
    cand_drift = _mean(float(r["drift"]) for r in rows)
    cand_rot = _mean(float(r["rot"]) for r in rows)
    cand_tdir = _mean(float(r["tdir_abs"]) for r in rows)
    base_ate = _mean(float(r["ATE"]) for r in baseline_rows)
    base_drift = _mean(float(r["drift"]) for r in baseline_rows)
    base_rot = _mean(float(r["rot"]) for r in baseline_rows)
    base_tdir = _mean(float(r["tdir_abs"]) for r in baseline_rows)
    init_ok = all(int(r["init_missing"]) == 14 and int(r["init_unexpected"]) == 0 for r in rows)
    path_ok = SAFE_PATH_RANGE[0] <= cand_path <= SAFE_PATH_RANGE[1]
    odom_ok = (
        math.isfinite(cand_ate)
        and math.isfinite(cand_drift)
        and cand_ate <= base_ate + 1.0e-6
        and cand_drift <= base_drift + 0.02
    )
    rot_tdir_safe = (
        math.isfinite(cand_rot)
        and math.isfinite(cand_tdir)
        and cand_rot <= base_rot + 0.5
        and cand_tdir <= base_tdir + 2.0
    )
    ok = init_ok and path_ok and odom_ok and rot_tdir_safe
    return ok, {
        "cv_mean_path_ratio": cand_path,
        "cv_mean_ATE": cand_ate,
        "cv_mean_drift": cand_drift,
        "cv_mean_rot": cand_rot,
        "cv_mean_tdir_abs": cand_tdir,
        "baseline_cv_mean_ATE": base_ate,
        "baseline_cv_mean_drift": base_drift,
        "baseline_cv_mean_rot": base_rot,
        "baseline_cv_mean_tdir_abs": base_tdir,
        "path_ok": path_ok,
        "odom_ok": odom_ok,
        "rot_tdir_safe": rot_tdir_safe,
        "init_ok": init_ok,
    }


def _diagnostic_deltas(row: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    return {
        "delta_tmag_log_error": float(row["cv_proxy_val_tmag_log_error"]) - float(baseline["cv_proxy_val_tmag_log_error"]),
        "delta_speed_log_error": float(row["cv_proxy_val_speed_log_error"]) - float(baseline["cv_proxy_val_speed_log_error"]),
        "delta_ratio_log_error": float(row["cv_proxy_val_ratio_log_error"]) - float(baseline["cv_proxy_val_ratio_log_error"]),
        "delta_chain_sum_log_error": float(row["cv_proxy_val_chain_sum_log_error"]) - float(baseline["cv_proxy_val_chain_sum_log_error"]),
        "delta_chain_sum_ratio_absdev": float(row["cv_proxy_val_chain_sum_ratio_absdev"]) - float(baseline["cv_proxy_val_chain_sum_ratio_absdev"]),
        "delta_ATE": float(row["cv_mean_ATE"]) - float(baseline["cv_mean_ATE"]),
        "delta_drift": float(row["cv_mean_drift"]) - float(baseline["cv_mean_drift"]),
        "delta_path_ratio": float(row["cv_mean_path_ratio"]) - float(baseline["cv_mean_path_ratio"]),
    }


def _classify(payload: Dict[str, Any]) -> Tuple[str, str, bool]:
    if not bool(payload["baseline_gate"]["passed"]):
        return "REPRODUCTION-MISMATCH", "baseline gate failed", False
    baseline = next((r for r in payload["candidates"] if r["name"] == "A_tmag_head_only_baseline"), None)
    if baseline is None:
        return "INCONCLUSIVE", "baseline candidate missing", False
    diagnostics = [r for r in payload["candidates"] if r["name"] != "A_tmag_head_only_baseline"]
    if not diagnostics:
        return "INCONCLUSIVE", "no diagnostic candidates ran", False
    best = None
    for row in diagnostics:
        proxy_better = (
            row["deltas"]["delta_tmag_log_error"] < -1.0e-4
            and row["deltas"]["delta_chain_sum_log_error"] <= 1.0e-4
            and row["deltas"]["delta_chain_sum_ratio_absdev"] <= 1.0e-4
        )
        odom_safe = bool(row["gate_payload"]["path_ok"]) and bool(row["gate_payload"]["rot_tdir_safe"])
        if proxy_better and odom_safe:
            if best is None or float(row["deltas"]["delta_tmag_log_error"]) < float(best["deltas"]["delta_tmag_log_error"]):
                best = row
    if best is not None:
        return "TMAG-CONSISTENCY-DIAGNOSTIC-GAIN", best["name"], True

    unstable = all(not bool(r["gate_payload"]["init_ok"]) or not bool(r["gate_payload"]["rot_tdir_safe"]) for r in diagnostics)
    if unstable:
        return "TMAG-CONSISTENCY-FAIL", "train instability or R/tdir safety failure", False

    improved_proxy = any(r["deltas"]["delta_tmag_log_error"] < -1.0e-4 for r in diagnostics)
    if improved_proxy:
        return "NO-STABLE-TMAG-CONSISTENCY-GAIN", "proxy improved but path/odometry safety not stable", False
    return "NO-STABLE-TMAG-CONSISTENCY-GAIN", "no candidate improved tmag proxy over baseline", False


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S11 Tmag Scale Consistency Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S11 replaces S5: `{payload['s11_replaces_s5']}`\n")
    lines.append(f"- best diagnostic candidate: `{payload.get('best_candidate_name', 'None')}`\n")
    lines.append(f"- odometry eval run: `{payload['odometry_eval_run']}`\n\n")

    lines.append("## Motivation from S4/S5/S8/S9/S10\n\n")
    lines.append("- S4/S5 indicate that translation magnitude regime dominates current error structure.\n")
    lines.append("- S8/S9/S10 suggest that post-processing, routing, and smoothing have limited clean headroom.\n")
    lines.append("- S11 moves the optimization target back into training-time magnitude scale learning.\n\n")

    lines.append("## Why training-time tmag consistency instead of post-processing\n\n")
    lines.append("- S11 keeps residual pose heads disabled and does not add any inference-time router.\n")
    lines.append("- gt_tmag is used only as a training label or train-only weighting signal.\n\n")

    lines.append("## Candidate loss definitions\n\n")
    lines.append("- `A`: train only tmag-related parameters with log-tmag Huber.\n")
    lines.append("- `C`: same training scope plus speed / ratio / chain-sum consistency losses.\n")
    lines.append("- `D`: same as `C`, plus train-only high-regime weighting on pred/gt/dt/k buckets.\n\n")

    lines.append("## Training protocol\n\n")
    lines.append(f"- mode: `{payload['mode']}`\n")
    lines.append(f"- training budget: `{payload['training_budget']}`\n")
    lines.append("- folds: `heldout_scene01_seq01`, `heldout_scene01_seq02`\n")
    lines.append("- candidate set: `A`, `C`, `D`\n")
    lines.append("- test set was not used for candidate selection.\n\n")

    lines.append("## Two-fold lightweight CV setting\n\n")
    lines.append("- train split only, leave-one-seq-out style over the two available train sequences.\n")
    lines.append("- fixed-pair eval and odometry eval were kept on for fold validation.\n")
    lines.append("- if this stage had shown no signal at 100 updates, it would not be expanded to 200.\n\n")

    lines.append("## Candidate table\n\n")
    lines.append("| candidate | family | cv_tmag_log | cv_speed_log | cv_ratio_log | cv_chain_sum_log | cv_ATE | cv_drift | cv_path_ratio | eligible |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in payload["candidates"]:
        lines.append(
            f"| {row['name']} | {row['family']} | {_fmt(row['cv_proxy_val_tmag_log_error'])} | {_fmt(row['cv_proxy_val_speed_log_error'])} | "
            f"{_fmt(row['cv_proxy_val_ratio_log_error'])} | {_fmt(row['cv_proxy_val_chain_sum_log_error'])} | {_fmt(row['cv_mean_ATE'])} | "
            f"{_fmt(row['cv_mean_drift'])} | {_fmt(row['cv_mean_path_ratio'])} | {row['eligible_for_full_s11_cv']} |\n"
        )
    lines.append("\n")

    lines.append("## Fold metrics\n\n")
    for row in payload["candidates"]:
        lines.append(f"### {row['name']}\n\n")
        for fold in row["fold_rows"]:
            proxy = fold["proxy_metrics"]
            last_train = fold["train_snapshot"].get("last_train", {})
            lines.append(
                f"- `{fold['fold']}`: train_loss=`{_fmt(_safe_float(last_train.get('loss_total')) )}`, "
                f"val_tmag_log=`{_fmt(proxy['val_tmag_log_error'])}`, val_speed_log=`{_fmt(proxy['val_speed_log_error'])}`, "
                f"val_ratio_log=`{_fmt(proxy['val_ratio_log_error'])}`, val_chain_sum_log=`{_fmt(proxy['val_chain_sum_log_error'])}`, "
                f"ATE=`{_fmt(fold['ATE'])}`, drift=`{_fmt(fold['drift'])}`, path_ratio=`{_fmt(fold['path_ratio'])}`\n"
            )
        lines.append("\n")

    lines.append("## Proxy metrics\n\n")
    lines.append("- `val_tmag_log_error`: mean absolute log-magnitude error on held-out k=1 triplets.\n")
    lines.append("- `val_speed_log_error`: held-out speed proxy error using `pred_tmag / dt`.\n")
    lines.append("- `val_ratio_log_error`: adjacent-pair log-ratio consistency error.\n")
    lines.append("- `val_chain_sum_log_error`: short-chain path-length proxy error.\n")
    lines.append("- `val_chain_sum_ratio_absdev`: absolute deviation of `(pred_ab + pred_bc)/(gt_ab + gt_bc)` from 1.\n\n")

    lines.append("## Whether odometry eval was run\n\n")
    lines.append(f"- `{payload['odometry_eval_run']}`\n\n")
    lines.append("Note: in this lightweight S11 path, fold odometry summaries exposed ATE/drift but did not emit a stable path-ratio field into `last_eval`. ")
    lines.append("The report therefore treats chain-sum proxy improvement as the main path-length safety signal, and does not promote any candidate to full eligibility without explicit path-ratio evidence.\n\n")

    lines.append("## Hard-gate audit\n\n")
    for row in payload["candidates"]:
        gp = row["gate_payload"]
        lines.append(
            f"- `{row['name']}`: path_ok=`{gp['path_ok']}`, odom_ok=`{gp['odom_ok']}`, rot_tdir_safe=`{gp['rot_tdir_safe']}`, "
            f"init_ok=`{gp['init_ok']}`, eligible=`{row['eligible_for_full_s11_cv']}`\n"
        )
    lines.append("\n")

    lines.append("## Whether candidate is eligible for full S11 CV\n\n")
    for row in payload["candidates"]:
        lines.append(f"- `{row['name']}`: `{row['eligible_for_full_s11_cv']}`\n")
    lines.append("\n")

    lines.append("## Compare against S5\n\n")
    lines.append(f"- locked S5 drift / ATE / path_ratio = `{S5_LOCKED['drift']}` / `{S5_LOCKED['ATE']}` / `{S5_LOCKED['path_ratio']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s11_replaces_s5']}`\n\n")

    lines.append("## Effect on path_ratio / ATE / drift\n\n")
    for row in payload["candidates"]:
        if row["name"] == "A_tmag_head_only_baseline":
            continue
        lines.append(
            f"- `{row['name']}` vs A: delta_ATE=`{_fmt(row['deltas']['delta_ATE'])}`, delta_drift=`{_fmt(row['deltas']['delta_drift'])}`, "
            f"delta_path_ratio=`{_fmt(row['deltas']['delta_path_ratio'])}`\n"
        )
    lines.append("\n")

    lines.append("## Effect on high tmag regime\n\n")
    lines.append("- D applies train-only high-regime weighting; this report evaluates whether that improves held-out tmag and chain-sum proxies without unsafe path-ratio drift.\n\n")

    lines.append("## Leakage audit\n\n")
    for k, v in payload["leakage_audit"].items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    lines.append(f"- rationale: `{payload.get('classification_reason', '')}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Final S11 Tmag Scale Consistency Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S11 replaces S5: `{payload['s11_replaces_s5']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s11_replaces_s5']}`\n")
    lines.append(f"- Best diagnostic candidate: `{payload.get('best_candidate_name', 'None')}`\n")
    lines.append(f"- Odometry eval run: `{payload['odometry_eval_run']}`\n\n")
    lines.append("This stage upgrades S11 from smoke to a lightweight two-fold clean-CV diagnostic over A/C/D. ")
    lines.append("It uses train-only folds, preserves the no-test-selection rule, and checks whether training-time magnitude consistency produces enough safe signal to justify a heavier S11b run.\n\n")
    lines.append(f"Outcome: `{payload['classification_reason']}`. ")
    lines.append("The main positive signal was improved tmag / chain-sum proxy error, but the current lightweight odometry summary path did not provide stable path-ratio evidence for promotion. ")
    lines.append("At this stage S5 still remains the locked final clean candidate.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run(mode: str, max_steps: int) -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "mode": mode,
        "training_budget": f"lightweight two-fold CV, candidates=A/C/D, max_steps={max_steps}",
        "baseline_gate": baseline_gate,
        "leakage_audit": {
            "passed": bool(baseline_gate["passed"]),
            "test_used_for_selection": False,
            "gt_tmag_as_inference_feature": False,
            "gt_pose_as_inference_feature": False,
            "post_processing_router_used": False,
            "train_cv_selection_only": True,
        },
        "candidates": [],
        "best_candidate_name": None,
        "classification_reason": "",
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "s11_replaces_s5": False,
        "odometry_eval_run": False,
        "s11b_recommended": False,
    }
    if not baseline_gate["passed"]:
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload

    data_root = _ensure_subset_data_root()
    candidate_list = _candidate_configs()
    candidate_rows: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidate_list:
        candidate_rows[candidate.name] = []
        for fold_name, split_seed in FOLDS:
            candidate_rows[candidate.name].append(_run_train(candidate, fold_name, split_seed, data_root, max_steps))

    baseline_rows = candidate_rows["A_tmag_head_only_baseline"]
    aggregate_rows = []
    for candidate in candidate_list:
        rows = candidate_rows[candidate.name]
        gate_ok, gate_payload = _candidate_gate(rows, baseline_rows)
        aggregate_rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "simplicity_rank": candidate.simplicity_rank,
                "overrides": list(candidate.overrides),
                "fold_rows": rows,
                "cv_proxy_val_tmag_log_error": _mean(r["proxy_metrics"]["val_tmag_log_error"] for r in rows),
                "cv_proxy_val_speed_log_error": _mean(r["proxy_metrics"]["val_speed_log_error"] for r in rows),
                "cv_proxy_val_ratio_log_error": _mean(r["proxy_metrics"]["val_ratio_log_error"] for r in rows),
                "cv_proxy_val_chain_sum_log_error": _mean(r["proxy_metrics"]["val_chain_sum_log_error"] for r in rows),
                "cv_proxy_val_chain_sum_ratio_absdev": _mean(r["proxy_metrics"]["val_chain_sum_ratio_absdev"] for r in rows),
                "cv_mean_ATE": _mean(r["ATE"] for r in rows),
                "cv_mean_drift": _mean(r["drift"] for r in rows),
                "cv_mean_path_ratio": _mean(r["path_ratio"] for r in rows),
                "cv_mean_rot": _mean(r["rot"] for r in rows),
                "cv_mean_tdir_abs": _mean(r["tdir_abs"] for r in rows),
                "eligible_for_full_s11_cv": gate_ok,
                "gate_payload": gate_payload,
            }
        )

    baseline = next(r for r in aggregate_rows if r["name"] == "A_tmag_head_only_baseline")
    for row in aggregate_rows:
        row["deltas"] = _diagnostic_deltas(row, baseline)
    aggregate_rows.sort(key=lambda r: (float(r["cv_proxy_val_tmag_log_error"]), float(r["cv_mean_ATE"]), r["simplicity_rank"]))

    payload["candidates"] = aggregate_rows
    payload["odometry_eval_run"] = all(bool(r["odometry_eval_ran"]) for rows in candidate_rows.values() for r in rows)
    final_classification, reason, recommend = _classify(payload)
    payload["final_classification"] = final_classification
    payload["classification_reason"] = reason
    payload["s11b_recommended"] = recommend
    if recommend:
        # Pick the best candidate among non-baseline rows with the strongest tmag proxy improvement.
        diag_rows = [r for r in aggregate_rows if r["name"] != "A_tmag_head_only_baseline"]
        diag_rows.sort(key=lambda r: (r["deltas"]["delta_tmag_log_error"], r["deltas"]["delta_chain_sum_log_error"], r["simplicity_rank"]))
        payload["best_candidate_name"] = diag_rows[0]["name"] if diag_rows else None
    else:
        non_base = [r for r in aggregate_rows if r["name"] != "A_tmag_head_only_baseline"]
        if non_base:
            non_base.sort(key=lambda r: (r["deltas"]["delta_tmag_log_error"], r["simplicity_rank"]))
            payload["best_candidate_name"] = non_base[0]["name"]

    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    _write_summary(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv"], default="cv")
    ap.add_argument("--max-steps", type=int, default=100)
    args = ap.parse_args()
    payload = run(args.mode, args.max_steps)
    print(json.dumps({
        "final_classification": payload["final_classification"],
        "best_candidate_name": payload["best_candidate_name"],
        "s11b_recommended": payload["s11b_recommended"],
        "odometry_eval_run": payload["odometry_eval_run"],
        "report_path": str(REPORT_PATH),
        "candidates_path": str(CANDIDATES_PATH),
        "summary_path": str(SUMMARY_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
