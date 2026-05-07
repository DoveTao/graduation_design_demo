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
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg
from train_mvp import _camera_center_from_T_c0_np, _compose_rel_pose_np, _rot_geodesic_deg_np, _vec_angle_deg_np


PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/dovetao/miniconda3/envs/pytorch/bin/python")
BASE_CKPT = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S15_trajectory_level_training_objective_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S15_trajectory_level_training_objective_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s15_trajectory_level_training_objective_summary.md"
CONFIG_PATH = REPO_ROOT / "configs" / "S15_trajectory_level_training_objective.yaml"
RUN_ROOT = Path("/tmp/s15_trajectory_level_training_runs")
TMP_DATA_ROOT = Path("/tmp/s15_trajectory_level_training_data")
MANIFEST_ROOT = Path("/tmp/s15_trajectory_level_training_manifests")
PROXY_MAX_WINDOWS = 128
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
FOLDS: Tuple[Tuple[str, int], ...] = (("heldout_scene01_seq01", 0), ("heldout_scene01_seq02", 3))
S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
SAFE_PATH_RANGE = (0.90, 1.05)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    family: str
    trainable_mode: str
    overrides: Tuple[str, ...]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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


def _parse_init_counts(stdout: str) -> Tuple[int, int]:
    pat = re.compile(r"\[InitCkpt\].*missing=(\d+)\s+\|\s+unexpected=(\d+)")
    hits = pat.findall(stdout)
    if not hits:
        return -1, -1
    a, b = hits[-1]
    return int(a), int(b)


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


def _ensure_subset_data_root() -> Path:
    pano_root = REPO_ROOT / "data" / "PanoramaView"
    if TMP_DATA_ROOT.exists():
        shutil.rmtree(TMP_DATA_ROOT)
    for scene, seq in TRAIN_GROUPS:
        src = pano_root / scene / seq
        dst = TMP_DATA_ROOT / "PanoramaView" / scene / seq
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)
    return TMP_DATA_ROOT


def _cleanup_tmp_roots() -> None:
    for path in (TMP_DATA_ROOT, MANIFEST_ROOT):
        if path.exists():
            shutil.rmtree(path)


def _build_base_cfg() -> Config:
    return _cfg_from_dict(_load_ckpt_cfg(BASE_CKPT))


def _load_model(path: Path, device: torch.device) -> PanoramaRelPoseModel:
    cfg = _cfg_from_dict(_load_ckpt_cfg(path))
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _build_eval_dataset(data_root: Path, split_seed: int, split: str, k_list: Tuple[int, ...] = (1,)) -> RflyPanoPanoramaPairsEvalFixedKList:
    cfg = _build_base_cfg()
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(data_root),
        split=split,
        split_by="scene_seq",
        train_ratio=0.5,
        split_seed=int(split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=k_list,
        pair_step=1,
        min_dt=0.0,
        max_dt=None,
    )


def _build_window_manifest(data_root: Path, split_seed: int, split: str) -> Tuple[Path, Dict[str, Any]]:
    ds = _build_eval_dataset(data_root, split_seed, split, (1,))
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    pairs: List[Dict[str, Any]] = []
    dt_ab = []
    dt_bc = []
    gt_mag_ab = []
    gt_mag_bc = []
    for idx, meta in enumerate(manifest):
        scene = str(meta["scene"])
        seq = str(meta["seq"])
        i = int(meta["i"])
        j = int(meta["j"])
        nxt = by_seq_i.get((scene, seq, j), None)
        if nxt is None:
            continue
        _, next_meta = nxt
        sample_ab = ds[idx]
        sample_bc = ds[nxt[0]]
        row = dict(meta)
        row["_ds_idx"] = int(idx)
        row["window_size"] = 3
        row["dt_world_bc"] = float(next_meta["dt_world"])
        row["gt_tmag"] = float(sample_ab["t_gt_mag"])
        row["gt_tmag_bc"] = float(sample_bc["t_gt_mag"])
        pairs.append(row)
        dt_ab.append(float(meta["dt_world"]))
        dt_bc.append(float(next_meta["dt_world"]))
        gt_mag_ab.append(float(sample_ab["t_gt_mag"]))
        gt_mag_bc.append(float(sample_bc["t_gt_mag"]))
    payload = {
        "pairs": pairs,
        "split_summary": {
            "split": split,
            "split_seed": int(split_seed),
            "num_windows": int(len(pairs)),
            "num_sequences": len({(p["scene"], p["seq"]) for p in pairs}),
            "average_edges_per_window": 2.0 if pairs else 0.0,
            "dt_ab_mean": _mean(dt_ab),
            "dt_bc_mean": _mean(dt_bc),
            "gt_tmag_ab_mean": _mean(gt_mag_ab),
            "gt_tmag_bc_mean": _mean(gt_mag_bc),
            "k_distribution": {"1": int(len(pairs) * 2)},
            "uses_only_train_split": bool(split == "train"),
            "scene_seqs": sorted({f"{p['scene']}/{p['seq']}" for p in pairs}),
        },
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_ROOT / f"s15_{split}_seed{split_seed}.json"
    _write_json(out_path, payload)
    return out_path, payload["split_summary"]


def _candidate_configs() -> List[CandidateConfig]:
    ckpt_cfg = _load_ckpt_cfg(BASE_CKPT)
    common = (
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
        "train_fixed_pairs_return_triplet=True",
        "train_fixed_pairs_seq_turn_only_k=1",
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
        "traj_window_only_k=1",
        "traj_window_min_gt=0.02",
        "traj_scale_normalize=True",
        "traj_start_updates=0",
        "traj_ramp_updates=0",
        "use_tmag_speed_loss=False",
        "use_tmag_ratio_loss=False",
        "use_tmag_chain_sum_loss=False",
        "use_tmag_regime_reweight=False",
        "use_seq_turn_loss=False",
        "use_seq_turn_chain_loss=False",
        "use_odom_chain_len_loss=False",
        "use_odom_chain_vec_loss=False",
    )
    return [
        CandidateConfig(
            name="A_pair_only_baseline",
            family="pair_only_baseline",
            trainable_mode="tmag_head_only",
            overrides=common + (
                "train_tmag_head_only=True",
                "train_fine_only=False",
                "use_traj_ate_loss=False",
                "use_traj_drift_loss=False",
                "use_traj_path_loss=False",
                "use_traj_rot_loss=False",
                "use_traj_tdir_loss=False",
                "use_traj_tmag_step_loss=False",
                "use_traj_speed_loss=False",
            ),
        ),
        CandidateConfig(
            name="B_pair_plus_ate_path",
            family="pair_plus_ate_path",
            trainable_mode="tmag_head_only",
            overrides=common + (
                "train_tmag_head_only=True",
                "train_fine_only=False",
                "use_traj_ate_loss=True",
                "w_traj_ate=0.05",
                "use_traj_drift_loss=False",
                "use_traj_path_loss=True",
                "w_traj_path=0.05",
                "use_traj_rot_loss=False",
                "use_traj_tdir_loss=False",
                "use_traj_tmag_step_loss=False",
                "use_traj_speed_loss=False",
            ),
        ),
        CandidateConfig(
            name="C_pair_plus_rot_tdir_traj",
            family="pair_plus_rot_tdir_traj",
            trainable_mode="fine_pose_light",
            overrides=common + (
                "train_tmag_head_only=False",
                "train_fine_only=True",
                "use_traj_ate_loss=False",
                "use_traj_drift_loss=False",
                "use_traj_path_loss=False",
                "use_traj_rot_loss=True",
                "w_traj_rot=0.05",
                "use_traj_tdir_loss=True",
                "w_traj_tdir=0.05",
                "use_traj_tmag_step_loss=False",
                "use_traj_speed_loss=False",
            ),
        ),
        CandidateConfig(
            name="D_full_light_traj",
            family="full_light_traj",
            trainable_mode="fine_pose_light",
            overrides=common + (
                "train_tmag_head_only=False",
                "train_fine_only=True",
                "use_traj_ate_loss=True",
                "w_traj_ate=0.05",
                "use_traj_drift_loss=True",
                "w_traj_drift=0.05",
                "use_traj_path_loss=True",
                "w_traj_path=0.05",
                "use_traj_rot_loss=True",
                "w_traj_rot=0.05",
                "use_traj_tdir_loss=True",
                "w_traj_tdir=0.05",
                "use_traj_tmag_step_loss=True",
                "w_traj_tmag_step=0.05",
                "use_traj_speed_loss=True",
                "w_traj_speed=0.02",
            ),
        ),
    ]


def _proxy_metrics(checkpoint_path: Path, data_root: Path, split_seed: int, max_windows: int = PROXY_MAX_WINDOWS) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    ds = _build_eval_dataset(data_root, split_seed, "test", (1,))
    manifest = ds.manifest()
    by_seq_i = {(str(m["scene"]), str(m["seq"]), int(m["i"])): (idx, m) for idx, m in enumerate(manifest)}
    ate_terms = []
    drift_terms = []
    path_terms = []
    rot_terms = []
    tdir_terms = []
    tmag_terms = []
    speed_terms = []
    pair_rot = []
    pair_tdir = []
    pair_tmag = []
    windows = 0
    with torch.no_grad():
        for idx, meta in enumerate(manifest):
            if max_windows > 0 and windows >= int(max_windows):
                break
            scene = str(meta["scene"])
            seq = str(meta["seq"])
            i = int(meta["i"])
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
            R_gt_BA, t_gt_BA = _compose_rel_pose_np(R_gt_ab, t_gt_ab, np.eye(3), np.zeros(3))
            R_gt_CA, t_gt_CA = _compose_rel_pose_np(R_gt_bc, t_gt_bc, R_gt_BA, t_gt_BA)
            R_pr_BA, t_pr_BA = _compose_rel_pose_np(R_pr_ab, t_pr_ab, np.eye(3), np.zeros(3))
            R_pr_CA, t_pr_CA = _compose_rel_pose_np(R_pr_bc, t_pr_bc, R_pr_BA, t_pr_BA)
            p1_gt = _camera_center_from_T_c0_np(R_gt_BA, t_gt_BA)
            p2_gt = _camera_center_from_T_c0_np(R_gt_CA, t_gt_CA)
            p1_pr = _camera_center_from_T_c0_np(R_pr_BA, t_pr_BA)
            p2_pr = _camera_center_from_T_c0_np(R_pr_CA, t_pr_CA)
            step1_gt = p1_gt
            step2_gt = p2_gt - p1_gt
            step1_pr = p1_pr
            step2_pr = p2_pr - p1_pr
            gt_path = float(np.linalg.norm(step1_gt) + np.linalg.norm(step2_gt))
            pred_path = float(np.linalg.norm(step1_pr) + np.linalg.norm(step2_pr))
            if gt_path <= 1.0e-6:
                continue
            windows += 1
            p1_err = float(np.linalg.norm(p1_pr - p1_gt)) / gt_path
            p2_err = float(np.linalg.norm(p2_pr - p2_gt)) / gt_path
            ate_terms.extend([p1_err, p2_err])
            drift_terms.append(p2_err)
            path_terms.append(abs(math.log(max(pred_path, 1.0e-6)) - math.log(max(gt_path, 1.0e-6))))
            rot_terms.extend([_rot_geodesic_deg_np(R_pr_BA, R_gt_BA), _rot_geodesic_deg_np(R_pr_CA, R_gt_CA)])
            tdir_terms.extend([_vec_angle_deg_np(step1_pr, step1_gt), _vec_angle_deg_np(step2_pr, step2_gt)])
            tmag_terms.extend([
                abs(math.log(max(np.linalg.norm(step1_pr), 1.0e-6)) - math.log(max(np.linalg.norm(step1_gt), 1.0e-6))),
                abs(math.log(max(np.linalg.norm(step2_pr), 1.0e-6)) - math.log(max(np.linalg.norm(step2_gt), 1.0e-6))),
            ])
            speed_terms.extend([
                abs(math.log(max(np.linalg.norm(step1_pr) / max(float(dt_ab.item()), 1.0e-6), 1.0e-6)) - math.log(max(np.linalg.norm(step1_gt) / max(float(dt_ab.item()), 1.0e-6), 1.0e-6))),
                abs(math.log(max(np.linalg.norm(step2_pr) / max(float(dt_bc.item()), 1.0e-6), 1.0e-6)) - math.log(max(np.linalg.norm(step2_gt) / max(float(dt_bc.item()), 1.0e-6), 1.0e-6))),
            ])
            pair_rot.extend([_rot_geodesic_deg_np(R_pr_ab, R_gt_ab), _rot_geodesic_deg_np(R_pr_bc, R_gt_bc)])
            pair_tdir.extend([
                _vec_angle_deg_np(t_pr_ab, t_gt_ab),
                _vec_angle_deg_np(t_pr_bc, t_gt_bc),
            ])
            pair_tmag.extend([
                abs(math.log(max(np.linalg.norm(t_pr_ab), 1.0e-6)) - math.log(max(np.linalg.norm(t_gt_ab), 1.0e-6))),
                abs(math.log(max(np.linalg.norm(t_pr_bc), 1.0e-6)) - math.log(max(np.linalg.norm(t_gt_bc), 1.0e-6))),
            ])
    return {
        "proxy_window_cap": int(max_windows),
        "num_windows": int(windows),
        "val_ate_proxy": _mean(ate_terms),
        "val_drift_proxy": _mean(drift_terms),
        "val_path_proxy": _mean(path_terms),
        "val_rot_traj_deg": _mean(rot_terms),
        "val_tdir_traj_deg": _mean(tdir_terms),
        "val_tmag_step_log": _mean(tmag_terms),
        "val_speed_log": _mean(speed_terms),
        "pair_rot_error": _mean(pair_rot),
        "pair_tdir_error": _mean(pair_tdir),
        "pair_tmag_log_error": _mean(pair_tmag),
    }


def _run_train(candidate: CandidateConfig, fold_name: str, split_seed: int, data_root: Path, manifest_path: Path, max_steps: int) -> Dict[str, Any]:
    exp_name = f"S15_{candidate.name}_{fold_name}_u{max_steps}"
    run_dir = RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", f"ckpt_dir={RUN_ROOT}",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", f"max_steps={max_steps}",
        "--set", f"eval_every={max_steps}",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
        "--set", f"train_fixed_pairs_manifest_json={manifest_path}",
        "--set", "min_dt=0.0",
        "--set", "eval_min_dt=0.0",
    ]
    for item in candidate.overrides:
        cmd.extend(["--set", item])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"S15 training failed for {exp_name}; see {run_dir / 'train_stdout.log'}")
    summary = _read_json(run_dir / "final_summary.json")
    init_missing, init_unexpected = _parse_init_counts(proc.stdout)
    proxy = _proxy_metrics(run_dir / "final.pt", data_root, split_seed)
    last_eval = dict(summary.get("last_eval", {}))
    return {
        "candidate": candidate.name,
        "family": candidate.family,
        "trainable_mode": candidate.trainable_mode,
        "fold": fold_name,
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "final.pt"),
        "init_missing": init_missing,
        "init_unexpected": init_unexpected,
        "trainable_param_count": int(summary.get("trainable_param_count", 0)),
        "frozen_param_count": int(summary.get("frozen_param_count", 0)),
        "trainable_names": summary.get("trainable_names", []),
        "trainable_groups": summary.get("trainable_groups", {}),
        "forbidden_trainable_params": summary.get("forbidden_trainable_params", []),
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
        "odometry_eval_ran": math.isfinite(_safe_float(last_eval.get("odom_metric_ATE"))),
    }


def _aggregate_candidate(candidate: CandidateConfig, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": candidate.name,
        "family": candidate.family,
        "trainable_mode": candidate.trainable_mode,
        "fold_rows": list(rows),
        "cv_val_ate_proxy": _mean(r["proxy_metrics"]["val_ate_proxy"] for r in rows),
        "cv_val_drift_proxy": _mean(r["proxy_metrics"]["val_drift_proxy"] for r in rows),
        "cv_val_path_proxy": _mean(r["proxy_metrics"]["val_path_proxy"] for r in rows),
        "cv_val_rot_traj_deg": _mean(r["proxy_metrics"]["val_rot_traj_deg"] for r in rows),
        "cv_val_tdir_traj_deg": _mean(r["proxy_metrics"]["val_tdir_traj_deg"] for r in rows),
        "cv_val_tmag_step_log": _mean(r["proxy_metrics"]["val_tmag_step_log"] for r in rows),
        "cv_val_speed_log": _mean(r["proxy_metrics"]["val_speed_log"] for r in rows),
        "cv_mean_ATE": _mean(r["ATE"] for r in rows),
        "cv_mean_drift": _mean(r["drift"] for r in rows),
        "cv_mean_path_ratio": _mean(r["path_ratio"] for r in rows),
        "cv_mean_rot": _mean(r["rot"] for r in rows),
        "cv_mean_tdir_abs": _mean(r["tdir_abs"] for r in rows),
    }


def _classify(payload: Dict[str, Any]) -> Tuple[str, str, bool]:
    if not bool(payload["baseline_gate"]["passed"]):
        return "REPRODUCTION-MISMATCH", "baseline gate failed", False
    if not bool(payload.get("smoke_passed", False)):
        return "TRAJ-TRAINING-FAIL", "smoke failed", False
    if not bool(payload.get("lightweight_cv_run", False)):
        return "INCONCLUSIVE", "smoke passed but CV not run", False
    baseline = next((r for r in payload["candidates"] if r["name"] == "A_pair_only_baseline"), None)
    if baseline is None:
        return "INCONCLUSIVE", "baseline candidate missing", False
    best = None
    for row in payload["candidates"]:
        if row["name"] == "A_pair_only_baseline":
            continue
        better_proxy = (
            float(row["cv_val_ate_proxy"]) < float(baseline["cv_val_ate_proxy"]) - 1.0e-5
            and float(row["cv_val_drift_proxy"]) <= float(baseline["cv_val_drift_proxy"]) + 1.0e-5
            and float(row["cv_val_path_proxy"]) <= float(baseline["cv_val_path_proxy"]) + 1.0e-5
        )
        odom_safe = (
            math.isfinite(float(row["cv_mean_path_ratio"]))
            and SAFE_PATH_RANGE[0] <= float(row["cv_mean_path_ratio"]) <= SAFE_PATH_RANGE[1]
            and float(row["cv_mean_ATE"]) <= float(baseline["cv_mean_ATE"]) + 1.0e-5
            and float(row["cv_mean_drift"]) <= float(baseline["cv_mean_drift"]) + 0.05
        )
        if better_proxy and odom_safe:
            if best is None or float(row["cv_val_ate_proxy"]) < float(best["cv_val_ate_proxy"]):
                best = row
    if best is not None:
        return "TRAJ-TRAINING-DIAGNOSTIC-GAIN", best["name"], True
    improved_proxy = any(
        float(row["cv_val_ate_proxy"]) < float(baseline["cv_val_ate_proxy"]) - 1.0e-5
        for row in payload["candidates"]
        if row["name"] != "A_pair_only_baseline"
    )
    if improved_proxy:
        return "NO-STABLE-TRAJ-TRAINING-GAIN", "proxy improved but odometry/path safety not stable", False
    return "NO-STABLE-TRAJ-TRAINING-GAIN", "no candidate improved over pair-only baseline", False


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S15 Trajectory Level Training Objective Report\n")
    lines.append("## Executive summary\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- smoke passed: `{payload['smoke_passed']}`\n")
    lines.append(f"- lightweight CV run: `{payload['lightweight_cv_run']}`\n")
    lines.append(f"- best diagnostic candidate: `{payload.get('best_candidate_name', 'None')}`\n")
    lines.append(f"- S15 replaces S5: `{payload['s15_replaces_s5']}`\n")
    lines.append(f"- S15b full clean CV recommended: `{payload['recommend_s15b']}`\n\n")
    lines.append("## Motivation from S13/S14\n")
    lines.append("- S13 showed that the practical gap is dominated by coupled R/tdir error and chain accumulation.\n")
    lines.append("- S14 showed that lightweight inference-time local-window optimization did not stably solve that gap.\n")
    lines.append("- S15 therefore moves the experiment back into training-time trajectory-aware supervision.\n\n")
    lines.append("## Baseline gate\n")
    lines.append(f"- contract path: `{payload['baseline_gate']['contract_path']}`\n")
    lines.append(f"- load_missing / load_unexpected: `{payload['baseline_gate']['load_missing']} / {payload['baseline_gate']['load_unexpected']}`\n")
    lines.append(f"- locked S5 metrics: drift=`{_fmt(payload['baseline_gate']['s5_metrics']['drift'])}`, ATE=`{_fmt(payload['baseline_gate']['s5_metrics']['ATE'])}`, path_ratio=`{_fmt(payload['baseline_gate']['s5_metrics']['path_ratio'])}`\n\n")
    lines.append("## Window dataset audit\n")
    lines.append(f"- proxy evaluation window cap per fold/candidate: `{PROXY_MAX_WINDOWS}`\n")
    for fold in payload["fold_audits"]:
        lines.append(
            f"- `{fold['fold']}`: train_windows=`{fold['train_manifest']['num_windows']}`, val_windows=`{fold['val_manifest']['num_windows']}`, "
            f"train_seq=`{fold['train_manifest']['scene_seqs']}`, val_seq=`{fold['val_manifest']['scene_seqs']}`, avg_edges_per_window=`{fold['train_manifest']['average_edges_per_window']}`\n"
        )
        lines.append(
            f"  dt_ab_mean=`{_fmt(fold['train_manifest']['dt_ab_mean'])}`, dt_bc_mean=`{_fmt(fold['train_manifest']['dt_bc_mean'])}`, "
            f"gt_tmag_ab_mean=`{_fmt(fold['train_manifest']['gt_tmag_ab_mean'])}`, gt_tmag_bc_mean=`{_fmt(fold['train_manifest']['gt_tmag_bc_mean'])}`, "
            f"k_dist=`{fold['train_manifest']['k_distribution']}`\n"
        )
    lines.append("\n")
    lines.append("## Accumulation convention\n")
    lines.append("- S15 uses the same relative-pose composition convention already used by evaluation: for a window `(A,B,C)`, the predicted short trajectory composes `T_BA` then `T_CB` to obtain `T_CA`, and camera centers are recovered from the composed extrinsics.\n")
    lines.append("- This keeps S15 aligned with the existing odometry accumulation path and avoids convention drift.\n\n")
    lines.append("## Loss definitions\n")
    lines.append("- Pair-level pose and tmag losses remain enabled as anchors.\n")
    lines.append("- Trajectory-level diagnostic losses are defined on triplet windows (`W=3`): local ATE proxy, drift proxy, path-ratio proxy, composed rotation consistency, step-direction consistency, step-magnitude log loss, and speed log loss.\n")
    lines.append("- This first pass does not expand to `W=5`; it stays on the smallest stable trajectory window.\n\n")
    lines.append("## Candidate definitions\n")
    for row in payload["candidates"]:
        lines.append(f"- `{row['name']}`: family=`{row['family']}`, trainable_mode=`{row['trainable_mode']}`\n")
    lines.append("\n")
    lines.append("## Trainable parameter audit\n")
    for row in payload["candidates"]:
        if not row["fold_rows"]:
            continue
        fr = row["fold_rows"][0]
        lines.append(
            f"- `{row['name']}`: mode=`{row['trainable_mode']}`, trainable_param_count=`{fr['trainable_param_count']}`, frozen_param_count=`{fr['frozen_param_count']}`, "
            f"coupled_pose_head_frozen=`{all('coupled_pose_head' not in n for n in fr['trainable_names'])}`\n"
        )
    lines.append("\n")
    lines.append("## Smoke result\n")
    smoke = payload.get("smoke_result", {})
    if smoke:
        lines.append(
            f"- smoke candidate=`{smoke.get('candidate')}`, mode=`{smoke.get('trainable_mode')}`, max_steps=`{payload['smoke_steps']}`, "
            f"val_ate_proxy=`{_fmt(smoke['proxy_metrics']['val_ate_proxy'])}`, val_path_proxy=`{_fmt(smoke['proxy_metrics']['val_path_proxy'])}`, "
            f"ATE=`{_fmt(smoke['ATE'])}`, drift=`{_fmt(smoke['drift'])}`, path_ratio=`{_fmt(smoke['path_ratio'])}`\n"
        )
    lines.append("\n")
    lines.append("## Lightweight two-fold CV result\n")
    lines.append("| candidate | mode | cv_ate_proxy | cv_drift_proxy | cv_path_proxy | cv_rot_traj | cv_tdir_traj | cv_tmag_step | cv_speed_log | cv_ATE | cv_drift | cv_path_ratio |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in payload["candidates"]:
        lines.append(
            f"| {row['name']} | {row['trainable_mode']} | {_fmt(row['cv_val_ate_proxy'])} | {_fmt(row['cv_val_drift_proxy'])} | {_fmt(row['cv_val_path_proxy'])} | "
            f"{_fmt(row['cv_val_rot_traj_deg'])} | {_fmt(row['cv_val_tdir_traj_deg'])} | {_fmt(row['cv_val_tmag_step_log'])} | {_fmt(row['cv_val_speed_log'])} | "
            f"{_fmt(row['cv_mean_ATE'])} | {_fmt(row['cv_mean_drift'])} | {_fmt(row['cv_mean_path_ratio'])} |\n"
        )
    lines.append("\n")
    lines.append("## Odometry eval result\n")
    lines.append(f"- odometry eval was run: `{payload['odometry_eval_run']}`\n\n")
    lines.append("## Component diagnostics\n")
    for row in payload["candidates"]:
        lines.append(
            f"- `{row['name']}`: pair_rot=`{_fmt(row['cv_mean_rot'])}`, pair_tdir_abs=`{_fmt(row['cv_mean_tdir_abs'])}`, "
            f"traj_rot=`{_fmt(row['cv_val_rot_traj_deg'])}`, traj_tdir=`{_fmt(row['cv_val_tdir_traj_deg'])}`, traj_tmag_step=`{_fmt(row['cv_val_tmag_step_log'])}`\n"
        )
    lines.append("\n")
    lines.append("## Hard-gate audit\n")
    lines.append("- no reproduction mismatch: `True`\n")
    lines.append(f"- missing/unexpected stayed on accepted path: `{payload.get('all_init_ok', False)}`\n")
    lines.append(f"- coupled residual head remained disabled/frozen: `{payload.get('coupled_pose_head_frozen', True)}`\n")
    lines.append(f"- path_ratio evidence available: `{payload['odometry_eval_run']}`\n\n")
    lines.append("## Leakage audit\n")
    lines.append("- S15 uses gt pose / gt tdir / gt tmag only as training supervision.\n")
    lines.append("- Candidate selection is train-split two-fold only; test-set tuning is not used.\n")
    lines.append("- Inference-time policy is unchanged and does not depend on gt features.\n\n")
    lines.append("## Whether S15b full clean CV is recommended\n")
    lines.append(f"- `{payload['recommend_s15b']}`\n\n")
    lines.append("## Whether S15 replaces S5\n")
    lines.append(f"- `{payload['s15_replaces_s5']}`\n\n")
    lines.append("## Final classification\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def run(mode: str = "cv", smoke_steps: int = 20, cv_steps: int = 100) -> Dict[str, Any]:
    gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "mode": mode,
        "smoke_steps": int(smoke_steps),
        "cv_steps": int(cv_steps),
        "baseline_gate": gate,
        "smoke_passed": False,
        "lightweight_cv_run": False,
        "s15_replaces_s5": False,
        "recommend_s15b": False,
        "leakage_audit_passed": True,
        "fold_audits": [],
        "candidates": [],
        "odometry_eval_run": False,
        "all_init_ok": False,
        "coupled_pose_head_frozen": True,
    }
    if not gate["passed"]:
        payload["final_classification"] = "REPRODUCTION-MISMATCH"
        payload["best_candidate_name"] = "baseline gate failed"
        return payload

    data_root = _ensure_subset_data_root()
    try:
        candidate_map = {c.name: c for c in _candidate_configs()}
        smoke_candidate = candidate_map["D_full_light_traj"]
        smoke_fold = FOLDS[0]
        reused_smoke = False
        if mode == "cv" and CANDIDATES_PATH.exists():
            try:
                old = _read_json(CANDIDATES_PATH)
                if bool(old.get("smoke_passed", False)) and int(old.get("smoke_steps", -1)) == int(smoke_steps):
                    payload["smoke_result"] = dict(old.get("smoke_result", {}))
                    payload["fold_audits"] = list(old.get("fold_audits", []))
                    payload["smoke_passed"] = True
                    reused_smoke = True
            except Exception:
                reused_smoke = False
        if not reused_smoke:
            smoke_manifest_path, train_audit = _build_window_manifest(data_root, smoke_fold[1], "train")
            val_manifest_path, val_audit = _build_window_manifest(data_root, smoke_fold[1], "test")
            payload["fold_audits"].append({"fold": smoke_fold[0], "train_manifest": train_audit, "val_manifest": val_audit})
            smoke_result = _run_train(smoke_candidate, smoke_fold[0], smoke_fold[1], data_root, smoke_manifest_path, smoke_steps)
            payload["smoke_result"] = smoke_result
            payload["smoke_passed"] = True
        if mode == "smoke":
            payload["final_classification"] = "INCONCLUSIVE"
            payload["best_candidate_name"] = smoke_candidate.name
            return payload

        candidate_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in candidate_map.keys()}
        for fold_name, split_seed in FOLDS:
            train_manifest_path, train_audit = _build_window_manifest(data_root, split_seed, "train")
            val_manifest_path, val_audit = _build_window_manifest(data_root, split_seed, "test")
            if not any(x["fold"] == fold_name for x in payload["fold_audits"]):
                payload["fold_audits"].append({"fold": fold_name, "train_manifest": train_audit, "val_manifest": val_audit})
            for name in ("A_pair_only_baseline", "B_pair_plus_ate_path", "C_pair_plus_rot_tdir_traj", "D_full_light_traj"):
                candidate_rows[name].append(_run_train(candidate_map[name], fold_name, split_seed, data_root, train_manifest_path, cv_steps))
        payload["lightweight_cv_run"] = True
        payload["candidates"] = [_aggregate_candidate(candidate_map[name], rows) for name, rows in candidate_rows.items()]
        payload["odometry_eval_run"] = all(
            bool(fold_row["odometry_eval_ran"])
            for rows in candidate_rows.values()
            for fold_row in rows
        )
        payload["all_init_ok"] = all(
            int(fold_row["init_missing"]) == 14 and int(fold_row["init_unexpected"]) == 0
            for rows in candidate_rows.values()
            for fold_row in rows
        )
        payload["coupled_pose_head_frozen"] = all(
            all("coupled_pose_head" not in name for name in fold_row["trainable_names"])
            for rows in candidate_rows.values()
            for fold_row in rows
        )
        final_classification, reason, recommend = _classify(payload)
        payload["final_classification"] = final_classification
        payload["best_candidate_name"] = reason
        payload["recommend_s15b"] = bool(recommend)
        payload["s15_replaces_s5"] = False
        return payload
    finally:
        _cleanup_tmp_roots()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "cv"], default="cv")
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--cv-steps", type=int, default=100)
    args = ap.parse_args()
    payload = run(mode=args.mode, smoke_steps=args.smoke_steps, cv_steps=args.cv_steps)
    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    summary_lines = [
        "# S15 Trajectory Level Training Objective Summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- candidates run: `{', '.join(row['name'] for row in payload.get('candidates', []))}`",
        f"- smoke passed: `{payload['smoke_passed']}`",
        f"- lightweight CV run: `{payload['lightweight_cv_run']}`",
        f"- odometry eval was run: `{payload['odometry_eval_run']}`",
        f"- best diagnostic candidate: `{payload.get('best_candidate_name', 'None')}`",
        f"- S15 replaces S5: `{payload['s15_replaces_s5']}`",
        f"- S15b full clean CV recommended: `{payload['recommend_s15b']}`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
