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
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList, RflyPanoPanoramaPairsMixedK
from model import PanoramaRelPoseModel
from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg


PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/dovetao/miniconda3/envs/pytorch/bin/python")
BASE_CKPT = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s12_regime_balanced_sampling_summary.md"
MANIFEST_DIR = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_manifests"
TMP_DATA_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s12_regime_balanced_data"
TMP_RUN_ROOT = Path("/tmp/s12_regime_balanced_sampling_runs")
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
FOLDS: Tuple[Tuple[str, int], ...] = (("heldout_scene01_seq01", 0), ("heldout_scene01_seq02", 3))
SAFE_PATH_RANGE = (0.90, 0.97)
S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    overrides: Tuple[str, ...] = ()


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
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _ensure_subset_data_root() -> Path:
    pano_root = REPO_ROOT / "data" / "PanoramaView"
    if not pano_root.is_dir():
        raise FileNotFoundError(f"PanoramaView root not found: {pano_root}")
    if TMP_DATA_ROOT.exists():
        shutil.rmtree(TMP_DATA_ROOT)
    for scene, seq in TRAIN_GROUPS:
        src = pano_root / scene / seq
        dst = TMP_DATA_ROOT / "PanoramaView" / scene / seq
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)
    return TMP_DATA_ROOT


def _cleanup_tmp_data_root() -> None:
    if TMP_DATA_ROOT.exists():
        shutil.rmtree(TMP_DATA_ROOT)


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


def _build_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Config]:
    cfg = _cfg_from_dict(_load_ckpt_cfg(BASE_CKPT))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = 0.45
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.pin_memory = False
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(BASE_CKPT), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def _load_model(path: Path, device: torch.device) -> PanoramaRelPoseModel:
    cfg = _cfg_from_dict(_load_ckpt_cfg(path))
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _quantile_edges(vals: Sequence[float], q: Sequence[float]) -> List[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return [0.0 for _ in q]
    return np.quantile(arr, q).astype(np.float64).tolist()


def _bucket_from_edges(v: float, edges: Sequence[float], prefix: str) -> str:
    if len(edges) < 2:
        return f"{prefix}_all"
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        is_last = idx == len(edges) - 2
        if (v >= lo and v < hi) or (is_last and v <= hi):
            return f"{prefix}{idx}"
    return f"{prefix}{len(edges) - 2}"


def _dt_bucket(v: float) -> str:
    if v < 0.1:
        return "dt<0.1"
    if v < 0.3:
        return "0.1<=dt<0.3"
    if v < 0.5:
        return "0.3<=dt<0.5"
    if v < 1.0:
        return "0.5<=dt<1.0"
    if v < 2.0:
        return "1.0<=dt<2.0"
    return "dt>=2.0"


def _load_rows_for_fold(data_root: Path, split_seed: int, model: PanoramaRelPoseModel, cfg: Config, device: torch.device) -> List[Dict[str, Any]]:
    ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(data_root),
        split="train",
        split_by="scene_seq",
        train_ratio=0.5,
        split_seed=int(split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=(1, 2, 3, 5, 10, 20),
        pair_step=1,
        min_dt=0.0,
        max_dt=None,
    )
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device)
            IB = sample["IB"].unsqueeze(0).to(device)
            dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            gt_tmag = float(sample["t_gt_mag"])
            rows.append(
                {
                    "scene": str(meta["scene"]),
                    "seq": str(meta["seq"]),
                    "i": int(meta["i"]),
                    "j": int(meta["j"]),
                    "k": int(meta["k"]),
                    "dt_world": float(dt_world),
                    "pred_tmag": float(pred_tmag),
                    "gt_tmag": float(gt_tmag),
                }
            )
    pred_edges = _quantile_edges([r["pred_tmag"] for r in rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    gt_edges = _quantile_edges([r["gt_tmag"] for r in rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    pred_q90 = float(pred_edges[-2]) if len(pred_edges) >= 2 else float("inf")
    for row in rows:
        row["pred_tmag_bucket"] = _bucket_from_edges(float(row["pred_tmag"]), pred_edges, "pred_q")
        row["gt_tmag_bucket"] = _bucket_from_edges(float(row["gt_tmag"]), gt_edges, "gt_q")
        row["dt_bucket"] = _dt_bucket(float(row["dt_world"]))
        row["k_bucket"] = f"k={int(row['k'])}"
        row["dt_k_bucket"] = f"{row['dt_bucket']}|{row['k_bucket']}"
        row["high_risk_bucket"] = bool(float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20)
        row["high_pred_bucket"] = bool(float(row["pred_tmag"]) >= pred_q90)
        row["mixed_hard_bucket"] = bool(float(row["pred_tmag"]) >= pred_q90 and float(row["dt_world"]) >= 1.0)
    return rows


def _counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        label = str(row[key])
        out[label] = out.get(label, 0) + 1
    return out


def _repeat_rows_by_inverse_freq(rows: Sequence[Dict[str, Any]], key: str, cap: int = 4) -> List[Dict[str, Any]]:
    freq = _counts(rows, key)
    max_count = max(freq.values()) if freq else 1
    out: List[Dict[str, Any]] = []
    for row in rows:
        count = max(freq.get(str(row[key]), 1), 1)
        rep = min(cap, max(1, int(round(max_count / count))))
        out.extend([dict(row)] * rep)
    return out


def _candidate_rows(rows: Sequence[Dict[str, Any]], spec: CandidateSpec) -> List[Dict[str, Any]]:
    base = [dict(r) for r in rows]
    if spec.name == "A_baseline_uniform_sampling":
        return base
    if spec.name == "B_pred_tmag_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "pred_tmag_bucket", cap=4)
    if spec.name == "C_gt_tmag_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "gt_tmag_bucket", cap=4)
    if spec.name == "D_dt_k_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "dt_k_bucket", cap=4)
    if spec.name == "E_hard_regime_oversampling":
        out: List[Dict[str, Any]] = []
        for row in base:
            rep = 1
            if bool(row["high_pred_bucket"]):
                rep += 1
            if bool(row["high_risk_bucket"]):
                rep += 2
            out.extend([dict(row)] * min(rep, 4))
        return out
    if spec.name == "F_mixed_balanced_sampling":
        pred_freq = _counts(base, "pred_tmag_bucket")
        dtk_freq = _counts(base, "dt_k_bucket")
        max_pred = max(pred_freq.values()) if pred_freq else 1
        max_dtk = max(dtk_freq.values()) if dtk_freq else 1
        out: List[Dict[str, Any]] = []
        for row in base:
            rep_pred = max(1, int(round(max_pred / max(pred_freq.get(str(row["pred_tmag_bucket"]), 1), 1))))
            rep_dtk = max(1, int(round(max_dtk / max(dtk_freq.get(str(row["dt_k_bucket"]), 1), 1))))
            rep = min(4, max(rep_pred, rep_dtk))
            if bool(row["mixed_hard_bucket"]):
                rep = min(4, rep + 1)
            out.extend([dict(row)] * rep)
        return out
    raise ValueError(spec.name)


def _manifest_payload(rows: Sequence[Dict[str, Any]], candidate_name: str, split_seed: int) -> Dict[str, Any]:
    pairs = []
    for row in rows:
        item = dict(row)
        item["weight_tag"] = candidate_name
        pairs.append(item)
    return {
        "candidate_name": candidate_name,
        "split_seed": int(split_seed),
        "pairs": pairs,
        "split_summary": {
            "num_pairs": int(len(pairs)),
            "num_unique_pairs": int(len({(p["scene"], p["seq"], int(p["i"]), int(p["j"]), int(p["k"])) for p in pairs})),
            "num_sequences_after_filter": int(len({(p["scene"], p["seq"]) for p in pairs})),
        },
    }


def _audit_candidate_specs() -> List[CandidateSpec]:
    return [
        CandidateSpec("A_baseline_uniform_sampling", "baseline_uniform_sampling"),
        CandidateSpec("B_pred_tmag_balanced_sampling", "pred_tmag_balanced_sampling"),
        CandidateSpec("C_gt_tmag_balanced_sampling", "gt_tmag_balanced_sampling"),
        CandidateSpec("D_dt_k_balanced_sampling", "dt_k_balanced_sampling"),
        CandidateSpec("E_hard_regime_oversampling", "hard_regime_oversampling"),
        CandidateSpec("F_mixed_balanced_sampling", "mixed_balanced_sampling"),
    ]


def _cv_candidate_specs() -> List[CandidateSpec]:
    ckpt_cfg = _load_ckpt_cfg(BASE_CKPT)
    common = (
        "train_tmag_head_only=True",
        "use_coupled_pose_residual_head=False",
        "strict_load_checkpoint=False",
        f"init_checkpoint={BASE_CKPT.relative_to(REPO_ROOT)}",
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
        "tmag_condition_on_dt=False",
        "use_tmag_speed_loss=False",
        "use_tmag_ratio_loss=False",
        "use_tmag_chain_sum_loss=False",
        "use_tmag_regime_reweight=False",
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
        CandidateSpec("A_baseline_uniform_sampling", "uniform_manifest_baseline", common),
        CandidateSpec("D_dt_k_balanced_sampling", "dt_k_balanced_sampling", common),
        CandidateSpec("E_hard_regime_oversampling", "hard_regime_oversampling", common),
        CandidateSpec("F_mixed_balanced_sampling", "mixed_balanced_sampling", common),
    ]


def _run_audit_core(data_root: Path, device: torch.device) -> Dict[str, Any]:
    model, cfg = _build_model(device)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    folds: List[Dict[str, Any]] = []
    for fold_name, split_seed in FOLDS:
        rows = _load_rows_for_fold(data_root, split_seed, model, cfg, device)
        pred_q90 = float(np.quantile(np.asarray([float(r["pred_tmag"]) for r in rows], dtype=np.float64), 0.90)) if rows else float("nan")
        fold_payload = {
            "fold_name": fold_name,
            "split_seed": int(split_seed),
            "base_num_rows": int(len(rows)),
            "pred_q90": pred_q90,
            "candidates": [],
        }
        for spec in _audit_candidate_specs():
            sampled = _candidate_rows(rows, spec)
            manifest = _manifest_payload(sampled, spec.name, split_seed)
            manifest_path = MANIFEST_DIR / f"{fold_name}_{spec.name}.json"
            _write_json(manifest_path, manifest)
            fold_payload["candidates"].append(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "manifest_path": str(manifest_path),
                    "num_rows": int(len(sampled)),
                    "num_unique_pairs": int(len({(r["scene"], r["seq"], int(r["i"]), int(r["j"]), int(r["k"])) for r in sampled})),
                    "high_pred_mass": int(sum(1 for r in sampled if bool(r["high_pred_bucket"]))),
                    "high_risk_mass": int(sum(1 for r in sampled if bool(r["high_risk_bucket"]))),
                    "mixed_hard_mass": int(sum(1 for r in sampled if bool(r["mixed_hard_bucket"]))),
                    "pred_bucket_counts": _counts(sampled, "pred_tmag_bucket"),
                    "gt_bucket_counts": _counts(sampled, "gt_tmag_bucket"),
                    "dt_k_bucket_counts": _counts(sampled, "dt_k_bucket"),
                }
            )
        folds.append(fold_payload)
    return {"folds": folds, "candidate_specs": [{"name": s.name, "family": s.family} for s in _audit_candidate_specs()]}


def _load_audit_payload(data_root: Path, device: torch.device) -> Dict[str, Any]:
    if CANDIDATES_PATH.exists():
        try:
            obj = _read_json(CANDIDATES_PATH)
            if obj.get("run_mode") == "audit" and isinstance(obj.get("folds"), list) and obj.get("folds"):
                return {
                    "folds": obj["folds"],
                    "candidate_specs": obj.get("candidate_specs", [{"name": s.name, "family": s.family} for s in _audit_candidate_specs()]),
                }
        except Exception:
            pass
    return _run_audit_core(data_root, device)


def _manifest_path_for_fold(fold_name: str, candidate_name: str) -> Path:
    return MANIFEST_DIR / f"{fold_name}_{candidate_name}.json"


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    items = payload.get("pairs", payload if isinstance(payload, list) else [])
    return [dict(x) for x in items]


def _pred_q90_from_base_manifest(fold_name: str) -> float:
    rows = _load_manifest_rows(_manifest_path_for_fold(fold_name, "A_baseline_uniform_sampling"))
    vals = [float(r["pred_tmag"]) for r in rows if math.isfinite(float(r["pred_tmag"]))]
    if not vals:
        return float("nan")
    return float(np.quantile(np.asarray(vals, dtype=np.float64), 0.90))


def _parse_init_counts(stdout: str) -> Tuple[int, int]:
    pat = re.compile(r"\[InitCkpt\].*missing=(\d+)\s+\|\s+unexpected=(\d+)")
    hits = pat.findall(stdout)
    if not hits:
        return -1, -1
    a, b = hits[-1]
    return int(a), int(b)


def _heldout_eval_dataset(data_root: Path, split_seed: int, cfg: Config) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(data_root),
        split="test",
        split_by="scene_seq",
        train_ratio=0.5,
        split_seed=int(split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=(1, 2, 3, 5, 10, 20),
        pair_step=1,
        min_dt=0.0,
        max_dt=None,
    )


def _heldout_triplet_dataset(data_root: Path, split_seed: int, cfg: Config) -> RflyPanoPanoramaPairsMixedK:
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


def _subset_mean(vals: Sequence[float], flags: Sequence[bool]) -> float:
    picked = [float(v) for v, flag in zip(vals, flags) if flag and math.isfinite(float(v))]
    return float(sum(picked) / len(picked)) if picked else float("nan")


def _proxy_metrics(model: PanoramaRelPoseModel, data_root: Path, split_seed: int, pred_q90: float) -> Dict[str, Any]:
    cfg = model.cfg
    device = next(model.parameters()).device
    eval_ds = _heldout_eval_dataset(data_root, split_seed, cfg)
    manifest = eval_ds.manifest()
    pair_log_errs: List[float] = []
    speed_log_errs: List[float] = []
    high_pred_flags: List[bool] = []
    high_risk_flags: List[bool] = []
    dt_k20_flags: List[bool] = []
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            sample = eval_ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device)
            IB = sample["IB"].unsqueeze(0).to(device)
            dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            gt_tmag = float(sample["t_gt_mag"])
            eps = 1.0e-6
            pair_log_err = abs(math.log(max(pred_tmag, eps)) - math.log(max(gt_tmag, eps)))
            speed_log_err = abs((math.log(max(pred_tmag, eps)) - math.log(max(dt_world, eps))) - (math.log(max(gt_tmag, eps)) - math.log(max(dt_world, eps))))
            pair_log_errs.append(pair_log_err)
            speed_log_errs.append(speed_log_err)
            high_pred_flags.append(bool(math.isfinite(pred_q90) and pred_tmag >= pred_q90))
            is_high_risk = bool(dt_world >= 1.0 and int(meta.get("k", -1)) == 20)
            high_risk_flags.append(is_high_risk)
            dt_k20_flags.append(is_high_risk)

    triplet_ds = _heldout_triplet_dataset(data_root, split_seed, cfg)
    chain_sum_log_errs: List[float] = []
    chain_sum_absdevs: List[float] = []
    with torch.no_grad():
        for idx in range(len(triplet_ds)):
            sample = triplet_ds[idx]
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
            chain_sum_log_errs.append(abs(math.log(max(pred_ab + pred_bc, eps)) - math.log(max(gt_ab + gt_bc, eps))))
            chain_sum_absdevs.append(abs((pred_ab + pred_bc) / max(gt_ab + gt_bc, eps) - 1.0))

    return {
        "num_eval_rows": int(len(pair_log_errs)),
        "num_triplets": int(len(chain_sum_log_errs)),
        "val_tmag_log_error": _mean(pair_log_errs),
        "val_speed_log_error": _mean(speed_log_errs),
        "val_chain_sum_log_error": _mean(chain_sum_log_errs),
        "val_chain_sum_ratio_absdev": _mean(chain_sum_absdevs),
        "high_pred_bucket_error": _subset_mean(pair_log_errs, high_pred_flags),
        "high_risk_bucket_error": _subset_mean(pair_log_errs, high_risk_flags),
        "dt1_k20_bucket_error": _subset_mean(pair_log_errs, dt_k20_flags),
        "high_pred_bucket_count": int(sum(high_pred_flags)),
        "high_risk_bucket_count": int(sum(high_risk_flags)),
        "dt1_k20_bucket_count": int(sum(dt_k20_flags)),
    }


def _run_train(candidate: CandidateSpec, fold_name: str, split_seed: int, data_root: Path, max_steps: int) -> Dict[str, Any]:
    manifest_path = _manifest_path_for_fold(fold_name, candidate.name)
    exp_name = f"S12_{candidate.name}_{fold_name}_u{max_steps}"
    run_dir = TMP_RUN_ROOT / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        PYTHON_BIN,
        "train_mvp.py",
        "--set", f"exp_name={exp_name}",
        "--set", f"ckpt_dir={run_dir.parent}",
        "--set", f"data_root={data_root}",
        "--set", "split_by=scene_seq",
        "--set", "train_ratio=0.5",
        "--set", f"split_seed={split_seed}",
        "--set", f"max_steps={max_steps}",
        "--set", f"eval_every={max_steps}",
        "--set", "max_eval_batches=0",
        "--set", "max_train_eval_batches=0",
        "--set", f"train_fixed_pairs_manifest_json={manifest_path}",
        "--set", "train_fixed_pairs_return_triplet=False",
    ]
    for item in candidate.overrides:
        cmd.extend(["--set", item])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_stdout.log").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"S12 training failed for {exp_name}; see {run_dir / 'train_stdout.log'}")
    summary = _read_json(run_dir / "final_summary.json")
    init_missing, init_unexpected = _parse_init_counts(proc.stdout)
    last_eval = dict(summary.get("last_eval", {}))
    pred_q90 = _pred_q90_from_base_manifest(fold_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run_dir / "final.pt", device)
    proxy = _proxy_metrics(model, data_root, split_seed, pred_q90)
    return {
        "candidate": candidate.name,
        "family": candidate.family,
        "fold": fold_name,
        "split_seed": int(split_seed),
        "exp_name": exp_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "final.pt"),
        "manifest_path": str(manifest_path),
        "manifest_distribution": _read_json(manifest_path).get("split_summary", {}),
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
        "tdir_local_A_abs": _safe_float(last_eval.get("tdir_local_A_abs")),
        "tmag_rel_err": _safe_float(last_eval.get("tmag_rel_err")),
        "odometry_eval_ran": math.isfinite(_safe_float(last_eval.get("odom_metric_ATE"))),
        "path_ratio_safe": bool(SAFE_PATH_RANGE[0] <= _safe_float(last_eval.get("odom_shape_metric_mean_path_length_ratio")) <= SAFE_PATH_RANGE[1]),
    }


def _cv_deltas(row: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    return {
        "delta_tmag_log_error": float(row["cv_proxy_val_tmag_log_error"]) - float(baseline["cv_proxy_val_tmag_log_error"]),
        "delta_speed_log_error": float(row["cv_proxy_val_speed_log_error"]) - float(baseline["cv_proxy_val_speed_log_error"]),
        "delta_chain_sum_log_error": float(row["cv_proxy_val_chain_sum_log_error"]) - float(baseline["cv_proxy_val_chain_sum_log_error"]),
        "delta_high_pred_bucket_error": float(row["cv_high_pred_bucket_error"]) - float(baseline["cv_high_pred_bucket_error"]),
        "delta_high_risk_bucket_error": float(row["cv_high_risk_bucket_error"]) - float(baseline["cv_high_risk_bucket_error"]),
        "delta_dt1_k20_bucket_error": float(row["cv_dt1_k20_bucket_error"]) - float(baseline["cv_dt1_k20_bucket_error"]),
        "delta_ATE": float(row["cv_mean_ATE"]) - float(baseline["cv_mean_ATE"]),
        "delta_drift": float(row["cv_mean_drift"]) - float(baseline["cv_mean_drift"]),
        "delta_path_ratio": float(row["cv_mean_path_ratio"]) - float(baseline["cv_mean_path_ratio"]),
    }


def _finalize_fold_relative_flags(candidate_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    baseline_rows = {row["fold"]: row for row in candidate_rows["A_baseline_uniform_sampling"]}
    for name, rows in candidate_rows.items():
        for row in rows:
            base = baseline_rows[row["fold"]]
            if name == "A_baseline_uniform_sampling":
                row["r_tdir_path_unchanged"] = True
            else:
                row["r_tdir_path_unchanged"] = bool(
                    math.isfinite(float(row["rot"]))
                    and math.isfinite(float(row["tdir_abs"]))
                    and float(row["rot"]) <= float(base["rot"]) + 0.5
                    and float(row["tdir_abs"]) <= float(base["tdir_abs"]) + 2.0
                )


def _aggregate_candidate_rows(candidate: CandidateSpec, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": candidate.name,
        "family": candidate.family,
        "overrides": list(candidate.overrides),
        "fold_rows": list(rows),
        "cv_proxy_val_tmag_log_error": _mean(r["proxy_metrics"]["val_tmag_log_error"] for r in rows),
        "cv_proxy_val_speed_log_error": _mean(r["proxy_metrics"]["val_speed_log_error"] for r in rows),
        "cv_proxy_val_chain_sum_log_error": _mean(r["proxy_metrics"]["val_chain_sum_log_error"] for r in rows),
        "cv_proxy_val_chain_sum_ratio_absdev": _mean(r["proxy_metrics"]["val_chain_sum_ratio_absdev"] for r in rows),
        "cv_high_pred_bucket_error": _mean(r["proxy_metrics"]["high_pred_bucket_error"] for r in rows),
        "cv_high_risk_bucket_error": _mean(r["proxy_metrics"]["high_risk_bucket_error"] for r in rows),
        "cv_dt1_k20_bucket_error": _mean(r["proxy_metrics"]["dt1_k20_bucket_error"] for r in rows),
        "cv_mean_ATE": _mean(r["ATE"] for r in rows),
        "cv_mean_drift": _mean(r["drift"] for r in rows),
        "cv_mean_path_ratio": _mean(r["path_ratio"] for r in rows),
        "cv_mean_rot": _mean(r["rot"] for r in rows),
        "cv_mean_tdir_abs": _mean(r["tdir_abs"] for r in rows),
        "odometry_eval_ran": all(bool(r["odometry_eval_ran"]) for r in rows),
        "path_ratio_safe_all_folds": all(bool(r["path_ratio_safe"]) for r in rows),
        "init_ok_all_folds": all(int(r["init_missing"]) == 14 and int(r["init_unexpected"]) == 0 for r in rows),
        "r_tdir_safe_all_folds": all(bool(r["r_tdir_path_unchanged"]) for r in rows),
    }


def _annotate_candidate_gates(aggregate_rows: List[Dict[str, Any]]) -> None:
    baseline = next(r for r in aggregate_rows if r["name"] == "A_baseline_uniform_sampling")
    base_fold_map = {row["fold"]: row for row in baseline["fold_rows"]}
    for row in aggregate_rows:
        row["deltas"] = _cv_deltas(row, baseline)
        no_worse_both = True
        better_high_risk_both = True
        for fold in row["fold_rows"]:
            base = base_fold_map[fold["fold"]]
            proxy_ok = float(fold["proxy_metrics"]["high_risk_bucket_error"]) <= float(base["proxy_metrics"]["high_risk_bucket_error"]) + 1.0e-6
            odom_ok = float(fold["ATE"]) <= float(base["ATE"]) + 1.0e-6 and float(fold["drift"]) <= float(base["drift"]) + 1.0e-6
            if not (proxy_ok and odom_ok):
                better_high_risk_both = False
            if not odom_ok:
                no_worse_both = False
        row["eligible_for_full_s12b_cv"] = bool(
            row["name"] in {"E_hard_regime_oversampling", "F_mixed_balanced_sampling"}
            and row["odometry_eval_ran"]
            and row["path_ratio_safe_all_folds"]
            and row["init_ok_all_folds"]
            and row["r_tdir_safe_all_folds"]
            and better_high_risk_both
        )
        row["no_worse_than_A_both_folds"] = bool(no_worse_both)
        row["high_risk_improved_both_folds"] = bool(
            row["name"] != "A_baseline_uniform_sampling"
            and all(
                float(fold["proxy_metrics"]["high_risk_bucket_error"]) <= float(base_fold_map[fold["fold"]]["proxy_metrics"]["high_risk_bucket_error"]) + 1.0e-6
                for fold in row["fold_rows"]
            )
        )


def _classify_cv(payload: Dict[str, Any]) -> Tuple[str, str, bool, str | None]:
    if not bool(payload["baseline_gate"]["passed"]):
        return "REPRODUCTION-MISMATCH", "baseline gate failed", False, None
    if not payload["odometry_eval_run"]:
        return "INCONCLUSIVE", "odometry eval did not complete on all folds", False, None
    diagnostics = [r for r in payload["candidates"] if r["name"] != "A_baseline_uniform_sampling"]
    if any((not r["init_ok_all_folds"]) for r in diagnostics):
        return "REGIME-SAMPLING-FAIL", "missing/unexpected worsened beyond accepted 14/0 path", False, None
    if any((not r["r_tdir_safe_all_folds"]) for r in diagnostics):
        return "REGIME-SAMPLING-FAIL", "R/tdir path changed unintentionally under sampler training", False, None

    eligible = [r for r in diagnostics if bool(r["eligible_for_full_s12b_cv"])]
    if eligible:
        eligible.sort(key=lambda r: (r["deltas"]["delta_high_risk_bucket_error"], r["deltas"]["delta_ATE"], r["deltas"]["delta_drift"]))
        best = eligible[0]
        return "REGIME-SAMPLING-DIAGNOSTIC-GAIN", "E/F improved high-risk proxy on both folds without unsafe odometry/path degradation", True, best["name"]

    improved_proxy = [r for r in diagnostics if bool(r["high_risk_improved_both_folds"])]
    if improved_proxy:
        improved_proxy.sort(key=lambda r: (r["deltas"]["delta_high_risk_bucket_error"], r["deltas"]["delta_ATE"], r["deltas"]["delta_drift"]))
        best = improved_proxy[0]
        return "NO-STABLE-REGIME-SAMPLING-GAIN", "sampler improved high-risk proxy but odometry/path safety was not stable enough for promotion", False, best["name"]

    diagnostics.sort(key=lambda r: (r["deltas"]["delta_high_risk_bucket_error"], r["deltas"]["delta_ATE"], r["deltas"]["delta_drift"]))
    best = diagnostics[0]["name"] if diagnostics else None
    return "NO-STABLE-REGIME-SAMPLING-GAIN", "no sampled candidate achieved stable high-risk proxy and odometry improvement over A_uniform", False, best


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S12 Regime Balanced Sampling Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- baseline gate passed: `{payload['baseline_gate']['passed']}`\n")
    lines.append(f"- run mode: `{payload['run_mode']}`\n")
    lines.append(f"- device used: `{payload['device']}`\n")
    lines.append(f"- training budget: `{payload.get('training_budget', 'audit only')}`\n")
    lines.append(f"- odometry eval run: `{payload.get('odometry_eval_run', False)}`\n")
    lines.append(f"- best diagnostic sampler: `{payload.get('best_candidate_name')}`\n")
    lines.append(f"- S12 replaces S5: `{payload.get('s12_replaces_s5', False)}`\n")
    lines.append(f"- S12b full clean CV recommended: `{payload.get('s12b_recommended', False)}`\n\n")

    lines.append("## Motivation\n\n")
    lines.append("- S4 identified tmag-regime-dominant error structure and a worst bucket around `dt>=1.0, k=20`.\n")
    lines.append("- S8/S9/S10/S11 suggest that changing heads, routers, smoothers, or tmag losses alone is unlikely to replace S5.\n")
    lines.append("- S12 therefore shifts to data-centric regime-balanced sampling and curriculum redesign.\n\n")

    lines.append("## Baseline gate using S8b contract\n\n")
    lines.append(f"- contract path: `{payload['baseline_gate']['contract_path']}`\n")
    lines.append(f"- load_missing / load_unexpected: `{payload['baseline_gate']['load_missing']} / {payload['baseline_gate']['load_unexpected']}`\n")
    lines.append(
        f"- cited locked S5 metrics: drift=`{_fmt(payload['baseline_gate']['s5_metrics']['drift'])}`, "
        f"ATE=`{_fmt(payload['baseline_gate']['s5_metrics']['ATE'])}`, "
        f"path_ratio=`{_fmt(payload['baseline_gate']['s5_metrics']['path_ratio'])}`\n\n"
    )

    if payload["run_mode"] == "audit":
        lines.append("## Candidate sampler families\n\n")
        for row in payload["candidate_specs"]:
            lines.append(f"- `{row['name']}`: `{row['family']}`\n")
        lines.append("\n")
        lines.append("## Fold sampler distribution summary\n\n")
        for fold in payload["folds"]:
            lines.append(f"### {fold['fold_name']}\n\n")
            lines.append(f"- base rows: `{fold['base_num_rows']}`\n")
            lines.append(f"- pred_q90 threshold: `{_fmt(fold['pred_q90'])}`\n")
            for cand in fold["candidates"]:
                lines.append(
                    f"- `{cand['name']}`: rows=`{cand['num_rows']}`, unique_pairs=`{cand['num_unique_pairs']}`, "
                    f"high_pred_mass=`{cand['high_pred_mass']}`, high_risk_mass=`{cand['high_risk_mass']}`, mixed_hard_mass=`{cand['mixed_hard_mass']}`\n"
                )
            lines.append("\n")
    else:
        lines.append("## Lightweight two-fold CV setting\n\n")
        lines.append("- folds: `heldout_scene01_seq01`, `heldout_scene01_seq02`\n")
        lines.append("- samplers run: `A_uniform`, `D_dt_k_balanced`, `E_hard_regime_oversampling`, `F_mixed_balanced`\n")
        lines.append("- updates per sampler per fold: `100`\n")
        lines.append("- batch_size: `1`\n")
        lines.append("- model structure kept fixed; sampler is the main variable.\n\n")

        lines.append("## Manifest paths used\n\n")
        for row in payload["candidates"]:
            lines.append(f"### {row['name']}\n\n")
            for fold in row["fold_rows"]:
                lines.append(f"- `{fold['fold']}`: `{fold['manifest_path']}`\n")
            lines.append("\n")

        lines.append("## Sampling distribution table\n\n")
        for fold in payload["folds"]:
            lines.append(f"### {fold['fold_name']}\n\n")
            lines.append("| sampler | rows | unique_pairs | high_pred_mass | high_risk_mass | mixed_hard_mass |\n")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            for cand in fold["candidates"]:
                if cand["name"] not in {"A_baseline_uniform_sampling", "D_dt_k_balanced_sampling", "E_hard_regime_oversampling", "F_mixed_balanced_sampling"}:
                    continue
                lines.append(
                    f"| {cand['name']} | {cand['num_rows']} | {cand['num_unique_pairs']} | {cand['high_pred_mass']} | {cand['high_risk_mass']} | {cand['mixed_hard_mass']} |\n"
                )
            lines.append("\n")

        lines.append("## Fold metrics table\n\n")
        for row in payload["candidates"]:
            lines.append(f"### {row['name']}\n\n")
            for fold in row["fold_rows"]:
                proxy = fold["proxy_metrics"]
                last_train = fold["train_snapshot"].get("last_train", {})
                lines.append(
                    f"- `{fold['fold']}`: manifest=`{fold['manifest_path']}`, train_loss=`{_fmt(_safe_float(last_train.get('loss_total')) )}`, "
                    f"val_tmag_log=`{_fmt(proxy['val_tmag_log_error'])}`, val_speed_log=`{_fmt(proxy['val_speed_log_error'])}`, "
                    f"val_chain_sum_log=`{_fmt(proxy['val_chain_sum_log_error'])}`, high_pred_err=`{_fmt(proxy['high_pred_bucket_error'])}`, "
                    f"high_risk_err=`{_fmt(proxy['high_risk_bucket_error'])}`, dt1_k20_err=`{_fmt(proxy['dt1_k20_bucket_error'])}`, "
                    f"ATE=`{_fmt(fold['ATE'])}`, drift=`{_fmt(fold['drift'])}`, path_ratio=`{_fmt(fold['path_ratio'])}`, "
                    f"missing/unexpected=`{fold['init_missing']}/{fold['init_unexpected']}`, path_safe=`{fold['path_ratio_safe']}`, "
                    f"R/tdir_unchanged=`{fold['r_tdir_path_unchanged']}`\n"
                )
            lines.append("\n")

        lines.append("## Hard-gate audit\n\n")
        for row in payload["candidates"]:
            lines.append(
                f"- `{row['name']}`: init_ok=`{row['init_ok_all_folds']}`, path_safe_all_folds=`{row['path_ratio_safe_all_folds']}`, "
                f"r_tdir_safe_all_folds=`{row['r_tdir_safe_all_folds']}`, high_risk_improved_both_folds=`{row.get('high_risk_improved_both_folds', False)}`, "
                f"eligible_for_full_s12b_cv=`{row.get('eligible_for_full_s12b_cv', False)}`\n"
            )
        lines.append("\n")

        lines.append("## Best diagnostic sampler\n\n")
        lines.append(f"- `{payload.get('best_candidate_name')}`\n\n")

        lines.append("## Whether S12b full clean CV is recommended\n\n")
        lines.append(f"- `{payload.get('s12b_recommended', False)}`\n\n")

        lines.append("## Whether S12 replaces S5\n\n")
        lines.append(f"- `{payload.get('s12_replaces_s5', False)}`\n\n")

    lines.append("## Leakage audit\n\n")
    for k, v in payload["leakage_audit"].items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    if payload.get("classification_reason"):
        lines.append(f"- rationale: `{payload['classification_reason']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Final S12 Regime Balanced Sampling Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- Run mode: `{payload['run_mode']}`\n")
    lines.append(f"- Device: `{payload['device']}`\n")
    lines.append(f"- Odometry eval run: `{payload.get('odometry_eval_run', False)}`\n")
    lines.append(f"- Best diagnostic sampler: `{payload.get('best_candidate_name')}`\n")
    lines.append(f"- S12 replaces S5: `{payload.get('s12_replaces_s5', False)}`\n")
    lines.append(f"- S12b full clean CV recommended: `{payload.get('s12b_recommended', False)}`\n")
    lines.append("- S5 remains final clean candidate at this stage\n\n")
    if payload["run_mode"] == "audit":
        lines.append("S12 has been started as a data-centric follow-up to S11. ")
        lines.append("The current stage builds train-only regime buckets and candidate sampling manifests for A/B/C/D/E/F, so the next S12 step can legally compare balanced-sampling training under the same clean-CV protocol.\n")
    else:
        lines.append("This stage upgrades S12 from audit to a lightweight two-fold CV comparison over A/D/E/F. ")
        lines.append("It reuses the train-only manifests generated in audit mode, keeps the model structure fixed, and asks whether sampler choice alone produces a stable high-risk regime benefit without unsafe path-ratio or odometry drift.\n\n")
        lines.append(f"Outcome: `{payload.get('classification_reason', '')}`. ")
        lines.append("At this stage S5 still remains the locked final clean candidate.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run(mode: str = "audit", max_steps: int = 100) -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload: Dict[str, Any] = {
        "run_mode": str(mode),
        "device": str(device),
        "manifest_dir": str(MANIFEST_DIR),
        "baseline_gate": baseline_gate,
        "candidate_specs": [{"name": s.name, "family": s.family} for s in _audit_candidate_specs()],
        "folds": [],
        "leakage_audit": {
            "passed": bool(baseline_gate["passed"]),
            "test_used_for_selection": False,
            "gt_tmag_as_inference_feature": False,
            "gt_pose_as_inference_feature": False,
            "train_only_regime_labels_used_for_sampling": True,
            "post_processing_router_used": False,
            "train_cv_selection_only": True,
        },
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "classification_reason": "",
        "s12_replaces_s5": False,
        "s12b_recommended": False,
        "best_candidate_name": None,
        "odometry_eval_run": False,
        "training_budget": f"lightweight two-fold CV, samplers=A/D/E/F, max_steps={max_steps}" if mode == "cv" else "audit only",
    }
    if not baseline_gate["passed"]:
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload

    data_root = _ensure_subset_data_root()
    try:
        if mode == "audit":
            audit_payload = _run_audit_core(data_root, device)
            payload["folds"] = audit_payload["folds"]
            payload["candidate_specs"] = audit_payload["candidate_specs"]
        else:
            audit_payload = _load_audit_payload(data_root, device)
            payload["folds"] = audit_payload["folds"]
            payload["candidate_specs"] = audit_payload["candidate_specs"]
            candidate_list = _cv_candidate_specs()
            candidate_rows: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidate_list:
                candidate_rows[candidate.name] = []
                for fold_name, split_seed in FOLDS:
                    candidate_rows[candidate.name].append(_run_train(candidate, fold_name, split_seed, data_root, max_steps))
            _finalize_fold_relative_flags(candidate_rows)
            aggregate_rows = [_aggregate_candidate_rows(candidate, candidate_rows[candidate.name]) for candidate in candidate_list]
            _annotate_candidate_gates(aggregate_rows)
            baseline = next(r for r in aggregate_rows if r["name"] == "A_baseline_uniform_sampling")
            diag_rows = [r for r in aggregate_rows if r["name"] != "A_baseline_uniform_sampling"]
            diag_rows.sort(key=lambda r: (r["deltas"]["delta_high_risk_bucket_error"], r["deltas"]["delta_ATE"], r["deltas"]["delta_drift"], r["name"]))
            payload["best_candidate_name"] = diag_rows[0]["name"] if diag_rows else None
            payload["candidates"] = aggregate_rows
            payload["odometry_eval_run"] = all(bool(r["odometry_eval_ran"]) for row in aggregate_rows for r in row["fold_rows"])
            final_classification, reason, recommend, best_name = _classify_cv(payload)
            payload["final_classification"] = final_classification
            payload["classification_reason"] = reason
            payload["s12b_recommended"] = recommend
            if best_name is not None:
                payload["best_candidate_name"] = best_name
            # Keep the baseline row first in the json for easier reading.
            others = [r for r in aggregate_rows if r["name"] != baseline["name"]]
            others.sort(key=lambda r: (r["deltas"]["delta_high_risk_bucket_error"], r["deltas"]["delta_ATE"], r["deltas"]["delta_drift"], r["name"]))
            payload["candidates"] = [baseline] + others
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload
    finally:
        _cleanup_tmp_data_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch S12 regime-balanced sampling diagnostic.")
    parser.add_argument("--mode", default="cv", choices=("audit", "cv"), help="Current S12 stage.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max training updates for CV mode.")
    args = parser.parse_args()
    payload = run(mode=args.mode, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "final_classification": payload["final_classification"],
                "run_mode": payload["run_mode"],
                "device": payload["device"],
                "best_candidate_name": payload.get("best_candidate_name"),
                "s12b_recommended": payload.get("s12b_recommended", False),
                "odometry_eval_run": payload.get("odometry_eval_run", False),
                "report_path": str(REPORT_PATH),
                "candidates_path": str(CANDIDATES_PATH),
                "summary_path": str(SUMMARY_PATH),
                "manifest_dir": str(MANIFEST_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
