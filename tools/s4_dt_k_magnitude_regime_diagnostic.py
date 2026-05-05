#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from train_mvp import (
    _apply_dt_bucket_scale_anchor_policy,
    _epipolar_matching_diagnostics,
    _restore_cfg_from_policy_base_checkpoint,
    eval_odometry_sequence,
)


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
BASE_CKPT = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S4_dt_k_magnitude_regime_diagnostic_report.md"
FIG_DIR = REPO_ROOT / "checkpoints" / "S4_dt_k_magnitude_regime_diagnostic_figures"
PAIR_K_LIST = (1, 2, 3, 5, 10, 20)
PROGRESS_EVERY = 64
MIN_BUCKET_COUNT = 20


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cfg() -> Tuple[Config, Dict[str, Any], Dict[str, Any]]:
    cfg = Config()
    cfg.dt_bucket_scale_anchor_policy_json = str(POLICY_PATH)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, [])
    policy_summary = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = 0.45
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.batch_size = 1
    cfg.pin_memory = False
    cfg.num_workers = 0
    return cfg, restore_summary, policy_summary


def _load_model(cfg: Config, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(BASE_CKPT), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_dataset(cfg: Config, split: str, *, k_list: Sequence[int]) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=split,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
    )


def _rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(R_pred.float().transpose(-1, -2), R_gt.float())
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _scalarize(diag: Dict[str, torch.Tensor], key: str) -> float:
    if key not in diag or diag[key] is None:
        return float("nan")
    v = diag[key]
    if torch.is_tensor(v):
        return float(v.detach().float().view(-1)[0].cpu())
    return _safe_float(v)


def _extract_pair_rows(
    model: PanoramaRelPoseModel,
    cfg: Config,
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    split_name: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest = ds.manifest()
    for ds_idx, meta in enumerate(manifest):
        if (ds_idx + 1) % PROGRESS_EVERY == 0 or ds_idx == 0 or (ds_idx + 1) == len(manifest):
            print(f"[S4] {split_name} pair extraction {ds_idx + 1}/{len(manifest)}", flush=True)
        sample = ds[ds_idx]
        IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
        IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
        R_gt = sample["R_gt"].unsqueeze(0).to(device, non_blocking=True)
        t_gt_vec = sample["t_gt_vec"].unsqueeze(0).to(device, non_blocking=True)
        dt_world = torch.tensor([float(meta.get("dt_world", float(sample["t_gt_mag"])))], device=device, dtype=torch.float32)

        with torch.no_grad():
            R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=False, dt_world=dt_world)
            t_gt = F.normalize(t_gt_vec.float(), dim=-1, eps=1.0e-6)
            t_gt_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), t_gt.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1.0e-6)
            t_dir_out = aux["t_dir_out"].float()
            t_dir_local = aux["t_dir_local"].float()
            rot_err = float(_rotation_error_deg(R_pred, R_gt).detach().cpu().view(-1)[0])
            tdir_abs = torch.rad2deg(torch.acos(torch.sum(t_dir_out * t_gt, dim=-1).clamp(-1.0, 1.0)))
            tdir_abs = torch.minimum(tdir_abs, 180.0 - tdir_abs)
            tdir_local = torch.rad2deg(torch.acos(torch.sum(t_dir_local * t_gt_Rt, dim=-1).clamp(-1.0, 1.0)))
            tdir_local = torch.minimum(tdir_local, 180.0 - tdir_local)
            tmag_gt = float(sample["t_gt_mag"])
            tmag_pred = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            tmag_ratio = float(tmag_pred / max(tmag_gt, 1.0e-8))
            tmag_rel_err = float(abs(tmag_ratio - 1.0))
            pair_pos_err = float(torch.linalg.norm(aux["t_vec_out"].float() - t_gt_vec.float(), dim=-1).detach().cpu().view(-1)[0])
            fine_diag = _epipolar_matching_diagnostics(
                aux.get("Wf_ab"),
                aux.get("Wf_ba"),
                aux.get("bearingA_f"),
                aux.get("bearingB_f"),
                R_gt,
                t_gt_vec,
                angle_thresh_deg=cfg.epi_angle_thresh_deg,
                allowed_mask=aux.get("allowed_mask"),
                routing_mask=aux.get("routing_mask"),
                topk=5,
            )
            coarse_diag = _epipolar_matching_diagnostics(
                aux.get("Wc_ab"),
                aux.get("Wc_ba"),
                aux.get("bearingA_c"),
                aux.get("bearingB_c"),
                R_gt,
                t_gt_vec,
                angle_thresh_deg=cfg.epi_angle_thresh_deg,
                allowed_mask=None,
                routing_mask=None,
                topk=5,
            )

        rows.append(
            {
                "split": split_name,
                "ds_idx": int(ds_idx),
                "scene": str(meta.get("scene")),
                "seq": str(meta.get("seq")),
                "scene_seq": str((meta.get("scene"), meta.get("seq"))),
                "i": int(meta.get("i", -1)),
                "j": int(meta.get("j", -1)),
                "k": int(meta.get("k", -1)),
                "dt_world": float(meta.get("dt_world", tmag_gt)),
                "gt_tmag": float(tmag_gt),
                "pred_tmag": float(tmag_pred),
                "tmag_ratio": float(tmag_ratio),
                "tmag_rel_err": float(tmag_rel_err),
                "rot_error": float(rot_err),
                "tdir_abs": float(tdir_abs.detach().cpu().view(-1)[0]),
                "tdir_local_A_abs": float(tdir_local.detach().cpu().view(-1)[0]),
                "RPE_rot": float(rot_err),
                "RPE_trans_dir": float(tdir_abs.detach().cpu().view(-1)[0]),
                "pair_pos_err": float(pair_pos_err),
                "fine_confidence": _scalarize(fine_diag, "max_matching_prob"),
                "fine_entropy": _scalarize(fine_diag, "matching_entropy"),
                "fine_epi_mass": _scalarize(fine_diag, "epi_mass_in_gt_band"),
                "fine_top1": _scalarize(fine_diag, "top1_in_gt_band"),
                "routing_recall": _scalarize(fine_diag, "routing_recall_in_gt_band"),
                "allowed_mask_density": _scalarize(fine_diag, "allowed_mask_density"),
                "coarse_confidence": _scalarize(coarse_diag, "max_matching_prob"),
                "coarse_entropy": _scalarize(coarse_diag, "matching_entropy"),
                "coarse_epi_mass": _scalarize(coarse_diag, "epi_mass_in_gt_band"),
                "coarse_top1": _scalarize(coarse_diag, "top1_in_gt_band"),
            }
        )
    return rows


def _dt_bucket(v: float) -> str:
    if 0.1 <= v < 0.3:
        return "[0.1,0.3)"
    if 0.3 <= v < 0.5:
        return "[0.3,0.5)"
    if 0.5 <= v < 1.0:
        return "[0.5,1.0)"
    if v < 0.1:
        return "<0.1"
    return ">=1.0"


def _k_bucket(v: int) -> str:
    mapping = {1: "k=1", 2: "k=2", 3: "k=3", 5: "k=5", 10: "k=10", 20: "k=20"}
    return mapping.get(int(v), f"k={int(v)}")


def _quantile_edges(vals: Sequence[float], q: Sequence[float]) -> List[float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0, 1.0]
    edges = np.quantile(arr, q).astype(np.float64).tolist()
    edges[0] -= 1.0e-6
    edges[-1] += 1.0e-6
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1.0e-6
    return [float(x) for x in edges]


def _bucket_from_edges(v: float, edges: Sequence[float], prefix: str) -> str:
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if lo <= v < hi:
            return f"{prefix}[{idx}]"
    return f"{prefix}[{len(edges) - 2}]"


def _mean(arr: Sequence[float]) -> float:
    vals = np.asarray(list(arr), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _median(arr: Sequence[float]) -> float:
    vals = np.asarray(list(arr), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _p90(arr: Sequence[float]) -> float:
    vals = np.asarray(list(arr), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 90)) if vals.size else float("nan")


def _distribution_distance(labels_a: Sequence[str], labels_b: Sequence[str]) -> Dict[str, Any]:
    keys = sorted(set(labels_a) | set(labels_b))
    ca = {k: 0 for k in keys}
    cb = {k: 0 for k in keys}
    for x in labels_a:
        ca[str(x)] += 1
    for x in labels_b:
        cb[str(x)] += 1
    pa = np.asarray([ca[k] for k in keys], dtype=np.float64)
    pb = np.asarray([cb[k] for k in keys], dtype=np.float64)
    pa = pa / max(pa.sum(), 1.0)
    pb = pb / max(pb.sum(), 1.0)
    eps = 1.0e-9
    kl = float(np.sum(pa * np.log((pa + eps) / (pb + eps))))
    l1 = float(np.abs(pa - pb).sum())
    return {
        "labels": keys,
        "train_probs": {k: float(pa[i]) for i, k in enumerate(keys)},
        "test_probs": {k: float(pb[i]) for i, k in enumerate(keys)},
        "l1": l1,
        "kl_train_to_test": kl,
    }


def _add_regime_labels(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_edges = _quantile_edges([float(r["gt_tmag"]) for r in rows if r["split"] == "train"], [0.0, 0.33, 0.66, 1.0])
    pred_edges = _quantile_edges([float(r["pred_tmag"]) for r in rows if r["split"] == "train"], [0.0, 0.33, 0.66, 1.0])
    conf_edges = _quantile_edges([float(r["fine_confidence"]) for r in rows if r["split"] == "train"], [0.0, 0.5, 1.0])
    ent_edges = _quantile_edges([float(r["fine_entropy"]) for r in rows if r["split"] == "train"], [0.0, 0.5, 1.0])

    train_comp = []
    for r in rows:
        r["dt_bucket"] = _dt_bucket(float(r["dt_world"]))
        r["k_bucket"] = _k_bucket(int(r["k"]))
        r["gt_tmag_bucket"] = _bucket_from_edges(float(r["gt_tmag"]), gt_edges, "gt_tmag_q")
        r["pred_tmag_bucket"] = _bucket_from_edges(float(r["pred_tmag"]), pred_edges, "pred_tmag_q")
        r["conf_bucket"] = _bucket_from_edges(float(r["fine_confidence"]), conf_edges, "conf_q")
        r["entropy_bucket"] = _bucket_from_edges(float(r["fine_entropy"]), ent_edges, "entropy_q")
        if r["split"] == "train":
            comp = (
                (float(r["rot_error"]) - _safe_float(r["rot_error"]))  # placeholder overwritten below
            )
            train_comp.append(comp)

    rot_train = np.asarray([float(r["rot_error"]) for r in rows if r["split"] == "train"], dtype=np.float64)
    tdir_train = np.asarray([float(r["tdir_local_A_abs"]) for r in rows if r["split"] == "train"], dtype=np.float64)
    pos_train = np.asarray([float(r["pair_pos_err"]) for r in rows if r["split"] == "train"], dtype=np.float64)
    rot_mu, rot_std = float(rot_train.mean()), float(max(rot_train.std(), 1.0e-6))
    tdir_mu, tdir_std = float(tdir_train.mean()), float(max(tdir_train.std(), 1.0e-6))
    pos_mu, pos_std = float(pos_train.mean()), float(max(pos_train.std(), 1.0e-6))
    train_score = []
    for r in rows:
        score = (
            (float(r["rot_error"]) - rot_mu) / rot_std
            + (float(r["tdir_local_A_abs"]) - tdir_mu) / tdir_std
            + (float(r["pair_pos_err"]) - pos_mu) / pos_std
        )
        r["high_error_score"] = float(score)
        if r["split"] == "train":
            train_score.append(score)
    high_thr = float(np.quantile(np.asarray(train_score, dtype=np.float64), 0.75))
    for r in rows:
        r["high_error"] = int(float(r["high_error_score"]) >= high_thr)
        r["dt_k_bucket"] = f"{r['dt_bucket']}|{r['k_bucket']}"
    return {
        "gt_tmag_edges": gt_edges,
        "pred_tmag_edges": pred_edges,
        "high_error_threshold": high_thr,
    }


def _bucket_summary(rows: Sequence[Dict[str, Any]], bucket_key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[bucket_key]), []).append(row)
    out: List[Dict[str, Any]] = []
    for label, items in sorted(groups.items(), key=lambda kv: kv[0]):
        rot = [float(r["rot_error"]) for r in items]
        tdir = [float(r["tdir_local_A_abs"]) for r in items]
        pos = [float(r["pair_pos_err"]) for r in items]
        out.append(
            {
                "bucket": label,
                "count": int(len(items)),
                "rot_mean": _mean(rot),
                "rot_median": _median(rot),
                "rot_p90": _p90(rot),
                "tdir_mean": _mean(tdir),
                "tdir_median": _median(tdir),
                "tdir_p90": _p90(tdir),
                "pos_mean": _mean(pos),
                "pos_median": _median(pos),
                "pos_p90": _p90(pos),
                "high_error_rate": _mean([float(r["high_error"]) for r in items]),
                "fine_conf_mean": _mean([float(r["fine_confidence"]) for r in items]),
                "fine_entropy_mean": _mean([float(r["fine_entropy"]) for r in items]),
                "pred_tmag_mean": _mean([float(r["pred_tmag"]) for r in items]),
                "gt_tmag_mean": _mean([float(r["gt_tmag"]) for r in items]),
                "tmag_ratio_mean": _mean([float(r["tmag_ratio"]) for r in items]),
            }
        )
    return out


def _reweighted_high_error_rate(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    train_count: Dict[str, int] = {}
    test_count: Dict[str, int] = {}
    test_rate: Dict[str, float] = {}
    for label in sorted(set(str(r[key]) for r in rows)):
        tr = [r for r in train if str(r[key]) == label]
        te = [r for r in test if str(r[key]) == label]
        train_count[label] = len(tr)
        test_count[label] = len(te)
        test_rate[label] = _mean([float(r["high_error"]) for r in te]) if te else float("nan")
    train_total = max(sum(train_count.values()), 1)
    test_total = max(sum(test_count.values()), 1)
    actual = sum((test_count[k] / test_total) * (0.0 if math.isnan(test_rate[k]) else test_rate[k]) for k in test_count)
    reweighted = sum((train_count[k] / train_total) * (0.0 if math.isnan(test_rate[k]) else test_rate[k]) for k in train_count)
    return {"actual_test_high_error_rate": float(actual), "reweighted_to_train": float(reweighted), "delta": float(actual - reweighted)}


def _worst_bucket_counterfactual(test_rows: Sequence[Dict[str, Any]], bucket_key: str) -> Dict[str, Any]:
    buckets = _bucket_summary(test_rows, bucket_key)
    eligible = [b for b in buckets if int(b["count"]) >= MIN_BUCKET_COUNT]
    if not eligible:
        return {"bucket": "N/A"}
    worst = max(eligible, key=lambda b: (float(b["high_error_rate"]), float(b["pos_mean"])))
    remain = [r for r in test_rows if str(r[bucket_key]) != str(worst["bucket"])]
    return {
        "bucket": worst["bucket"],
        "count": int(worst["count"]),
        "high_error_rate": float(worst["high_error_rate"]),
        "pair_pos_err_mean_before": _mean([float(r["pair_pos_err"]) for r in test_rows]),
        "pair_pos_err_mean_after": _mean([float(r["pair_pos_err"]) for r in remain]),
        "rot_mean_before": _mean([float(r["rot_error"]) for r in test_rows]),
        "rot_mean_after": _mean([float(r["rot_error"]) for r in remain]),
        "tdir_mean_before": _mean([float(r["tdir_local_A_abs"]) for r in test_rows]),
        "tdir_mean_after": _mean([float(r["tdir_local_A_abs"]) for r in remain]),
    }


def _dominant_train_like_bucket_eval(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    counts: Dict[str, int] = {}
    for r in train:
        counts[str(r[key])] = counts.get(str(r[key]), 0) + 1
    total = max(sum(counts.values()), 1)
    dominant = {k for k, v in counts.items() if v / total >= 0.10}
    test_dom = [r for r in test if str(r[key]) in dominant]
    return {
        "dominant_buckets": sorted(dominant),
        "test_count": int(len(test_dom)),
        "rot_mean": _mean([float(r["rot_error"]) for r in test_dom]),
        "tdir_mean": _mean([float(r["tdir_local_A_abs"]) for r in test_dom]),
        "pair_pos_err_mean": _mean([float(r["pair_pos_err"]) for r in test_dom]),
        "high_error_rate": _mean([float(r["high_error"]) for r in test_dom]),
    }


def _run_odometry_debug(
    model: PanoramaRelPoseModel,
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    cfg: Config,
    split_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"s4_{split_name}_", dir=str(REPO_ROOT / "checkpoints")))
    old_flag = bool(getattr(cfg, "save_odom_trajectory_debug", False))
    cfg.save_odom_trajectory_debug = True
    try:
        payload = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(tmp_dir), step=0, upd=0)
        debug_json = _read_json(tmp_dir / "odom_trajectory_debug_latest.json")
        step_rows: List[Dict[str, Any]] = []
        with (tmp_dir / "odom_trajectory_steps_latest.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_rows.append({k: row[k] for k in reader.fieldnames or []})
    finally:
        cfg.save_odom_trajectory_debug = old_flag
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"payload": payload, "debug_json": debug_json, "step_rows": step_rows}


def _chain_regime_summary(odom_debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    chains = odom_debug["debug_json"].get("chains", [])
    steps = odom_debug["step_rows"]
    by_chain: Dict[int, List[Dict[str, Any]]] = {}
    for step in steps:
        by_chain.setdefault(int(step["chain_id"]), []).append(step)
    out: List[Dict[str, Any]] = []
    for cid, chain in enumerate(chains):
        items = by_chain.get(cid, [])
        dt_labels = [_dt_bucket(_safe_float(s.get("dt_gt"))) for s in items]
        k_labels = [_k_bucket(int(s.get("k", -1))) for s in items]
        gt_tmag = [_safe_float(s.get("tmag_gt")) for s in items]
        pred_tmag = [_safe_float(s.get("tmag_pred")) for s in items]
        pos_err = [_safe_float(s.get("metric_pos_err")) for s in items]
        high_steps = [1.0 if _safe_float(s.get("metric_pos_err")) >= np.nanpercentile(np.asarray(pos_err, dtype=np.float64), 75) else 0.0 for s in items] if items else []
        out.append(
            {
                "chain_id": int(cid),
                "scene_seq": str(chain.get("scene_seq")),
                "num_steps": int(chain.get("num_steps", 0)),
                "ATE_proxy": _safe_float(chain.get("mean_metric_pos_err")),
                "drift_proxy": _safe_float(chain.get("final_metric_pos_err")),
                "path_ratio": _safe_float(chain.get("shape_metric", {}).get("path_length_ratio")),
                "dt_comp": {k: dt_labels.count(k) for k in sorted(set(dt_labels))},
                "k_comp": {k: k_labels.count(k) for k in sorted(set(k_labels))},
                "gt_tmag_mean": _mean(gt_tmag),
                "pred_tmag_mean": _mean(pred_tmag),
                "tmag_ratio_mean": _mean([p / max(g, 1.0e-8) for p, g in zip(pred_tmag, gt_tmag) if math.isfinite(p) and math.isfinite(g)]),
                "high_error_step_ratio": _mean(high_steps),
            }
        )
    return out


def _plot_distribution(comp: Dict[str, Any], path: Path, title: str) -> None:
    labels = comp["labels"]
    train_vals = [comp["train_probs"][k] for k in labels]
    test_vals = [comp["test_probs"][k] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.bar(x - 0.18, train_vals, width=0.36, label="train")
    ax.bar(x + 0.18, test_vals, width=0.36, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_error_by_bucket(summary: Sequence[Dict[str, Any]], value_key: str, path: Path, title: str) -> None:
    labels = [str(r["bucket"]) for r in summary]
    vals = [float(r[value_key]) for r in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _classify(
    shift_dt: Dict[str, Any],
    shift_k: Dict[str, Any],
    shift_gt_tmag: Dict[str, Any],
    shift_pred_tmag: Dict[str, Any],
    worst_dt: Dict[str, Any],
    worst_k: Dict[str, Any],
    worst_dtk: Dict[str, Any],
    worst_gt: Dict[str, Any],
    reweight_dtk: Dict[str, Any],
) -> Tuple[str, str]:
    shift_score = max(float(shift_dt["l1"]), float(shift_k["l1"]), float(shift_gt_tmag["l1"]), float(shift_pred_tmag["l1"]))
    dt_drop = float(worst_dt.get("pair_pos_err_mean_before", float("nan"))) - float(worst_dt.get("pair_pos_err_mean_after", float("nan")))
    k_drop = float(worst_k.get("pair_pos_err_mean_before", float("nan"))) - float(worst_k.get("pair_pos_err_mean_after", float("nan")))
    dtk_drop = float(worst_dtk.get("pair_pos_err_mean_before", float("nan"))) - float(worst_dtk.get("pair_pos_err_mean_after", float("nan")))
    gt_drop = float(worst_gt.get("pair_pos_err_mean_before", float("nan"))) - float(worst_gt.get("pair_pos_err_mean_after", float("nan")))
    if reweight_dtk["delta"] >= 0.05 and shift_score >= 0.20:
        return "TRAIN-TEST-REGIME-SHIFT", "test high-error rate drops noticeably when reweighted to the train dt×k regime, indicating a meaningful regime shift"
    if gt_drop >= max(dtk_drop, dt_drop, k_drop) + 0.03 and max(float(shift_gt_tmag["l1"]), float(shift_pred_tmag["l1"])) >= 0.20:
        return "TMAG-REGIME-DOMINANT", "magnitude regime shift is the strongest single-axis effect: predicted/ground-truth magnitude buckets move the most, and removing the worst magnitude bucket yields the largest error drop"
    if worst_dtk.get("bucket", "N/A") != "N/A" and worst_dtk["pair_pos_err_mean_before"] - worst_dtk["pair_pos_err_mean_after"] >= 0.05:
        return "MIXED-REGIME-EFFECT", "a combined dt×k bucket dominates much of the removable error mass, suggesting a mixed regime effect rather than a single pure axis"
    if worst_dt.get("bucket", "N/A") != "N/A" and worst_dt["pair_pos_err_mean_before"] - worst_dt["pair_pos_err_mean_after"] >= 0.05:
        return "DT-REGIME-DOMINANT", "removing the worst dt bucket reduces pair error materially, pointing to dt-dominant failure"
    if worst_k.get("bucket", "N/A") != "N/A" and worst_k["pair_pos_err_mean_before"] - worst_k["pair_pos_err_mean_after"] >= 0.05:
        return "K-REGIME-DOMINANT", "removing the worst k bucket reduces pair error materially, pointing to k-dominant failure"
    if max(float(shift_gt_tmag["l1"]), float(shift_pred_tmag["l1"])) >= 0.20:
        return "TMAG-REGIME-DOMINANT", "magnitude-regime distribution shift is the strongest single-axis signal"
    return "INCONCLUSIVE", "regime statistics move in the same direction as S3b, but no single axis is dominant enough for a stronger label"


def _write_table(lines: List[str], title: str, rows: Sequence[Dict[str, Any]]) -> None:
    lines.append(f"### {title}\n\n")
    lines.append("| bucket | count | rot_mean | rot_p90 | tdir_mean | tdir_p90 | pos_mean | pos_p90 | high_err | conf | entropy | tmag_ratio |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in rows:
        lines.append(
            f"| {r['bucket']} | {r['count']} | {_fmt(r['rot_mean'])} | {_fmt(r['rot_p90'])} | {_fmt(r['tdir_mean'])} | {_fmt(r['tdir_p90'])} | {_fmt(r['pos_mean'])} | {_fmt(r['pos_p90'])} | {_fmt(r['high_error_rate'])} | {_fmt(r['fine_conf_mean'])} | {_fmt(r['fine_entropy_mean'])} | {_fmt(r['tmag_ratio_mean'])} |\n"
        )
    lines.append("\n")


def main() -> None:
    cfg, restore_summary, policy_summary = _build_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_summary = _load_model(cfg, device)
    if load_summary["unexpected"]:
        raise RuntimeError(f"unexpected checkpoint keys found: {load_summary['unexpected'][:8]}")

    train_ds = _build_dataset(cfg, "train", k_list=PAIR_K_LIST)
    test_ds = _build_dataset(cfg, "test", k_list=PAIR_K_LIST)
    print(f"[S4] extracting train pairs: {len(train_ds)}", flush=True)
    train_rows = _extract_pair_rows(model, cfg, train_ds, "train", device)
    print(f"[S4] extracting test pairs: {len(test_ds)}", flush=True)
    test_rows = _extract_pair_rows(model, cfg, test_ds, "test", device)
    rows = train_rows + test_rows
    bucket_meta = _add_regime_labels(rows)

    train_dt = _bucket_summary(train_rows, "dt_bucket")
    test_dt = _bucket_summary(test_rows, "dt_bucket")
    train_k = _bucket_summary(train_rows, "k_bucket")
    test_k = _bucket_summary(test_rows, "k_bucket")
    train_gt = _bucket_summary(train_rows, "gt_tmag_bucket")
    test_gt = _bucket_summary(test_rows, "gt_tmag_bucket")
    train_dtk = _bucket_summary(train_rows, "dt_k_bucket")
    test_dtk = _bucket_summary(test_rows, "dt_k_bucket")

    shift_dt = _distribution_distance([str(r["dt_bucket"]) for r in train_rows], [str(r["dt_bucket"]) for r in test_rows])
    shift_k = _distribution_distance([str(r["k_bucket"]) for r in train_rows], [str(r["k_bucket"]) for r in test_rows])
    shift_gt = _distribution_distance([str(r["gt_tmag_bucket"]) for r in train_rows], [str(r["gt_tmag_bucket"]) for r in test_rows])
    shift_pred = _distribution_distance([str(r["pred_tmag_bucket"]) for r in train_rows], [str(r["pred_tmag_bucket"]) for r in test_rows])
    shift_high = _distribution_distance([str(r["dt_k_bucket"]) for r in train_rows if r["high_error"] == 1], [str(r["dt_k_bucket"]) for r in test_rows if r["high_error"] == 1])
    shift_conf = _distribution_distance([str(r["conf_bucket"]) for r in train_rows], [str(r["conf_bucket"]) for r in test_rows])
    shift_ent = _distribution_distance([str(r["entropy_bucket"]) for r in train_rows], [str(r["entropy_bucket"]) for r in test_rows])

    worst_dt = _worst_bucket_counterfactual(test_rows, "dt_bucket")
    worst_k = _worst_bucket_counterfactual(test_rows, "k_bucket")
    worst_dtk = _worst_bucket_counterfactual(test_rows, "dt_k_bucket")
    worst_gt = _worst_bucket_counterfactual(test_rows, "gt_tmag_bucket")
    reweight_dt = _reweighted_high_error_rate(rows, "dt_bucket")
    reweight_k = _reweighted_high_error_rate(rows, "k_bucket")
    reweight_dtk = _reweighted_high_error_rate(rows, "dt_k_bucket")
    dominant_train_like = _dominant_train_like_bucket_eval(rows, "dt_k_bucket")

    print("[S4] running chain regime summaries", flush=True)
    train_chain_ds = _build_dataset(cfg, "train", k_list=(1,))
    test_chain_ds = _build_dataset(cfg, "test", k_list=(1,))
    odom_train = _run_odometry_debug(model, train_chain_ds, cfg, "train_chain", device)
    odom_test = _run_odometry_debug(model, test_chain_ds, cfg, "test_chain", device)
    train_chain_summary = _chain_regime_summary(odom_train)
    test_chain_summary = _chain_regime_summary(odom_test)

    classification, reason = _classify(shift_dt, shift_k, shift_gt, shift_pred, worst_dt, worst_k, worst_dtk, worst_gt, reweight_dtk)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_distribution(shift_dt, FIG_DIR / "train_test_dt_distribution.png", "Train/Test dt Distribution")
    _plot_distribution(shift_gt, FIG_DIR / "train_test_tmag_distribution.png", "Train/Test gt_tmag Distribution")
    _plot_error_by_bucket(test_dt, "pos_mean", FIG_DIR / "error_by_dt_bucket.png", "Test Pair Position Error by dt Bucket")
    _plot_error_by_bucket(test_k, "pos_mean", FIG_DIR / "error_by_k_bucket.png", "Test Pair Position Error by k Bucket")
    _plot_error_by_bucket(test_dtk, "high_error_rate", FIG_DIR / "high_error_rate_by_bucket.png", "Test High-Error Rate by dt×k Bucket")

    lines: List[str] = []
    lines.append("# S4 dt k magnitude regime diagnostic report\n\n")
    lines.append("## 1. Executive summary\n\n")
    lines.append(f"- final classification: `{classification}`\n")
    lines.append(f"- summary: {reason}\n")
    lines.append(f"- dt/k/tmag regime dominates error: `{classification in {'DT-REGIME-DOMINANT', 'K-REGIME-DOMINANT', 'TMAG-REGIME-DOMINANT', 'TRAIN-TEST-REGIME-SHIFT', 'MIXED-REGIME-EFFECT'}}`\n")
    lines.append(f"- train/test regime shift exists: `{max(shift_dt['l1'], shift_k['l1'], shift_gt['l1'], shift_pred['l1']) >= 0.20}`\n")
    lines.append(f"- explains S3b DT/K-dominated result: `{classification != 'INCONCLUSIVE'}`\n\n")

    lines.append("## 2. Pair-level bucket tables\n\n")
    _write_table(lines, "Train dt buckets", train_dt)
    _write_table(lines, "Test dt buckets", test_dt)
    _write_table(lines, "Train k buckets", train_k)
    _write_table(lines, "Test k buckets", test_k)
    _write_table(lines, "Train gt_tmag buckets", train_gt)
    _write_table(lines, "Test gt_tmag buckets", test_gt)
    _write_table(lines, "Test dt x k buckets", test_dtk)

    lines.append("## 3. Train/test distribution shift\n\n")
    for name, comp in [
        ("dt", shift_dt),
        ("k", shift_k),
        ("gt_tmag", shift_gt),
        ("pred_tmag", shift_pred),
        ("high-error dt×k", shift_high),
        ("confidence", shift_conf),
        ("entropy", shift_ent),
    ]:
        lines.append(f"- {name}: L1=`{_fmt(comp['l1'])}` KL(train||test)=`{_fmt(comp['kl_train_to_test'])}`\n")
    lines.append(f"- strongest shift axis: `{max([('dt', shift_dt['l1']), ('k', shift_k['l1']), ('gt_tmag', shift_gt['l1']), ('pred_tmag', shift_pred['l1'])], key=lambda x: x[1])[0]}`\n\n")

    lines.append("## 4. Chain-level regime summary\n\n")
    lines.append("### Train chains\n\n")
    for rec in train_chain_summary:
        lines.append(f"- chain `{rec['chain_id']}` {rec['scene_seq']}: ATE_proxy=`{_fmt(rec['ATE_proxy'])}` drift_proxy=`{_fmt(rec['drift_proxy'])}` path_ratio=`{_fmt(rec['path_ratio'])}` dt_comp=`{rec['dt_comp']}` k_comp=`{rec['k_comp']}` high_error_step_ratio=`{_fmt(rec['high_error_step_ratio'])}`\n")
    lines.append("\n### Test chains\n\n")
    for rec in test_chain_summary:
        lines.append(f"- chain `{rec['chain_id']}` {rec['scene_seq']}: ATE_proxy=`{_fmt(rec['ATE_proxy'])}` drift_proxy=`{_fmt(rec['drift_proxy'])}` path_ratio=`{_fmt(rec['path_ratio'])}` dt_comp=`{rec['dt_comp']}` k_comp=`{rec['k_comp']}` high_error_step_ratio=`{_fmt(rec['high_error_step_ratio'])}`\n")
    lines.append("\n")

    lines.append("## 5. Counterfactual / reweighting diagnostic\n\n")
    lines.append(f"- reweight test high-error to train dt distribution: actual=`{_fmt(reweight_dt['actual_test_high_error_rate'])}` reweighted=`{_fmt(reweight_dt['reweighted_to_train'])}` delta=`{_fmt(reweight_dt['delta'])}`\n")
    lines.append(f"- reweight test high-error to train k distribution: actual=`{_fmt(reweight_k['actual_test_high_error_rate'])}` reweighted=`{_fmt(reweight_k['reweighted_to_train'])}` delta=`{_fmt(reweight_k['delta'])}`\n")
    lines.append(f"- reweight test high-error to train dt×k distribution: actual=`{_fmt(reweight_dtk['actual_test_high_error_rate'])}` reweighted=`{_fmt(reweight_dtk['reweighted_to_train'])}` delta=`{_fmt(reweight_dtk['delta'])}`\n")
    lines.append(
        f"- remove worst dt bucket `{worst_dt.get('bucket', 'N/A')}`: pair_pos_err "
        f"`{_fmt(worst_dt.get('pair_pos_err_mean_before', float('nan')))} -> {_fmt(worst_dt.get('pair_pos_err_mean_after', float('nan')))}`\n"
    )
    lines.append(
        f"- remove worst k bucket `{worst_k.get('bucket', 'N/A')}`: pair_pos_err "
        f"`{_fmt(worst_k.get('pair_pos_err_mean_before', float('nan')))} -> {_fmt(worst_k.get('pair_pos_err_mean_after', float('nan')))}`\n"
    )
    lines.append(
        f"- remove worst dt×k bucket `{worst_dtk.get('bucket', 'N/A')}`: pair_pos_err "
        f"`{_fmt(worst_dtk.get('pair_pos_err_mean_before', float('nan')))} -> {_fmt(worst_dtk.get('pair_pos_err_mean_after', float('nan')))}`\n"
    )
    lines.append(
        f"- remove worst gt_tmag bucket `{worst_gt.get('bucket', 'N/A')}`: pair_pos_err "
        f"`{_fmt(worst_gt.get('pair_pos_err_mean_before', float('nan')))} -> {_fmt(worst_gt.get('pair_pos_err_mean_after', float('nan')))}`\n"
    )
    lines.append(f"- dominant train-like dt×k buckets on test: `{dominant_train_like['dominant_buckets']}` with high_error_rate=`{_fmt(dominant_train_like['high_error_rate'])}`\n")
    lines.append("- interpretation on train-CV stability: if test high-error falls under train-style reweighting, then train-CV may indeed prefer train-optimal buckets that are less stable on test.\n\n")

    lines.append("## 6. Final classification\n\n")
    lines.append(f"- `{classification}`\n\n")

    lines.append("## 7. Next-step recommendation\n\n")
    if classification == "TRAIN-TEST-REGIME-SHIFT":
        lines.append("- redesign data split / sampling / CV before any model change\n")
    elif classification in {"DT-REGIME-DOMINANT", "K-REGIME-DOMINANT", "TMAG-REGIME-DOMINANT", "MIXED-REGIME-EFFECT"}:
        lines.append("- prioritize regime-aware calibration / evaluation / bucket-specific diagnostics, not new residual-head training\n")
    elif classification == "CONFIDENCE/ENTROPY-DOMINANT":
        lines.append("- try confidence-gated inference policy rather than a new head\n")
    else:
        lines.append("- stop new model experiments and write the final report from current evidence\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "classification": classification,
                "reason": reason,
                "worst_dt_k_bucket": worst_dtk.get("bucket", "N/A"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
