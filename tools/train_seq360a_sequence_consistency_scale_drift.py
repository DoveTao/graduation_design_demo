#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from datasets.dset2c_sequence_clip_dataset import Dset2CSequenceClipDataset
from miniyaml import load_yaml_like
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel, count_parameters
from pose_head import matrix_geodesic_distance
from train_struct360b_match_free_coarse_to_fine import (
    _cfg_from_dict,
    _inject_struct360b_cfg,
    evaluate_struct360b_pose,
)
import train360e_sequence_trajectory_export_and_ate_eval as traj


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seq360a_sequence_consistency_scale_drift.yaml"
TMAG_EPS = 1.0e-6


def _git(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout or "").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    x = _safe_float(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def _mean(vals: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _extract_final360_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return {}


def _extract_train360e_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for key in ("trajectory_path_ratio", "pose_coverage", "pred_path_length", "gt_path_length"):
        out[key] = payload.get(key)
    for mode in ("none", "se3", "sim3"):
        rmse = payload.get(f"ate_{mode}", {}).get("rmse") if isinstance(payload.get(f"ate_{mode}"), Mapping) else None
        out[f"ate_{mode}_rmse"] = rmse
    return out


def _extract_base360_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    pair = payload.get("pair_metrics", {}).get("all_pairs", {})
    out = {
        "trajectory_path_ratio": payload.get("trajectory_path_ratio"),
        "pair_path_ratio": pair.get("pair_component_path_ratio"),
        "signed_tdir_mean_deg": pair.get("signed_tdir_mean_deg"),
        "anti_parallel_rate": pair.get("anti_parallel_rate"),
        "tmag_median_ratio": pair.get("tmag_median_ratio"),
    }
    for mode in ("none", "se3", "sim3"):
        entry = payload.get(f"ate_{mode}", {})
        out[f"ate_{mode}_rmse"] = entry.get("rmse") if isinstance(entry, Mapping) else None
    return out


def _aggregate_existing_trajectory_root(root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for split in ("val", "test"):
        per_seq: Dict[str, Any] = {}
        error_buckets = {mode: [] for mode in ("none", "se3", "sim3")}
        pred_path = 0.0
        gt_path = 0.0
        coverage = []
        for path in sorted(root.glob(f"{split}/*/trajectory_metrics.json")):
            payload = _read_json(path)
            seq_id = path.parent.name
            per_seq[seq_id] = payload
            coverage.append(float(payload.get("pose_coverage") or 0.0))
            traj_eval = payload.get("trajectory_eval", {})
            pred_path += float(traj_eval.get("none", {}).get("pred_path_length") or 0.0)
            gt_path += float(traj_eval.get("none", {}).get("gt_path_length") or 0.0)
            for mode in error_buckets:
                error_buckets[mode].extend(float(x) for x in traj_eval.get(mode, {}).get("errors", []))
        split_metrics = {
            "sequence_count": len(per_seq),
            "pose_coverage": _mean(coverage),
            "trajectory_path_ratio": float(pred_path / max(gt_path, TMAG_EPS)) if gt_path > 0.0 else None,
            "pred_path_length": pred_path,
            "gt_path_length": gt_path,
            "per_sequence": per_seq,
        }
        for mode in ("none", "se3", "sim3"):
            errs = np.asarray(error_buckets[mode], dtype=np.float64)
            split_metrics[f"ate_{mode}_rmse"] = float(np.sqrt(np.mean(errs ** 2))) if errs.size else None
        out[split] = split_metrics
    return out


def _load_model(checkpoint_path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Config, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = _inject_struct360b_cfg(dict(payload.get("cfg", {})), cfg)
    ckpt_cfg = _cfg_from_dict(cfg_dict)
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=bool(cfg["model"].get("strict_load_attempt", False)))
    model.train()
    return model, ckpt_cfg, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def _subset_clip_dataset(ds: Dset2CSequenceClipDataset, max_count: Optional[int], seed: int):
    if max_count is None or int(max_count) <= 0 or len(ds) <= int(max_count):
        return ds, {"subset_used": False, "subset_count": len(ds), "original_count": len(ds)}
    rng = np.random.default_rng(int(seed))
    idx = sorted(int(x) for x in rng.permutation(len(ds))[: int(max_count)].tolist())
    return Subset(ds, idx), {
        "subset_used": True,
        "subset_count": len(idx),
        "original_count": len(ds),
        "subset_seed": int(seed),
    }


def _build_clip_dataset(cfg: Mapping[str, Any], split: str) -> Dset2CSequenceClipDataset:
    data = cfg["data"]
    return Dset2CSequenceClipDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        clip_len=int(data["clip_len"]),
        image_hw=tuple(int(x) for x in data["image_hw"]),
        max_frame_gap=int(data["max_frame_gap"]),
        max_timestamp_gap_factor=float(data["max_timestamp_gap_factor"]),
        tmag_epsilon=float(data["tmag_epsilon"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )


def _build_pair_dataset(cfg: Mapping[str, Any], split: str) -> Dset2CCanonicalPairDataset:
    data = cfg["data"]
    return Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        image_hw=tuple(int(x) for x in data["image_hw"]),
        tmag_epsilon=float(data["tmag_epsilon"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )


def _predict_clip_pairs(model: STRUCT360BMatchFreeCoarseToFineModel, batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    pair_i = batch["pair_i"].to(device, non_blocking=True)
    pair_j = batch["pair_j"].to(device, non_blocking=True)
    timestamps = batch["timestamps"].to(device, non_blocking=True)
    B, P = pair_i.shape
    flat_a: List[torch.Tensor] = []
    flat_b: List[torch.Tensor] = []
    flat_dt: List[torch.Tensor] = []
    for b in range(B):
        flat_a.append(images[b, pair_i[b]])
        flat_b.append(images[b, pair_j[b]])
        flat_dt.append((timestamps[b, pair_j[b]] - timestamps[b, pair_i[b]]).float())
    IA = torch.cat(flat_a, dim=0)
    IB = torch.cat(flat_b, dim=0)
    dt_world = torch.cat(flat_dt, dim=0)
    R, _t_local, aux = model(IA, IB, dt_world=dt_world)
    return {
        "R": R.view(B, P, 3, 3),
        "tdir": aux["t_dir_out"].view(B, P, 3),
        "tmag": aux["t_mag"].view(B, P),
        "log_tmag": torch.log(aux["t_mag"].view(B, P).clamp_min(TMAG_EPS)),
        "coarse_R": aux["coarse_R"].view(B, P, 3, 3),
        "coarse_tdir": aux["coarse_t_dir_out"].view(B, P, 3),
        "coarse_tmag": aux["coarse_t_mag"].view(B, P),
        "coarse_log_tmag": torch.log(aux["coarse_t_mag"].view(B, P).clamp_min(TMAG_EPS)),
        "aux": aux,
    }


def _compose_ba(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.matmul(R2, R1), torch.matmul(R2, t1.unsqueeze(-1)).squeeze(-1) + t2


def _vector_angle_batch_deg(a: torch.Tensor, b: torch.Tensor, *, absolute: bool) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1, eps=TMAG_EPS)
    b = F.normalize(b.float(), dim=-1, eps=TMAG_EPS)
    cosine = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    if absolute:
        cosine = cosine.abs()
    return torch.rad2deg(torch.acos(cosine))


def _pair_pose_loss(
    pred_R: torch.Tensor,
    pred_tdir: torch.Tensor,
    pred_log_tmag: torch.Tensor,
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    *,
    tmag_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gt_log_tmag = torch.log(torch.linalg.norm(gt_t.float(), dim=-1).clamp_min(TMAG_EPS))
    rot_loss = matrix_geodesic_distance(pred_R.float(), gt_R.float()).mean()
    tdir_loss = (1.0 - (F.normalize(pred_tdir.float(), dim=-1, eps=TMAG_EPS) * F.normalize(gt_t.float(), dim=-1, eps=TMAG_EPS)).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    log_tmag_loss = F.smooth_l1_loss(pred_log_tmag.float(), gt_log_tmag.float())
    total = rot_loss + tdir_loss + float(tmag_weight) * log_tmag_loss
    return total, {
        "rot_loss": rot_loss,
        "tdir_loss": tdir_loss,
        "log_tmag_loss": log_tmag_loss,
    }


def _sequence_losses(pred: Mapping[str, torch.Tensor], batch: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    loss_cfg = cfg["loss"]
    gt_R = batch["R_BA"].to(pred["R"].device).float()
    gt_t = batch["t_BA_B"].to(pred["R"].device).float()
    pair_i = batch["pair_i"].to(pred["R"].device)
    pair_j = batch["pair_j"].to(pred["R"].device)
    adjacent_mask = batch["adjacent_mask"].to(pred["R"].device).bool()
    B = gt_R.shape[0]

    pair_losses: List[torch.Tensor] = []
    pair_rot: List[torch.Tensor] = []
    pair_tdir: List[torch.Tensor] = []
    pair_log: List[torch.Tensor] = []
    coarse_pair_losses: List[torch.Tensor] = []
    comp_rot_terms: List[torch.Tensor] = []
    comp_tdir_terms: List[torch.Tensor] = []
    comp_log_terms: List[torch.Tensor] = []
    rot_accum_terms: List[torch.Tensor] = []
    path_terms: List[torch.Tensor] = []

    for b in range(B):
        adj_idx = torch.where(adjacent_mask[b])[0]
        pair_total, pair_detail = _pair_pose_loss(
            pred["R"][b, adj_idx],
            pred["tdir"][b, adj_idx],
            pred["log_tmag"][b, adj_idx],
            gt_R[b, adj_idx],
            gt_t[b, adj_idx],
            tmag_weight=float(loss_cfg["pair_tmag_weight"]),
        )
        pair_losses.append(pair_total)
        pair_rot.append(pair_detail["rot_loss"])
        pair_tdir.append(pair_detail["tdir_loss"])
        pair_log.append(pair_detail["log_tmag_loss"])
        coarse_total, _ = _pair_pose_loss(
            pred["coarse_R"][b, adj_idx],
            pred["coarse_tdir"][b, adj_idx],
            pred["coarse_log_tmag"][b, adj_idx],
            gt_R[b, adj_idx],
            gt_t[b, adj_idx],
            tmag_weight=float(loss_cfg["pair_tmag_weight"]),
        )
        coarse_pair_losses.append(coarse_total)

        lookup = {(int(pair_i[b, p]), int(pair_j[b, p])): p for p in range(pair_i.shape[1])}
        clip_len = int(batch["images"].shape[1])
        pred_path = pred["tmag"][b, adj_idx].sum()
        gt_path = torch.linalg.norm(gt_t[b, adj_idx], dim=-1).sum()
        path_terms.append(torch.abs(torch.log((pred_path + TMAG_EPS) / (gt_path + TMAG_EPS))))

        for start in range(clip_len - 2):
            R_comp = pred["R"][b, lookup[(start, start + 1)]]
            t_comp = pred["tdir"][b, lookup[(start, start + 1)]] * pred["tmag"][b, lookup[(start, start + 1)]]
            for end in range(start + 2, clip_len):
                step = lookup[(end - 1, end)]
                R_comp, t_comp = _compose_ba(
                    R_comp,
                    t_comp,
                    pred["R"][b, step],
                    pred["tdir"][b, step] * pred["tmag"][b, step],
                )
                target = lookup[(start, end)]
                gt_target_t = gt_t[b, target]
                gt_target_R = gt_R[b, target]
                comp_rot = matrix_geodesic_distance(R_comp.unsqueeze(0), gt_target_R.unsqueeze(0)).mean()
                comp_tdir = (1.0 - (F.normalize(t_comp.unsqueeze(0), dim=-1, eps=TMAG_EPS) * F.normalize(gt_target_t.unsqueeze(0), dim=-1, eps=TMAG_EPS)).sum(dim=-1).clamp(-1.0, 1.0)).mean()
                comp_log = torch.abs(
                    torch.log(torch.linalg.norm(t_comp).clamp_min(TMAG_EPS))
                    - torch.log(torch.linalg.norm(gt_target_t).clamp_min(TMAG_EPS))
                )
                comp_rot_terms.append(comp_rot)
                comp_tdir_terms.append(comp_tdir)
                comp_log_terms.append(comp_log)
                rot_accum_terms.append(comp_rot)

    zero = pred["R"].sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    coarse_pair_loss = torch.stack(coarse_pair_losses).mean() if coarse_pair_losses else zero
    comp_rot = torch.stack(comp_rot_terms).mean() if comp_rot_terms else zero
    comp_tdir = torch.stack(comp_tdir_terms).mean() if comp_tdir_terms else zero
    comp_log = torch.stack(comp_log_terms).mean() if comp_log_terms else zero
    path_loss = torch.stack(path_terms).mean() if path_terms else zero
    rot_accum = torch.stack(rot_accum_terms).mean() if rot_accum_terms else zero
    residual_reg = model_residual = pred["aux"]["residual_gate"].sum() * 0.0
    if hasattr(pred["aux"], "get"):
        pass
    return {
        "pair_loss": pair_loss,
        "pair_rot_loss": torch.stack(pair_rot).mean() if pair_rot else zero,
        "pair_tdir_loss": torch.stack(pair_tdir).mean() if pair_tdir else zero,
        "pair_log_tmag_loss": torch.stack(pair_log).mean() if pair_log else zero,
        "coarse_pair_loss": coarse_pair_loss,
        "comp_rot_loss": comp_rot,
        "comp_tdir_loss": comp_tdir,
        "comp_log_tmag_loss": comp_log,
        "path_loss": path_loss,
        "rot_accum_loss": rot_accum,
        "residual_reg_loss": residual_reg + model_residual,
    }


def _loss_total(losses: Mapping[str, torch.Tensor], residual_reg: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    loss_cfg = cfg["loss"]
    return (
        float(loss_cfg["pair_loss"]) * losses["pair_loss"]
        + float(loss_cfg["coarse_aux_weight"]) * losses["coarse_pair_loss"]
        + float(loss_cfg["lambda_comp_rot"]) * losses["comp_rot_loss"]
        + float(loss_cfg["lambda_comp_tdir"]) * losses["comp_tdir_loss"]
        + float(loss_cfg["lambda_comp_log_tmag"]) * losses["comp_log_tmag_loss"]
        + float(loss_cfg["lambda_path"]) * losses["path_loss"]
        + float(loss_cfg["lambda_rot_accum"]) * losses["rot_accum_loss"]
        + float(loss_cfg["lambda_delta_reg"]) * residual_reg
    )


@torch.no_grad()
def _evaluate_clip_metrics(
    model: STRUCT360BMatchFreeCoarseToFineModel,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    comp_rot_deg: List[float] = []
    comp_tdir_deg: List[float] = []
    comp_tmag_ratio: List[float] = []
    clip_path_ratio: List[float] = []
    delta_log_abs: List[float] = []
    residual_gate: List[float] = []
    skipped = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        pred = _predict_clip_pairs(model, batch, device)
        gt_R = batch["R_BA"].to(device).float()
        gt_t = batch["t_BA_B"].to(device).float()
        pair_i = batch["pair_i"].to(device)
        pair_j = batch["pair_j"].to(device)
        adjacent_mask = batch["adjacent_mask"].to(device).bool()
        B = gt_R.shape[0]
        for b in range(B):
            lookup = {(int(pair_i[b, p]), int(pair_j[b, p])): p for p in range(pair_i.shape[1])}
            clip_len = int(batch["images"].shape[1])
            adj_idx = torch.where(adjacent_mask[b])[0]
            pred_path = float(pred["tmag"][b, adj_idx].sum().detach().cpu())
            gt_path = float(torch.linalg.norm(gt_t[b, adj_idx], dim=-1).sum().detach().cpu())
            clip_path_ratio.append(pred_path / max(gt_path, TMAG_EPS))
            for start in range(clip_len - 2):
                R_comp = pred["R"][b, lookup[(start, start + 1)]]
                t_comp = pred["tdir"][b, lookup[(start, start + 1)]] * pred["tmag"][b, lookup[(start, start + 1)]]
                for end in range(start + 2, clip_len):
                    step = lookup[(end - 1, end)]
                    R_comp, t_comp = _compose_ba(
                        R_comp,
                        t_comp,
                        pred["R"][b, step],
                        pred["tdir"][b, step] * pred["tmag"][b, step],
                    )
                    target = lookup[(start, end)]
                    gt_target_t = gt_t[b, target]
                    gt_target_R = gt_R[b, target]
                    comp_rot_deg.append(
                        float(matrix_geodesic_distance(R_comp.unsqueeze(0), gt_target_R.unsqueeze(0)).detach().cpu().item() * (180.0 / math.pi))
                    )
                    comp_tdir_deg.append(
                        float(_vector_angle_batch_deg(t_comp.unsqueeze(0), gt_target_t.unsqueeze(0), absolute=False).detach().cpu().item())
                    )
                    comp_tmag_ratio.append(
                        float(torch.linalg.norm(t_comp).detach().cpu() / max(float(torch.linalg.norm(gt_target_t).detach().cpu()), TMAG_EPS))
                    )
        delta_log_abs.extend(pred["aux"]["delta_log_tmag_abs"].detach().cpu().view(-1).numpy().tolist())
        residual_gate.extend(pred["aux"]["residual_gate"].detach().cpu().view(-1).numpy().tolist())
    return {
        "count": len(comp_rot_deg),
        "comp_rot_error_deg": _mean(comp_rot_deg),
        "comp_tdir_error_deg": _mean(comp_tdir_deg),
        "comp_tmag_ratio": float(np.median(np.asarray(comp_tmag_ratio, dtype=np.float64))) if comp_tmag_ratio else None,
        "clip_path_ratio": _mean(clip_path_ratio),
        "delta_log_tmag_abs_mean": _mean(delta_log_abs),
        "residual_gate_mean": _mean(residual_gate),
        "skipped": int(skipped),
    }


def _selection_score(pair_metrics: Mapping[str, Any], clip_metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    score_cfg = cfg["evaluation"]["selection_score"]
    signed = float(pair_metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(pair_metrics.get("anti_parallel_rate") or 1.0)
    tmag_ratio = float(pair_metrics.get("tmag_median_ratio") or TMAG_EPS)
    path_ratio = float(pair_metrics.get("path_ratio") or TMAG_EPS)
    clip_path_ratio = float(clip_metrics.get("clip_path_ratio") or TMAG_EPS)
    comp_rot = float(clip_metrics.get("comp_rot_error_deg") or 0.0)
    comp_tdir = float(clip_metrics.get("comp_tdir_error_deg") or 0.0)
    return (
        float(score_cfg["signed_tdir_weight"]) * signed
        + float(score_cfg["anti_parallel_weight"]) * anti
        + float(score_cfg["tmag_ratio_weight"]) * abs(math.log(max(tmag_ratio, TMAG_EPS)))
        + float(score_cfg["path_ratio_weight"]) * abs(math.log(max(path_ratio, TMAG_EPS)))
        + float(score_cfg["clip_path_ratio_weight"]) * abs(math.log(max(clip_path_ratio, TMAG_EPS)))
        + float(score_cfg["comp_rot_weight"]) * comp_rot
        + float(score_cfg["comp_tdir_weight"]) * comp_tdir
    )


def _make_optimizer(model: STRUCT360BMatchFreeCoarseToFineModel, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    base_params: List[torch.nn.Parameter] = []
    fine_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("struct360b_"):
            fine_params.append(param)
        else:
            base_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": float(train_cfg["coarse_lr"]), "name": "base"},
            {"params": fine_params, "lr": float(train_cfg["fine_lr"]), "name": "fine"},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def _set_epoch_lrs(optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any], epoch: int) -> Dict[str, float]:
    train_cfg = cfg["training"]
    total_epochs = max(int(train_cfg["epochs"]), 1)
    if str(train_cfg.get("scheduler", "none")) == "cosine":
        progress = 0.0 if total_epochs == 1 else float(epoch) / float(total_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        cosine = 1.0
    coarse_target = float(train_cfg["coarse_lr"])
    fine_target = float(train_cfg["fine_lr"])
    min_lr = float(train_cfg.get("min_lr", 0.0))
    coarse_lr = min_lr + (coarse_target - min_lr) * cosine
    fine_lr = min_lr + (fine_target - min_lr) * cosine
    if bool(cfg["model"].get("freeze_coarse_warmup", False)) and epoch < int(train_cfg.get("warmup_epochs", 0)):
        coarse_lr = float(train_cfg.get("warmup_coarse_lr", 0.0))
        fine_lr = float(train_cfg.get("warmup_fine_lr", fine_target))
    for group in optimizer.param_groups:
        if group.get("name") == "base":
            group["lr"] = coarse_lr
        else:
            group["lr"] = fine_lr
    return {"base_lr": coarse_lr, "fine_lr": fine_lr}


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, ckpt_cfg: Config, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "cfg": dict(ckpt_cfg.__dict__),
            "metadata": _jsonable(dict(metadata)),
        },
        str(path),
    )


def _precheck(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    branch = _git(["git", "branch", "--show-current"])
    git_status_short = _git(["git", "status", "--short"]).splitlines()
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    blockers: List[str] = []
    if branch not in [str(x) for x in cfg["prechecks"]["expected_branches"]]:
        blockers.append(f"current branch mismatch: {branch}")
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB")
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"]["use_cuda_if_available"]) else "cpu")
    if device.type != "cuda":
        blockers.append("CUDA unavailable in TRAIN360 env")

    required_paths = {name: (REPO_ROOT / rel) for name, rel in cfg["inputs"].items()}
    for name, path in required_paths.items():
        if not path.exists():
            blockers.append(f"missing required path: {name} -> {path}")

    train_pair = _build_pair_dataset(cfg, "train")
    val_pair = _build_pair_dataset(cfg, "val")
    test_pair = _build_pair_dataset(cfg, "test")
    split_audit = summarize_manifest_group([train_pair, val_pair, test_pair])
    if split_audit["has_overlap"]:
        blockers.append(f"manifest overlap detected: {split_audit['sequence_overlap']}")

    train_clip = _build_clip_dataset(cfg, "train")
    val_clip = _build_clip_dataset(cfg, "val")
    test_clip = _build_clip_dataset(cfg, "test")

    init_checkpoint = REPO_ROOT / cfg["inputs"]["init_checkpoint"]
    if not init_checkpoint.is_file():
        blockers.append(f"missing init checkpoint: {init_checkpoint}")

    final360i_val = _extract_final360_metrics(_read_json(REPO_ROOT / cfg["inputs"]["final360i_val_metrics"]))
    final360i_test = _extract_final360_metrics(_read_json(REPO_ROOT / cfg["inputs"]["final360i_test_metrics"]))
    train360e_val = _extract_train360e_metrics(_read_json(REPO_ROOT / cfg["inputs"]["train360e_val_metrics"]))
    train360e_test = _extract_train360e_metrics(_read_json(REPO_ROOT / cfg["inputs"]["train360e_test_metrics"]))
    seq360b_traj = _aggregate_existing_trajectory_root(REPO_ROOT / "external_baselines" / "results" / "seq360b_scale_smoothing_trajectory")
    base360d_val = _extract_base360_metrics(_read_json(REPO_ROOT / cfg["inputs"]["base360d_val_metrics"]))
    base360d_test = _extract_base360_metrics(_read_json(REPO_ROOT / cfg["inputs"]["base360d_test_metrics"]))

    return {
        "blockers": blockers,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_status_short": git_status_short,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "disk_free_gb": disk_free_gb,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "train_pair_count": len(train_pair),
        "val_pair_count": len(val_pair),
        "test_pair_count": len(test_pair),
        "train_clip_summary": train_clip.get_clip_summary(),
        "val_clip_summary": val_clip.get_clip_summary(),
        "test_clip_summary": test_clip.get_clip_summary(),
        "split_audit": split_audit,
        "baseline_recap": {
            "FINAL360I_val_pair": final360i_val,
            "FINAL360I_test_pair": final360i_test,
            "TRAIN360E_val_trajectory": train360e_val,
            "TRAIN360E_test_trajectory": train360e_test,
            "SEQ360B_val_trajectory": seq360b_traj.get("val", {}),
            "SEQ360B_test_trajectory": seq360b_traj.get("test", {}),
            "BASE360D_val_trajectory": base360d_val,
            "BASE360D_test_trajectory": base360d_test,
        },
    }


def _write_blocker_artifacts(cfg: Mapping[str, Any], precheck: Mapping[str, Any]) -> None:
    report_lines = [
        "# SEQ360A sequence consistency scale drift stabilization",
        "",
        "## Blockers",
    ]
    report_lines.extend([f"- {line}" for line in precheck["blockers"]])
    report_lines.extend(
        [
            "",
            "## Precheck snapshot",
            f"- branch: `{precheck['branch']}`",
            f"- disk_free_gb: `{precheck['disk_free_gb']}`",
            f"- cuda_available: `{precheck['cuda_available']}`",
        ]
    )
    report_path = REPO_ROOT / cfg["outputs"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    blocker_payload = {
        "task_name": cfg["task_name"],
        "training_executed": False,
        "checkpoint_saved": False,
        "blockers": list(precheck["blockers"]),
        "precheck": _jsonable(dict(precheck)),
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], blocker_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], blocker_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_val_metrics_path"], blocker_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_test_metrics_path"], blocker_payload)
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text(
        "# SEQ360A blocked\n\n" + "\n".join(f"- {line}" for line in precheck["blockers"]) + "\n",
        encoding="utf-8",
    )


def _run_trajectory_eval(cfg: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = cfg["data"]
    val_pair = _build_pair_dataset(cfg, "val")
    test_pair = _build_pair_dataset(cfg, "test")
    val_specs, val_blockers = traj._build_sequence_specs(val_pair, "val")
    test_specs, test_blockers = traj._build_sequence_specs(test_pair, "test")
    convention, convention_summary, convention_warnings = traj._convention_sanity({**val_specs, **test_specs})
    if convention == "blocked_convention" or val_blockers or test_blockers:
        raise RuntimeError(f"trajectory convention/spec blockers: {val_blockers + test_blockers + list(convention_warnings)}")

    original_results_root = traj.RESULTS_ROOT
    original_checkpoint = traj.CHECKPOINT_PATH
    original_task = traj.TASK_NAME
    try:
        traj.RESULTS_ROOT = REPO_ROOT / cfg["outputs"]["trajectory_dir"]
        traj.CHECKPOINT_PATH = checkpoint_path
        traj.TASK_NAME = str(cfg["task_name"])
        model, load_summary = traj._load_model_from_checkpoint(checkpoint_path, device)
        predictions = traj._iterate_adjacent_predictions(
            model,
            val_pair,
            device,
            batch_size=int(data["eval_batch_size"]),
            num_workers=int(data["num_workers"]),
        )
        predictions.extend(
            traj._iterate_adjacent_predictions(
                model,
                test_pair,
                device,
                batch_size=int(data["eval_batch_size"]),
                num_workers=int(data["num_workers"]),
            )
        )
        pred_by_split_seq: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in predictions:
            pred_by_split_seq.setdefault((row["split"], row["sequence"]), []).append(row)
        git_commit = _git(["git", "rev-parse", "HEAD"])
        git_branch = _git(["git", "branch", "--show-current"])
        val_per_seq = {
            seq_id: traj._compose_sequence_trajectory(
                "val",
                seq_id,
                spec,
                pred_by_split_seq.get(("val", seq_id), []),
                selected_convention=convention,
                git_commit=git_commit,
                git_branch=git_branch,
            )
            for seq_id, spec in val_specs.items()
        }
        test_per_seq = {
            seq_id: traj._compose_sequence_trajectory(
                "test",
                seq_id,
                spec,
                pred_by_split_seq.get(("test", seq_id), []),
                selected_convention=convention,
                git_commit=git_commit,
                git_branch=git_branch,
            )
            for seq_id, spec in test_specs.items()
        }
        val_metrics = traj._aggregate_split("val", val_per_seq)
        test_metrics = traj._aggregate_split("test", test_per_seq)
        val_metrics["selected_convention"] = convention
        test_metrics["selected_convention"] = convention
        val_metrics["convention_sanity"] = convention_summary
        test_metrics["convention_sanity"] = convention_summary
        val_metrics["checkpoint_load_summary"] = load_summary
        test_metrics["checkpoint_load_summary"] = load_summary
        return val_metrics, test_metrics
    finally:
        traj.RESULTS_ROOT = original_results_root
        traj.CHECKPOINT_PATH = original_checkpoint
        traj.TASK_NAME = original_task


def _compare_pair(metrics: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    improved = 0
    total = 0
    for key, smaller_better in (
        ("signed_tdir_mean_deg", True),
        ("anti_parallel_rate", True),
        ("tmag_median_ratio", False),
        ("path_ratio", False),
    ):
        ours = _safe_float(metrics.get(key))
        theirs = _safe_float(ref.get(key))
        if ours is None or theirs is None:
            continue
        total += 1
        if key == "tmag_median_ratio":
            if abs(math.log(max(ours, TMAG_EPS))) < abs(math.log(max(theirs, TMAG_EPS))):
                improved += 1
        elif key == "path_ratio":
            if abs(math.log(max(ours, TMAG_EPS))) < abs(math.log(max(theirs, TMAG_EPS))):
                improved += 1
        elif smaller_better and ours < theirs:
            improved += 1
    if improved >= 3:
        return "better"
    if improved >= 1:
        return "partial"
    return "worse"


def _compare_trajectory(metrics: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    improved = 0
    total = 0
    for key in ("trajectory_path_ratio", "ate_se3_rmse", "ate_sim3_rmse"):
        ours = _safe_float(metrics.get(key))
        theirs = _safe_float(ref.get(key))
        if ours is None or theirs is None:
            continue
        total += 1
        if key == "trajectory_path_ratio":
            if abs(math.log(max(ours, TMAG_EPS))) < abs(math.log(max(theirs, TMAG_EPS))):
                improved += 1
        elif ours < theirs:
            improved += 1
    if improved >= 2:
        return "better"
    if improved >= 1:
        return "partial"
    return "worse"


def _classify(pair_test: Mapping[str, Any], traj_test: Mapping[str, Any], final360i_pair: Mapping[str, Any], seq360b_traj: Mapping[str, Any]) -> str:
    signed = float(pair_test.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(pair_test.get("anti_parallel_rate") or 1.0)
    tmag = float(pair_test.get("tmag_median_ratio") or 0.0)
    path = float(traj_test.get("trajectory_path_ratio") or float("inf"))
    ate_se3 = float(traj_test.get("ate_se3_rmse") or float("inf"))
    ate_sim3 = float(traj_test.get("ate_sim3_rmse") or float("inf"))
    if (
        signed > 54.96
        or anti > 0.2157
        or tmag > 1.5
        or path >= 1.7563
        or int(pair_test.get("nan_inf_count") or 0) > 0
    ):
        return "regression"
    if path < 1.3502 and (ate_se3 < 75.9469 or ate_sim3 < 27.5647):
        return "better"
    if path <= 1.35 and ate_se3 <= 75.95 and ate_sim3 <= 27.57 and 0.75 <= tmag <= 1.25:
        return "balanced_success"
    if ate_sim3 < float(seq360b_traj.get("ate_sim3_rmse") or float("inf")) and (
        signed > float(final360i_pair.get("signed_tdir_mean_deg") or float("inf"))
        or anti > float(final360i_pair.get("anti_parallel_rate") or 1.0)
    ):
        return "trajectory_improved_pair_tradeoff"
    if path < 1.3502 and ate_sim3 >= 27.5647:
        return "scale_improved_shape_unchanged"
    return "no_improvement"


def _report_text(
    cfg: Mapping[str, Any],
    precheck: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    pair_val: Mapping[str, Any],
    pair_test: Mapping[str, Any],
    clip_val: Mapping[str, Any],
    clip_test: Mapping[str, Any],
    traj_val: Mapping[str, Any],
    traj_test: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    classification: str,
) -> str:
    lines = [
        "# SEQ360A sequence consistency scale drift stabilization",
        "",
        "## 1. Executive summary",
        f"- training executed: `true`",
        f"- checkpoint saved: `true`",
        f"- best checkpoint: `{train_summary['best_checkpoint']}`",
        f"- init checkpoint: `{REPO_ROOT / cfg['inputs']['init_checkpoint']}`",
        f"- classification: `{classification}`",
        f"- pair test metrics: `signed_tdir={pair_test.get('signed_tdir_mean_deg')}`, `anti_parallel={pair_test.get('anti_parallel_rate')}`, `tmag_median_ratio={pair_test.get('tmag_median_ratio')}`, `path_ratio={pair_test.get('path_ratio')}`",
        f"- trajectory test metrics: `ATE none={traj_test.get('ate_none_rmse')}`, `ATE SE3={traj_test.get('ate_se3_rmse')}`, `ATE Sim3={traj_test.get('ate_sim3_rmse')}`, `path_ratio={traj_test.get('trajectory_path_ratio')}`",
        "",
        "## 2. Motivation",
        "- FINAL360I pair-level strong.",
        "- TRAIN360E exposed trajectory drift under sequential composition.",
        "- SEQ360B partially corrected scale/path drift but left Sim3 unchanged and over-corrected pair tmag.",
        "- SEQ360A targets contiguous composition consistency and trajectory shape stability.",
        "",
        "## 3. Data protocol",
        "- DSET2C canonical split used: `true`",
        "- train clips come from train manifest only: `true`",
        "- val used for model selection only: `true`",
        "- test used for final evaluation only: `true`",
        "- random pair split used: `false`",
        "- direct raw glob split used: `false`",
        "",
        "## 4. Sequence clip construction",
        f"- clip_len: `{cfg['data']['clip_len']}`",
        f"- train clips: `{precheck['train_clip_summary']['clip_count']}`",
        f"- val clips: `{precheck['val_clip_summary']['clip_count']}`",
        f"- test clips: `{precheck['test_clip_summary']['clip_count']}`",
        f"- skipped train clips: `{precheck['train_clip_summary']['skipped_clip_count']}`",
        f"- continuity checks enforced: `frame continuity`, `timestamp gap`, `no cross-sequence clip`",
        "",
        "## 5. Model/init",
        f"- init checkpoint: `{REPO_ROOT / cfg['inputs']['init_checkpoint']}`",
        "- architecture: `STRUCT360B match-free coarse-to-fine pose residual refinement`",
        "- explicit matching: `false`",
        "- RANSAC / PnP / BA: `false / false / false`",
        "",
        "## 6. Loss design",
        "- adjacent pair loss: `rot + tdir + log_tmag`",
        "- k-step composition loss: `comp_rot + comp_tdir + comp_log_tmag`",
        "- scale/path loss: `clip path log-ratio`",
        "- rotation accumulation loss: `SO3 geodesic on composed k-step rotation`",
        "- residual regularization: `delta magnitude penalty`",
        f"- weights: `{cfg['loss']}`",
        "",
        "## 7. Training setup",
        f"- epochs: `{cfg['training']['epochs']}`",
        f"- batch_size: `{cfg['data']['train_batch_size']}`",
        f"- lr groups: `base={cfg['training']['coarse_lr']}`, `fine={cfg['training']['fine_lr']}`",
        f"- freeze_coarse_warmup: `{cfg['model']['freeze_coarse_warmup']}`",
        f"- seed: `{cfg['training']['seed']}`",
        f"- val score: `pair_signed + 60*anti + 20*|log tmag| + 10*|log pair path| + 30*|log clip path| + 10*comp_rot + 10*comp_tdir`",
        "",
        "## 8. Validation results",
        f"- selected epoch: `{train_summary['best_epoch']}`",
        f"- pair val metrics: `{pair_val}`",
        f"- clip val metrics: `{clip_val}`",
        "",
        "## 9. Test pair-level results",
        f"- FINAL360I vs SEQ360A: `FINAL360I={precheck['baseline_recap']['FINAL360I_test_pair']}`, `SEQ360A={pair_test}`",
        "",
        "## 10. Test trajectory results",
        f"- TRAIN360E FINAL360I trajectory: `{precheck['baseline_recap']['TRAIN360E_test_trajectory']}`",
        f"- SEQ360B trajectory: `{precheck['baseline_recap']['SEQ360B_test_trajectory']}`",
        f"- SEQ360A trajectory: `{traj_test}`",
        f"- BASE360D trajectory: `{precheck['baseline_recap']['BASE360D_test_trajectory']}`",
        "",
        "## 11. Analysis",
        f"- path ratio improved: `{comparisons['vs_seq360b'] in {'better', 'partial'}}`",
        f"- ATE SE3 improved: `{_safe_float(traj_test.get('ate_se3_rmse')) is not None and _safe_float(traj_test.get('ate_se3_rmse')) < 75.94691348103409}`",
        f"- ATE Sim3 improved: `{_safe_float(traj_test.get('ate_sim3_rmse')) is not None and _safe_float(traj_test.get('ate_sim3_rmse')) < 27.564661865900444}`",
        f"- pair tmag over-corrected: `{_safe_float(pair_test.get('tmag_median_ratio')) is not None and _safe_float(pair_test.get('tmag_median_ratio')) > 1.25}`",
        f"- clip composition val/test: `val={clip_val}`, `test={clip_test}`",
        "",
        "## 12. Recommendation",
        f"- `{train_summary['recommendation']}`",
        "",
        "## 13. Compliance checklist",
        "- `training_executed = true`",
        "- `fine_tune_executed = true`",
        "- `learned_weights_saved = true`",
        "- `final360i_checkpoint_modified = false`",
        "- `seq360b_checkpoint_modified = false`",
        "- `explicit_matching_used = false`",
        "- `match_list_output = false`",
        "- `ransac_used = false`",
        "- `pnp_used = false`",
        "- `bundle_adjustment_used = false`",
        "- `hkust_360dvo_teacher_used = false`",
        "- `base360_outputs_used_as_training_input = false`",
        "- `train_manifest_used = true`",
        "- `val_manifest_used_for_selection_only = true`",
        "- `test_manifest_used_for_final_eval_only = true`",
        "- `dset2c_canonical_split_used = true`",
        "- `random_pair_split_used = false`",
        "- `direct_glob_data_360dvo_sequences = false`",
        "- `s5_locked_metrics_modified = false`",
        "- `large_checkpoints_committed_to_git = false`",
    ]
    return "\n".join(lines) + "\n"


def _comparison_summary(
    precheck: Mapping[str, Any],
    pair_test: Mapping[str, Any],
    traj_test: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    classification: str,
) -> str:
    return "\n".join(
        [
            "# SEQ360A vs FINAL360I SEQ360B TRAIN360E BASE360D summary",
            "",
            "| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | pair_path_ratio | trajectory_path_ratio | ATE none | ATE SE3 | ATE Sim3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| FINAL360I pair / TRAIN360E traj | {_fmt(precheck['baseline_recap']['FINAL360I_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(precheck['baseline_recap']['FINAL360I_test_pair'].get('anti_parallel_rate'))} | {_fmt(precheck['baseline_recap']['FINAL360I_test_pair'].get('tmag_median_ratio'))} | {_fmt(precheck['baseline_recap']['FINAL360I_test_pair'].get('path_ratio'))} | {_fmt(precheck['baseline_recap']['TRAIN360E_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(precheck['baseline_recap']['TRAIN360E_test_trajectory'].get('ate_none_rmse'))} | {_fmt(precheck['baseline_recap']['TRAIN360E_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(precheck['baseline_recap']['TRAIN360E_test_trajectory'].get('ate_sim3_rmse'))} |",
            f"| SEQ360B trajectory | N/A | N/A | N/A | N/A | {_fmt(precheck['baseline_recap']['SEQ360B_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(precheck['baseline_recap']['SEQ360B_test_trajectory'].get('ate_none_rmse'))} | {_fmt(precheck['baseline_recap']['SEQ360B_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(precheck['baseline_recap']['SEQ360B_test_trajectory'].get('ate_sim3_rmse'))} |",
            f"| BASE360D trajectory | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('signed_tdir_mean_deg'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('anti_parallel_rate'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('tmag_median_ratio'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('pair_path_ratio'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('ate_none_rmse'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(precheck['baseline_recap']['BASE360D_test_trajectory'].get('ate_sim3_rmse'))} |",
            f"| SEQ360A | {_fmt(pair_test.get('signed_tdir_mean_deg'))} | {_fmt(pair_test.get('anti_parallel_rate'))} | {_fmt(pair_test.get('tmag_median_ratio'))} | {_fmt(pair_test.get('path_ratio'))} | {_fmt(traj_test.get('trajectory_path_ratio'))} | {_fmt(traj_test.get('ate_none_rmse'))} | {_fmt(traj_test.get('ate_se3_rmse'))} | {_fmt(traj_test.get('ate_sim3_rmse'))} |",
            "",
            f"- compared to FINAL360I pair-level: `{comparisons['vs_final360i_pair']}`",
            f"- compared to SEQ360B trajectory: `{comparisons['vs_seq360b']}`",
            f"- compared to TRAIN360E trajectory: `{comparisons['vs_train360e']}`",
            f"- compared to BASE360D trajectory: `{comparisons['vs_base360d']}`",
            f"- classification: `{classification}`",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_yaml_like(Path(args.config))
    _seed(int(cfg["training"]["seed"]))

    precheck = _precheck(cfg)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck)
        print(json.dumps({"status": "blocked", "blockers": precheck["blockers"]}, ensure_ascii=False, indent=2))
        return 1

    device = torch.device(precheck["device"])
    outputs = cfg["outputs"]
    ckpt_dir = REPO_ROOT / outputs["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_clip_ds = _build_clip_dataset(cfg, "train")
    train_clip_subset, subset_info = _subset_clip_dataset(train_clip_ds, cfg["data"].get("train_subset_max"), int(cfg["training"]["seed"]))
    val_clip_ds = _build_clip_dataset(cfg, "val")
    test_clip_ds = _build_clip_dataset(cfg, "test")
    train_loader = DataLoader(
        train_clip_subset,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=bool(cfg["data"]["shuffle_train"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    clip_eval_batch = int(cfg["data"].get("clip_eval_batch_size", 1))
    val_clip_loader = DataLoader(val_clip_ds, batch_size=clip_eval_batch, shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", drop_last=False)
    test_clip_loader = DataLoader(test_clip_ds, batch_size=clip_eval_batch, shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", drop_last=False)

    val_pair_ds = _build_pair_dataset(cfg, "val")
    test_pair_ds = _build_pair_dataset(cfg, "test")
    val_pair_loader = DataLoader(val_pair_ds, batch_size=int(cfg["data"]["eval_batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", drop_last=False)
    test_pair_loader = DataLoader(test_pair_ds, batch_size=int(cfg["data"]["eval_batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", drop_last=False)

    model, ckpt_cfg, load_summary = _load_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    optimizer = _make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"].get("amp", False) and device.type == "cuda"))

    best_score = float("inf")
    best_epoch = -1
    best_path = ckpt_dir / "best_val.pt"
    final_path = ckpt_dir / "final.pt"
    history: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        lr_info = _set_epoch_lrs(optimizer, cfg, epoch)
        running: Dict[str, List[float]] = {
            "total_loss": [],
            "pair_loss": [],
            "comp_rot_loss": [],
            "comp_tdir_loss": [],
            "comp_log_tmag_loss": [],
            "path_loss": [],
            "rot_accum_loss": [],
            "residual_reg": [],
        }
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg["training"].get("amp", False) and device.type == "cuda")):
                pred = _predict_clip_pairs(model, batch, device)
                losses = _sequence_losses(pred, batch, cfg)
                residual_reg = model.residual_regularization(pred["aux"])["loss"]
                total_loss = _loss_total(losses, residual_reg, cfg)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running["total_loss"].append(float(total_loss.detach().cpu()))
            running["pair_loss"].append(float(losses["pair_loss"].detach().cpu()))
            running["comp_rot_loss"].append(float(losses["comp_rot_loss"].detach().cpu()))
            running["comp_tdir_loss"].append(float(losses["comp_tdir_loss"].detach().cpu()))
            running["comp_log_tmag_loss"].append(float(losses["comp_log_tmag_loss"].detach().cpu()))
            running["path_loss"].append(float(losses["path_loss"].detach().cpu()))
            running["rot_accum_loss"].append(float(losses["rot_accum_loss"].detach().cpu()))
            running["residual_reg"].append(float(residual_reg.detach().cpu()))

        pair_val = evaluate_struct360b_pose(
            model,
            val_pair_loader,
            device,
            tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
            max_batches=cfg["evaluation"].get("max_val_batches"),
        )["final_metrics"]
        clip_val = _evaluate_clip_metrics(
            model,
            val_clip_loader,
            device,
            max_batches=cfg["evaluation"].get("max_clip_val_batches"),
        )
        score = _selection_score(pair_val, clip_val, cfg)
        epoch_payload = {
            "epoch": epoch + 1,
            "lrs": lr_info,
            "train": {k: _mean(v) for k, v in running.items()},
            "pair_val": pair_val,
            "clip_val": clip_val,
            "selection_score": score,
        }
        history.append(_jsonable(epoch_payload))
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch + 1,
                ckpt_cfg,
                {
                    "task_name": cfg["task_name"],
                    "run_name": cfg["training"]["run_name"],
                    "best_score": best_score,
                    "history_tail": history[-1],
                    "subset_info": subset_info,
                    "checkpoint_load": load_summary,
                },
            )

    _save_checkpoint(
        final_path,
        model,
        optimizer,
        int(cfg["training"]["epochs"]),
        ckpt_cfg,
        {
            "task_name": cfg["task_name"],
            "run_name": cfg["training"]["run_name"],
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
            "subset_info": subset_info,
            "checkpoint_load": load_summary,
        },
    )

    best_model, _best_cfg, best_load_summary = _load_model(best_path, cfg, device)
    pair_val = evaluate_struct360b_pose(best_model, val_pair_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_val_batches"))["final_metrics"]
    pair_test = evaluate_struct360b_pose(best_model, test_pair_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_test_batches"))["final_metrics"]
    clip_val = _evaluate_clip_metrics(best_model, val_clip_loader, device, max_batches=cfg["evaluation"].get("max_clip_val_batches"))
    clip_test = _evaluate_clip_metrics(best_model, test_clip_loader, device, max_batches=None)
    traj_val_raw, traj_test_raw = _run_trajectory_eval(cfg, best_path, device)

    traj_val = {
        "trajectory_path_ratio": traj_val_raw.get("trajectory_path_ratio"),
        "coverage": traj_val_raw.get("pose_coverage"),
        "pred_path_length": traj_val_raw.get("pred_path_length"),
        "gt_path_length": traj_val_raw.get("gt_path_length"),
        "ate_none_rmse": traj_val_raw.get("ate_none", {}).get("rmse"),
        "ate_se3_rmse": traj_val_raw.get("ate_se3", {}).get("rmse"),
        "ate_sim3_rmse": traj_val_raw.get("ate_sim3", {}).get("rmse"),
        "selected_convention": traj_val_raw.get("selected_convention"),
    }
    traj_test = {
        "trajectory_path_ratio": traj_test_raw.get("trajectory_path_ratio"),
        "coverage": traj_test_raw.get("pose_coverage"),
        "pred_path_length": traj_test_raw.get("pred_path_length"),
        "gt_path_length": traj_test_raw.get("gt_path_length"),
        "ate_none_rmse": traj_test_raw.get("ate_none", {}).get("rmse"),
        "ate_se3_rmse": traj_test_raw.get("ate_se3", {}).get("rmse"),
        "ate_sim3_rmse": traj_test_raw.get("ate_sim3", {}).get("rmse"),
        "selected_convention": traj_test_raw.get("selected_convention"),
    }

    comparisons = {
        "vs_final360i_pair": _compare_pair(pair_test, precheck["baseline_recap"]["FINAL360I_test_pair"]),
        "vs_seq360b": _compare_trajectory(traj_test, precheck["baseline_recap"]["SEQ360B_test_trajectory"]),
        "vs_train360e": _compare_trajectory(traj_test, precheck["baseline_recap"]["TRAIN360E_test_trajectory"]),
        "vs_base360d": _compare_trajectory(traj_test, precheck["baseline_recap"]["BASE360D_test_trajectory"]),
    }
    classification = _classify(
        pair_test,
        traj_test,
        precheck["baseline_recap"]["FINAL360I_test_pair"],
        precheck["baseline_recap"]["SEQ360B_test_trajectory"],
    )
    recommendation = (
        "promote_SEQ360A_as_sequence_enhanced_model"
        if classification in {"better", "balanced_success"}
        else "proceed_to_STRUCT360C_rotation_aware_fine_refinement"
        if classification in {"trajectory_improved_pair_tradeoff", "scale_improved_shape_unchanged"}
        else "keep_FINAL360I_as_pair_main_and_SEQ360A_as_sequence_variant"
        if classification == "no_improvement"
        else "prepare_thesis_experiment_section"
    )

    train_summary = {
        "training_executed": True,
        "checkpoint_saved": True,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "init_checkpoint": str(REPO_ROOT / cfg["inputs"]["init_checkpoint"]),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "subset_info": subset_info,
        "checkpoint_load": load_summary,
        "best_checkpoint_load": best_load_summary,
        "history": history,
        "runtime_seconds": time.time() - start_time,
        "parameter_count": count_parameters(best_model),
        "recommendation": recommendation,
    }

    val_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "pair_metrics": pair_val,
        "clip_metrics": clip_val,
        "selection_score": _selection_score(pair_val, clip_val, cfg),
        "precheck": _jsonable(precheck),
        "training_summary": _jsonable(train_summary),
    }
    test_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "pair_metrics": pair_test,
        "clip_metrics": clip_test,
        "classification": classification,
        "comparisons": comparisons,
        "precheck": _jsonable(precheck),
        "training_summary": _jsonable(train_summary),
    }

    _write_json(REPO_ROOT / outputs["val_metrics_path"], _jsonable(val_payload))
    _write_json(REPO_ROOT / outputs["test_metrics_path"], _jsonable(test_payload))
    _write_json(REPO_ROOT / outputs["trajectory_val_metrics_path"], _jsonable(traj_val_raw))
    _write_json(REPO_ROOT / outputs["trajectory_test_metrics_path"], _jsonable(traj_test_raw))
    (REPO_ROOT / outputs["report_path"]).write_text(
        _report_text(
            cfg,
            precheck,
            train_summary,
            pair_val,
            pair_test,
            clip_val,
            clip_test,
            traj_val,
            traj_test,
            comparisons,
            classification,
        ),
        encoding="utf-8",
    )
    (REPO_ROOT / outputs["comparison_summary_path"]).write_text(
        _comparison_summary(precheck, pair_test, traj_test, comparisons, classification),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "task_name": cfg["task_name"],
                "training_executed": True,
                "best_checkpoint": str(best_path),
                "best_epoch": best_epoch,
                "pair_test_signed_tdir_mean": pair_test.get("signed_tdir_mean_deg"),
                "pair_test_anti_parallel_rate": pair_test.get("anti_parallel_rate"),
                "pair_test_tmag_median_ratio": pair_test.get("tmag_median_ratio"),
                "pair_test_path_ratio": pair_test.get("path_ratio"),
                "trajectory_test_ate_none": traj_test.get("ate_none_rmse"),
                "trajectory_test_ate_se3": traj_test.get("ate_se3_rmse"),
                "trajectory_test_ate_sim3": traj_test.get("ate_sim3_rmse"),
                "trajectory_test_path_ratio": traj_test.get("trajectory_path_ratio"),
                "classification": classification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
