#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_sequence_clip_dataset import Dset2CSequenceClipDataset
from miniyaml import load_yaml_like
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from pose_head import matrix_geodesic_distance
from train_struct360b_match_free_coarse_to_fine import _cfg_from_dict, _inject_struct360b_cfg


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seq360a_sequence_consistency_scale_drift.yaml"


def _load_model(checkpoint_path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    ckpt_cfg = _cfg_from_dict(_inject_struct360b_cfg(dict(payload.get("cfg", {})), cfg))
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def _compose_ba(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.matmul(R2, R1), torch.matmul(R2, t1.unsqueeze(-1)).squeeze(-1) + t2


def _tdir_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.float(), dim=-1, eps=1.0e-6)
    b = torch.nn.functional.normalize(b.float(), dim=-1, eps=1.0e-6)
    return (1.0 - (a * b).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def predict_clip_pairs(model: STRUCT360BMatchFreeCoarseToFineModel, batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    pair_i = batch["pair_i"].to(device, non_blocking=True)
    pair_j = batch["pair_j"].to(device, non_blocking=True)
    timestamps = batch["timestamps"].to(device, non_blocking=True)
    B, P = pair_i.shape
    IA: List[torch.Tensor] = []
    IB: List[torch.Tensor] = []
    dt: List[torch.Tensor] = []
    for b in range(B):
        IA.append(images[b, pair_i[b]])
        IB.append(images[b, pair_j[b]])
        dt.append(timestamps[b, pair_j[b]] - timestamps[b, pair_i[b]])
    flat_a = torch.cat(IA, dim=0)
    flat_b = torch.cat(IB, dim=0)
    flat_dt = torch.cat(dt, dim=0).float()
    R, _t_local, aux = model(flat_a, flat_b, dt_world=flat_dt)
    return {
        "R": R.view(B, P, 3, 3),
        "tdir": aux["t_dir_out"].view(B, P, 3),
        "tmag": aux["t_mag"].view(B, P),
        "log_tmag": torch.log(aux["t_mag"].view(B, P).clamp_min(1.0e-6)),
        "aux": aux,
    }


def sequence_composition_losses(pred: Mapping[str, torch.Tensor], batch: Mapping[str, Any], *, eps: float = 1.0e-6) -> Dict[str, torch.Tensor]:
    R_gt = batch["R_BA"].to(pred["R"].device).float()
    t_gt = batch["t_BA_B"].to(pred["R"].device).float()
    pair_i = batch["pair_i"].to(pred["R"].device)
    pair_j = batch["pair_j"].to(pred["R"].device)
    B, P = pair_i.shape
    rot_terms: List[torch.Tensor] = []
    tdir_terms: List[torch.Tensor] = []
    log_terms: List[torch.Tensor] = []
    path_terms: List[torch.Tensor] = []
    adjacent_terms: List[torch.Tensor] = []
    for b in range(B):
        lookup = {(int(pair_i[b, p]), int(pair_j[b, p])): p for p in range(P)}
        clip_len = int(batch["images"].shape[1])
        adj = [lookup[(i, i + 1)] for i in range(clip_len - 1)]
        adjacent_terms.append(
            matrix_geodesic_distance(pred["R"][b, adj], R_gt[b, adj]).mean()
            + _tdir_loss(pred["tdir"][b, adj], t_gt[b, adj])
            + torch.nn.functional.smooth_l1_loss(pred["log_tmag"][b, adj], torch.log(torch.linalg.norm(t_gt[b, adj], dim=-1).clamp_min(eps)))
        )
        pred_path = pred["tmag"][b, adj].sum()
        gt_path = torch.linalg.norm(t_gt[b, adj], dim=-1).sum()
        path_terms.append(torch.abs(torch.log((pred_path + eps) / (gt_path + eps))))
        for end in range(2, clip_len):
            R_comp = pred["R"][b, adj[0]]
            t_comp = pred["tdir"][b, adj[0]] * pred["tmag"][b, adj[0]]
            for step in range(1, end):
                p = adj[step]
                R_comp, t_comp = _compose_ba(R_comp, t_comp, pred["R"][b, p], pred["tdir"][b, p] * pred["tmag"][b, p])
            target = lookup[(0, end)]
            rot_terms.append(matrix_geodesic_distance(R_comp.unsqueeze(0), R_gt[b, target].unsqueeze(0)).mean())
            tdir_terms.append(_tdir_loss(t_comp.unsqueeze(0), t_gt[b, target].unsqueeze(0)))
            log_terms.append(torch.abs(torch.log(torch.linalg.norm(t_comp).clamp_min(eps)) - torch.log(torch.linalg.norm(t_gt[b, target]).clamp_min(eps))))
    zero = pred["R"].sum() * 0.0
    return {
        "adjacent_pair_loss": torch.stack(adjacent_terms).mean() if adjacent_terms else zero,
        "kstep_comp_rot_loss": torch.stack(rot_terms).mean() if rot_terms else zero,
        "kstep_comp_tdir_loss": torch.stack(tdir_terms).mean() if tdir_terms else zero,
        "kstep_comp_log_tmag_loss": torch.stack(log_terms).mean() if log_terms else zero,
        "scale_path_ratio_loss": torch.stack(path_terms).mean() if path_terms else zero,
    }


def _build_loader(cfg: Mapping[str, Any], split: str, smoke_batches: int) -> Tuple[DataLoader, Dict[str, Any]]:
    data = cfg["data"]
    manifest = REPO_ROOT / cfg["inputs"][f"{split}_manifest"]
    ds = Dset2CSequenceClipDataset(
        str(manifest),
        expected_split=split,
        clip_len=int(data["clip_len"]),
        image_hw=tuple(int(x) for x in data["image_hw"]),
        max_frame_gap=int(data["max_frame_gap"]),
        max_timestamp_gap_factor=float(data["max_timestamp_gap_factor"]),
        tmag_epsilon=float(data["tmag_epsilon"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )
    max_items = max(1, int(smoke_batches)) * int(data.get("train_batch_size", 1))
    subset = Subset(ds, list(range(min(len(ds), max_items))))
    loader = DataLoader(subset, batch_size=int(data.get("train_batch_size", 1)), shuffle=False, num_workers=0)
    return loader, ds.get_clip_summary()


def run_dry_smoke(cfg: Mapping[str, Any], smoke_batches: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"].get("use_cuda_if_available", True)) else "cpu")
    loader, clip_summary = _build_loader(cfg, "train", smoke_batches)
    model, load_summary = _load_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    losses: List[Dict[str, float]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= int(smoke_batches):
                break
            pred = predict_clip_pairs(model, batch, device)
            loss = sequence_composition_losses(pred, batch, eps=float(cfg["data"]["tmag_epsilon"]))
            losses.append({k: float(v.detach().cpu()) for k, v in loss.items()})
    return {
        "variant": "SEQ360A",
        "dry_run": True,
        "training_executed": False,
        "optimizer_step_executed": False,
        "smoke_batches": int(smoke_batches),
        "device": str(device),
        "clip_summary": clip_summary,
        "checkpoint_load": load_summary,
        "losses": losses,
        "smoke_pass": bool(losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_yaml_like(Path(args.config))
    if not args.dry_run and not args.no_train and not bool(cfg.get("dry_run", True)):
        raise SystemExit("SEQ360A preparation script refuses to train; pass --dry-run for smoke validation.")
    result = run_dry_smoke(cfg, args.smoke_batches)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
