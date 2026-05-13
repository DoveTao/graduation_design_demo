#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_sequence_clip_dataset import Dset2CSequenceClipDataset
from miniyaml import load_yaml_like
from models.seq360b_scale_smoothing_head import Seq360BScaleSmoothingHead, seq360b_scale_losses
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train_struct360b_match_free_coarse_to_fine import _cfg_from_dict, _inject_struct360b_cfg


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seq360b_lightweight_scale_smoothing.yaml"


def _load_pair_model(path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    ckpt_cfg = _cfg_from_dict(_inject_struct360b_cfg(dict(payload.get("cfg", {})), cfg))
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "D": int(ckpt_cfg.D)}


def _loader(cfg: Mapping[str, Any], smoke_batches: int) -> Tuple[DataLoader, Dict[str, Any]]:
    data = cfg["data"]
    ds = Dset2CSequenceClipDataset(
        str(REPO_ROOT / cfg["inputs"]["train_manifest"]),
        expected_split="train",
        clip_len=int(data["clip_len"]),
        image_hw=tuple(int(x) for x in data["image_hw"]),
        max_frame_gap=int(data["max_frame_gap"]),
        max_timestamp_gap_factor=float(data["max_timestamp_gap_factor"]),
        tmag_epsilon=float(data["tmag_epsilon"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )
    subset = Subset(ds, list(range(min(len(ds), max(1, smoke_batches)))))
    return DataLoader(subset, batch_size=int(data["train_batch_size"]), shuffle=False, num_workers=0), ds.get_clip_summary()


def _adjacent_predictions(pair_model: STRUCT360BMatchFreeCoarseToFineModel, batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    images = batch["images"].to(device)
    timestamps = batch["timestamps"].to(device)
    adj_mask = batch["adjacent_mask"][0].bool()
    pair_i = batch["pair_i"][0, adj_mask].to(device)
    pair_j = batch["pair_j"][0, adj_mask].to(device)
    IA = images[0, pair_i]
    IB = images[0, pair_j]
    dt = timestamps[0, pair_j] - timestamps[0, pair_i]
    R, _t, aux = pair_model(IA, IB, dt_world=dt.float())
    return {
        "R": R.unsqueeze(0),
        "tdir": aux["t_dir_out"].unsqueeze(0),
        "log_tmag": torch.log(aux["t_mag"].clamp_min(1.0e-6)).unsqueeze(0),
        "pair_context": aux["coarse_pair_context"].unsqueeze(0),
        "gt_tmag": batch["tmag"][:, adj_mask].to(device),
    }


def run_dry_smoke(cfg: Mapping[str, Any], smoke_batches: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"].get("use_cuda_if_available", True)) else "cpu")
    loader, clip_summary = _loader(cfg, smoke_batches)
    pair_model, load_summary = _load_pair_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    context_dim = 3 * int(load_summary["D"])
    head = Seq360BScaleSmoothingHead(context_dim, hidden_dim=int(cfg["model"]["hidden_dim"]), delta_clamp=float(cfg["model"]["delta_clamp"])).to(device)
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= int(smoke_batches):
                break
            pred = _adjacent_predictions(pair_model, batch, device)
            out = head(pred["pair_context"], pred["log_tmag"])
            losses = seq360b_scale_losses(out["corrected_log_tmag"], pred["gt_tmag"], eps=float(cfg["data"]["tmag_epsilon"]))
            results.append({
                "batch": idx,
                "adjacent_pairs": int(pred["log_tmag"].shape[1]),
                "delta_abs_mean": float(out["delta_log_tmag_abs_mean"].detach().cpu()),
                "losses": {k: float(v.detach().cpu()) for k, v in losses.items()},
                "rotation_modified": False,
                "tdir_modified": False,
            })
    return {
        "variant": "SEQ360B",
        "dry_run": True,
        "training_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_load": load_summary,
        "clip_summary": clip_summary,
        "results": results,
        "smoke_pass": bool(results),
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
        raise SystemExit("SEQ360B preparation script refuses to train; pass --dry-run.")
    result = run_dry_smoke(cfg, args.smoke_batches)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
