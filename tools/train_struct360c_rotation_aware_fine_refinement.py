#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from miniyaml import load_yaml_like
from models.struct360c_rotation_aware_fine_refinement import STRUCT360CRotationAwareFineRefinementModel
from train_struct360b_match_free_coarse_to_fine import _cfg_from_dict, _inject_struct360b_cfg


DEFAULT_CONFIG = REPO_ROOT / "configs" / "struct360c_rotation_aware_fine_refinement.yaml"


def _load_model(path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360CRotationAwareFineRefinementModel, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    ckpt_cfg = _cfg_from_dict(_inject_struct360b_cfg(dict(payload.get("cfg", {})), cfg))
    ckpt_cfg.struct360c_bias_temperature = float(cfg["model"]["struct360c_bias_temperature"])
    ckpt_cfg.struct360c_bias_clamp_abs = float(cfg["model"]["struct360c_bias_clamp_abs"])
    model = STRUCT360CRotationAwareFineRefinementModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def _loader(cfg: Mapping[str, Any], smoke_batches: int) -> DataLoader:
    data = cfg["data"]
    ds = Dset2CCanonicalPairDataset(
        str(REPO_ROOT / cfg["inputs"]["train_manifest"]),
        expected_split="train",
        image_hw=tuple(int(x) for x in data["image_hw"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )
    subset = Subset(ds, list(range(min(len(ds), max(1, smoke_batches)))))
    return DataLoader(subset, batch_size=int(data["eval_batch_size"]), shuffle=False, num_workers=0)


def run_dry_smoke(cfg: Mapping[str, Any], smoke_batches: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"].get("use_cuda_if_available", True)) else "cpu")
    model, load_summary = _load_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    loader = _loader(cfg, smoke_batches)
    outputs = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= int(smoke_batches):
                break
            IA = batch["IA"].to(device)
            IB = batch["IB"].to(device)
            dt = batch["meta"]["dt_world"].to(device=device, dtype=torch.float32).view(-1)
            R, _t, aux = model(IA, IB, dt_world=dt)
            outputs.append({
                "batch": idx,
                "R_shape": list(R.shape),
                "tdir_shape": list(aux["t_dir_out"].shape),
                "tmag_shape": list(aux["t_mag"].shape),
                "bias_mean": float(aux["struct360c_rotation_attention_bias_mean"].detach().cpu()),
                "bias_std": float(aux["struct360c_rotation_attention_bias_std"].detach().cpu()),
                "bias_min": float(aux["struct360c_rotation_attention_bias_min"].detach().cpu()),
                "bias_max": float(aux["struct360c_rotation_attention_bias_max"].detach().cpu()),
                "outputs_correspondences": bool(aux["struct360c_outputs_correspondences"].detach().cpu().item()),
            })
    return {
        "variant": "STRUCT360C",
        "dry_run": True,
        "training_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_load": load_summary,
        "outputs": outputs,
        "smoke_pass": bool(outputs),
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
        raise SystemExit("STRUCT360C preparation script refuses to train; pass --dry-run.")
    result = run_dry_smoke(cfg, args.smoke_batches)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
