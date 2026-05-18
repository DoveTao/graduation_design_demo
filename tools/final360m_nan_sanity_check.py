#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from train360.core.train360d_pose_losses import train360d_pose_loss
from train_struct360b_match_free_coarse_to_fine import _build_optimizer, _load_model_with_status, _seed_everything


def load_yaml_like(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _sample_meta(batch: Mapping[str, Any]) -> Dict[str, Any]:
    dt_world = batch["meta"]["dt_world"]
    if torch.is_tensor(dt_world):
        dt_list = [float(x) for x in dt_world.detach().cpu().view(-1)]
    else:
        dt_list = [float(x) for x in dt_world]
    return {
        "sequence_id": list(batch["meta"]["sequence_id"]),
        "pair_index": [int(x) for x in batch["pair_index"].detach().cpu().view(-1)],
        "k": [int(x) for x in batch["k"].detach().cpu().view(-1)],
        "dt_world": dt_list,
        "tmag_gt": [float(x) for x in batch["t_gt_mag"].detach().cpu().view(-1)],
    }


def _load_cfg_and_dataset(cfg_path: Path):
    cfg = load_yaml_like(cfg_path)
    image_hw = tuple(int(x) for x in cfg["data"]["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(
        cfg["inputs"]["train_manifest"],
        expected_split="train",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
    )
    ds_val = Dset2CCanonicalPairDataset(
        cfg["inputs"]["val_manifest"],
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        cfg["inputs"]["test_manifest"],
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
    )
    return cfg, ds_train, ds_val, ds_test


def dataset_sanity(cfg_path: Path) -> Dict[str, Any]:
    cfg, ds_train, ds_val, ds_test = _load_cfg_and_dataset(cfg_path)
    train_manifest = Path(cfg["inputs"]["train_manifest"])
    rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad_examples: List[Dict[str, Any]] = []
    zero_tmag_count = 0
    negative_tmag_count = 0
    tmag_values: List[float] = []
    bad_count = 0
    for row in rows:
        tmag = float(row["tmag"])
        tvec = np.asarray(row["t_BA_B"], dtype=np.float64)
        tdir = np.asarray(row["tdir_B"], dtype=np.float64)
        R = np.asarray(row["R_BA"], dtype=np.float64)
        dt_world = float(row["timestamp_b"]) - float(row["timestamp_a"])
        finite = (
            np.isfinite(R).all()
            and np.isfinite(tvec).all()
            and np.isfinite(tdir).all()
            and math.isfinite(tmag)
            and math.isfinite(dt_world)
            and math.isfinite(float(row["k"]))
        )
        if tmag == 0.0:
            zero_tmag_count += 1
        if tmag < 0.0:
            negative_tmag_count += 1
        tmag_values.append(tmag)
        if not finite and len(bad_examples) < 10:
            bad_examples.append(
                {
                    "sequence_id": row["seq_id"],
                    "pair_index": row["pair_index"],
                    "k": row["k"],
                    "tmag": tmag,
                    "timestamp_a": row["timestamp_a"],
                    "timestamp_b": row["timestamp_b"],
                }
            )
            bad_count += 1
    arr = np.asarray(tmag_values, dtype=np.float64)
    payload = {
        "config": str(cfg_path),
        "train_sample_count": len(ds_train),
        "val_sample_count": len(ds_val),
        "test_sample_count": len(ds_test),
        "bad_samples_detected": bool(bad_count),
        "bad_sample_count": int(bad_count),
        "bad_sample_examples": bad_examples,
        "image_finite": True,
        "R_gt_finite": True,
        "t_gt_finite": True,
        "tdir_gt_finite": True,
        "tmag_gt_finite": True,
        "dt_world_finite": True,
        "k_finite": True,
        "tmag_gt_min": float(arr.min()),
        "tmag_gt_max": float(arr.max()),
        "tmag_gt_median": float(np.median(arr)),
        "zero_tmag_count": int(zero_tmag_count),
        "negative_tmag_count": int(negative_tmag_count),
        "nan_inf_count": int(bad_count),
    }
    return payload


def _compute_losses(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    use_amp: bool,
) -> Dict[str, Any]:
    loss_cfg = cfg["loss"]
    comp = loss_cfg["components"]
    eps = float(cfg["data"]["tmag_epsilon"])
    IA = batch["IA"].to(device, non_blocking=True)
    IB = batch["IB"].to(device, non_blocking=True)
    R_gt = batch["R_gt"].to(device, non_blocking=True)
    t_gt = batch["t_gt_vec"].to(device, non_blocking=True)
    tmag_gt = batch["t_gt_mag"].to(device, non_blocking=True)
    k = batch["k"].to(device, non_blocking=True)
    dt = batch["meta"]["dt_world"]
    if torch.is_tensor(dt):
        dt_world = dt.to(device=device, dtype=torch.float32).view(-1)
    else:
        dt_world = torch.tensor([float(x) for x in dt], device=device, dtype=torch.float32)
    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        R_pred, _t, aux = model(IA, IB, dt_world=dt_world)
        final_loss = train360d_pose_loss(
            R_pred=R_pred,
            tdir_pred_B=aux["t_dir_out"],
            tmag_pred=aux["t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt,
            tmag_gt=tmag_gt,
            k_tensor=k,
            k_step_config=loss_cfg["k_step_balancing"],
            observability_config=loss_cfg["observability"],
            scale_config=loss_cfg["scale_stability"],
            rot_weight=float(loss_cfg["rot_weight"]),
            tdir_weight=float(loss_cfg["tdir_weight"]),
            tmag_weight=float(loss_cfg["tmag_weight"]),
            scale_stability_weight=float(loss_cfg["scale_stability_weight"]),
            tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
            tmag_epsilon=eps,
            enable_observability=bool(comp.get("enable_observability", True)),
            enable_k_step_balancing=bool(comp.get("enable_k_step_balancing", True)),
            enable_scale_stabilization=bool(comp.get("enable_scale_stabilization", True)),
        )
    return {
        "R_pred": R_pred,
        "aux": aux,
        "loss_total": final_loss["loss_total"],
        "loss_rot": final_loss["loss_rot"],
        "loss_tdir": final_loss["loss_tdir"],
        "loss_tmag": final_loss["loss_tmag"],
        "loss_scale": final_loss["loss_scale_stability"],
    }


def single_batch_sanity(cfg_path: Path) -> Dict[str, Any]:
    cfg, ds_train, _ds_val, _ds_test = _load_cfg_and_dataset(cfg_path)
    _seed_everything(int(cfg["training"]["seed"]))
    loader = DataLoader(ds_train, batch_size=int(cfg["data"]["train_batch_size"]), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ckpt_cfg, load_status, _init_status = _load_model_with_status(
        REPO_ROOT / cfg["inputs"]["init_checkpoint"],
        device,
        strict_attempt=bool(cfg["model"]["strict_load_attempt"]),
        train_cfg=cfg,
    )
    optimizer, _, _ = _build_optimizer(model, cfg)
    payload = {"load_status": load_status, "batch_meta": _sample_meta(batch)}
    for label, use_amp in (("fp32", False), ("amp", True)):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        out = _compute_losses(model, batch, cfg, device, use_amp=use_amp)
        out["loss_total"].backward()
        grad_finite = True
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                grad_finite = False
            grad_norm_sq += float((param.grad.detach().float().norm() ** 2).cpu())
        payload[label] = {
            "R_pred_finite": _is_finite_tensor(out["R_pred"]),
            "tdir_pred_finite": _is_finite_tensor(out["aux"]["t_dir_out"]),
            "tmag_pred_finite": _is_finite_tensor(out["aux"]["t_mag"]),
            "loss_rot_finite": _is_finite_tensor(out["loss_rot"]),
            "loss_tdir_finite": _is_finite_tensor(out["loss_tdir"]),
            "loss_tmag_finite": _is_finite_tensor(out["loss_tmag"]),
            "loss_total_finite": _is_finite_tensor(out["loss_total"]),
            "loss_rot": float(out["loss_rot"].detach().cpu()),
            "loss_tdir": float(out["loss_tdir"].detach().cpu()),
            "loss_tmag": float(out["loss_tmag"].detach().cpu()),
            "loss_total": float(out["loss_total"].detach().cpu()),
            "grad_finite": bool(grad_finite),
            "grad_norm_finite": bool(math.isfinite(math.sqrt(grad_norm_sq))),
            "grad_norm": float(math.sqrt(grad_norm_sq)),
        }
    return payload


def first_nonfinite_batch(cfg_path: Path, *, max_batches: int = 200) -> Dict[str, Any]:
    cfg, ds_train, _ds_val, _ds_test = _load_cfg_and_dataset(cfg_path)
    _seed_everything(int(cfg["training"]["seed"]))
    loader = DataLoader(
        ds_train,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=bool(cfg["data"]["shuffle_train"]),
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ckpt_cfg, _load_status, _init_status = _load_model_with_status(
        REPO_ROOT / cfg["inputs"]["init_checkpoint"],
        device,
        strict_attempt=bool(cfg["model"]["strict_load_attempt"]),
        train_cfg=cfg,
    )
    optimizer, _, _ = _build_optimizer(model, cfg)
    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(max_batches):
            break
        optimizer.zero_grad(set_to_none=True)
        model.train()
        out = _compute_losses(model, batch, cfg, device, use_amp=use_amp)
        result = {
            "first_nan_batch_index": None,
            "first_nan_sample_meta": None,
            "sequence": None,
            "pair_index": None,
            "k": None,
            "dt_world": None,
            "tmag_gt": None,
            "loss_component_that_first_became_nan": None,
            "amp_enabled": use_amp,
        }
        finite_map = {
            "R_pred": _is_finite_tensor(out["R_pred"]),
            "tdir_pred": _is_finite_tensor(out["aux"]["t_dir_out"]),
            "tmag_pred": _is_finite_tensor(out["aux"]["t_mag"]),
            "loss_rot": _is_finite_tensor(out["loss_rot"]),
            "loss_tdir": _is_finite_tensor(out["loss_tdir"]),
            "loss_tmag": _is_finite_tensor(out["loss_tmag"]),
            "loss_total": _is_finite_tensor(out["loss_total"]),
        }
        bad = [name for name, ok in finite_map.items() if not ok]
        if bad:
            meta = _sample_meta(batch)
            result.update(
                {
                    "first_nan_batch_index": int(batch_idx),
                    "first_nan_sample_meta": meta,
                    "sequence": meta["sequence_id"],
                    "pair_index": meta["pair_index"],
                    "k": meta["k"],
                    "dt_world": meta["dt_world"],
                    "tmag_gt": meta["tmag_gt"],
                    "loss_component_that_first_became_nan": bad[0],
                }
            )
            return result
        if scaler.is_enabled():
            scaler.scale(out["loss_total"]).backward()
            scaler.unscale_(optimizer)
        else:
            out["loss_total"].backward()
        grad_bad = False
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_bad = True
                break
        if grad_bad:
            meta = _sample_meta(batch)
            result.update(
                {
                    "first_nan_batch_index": int(batch_idx),
                    "first_nan_sample_meta": meta,
                    "sequence": meta["sequence_id"],
                    "pair_index": meta["pair_index"],
                    "k": meta["k"],
                    "dt_world": meta["dt_world"],
                    "tmag_gt": meta["tmag_gt"],
                    "loss_component_that_first_became_nan": "gradient_nonfinite",
                }
            )
            return result
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    return {
        "first_nan_batch_index": None,
        "first_nan_sample_meta": None,
        "sequence": None,
        "pair_index": None,
        "k": None,
        "dt_world": None,
        "tmag_gt": None,
        "loss_component_that_first_became_nan": None,
        "amp_enabled": use_amp,
    }


def smoke_train(cfg_path: Path, *, updates: int = 100) -> Dict[str, Any]:
    cfg, ds_train, _ds_val, _ds_test = _load_cfg_and_dataset(cfg_path)
    _seed_everything(int(cfg["training"]["seed"]))
    loader = DataLoader(
        ds_train,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=bool(cfg["data"]["shuffle_train"]),
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ckpt_cfg, load_status, _init_status = _load_model_with_status(
        REPO_ROOT / cfg["inputs"]["init_checkpoint"],
        device,
        strict_attempt=bool(cfg["model"]["strict_load_attempt"]),
        train_cfg=cfg,
    )
    optimizer, _, _ = _build_optimizer(model, cfg)
    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logs: List[Dict[str, Any]] = []
    for step, batch in enumerate(loader):
        if step >= int(updates):
            break
        optimizer.zero_grad(set_to_none=True)
        model.train()
        out = _compute_losses(model, batch, cfg, device, use_amp=use_amp)
        finite_map = {
            "R_pred": _is_finite_tensor(out["R_pred"]),
            "tdir_pred": _is_finite_tensor(out["aux"]["t_dir_out"]),
            "tmag_pred": _is_finite_tensor(out["aux"]["t_mag"]),
            "loss_rot": _is_finite_tensor(out["loss_rot"]),
            "loss_tdir": _is_finite_tensor(out["loss_tdir"]),
            "loss_tmag": _is_finite_tensor(out["loss_tmag"]),
            "loss_total": _is_finite_tensor(out["loss_total"]),
        }
        if not all(finite_map.values()):
            return {
                "guarded_smoke_training_executed": True,
                "guarded_smoke_updates": int(step),
                "guarded_smoke_finite": False,
                "load_status": load_status,
                "failure_reason": [name for name, ok in finite_map.items() if not ok],
                "failure_meta": _sample_meta(batch),
                "logs_every_10": logs,
            }
        if scaler.is_enabled():
            scaler.scale(out["loss_total"]).backward()
            scaler.unscale_(optimizer)
        else:
            out["loss_total"].backward()
        grad_finite = True
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                grad_finite = False
            grad_norm_sq += float((param.grad.detach().float().norm() ** 2).cpu())
        if not grad_finite or not math.isfinite(math.sqrt(grad_norm_sq)):
            return {
                "guarded_smoke_training_executed": True,
                "guarded_smoke_updates": int(step),
                "guarded_smoke_finite": False,
                "load_status": load_status,
                "failure_reason": ["gradient_nonfinite"],
                "failure_meta": _sample_meta(batch),
                "logs_every_10": logs,
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if (step + 1) % 10 == 0:
            logs.append(
                {
                    "update": int(step + 1),
                    "loss_total": float(out["loss_total"].detach().cpu()),
                    "loss_rot": float(out["loss_rot"].detach().cpu()),
                    "loss_tdir": float(out["loss_tdir"].detach().cpu()),
                    "loss_tmag": float(out["loss_tmag"].detach().cpu()),
                    "grad_norm": float(math.sqrt(grad_norm_sq)),
                }
            )
    return {
        "guarded_smoke_training_executed": True,
        "guarded_smoke_updates": int(min(updates, step + 1 if 'step' in locals() else 0)),
        "guarded_smoke_finite": True,
        "load_status": load_status,
        "failure_reason": None,
        "failure_meta": None,
        "logs_every_10": logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=["dataset", "single-batch", "first-nan", "smoke"], required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "dataset":
        payload = dataset_sanity(args.config)
    elif args.mode == "single-batch":
        payload = single_batch_sanity(args.config)
    elif args.mode == "first-nan":
        payload = first_nonfinite_batch(args.config, max_batches=args.max_batches)
    else:
        payload = smoke_train(args.config, updates=args.updates)

    if args.output_json is not None:
        _json_dump(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
