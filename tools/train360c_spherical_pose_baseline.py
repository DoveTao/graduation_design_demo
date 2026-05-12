#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from miniyaml import load_yaml_like
from model import PanoramaRelPoseModel
from train360_pose_losses import build_k_step_weights, train360_pose_loss
from tools.eval_train360_pose import evaluate_train360_pose


T57B_EXTERNAL_REFERENCE = {
    "signed_tdir_mean_deg": 111.96493221327962,
    "anti_parallel_rate": 0.6746724890829694,
    "tmag_median_ratio": 0.1751560082454769,
    "path_ratio": 0.14878731297064046,
    "rot_mean_deg": 2.12066772555611,
    "split": "val+test external ref",
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(cmd: Sequence[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def _system_info(device: torch.device) -> Dict[str, Any]:
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        return {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(idx),
            "cuda_available": True,
        }
    return {"device": str(device), "gpu_name": None, "cuda_available": False}


def _load_model_with_status(
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict_attempt: bool = True,
    cfg_overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[PanoramaRelPoseModel, Config, Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg type in checkpoint: {type(cfg_dict)}")
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            cfg_dict[key] = value
    cfg = _cfg_from_dict(cfg_dict)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no model state_dict: {checkpoint_path}")

    strict_ok = False
    strict_error = None
    strict_missing: List[str] = []
    strict_unexpected: List[str] = []
    if strict_attempt:
        try:
            missing, unexpected = model.load_state_dict(state, strict=True)
            strict_missing = list(missing)
            strict_unexpected = list(unexpected)
            strict_ok = True
        except Exception as exc:
            strict_error = f"{type(exc).__name__}: {exc}"

    if strict_ok:
        load_status = {
            "status": "strict",
            "strict_attempted": True,
            "strict_ok": True,
            "missing_keys": strict_missing,
            "unexpected_keys": strict_unexpected,
            "strict_error": None,
            "skipped_shape_mismatch": [],
        }
        new_init = {
            "newly_initialized_params": [],
            "checkpoint_loaded_params": list(state.keys()),
        }
    else:
        model = PanoramaRelPoseModel(cfg, device).to(device)
        model_state = model.state_dict()
        filtered = {}
        skipped_shape_mismatch: List[str] = []
        for key, value in state.items():
            if key in model_state and hasattr(value, "shape") and model_state[key].shape != value.shape:
                skipped_shape_mismatch.append(key)
                continue
            filtered[key] = value
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        load_status = {
            "status": "non-strict",
            "strict_attempted": bool(strict_attempt),
            "strict_ok": False,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "strict_error": strict_error,
            "skipped_shape_mismatch": skipped_shape_mismatch,
        }
        new_init = {
            "newly_initialized_params": list(missing),
            "checkpoint_loaded_params": sorted(filtered.keys()),
        }
    return model, cfg, load_status, new_init


def _subset_dataset(dataset: Dset2CCanonicalPairDataset, max_count: Optional[int], seed: int) -> Tuple[torch.utils.data.Dataset, Dict[str, Any]]:
    if max_count is None or int(max_count) <= 0 or len(dataset) <= int(max_count):
        return dataset, {"subset_used": False, "subset_count": len(dataset), "original_count": len(dataset)}
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(dataset))[: int(max_count)]
    idx = sorted(int(x) for x in idx.tolist())
    return Subset(dataset, idx), {
        "subset_used": True,
        "subset_count": len(idx),
        "original_count": len(dataset),
        "subset_seed": int(seed),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    ckpt_cfg: Config,
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "cfg": dict(ckpt_cfg.__dict__),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_prechecks(cfg: Dict[str, Any]) -> Dict[str, Any]:
    inputs = cfg["inputs"]
    train360b_json = REPO_ROOT / inputs["train360b_sanity_json"]
    sanity = _read_json(train360b_json)
    if not sanity:
        raise RuntimeError(f"Missing TRAIN360B sanity json: {train360b_json}")
    if not sanity.get("readiness_assessment", {}).get("recommend_enter_train360_spherical_pose_baseline", False):
        raise RuntimeError("TRAIN360B sanity did not recommend entering TRAIN360 baseline.")

    manifests = {
        "train": REPO_ROOT / inputs["train_manifest"],
        "val": REPO_ROOT / inputs["val_manifest"],
        "test": REPO_ROOT / inputs["test_manifest"],
    }
    for path in manifests.values():
        if not path.is_file():
            raise FileNotFoundError(f"Manifest missing: {path}")
    hygiene_path = REPO_ROOT / inputs["hygiene_json"]
    if not hygiene_path.is_file():
        raise FileNotFoundError(f"Hygiene json missing: {hygiene_path}")

    image_hw = tuple(int(x) for x in cfg["data"]["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(str(manifests["train"]), expected_split="train", image_hw=image_hw)
    ds_val = Dset2CCanonicalPairDataset(str(manifests["val"]), expected_split="val", image_hw=image_hw)
    ds_test = Dset2CCanonicalPairDataset(str(manifests["test"]), expected_split="test", image_hw=image_hw)
    split_audit = summarize_manifest_group([ds_train, ds_val, ds_test])
    if split_audit["has_overlap"]:
        raise RuntimeError(f"Sequence overlap detected: {split_audit['sequence_overlap']}")

    git_status = _git(["git", "status", "--short"])
    branch = _git(["git", "branch", "--show-current"])
    return {
        "git_status_short": git_status.splitlines(),
        "branch": branch,
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "dataset_summaries": {
            "train": ds_train.get_schema_summary(),
            "val": ds_val.get_schema_summary(),
            "test": ds_test.get_schema_summary(),
        },
        "split_audit": split_audit,
        "train360b_forward_sanity_pass": True,
        "train360b_sanity_json": sanity,
    }


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "train360_v0_baseline.yaml")
    cfg = load_yaml_like(cfg_path)
    task_name = str(cfg["task_name"])
    start_time = time.time()

    try:
        precheck = _run_prechecks(cfg)
    except Exception as exc:
        blocker_report = {
            "task_name": task_name,
            "training_executed": False,
            "blocker": f"{type(exc).__name__}: {exc}",
        }
        out_path = REPO_ROOT / cfg["outputs"]["report_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "# TRAIN360C blocked before training\n\n"
            f"- blocker: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        _json_dump(REPO_ROOT / cfg["outputs"]["val_metrics_path"], blocker_report)
        _json_dump(REPO_ROOT / cfg["outputs"]["test_metrics_path"], blocker_report)
        raise

    _seed_everything(int(cfg["training"]["seed"]))

    use_cuda = bool(cfg["model"]["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    system_info = _system_info(device)

    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    ds_train: Dset2CCanonicalPairDataset = precheck["datasets"]["train"]
    ds_val: Dset2CCanonicalPairDataset = precheck["datasets"]["val"]
    ds_test: Dset2CCanonicalPairDataset = precheck["datasets"]["test"]

    train_subset, subset_info = _subset_dataset(
        ds_train,
        max_count=data_cfg.get("train_subset_max"),
        seed=int(train_cfg["seed"]),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=int(data_cfg["train_batch_size"]),
        shuffle=bool(data_cfg["shuffle_train"]),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(data_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=int(data_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )

    init_ckpt_path = REPO_ROOT / inputs["init_checkpoint"]
    model_hw = tuple(int(x) for x in data_cfg["image_hw"])
    model, ckpt_cfg, load_status, init_status = _load_model_with_status(
        init_ckpt_path,
        device,
        strict_attempt=bool(model_cfg["strict_load_attempt"]),
        cfg_overrides={
            "H": int(model_hw[0]),
            "W": int(model_hw[1]),
            "use_fine_stage": bool(model_cfg["enable_fine_stage"]),
            "use_coupled_pose_residual_head": bool(model_cfg["enable_coupled_pose_head"]),
            "use_depth_branch": False,
        },
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if str(train_cfg["scheduler"]).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(train_cfg["epochs"]), 1),
            eta_min=float(train_cfg["min_lr"]),
        )
    else:
        scheduler = None

    scaler = None
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    checkpoint_dir = REPO_ROOT / outputs["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = checkpoint_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    best_val_score = float("inf")
    best_epoch = -1
    best_val_metrics: Optional[Dict[str, Any]] = None
    best_checkpoint_path = checkpoint_dir / "best_val.pt"

    metadata_common = {
        "task_name": task_name,
        "git_branch": precheck["branch"],
        "git_status_short": precheck["git_status_short"],
        "config_path": str(cfg_path),
        "train_manifest": str(REPO_ROOT / inputs["train_manifest"]),
        "val_manifest": str(REPO_ROOT / inputs["val_manifest"]),
        "test_manifest": str(REPO_ROOT / inputs["test_manifest"]),
        "train_sequences": precheck["dataset_summaries"]["train"]["sequence_ids"],
        "val_sequences": precheck["dataset_summaries"]["val"]["sequence_ids"],
        "test_sequences": precheck["dataset_summaries"]["test"]["sequence_ids"],
        "no_legacy_dependency_flags": {
            "legacy_scene01_artifact_dependency": False,
            "direct_glob_data_360dvo_sequences": False,
            "uses_orbslam3_teacher": False,
            "uses_hkust_360dvo_teacher": False,
            "uses_eval_gt_for_training": False,
            "uses_test_gt_for_training": False,
        },
    }

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_losses = {"loss_total": [], "loss_rot": [], "loss_tdir": [], "loss_tmag": []}
        nan_inf_count = 0

        for batch in train_loader:
            IA = batch["IA"].to(device, non_blocking=True)
            IB = batch["IB"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)
            t_gt_vec = batch["t_gt_vec"].to(device, non_blocking=True)
            tmag_gt = batch["t_gt_mag"].to(device, non_blocking=True)
            k_tensor = batch["k"].to(device, non_blocking=True)
            dt_world = batch["meta"]["dt_world"]
            if torch.is_tensor(dt_world):
                dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
            else:
                dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    R_pred, _t_pred_local, aux = model(
                        IA,
                        IB,
                        enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
                        dt_world=dt_world_t,
                    )
                    tdir_pred_B = aux.get("t_dir_out", aux["t_dir"])
                    tmag_pred = aux.get("t_mag", aux.get("final_tmag"))
                    if tmag_pred is None:
                        log_tmag = aux.get("log_t_mag", aux.get("final_log_tmag"))
                        tmag_pred = torch.exp(log_tmag.float())
                    k_weights = build_k_step_weights(
                        k_tensor,
                        weight_k1=float(loss_cfg["k_weight_k1"]),
                        weight_k2=float(loss_cfg["k_weight_k2"]),
                        weight_k3=float(loss_cfg["k_weight_k3"]),
                        weight_k5=float(loss_cfg["k_weight_k5"]),
                    ).to(device)
                    loss_dict = train360_pose_loss(
                        R_pred=R_pred,
                        tdir_pred_B=tdir_pred_B,
                        tmag_pred=tmag_pred,
                        R_gt=R_gt,
                        t_gt_vec_B=t_gt_vec,
                        tmag_gt=tmag_gt,
                        k_weights=k_weights,
                        rot_weight=float(loss_cfg["rot_weight"]),
                        tdir_weight=float(loss_cfg["tdir_weight"]),
                        tmag_weight=float(loss_cfg["tmag_weight"]),
                        tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
                        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
                    )
                scaler.scale(loss_dict["loss_total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                R_pred, _t_pred_local, aux = model(
                    IA,
                    IB,
                    enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
                    dt_world=dt_world_t,
                )
                tdir_pred_B = aux.get("t_dir_out", aux["t_dir"])
                tmag_pred = aux.get("t_mag", aux.get("final_tmag"))
                if tmag_pred is None:
                    log_tmag = aux.get("log_t_mag", aux.get("final_log_tmag"))
                    tmag_pred = torch.exp(log_tmag.float())
                k_weights = build_k_step_weights(
                    k_tensor,
                    weight_k1=float(loss_cfg["k_weight_k1"]),
                    weight_k2=float(loss_cfg["k_weight_k2"]),
                    weight_k3=float(loss_cfg["k_weight_k3"]),
                    weight_k5=float(loss_cfg["k_weight_k5"]),
                ).to(device)
                loss_dict = train360_pose_loss(
                    R_pred=R_pred,
                    tdir_pred_B=tdir_pred_B,
                    tmag_pred=tmag_pred,
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_weights=k_weights,
                    rot_weight=float(loss_cfg["rot_weight"]),
                    tdir_weight=float(loss_cfg["tdir_weight"]),
                    tmag_weight=float(loss_cfg["tmag_weight"]),
                    tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
                    tmag_epsilon=float(data_cfg["tmag_epsilon"]),
                )
                loss_dict["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(float(loss_dict[key].detach().cpu()))
            nan_inf_count += int(
                sum((~torch.isfinite(v)).sum().item() for v in [R_pred, tdir_pred_B, tmag_pred, loss_dict["loss_total"]])
            )

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate_train360_pose(
            model,
            val_loader,
            device,
            enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
            tmag_epsilon=float(data_cfg["tmag_epsilon"]),
            max_batches=eval_cfg.get("max_val_batches"),
        )

        epoch_log = {
            "epoch": int(epoch),
            "train_loss_total": float(np.mean(epoch_losses["loss_total"])) if epoch_losses["loss_total"] else None,
            "train_loss_rot": float(np.mean(epoch_losses["loss_rot"])) if epoch_losses["loss_rot"] else None,
            "train_loss_tdir": float(np.mean(epoch_losses["loss_tdir"])) if epoch_losses["loss_tdir"] else None,
            "train_loss_tmag": float(np.mean(epoch_losses["loss_tmag"])) if epoch_losses["loss_tmag"] else None,
            "train_nan_inf_count": int(nan_inf_count),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_metrics": val_metrics,
        }
        _append_jsonl(train_log_path, epoch_log)

        val_score = float(val_metrics["signed_tdir_mean_deg"]) if val_metrics["signed_tdir_mean_deg"] is not None else float("inf")
        if val_score < best_val_score and int(val_metrics["nan_inf_count"]) == 0:
            best_val_score = val_score
            best_epoch = int(epoch)
            best_val_metrics = dict(val_metrics)
            ckpt_metadata = {
                **metadata_common,
                "epoch": int(epoch),
                "val_metric": val_score,
                "selection_metric": str(train_cfg["val_selection_metric"]),
                "checkpoint_role": "best_val",
                "checkpoint_load_status": load_status,
                "subset_info": subset_info,
            }
            _save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, ckpt_cfg, ckpt_metadata)

    final_checkpoint_path = checkpoint_dir / "final.pt"
    final_metadata = {
        **metadata_common,
        "epoch": int(train_cfg["epochs"]),
        "val_metric": best_val_score,
        "selection_metric": str(train_cfg["val_selection_metric"]),
        "checkpoint_role": "final",
        "checkpoint_load_status": load_status,
        "subset_info": subset_info,
    }
    _save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, int(train_cfg["epochs"]), ckpt_cfg, final_metadata)

    selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
    selected_payload = torch.load(str(selected_checkpoint_path), map_location=device)
    selected_model = PanoramaRelPoseModel(ckpt_cfg, device).to(device)
    selected_model.load_state_dict(selected_payload["model"], strict=False)
    selected_model.eval()

    val_metrics = evaluate_train360_pose(
        selected_model,
        val_loader,
        device,
        enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
        max_batches=eval_cfg.get("max_val_batches"),
    )
    test_metrics = evaluate_train360_pose(
        selected_model,
        test_loader,
        device,
        enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
        max_batches=eval_cfg.get("max_test_batches"),
    )

    val_payload = {
        "task_name": task_name,
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "metrics": val_metrics,
    }
    test_payload = {
        "task_name": task_name,
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "metrics": test_metrics,
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], val_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], test_payload)

    runtime_sec = float(time.time() - start_time)
    improves_translation = "no"
    if test_metrics["signed_tdir_mean_deg"] is not None and test_metrics["anti_parallel_rate"] is not None:
        improved_signed = float(test_metrics["signed_tdir_mean_deg"]) < float(T57B_EXTERNAL_REFERENCE["signed_tdir_mean_deg"])
        improved_anti = float(test_metrics["anti_parallel_rate"]) < float(T57B_EXTERNAL_REFERENCE["anti_parallel_rate"])
        improved_scale = (
            test_metrics["tmag_median_ratio"] is not None
            and float(test_metrics["tmag_median_ratio"]) > 0.3
        )
        improved_path = (
            test_metrics["path_ratio"] is not None
            and float(test_metrics["path_ratio"]) > float(T57B_EXTERNAL_REFERENCE["path_ratio"])
        )
        if improved_signed and improved_anti:
            improves_translation = "yes"
        elif improved_signed or improved_anti or improved_scale or improved_path:
            improves_translation = "partial"

    next_task = (
        "proceed_to_BASE360_HKUST_360DVO_official_baseline_eval"
        if improves_translation == "yes"
        else "proceed_to_TRAIN360D_observability_or_kstep"
        if improves_translation == "partial"
        else "rerun_TRAIN360C_with_adjusted_loss"
    )

    comparison_rows = [
        {
            "model": "T57b external reference",
            "split": T57B_EXTERNAL_REFERENCE["split"],
            "signed_tdir_mean": T57B_EXTERNAL_REFERENCE["signed_tdir_mean_deg"],
            "anti_parallel_rate": T57B_EXTERNAL_REFERENCE["anti_parallel_rate"],
            "tmag_median_ratio": T57B_EXTERNAL_REFERENCE["tmag_median_ratio"],
            "path_ratio": T57B_EXTERNAL_REFERENCE["path_ratio"],
            "rot_mean": T57B_EXTERNAL_REFERENCE["rot_mean_deg"],
            "notes": "GEN5 external reference from TRAIN360A/B audit",
        },
        {
            "model": task_name,
            "split": "val",
            "signed_tdir_mean": val_metrics["signed_tdir_mean_deg"],
            "anti_parallel_rate": val_metrics["anti_parallel_rate"],
            "tmag_median_ratio": val_metrics["tmag_median_ratio"],
            "path_ratio": val_metrics["path_ratio"],
            "rot_mean": val_metrics["rot_mean_deg"],
            "notes": "best-val selected checkpoint",
        },
        {
            "model": task_name,
            "split": "test",
            "signed_tdir_mean": test_metrics["signed_tdir_mean_deg"],
            "anti_parallel_rate": test_metrics["anti_parallel_rate"],
            "tmag_median_ratio": test_metrics["tmag_median_ratio"],
            "path_ratio": test_metrics["path_ratio"],
            "rot_mean": test_metrics["rot_mean_deg"],
            "notes": "test evaluated once with best-val checkpoint",
        },
    ]

    report_lines = [
        "# TRAIN360C spherical pose baseline",
        "",
        "## 1. Executive summary",
        "- real training executed: `true`",
        "- learned weights saved: `true`",
        f"- recommended main checkpoint: `{selected_checkpoint_path}`",
        f"- improves over T57b external translation: `{improves_translation}`",
        f"- next step recommendation: `{next_task}`",
        "",
        "## 2. Data compliance",
        f"- train manifest: `{REPO_ROOT / inputs['train_manifest']}`",
        f"- val manifest: `{REPO_ROOT / inputs['val_manifest']}`",
        f"- test manifest: `{REPO_ROOT / inputs['test_manifest']}`",
        f"- train pairs: `{precheck['dataset_summaries']['train']['kept_row_count']}`",
        f"- val pairs: `{precheck['dataset_summaries']['val']['kept_row_count']}`",
        f"- test pairs: `{precheck['dataset_summaries']['test']['kept_row_count']}`",
        f"- train sequences: `{precheck['dataset_summaries']['train']['sequence_ids']}`",
        f"- val sequences: `{precheck['dataset_summaries']['val']['sequence_ids']}`",
        f"- test sequences: `{precheck['dataset_summaries']['test']['sequence_ids']}`",
        f"- no sequence overlap: `{not precheck['split_audit']['has_overlap']}`",
        "- no random pair split: `true`",
        "- no direct glob `data/360DVO/Sequences/*`: `true`",
        "",
        "## 3. Model configuration",
        f"- backbone: `PanoramaRelPoseModel`",
        f"- checkpoint init: `{init_ckpt_path}`",
        f"- strict/non-strict load: `{load_status['status']}`",
        f"- missing keys: `{load_status['missing_keys']}`",
        f"- unexpected keys: `{load_status['unexpected_keys']}`",
        "- enabled heads: `rotation`, `translation direction`, `translation magnitude`",
        f"- disabled heads: `fine_stage={not bool(model_cfg['enable_fine_stage'])}`, `coupled_pose_head={not bool(model_cfg['enable_coupled_pose_head'])}`",
        "- bounded log_tmag: `true`",
        "",
        "## 4. Loss configuration",
        "- rotation loss: `SO(3) geodesic`",
        "- tdir loss: `1 - cosine` in B frame",
        f"- tmag/log_tmag loss: `{loss_cfg['tmag_loss_type']}`",
        f"- weights: `rot={loss_cfg['rot_weight']}, tdir={loss_cfg['tdir_weight']}, tmag={loss_cfg['tmag_weight']}`",
        f"- k-step weighting: `k1={loss_cfg['k_weight_k1']}, k2={loss_cfg['k_weight_k2']}, k3={loss_cfg['k_weight_k3']}, k5={loss_cfg['k_weight_k5']}`",
        f"- epsilon guard: `{data_cfg['tmag_epsilon']}`",
        "",
        "## 5. Training details",
        f"- epochs: `{train_cfg['epochs']}`",
        f"- batch size: `{data_cfg['train_batch_size']}`",
        f"- optimizer: `{train_cfg['optimizer']}`",
        f"- learning rate: `{train_cfg['lr']}`",
        f"- scheduler: `{train_cfg['scheduler']}`",
        f"- grad clipping: `{train_cfg['grad_clip_norm']}`",
        f"- AMP: `{str(train_cfg['amp']).lower()}`",
        f"- seed: `{train_cfg['seed']}`",
        f"- hardware: `{system_info}`",
        f"- runtime_sec: `{runtime_sec:.2f}`",
        f"- train subset info: `{subset_info}`",
        "",
        "## 6. Validation metrics",
        f"- best epoch: `{best_epoch}`",
        f"- metrics: `{val_metrics}`",
        "",
        "## 7. Test metrics",
        f"- final selected checkpoint: `{selected_checkpoint_path}`",
        f"- metrics: `{test_metrics}`",
        "",
        "## 8. Comparison to T57b",
    ]
    for row in comparison_rows:
        report_lines.append(f"- {row}")

    report_lines.extend(
        [
            "",
            "## 9. Failure analysis",
            (
                "- signed direction still anti-parallel heavy."
                if test_metrics.get("anti_parallel_rate") is not None and float(test_metrics["anti_parallel_rate"]) >= 0.5
                else "- anti-parallel behavior is materially reduced relative to the weak T57b reference."
            ),
            (
                "- scale remains under-estimated."
                if test_metrics.get("tmag_median_ratio") is not None and float(test_metrics["tmag_median_ratio"]) < 0.3
                else "- scale no longer collapses as severely as the T57b reference."
            ),
            (
                "- rotation is better than translation, suggesting correspondence ambiguity still dominates translation."
                if test_metrics.get("rot_mean_deg") is not None and test_metrics.get("signed_tdir_mean_deg") is not None and float(test_metrics["signed_tdir_mean_deg"]) > 5 * float(test_metrics["rot_mean_deg"])
                else "- rotation/translation gap is not the dominant failure mode."
            ),
            (
                "- path ratio still collapsed."
                if test_metrics.get("path_ratio") is not None and float(test_metrics["path_ratio"]) <= float(T57B_EXTERNAL_REFERENCE["path_ratio"])
                else "- path ratio improved over the T57b external reference."
            ),
            "",
            "## 10. Next step recommendation",
            f"- `{next_task}`",
            "",
            "## 11. Compliance checklist",
            "- `real_training_executed = true`",
            "- `learned_weights_saved = true`",
            "- `train_manifest_used = true`",
            "- `val_manifest_used_for_validation_only = true`",
            "- `test_manifest_used_for_final_eval_only = true`",
            "- `uses_eval_gt_for_training = false`",
            "- `uses_test_gt_for_training = false`",
            "- `uses_orbslam3_teacher = false`",
            "- `uses_hkust_360dvo_teacher = false`",
            "- `s5_locked_metrics_modified = false`",
            "- `legacy_scene01_artifact_dependency = false`",
            "- `direct_glob_data_360dvo_sequences = false`",
            "- `dset2c_canonical_manifest_required = true`",
            "- `s5e15_external_inference_model_claimed = false`",
        ]
    )
    (REPO_ROOT / outputs["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("training executed: true")
    print("checkpoint saved: true")
    print(f"best checkpoint: {selected_checkpoint_path}")
    print(f"val signed_tdir_mean: {val_metrics['signed_tdir_mean_deg']}")
    print(f"test signed_tdir_mean: {test_metrics['signed_tdir_mean_deg']}")
    print(f"test anti_parallel_rate: {test_metrics['anti_parallel_rate']}")
    print(f"test tmag_median_ratio: {test_metrics['tmag_median_ratio']}")
    print(f"test path_ratio: {test_metrics['path_ratio']}")
    print(f"improves over T57b translation: {improves_translation}")
    print(f"next recommended task: {next_task}")


if __name__ == "__main__":
    train()
