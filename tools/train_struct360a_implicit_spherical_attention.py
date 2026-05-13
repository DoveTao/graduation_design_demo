#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
from models.struct360a_implicit_spherical_attention import (
    STRUCT360AImplicitSphericalAttentionModel,
    count_parameters,
)
from tools.eval_train360_pose import evaluate_train360_pose
from train360d_pose_losses import train360d_pose_loss


T57B_REFERENCE = {
    "signed_tdir_mean_deg": 111.96493221327962,
    "anti_parallel_rate": 0.6746724890829694,
    "tmag_median_ratio": 0.1751560082454769,
    "path_ratio": 0.14878731297064046,
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def _subset_dataset(dataset: Dset2CCanonicalPairDataset, max_count: Optional[int], seed: int):
    if max_count is None or int(max_count) <= 0 or len(dataset) <= int(max_count):
        return dataset, {"subset_used": False, "subset_count": len(dataset), "original_count": len(dataset)}
    rng = np.random.default_rng(int(seed))
    idx = sorted(int(x) for x in rng.permutation(len(dataset))[: int(max_count)].tolist())
    return Subset(dataset, idx), {
        "subset_used": True,
        "subset_count": len(idx),
        "original_count": len(dataset),
        "subset_seed": int(seed),
    }


def _hist_from_dataset(dataset: Dset2CCanonicalPairDataset) -> Dict[str, Any]:
    ks: Dict[int, int] = {}
    adjacent = 0
    non_adjacent = 0
    tmags: List[float] = []
    for sample in dataset.samples:
        k = int(sample["k"])
        ks[k] = int(ks.get(k, 0)) + 1
        if k == 1 or str(sample.get("pair_type", "")) == "adjacent":
            adjacent += 1
        else:
            non_adjacent += 1
        tmags.append(float(sample["tmag"]))
    arr = np.asarray(tmags, dtype=np.float64)
    return {
        "pair_count": len(dataset.samples),
        "sequence_ids": sorted({str(s["sequence_id"]) for s in dataset.samples}),
        "k_histogram": {str(k): int(v) for k, v in sorted(ks.items())},
        "adjacent_pair_count": int(adjacent),
        "non_adjacent_pair_count": int(non_adjacent),
        "tmag_quantiles": {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        },
    }


def _extract_base360_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    all_pairs = payload.get("pair_metrics", {}).get("all_pairs", {})
    return {
        "rot_mean_deg": all_pairs.get("rot_mean_deg"),
        "rot_median_deg": all_pairs.get("rot_median_deg"),
        "signed_tdir_mean_deg": all_pairs.get("signed_tdir_mean_deg"),
        "signed_tdir_median_deg": all_pairs.get("signed_tdir_median_deg"),
        "unsigned_tdir_mean_deg": all_pairs.get("unsigned_tdir_mean_deg"),
        "anti_parallel_rate": all_pairs.get("anti_parallel_rate"),
        "tmag_median_ratio": all_pairs.get("tmag_median_ratio"),
        "tmag_mean_ratio": all_pairs.get("tmag_mean_ratio"),
        "tmag_ratio_p10": all_pairs.get("tmag_p10_ratio"),
        "tmag_ratio_p90": all_pairs.get("tmag_p90_ratio"),
        "log_tmag_mae": all_pairs.get("log_tmag_mae"),
        "scale_collapse_rate": all_pairs.get("scale_collapse_rate"),
        "scale_explosion_rate": all_pairs.get("scale_explosion_rate"),
        "path_ratio": all_pairs.get("pair_component_path_ratio"),
        "pair_component_path_ratio": all_pairs.get("pair_component_path_ratio"),
        "coverage": all_pairs.get("pair_coverage"),
    }


def _extract_metrics_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _compute_val_score(metrics: Mapping[str, Any], score_cfg: Optional[Mapping[str, Any]] = None, eps: float = 1.0e-6) -> float:
    score_cfg = score_cfg or {}
    signed = float(metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(metrics.get("anti_parallel_rate") or 1.0)
    tmag_ratio = float(metrics.get("tmag_median_ratio") or eps)
    path_ratio = float(metrics.get("path_ratio") or eps)
    rot = float(metrics.get("rot_mean_deg") or 0.0)
    return (
        float(score_cfg.get("signed_tdir_weight", 1.0)) * signed
        + float(score_cfg.get("anti_parallel_weight", 50.0)) * anti
        + float(score_cfg.get("tmag_ratio_weight", 20.0)) * abs(math.log(max(tmag_ratio, eps)))
        + float(score_cfg.get("path_ratio_weight", 10.0)) * abs(math.log(max(path_ratio, eps)))
        + float(score_cfg.get("rot_weight", 0.0)) * rot
    )


def _classification(test_metrics: Mapping[str, Any], train360d_test: Mapping[str, Any], train360h_test: Mapping[str, Any]) -> str:
    signed = float(test_metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(test_metrics.get("anti_parallel_rate") or 1.0)
    tmag = float(test_metrics.get("tmag_median_ratio") or 0.0)
    path = float(test_metrics.get("path_ratio") or 0.0)
    d_signed = float(train360d_test.get("signed_tdir_mean_deg") or float("inf"))
    d_anti = float(train360d_test.get("anti_parallel_rate") or 1.0)
    h_tmag = float(train360h_test.get("tmag_median_ratio") or 0.0)
    h_path = float(train360h_test.get("path_ratio") or 0.0)
    if signed < d_signed or anti < d_anti:
        if tmag >= h_tmag * 0.95 and path >= h_path * 0.95:
            return "balanced"
        return "direction_best"
    if tmag >= h_tmag or path >= h_path:
        return "scale_best"
    if signed <= d_signed * 1.1 and path >= h_path * 0.9:
        return "partial"
    return "regression"


def _inject_struct360a_cfg(base_cfg: Dict[str, Any], train_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base_cfg)
    out["H"] = int(train_cfg["data"]["image_hw"][0])
    out["W"] = int(train_cfg["data"]["image_hw"][1])
    out["use_fine_stage"] = bool(train_cfg["model"]["enable_fine_stage"])
    out["use_coupled_pose_residual_head"] = bool(train_cfg["model"]["enable_coupled_pose_head"])
    out["use_depth_branch"] = False
    out["use_cross_context"] = False
    struct_cfg = train_cfg["model"]["struct360a"]
    out["struct360a_attention_layers"] = int(struct_cfg["attention_layers"])
    out["struct360a_attention_heads"] = int(struct_cfg["attention_heads"])
    out["struct360a_attention_dropout"] = float(struct_cfg["attention_dropout"])
    out["struct360a_attention_mlp_ratio"] = float(struct_cfg["attention_mlp_ratio"])
    out["struct360a_pair_context_strength"] = float(struct_cfg["pair_context_strength"])
    out["struct360a_share_cross_for_translation_branch"] = bool(struct_cfg["share_cross_for_translation_branch"])
    return out


def _load_model_with_status(
    checkpoint_path: Path,
    device: torch.device,
    *,
    strict_attempt: bool,
    train_cfg: Mapping[str, Any],
) -> Tuple[STRUCT360AImplicitSphericalAttentionModel, Config, Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = payload.get("cfg", {})
    cfg_dict = _inject_struct360a_cfg(dict(cfg_dict), train_cfg)
    cfg = _cfg_from_dict(cfg_dict)
    model = STRUCT360AImplicitSphericalAttentionModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
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
        init_status = {"newly_initialized_params": [], "checkpoint_loaded_params": list(state.keys())}
        return model, cfg, load_status, init_status

    model = STRUCT360AImplicitSphericalAttentionModel(cfg, device).to(device)
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
    init_status = {
        "newly_initialized_params": list(missing),
        "checkpoint_loaded_params": sorted(filtered.keys()),
    }
    return model, cfg, load_status, init_status


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


def _run_prechecks(cfg: Mapping[str, Any], cfg_path: Path) -> Dict[str, Any]:
    git_status_short = _git(["git", "status", "--short"]).splitlines()
    branch = _git(["git", "branch", "--show-current"])
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    commit = _git(["git", "rev-parse", "HEAD"])
    blockers: List[str] = []
    if branch != str(cfg["prechecks"]["expected_branch"]):
        blockers.append(f"current branch mismatch: {branch} != {cfg['prechecks']['expected_branch']}")

    inputs = cfg["inputs"]
    required_paths = {
        "train_manifest": REPO_ROOT / inputs["train_manifest"],
        "val_manifest": REPO_ROOT / inputs["val_manifest"],
        "test_manifest": REPO_ROOT / inputs["test_manifest"],
        "hygiene_json": REPO_ROOT / inputs["hygiene_json"],
        "init_checkpoint": REPO_ROOT / inputs["init_checkpoint"],
        "dataset_adapter": REPO_ROOT / inputs["dataset_adapter"],
        "train360c_val_metrics": REPO_ROOT / inputs["train360c_val_metrics"],
        "train360c_test_metrics": REPO_ROOT / inputs["train360c_test_metrics"],
        "train360d_val_metrics": REPO_ROOT / inputs["train360d_val_metrics"],
        "train360d_test_metrics": REPO_ROOT / inputs["train360d_test_metrics"],
        "train360h_val_metrics": REPO_ROOT / inputs["train360h_val_metrics"],
        "train360h_test_metrics": REPO_ROOT / inputs["train360h_test_metrics"],
        "base360d_val_metrics": REPO_ROOT / inputs["base360d_val_metrics"],
        "base360d_test_metrics": REPO_ROOT / inputs["base360d_test_metrics"],
    }
    for name, path in required_paths.items():
        if not path.is_file():
            blockers.append(f"missing required file: {name} -> {path}")

    adapter_text = required_paths["dataset_adapter"].read_text(encoding="utf-8")
    adapter_text_norm = " ".join(adapter_text.split())
    if "manifest-native" not in adapter_text_norm and "jsonl manifest and never scans raw sequence directories" not in adapter_text_norm:
        blockers.append("dataset adapter no longer advertises manifest-native / no-scan behavior")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to use raw sequence globbing")

    cfg_text = cfg_path.read_text(encoding="utf-8")
    forbidden_refs = ["S5E15", "ORB-SLAM3", "hkust_360dvo_teacher", "base360_outputs_used_as_training_input"]
    for ref in forbidden_refs:
        if ref in cfg_text:
            blockers.append(f"config contains forbidden reference: {ref}")

    image_hw = tuple(int(x) for x in cfg["data"]["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(
        str(required_paths["train_manifest"]),
        expected_split="train",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    ds_val = Dset2CCanonicalPairDataset(
        str(required_paths["val_manifest"]),
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        str(required_paths["test_manifest"]),
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(cfg["data"]["require_paths"]),
        skip_invalid=bool(cfg["data"]["skip_invalid"]),
    )
    split_audit = summarize_manifest_group([ds_train, ds_val, ds_test])
    if split_audit["has_overlap"]:
        blockers.append(f"sequence overlap detected: {split_audit['sequence_overlap']}")

    train360c_val = _extract_metrics_payload(_read_json(required_paths["train360c_val_metrics"]))
    train360c_test = _extract_metrics_payload(_read_json(required_paths["train360c_test_metrics"]))
    train360d_val = _extract_metrics_payload(_read_json(required_paths["train360d_val_metrics"]))
    train360d_test = _extract_metrics_payload(_read_json(required_paths["train360d_test_metrics"]))
    train360h_val = _extract_metrics_payload(_read_json(required_paths["train360h_val_metrics"]))
    train360h_test = _extract_metrics_payload(_read_json(required_paths["train360h_test_metrics"]))
    base360d_val = _extract_base360_metrics(_read_json(required_paths["base360d_val_metrics"]))
    base360d_test = _extract_base360_metrics(_read_json(required_paths["base360d_test_metrics"]))

    return {
        "blockers": blockers,
        "git_status_short": git_status_short,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_commit": commit,
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "dataset_histograms": {
            "train": _hist_from_dataset(ds_train),
            "val": _hist_from_dataset(ds_val),
            "test": _hist_from_dataset(ds_test),
        },
        "split_audit": split_audit,
        "required_paths": {k: str(v) for k, v in required_paths.items()},
        "train360c_val_metrics": train360c_val,
        "train360c_test_metrics": train360c_test,
        "train360d_val_metrics": train360d_val,
        "train360d_test_metrics": train360d_test,
        "train360h_val_metrics": train360h_val,
        "train360h_test_metrics": train360h_test,
        "base360d_val_metrics": base360d_val,
        "base360d_test_metrics": base360d_test,
    }


def _write_blocker_artifacts(cfg: Mapping[str, Any], blocker_lines: List[str], precheck: Mapping[str, Any]) -> None:
    outputs = cfg["outputs"]
    report_path = REPO_ROOT / outputs["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = ["# STRUCT360A blocked before training", "", "## Blockers"]
    body.extend([f"- {line}" for line in blocker_lines])
    report_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    precheck_summary = {
        "blockers": list(precheck.get("blockers", [])),
        "git_status_short": list(precheck.get("git_status_short", [])),
        "branch": precheck.get("branch"),
        "branch_vv": list(precheck.get("branch_vv", [])),
        "git_commit": precheck.get("git_commit"),
        "dataset_histograms": precheck.get("dataset_histograms"),
        "split_audit": precheck.get("split_audit"),
        "required_paths": precheck.get("required_paths"),
    }
    blocker_payload = {
        "task_name": str(cfg.get("task_name", "STRUCT360A_main_Dinit")),
        "training_executed": False,
        "checkpoint_saved": False,
        "blockers": blocker_lines,
        "precheck": precheck_summary,
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], blocker_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], blocker_payload)


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "struct360a_implicit_spherical_cross_attention.yaml")
    cfg = load_yaml_like(cfg_path)
    start_time = time.time()
    precheck = _run_prechecks(cfg, cfg_path)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck["blockers"], precheck)
        raise RuntimeError("STRUCT360A precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

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

    ds_train = precheck["datasets"]["train"]
    ds_val = precheck["datasets"]["val"]
    ds_test = precheck["datasets"]["test"]
    train_subset, subset_info = _subset_dataset(ds_train, max_count=data_cfg.get("train_subset_max"), seed=int(train_cfg["seed"]))

    train_loader = DataLoader(train_subset, batch_size=int(data_cfg["train_batch_size"]), shuffle=bool(data_cfg["shuffle_train"]), num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=int(data_cfg["eval_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=int(data_cfg["eval_batch_size"]), shuffle=False, num_workers=int(data_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(data_cfg["num_workers"]) > 0), drop_last=False)

    init_ckpt_path = REPO_ROOT / inputs["init_checkpoint"]
    model, ckpt_cfg, load_status, init_status = _load_model_with_status(
        init_ckpt_path,
        device,
        strict_attempt=bool(model_cfg["strict_load_attempt"]),
        train_cfg=cfg,
    )
    param_counts = count_parameters(model)
    new_prefixes = ("module2.struct360a_cross", "module2.struct360a_translation_cross")
    base_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(new_prefixes):
            new_params.append(param)
        else:
            base_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": float(train_cfg["lr"])},
            {"params": new_params, "lr": float(train_cfg.get("new_module_lr", train_cfg["lr"]))},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg["epochs"]), 1), eta_min=float(train_cfg["min_lr"]))
    scaler = torch.cuda.amp.GradScaler() if bool(train_cfg["amp"]) and device.type == "cuda" else None

    checkpoint_dir = REPO_ROOT / outputs["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = checkpoint_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    (checkpoint_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    best_val_score = float("inf")
    best_epoch = -1
    best_val_metrics: Optional[Dict[str, Any]] = None
    best_checkpoint_path = checkpoint_dir / "best_val.pt"

    metadata_common = {
        "task_name": str(cfg.get("task_name", "STRUCT360A_main_Dinit")),
        "init_checkpoint": str(init_ckpt_path),
        "git_branch": precheck["branch"],
        "git_commit": precheck["git_commit"],
        "parameter_count": param_counts,
        "compliance_flags": {
            "train360c_checkpoint_modified": False,
            "train360d_checkpoint_modified": False,
            "train360h_checkpoint_modified": False,
            "explicit_matching_used": False,
            "match_list_output": False,
            "ransac_used": False,
            "pnp_used": False,
            "bundle_adjustment_used": False,
            "hkust_360dvo_teacher_used": False,
            "base360_outputs_used_as_training_input": False,
            "train_manifest_used": True,
            "val_manifest_used_for_selection_only": True,
            "test_manifest_used_for_final_eval_only": True,
            "direct_glob_data_360dvo_sequences": False,
            "random_pair_split_used": False,
            "s5_locked_metrics_modified": False,
        },
    }

    component_cfg = loss_cfg.get("components", {})
    run_test_during_train = bool(eval_cfg.get("run_test", True))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_losses = {"loss_total": [], "loss_rot": [], "loss_tdir": [], "loss_tmag": [], "loss_scale_stability": []}
        attention_stats: Dict[str, List[float]] = {
            "entropy_ab": [],
            "entropy_ba": [],
            "max_ab": [],
            "max_ba": [],
            "concentration_ab": [],
            "concentration_ba": [],
            "relation_gate_mean": [],
        }
        train_nan_inf_count = 0
        obs_weights_all: List[float] = []
        obs_weights_adjacent: List[float] = []
        obs_weights_kstep: List[float] = []
        k_weights_all: List[float] = []

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
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                R_pred, _t_pred_local, aux = model(IA, IB, enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]), dt_world=dt_world_t)
                tdir_pred_B = aux.get("t_dir_out", aux["t_dir"])
                tmag_pred = aux.get("t_mag", aux.get("final_tmag"))
                if tmag_pred is None:
                    log_tmag = aux.get("log_t_mag", aux.get("final_log_tmag"))
                    tmag_pred = torch.exp(log_tmag.float())
                loss_dict = train360d_pose_loss(
                    R_pred=R_pred,
                    tdir_pred_B=tdir_pred_B,
                    tmag_pred=tmag_pred,
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    k_step_config=loss_cfg["k_step_balancing"],
                    observability_config=loss_cfg["observability"],
                    scale_config=loss_cfg["scale_stability"],
                    rot_weight=float(loss_cfg["rot_weight"]),
                    tdir_weight=float(loss_cfg["tdir_weight"]),
                    tmag_weight=float(loss_cfg["tmag_weight"]),
                    scale_stability_weight=float(loss_cfg["scale_stability_weight"]),
                    tmag_loss_type=str(loss_cfg["tmag_loss_type"]),
                    tmag_epsilon=float(data_cfg["tmag_epsilon"]),
                    enable_observability=bool(component_cfg.get("enable_observability", True)),
                    enable_k_step_balancing=bool(component_cfg.get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(component_cfg.get("enable_scale_stabilization", True)),
                )
            if scaler is not None:
                scaler.scale(loss_dict["loss_total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                optimizer.step()
            for key in epoch_losses:
                epoch_losses[key].append(float(loss_dict[key].detach().cpu()))

            obs_np = loss_dict["obs_weights"].detach().cpu().numpy().astype(np.float64)
            k_np = k_tensor.detach().cpu().numpy().astype(np.int64)
            k_weight_np = loss_dict["k_weights"].detach().cpu().numpy().astype(np.float64)
            obs_weights_all.extend(obs_np.tolist())
            k_weights_all.extend(k_weight_np.tolist())
            obs_weights_adjacent.extend(obs_np[k_np == 1].tolist())
            obs_weights_kstep.extend(obs_np[k_np != 1].tolist())
            train_nan_inf_count += int(sum((~torch.isfinite(v)).sum().item() for v in [R_pred, tdir_pred_B, tmag_pred, loss_dict["loss_total"]]))

            for key, bucket in (
                ("struct360a_attention_affinity_entropy_ab", "entropy_ab"),
                ("struct360a_attention_affinity_entropy_ba", "entropy_ba"),
                ("struct360a_attention_affinity_max_ab", "max_ab"),
                ("struct360a_attention_affinity_max_ba", "max_ba"),
                ("struct360a_attention_affinity_concentration_ab", "concentration_ab"),
                ("struct360a_attention_affinity_concentration_ba", "concentration_ba"),
                ("struct360a_relation_gate_mean", "relation_gate_mean"),
            ):
                value = aux.get(key)
                if torch.is_tensor(value):
                    attention_stats[bucket].append(float(value.detach().float().mean().cpu()))

        scheduler.step()
        val_metrics = evaluate_train360_pose(model, val_loader, device, enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]), tmag_epsilon=float(data_cfg["tmag_epsilon"]), max_batches=eval_cfg.get("max_val_batches"))
        val_score = _compute_val_score(val_metrics, score_cfg=eval_cfg.get("selection_score"), eps=float(data_cfg["tmag_epsilon"]))
        epoch_log = {
            "epoch": int(epoch),
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_new_module": float(optimizer.param_groups[1]["lr"]),
            "train_loss_total": float(np.mean(epoch_losses["loss_total"])) if epoch_losses["loss_total"] else None,
            "train_loss_rot": float(np.mean(epoch_losses["loss_rot"])) if epoch_losses["loss_rot"] else None,
            "train_loss_tdir": float(np.mean(epoch_losses["loss_tdir"])) if epoch_losses["loss_tdir"] else None,
            "train_loss_tmag": float(np.mean(epoch_losses["loss_tmag"])) if epoch_losses["loss_tmag"] else None,
            "train_loss_scale_stability": float(np.mean(epoch_losses["loss_scale_stability"])) if epoch_losses["loss_scale_stability"] else None,
            "train_nan_inf_count": int(train_nan_inf_count),
            "val_score": float(val_score),
            "val_metrics": val_metrics,
            "latent_cross_image_association": {k: (float(np.mean(v)) if v else None) for k, v in attention_stats.items()},
            "obs_weight_stats": {
                "min": float(np.min(obs_weights_all)) if obs_weights_all else None,
                "median": float(np.percentile(np.asarray(obs_weights_all, dtype=np.float64), 50)) if obs_weights_all else None,
                "max": float(np.max(obs_weights_all)) if obs_weights_all else None,
                "adjacent_mean": float(np.mean(np.asarray(obs_weights_adjacent, dtype=np.float64))) if obs_weights_adjacent else None,
                "kstep_mean": float(np.mean(np.asarray(obs_weights_kstep, dtype=np.float64))) if obs_weights_kstep else None,
            },
            "k_weight_stats": {
                "min": float(np.min(k_weights_all)) if k_weights_all else None,
                "median": float(np.percentile(np.asarray(k_weights_all, dtype=np.float64), 50)) if k_weights_all else None,
                "max": float(np.max(k_weights_all)) if k_weights_all else None,
            },
        }
        _append_jsonl(train_log_path, epoch_log)
        if val_score < best_val_score and int(val_metrics["nan_inf_count"]) == 0:
            best_val_score = float(val_score)
            best_epoch = int(epoch)
            best_val_metrics = dict(val_metrics)
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                ckpt_cfg,
                {
                    **metadata_common,
                    "epoch": int(epoch),
                    "val_score": float(val_score),
                    "checkpoint_role": "best_val",
                    "checkpoint_load_status": load_status,
                    "new_parameter_init_status": init_status,
                    "subset_info": subset_info,
                },
            )

    final_checkpoint_path = checkpoint_dir / "final.pt"
    _save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        scheduler,
        int(train_cfg["epochs"]),
        ckpt_cfg,
        {
            **metadata_common,
            "epoch": int(train_cfg["epochs"]),
            "val_score": float(best_val_score),
            "checkpoint_role": "final",
            "checkpoint_load_status": load_status,
            "new_parameter_init_status": init_status,
            "subset_info": subset_info,
        },
    )

    selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
    selected_payload = torch.load(str(selected_checkpoint_path), map_location=device)
    selected_model = STRUCT360AImplicitSphericalAttentionModel(ckpt_cfg, device).to(device)
    selected_model.load_state_dict(selected_payload["model"], strict=False)
    selected_model.eval()

    val_metrics = evaluate_train360_pose(selected_model, val_loader, device, enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]), tmag_epsilon=float(data_cfg["tmag_epsilon"]), max_batches=eval_cfg.get("max_val_batches"))
    test_metrics = evaluate_train360_pose(selected_model, test_loader, device, enable_depth_fusion=bool(model_cfg["enable_depth_fusion"]), tmag_epsilon=float(data_cfg["tmag_epsilon"]), max_batches=eval_cfg.get("max_test_batches")) if run_test_during_train else {"skipped": True}
    val_score = _compute_val_score(val_metrics, score_cfg=eval_cfg.get("selection_score"), eps=float(data_cfg["tmag_epsilon"]))

    val_payload = {
        "task_name": str(cfg.get("task_name")),
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "val_score": float(val_score),
        "metrics": val_metrics,
    }
    test_payload = {
        "task_name": str(cfg.get("task_name")),
        "checkpoint_used": str(selected_checkpoint_path),
        "best_epoch": int(best_epoch),
        "val_score_of_selected_checkpoint": float(val_score),
        "metrics": test_metrics,
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], val_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], test_payload)

    train360c_val = precheck["train360c_val_metrics"]
    train360c_test = precheck["train360c_test_metrics"]
    train360d_val = precheck["train360d_val_metrics"]
    train360d_test = precheck["train360d_test_metrics"]
    train360h_val = precheck["train360h_val_metrics"]
    train360h_test = precheck["train360h_test_metrics"]
    base360d_test = precheck["base360d_test_metrics"]
    discrepancy = abs(float(val_metrics["signed_tdir_mean_deg"]) - float(test_metrics["signed_tdir_mean_deg"]))
    classification = _classification(test_metrics, train360d_test, train360h_test)
    versus_d = "better" if float(test_metrics["signed_tdir_mean_deg"]) < float(train360d_test["signed_tdir_mean_deg"]) or float(test_metrics["anti_parallel_rate"]) < float(train360d_test["anti_parallel_rate"]) else "partial" if float(test_metrics["path_ratio"]) > float(train360d_test["path_ratio"]) or float(test_metrics["tmag_median_ratio"]) > float(train360d_test["tmag_median_ratio"]) else "worse"
    versus_h = "better" if float(test_metrics["tmag_median_ratio"]) >= float(train360h_test["tmag_median_ratio"]) and float(test_metrics["path_ratio"]) >= float(train360h_test["path_ratio"]) else "partial" if float(test_metrics["signed_tdir_mean_deg"]) < float(train360h_test["signed_tdir_mean_deg"]) else "worse"

    train_log_rows = []
    if train_log_path.exists():
        train_log_rows = [json.loads(line) for line in train_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest_assoc = train_log_rows[-1]["latent_cross_image_association"] if train_log_rows else {}

    next_recommendation = (
        "prepare_thesis_experiment_section"
        if classification in {"balanced", "direction_best"}
        else "proceed_to_STRUCT360B_rotation_compensated_interaction"
    )

    report_lines = [
        "# STRUCT360A implicit spherical cross attention",
        "",
        "## 1. Executive summary",
        "- training executed true/false: `true`",
        "- checkpoint saved true/false: `true`",
        f"- best checkpoint: `{selected_checkpoint_path}`",
        f"- whether STRUCT360A improves over TRAIN360D/H: `vs TRAIN360D={versus_d}, vs TRAIN360H={versus_h}`",
        f"- classification: `{classification}`",
        "",
        "## 2. Research alignment",
        "- method type: `match-free, end-to-end, implicit cross-image spherical attention`",
        "- no explicit matching: `true`",
        "- no RANSAC / PnP / BA: `true`",
        "- no HKUST 360DVO teacher: `true`",
        "- no explicit correspondence supervision or match list output: `true`",
        "",
        "## 3. Architecture",
        "- base model inherited from TRAIN360D coarse pose-regression backbone with manifest-native DSET2C pipeline.",
        "- new module: `ImplicitSphericalCrossAttentionEncoder` over coarse spherical tokens.",
        "- spherical positional encoding: `bearing xyz + lat/lon sincos + ERP latitude distortion channels`.",
        "- relation pooling: `pair_context MLP over pooled A/B/cross responses with sigmoid relation gate`.",
        "- pose decoder: `unchanged coarse interaction + pose head`, so the experiment isolates feature-level cross-image relation learning.",
        f"- parameter count: `{param_counts}`",
        "",
        "## 4. Initialization",
        f"- init checkpoint: `{init_ckpt_path}`",
        f"- strict/non-strict load: `{load_status['status']}`",
        f"- missing keys: `{load_status['missing_keys']}`",
        f"- unexpected keys: `{load_status['unexpected_keys']}`",
        f"- new random params: `{init_status['newly_initialized_params']}`",
        "",
        "## 5. Training setup",
        f"- train manifest: `{REPO_ROOT / inputs['train_manifest']}`",
        f"- val manifest: `{REPO_ROOT / inputs['val_manifest']}`",
        f"- test manifest: `{REPO_ROOT / inputs['test_manifest']}`",
        f"- epochs: `{train_cfg['epochs']}`",
        f"- batch size: `{data_cfg['train_batch_size']}`",
        f"- LR backbone: `{train_cfg['lr']}`",
        f"- LR new module: `{train_cfg['new_module_lr']}`",
        f"- loss weights: `rot={loss_cfg['rot_weight']}, tdir={loss_cfg['tdir_weight']}, tmag={loss_cfg['tmag_weight']}, scale={loss_cfg['scale_stability_weight']}`",
        f"- observability / k-step / scale settings: `{loss_cfg}`",
        f"- val score: `{eval_cfg['selection_score']}`",
        f"- train subset info: `{subset_info}`",
        f"- system info: `{system_info}`",
        "",
        "## 6. Val results",
        f"- STRUCT360A val metrics: `{val_metrics}`",
        f"- comparison to TRAIN360D val: `signed_tdir {train360d_val.get('signed_tdir_mean_deg')} -> {val_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360d_val.get('anti_parallel_rate')} -> {val_metrics.get('anti_parallel_rate')}, tmag_ratio {train360d_val.get('tmag_median_ratio')} -> {val_metrics.get('tmag_median_ratio')}, path_ratio {train360d_val.get('path_ratio')} -> {val_metrics.get('path_ratio')}`",
        f"- comparison to TRAIN360H val: `signed_tdir {train360h_val.get('signed_tdir_mean_deg')} -> {val_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360h_val.get('anti_parallel_rate')} -> {val_metrics.get('anti_parallel_rate')}, tmag_ratio {train360h_val.get('tmag_median_ratio')} -> {val_metrics.get('tmag_median_ratio')}, path_ratio {train360h_val.get('path_ratio')} -> {val_metrics.get('path_ratio')}`",
        "",
        "## 7. Test results",
        f"- STRUCT360A test metrics: `{test_metrics}`",
        f"- comparison to TRAIN360C: `signed_tdir {train360c_test.get('signed_tdir_mean_deg')} -> {test_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360c_test.get('anti_parallel_rate')} -> {test_metrics.get('anti_parallel_rate')}, tmag_ratio {train360c_test.get('tmag_median_ratio')} -> {test_metrics.get('tmag_median_ratio')}, path_ratio {train360c_test.get('path_ratio')} -> {test_metrics.get('path_ratio')}`",
        f"- comparison to TRAIN360D: `signed_tdir {train360d_test.get('signed_tdir_mean_deg')} -> {test_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360d_test.get('anti_parallel_rate')} -> {test_metrics.get('anti_parallel_rate')}, tmag_ratio {train360d_test.get('tmag_median_ratio')} -> {test_metrics.get('tmag_median_ratio')}, path_ratio {train360d_test.get('path_ratio')} -> {test_metrics.get('path_ratio')}`",
        f"- comparison to TRAIN360H: `signed_tdir {train360h_test.get('signed_tdir_mean_deg')} -> {test_metrics.get('signed_tdir_mean_deg')}, anti_parallel {train360h_test.get('anti_parallel_rate')} -> {test_metrics.get('anti_parallel_rate')}, tmag_ratio {train360h_test.get('tmag_median_ratio')} -> {test_metrics.get('tmag_median_ratio')}, path_ratio {train360h_test.get('path_ratio')} -> {test_metrics.get('path_ratio')}`",
        f"- comparison to T57b: `signed_tdir {T57B_REFERENCE['signed_tdir_mean_deg']} vs {test_metrics.get('signed_tdir_mean_deg')}, anti_parallel {T57B_REFERENCE['anti_parallel_rate']} vs {test_metrics.get('anti_parallel_rate')}, path_ratio {T57B_REFERENCE['path_ratio']} vs {test_metrics.get('path_ratio')}`",
        f"- comparison to BASE360D: `signed_tdir {base360d_test.get('signed_tdir_mean_deg')} vs {test_metrics.get('signed_tdir_mean_deg')}, anti_parallel {base360d_test.get('anti_parallel_rate')} vs {test_metrics.get('anti_parallel_rate')}, path_ratio {base360d_test.get('pair_component_path_ratio')} vs {test_metrics.get('path_ratio')}`",
        "",
        "## 8. Attention diagnostics",
        f"- latent cross-image association summary: `{latest_assoc}`",
        "- terminology note: `the attention statistics above describe latent association / attention affinity only, not explicit correspondences.`",
        "",
        "## 9. Failure analysis",
        f"- if direction worsens: `check whether latent association entropy stays too high and whether coarse pose head still under-uses cross-image relation features.`",
        f"- if scale worsens: `STRUCT360A kept TRAIN360D loss weights, so scale may still lag TRAIN360H if relation pooling helps direction more than magnitude calibration.`",
        f"- if attention overfits: `watch val/test discrepancy = {discrepancy}`",
        "- if discrepancy remains: `next step should adjust rotation-compensated interaction or balanced scale weighting rather than adding explicit matching.`",
        "",
        "## 10. Next recommendation",
        f"- `{next_recommendation}`",
        "",
        "## 11. Compliance checklist",
        "- `real_training_executed = true`",
        "- `learned_weights_saved = true`",
        "- `train360c_checkpoint_modified = false`",
        "- `train360d_checkpoint_modified = false`",
        "- `train360h_checkpoint_modified = false`",
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
        "- `direct_glob_data_360dvo_sequences = false`",
        "- `random_pair_split_used = false`",
        "- `s5_locked_metrics_modified = false`",
        "- `large_checkpoints_committed_to_git = false`",
    ]
    (REPO_ROOT / outputs["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# STRUCT360A vs TRAIN360C / D / H / BASE360D",
        "",
        "| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| STRUCT360A | val | {val_metrics.get('signed_tdir_mean_deg')} | {val_metrics.get('anti_parallel_rate')} | {val_metrics.get('tmag_median_ratio')} | {val_metrics.get('path_ratio')} | {val_metrics.get('coverage')} |",
        f"| STRUCT360A | test | {test_metrics.get('signed_tdir_mean_deg')} | {test_metrics.get('anti_parallel_rate')} | {test_metrics.get('tmag_median_ratio')} | {test_metrics.get('path_ratio')} | {test_metrics.get('coverage')} |",
        f"| TRAIN360C | test | {train360c_test.get('signed_tdir_mean_deg')} | {train360c_test.get('anti_parallel_rate')} | {train360c_test.get('tmag_median_ratio')} | {train360c_test.get('path_ratio')} | {train360c_test.get('coverage')} |",
        f"| TRAIN360D | test | {train360d_test.get('signed_tdir_mean_deg')} | {train360d_test.get('anti_parallel_rate')} | {train360d_test.get('tmag_median_ratio')} | {train360d_test.get('path_ratio')} | {train360d_test.get('coverage')} |",
        f"| TRAIN360H | test | {train360h_test.get('signed_tdir_mean_deg')} | {train360h_test.get('anti_parallel_rate')} | {train360h_test.get('tmag_median_ratio')} | {train360h_test.get('path_ratio')} | {train360h_test.get('coverage')} |",
        f"| T57b | reference | {T57B_REFERENCE['signed_tdir_mean_deg']} | {T57B_REFERENCE['anti_parallel_rate']} | {T57B_REFERENCE['tmag_median_ratio']} | {T57B_REFERENCE['path_ratio']} | 1.0 |",
        f"| BASE360D | test component | {base360d_test.get('signed_tdir_mean_deg')} | {base360d_test.get('anti_parallel_rate')} | {base360d_test.get('tmag_median_ratio')} | {base360d_test.get('pair_component_path_ratio')} | {base360d_test.get('coverage')} |",
        "",
        f"- classification: `{classification}`",
        f"- compared to TRAIN360D: `{versus_d}`",
        f"- compared to TRAIN360H: `{versus_h}`",
        f"- val/test discrepancy: `{discrepancy}`",
    ]
    (REPO_ROOT / outputs["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"- STRUCT360A training executed: true")
    print(f"- checkpoint saved: true")
    print(f"- best checkpoint: {selected_checkpoint_path}")
    print(f"- init source: TRAIN360D")
    print(f"- explicit matching used: false")
    print(f"- test signed_tdir_mean: {test_metrics.get('signed_tdir_mean_deg')}")
    print(f"- test anti_parallel_rate: {test_metrics.get('anti_parallel_rate')}")
    print(f"- test tmag_median_ratio: {test_metrics.get('tmag_median_ratio')}")
    print(f"- test path_ratio: {test_metrics.get('path_ratio')}")
    print(f"- val/test discrepancy: {discrepancy}")
    print(f"- compared to TRAIN360D: {versus_d}")
    print(f"- compared to TRAIN360H: {versus_h}")
    print(f"- classification: {classification}")
    print(f"- committed to git: false")
    print(f"- next recommended task: {next_recommendation}")


if __name__ == "__main__":
    train()
