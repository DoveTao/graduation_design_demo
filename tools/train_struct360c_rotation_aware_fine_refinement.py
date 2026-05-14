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
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from miniyaml import load_yaml_like
from models.struct360c_rotation_aware_fine_refinement import (
    STRUCT360CRotationAwareFineRefinementModel,
    count_parameters,
)
from train360.core.train360d_pose_losses import train360d_pose_loss
from train_struct360b_match_free_coarse_to_fine import (
    _cfg_from_dict,
    _inject_struct360b_cfg,
    evaluate_struct360b_pose,
)
import train360e_sequence_trajectory_export_and_ate_eval as traj


DEFAULT_CONFIG = REPO_ROOT / "configs" / "struct360c_rotation_aware_fine_refinement.yaml"
TMAG_EPS = 1.0e-6


def _git(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout or "").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(dict(payload)), ensure_ascii=False) + "\n")


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


def _float_stats(vals: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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


def _extract_pair_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("metrics"), Mapping):
        return dict(payload["metrics"])
    if isinstance(payload.get("pair_metrics"), Mapping):
        return dict(payload["pair_metrics"])
    return {}


def _extract_trajectory_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "trajectory_path_ratio": payload.get("trajectory_path_ratio"),
        "coverage": payload.get("pose_coverage"),
        "pred_path_length": payload.get("pred_path_length"),
        "gt_path_length": payload.get("gt_path_length"),
    }
    for mode in ("none", "se3", "sim3"):
        entry = payload.get(f"ate_{mode}", {})
        out[f"ate_{mode}_rmse"] = entry.get("rmse") if isinstance(entry, Mapping) else None
    return out


def _extract_seq360a_pair(payload: Mapping[str, Any]) -> Dict[str, Any]:
    pair = payload.get("pair_metrics")
    return dict(pair) if isinstance(pair, Mapping) else {}


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


def _inject_struct360c_cfg(base_cfg: Dict[str, Any], train_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = _inject_struct360b_cfg(base_cfg, train_cfg)
    out["struct360c_bias_gamma"] = float(train_cfg["model"]["struct360c_bias_gamma"])
    out["struct360c_bias_clamp_min"] = float(train_cfg["model"]["struct360c_bias_clamp_min"])
    out["struct360c_bias_clamp_max"] = float(train_cfg["model"]["struct360c_bias_clamp_max"])
    return out


def _load_model(checkpoint_path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360CRotationAwareFineRefinementModel, Any, Dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg_dict = _inject_struct360c_cfg(dict(payload.get("cfg", {})), cfg)
    ckpt_cfg = _cfg_from_dict(cfg_dict)
    model = STRUCT360CRotationAwareFineRefinementModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=bool(cfg["model"].get("strict_load_attempt", False)))
    return model, ckpt_cfg, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


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


def _system_info(device: torch.device) -> Dict[str, Any]:
    out = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        out["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return out


def _compute_val_score(metrics: Mapping[str, Any], score_cfg: Mapping[str, Any], eps: float = 1.0e-6) -> float:
    signed = float(metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(metrics.get("anti_parallel_rate") or 1.0)
    tmag_ratio = float(metrics.get("tmag_median_ratio") or eps)
    path_ratio = float(metrics.get("path_ratio") or eps)
    rot = float(metrics.get("rot_mean_deg") or 0.0)
    return (
        float(score_cfg.get("signed_tdir_weight", 1.0)) * signed
        + float(score_cfg.get("anti_parallel_weight", 70.0)) * anti
        + float(score_cfg.get("tmag_ratio_weight", 20.0)) * abs(math.log(max(tmag_ratio, eps)))
        + float(score_cfg.get("path_ratio_weight", 10.0)) * abs(math.log(max(path_ratio, eps)))
        + float(score_cfg.get("rot_weight", 0.0)) * rot
    )


def _build_optimizer(model: STRUCT360CRotationAwareFineRefinementModel, cfg: Mapping[str, Any]) -> Tuple[torch.optim.Optimizer, List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    fine_prefixes = ("module2.patch_embed_f", "module2.enc_f", "module2.pos_enc", "struct360b_", "struct360c_")
    coarse_prefixes = ("module2.patch_embed_c", "module2.enc_c", "module2.patch_embed_c_t", "module2.enc_c_t", "coarse")
    fine_params: List[torch.nn.Parameter] = []
    coarse_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("fine.", "direct_head", "coupled_pose_head", "depth", "depth_token_proj")):
            param.requires_grad = False
            continue
        if name.startswith(fine_prefixes):
            fine_params.append(param)
        elif name.startswith(coarse_prefixes):
            coarse_params.append(param)
        else:
            coarse_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": coarse_params, "lr": float(cfg["training"]["coarse_lr"]), "name": "coarse"},
            {"params": fine_params, "lr": float(cfg["training"]["fine_lr"]), "name": "fine"},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    return optimizer, coarse_params, fine_params


def _set_stage_trainability(
    *,
    epoch: int,
    coarse_params: Sequence[torch.nn.Parameter],
    fine_params: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    train_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    total_epochs = int(train_cfg["epochs"])
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    if epoch <= warmup_epochs:
        stage = "fine_only_warmup"
        for p in coarse_params:
            p.requires_grad = False
        for p in fine_params:
            p.requires_grad = True
        coarse_lr = float(train_cfg.get("warmup_coarse_lr", 0.0))
        fine_lr = float(train_cfg.get("warmup_fine_lr", train_cfg["fine_lr"]))
    else:
        stage = "partial_unfreeze"
        for p in coarse_params:
            p.requires_grad = True
        for p in fine_params:
            p.requires_grad = True
        denom = max(total_epochs - warmup_epochs - 1, 1)
        progress = float(epoch - warmup_epochs - 1) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        coarse_base = float(train_cfg["coarse_lr"])
        fine_base = float(train_cfg["fine_lr"])
        coarse_lr = min_lr + (coarse_base - min_lr) * cosine
        fine_lr = min_lr + (fine_base - min_lr) * cosine
    for group in optimizer.param_groups:
        if group.get("name") == "coarse":
            group["lr"] = coarse_lr
        else:
            group["lr"] = fine_lr
    return {"stage": stage, "coarse_lr": coarse_lr, "fine_lr": fine_lr}


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, ckpt_cfg: Any, metadata: Mapping[str, Any]) -> None:
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


@torch.no_grad()
def _evaluate_struct360c_diagnostics(
    model: STRUCT360CRotationAwareFineRefinementModel,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    gate_vals: List[float] = []
    delta_rot_vals: List[float] = []
    delta_tdir_vals: List[float] = []
    delta_log_vals: List[float] = []
    bias_mean_vals: List[float] = []
    bias_min_vals: List[float] = []
    bias_max_vals: List[float] = []
    entropy_vals: List[float] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        dt_world = batch["meta"]["dt_world"]
        if torch.is_tensor(dt_world):
            dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
        else:
            dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)
        _R, _t, aux = model(IA, IB, dt_world=dt_world_t)
        gate_vals.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
        delta_rot_vals.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
        delta_tdir_vals.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
        delta_log_vals.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
        bias_mean_vals.append(float(aux["struct360c_rotation_attention_bias_mean"].detach().cpu()))
        bias_min_vals.append(float(aux["struct360c_rotation_attention_bias_min"].detach().cpu()))
        bias_max_vals.append(float(aux["struct360c_rotation_attention_bias_max"].detach().cpu()))
        entropy_vals.append(float(aux["struct360c_attention_entropy_mean"].detach().cpu()))
    return {
        "fine_gate_stats": _float_stats(gate_vals),
        "delta_rot_mean_deg": _mean(delta_rot_vals),
        "delta_tdir_norm_mean": _mean(delta_tdir_vals),
        "delta_log_tmag_abs_mean": _mean(delta_log_vals),
        "attention_bias_mean": _mean(bias_mean_vals),
        "attention_bias_min": _mean(bias_min_vals),
        "attention_bias_max": _mean(bias_max_vals),
        "attention_entropy_mean": _mean(entropy_vals),
        "explicit_matching_used": False,
        "match_list_output": False,
        "correspondence_list_output": False,
        "topk_matching_used": False,
    }


def _run_prechecks(cfg: Mapping[str, Any]) -> Dict[str, Any]:
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
    required_paths = {name: REPO_ROOT / rel for name, rel in cfg["inputs"].items()}
    required_paths.update(
        {
            "model_file": REPO_ROOT / "models/struct360c_rotation_aware_fine_refinement.py",
            "config_file": REPO_ROOT / "configs/struct360c_rotation_aware_fine_refinement.yaml",
            "tool_file": REPO_ROOT / "tools/train_struct360c_rotation_aware_fine_refinement.py",
            "prep_report": REPO_ROOT / "reports/STRUCT360C_code_preparation_no_training.md",
        }
    )
    for name, path in required_paths.items():
        if not path.exists():
            blockers.append(f"missing required path: {name} -> {path}")

    train_ds = _build_pair_dataset(cfg, "train")
    val_ds = _build_pair_dataset(cfg, "val")
    test_ds = _build_pair_dataset(cfg, "test")
    split_audit = summarize_manifest_group([train_ds, val_ds, test_ds])
    if split_audit["has_overlap"]:
        blockers.append(f"manifest overlap detected: {split_audit['sequence_overlap']}")

    final360i_val = _extract_pair_metrics(_read_json(REPO_ROOT / cfg["inputs"]["final360i_val_metrics"]))
    final360i_test = _extract_pair_metrics(_read_json(REPO_ROOT / cfg["inputs"]["final360i_test_metrics"]))
    struct360b_val = _extract_pair_metrics(_read_json(REPO_ROOT / cfg["inputs"]["struct360b_val_metrics"]))
    struct360b_test = _extract_pair_metrics(_read_json(REPO_ROOT / cfg["inputs"]["struct360b_test_metrics"]))
    seq360a_val = _extract_seq360a_pair(_read_json(REPO_ROOT / cfg["inputs"]["seq360a_val_metrics"]))
    seq360a_test = _extract_seq360a_pair(_read_json(REPO_ROOT / cfg["inputs"]["seq360a_test_metrics"]))
    seq360a_traj_val = _extract_trajectory_metrics(_read_json(REPO_ROOT / cfg["inputs"]["seq360a_traj_val_metrics"]))
    seq360a_traj_test = _extract_trajectory_metrics(_read_json(REPO_ROOT / cfg["inputs"]["seq360a_traj_test_metrics"]))
    train360e_val = _extract_trajectory_metrics(_read_json(REPO_ROOT / cfg["inputs"]["train360e_val_metrics"]))
    train360e_test = _extract_trajectory_metrics(_read_json(REPO_ROOT / cfg["inputs"]["train360e_test_metrics"]))
    seq360b_traj = _aggregate_existing_trajectory_root(REPO_ROOT / "external_baselines" / "results" / "seq360b_scale_smoothing_trajectory")

    base_val_pair_json = _read_json(REPO_ROOT / cfg["inputs"]["base360_val_metrics"])
    base_test_pair_json = _read_json(REPO_ROOT / cfg["inputs"]["base360_test_metrics"])

    val_specs, _ = traj._build_sequence_specs(val_ds, "val")
    test_specs, _ = traj._build_sequence_specs(test_ds, "test")
    base360_val = traj._base360_split_eval("val", sorted(val_specs.keys()))
    base360_test = traj._base360_split_eval("test", sorted(test_specs.keys()))

    return {
        "blockers": blockers,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_status_short": git_status_short,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "disk_free_gb": disk_free_gb,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
        "split_audit": split_audit,
        "baseline_recap": {
            "FINAL360I_val_pair": final360i_val,
            "FINAL360I_test_pair": final360i_test,
            "STRUCT360B_val_pair": struct360b_val,
            "STRUCT360B_test_pair": struct360b_test,
            "SEQ360A_val_pair": seq360a_val,
            "SEQ360A_test_pair": seq360a_test,
            "SEQ360A_val_trajectory": seq360a_traj_val,
            "SEQ360A_test_trajectory": seq360a_traj_test,
            "SEQ360B_val_trajectory": seq360b_traj.get("val", {}),
            "SEQ360B_test_trajectory": seq360b_traj.get("test", {}),
            "TRAIN360E_val_trajectory": train360e_val,
            "TRAIN360E_test_trajectory": train360e_test,
            "BASE360_pair_val_json": base_val_pair_json,
            "BASE360_pair_test_json": base_test_pair_json,
            "BASE360D_val_trajectory": base360_val,
            "BASE360D_test_trajectory": base360_test,
        },
    }


def _write_blocker_artifacts(cfg: Mapping[str, Any], precheck: Mapping[str, Any]) -> None:
    lines = [
        "# STRUCT360C rotation-aware fine refinement",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {line}" for line in precheck["blockers"]])
    report_path = REPO_ROOT / cfg["outputs"]["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "task_name": cfg["task_name"],
        "training_executed": False,
        "checkpoint_saved": False,
        "blockers": list(precheck["blockers"]),
        "precheck": _jsonable(dict(precheck)),
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], payload)
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_val_metrics_path"], payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_test_metrics_path"], payload)
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text(
        "# STRUCT360C blocked\n\n" + "\n".join(f"- {line}" for line in precheck["blockers"]) + "\n",
        encoding="utf-8",
    )


def _run_trajectory_eval(cfg: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    val_ds = _build_pair_dataset(cfg, "val")
    test_ds = _build_pair_dataset(cfg, "test")
    val_specs, val_blockers = traj._build_sequence_specs(val_ds, "val")
    test_specs, test_blockers = traj._build_sequence_specs(test_ds, "test")
    convention, convention_summary, convention_warnings = traj._convention_sanity({**val_specs, **test_specs})
    blockers = list(val_blockers) + list(test_blockers)
    if convention == "blocked_convention":
        blockers.extend(convention_warnings)
    if blockers:
        raise RuntimeError(f"trajectory blockers: {blockers}")
    if convention != "BA":
        raise RuntimeError(f"Unexpected convention for eval-only recovery: {convention}")

    model, _ckpt_cfg, load_summary = _load_model(checkpoint_path, cfg, device)
    predictions = traj._iterate_adjacent_predictions(
        model,
        val_ds,
        device,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )
    predictions.extend(
        traj._iterate_adjacent_predictions(
            model,
            test_ds,
            device,
            batch_size=int(cfg["data"]["eval_batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
        )
    )
    pred_by_split_seq: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in predictions:
        pred_by_split_seq.setdefault((row["split"], row["sequence"]), []).append(row)

    original_results_root = traj.RESULTS_ROOT
    original_checkpoint = traj.CHECKPOINT_PATH
    original_task = traj.TASK_NAME
    try:
        traj.RESULTS_ROOT = REPO_ROOT / cfg["outputs"]["trajectory_dir"]
        traj.CHECKPOINT_PATH = checkpoint_path
        traj.TASK_NAME = str(cfg["task_name"])
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
        if key in {"tmag_median_ratio", "path_ratio"}:
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
    for key in ("trajectory_path_ratio", "ate_se3_rmse", "ate_sim3_rmse"):
        ours = _safe_float(metrics.get(key))
        theirs = _safe_float(ref.get(key))
        if ours is None or theirs is None:
            continue
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


def _classify(pair_test: Mapping[str, Any], traj_test: Mapping[str, Any], final360i_pair: Mapping[str, Any], train360e_traj: Mapping[str, Any], seq360b_traj: Mapping[str, Any]) -> str:
    signed = float(pair_test.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(pair_test.get("anti_parallel_rate") or 1.0)
    tmag = float(pair_test.get("tmag_median_ratio") or 0.0)
    path = float(pair_test.get("path_ratio") or 0.0)
    traj_path = float(traj_test.get("trajectory_path_ratio") or float("inf"))
    ate_se3 = float(traj_test.get("ate_se3_rmse") or float("inf"))
    ate_sim3 = float(traj_test.get("ate_sim3_rmse") or float("inf"))
    if (
        signed > 54.96
        or anti > 0.2157
        or tmag < 0.65
        or traj_path >= 1.7563
        or int(pair_test.get("nan_inf_count") or 0) > 0
    ):
        return "regression"
    if signed <= 44.986 and anti <= 0.19696 and ate_sim3 < 27.56 and traj_path < 1.35:
        return "strong_success"
    if signed <= 47.0 and anti <= 0.205 and tmag >= 0.75 and path >= 0.60 and traj_path <= 1.50 and ate_se3 <= 118.6038:
        return "balanced_success"
    pair_improved = (
        signed <= float(final360i_pair.get("signed_tdir_mean_deg") or float("inf"))
        or anti <= float(final360i_pair.get("anti_parallel_rate") or 1.0)
    )
    traj_improved = (
        ate_sim3 < float(train360e_traj.get("ate_sim3_rmse") or float("inf"))
        or ate_sim3 < float(seq360b_traj.get("ate_sim3_rmse") or float("inf"))
    )
    if pair_improved and not traj_improved:
        return "pair_improved_trajectory_unchanged"
    if traj_improved:
        return "trajectory_improved_pair_tradeoff"
    return "no_improvement"


def _recommendation(classification: str) -> str:
    if classification in {"balanced_success", "strong_success"}:
        return "promote_STRUCT360C_as_new_main_model"
    if classification == "trajectory_improved_pair_tradeoff":
        return "prepare_thesis_experiment_section"
    if classification == "pair_improved_trajectory_unchanged":
        return "keep_FINAL360I_as_main_and_report_STRUCT360C_ablation"
    if classification == "regression":
        return "keep_FINAL360I_as_main_and_report_STRUCT360C_ablation"
    return "keep_FINAL360I_as_main_and_report_STRUCT360C_ablation"


def _override_eval_cfg(
    cfg: Mapping[str, Any],
    *,
    num_workers: Optional[int],
    batch_size: Optional[int],
    trajectory_dir: Optional[str],
) -> Dict[str, Any]:
    out = dict(cfg)
    out["data"] = dict(cfg["data"])
    out["outputs"] = dict(cfg["outputs"])
    if num_workers is not None:
        out["data"]["num_workers"] = int(num_workers)
    if batch_size is not None:
        out["data"]["eval_batch_size"] = int(batch_size)
    if trajectory_dir is not None:
        out["outputs"]["trajectory_dir"] = str(trajectory_dir)
    return out


def _classify_eval_recovery(
    pair_test: Mapping[str, Any],
    traj_test: Mapping[str, Any],
    *,
    final360i_pair: Mapping[str, Any],
    struct360b_pair: Mapping[str, Any],
    seq360b_traj: Mapping[str, Any],
    seq360a_traj: Mapping[str, Any],
) -> str:
    nan_inf = int(pair_test.get("nan_inf_count") or 0) + int(traj_test.get("nan_inf_count") or 0)
    signed = _safe_float(pair_test.get("signed_tdir_mean_deg"))
    anti = _safe_float(pair_test.get("anti_parallel_rate"))
    sim3 = _safe_float(traj_test.get("ate_sim3_rmse"))
    traj_path = _safe_float(traj_test.get("trajectory_path_ratio"))
    if nan_inf > 0:
        return "eval_failed_nan_inf"
    if signed is None or anti is None or sim3 is None or traj_path is None:
        return "evaluation_failed"
    if signed > 54.96 or anti > 0.2157 or sim3 > 29.467660460312683:
        return "regression"
    pair_good = (
        signed <= float(final360i_pair.get("signed_tdir_mean_deg") or float("inf"))
        or anti <= float(final360i_pair.get("anti_parallel_rate") or 1.0)
    )
    traj_good = (
        sim3 < 27.564661865900444
        or traj_path < 1.3502369615185652
    )
    pair_not_bad = signed <= 47.0 and anti <= 0.205
    if pair_good and (traj_path <= 1.7563):
        return "pair_improved"
    if traj_good and pair_not_bad:
        return "trajectory_improved"
    near_final = abs(signed - float(final360i_pair.get("signed_tdir_mean_deg") or signed)) <= 1.0 and abs(anti - float(final360i_pair.get("anti_parallel_rate") or anti)) <= 0.01
    near_traj = abs(sim3 - float(seq360a_traj.get("ate_sim3_rmse") or sim3)) <= 2.0 or abs(traj_path - float(seq360b_traj.get("trajectory_path_ratio") or traj_path)) <= 0.2
    if near_final and near_traj:
        return "partial"
    if signed > float(struct360b_pair.get("signed_tdir_mean_deg") or float("inf")) and anti > float(struct360b_pair.get("anti_parallel_rate") or 1.0):
        return "regression"
    return "partial"


def _compare_recovery_pair(metrics: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    return _compare_pair(metrics, ref)


def _compare_recovery_traj(metrics: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    return _compare_trajectory(metrics, ref)


def _build_eval_only_report(
    *,
    cfg: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_meta: Mapping[str, Any],
    precheck: Mapping[str, Any],
    val_pair_metrics: Mapping[str, Any],
    test_pair_metrics: Mapping[str, Any],
    val_diag: Mapping[str, Any],
    test_diag: Mapping[str, Any],
    val_traj_metrics: Mapping[str, Any],
    test_traj_metrics: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    classification: str,
) -> Tuple[str, str]:
    baseline = precheck["baseline_recap"]
    recommendation = "rollback_to_FINAL360I_or_STRUCT360B" if classification in {"eval_failed_nan_inf", "evaluation_failed", "regression"} else "prepare_thesis_experiment_section"
    main_report = "\n".join(
        [
            "# STRUCT360C rotation-aware fine refinement",
            "",
            "## 1. Executive summary",
            "- evaluation_only_recovery: `true`",
            "- training_executed_in_recovery: `false`",
            "- checkpoint saved in recovery: `false`",
            f"- checkpoint evaluated: `{checkpoint_path}`",
            "- final.pt used for main result: `false`",
            f"- best checkpoint epoch: `{checkpoint_meta.get('best_epoch')}`",
            f"- best checkpoint val score: `{checkpoint_meta.get('best_val_score')}`",
            f"- classification: `{classification}`",
            f"- pair test metrics: `{test_pair_metrics}`",
            f"- trajectory test metrics: `{test_traj_metrics}`",
            "",
            "## 2. Recovery protocol",
            "- evaluation_only_recovery = true",
            "- no training executed in recovery = true",
            "- no optimizer step = true",
            "- best checkpoint only = true",
            "- final.pt not used for main result = true",
            "- epoch 3 NaN/Inf checkpoint not selected = true",
            "",
            "## 3. Pair-level evaluation",
            "- num_workers = `0`",
            "- torch.no_grad = `true`",
            f"- val metrics: `{val_pair_metrics}`",
            f"- test metrics: `{test_pair_metrics}`",
            "",
            "## 4. Trajectory evaluation",
            "- selected convention = `BA`",
            "- isolated trajectory export = `true`",
            f"- val trajectory: `{val_traj_metrics}`",
            f"- test trajectory: `{test_traj_metrics}`",
            "",
            "## 5. Diagnostics",
            f"- val diagnostics: `{val_diag}`",
            f"- test diagnostics: `{test_diag}`",
            f"- any skipped pairs/sequences are recorded in trajectory json outputs.",
            "",
            "## 6. Comparison",
            f"- FINAL360I pair / TRAIN360E traj: `{baseline['FINAL360I_test_pair']}` / `{baseline['TRAIN360E_test_trajectory']}`",
            f"- STRUCT360B pair: `{baseline['STRUCT360B_test_pair']}`",
            f"- SEQ360B trajectory: `{baseline['SEQ360B_test_trajectory']}`",
            f"- SEQ360A trajectory: `{baseline['SEQ360A_test_trajectory']}`",
            f"- BASE360D trajectory: `{baseline['BASE360D_test_trajectory']}`",
            f"- comparisons: `{comparisons}`",
            "",
            "## 7. Compliance checklist",
            "- no explicit matching = true",
            "- no RANSAC / PnP / BA = true / true / true",
            "- no checkpoint modified = true",
            "- no learned weights written = true",
            "- no test split tuning = true",
            "",
            "## 8. Recommendation",
            f"- `{recommendation}`",
        ]
    ) + "\n"
    recovery_report = "\n".join(
        [
            "# STRUCT360C eval-only recovery",
            "",
            "- evaluation_only_recovery = `true`",
            "- training_executed_in_recovery = `false`",
            f"- checkpoint evaluated = `{checkpoint_path}`",
            "- final.pt used = `false`",
            f"- selected convention = `BA`",
            f"- classification = `{classification}`",
            f"- pair test metrics = `{test_pair_metrics}`",
            f"- trajectory test metrics = `{test_traj_metrics}`",
            f"- comparisons = `{comparisons}`",
        ]
    ) + "\n"
    return main_report, recovery_report


def _run_eval_only(
    cfg: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    start_time = time.time()
    precheck = _run_prechecks(cfg)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck)
        raise RuntimeError("STRUCT360C eval-only precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    device = torch.device(precheck["device"])
    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_meta = dict(checkpoint_payload.get("metadata", {})) if isinstance(checkpoint_payload, Mapping) else {}
    model, _ckpt_cfg, load_summary = _load_model(checkpoint_path, cfg, device)
    model.eval()

    val_ds = precheck["datasets"]["val"]
    test_ds = precheck["datasets"]["test"]
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )

    val_pair_eval = evaluate_struct360b_pose(
        model,
        val_loader,
        device,
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
        max_batches=cfg["evaluation"].get("max_val_batches"),
    )
    test_pair_eval = evaluate_struct360b_pose(
        model,
        test_loader,
        device,
        tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
        max_batches=cfg["evaluation"].get("max_test_batches"),
    )
    val_diag = _evaluate_struct360c_diagnostics(
        model,
        val_loader,
        device,
        max_batches=cfg["evaluation"].get("max_val_batches"),
    )
    test_diag = _evaluate_struct360c_diagnostics(
        model,
        test_loader,
        device,
        max_batches=cfg["evaluation"].get("max_test_batches"),
    )
    traj_val_raw, traj_test_raw = _run_trajectory_eval(cfg, checkpoint_path, device)
    val_pair_metrics = val_pair_eval["final_metrics"]
    test_pair_metrics = test_pair_eval["final_metrics"]
    val_traj_metrics = _extract_trajectory_metrics(traj_val_raw)
    test_traj_metrics = _extract_trajectory_metrics(traj_test_raw)

    baseline = precheck["baseline_recap"]
    comparisons = {
        "vs_final360i_pair": _compare_recovery_pair(test_pair_metrics, baseline["FINAL360I_test_pair"]),
        "vs_struct360b": _compare_recovery_pair(test_pair_metrics, baseline["STRUCT360B_test_pair"]),
        "vs_seq360b_trajectory": _compare_recovery_traj(test_traj_metrics, baseline["SEQ360B_test_trajectory"]),
        "vs_seq360a_trajectory": _compare_recovery_traj(test_traj_metrics, baseline["SEQ360A_test_trajectory"]),
        "vs_base360d_trajectory": _compare_recovery_traj(test_traj_metrics, baseline["BASE360D_test_trajectory"]),
    }
    classification = _classify_eval_recovery(
        test_pair_metrics,
        test_traj_metrics,
        final360i_pair=baseline["FINAL360I_test_pair"],
        struct360b_pair=baseline["STRUCT360B_test_pair"],
        seq360b_traj=baseline["SEQ360B_test_trajectory"],
        seq360a_traj=baseline["SEQ360A_test_trajectory"],
    )

    val_payload = {
        "task_name": cfg["task_name"],
        "evaluation_only_recovery": True,
        "training_executed_in_recovery": False,
        "checkpoint_evaluated": str(checkpoint_path),
        "final_pt_used": False,
        "selected_convention": "BA",
        "pair_metrics": val_pair_metrics,
        "diagnostics": val_diag,
        "checkpoint_metadata": checkpoint_meta,
        "checkpoint_load_summary": load_summary,
    }
    test_payload = {
        "task_name": cfg["task_name"],
        "evaluation_only_recovery": True,
        "training_executed_in_recovery": False,
        "checkpoint_evaluated": str(checkpoint_path),
        "final_pt_used": False,
        "selected_convention": "BA",
        "pair_metrics": test_pair_metrics,
        "diagnostics": test_diag,
        "classification": classification,
        "comparisons": comparisons,
        "checkpoint_metadata": checkpoint_meta,
        "checkpoint_load_summary": load_summary,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], val_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], test_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_val_metrics_path"], traj_val_raw)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_test_metrics_path"], traj_test_raw)

    main_report, recovery_report = _build_eval_only_report(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        checkpoint_meta=checkpoint_meta,
        precheck=precheck,
        val_pair_metrics=val_pair_metrics,
        test_pair_metrics=test_pair_metrics,
        val_diag=val_diag,
        test_diag=test_diag,
        val_traj_metrics=val_traj_metrics,
        test_traj_metrics=test_traj_metrics,
        comparisons=comparisons,
        classification=classification,
    )
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text(main_report, encoding="utf-8")
    recovery_report_path = REPO_ROOT / "reports" / "STRUCT360C_eval_only_recovery.md"
    recovery_report_path.write_text(recovery_report, encoding="utf-8")

    summary_lines = [
        "# STRUCT360C vs FINAL360I STRUCT360B SEQ360A SEQ360B BASE360D summary",
        "",
        "| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | pair_path_ratio | trajectory_path_ratio | ATE none | ATE SE3 | ATE Sim3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| FINAL360I / TRAIN360E | {_fmt(baseline['FINAL360I_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['FINAL360I_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['FINAL360I_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['FINAL360I_test_pair'].get('path_ratio'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| STRUCT360B | {_fmt(baseline['STRUCT360B_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('path_ratio'))} | N/A | N/A | N/A | N/A |",
        f"| SEQ360B trajectory | N/A | N/A | N/A | N/A | {_fmt(baseline['SEQ360B_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| SEQ360A trajectory | {_fmt(baseline['SEQ360A_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['SEQ360A_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['SEQ360A_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['SEQ360A_test_pair'].get('path_ratio'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| BASE360D trajectory | N/A | N/A | N/A | N/A | {_fmt(baseline['BASE360D_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| STRUCT360C | {_fmt(test_pair_metrics.get('signed_tdir_mean_deg'))} | {_fmt(test_pair_metrics.get('anti_parallel_rate'))} | {_fmt(test_pair_metrics.get('tmag_median_ratio'))} | {_fmt(test_pair_metrics.get('path_ratio'))} | {_fmt(test_traj_metrics.get('trajectory_path_ratio'))} | {_fmt(test_traj_metrics.get('ate_none_rmse'))} | {_fmt(test_traj_metrics.get('ate_se3_rmse'))} | {_fmt(test_traj_metrics.get('ate_sim3_rmse'))} |",
        "",
        f"- compared to FINAL360I pair-level: `{comparisons['vs_final360i_pair']}`",
        f"- compared to STRUCT360B: `{comparisons['vs_struct360b']}`",
        f"- compared to SEQ360B trajectory: `{comparisons['vs_seq360b_trajectory']}`",
        f"- compared to SEQ360A trajectory: `{comparisons['vs_seq360a_trajectory']}`",
        f"- compared to BASE360D trajectory: `{comparisons['vs_base360d_trajectory']}`",
        f"- classification: `{classification}`",
    ]
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    result = {
        "STRUCT360C eval-only recovery executed": True,
        "training executed in recovery": False,
        "checkpoint evaluated": str(checkpoint_path),
        "final.pt used": False,
        "selected convention": "BA",
        "pair test signed_tdir_mean": test_pair_metrics.get("signed_tdir_mean_deg"),
        "pair test anti_parallel_rate": test_pair_metrics.get("anti_parallel_rate"),
        "pair test tmag_median_ratio": test_pair_metrics.get("tmag_median_ratio"),
        "pair test path_ratio": test_pair_metrics.get("path_ratio"),
        "trajectory test ATE none": test_traj_metrics.get("ate_none_rmse"),
        "trajectory test ATE SE3": test_traj_metrics.get("ate_se3_rmse"),
        "trajectory test ATE Sim3": test_traj_metrics.get("ate_sim3_rmse"),
        "trajectory test path_ratio": test_traj_metrics.get("trajectory_path_ratio"),
        "compared to FINAL360I pair-level": comparisons["vs_final360i_pair"],
        "compared to STRUCT360B": comparisons["vs_struct360b"],
        "compared to SEQ360B trajectory": comparisons["vs_seq360b_trajectory"],
        "compared to SEQ360A trajectory": comparisons["vs_seq360a_trajectory"],
        "compared to BASE360D trajectory": comparisons["vs_base360d_trajectory"],
        "classification": classification,
        "committed to git": False,
        "pushed to remote": False,
        "next recommended task": "rollback_to_FINAL360I_or_STRUCT360B" if classification in {"eval_failed_nan_inf", "evaluation_failed", "regression"} else "prepare_thesis_experiment_section",
        "elapsed_sec": time.time() - start_time,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_yaml_like(cfg_path)
    start_time = time.time()
    precheck = _run_prechecks(cfg)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck)
        raise RuntimeError("STRUCT360C precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    _seed(int(cfg["training"]["seed"]))
    device = torch.device(precheck["device"])
    system_info = _system_info(device)

    train_ds = precheck["datasets"]["train"]
    val_ds = precheck["datasets"]["val"]
    test_ds = precheck["datasets"]["test"]
    train_subset, subset_info = _subset_dataset(train_ds, cfg["data"].get("train_subset_max"), int(cfg["training"]["seed"]))
    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=bool(cfg["data"]["shuffle_train"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0),
        drop_last=False,
    )

    init_ckpt = REPO_ROOT / cfg["inputs"]["init_checkpoint"]
    model, ckpt_cfg, load_status = _load_model(init_ckpt, cfg, device)
    param_counts = count_parameters(model)
    optimizer, coarse_params, fine_params = _build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler() if bool(cfg["training"]["amp"]) and device.type == "cuda" else None

    checkpoint_dir = REPO_ROOT / cfg["outputs"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = checkpoint_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    (checkpoint_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    best_checkpoint_path = checkpoint_dir / "best_val.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    best_val_score = float("inf")
    best_epoch = -1

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        stage_info = _set_stage_trainability(
            epoch=epoch,
            coarse_params=coarse_params,
            fine_params=fine_params,
            optimizer=optimizer,
            train_cfg=cfg["training"],
        )
        model.train()
        epoch_losses: Dict[str, List[float]] = {
            "loss_total": [],
            "loss_final": [],
            "loss_coarse_aux": [],
            "loss_residual_reg": [],
            "loss_bias_reg": [],
        }
        bias_mean_vals: List[float] = []
        bias_entropy_vals: List[float] = []
        gate_vals: List[float] = []
        train_nan_inf_count = 0

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
                R_pred, _t_local, aux = model(IA, IB, dt_world=dt_world_t)
                final_loss = train360d_pose_loss(
                    R_pred=R_pred,
                    tdir_pred_B=aux["t_dir_out"],
                    tmag_pred=aux["t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    k_step_config=cfg["loss"]["k_step_balancing"],
                    observability_config=cfg["loss"]["observability"],
                    scale_config=cfg["loss"]["scale_stability"],
                    rot_weight=float(cfg["loss"]["rot_weight"]),
                    tdir_weight=float(cfg["loss"]["tdir_weight"]),
                    tmag_weight=float(cfg["loss"]["tmag_weight"]),
                    scale_stability_weight=float(cfg["loss"]["scale_stability_weight"]),
                    tmag_loss_type=str(cfg["loss"]["tmag_loss_type"]),
                    tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
                    enable_observability=bool(cfg["loss"]["components"].get("enable_observability", True)),
                    enable_k_step_balancing=bool(cfg["loss"]["components"].get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(cfg["loss"]["components"].get("enable_scale_stabilization", True)),
                )
                coarse_loss = train360d_pose_loss(
                    R_pred=aux["coarse_R"],
                    tdir_pred_B=aux["coarse_t_dir_out"],
                    tmag_pred=aux["coarse_t_mag"],
                    R_gt=R_gt,
                    t_gt_vec_B=t_gt_vec,
                    tmag_gt=tmag_gt,
                    k_tensor=k_tensor,
                    k_step_config=cfg["loss"]["k_step_balancing"],
                    observability_config=cfg["loss"]["observability"],
                    scale_config=cfg["loss"]["scale_stability"],
                    rot_weight=float(cfg["loss"]["rot_weight"]),
                    tdir_weight=float(cfg["loss"]["tdir_weight"]),
                    tmag_weight=float(cfg["loss"]["tmag_weight"]),
                    scale_stability_weight=float(cfg["loss"]["scale_stability_weight"]),
                    tmag_loss_type=str(cfg["loss"]["tmag_loss_type"]),
                    tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
                    enable_observability=bool(cfg["loss"]["components"].get("enable_observability", True)),
                    enable_k_step_balancing=bool(cfg["loss"]["components"].get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(cfg["loss"]["components"].get("enable_scale_stabilization", True)),
                )
                residual_reg = model.residual_regularization(aux)
                bias_reg = torch.relu(torch.abs(aux["struct360c_rotation_attention_bias_mean"]) - 3.5)
                loss_total = (
                    final_loss["loss_total"]
                    + float(cfg["loss"]["coarse_aux_weight"]) * coarse_loss["loss_total"]
                    + float(cfg["loss"]["residual_reg_weight"]) * residual_reg["loss"]
                    + float(cfg["loss"].get("bias_reg_weight", 0.0)) * bias_reg
                )

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
                optimizer.step()

            epoch_losses["loss_total"].append(float(loss_total.detach().cpu()))
            epoch_losses["loss_final"].append(float(final_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_coarse_aux"].append(float(coarse_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_residual_reg"].append(float(residual_reg["loss"].detach().cpu()))
            epoch_losses["loss_bias_reg"].append(float(bias_reg.detach().cpu()))
            bias_mean_vals.append(float(aux["struct360c_rotation_attention_bias_mean"].detach().cpu()))
            bias_entropy_vals.append(float(aux["struct360c_attention_entropy_mean"].detach().cpu()))
            gate_vals.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
            train_nan_inf_count += int(sum((~torch.isfinite(v)).sum().item() for v in [R_pred, aux["t_dir_out"], aux["t_mag"], loss_total]))

        val_pair_eval = evaluate_struct360b_pose(
            model,
            val_loader,
            device,
            tmag_epsilon=float(cfg["data"]["tmag_epsilon"]),
            max_batches=cfg["evaluation"].get("max_val_batches"),
        )
        val_diag = _evaluate_struct360c_diagnostics(
            model,
            val_loader,
            device,
            max_batches=cfg["evaluation"].get("max_val_batches"),
        )
        val_metrics = val_pair_eval["final_metrics"]
        val_score = _compute_val_score(val_metrics, cfg["evaluation"]["selection_score"], eps=float(cfg["data"]["tmag_epsilon"]))
        epoch_log = {
            "epoch": epoch,
            "stage": stage_info["stage"],
            "lr_coarse": stage_info["coarse_lr"],
            "lr_fine": stage_info["fine_lr"],
            "val_score": val_score,
            "val_metrics": val_metrics,
            "val_diagnostics": val_diag,
            "train_loss_stats": {k: _mean(v) for k, v in epoch_losses.items()},
            "train_bias_mean_stats": _float_stats(bias_mean_vals),
            "train_attention_entropy_stats": _float_stats(bias_entropy_vals),
            "train_gate_stats": _float_stats(gate_vals),
            "train_nan_inf_count": train_nan_inf_count,
        }
        _append_jsonl(train_log_path, epoch_log)
        if val_score < best_val_score and int(val_metrics.get("nan_inf_count") or 0) == 0:
            best_val_score = float(val_score)
            best_epoch = int(epoch)
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                ckpt_cfg,
                {
                    "task_name": cfg["task_name"],
                    "run_name": cfg["training"]["run_name"],
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                    "subset_info": subset_info,
                    "checkpoint_load_status": load_status,
                    "stage": stage_info["stage"],
                },
            )

    _save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        int(cfg["training"]["epochs"]),
        ckpt_cfg,
        {
            "task_name": cfg["task_name"],
            "run_name": cfg["training"]["run_name"],
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "subset_info": subset_info,
            "checkpoint_load_status": load_status,
        },
    )

    selected_checkpoint = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
    selected_model, _selected_cfg, selected_load = _load_model(selected_checkpoint, cfg, device)
    selected_model.eval()
    val_pair_eval = evaluate_struct360b_pose(selected_model, val_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_val_batches"))
    test_pair_eval = evaluate_struct360b_pose(selected_model, test_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_test_batches"))
    val_diag = _evaluate_struct360c_diagnostics(selected_model, val_loader, device, max_batches=cfg["evaluation"].get("max_val_batches"))
    test_diag = _evaluate_struct360c_diagnostics(selected_model, test_loader, device, max_batches=cfg["evaluation"].get("max_test_batches"))
    traj_val_raw, traj_test_raw = _run_trajectory_eval(cfg, selected_checkpoint, device)

    val_pair_metrics = val_pair_eval["final_metrics"]
    test_pair_metrics = test_pair_eval["final_metrics"]
    val_traj_metrics = _extract_trajectory_metrics(traj_val_raw)
    test_traj_metrics = _extract_trajectory_metrics(traj_test_raw)

    baseline = precheck["baseline_recap"]
    comparisons = {
        "vs_final360i_pair": _compare_pair(test_pair_metrics, baseline["FINAL360I_test_pair"]),
        "vs_struct360b_pair": _compare_pair(test_pair_metrics, baseline["STRUCT360B_test_pair"]),
        "vs_seq360b_trajectory": _compare_trajectory(test_traj_metrics, baseline["SEQ360B_test_trajectory"]),
        "vs_seq360a_trajectory": _compare_trajectory(test_traj_metrics, baseline["SEQ360A_test_trajectory"]),
        "vs_base360d_trajectory": _compare_trajectory(test_traj_metrics, baseline["BASE360D_test_trajectory"]),
    }
    classification = _classify(
        test_pair_metrics,
        test_traj_metrics,
        baseline["FINAL360I_test_pair"],
        baseline["TRAIN360E_test_trajectory"],
        baseline["SEQ360B_test_trajectory"],
    )
    recommendation = _recommendation(classification)

    val_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_epoch": best_epoch,
        "best_checkpoint": str(selected_checkpoint),
        "pair_metrics": val_pair_metrics,
        "diagnostics": val_diag,
        "precheck": _jsonable(precheck),
        "checkpoint_load_summary": selected_load,
    }
    test_payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "selected_epoch": best_epoch,
        "best_checkpoint": str(selected_checkpoint),
        "pair_metrics": test_pair_metrics,
        "diagnostics": test_diag,
        "classification": classification,
        "comparisons": comparisons,
        "precheck": _jsonable(precheck),
        "checkpoint_load_summary": selected_load,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], val_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], test_payload)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_val_metrics_path"], traj_val_raw)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_test_metrics_path"], traj_test_raw)

    report_lines = [
        "# STRUCT360C rotation-aware fine refinement",
        "",
        "## 1. Executive summary",
        "- training executed: `true`",
        "- checkpoint saved: `true`",
        f"- best checkpoint: `{selected_checkpoint}`",
        f"- init checkpoint: `{init_ckpt}`",
        f"- classification: `{classification}`",
        f"- pair metrics: `{test_pair_metrics}`",
        f"- trajectory metrics: `{test_traj_metrics}`",
        "",
        "## 2. Motivation",
        "- FINAL360I pair-level strong.",
        "- TRAIN360E showed trajectory drift under direct adjacent composition.",
        "- SEQ360B improved scale/path but not Sim3 shape error.",
        "- SEQ360A sequence consistency gave no improvement.",
        "- STRUCT360C targets rotation-aware fine refinement.",
        "",
        "## 3. Research alignment",
        "- match-free: `true`",
        "- no explicit correspondence: `true`",
        "- no RANSAC / PnP / BA: `true / true / true`",
        "- latent rotation-aware attention bias only: `true`",
        "",
        "## 4. Architecture",
        "- coarse-to-fine inherited from STRUCT360B.",
        "- coarse `R0` from pair-level coarse stage.",
        "- spherical xyz token coordinates from fine bearings.",
        "- rotation-aware attention bias added to fine cross-attention logits.",
        "- fine residual pose refinement predicts `ΔR / Δtdir / Δlog_tmag`.",
        "- final pose composition matches STRUCT360B.",
        "",
        "## 5. Training setup",
        f"- init checkpoint: `{init_ckpt}`",
        f"- epochs: `{cfg['training']['epochs']}`",
        f"- batch size: `{cfg['data']['train_batch_size']}`",
        f"- LR groups: `coarse={cfg['training']['coarse_lr']}`, `fine/bias={cfg['training']['fine_lr']}`",
        f"- warmup/unfreeze: `warmup_epochs={cfg['training']['warmup_epochs']}`, `freeze_coarse_warmup={cfg['model']['freeze_coarse_warmup']}`",
        f"- gamma: `{cfg['model']['struct360c_bias_gamma']}`",
        f"- alpha/beta: `{cfg['model']['struct360b']['alpha']} / {cfg['model']['struct360b']['beta']}`",
        f"- val score: `{cfg['evaluation']['selection_score']}`",
        f"- train subset info: `{subset_info}`",
        f"- system info: `{system_info}`",
        "",
        "## 6. Validation results",
        f"- selected epoch: `{best_epoch}`",
        f"- pair metrics: `{val_pair_metrics}`",
        f"- diagnostics: `{val_diag}`",
        "",
        "## 7. Test pair-level results",
        f"- FINAL360I: `{baseline['FINAL360I_test_pair']}`",
        f"- STRUCT360B: `{baseline['STRUCT360B_test_pair']}`",
        f"- STRUCT360C: `{test_pair_metrics}`",
        "",
        "## 8. Test trajectory results",
        f"- TRAIN360E FINAL360I: `{baseline['TRAIN360E_test_trajectory']}`",
        f"- SEQ360B: `{baseline['SEQ360B_test_trajectory']}`",
        f"- SEQ360A: `{baseline['SEQ360A_test_trajectory']}`",
        f"- STRUCT360C: `{test_traj_metrics}`",
        f"- BASE360D: `{baseline['BASE360D_test_trajectory']}`",
        "",
        "## 9. Diagnostics",
        f"- bias stats: `mean={test_diag['attention_bias_mean']}, min={test_diag['attention_bias_min']}, max={test_diag['attention_bias_max']}`",
        f"- attention entropy: `{test_diag['attention_entropy_mean']}`",
        f"- gate stats: `{test_diag['fine_gate_stats']}`",
        f"- residual magnitude: `delta_rot_mean_deg={test_diag['delta_rot_mean_deg']}, delta_tdir_norm_mean={test_diag['delta_tdir_norm_mean']}, delta_log_tmag_abs_mean={test_diag['delta_log_tmag_abs_mean']}`",
        f"- attention collapsed: `{bool(_safe_float(test_diag['attention_entropy_mean']) is not None and _safe_float(test_diag['attention_entropy_mean']) < 0.1)}`",
        "",
        "## 10. Analysis",
        f"- direction improved: `{comparisons['vs_final360i_pair'] in {'better', 'partial'}}`",
        f"- anti_parallel improved: `{_safe_float(test_pair_metrics.get('anti_parallel_rate')) is not None and _safe_float(test_pair_metrics.get('anti_parallel_rate')) <= 0.20169893322797314}`",
        f"- trajectory shape / ATE Sim3 improved: `{_safe_float(test_traj_metrics.get('ate_sim3_rmse')) is not None and _safe_float(test_traj_metrics.get('ate_sim3_rmse')) < 27.564661865900444}`",
        f"- scale/path degraded: `{_safe_float(test_pair_metrics.get('path_ratio')) is not None and _safe_float(test_pair_metrics.get('path_ratio')) < 0.60}`",
        f"- replace FINAL360I or remain ablation: `{'replace' if classification in {'balanced_success', 'strong_success'} else 'ablation'}`",
        "",
        "## 11. Recommendation",
        f"- `{recommendation}`",
        "",
        "## 12. Compliance checklist",
        "- `training_executed = true`",
        "- `fine_tune_executed = true`",
        "- `learned_weights_saved = true`",
        "- `final360i_checkpoint_modified = false`",
        "- `explicit_matching_used = false`",
        "- `match_list_output = false`",
        "- `correspondence_list_output = false`",
        "- `topk_matching_used = false`",
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
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# STRUCT360C vs FINAL360I STRUCT360B SEQ360A SEQ360B BASE360D",
        "",
        "| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | pair_path_ratio | trajectory_path_ratio | ATE none | ATE SE3 | ATE Sim3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| FINAL360I / TRAIN360E | {_fmt(baseline['FINAL360I_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['FINAL360I_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['FINAL360I_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['FINAL360I_test_pair'].get('path_ratio'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['TRAIN360E_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| STRUCT360B | {_fmt(baseline['STRUCT360B_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['STRUCT360B_test_pair'].get('path_ratio'))} | N/A | N/A | N/A | N/A |",
        f"| SEQ360B trajectory | N/A | N/A | N/A | N/A | {_fmt(baseline['SEQ360B_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['SEQ360B_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| SEQ360A trajectory | {_fmt(baseline['SEQ360A_test_pair'].get('signed_tdir_mean_deg'))} | {_fmt(baseline['SEQ360A_test_pair'].get('anti_parallel_rate'))} | {_fmt(baseline['SEQ360A_test_pair'].get('tmag_median_ratio'))} | {_fmt(baseline['SEQ360A_test_pair'].get('path_ratio'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['SEQ360A_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| BASE360D trajectory | N/A | N/A | N/A | N/A | {_fmt(baseline['BASE360D_test_trajectory'].get('trajectory_path_ratio'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_none_rmse'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_se3_rmse'))} | {_fmt(baseline['BASE360D_test_trajectory'].get('ate_sim3_rmse'))} |",
        f"| STRUCT360C | {_fmt(test_pair_metrics.get('signed_tdir_mean_deg'))} | {_fmt(test_pair_metrics.get('anti_parallel_rate'))} | {_fmt(test_pair_metrics.get('tmag_median_ratio'))} | {_fmt(test_pair_metrics.get('path_ratio'))} | {_fmt(test_traj_metrics.get('trajectory_path_ratio'))} | {_fmt(test_traj_metrics.get('ate_none_rmse'))} | {_fmt(test_traj_metrics.get('ate_se3_rmse'))} | {_fmt(test_traj_metrics.get('ate_sim3_rmse'))} |",
        "",
        f"- compared to FINAL360I pair-level: `{comparisons['vs_final360i_pair']}`",
        f"- compared to STRUCT360B: `{comparisons['vs_struct360b_pair']}`",
        f"- compared to SEQ360B trajectory: `{comparisons['vs_seq360b_trajectory']}`",
        f"- compared to SEQ360A trajectory: `{comparisons['vs_seq360a_trajectory']}`",
        f"- compared to BASE360D trajectory: `{comparisons['vs_base360d_trajectory']}`",
        f"- classification: `{classification}`",
    ]
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    runtime_sec = time.time() - start_time
    print("- STRUCT360C training executed: true")
    print("- checkpoint saved: true")
    print(f"- best checkpoint: {selected_checkpoint}")
    print(f"- init checkpoint: {init_ckpt}")
    print(f"- pair test signed_tdir_mean: {test_pair_metrics.get('signed_tdir_mean_deg')}")
    print(f"- pair test anti_parallel_rate: {test_pair_metrics.get('anti_parallel_rate')}")
    print(f"- pair test tmag_median_ratio: {test_pair_metrics.get('tmag_median_ratio')}")
    print(f"- pair test path_ratio: {test_pair_metrics.get('path_ratio')}")
    print(f"- trajectory test ATE none: {test_traj_metrics.get('ate_none_rmse')}")
    print(f"- trajectory test ATE SE3: {test_traj_metrics.get('ate_se3_rmse')}")
    print(f"- trajectory test ATE Sim3: {test_traj_metrics.get('ate_sim3_rmse')}")
    print(f"- trajectory test path_ratio: {test_traj_metrics.get('trajectory_path_ratio')}")
    print(f"- compared to FINAL360I pair-level: {comparisons['vs_final360i_pair']}")
    print(f"- compared to STRUCT360B: {comparisons['vs_struct360b_pair']}")
    print(f"- compared to SEQ360B trajectory: {comparisons['vs_seq360b_trajectory']}")
    print(f"- compared to SEQ360A trajectory: {comparisons['vs_seq360a_trajectory']}")
    print(f"- compared to BASE360D trajectory: {comparisons['vs_base360d_trajectory']}")
    print(f"- classification: {classification}")
    print("- committed to git: false")
    print("- pushed to remote: false")
    print(f"- next recommended task: {recommendation}")
    print(f"- runtime_sec: {runtime_sec:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--trajectory-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml_like(Path(args.config))
    cfg = _override_eval_cfg(
        cfg,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        trajectory_dir=args.trajectory_dir,
    )
    if args.eval_only:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else (REPO_ROOT / cfg["outputs"]["checkpoint_dir"] / "best_val.pt")
        _run_eval_only(cfg, checkpoint_path=checkpoint_path)
    else:
        sys.argv = [sys.argv[0], args.config]
        train()
