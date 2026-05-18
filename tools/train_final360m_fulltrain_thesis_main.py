#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel, count_parameters
from train360.core.train360d_pose_losses import train360d_pose_loss
from train_struct360b_match_free_coarse_to_fine import (
    T57B_REFERENCE,
    _append_jsonl,
    _build_optimizer,
    _cfg_from_dict,
    _compute_val_score,
    _extract_base360_metrics,
    _float_stats,
    _git,
    _inject_struct360b_cfg,
    _json_dump,
    _load_model_with_status,
    _read_json,
    _seed_everything,
    _set_stage_trainability,
    _system_info,
    evaluate_struct360b_pose,
)


def load_yaml_like(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


EXPECTED_OLD_FINAL360I_TEST = {
    "rot_mean_deg": 2.330108616583517,
    "signed_tdir_mean_deg": 45.264702006380205,
    "anti_parallel_rate": 0.20169893322797314,
    "tmag_median_ratio": 0.8344251368086006,
    "path_ratio": 0.6403519796204528,
}


def _manifest_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _safe_metric_compare(new_value: Any, old_value: Any, *, lower_is_better: bool) -> str:
    if new_value is None or old_value is None:
        return "missing"
    if lower_is_better:
        if float(new_value) < float(old_value):
            return "improved"
        if float(new_value) > float(old_value):
            return "worse"
        return "same"
    if float(new_value) > float(old_value):
        return "improved"
    if float(new_value) < float(old_value):
        return "worse"
    return "same"


def _overall_vs_old(metric_status: Mapping[str, str]) -> str:
    improved = sum(1 for value in metric_status.values() if value == "improved")
    worse = sum(1 for value in metric_status.values() if value == "worse")
    if improved >= 3 and worse == 0:
        return "better"
    if worse >= 3 and improved == 0:
        return "worse"
    return "mixed"


def _extract_final360i_old_test_metrics(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = cfg["inputs"]
    old_test = _read_json(REPO_ROOT / inputs["old_final360i_metrics_test"])
    metrics = old_test.get("metrics") if isinstance(old_test.get("metrics"), Mapping) else None
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return dict(EXPECTED_OLD_FINAL360I_TEST)


def _baseline_from_final360i_table(cfg: Mapping[str, Any], name: str) -> Optional[Dict[str, Any]]:
    table = _read_json(REPO_ROOT / cfg["inputs"]["old_final360i_model_selection_table"])
    for item in table.get("baseline_recap", []):
        if str(item.get("name")) == name and isinstance(item.get("metrics"), Mapping):
            return dict(item["metrics"])
    return None


def _build_mini_val_subset(dataset: Dset2CCanonicalPairDataset, *, count: int, seed: int) -> Tuple[Subset, Dict[str, Any]]:
    if len(dataset) < count:
        raise RuntimeError(f"mini-val count {count} exceeds val dataset size {len(dataset)}")
    rng = np.random.default_rng(int(seed))
    idx = sorted(int(x) for x in rng.permutation(len(dataset))[: int(count)].tolist())
    return Subset(dataset, idx), {"subset_count": len(idx), "subset_seed": int(seed)}


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_cfg: Any,
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "cfg": dict(ckpt_cfg.__dict__),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _load_resume_checkpoint(
    last_checkpoint_path: Path,
    *,
    device: torch.device,
) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Any, Dict[str, Any], int]:
    payload = torch.load(str(last_checkpoint_path), map_location=device)
    ckpt_cfg = _cfg_from_dict(dict(payload["cfg"]))
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    metadata = dict(payload.get("metadata") or {})
    epoch = int(payload.get("epoch") or metadata.get("epoch") or 0)
    return model, ckpt_cfg, metadata, epoch


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    loader: DataLoader,
    device: torch.device,
    tmag_epsilon: float,
) -> Dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    ckpt_cfg = _cfg_from_dict(dict(payload["cfg"]))
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return evaluate_struct360b_pose(model, loader, device, tmag_epsilon=tmag_epsilon, max_batches=None)


def _write_blocker(cfg: Mapping[str, Any], blocker_lines: Sequence[str], *, classification: str = "blocked") -> None:
    outputs = cfg["outputs"]
    report_path = REPO_ROOT / outputs["blocker_or_partial_path"]
    payload = {
        "task_name": cfg["task_name"],
        "classification": classification,
        "training_executed": False,
        "blockers": list(blocker_lines),
    }
    report_lines = [
        "# FINAL360M full-train blocker or partial report",
        "",
        f"- classification: `{classification}`",
    ]
    report_lines.extend(f"- blocker: `{line}`" for line in blocker_lines)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _json_dump(REPO_ROOT / outputs["training_protocol_path"], payload)


def _run_prechecks(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = cfg["inputs"]
    pre = cfg["prechecks"]
    data_cfg = cfg["data"]
    blockers: List[str] = []
    warnings: List[str] = []

    branch = _git(["git", "branch", "--show-current"])
    git_status = _git(["git", "status", "--short"]).splitlines()
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    git_commit = _git(["git", "rev-parse", "HEAD"])
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)

    expected_branches = [str(x) for x in pre["expected_branches"]]
    if branch not in expected_branches:
        warnings.append(f"current branch {branch} not in recommended set {expected_branches}")
    if disk_free_gb <= float(pre.get("min_disk_free_gb_warn", 5.0)):
        warnings.append(f"disk free is low: {disk_free_gb:.2f} GiB")

    path_map = {name: REPO_ROOT / rel for name, rel in inputs.items()}
    for name, path in path_map.items():
        if not path.exists():
            blockers.append(f"missing required input: {name} -> {path}")

    adapter_text = path_map["dataset_adapter"].read_text(encoding="utf-8")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to scan raw sequences")

    if bool(data_cfg.get("use_train_subset")):
        blockers.append("use_train_subset must be false")
    if _normalize_optional_int(data_cfg.get("train_subset_max")) not in {None, 0}:
        blockers.append(f"train_subset_max must be null/0 but got {data_cfg.get('train_subset_max')}")
    if _normalize_optional_int(data_cfg.get("train_subset_count")) not in {None, 0}:
        blockers.append(f"train_subset_count must be null/0 but got {data_cfg.get('train_subset_count')}")

    image_hw = tuple(int(x) for x in data_cfg["image_hw"])
    ds_train = Dset2CCanonicalPairDataset(
        str(path_map["train_manifest"]),
        expected_split="train",
        image_hw=image_hw,
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
    )
    ds_val = Dset2CCanonicalPairDataset(
        str(path_map["val_manifest"]),
        expected_split="val",
        image_hw=image_hw,
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
    )
    ds_test = Dset2CCanonicalPairDataset(
        str(path_map["test_manifest"]),
        expected_split="test",
        image_hw=image_hw,
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
    )
    split_audit = summarize_manifest_group([ds_train, ds_val, ds_test])
    if split_audit["has_overlap"]:
        blockers.append(f"sequence overlap detected: {split_audit['sequence_overlap']}")

    expected_counts = {k: int(v) for k, v in pre["expected_counts"].items()}
    actual_counts = {
        "train": len(ds_train),
        "val": len(ds_val),
        "test": len(ds_test),
    }
    manifest_line_counts = {
        "train": _manifest_line_count(path_map["train_manifest"]),
        "val": _manifest_line_count(path_map["val_manifest"]),
        "test": _manifest_line_count(path_map["test_manifest"]),
    }
    for split in ("train", "val", "test"):
        if actual_counts[split] != expected_counts[split]:
            blockers.append(
                f"{split} dataset count mismatch: dataset={actual_counts[split]} expected={expected_counts[split]}"
            )
        if manifest_line_counts[split] != expected_counts[split]:
            blockers.append(
                f"{split} manifest line count mismatch: manifest={manifest_line_counts[split]} expected={expected_counts[split]}"
            )

    outputs = cfg["outputs"]
    checkpoint_dir = str(outputs["checkpoint_dir"])
    if checkpoint_dir == "checkpoints/FINAL360I_struct360b_final":
        blockers.append("checkpoint_dir must not overwrite old FINAL360I directory")

    return {
        "blockers": blockers,
        "warnings": warnings,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_status_short": git_status,
        "git_commit": git_commit,
        "disk_free_gb": float(disk_free_gb),
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "actual_counts": actual_counts,
        "manifest_line_counts": manifest_line_counts,
        "split_audit": split_audit,
    }


def _candidate_checkpoint_name(epoch: int) -> str:
    return f"candidate_epoch_{epoch:02d}.pt"


def _update_top_candidates(
    candidates: List[Dict[str, Any]],
    *,
    epoch: int,
    mini_val_score: float,
    checkpoint_dir: Path,
    last_checkpoint_path: Path,
    top_k: int,
) -> List[Dict[str, Any]]:
    candidate_path = checkpoint_dir / _candidate_checkpoint_name(epoch)
    existing = [item for item in candidates if int(item["epoch"]) != int(epoch)]
    existing.append(
        {
            "epoch": int(epoch),
            "mini_val_score": float(mini_val_score),
            "checkpoint_path": str(candidate_path),
            "full_val_score": None,
            "full_val_metrics": None,
        }
    )
    existing.sort(key=lambda item: float(item["mini_val_score"]))
    keep = existing[: int(top_k)]
    keep_epochs = {int(item["epoch"]) for item in keep}
    if int(epoch) in keep_epochs:
        candidate_path.write_bytes(last_checkpoint_path.read_bytes())
    for item in existing[int(top_k):]:
        old_path = Path(item["checkpoint_path"])
        if old_path.exists():
            old_path.unlink()
    return keep


def _write_training_protocol(
    cfg: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
) -> None:
    _json_dump(REPO_ROOT / cfg["outputs"]["training_protocol_path"], dict(payload))


def _nonfinite_names(named_tensors: Mapping[str, torch.Tensor]) -> List[str]:
    bad: List[str] = []
    for name, tensor in named_tensors.items():
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            bad.append(name)
    return bad


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "final360m_fulltrain_struct360b_thesis_main.yaml")
    cfg = load_yaml_like(cfg_path)
    start_time = time.time()
    precheck = _run_prechecks(cfg)
    if precheck["blockers"]:
        _write_blocker(cfg, precheck["blockers"], classification="blocked")
        raise RuntimeError("FINAL360M precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    outputs = cfg["outputs"]
    checkpoint_dir = REPO_ROOT / outputs["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FINAL360M] started at {_now_ts()}")
    print(f"[FINAL360M] branch={precheck['branch']} commit={precheck['git_commit']}")
    print(
        f"[FINAL360M] dataset counts train={precheck['actual_counts']['train']} "
        f"val={precheck['actual_counts']['val']} test={precheck['actual_counts']['test']}"
    )
    for warning in precheck["warnings"]:
        print(f"[FINAL360M][warning] {warning}")

    _seed_everything(int(cfg["training"]["seed"]))
    use_cuda = bool(cfg["model"]["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    system_info = _system_info(device)

    ds_train = precheck["datasets"]["train"]
    ds_val = precheck["datasets"]["val"]
    ds_test = precheck["datasets"]["test"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    val_cfg = cfg["validation"]
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]
    tmag_epsilon = float(data_cfg["tmag_epsilon"])
    batch_eval = int(data_cfg["eval_batch_size"])

    train_loader = DataLoader(
        ds_train,
        batch_size=int(data_cfg["train_batch_size"]),
        shuffle=bool(data_cfg["shuffle_train"]),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )
    mini_val_subset, mini_val_info = _build_mini_val_subset(
        ds_val,
        count=int(val_cfg["mini_val_count"]),
        seed=int(data_cfg["mini_val_seed"]),
    )
    mini_val_loader = DataLoader(
        mini_val_subset,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_eval,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(data_cfg["num_workers"]) > 0),
        drop_last=False,
    )

    init_ckpt_path = REPO_ROOT / cfg["inputs"]["init_checkpoint"]
    last_checkpoint_path = checkpoint_dir / "last.pt"
    best_full_val_checkpoint_path = checkpoint_dir / "best_full_val.pt"

    resumed = False
    resume_epoch = 0
    best_full_val_score = float("inf")
    best_full_val_epoch = -1
    best_full_val_metrics: Optional[Dict[str, Any]] = None
    candidate_records: List[Dict[str, Any]] = []
    wall_time_before_resume = 0.0
    load_status: Dict[str, Any]
    init_status: Dict[str, Any]

    if bool(train_cfg.get("resume_if_available")) and last_checkpoint_path.exists():
        model, ckpt_cfg, resume_meta, resume_epoch = _load_resume_checkpoint(last_checkpoint_path, device=device)
        optimizer, coarse_params, fine_params = _build_optimizer(model, cfg)
        if "optimizer" in torch.load(str(last_checkpoint_path), map_location=device):
            payload = torch.load(str(last_checkpoint_path), map_location=device)
            optimizer.load_state_dict(payload["optimizer"])
        resumed = True
        wall_time_before_resume = float(resume_meta.get("wall_time_sec") or 0.0)
        best_full_val_score = float(resume_meta.get("best_full_val_score") or float("inf"))
        best_full_val_epoch = int(resume_meta.get("best_full_val_epoch") or -1)
        best_full_val_metrics = resume_meta.get("best_full_val_metrics")
        candidate_records = list(resume_meta.get("candidate_records") or [])
        load_status = dict(resume_meta.get("checkpoint_load_status") or {"status": "resume"})
        init_status = dict(resume_meta.get("new_parameter_init_status") or {})
        print(f"[FINAL360M] resuming from {last_checkpoint_path} at epoch {resume_epoch}")
    else:
        model, ckpt_cfg, load_status, init_status = _load_model_with_status(
            init_ckpt_path,
            device,
            strict_attempt=bool(model_cfg["strict_load_attempt"]),
            train_cfg=cfg,
        )
        optimizer, coarse_params, fine_params = _build_optimizer(model, cfg)
        print(f"[FINAL360M] initialized from {init_ckpt_path}")

    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")
    param_counts = count_parameters(model)
    train_log_path = checkpoint_dir / "train_log.jsonl"
    if not resumed and train_log_path.exists():
        train_log_path.unlink()
    (checkpoint_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    min_epochs = int(train_cfg["min_epochs_for_meaningful"])
    epochs_requested = min(int(train_cfg["epochs_requested"]), int(train_cfg["max_epochs"]))
    time_budget_sec = float(train_cfg["max_wall_time_hours"]) * 3600.0
    top_k = int(val_cfg["top_k_mini_val_candidates"])
    full_val_every = int(val_cfg["full_val_every_n_epochs"])
    score_cfg = dict(val_cfg["selection_score"])
    component_cfg = loss_cfg.get("components", {})
    abort_on_nonfinite_loss = bool(train_cfg.get("abort_on_nonfinite_loss", False))
    abort_on_nonfinite_grad = bool(train_cfg.get("abort_on_nonfinite_grad", False))
    _write_training_protocol(
        cfg,
        payload={
            "task_name": cfg["task_name"],
            "experiment_name": cfg["experiment_name"],
            "status": "running",
            "started_at": _now_ts(),
            "full_retrain_executed": False,
            "full_train_manifest_used": True,
            "train_subset_disabled": True,
            "train_sample_count": len(ds_train),
            "val_sample_count": len(ds_val),
            "test_sample_count": len(ds_test),
            "epochs_requested": epochs_requested,
            "epochs_completed": resume_epoch,
            "current_branch": precheck["branch"],
            "git_commit": precheck["git_commit"],
            "warnings": list(precheck["warnings"]),
        },
    )

    for epoch in range(resume_epoch + 1, epochs_requested + 1):
        print(f"[FINAL360M] epoch {epoch}/{epochs_requested} train start")
        stage_info = _set_stage_trainability(
            epoch=epoch,
            model=model,
            coarse_params=coarse_params,
            fine_params=fine_params,
            optimizer=optimizer,
            train_cfg={
                "epochs": epochs_requested,
                "warmup_epochs": int(train_cfg["warmup_epochs"]),
                "warmup_coarse_lr": float(train_cfg["warmup_coarse_lr"]),
                "warmup_fine_lr": float(train_cfg["warmup_fine_lr"]),
                "coarse_lr": float(train_cfg["coarse_lr"]),
                "fine_lr": float(train_cfg["fine_lr"]),
                "min_lr": float(train_cfg["min_lr"]),
            },
        )
        model.train()
        epoch_losses = {
            "loss_total": [],
            "loss_final": [],
            "loss_coarse_aux": [],
            "loss_residual_reg": [],
            "loss_rot": [],
            "loss_tdir": [],
            "loss_tmag": [],
            "loss_scale_stability": [],
        }
        train_nan_inf_count = 0
        gate_values: List[float] = []
        delta_rot_deg_values: List[float] = []
        delta_tdir_norm_values: List[float] = []
        delta_log_abs_values: List[float] = []
        obs_weights_all: List[float] = []
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
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                R_pred, _t_pred_local, aux = model(IA, IB, dt_world=dt_world_t)
                final_loss = train360d_pose_loss(
                    R_pred=R_pred,
                    tdir_pred_B=aux["t_dir_out"],
                    tmag_pred=aux["t_mag"],
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
                    tmag_epsilon=tmag_epsilon,
                    enable_observability=bool(component_cfg.get("enable_observability", True)),
                    enable_k_step_balancing=bool(component_cfg.get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(component_cfg.get("enable_scale_stabilization", True)),
                )
                coarse_loss = train360d_pose_loss(
                    R_pred=aux["coarse_R"],
                    tdir_pred_B=aux["coarse_t_dir_out"],
                    tmag_pred=aux["coarse_t_mag"],
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
                    tmag_epsilon=tmag_epsilon,
                    enable_observability=bool(component_cfg.get("enable_observability", True)),
                    enable_k_step_balancing=bool(component_cfg.get("enable_k_step_balancing", True)),
                    enable_scale_stabilization=bool(component_cfg.get("enable_scale_stabilization", True)),
                )
                residual_reg = model.residual_regularization(aux)
                loss_total = (
                    final_loss["loss_total"]
                    + float(loss_cfg["coarse_aux_weight"]) * coarse_loss["loss_total"]
                    + float(loss_cfg["residual_reg_weight"]) * residual_reg["loss"]
                )
            if abort_on_nonfinite_loss:
                bad_names = _nonfinite_names(
                    {
                        "R_pred": R_pred,
                        "tdir_pred": aux["t_dir_out"],
                        "tmag_pred": aux["t_mag"],
                        "coarse_R": aux["coarse_R"],
                        "coarse_tdir": aux["coarse_t_dir_out"],
                        "coarse_tmag": aux["coarse_t_mag"],
                        "loss_total": loss_total,
                        "loss_rot": final_loss["loss_rot"],
                        "loss_tdir": final_loss["loss_tdir"],
                        "loss_tmag": final_loss["loss_tmag"],
                        "loss_scale_stability": final_loss["loss_scale_stability"],
                    }
                )
                if bad_names:
                    raise RuntimeError(
                        "FINAL360M nonfinite loss/output detected "
                        f"batch_seq={batch['meta']['sequence_id']} "
                        f"pair_index={batch['pair_index'].tolist()} "
                        f"k={batch['k'].tolist()} bad={bad_names}"
                    )
            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                if abort_on_nonfinite_grad:
                    grad_bad = False
                    for param in model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grad_bad = True
                            break
                    if grad_bad:
                        raise RuntimeError(
                            "FINAL360M nonfinite gradient detected "
                            f"batch_seq={batch['meta']['sequence_id']} "
                            f"pair_index={batch['pair_index'].tolist()} "
                            f"k={batch['k'].tolist()}"
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                if abort_on_nonfinite_grad:
                    grad_bad = False
                    for param in model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grad_bad = True
                            break
                    if grad_bad:
                        raise RuntimeError(
                            "FINAL360M nonfinite gradient detected "
                            f"batch_seq={batch['meta']['sequence_id']} "
                            f"pair_index={batch['pair_index'].tolist()} "
                            f"k={batch['k'].tolist()}"
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                optimizer.step()

            epoch_losses["loss_total"].append(float(loss_total.detach().cpu()))
            epoch_losses["loss_final"].append(float(final_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_coarse_aux"].append(float(coarse_loss["loss_total"].detach().cpu()))
            epoch_losses["loss_residual_reg"].append(float(residual_reg["loss"].detach().cpu()))
            epoch_losses["loss_rot"].append(float(final_loss["loss_rot"].detach().cpu()))
            epoch_losses["loss_tdir"].append(float(final_loss["loss_tdir"].detach().cpu()))
            epoch_losses["loss_tmag"].append(float(final_loss["loss_tmag"].detach().cpu()))
            epoch_losses["loss_scale_stability"].append(float(final_loss["loss_scale_stability"].detach().cpu()))
            gate_values.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())
            delta_rot_deg_values.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
            delta_tdir_norm_values.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
            delta_log_abs_values.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
            obs_weights_all.extend(final_loss["obs_weights"].detach().cpu().numpy().astype(np.float64).tolist())
            k_weights_all.extend(final_loss["k_weights"].detach().cpu().numpy().astype(np.float64).tolist())
            train_nan_inf_count += int(sum((~torch.isfinite(v)).sum().item() for v in [R_pred, aux["t_dir_out"], aux["t_mag"], loss_total]))

        mini_val_eval = evaluate_struct360b_pose(model, mini_val_loader, device, tmag_epsilon=tmag_epsilon, max_batches=None)
        mini_val_metrics = mini_val_eval["final_metrics"]
        mini_val_score = _compute_val_score(mini_val_metrics, score_cfg=score_cfg, eps=tmag_epsilon)
        print(
            f"[FINAL360M] epoch {epoch} mini-val score={mini_val_score:.6f} "
            f"rot={mini_val_metrics.get('rot_mean_deg')} "
            f"signed_tdir={mini_val_metrics.get('signed_tdir_mean_deg')} "
            f"anti={mini_val_metrics.get('anti_parallel_rate')} "
            f"tmag={mini_val_metrics.get('tmag_median_ratio')} "
            f"path={mini_val_metrics.get('path_ratio')}"
        )

        elapsed_total = wall_time_before_resume + (time.time() - start_time)
        last_meta = {
            "task_name": cfg["task_name"],
            "epoch": int(epoch),
            "wall_time_sec": float(elapsed_total),
            "best_full_val_score": float(best_full_val_score) if math.isfinite(best_full_val_score) else None,
            "best_full_val_epoch": int(best_full_val_epoch),
            "best_full_val_metrics": best_full_val_metrics,
            "candidate_records": candidate_records,
            "checkpoint_load_status": load_status,
            "new_parameter_init_status": init_status,
            "train_sample_count": len(ds_train),
            "train_subset_disabled": True,
        }
        _save_checkpoint(last_checkpoint_path, model=model, optimizer=optimizer, epoch=epoch, ckpt_cfg=ckpt_cfg, metadata=last_meta)
        candidate_records = _update_top_candidates(
            candidate_records,
            epoch=epoch,
            mini_val_score=mini_val_score,
            checkpoint_dir=checkpoint_dir,
            last_checkpoint_path=last_checkpoint_path,
            top_k=top_k,
        )

        full_val_score = None
        full_val_metrics = None
        if epoch % full_val_every == 0 or epoch == epochs_requested:
            print(f"[FINAL360M] epoch {epoch} full-val start")
            full_val_eval = evaluate_struct360b_pose(model, val_loader, device, tmag_epsilon=tmag_epsilon, max_batches=None)
            full_val_metrics = full_val_eval["final_metrics"]
            full_val_score = _compute_val_score(full_val_metrics, score_cfg=score_cfg, eps=tmag_epsilon)
            print(
                f"[FINAL360M] epoch {epoch} full-val score={full_val_score:.6f} "
                f"rot={full_val_metrics.get('rot_mean_deg')} "
                f"signed_tdir={full_val_metrics.get('signed_tdir_mean_deg')} "
                f"anti={full_val_metrics.get('anti_parallel_rate')} "
                f"tmag={full_val_metrics.get('tmag_median_ratio')} "
                f"path={full_val_metrics.get('path_ratio')}"
            )
            if float(full_val_score) < float(best_full_val_score) and int(full_val_metrics["nan_inf_count"]) == 0:
                best_full_val_score = float(full_val_score)
                best_full_val_epoch = int(epoch)
                best_full_val_metrics = dict(full_val_metrics)
                _save_checkpoint(
                    best_full_val_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    ckpt_cfg=ckpt_cfg,
                    metadata={
                        **last_meta,
                        "checkpoint_role": "best_full_val",
                        "full_val_score": float(full_val_score),
                        "full_val_metrics": dict(full_val_metrics),
                    },
                )
                print(f"[FINAL360M] epoch {epoch} updated best_full_val checkpoint -> {best_full_val_checkpoint_path}")
            for item in candidate_records:
                if int(item["epoch"]) == int(epoch):
                    item["full_val_score"] = float(full_val_score)
                    item["full_val_metrics"] = dict(full_val_metrics)
                    break

        epoch_log = {
            "timestamp": _now_ts(),
            "epoch": int(epoch),
            "stage": stage_info["stage"],
            "lr_coarse": stage_info["coarse_lr"],
            "lr_fine": stage_info["fine_lr"],
            "train_loss_total": float(np.mean(epoch_losses["loss_total"])) if epoch_losses["loss_total"] else None,
            "train_loss_final": float(np.mean(epoch_losses["loss_final"])) if epoch_losses["loss_final"] else None,
            "train_loss_coarse_aux": float(np.mean(epoch_losses["loss_coarse_aux"])) if epoch_losses["loss_coarse_aux"] else None,
            "train_loss_residual_reg": float(np.mean(epoch_losses["loss_residual_reg"])) if epoch_losses["loss_residual_reg"] else None,
            "train_loss_rot": float(np.mean(epoch_losses["loss_rot"])) if epoch_losses["loss_rot"] else None,
            "train_loss_tdir": float(np.mean(epoch_losses["loss_tdir"])) if epoch_losses["loss_tdir"] else None,
            "train_loss_tmag": float(np.mean(epoch_losses["loss_tmag"])) if epoch_losses["loss_tmag"] else None,
            "train_loss_scale_stability": float(np.mean(epoch_losses["loss_scale_stability"])) if epoch_losses["loss_scale_stability"] else None,
            "train_nan_inf_count": int(train_nan_inf_count),
            "mini_val_score": float(mini_val_score),
            "mini_val_metrics": mini_val_metrics,
            "full_val_score": full_val_score,
            "full_val_metrics": full_val_metrics,
            "obs_weight_stats": _float_stats(obs_weights_all),
            "k_weight_stats": _float_stats(k_weights_all),
            "train_gate_stats": _float_stats(gate_values),
            "train_delta_rot_deg_stats": _float_stats(delta_rot_deg_values),
            "train_delta_tdir_norm_stats": _float_stats(delta_tdir_norm_values),
            "train_delta_log_tmag_abs_stats": _float_stats(delta_log_abs_values),
            "elapsed_time_sec": float(elapsed_total),
        }
        _append_jsonl(train_log_path, epoch_log)

        if elapsed_total >= time_budget_sec:
            print(f"[FINAL360M] time budget reached after epoch {epoch}")
            break

    epochs_completed = int(max(best_full_val_epoch, resume_epoch, 0))
    try:
        epochs_completed = max(int(row["epoch"]) for row in [json.loads(line) for line in train_log_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    except Exception:
        epochs_completed = max(epochs_completed, 0)

    for item in candidate_records:
        candidate_path = Path(item["checkpoint_path"])
        if not candidate_path.exists():
            continue
        if item.get("full_val_score") is not None:
            continue
        eval_payload = _evaluate_checkpoint(candidate_path, loader=val_loader, device=device, tmag_epsilon=tmag_epsilon)
        full_val_metrics = eval_payload["final_metrics"]
        full_val_score = _compute_val_score(full_val_metrics, score_cfg=score_cfg, eps=tmag_epsilon)
        item["full_val_score"] = float(full_val_score)
        item["full_val_metrics"] = dict(full_val_metrics)
        if float(full_val_score) < float(best_full_val_score) and int(full_val_metrics["nan_inf_count"]) == 0:
            best_full_val_score = float(full_val_score)
            best_full_val_epoch = int(item["epoch"])
            best_full_val_metrics = dict(full_val_metrics)
            best_full_val_checkpoint_path.write_bytes(candidate_path.read_bytes())
            print(f"[FINAL360M] post-train candidate epoch {item['epoch']} became best_full_val")

    if not best_full_val_checkpoint_path.exists():
        _write_blocker(cfg, ["no full-val-selected checkpoint was produced"], classification="partial_time_budget")
        raise RuntimeError("FINAL360M did not produce any full-val-selected checkpoint")

    selected_val_eval = _evaluate_checkpoint(best_full_val_checkpoint_path, loader=val_loader, device=device, tmag_epsilon=tmag_epsilon)
    selected_val_metrics = selected_val_eval["final_metrics"]
    selected_val_score = _compute_val_score(selected_val_metrics, score_cfg=score_cfg, eps=tmag_epsilon)
    selected_test_eval = _evaluate_checkpoint(best_full_val_checkpoint_path, loader=test_loader, device=device, tmag_epsilon=tmag_epsilon)
    selected_test_metrics = selected_test_eval["final_metrics"]
    print(
        f"[FINAL360M] selected checkpoint {best_full_val_checkpoint_path} "
        f"best_epoch={best_full_val_epoch} test_signed_tdir={selected_test_metrics.get('signed_tdir_mean_deg')} "
        f"test_anti={selected_test_metrics.get('anti_parallel_rate')}"
    )

    old_final360i_test = _extract_final360i_old_test_metrics(cfg)
    train360d_test = _baseline_from_final360i_table(cfg, "TRAIN360D")
    train360h_test = _baseline_from_final360i_table(cfg, "TRAIN360H")
    base360d_test = _extract_base360_metrics(_read_json(REPO_ROOT / cfg["inputs"]["base360d_test_metrics"]))

    metric_status = {
        "rot_mean_deg": _safe_metric_compare(selected_test_metrics.get("rot_mean_deg"), old_final360i_test.get("rot_mean_deg"), lower_is_better=True),
        "signed_tdir_mean_deg": _safe_metric_compare(selected_test_metrics.get("signed_tdir_mean_deg"), old_final360i_test.get("signed_tdir_mean_deg"), lower_is_better=True),
        "anti_parallel_rate": _safe_metric_compare(selected_test_metrics.get("anti_parallel_rate"), old_final360i_test.get("anti_parallel_rate"), lower_is_better=True),
        "tmag_median_ratio": _safe_metric_compare(selected_test_metrics.get("tmag_median_ratio"), old_final360i_test.get("tmag_median_ratio"), lower_is_better=False),
        "path_ratio": _safe_metric_compare(selected_test_metrics.get("path_ratio"), old_final360i_test.get("path_ratio"), lower_is_better=False),
    }
    overall_vs_old = _overall_vs_old(metric_status)

    final_elapsed = wall_time_before_resume + (time.time() - start_time)
    selected_by_full_val = True
    full_train_used = precheck["actual_counts"]["train"] == 12154
    train_subset_disabled = True
    thesis_main_ok = (
        full_train_used
        and train_subset_disabled
        and selected_by_full_val
        and not bool(cfg["test_evaluation"]["test_used_for_selection"])
        and epochs_completed >= min_epochs
    )
    classification = "thesis_main_model" if thesis_main_ok else "partial_time_budget"

    selected_checkpoint = str(best_full_val_checkpoint_path)
    val_payload = {
        "task_name": cfg["task_name"],
        "experiment_name": cfg["experiment_name"],
        "full_retrain_executed": True,
        "selected_model_name": cfg["experiment_name"],
        "selected_checkpoint": selected_checkpoint,
        "selected_by_full_val": True,
        "selection_basis": "full validation only",
        "test_used_for_selection": False,
        "train_subset_disabled": True,
        "train_sample_count": len(ds_train),
        "val_sample_count": len(ds_val),
        "test_sample_count": len(ds_test),
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_completed,
        "best_epoch": int(best_full_val_epoch),
        "full_val_score": float(selected_val_score),
        "metrics": selected_val_metrics,
        "coarse_metrics": selected_val_eval["coarse_metrics"],
        "residual_stats": selected_val_eval["residual_stats"],
        "gate_stats": selected_val_eval["gate_stats"],
        "alpha": selected_val_eval["alpha"],
        "beta": selected_val_eval["beta"],
    }
    test_payload = {
        "task_name": cfg["task_name"],
        "experiment_name": cfg["experiment_name"],
        "full_retrain_executed": True,
        "selected_model_name": cfg["experiment_name"],
        "selected_checkpoint": selected_checkpoint,
        "selected_by_full_val": True,
        "selection_basis": "full validation only",
        "test_used_for_selection": False,
        "best_epoch": int(best_full_val_epoch),
        "full_val_score_of_selected_checkpoint": float(selected_val_score),
        "metrics": selected_test_metrics,
        "coarse_metrics": selected_test_eval["coarse_metrics"],
        "residual_stats": selected_test_eval["residual_stats"],
        "gate_stats": selected_test_eval["gate_stats"],
        "alpha": selected_test_eval["alpha"],
        "beta": selected_test_eval["beta"],
    }
    _json_dump(REPO_ROOT / outputs["val_metrics_path"], val_payload)
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], test_payload)

    protocol_payload = {
        "task_name": cfg["task_name"],
        "experiment_name": cfg["experiment_name"],
        "full_retrain_executed": True,
        "wall_time_budget_hours": float(train_cfg["max_wall_time_hours"]),
        "wall_time_used_sec": float(final_elapsed),
        "full_train_manifest_used": full_train_used,
        "train_subset_disabled": train_subset_disabled,
        "train_sample_count": len(ds_train),
        "val_sample_count": len(ds_val),
        "test_sample_count": len(ds_test),
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_completed,
        "best_epoch": int(best_full_val_epoch),
        "selected_by_full_val": True,
        "test_used_for_selection": False,
        "init_checkpoint": str(init_ckpt_path),
        "selected_checkpoint": selected_checkpoint,
        "classification": classification,
        "thesis_main_model_recommendation": bool(thesis_main_ok),
        "current_branch": precheck["branch"],
        "git_commit": precheck["git_commit"],
        "warnings": list(precheck["warnings"]),
        "system_info": system_info,
        "manifest_line_counts": precheck["manifest_line_counts"],
        "mini_val_info": mini_val_info,
    }
    _write_training_protocol(cfg, payload=protocol_payload)

    model_selection_payload = {
        "task_name": cfg["task_name"],
        "selection_protocol": {
            "val_used_for_selection": True,
            "test_used_for_selection": False,
            "selected_by_full_val": True,
            "mini_val_count": int(val_cfg["mini_val_count"]),
            "full_val_every_n_epochs": int(val_cfg["full_val_every_n_epochs"]),
            "score_formula": "signed_tdir_mean + 80 * anti_parallel_rate + 0.2 * rot_mean + 15 * abs(log(tmag_median_ratio + eps)) + 8 * abs(log(path_ratio + eps))",
        },
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_completed,
        "best_epoch": int(best_full_val_epoch),
        "selected_checkpoint": selected_checkpoint,
        "selected_val_score": float(selected_val_score),
        "candidate_records": candidate_records,
    }
    _json_dump(REPO_ROOT / outputs["model_selection_table_path"], model_selection_payload)

    metric_source_manifest = {
        "task_name": cfg["task_name"],
        "selected_checkpoint": selected_checkpoint,
        "val_metrics_source": outputs["val_metrics_path"],
        "test_metrics_source": outputs["test_metrics_path"],
        "selected_by_full_val": True,
        "test_used_for_selection": False,
        "full_train_manifest": cfg["inputs"]["train_manifest"],
        "old_final360i_subset_candidate": cfg["inputs"]["old_final360i_metrics_test"],
        "train360d_source": "derived from reports/FINAL360I_model_selection_table.json baseline_recap",
        "train360h_source": "derived from reports/FINAL360I_model_selection_table.json baseline_recap",
        "base360d_source": cfg["inputs"]["base360d_test_metrics"],
    }
    _json_dump(REPO_ROOT / outputs["metric_source_manifest_path"], metric_source_manifest)

    comparison_lines = [
        "# FINAL360M vs old FINAL360I subset candidate",
        "",
        f"- overall comparison: `{overall_vs_old}`",
        f"- rot_mean_deg: old `{old_final360i_test.get('rot_mean_deg')}` vs new `{selected_test_metrics.get('rot_mean_deg')}` -> `{metric_status['rot_mean_deg']}`",
        f"- signed_tdir_mean_deg: old `{old_final360i_test.get('signed_tdir_mean_deg')}` vs new `{selected_test_metrics.get('signed_tdir_mean_deg')}` -> `{metric_status['signed_tdir_mean_deg']}`",
        f"- anti_parallel_rate: old `{old_final360i_test.get('anti_parallel_rate')}` vs new `{selected_test_metrics.get('anti_parallel_rate')}` -> `{metric_status['anti_parallel_rate']}`",
        f"- tmag_median_ratio: old `{old_final360i_test.get('tmag_median_ratio')}` vs new `{selected_test_metrics.get('tmag_median_ratio')}` -> `{metric_status['tmag_median_ratio']}`",
        f"- path_ratio: old `{old_final360i_test.get('path_ratio')}` vs new `{selected_test_metrics.get('path_ratio')}` -> `{metric_status['path_ratio']}`",
        f"- thesis main model recommendation: `{_bool_text(thesis_main_ok)}`",
        "- needs trajectory reevaluation: `true`",
        "- needs THESIS362 table update: `true`",
    ]
    (REPO_ROOT / outputs["comparison_path"]).write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# FINAL360M full-train thesis main",
        "",
        "## Summary",
        f"- FINAL360M full retrain executed: `{_bool_text(True)}`",
        f"- wall time budget hours: `{train_cfg['max_wall_time_hours']}`",
        f"- wall time used: `{final_elapsed:.2f} sec`",
        f"- full train manifest used: `{_bool_text(full_train_used)}`",
        f"- train subset disabled: `{_bool_text(train_subset_disabled)}`",
        f"- train sample count: `{len(ds_train)}`",
        f"- val sample count: `{len(ds_val)}`",
        f"- test sample count: `{len(ds_test)}`",
        f"- epochs requested: `{epochs_requested}`",
        f"- epochs completed: `{epochs_completed}`",
        f"- best epoch: `{best_full_val_epoch}`",
        f"- selected by full val: `{_bool_text(selected_by_full_val)}`",
        "- test used for selection: `false`",
        f"- init checkpoint: `{init_ckpt_path}`",
        f"- selected checkpoint: `{selected_checkpoint}`",
        f"- FINAL360M test rot_mean: `{selected_test_metrics.get('rot_mean_deg')}`",
        f"- FINAL360M test signed_tdir_mean: `{selected_test_metrics.get('signed_tdir_mean_deg')}`",
        f"- FINAL360M test anti_parallel_rate: `{selected_test_metrics.get('anti_parallel_rate')}`",
        f"- FINAL360M test tmag_median_ratio: `{selected_test_metrics.get('tmag_median_ratio')}`",
        f"- FINAL360M test path_ratio: `{selected_test_metrics.get('path_ratio')}`",
        f"- old FINAL360I test rot_mean: `{old_final360i_test.get('rot_mean_deg')}`",
        f"- old FINAL360I test signed_tdir_mean: `{old_final360i_test.get('signed_tdir_mean_deg')}`",
        f"- old FINAL360I test anti_parallel_rate: `{old_final360i_test.get('anti_parallel_rate')}`",
        f"- old FINAL360I test tmag_median_ratio: `{old_final360i_test.get('tmag_median_ratio')}`",
        f"- old FINAL360I test path_ratio: `{old_final360i_test.get('path_ratio')}`",
        f"- compared to old FINAL360I: `{overall_vs_old}`",
        f"- thesis main model recommendation: `{classification}`",
        "- needs trajectory reevaluation: `true`",
        "- needs THESIS362 table update: `true`",
        "- existing metrics modified: `false`",
        "- checkpoints committed: `false`",
        "",
        "## Baseline comparisons",
        f"- TRAIN360D: `{train360d_test if train360d_test is not None else 'missing'}`",
        f"- TRAIN360H: `{train360h_test if train360h_test is not None else 'missing'}`",
        f"- BASE360D: `{base360d_test if base360d_test else 'missing'}`",
        "",
        "## Selection and lineage",
        f"- selected checkpoint was chosen by full validation score only: `{_bool_text(True)}`",
        f"- mini-val monitor subset: `{mini_val_info}`",
        f"- candidate records: `{candidate_records}`",
        f"- current artifact source manifest: `{outputs['metric_source_manifest_path']}`",
        "",
        "## Next task",
        "- recommended next task: `rerun trajectory-level evaluation and update THESIS362 tables/figures if FINAL360M remains selected`",
    ]
    (REPO_ROOT / outputs["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if classification != "thesis_main_model":
        partial_lines = [
            "# FINAL360M full-train blocker or partial report",
            "",
            f"- classification: `{classification}`",
            f"- epochs_completed: `{epochs_completed}`",
            f"- selected_by_full_val: `{_bool_text(True)}`",
            f"- thesis_main_model_recommendation: `{_bool_text(thesis_main_ok)}`",
        ]
        (REPO_ROOT / outputs["blocker_or_partial_path"]).write_text("\n".join(partial_lines) + "\n", encoding="utf-8")
    print(f"[FINAL360M] finished classification={classification}")


if __name__ == "__main__":
    train()
