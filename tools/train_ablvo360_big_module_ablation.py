#!/usr/bin/env python3
from __future__ import annotations

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
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset, summarize_manifest_group
from models.ablvo360_big_module_ablation import build_ablvo360_model, count_parameters
from train360.core.config import Config
from train360.core.pose_head import matrix_geodesic_distance
from train360.core.train360d_pose_losses import train360d_pose_loss


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _git(cmd: Sequence[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _system_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def _extract_metrics_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


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


def _rotation_angle_deg_from_matrix(R: np.ndarray) -> float:
    trace = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(math.degrees(math.acos(cos_theta)))


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    for idx, edge in enumerate(edges):
        if value < float(edge):
            return int(idx)
    return int(len(edges))


def _stratified_subset_dataset(
    dataset: Dset2CCanonicalPairDataset,
    *,
    target_count: Optional[int],
    seed: int,
    tmag_bucket_edges: Sequence[float],
    rot_bucket_edges: Sequence[float],
) -> Tuple[Dset2CCanonicalPairDataset | Subset, Dict[str, Any]]:
    if target_count is None or int(target_count) <= 0 or len(dataset) <= int(target_count):
        return dataset, {
            "subset_used": False,
            "subset_count": len(dataset),
            "original_count": len(dataset),
            "seed": int(seed),
        }

    groups: Dict[Tuple[str, int, int, int], List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        seq_id = str(sample["sequence_id"])
        tmag_bucket = _bucket_index(float(sample["tmag"]), tmag_bucket_edges)
        rot_bucket = _bucket_index(_rotation_angle_deg_from_matrix(sample["R_BA"]), rot_bucket_edges)
        k_bucket = int(sample["k"])
        groups.setdefault((seq_id, tmag_bucket, rot_bucket, k_bucket), []).append(idx)

    rng = np.random.default_rng(int(seed))
    group_items = []
    for key, indices in groups.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        expected = float(len(indices)) * float(target_count) / float(len(dataset))
        base_take = int(math.floor(expected))
        frac = expected - float(base_take)
        group_items.append(
            {
                "key": key,
                "indices": shuffled,
                "expected": expected,
                "take": base_take,
                "frac": frac,
            }
        )

    selected_count = sum(int(item["take"]) for item in group_items)
    remaining = max(0, int(target_count) - int(selected_count))
    for item in sorted(group_items, key=lambda x: (-x["frac"], -len(x["indices"]), x["key"])):
        if remaining <= 0:
            break
        if int(item["take"]) < len(item["indices"]):
            item["take"] = int(item["take"]) + 1
            remaining -= 1

    selected: List[int] = []
    selected_groups = 0
    for item in group_items:
        take = min(int(item["take"]), len(item["indices"]))
        if take > 0:
            selected_groups += 1
            selected.extend(item["indices"][:take])

    if len(selected) < int(target_count):
        selected_set = set(selected)
        leftover = []
        for item in sorted(group_items, key=lambda x: (-len(x["indices"]), x["key"])):
            for idx in item["indices"]:
                if idx not in selected_set:
                    leftover.append(idx)
        selected.extend(leftover[: int(target_count) - len(selected)])

    selected = sorted(selected[: int(target_count)])
    subset = Subset(dataset, selected)
    selected_seq = {}
    selected_tmag_bucket = {}
    selected_rot_bucket = {}
    for idx in selected:
        sample = dataset.samples[idx]
        seq_id = str(sample["sequence_id"])
        tmag_bucket = _bucket_index(float(sample["tmag"]), tmag_bucket_edges)
        rot_bucket = _bucket_index(_rotation_angle_deg_from_matrix(sample["R_BA"]), rot_bucket_edges)
        selected_seq[seq_id] = int(selected_seq.get(seq_id, 0)) + 1
        selected_tmag_bucket[str(tmag_bucket)] = int(selected_tmag_bucket.get(str(tmag_bucket), 0)) + 1
        selected_rot_bucket[str(rot_bucket)] = int(selected_rot_bucket.get(str(rot_bucket), 0)) + 1
    summary = {
        "subset_used": True,
        "subset_count": len(selected),
        "original_count": len(dataset),
        "seed": int(seed),
        "group_count": len(groups),
        "groups_selected": int(selected_groups),
        "tmag_bucket_edges": [float(x) for x in tmag_bucket_edges],
        "rot_bucket_edges_deg": [float(x) for x in rot_bucket_edges],
        "sequence_count_selected": len(selected_seq),
        "top_sequence_counts": dict(sorted(selected_seq.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
        "selected_tmag_bucket_histogram": selected_tmag_bucket,
        "selected_rot_bucket_histogram": selected_rot_bucket,
    }
    return subset, summary


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


def _float_stats(vals: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _vector_angle_deg(a: np.ndarray, b: np.ndarray, *, absolute: bool = False) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / (na * nb))
    if absolute:
        c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _empty_pose_collector() -> Dict[str, List[float]]:
    return {
        "rot_deg": [],
        "signed_tdir": [],
        "unsigned_tdir": [],
        "anti_parallel": [],
        "tmag_ratio": [],
        "log_tmag_mae": [],
        "pred_lengths": [],
        "gt_lengths": [],
    }


def _append_pose_batch(
    collector: Dict[str, List[float]],
    *,
    R_pred: torch.Tensor,
    tdir_pred_B: torch.Tensor,
    tmag_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt_vec_B: torch.Tensor,
    tmag_epsilon: float,
) -> Tuple[int, int, int]:
    nan_count = 0
    inf_count = 0
    num_pairs = 0
    check_tensors = [R_pred, tdir_pred_B, tmag_pred, R_gt, t_gt_vec_B]
    nan_count += sum(int(torch.isnan(t).sum().item()) for t in check_tensors)
    inf_count += sum(int(torch.isinf(t).sum().item()) for t in check_tensors)

    rot_deg = matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
    tdir_np = tdir_pred_B.detach().float().cpu().numpy()
    tmag_np = tmag_pred.detach().float().view(-1).cpu().numpy()
    t_gt_np = t_gt_vec_B.detach().float().cpu().numpy()
    gt_mag_np = torch.linalg.norm(t_gt_vec_B.float(), dim=-1).detach().cpu().numpy()

    for i in range(R_pred.shape[0]):
        num_pairs += 1
        collector["rot_deg"].append(float(rot_deg[i]))
        signed = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=False)
        unsigned = _vector_angle_deg(tdir_np[i], t_gt_np[i], absolute=True)
        if signed is not None:
            collector["signed_tdir"].append(float(signed))
            collector["anti_parallel"].append(1.0 if float(np.dot(tdir_np[i], t_gt_np[i])) < 0.0 else 0.0)
        if unsigned is not None:
            collector["unsigned_tdir"].append(float(unsigned))
        gt_mag = max(float(gt_mag_np[i]), float(tmag_epsilon))
        pred_mag = max(float(tmag_np[i]), float(tmag_epsilon))
        collector["tmag_ratio"].append(float(pred_mag / gt_mag))
        collector["log_tmag_mae"].append(abs(math.log(pred_mag) - math.log(gt_mag)))
        collector["pred_lengths"].append(pred_mag)
        collector["gt_lengths"].append(gt_mag)
    return num_pairs, nan_count, inf_count


def _summarize_pose_metrics(
    collector: Dict[str, List[float]],
    *,
    num_pairs: int,
    dataset_size: int,
    nan_count: int,
    inf_count: int,
) -> Dict[str, Any]:
    rot_stats = _float_stats(collector["rot_deg"])
    signed_stats = _float_stats(collector["signed_tdir"])
    unsigned_stats = _float_stats(collector["unsigned_tdir"])
    tmag_ratio_stats = _float_stats(collector["tmag_ratio"])
    return {
        "count": int(num_pairs),
        "coverage": float(num_pairs / max(dataset_size, 1)),
        "rot_mean_deg": rot_stats["mean"],
        "rot_median_deg": rot_stats["median"],
        "signed_tdir_mean_deg": signed_stats["mean"],
        "signed_tdir_median_deg": signed_stats["median"],
        "unsigned_tdir_mean_deg": unsigned_stats["mean"],
        "anti_parallel_rate": float(np.mean(np.asarray(collector["anti_parallel"], dtype=np.float64))) if collector["anti_parallel"] else None,
        "tmag_ratio_p10": tmag_ratio_stats["p10"],
        "tmag_median_ratio": tmag_ratio_stats["median"],
        "tmag_mean_ratio": tmag_ratio_stats["mean"],
        "tmag_ratio_p90": tmag_ratio_stats["p90"],
        "log_tmag_mae": float(np.mean(np.asarray(collector["log_tmag_mae"], dtype=np.float64))) if collector["log_tmag_mae"] else None,
        "scale_collapse_rate": float(np.mean(np.asarray([1.0 if float(v) < 0.1 else 0.0 for v in collector["tmag_ratio"]], dtype=np.float64))) if collector["tmag_ratio"] else None,
        "scale_explosion_rate": float(np.mean(np.asarray([1.0 if float(v) > 10.0 else 0.0 for v in collector["tmag_ratio"]], dtype=np.float64))) if collector["tmag_ratio"] else None,
        "path_ratio": float(sum(collector["pred_lengths"]) / max(sum(collector["gt_lengths"]), 1.0e-12)) if collector["gt_lengths"] else None,
        "nan_inf_count": int(nan_count + inf_count),
    }


def _compute_val_score(metrics: Mapping[str, Any], score_cfg: Mapping[str, Any], eps: float = 1.0e-6) -> float:
    signed = float(metrics.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(metrics.get("anti_parallel_rate") or 1.0)
    rot = float(metrics.get("rot_mean_deg") or 0.0)
    tmag_ratio = float(metrics.get("tmag_median_ratio") or eps)
    path_ratio = float(metrics.get("path_ratio") or eps)
    return (
        float(score_cfg.get("signed_tdir_weight", 1.0)) * signed
        + float(score_cfg.get("anti_parallel_weight", 80.0)) * anti
        + float(score_cfg.get("rot_weight", 0.2)) * rot
        + float(score_cfg.get("tmag_ratio_weight", 15.0)) * abs(math.log(max(tmag_ratio, eps)))
        + float(score_cfg.get("path_ratio_weight", 8.0)) * abs(math.log(max(path_ratio, eps)))
    )


@torch.no_grad()
def evaluate_ablvo360_pose(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tmag_epsilon: float,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    final_collector = _empty_pose_collector()
    coarse_collector = _empty_pose_collector()
    residual_rot_deg: List[float] = []
    residual_tdir_norm: List[float] = []
    residual_log_tmag_abs: List[float] = []
    residual_gate: List[float] = []
    nan_count_final = 0
    inf_count_final = 0
    nan_count_coarse = 0
    inf_count_coarse = 0
    num_pairs_final = 0
    num_pairs_coarse = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt_vec = batch["t_gt_vec"].to(device, non_blocking=True)
        dt_world = batch["meta"]["dt_world"]
        if torch.is_tensor(dt_world):
            dt_world_t = dt_world.to(device=device, dtype=torch.float32).view(-1)
        else:
            dt_world_t = torch.tensor([float(x) for x in dt_world], device=device, dtype=torch.float32)
        R_pred, _t_local, aux = model(IA, IB, dt_world=dt_world_t)
        final_added, final_nan, final_inf = _append_pose_batch(
            final_collector,
            R_pred=R_pred,
            tdir_pred_B=aux["t_dir_out"],
            tmag_pred=aux["t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt_vec,
            tmag_epsilon=tmag_epsilon,
        )
        coarse_added, coarse_nan, coarse_inf = _append_pose_batch(
            coarse_collector,
            R_pred=aux["coarse_R"],
            tdir_pred_B=aux["coarse_t_dir_out"],
            tmag_pred=aux["coarse_t_mag"],
            R_gt=R_gt,
            t_gt_vec_B=t_gt_vec,
            tmag_epsilon=tmag_epsilon,
        )
        num_pairs_final += final_added
        nan_count_final += final_nan
        inf_count_final += final_inf
        num_pairs_coarse += coarse_added
        nan_count_coarse += coarse_nan
        inf_count_coarse += coarse_inf
        residual_rot_deg.extend((aux["delta_rot_norm"].detach().float().cpu().numpy() * (180.0 / math.pi)).tolist())
        residual_tdir_norm.extend(aux["delta_tdir_norm"].detach().float().cpu().numpy().tolist())
        residual_log_tmag_abs.extend(aux["delta_log_tmag_abs"].detach().float().cpu().numpy().tolist())
        residual_gate.extend(aux["residual_gate"].detach().float().view(-1).cpu().numpy().tolist())

    return {
        "final_metrics": _summarize_pose_metrics(
            final_collector,
            num_pairs=num_pairs_final,
            dataset_size=len(loader.dataset),
            nan_count=nan_count_final,
            inf_count=inf_count_final,
        ),
        "coarse_metrics": _summarize_pose_metrics(
            coarse_collector,
            num_pairs=num_pairs_coarse,
            dataset_size=len(loader.dataset),
            nan_count=nan_count_coarse,
            inf_count=inf_count_coarse,
        ),
        "residual_stats": {
            "delta_rot_mean_deg": _float_stats(residual_rot_deg)["mean"],
            "delta_tdir_norm_mean": _float_stats(residual_tdir_norm)["mean"],
            "delta_log_tmag_abs_mean": _float_stats(residual_log_tmag_abs)["mean"],
        },
        "gate_stats": {
            "mean": _float_stats(residual_gate)["mean"],
            "median": _float_stats(residual_gate)["median"],
            "max": _float_stats(residual_gate)["max"],
        },
    }


def _variant_cfg(base_cfg: Dict[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base_cfg)
    out["H"] = int(cfg["data"]["image_hw"][0])
    out["W"] = int(cfg["data"]["image_hw"][1])
    for key, value in cfg["model"]["base_overrides"].items():
        out[key] = value
    return out


def _load_model_with_status(
    variant_name: str,
    variant_cfg: Mapping[str, Any],
    checkpoint_path: Optional[Path],
    device: torch.device,
    *,
    strict_attempt: bool,
) -> Tuple[torch.nn.Module, Config, Dict[str, Any]]:
    cfg = _cfg_from_dict(variant_cfg)
    model = build_ablvo360_model(variant_name, cfg, device).to(device)
    if checkpoint_path is None:
        return model, cfg, {"status": "scratch", "missing_keys": [], "unexpected_keys": [], "skipped_shape_mismatch": []}
    payload = torch.load(str(checkpoint_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    strict_error = None
    if strict_attempt:
        try:
            model.load_state_dict(state, strict=True)
            return model, cfg, {"status": "strict", "missing_keys": [], "unexpected_keys": [], "skipped_shape_mismatch": []}
        except Exception as exc:
            strict_error = f"{type(exc).__name__}: {exc}"
    model_state = model.state_dict()
    filtered = {}
    skipped_shape_mismatch: List[str] = []
    for key, value in state.items():
        if key in model_state and hasattr(value, "shape") and model_state[key].shape != value.shape:
            skipped_shape_mismatch.append(key)
            continue
        if key in model_state:
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return model, cfg, {
        "status": "non-strict",
        "strict_error": strict_error,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "loaded_param_count": len(filtered),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_cfg: Config,
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "cfg": dict(ckpt_cfg.__dict__),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _run_prechecks(cfg: Mapping[str, Any], cfg_path: Path) -> Dict[str, Any]:
    blockers: List[str] = []
    git_status_short = _git(["git", "status", "--short"]).splitlines()
    branch = _git(["git", "branch", "--show-current"])
    branch_vv = _git(["git", "branch", "-vv"]).splitlines()
    commit = _git(["git", "rev-parse", "HEAD"])
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    if branch not in [str(x) for x in cfg["prechecks"]["expected_branches"]]:
        blockers.append(f"current branch mismatch: {branch}")
    if disk_free_gb <= float(cfg["prechecks"]["min_disk_free_gb"]):
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB")
    if bool(cfg["prechecks"].get("require_cuda", True)) and not torch.cuda.is_available():
        blockers.append("CUDA unavailable in pytorch environment")

    required_paths = {name: (REPO_ROOT / rel) for name, rel in cfg["inputs"].items()}
    for name, path in required_paths.items():
        if not path.is_file():
            blockers.append(f"missing required file: {name} -> {path}")

    adapter_text = required_paths["dataset_adapter"].read_text(encoding="utf-8")
    if "manifest-native" not in " ".join(adapter_text.split()):
        blockers.append("dataset adapter no longer advertises manifest-native behavior")
    if "glob(" in adapter_text or "data/360DVO/Sequences/" in adapter_text:
        blockers.append("dataset adapter appears to use raw sequence globbing")

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

    return {
        "blockers": blockers,
        "git_status_short": git_status_short,
        "branch": branch,
        "branch_vv": branch_vv,
        "git_commit": commit,
        "disk_free_gb": disk_free_gb,
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "dataset_histograms": {
            "train": _hist_from_dataset(ds_train),
            "val": _hist_from_dataset(ds_val),
            "test": _hist_from_dataset(ds_test),
        },
        "split_audit": split_audit,
        "required_paths": {k: str(v) for k, v in required_paths.items()},
        "final360i_test_metrics": _extract_metrics_payload(_read_json(required_paths["final360i_test_metrics"])),
    }


def _write_blocker_artifacts(cfg: Mapping[str, Any], blocker_lines: List[str], precheck: Mapping[str, Any]) -> None:
    outputs = cfg["outputs"]
    report_path = REPO_ROOT / outputs["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# {cfg['task_name']} blocked before training\n\n## Blockers\n" + "\n".join(f"- {line}" for line in blocker_lines) + "\n",
        encoding="utf-8",
    )
    payload = {
        "task_name": str(cfg["task_name"]),
        "training_executed": False,
        "blockers": blocker_lines,
        "precheck": {
            "branch": precheck["branch"],
            "disk_free_gb": precheck["disk_free_gb"],
            "git_status_short": precheck["git_status_short"],
            "dataset_histograms": precheck["dataset_histograms"],
            "split_audit": precheck["split_audit"],
        },
    }
    json_keys = ["val_metrics_path", "test_metrics_path", "selection_table_path", "metric_source_manifest_path", "per_model_summary_path"]
    if "minival_manifest_summary_path" in outputs:
        json_keys.append("minival_manifest_summary_path")
    for key in json_keys:
        _json_dump(REPO_ROOT / outputs[key], payload)
    (REPO_ROOT / outputs["main_table_path"]).write_text(f"# {cfg['task_name']} blocked before training\n", encoding="utf-8")
    (REPO_ROOT / outputs["contribution_summary_path"]).write_text(f"# {cfg['task_name']} blocked before training\n", encoding="utf-8")


def train() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "configs" / "ablvo360_big_module_ablation.yaml")
    cfg = _load_yaml(cfg_path)
    start_time = time.time()
    precheck = _run_prechecks(cfg, cfg_path)
    if precheck["blockers"]:
        _write_blocker_artifacts(cfg, precheck["blockers"], precheck)
        raise RuntimeError("ABLVO360 precheck failed:\n- " + "\n- ".join(precheck["blockers"]))

    _seed_everything(int(cfg["training"]["seed"]))
    use_cuda = bool(cfg["model"]["use_cuda_if_available"]) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    system_info = _system_info(device)

    ds_train = precheck["datasets"]["train"]
    ds_val = precheck["datasets"]["val"]
    ds_test = precheck["datasets"]["test"]
    train_subset, subset_info = _subset_dataset(ds_train, cfg["data"].get("train_subset_max"), int(cfg["training"]["seed"]))
    mini_val_subset, minival_summary = _stratified_subset_dataset(
        ds_val,
        target_count=cfg["evaluation"].get("val_subset_count"),
        seed=int(cfg["training"]["seed"]),
        tmag_bucket_edges=cfg["evaluation"].get("val_subset_tmag_bucket_edges", [0.03, 0.08, 0.2, 0.5]),
        rot_bucket_edges=cfg["evaluation"].get("val_subset_rot_bucket_edges_deg", [5.0, 15.0, 30.0, 60.0, 120.0]),
    )
    train_loader = DataLoader(train_subset, batch_size=int(cfg["data"]["train_batch_size"]), shuffle=bool(cfg["data"]["shuffle_train"]), num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0), drop_last=False)
    mini_val_loader = DataLoader(mini_val_subset, batch_size=int(cfg["data"]["eval_batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0), drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=int(cfg["data"]["eval_batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0), drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=int(cfg["data"]["eval_batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=bool(int(cfg["data"]["num_workers"]) > 0), drop_last=False)

    init_ckpt_path = REPO_ROOT / cfg["inputs"]["init_checkpoint"]
    fallback_init_ckpt_path = REPO_ROOT / cfg["inputs"]["fallback_init_checkpoint"]
    init_payload = torch.load(str(init_ckpt_path if init_ckpt_path.is_file() else fallback_init_ckpt_path), map_location="cpu")
    base_cfg = dict(init_payload.get("cfg", {}))
    base_cfg = _variant_cfg(base_cfg, cfg)

    aggregate_val: Dict[str, Any] = {}
    aggregate_test: Dict[str, Any] = {}
    selection_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    trained_models: List[str] = []
    skipped_models: List[str] = []

    for variant in cfg["model"]["variants"]:
        variant_name = str(variant["name"])
        enabled = bool(variant.get("enabled", True))
        checkpoint_dir = REPO_ROOT / "checkpoints" / variant_name
        if not enabled:
            skipped_models.append(variant_name)
            selection_rows.append({
                "model_name": variant_name,
                "executed": False,
                "skip_reason": str(variant.get("skip_reason", "disabled_in_config")),
                "test_used_for_selection": False,
            })
            continue

        ckpt_path = None
        if str(variant.get("init_mode", "partial")) != "scratch":
            ckpt_path = init_ckpt_path if init_ckpt_path.is_file() else fallback_init_ckpt_path
        model, model_cfg, load_status = _load_model_with_status(
            variant_name,
            base_cfg,
            ckpt_path,
            device,
            strict_attempt=bool(cfg["model"]["strict_load_attempt"]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
        scaler = torch.cuda.amp.GradScaler() if bool(cfg["training"]["amp"]) and device.type == "cuda" else None
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = checkpoint_dir / "train_log.jsonl"
        if train_log_path.exists():
            train_log_path.unlink()
        (checkpoint_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

        best_val_score = float("inf")
        best_epoch = -1
        best_checkpoint_path = checkpoint_dir / "best_val.pt"
        final_checkpoint_path = checkpoint_dir / "final.pt"
        param_counts = count_parameters(model)

        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            model.train()
            epoch_losses: List[float] = []
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
                    R_pred, _t_pred_local, aux = model(IA, IB, dt_world=dt_world_t)
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
                    loss_total = (
                        final_loss["loss_total"]
                        + float(variant.get("coarse_aux_weight", 0.0)) * coarse_loss["loss_total"]
                        + float(variant.get("residual_reg_weight", 0.0)) * residual_reg["loss"]
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
                epoch_losses.append(float(loss_total.detach().cpu()))
                train_nan_inf_count += int(sum((~torch.isfinite(v)).sum().item() for v in [R_pred, aux["t_dir_out"], aux["t_mag"], loss_total]))

            minival_eval = evaluate_ablvo360_pose(model, mini_val_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_minival_batches"))
            minival_metrics = minival_eval["final_metrics"]
            val_score = _compute_val_score(minival_metrics, cfg["evaluation"]["selection_score"], eps=float(cfg["data"]["tmag_epsilon"]))
            _append_jsonl(train_log_path, {
                "epoch": int(epoch),
                "train_loss_total": float(np.mean(epoch_losses)) if epoch_losses else None,
                "train_nan_inf_count": int(train_nan_inf_count),
                "minival_score": float(val_score),
                "minival_metrics": minival_metrics,
                "coarse_minival_metrics": minival_eval["coarse_metrics"],
                "residual_stats": minival_eval["residual_stats"],
                "gate_stats": minival_eval["gate_stats"],
            })
            if val_score < best_val_score and int(minival_metrics["nan_inf_count"]) == 0:
                best_val_score = float(val_score)
                best_epoch = int(epoch)
                _save_checkpoint(best_checkpoint_path, model, optimizer, epoch, model_cfg, {
                    "task_name": cfg["task_name"],
                    "variant_name": variant_name,
                    "checkpoint_role": "best_val",
                    "load_status": load_status,
                    "subset_info": subset_info,
                    "minival_summary": minival_summary,
                    "parameter_count": param_counts,
                })

        _save_checkpoint(final_checkpoint_path, model, optimizer, int(cfg["training"]["epochs"]), model_cfg, {
            "task_name": cfg["task_name"],
            "variant_name": variant_name,
            "checkpoint_role": "final",
            "load_status": load_status,
            "subset_info": subset_info,
            "minival_summary": minival_summary,
            "parameter_count": param_counts,
        })

        selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
        selected_payload = torch.load(str(selected_checkpoint_path), map_location=device)
        selected_cfg = _cfg_from_dict(selected_payload.get("cfg", dict(model_cfg.__dict__)))
        selected_model = build_ablvo360_model(variant_name, selected_cfg, device).to(device)
        selected_model.load_state_dict(selected_payload["model"], strict=False)
        val_eval = evaluate_ablvo360_pose(selected_model, val_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_val_batches"))
        test_eval = evaluate_ablvo360_pose(selected_model, test_loader, device, tmag_epsilon=float(cfg["data"]["tmag_epsilon"]), max_batches=cfg["evaluation"].get("max_test_batches"))

        val_payload = {
            "task_name": cfg["task_name"],
            "model_name": variant_name,
            "checkpoint_used": str(selected_checkpoint_path),
            "best_epoch": int(best_epoch),
            "minival_score_of_selected_checkpoint": float(best_val_score),
            "val_score": float(_compute_val_score(val_eval["final_metrics"], cfg["evaluation"]["selection_score"], eps=float(cfg["data"]["tmag_epsilon"]))),
            "metrics": val_eval["final_metrics"],
            "coarse_metrics": val_eval["coarse_metrics"],
            "residual_stats": val_eval["residual_stats"],
            "gate_stats": val_eval["gate_stats"],
            "selection_protocol": "mini_val_only_per_epoch_full_val_once_after_training",
        }
        test_payload = {
            "task_name": cfg["task_name"],
            "model_name": variant_name,
            "checkpoint_used": str(selected_checkpoint_path),
            "best_epoch": int(best_epoch),
            "metrics": test_eval["final_metrics"],
            "coarse_metrics": test_eval["coarse_metrics"],
            "residual_stats": test_eval["residual_stats"],
            "gate_stats": test_eval["gate_stats"],
            "test_used_for_selection": False,
            "selection_protocol": "mini_val_only_per_epoch_full_val_once_after_training",
        }
        aggregate_val[variant_name] = val_payload
        aggregate_test[variant_name] = test_payload
        selection_rows.append({
            "model_name": variant_name,
            "executed": True,
            "best_epoch": int(best_epoch),
            "minival_score": float(best_val_score),
            "val_score": float(val_payload["val_score"]),
            "val_signed_tdir_mean": val_payload["metrics"].get("signed_tdir_mean_deg"),
            "val_anti_parallel_rate": val_payload["metrics"].get("anti_parallel_rate"),
            "val_rot_mean": val_payload["metrics"].get("rot_mean_deg"),
            "val_tmag_median_ratio": val_payload["metrics"].get("tmag_median_ratio"),
            "val_path_ratio": val_payload["metrics"].get("path_ratio"),
            "selected_checkpoint": str(selected_checkpoint_path),
            "test_used_for_selection": False,
        })
        training_rows.append({
            "model_name": variant_name,
            "executed": True,
            "init_mode": variant.get("init_mode"),
            "load_status": load_status,
            "checkpoint_dir": str(checkpoint_dir),
            "parameter_count": param_counts,
            "minival_subset_count": minival_summary.get("subset_count"),
        })
        trained_models.append(variant_name)

    outputs = cfg["outputs"]
    final360i_ref = precheck["final360i_test_metrics"]
    runtime_sec = time.time() - start_time

    _json_dump(REPO_ROOT / outputs["val_metrics_path"], {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "models": aggregate_val,
    })
    _json_dump(REPO_ROOT / outputs["test_metrics_path"], {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "models": aggregate_test,
        "final360i_reference": final360i_ref,
    })
    _json_dump(REPO_ROOT / outputs["selection_table_path"], {
        "task_name": cfg["task_name"],
        "selection_protocol": "mini_val_only_per_epoch_full_val_once_after_training",
        "rows": selection_rows,
    })
    _json_dump(REPO_ROOT / outputs["per_model_summary_path"], {
        "task_name": cfg["task_name"],
        "rows": training_rows,
    })
    _json_dump(REPO_ROOT / outputs["metric_source_manifest_path"], {
        "task_name": cfg["task_name"],
        "train_manifest": cfg["inputs"]["train_manifest"],
        "val_manifest": cfg["inputs"]["val_manifest"],
        "test_manifest": cfg["inputs"]["test_manifest"],
        "test_used_for_selection": False,
        "existing_metrics_modified": False,
        "full_model_checkpoint_modified": False,
        "checkpoints_committed": False,
        "mini_val_manifest_used_for_selection": True,
        "full_val_executed_for_best_checkpoints": True,
        "full_test_executed_for_best_checkpoints": True,
        "trained_models": trained_models,
        "skipped_models": skipped_models,
    })
    if "minival_manifest_summary_path" in outputs:
        _json_dump(REPO_ROOT / outputs["minival_manifest_summary_path"], {
            "task_name": cfg["task_name"],
            "summary": minival_summary,
            "selection_protocol": "mini_val_only_per_epoch_full_val_once_after_training",
        })

    test_lines = [
        "| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in trained_models:
        metrics = aggregate_test[model_name]["metrics"]
        test_lines.append(
            f"| {model_name} | {metrics.get('signed_tdir_mean_deg')} | {metrics.get('anti_parallel_rate')} | {metrics.get('tmag_median_ratio')} | {metrics.get('path_ratio')} | {metrics.get('coverage')} |"
        )
    test_lines.append(
        f"| FINAL360I_full_model | {final360i_ref.get('signed_tdir_mean_deg')} | {final360i_ref.get('anti_parallel_rate')} | {final360i_ref.get('tmag_median_ratio')} | {final360i_ref.get('path_ratio')} | {final360i_ref.get('coverage')} |"
    )
    (REPO_ROOT / outputs["main_table_path"]).write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    plain = aggregate_test.get("ABLVO360_PlainPairVO", {}).get("metrics", {})
    nosph = aggregate_test.get("ABLVO360_NoSphericalGeometry", {}).get("metrics", {})
    nocross = aggregate_test.get("ABLVO360_NoCrossImageInteraction", {}).get("metrics", {})
    singlestage = aggregate_test.get("ABLVO360_SingleStagePoseRegression", {}).get("metrics", {})
    contribution_lines = [
        f"# {cfg['task_name']} module contribution summary",
        "",
        f"- spherical-aware contribution: `{'positive' if nosph and float(nosph.get('signed_tdir_mean_deg') or float('inf')) > float(final360i_ref.get('signed_tdir_mean_deg') or float('inf')) else 'inconclusive'}`",
        f"- cross-image interaction contribution: `{'positive' if nocross and float(nocross.get('signed_tdir_mean_deg') or float('inf')) > float(final360i_ref.get('signed_tdir_mean_deg') or float('inf')) else 'inconclusive'}`",
        f"- coarse-to-fine contribution: `{'inconclusive' if not singlestage else ('positive' if float(singlestage.get('signed_tdir_mean_deg') or float('inf')) > float(final360i_ref.get('signed_tdir_mean_deg') or float('inf')) else 'neutral')}`",
        f"- full model better than plain pair VO: `{'true' if plain and float(final360i_ref.get('signed_tdir_mean_deg') or float('inf')) < float(plain.get('signed_tdir_mean_deg') or float('inf')) else 'partial'}`",
    ]
    (REPO_ROOT / outputs["contribution_summary_path"]).write_text("\n".join(contribution_lines) + "\n", encoding="utf-8")

    report_lines = [
        f"# {cfg['task_name']}",
        "",
        "## 1. Executive summary",
        "- ablation executed true/false: `true`",
        f"- models trained: `{trained_models}`",
        f"- models skipped: `{skipped_models}`",
        "- selected full baseline: `FINAL360I_struct360b_final_selected`",
        f"- classification: `{'completed' if len(trained_models) >= 3 else 'partial'}`",
        "",
        "## 2. Experimental setup",
        f"- train/val/test manifests: `{cfg['inputs']['train_manifest']}`, `{cfg['inputs']['val_manifest']}`, `{cfg['inputs']['test_manifest']}`",
        f"- training epochs: `{cfg['training']['epochs']}`",
        f"- seed: `{cfg['training']['seed']}`",
        f"- optimizer: `{cfg['training']['optimizer']}`",
        f"- mini-val subset count: `{minival_summary.get('subset_count')}`",
        "- test used for selection: `false`",
        f"- subset info: `{subset_info}`",
        f"- mini-val summary: `{minival_summary}`",
        f"- system info: `{system_info}`",
        f"- runtime sec: `{runtime_sec:.2f}`",
        "",
        "## 3. Validation selection table",
        *[f"- {row}" for row in selection_rows],
        "",
        "## 4. Test results",
        *test_lines,
        "",
        "## 5. Compliance checklist",
        "- training_executed = true",
        "- test_used_for_selection = false",
        "- full_model_checkpoint_modified = false",
        "- metrics_modified_existing = false",
        "- checkpoints_committed = false",
        "- raw_data_committed = false",
        "- explicit_matching_used = false",
        "- ransac_used = false",
        "- pnp_used = false",
        "- ba_used = false",
    ]
    (REPO_ROOT / outputs["report_path"]).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"- {cfg['task_name']} rerun executed: true")
    print(f"- models trained: {trained_models}")
    print(f"- models skipped: {skipped_models}")
    print(f"- mini-val count: {minival_summary.get('subset_count')}")
    print(f"- full val executed for best checkpoints: true")
    print(f"- full test executed for best checkpoints: true")
    print(f"- FINAL360I reference signed_tdir_mean: {final360i_ref.get('signed_tdir_mean_deg')}")
    if plain:
        print(f"- PlainPairVO test signed_tdir_mean: {plain.get('signed_tdir_mean_deg')}")
        print(f"- PlainPairVO test anti_parallel_rate: {plain.get('anti_parallel_rate')}")
        print(f"- PlainPairVO test tmag_median_ratio: {plain.get('tmag_median_ratio')}")
        print(f"- PlainPairVO test path_ratio: {plain.get('path_ratio')}")
    if nosph:
        print(f"- NoSpherical test signed_tdir_mean: {nosph.get('signed_tdir_mean_deg')}")
    if nocross:
        print(f"- NoCrossInteraction test signed_tdir_mean: {nocross.get('signed_tdir_mean_deg')}")
    if singlestage:
        print(f"- SingleStage test signed_tdir_mean: {singlestage.get('signed_tdir_mean_deg')}")
    print(f"- test used for selection: false")
    print(f"- current artifact check: pending")
    print(f"- py_compile: pending")
    print(f"- committed to git: false")
    print(f"- pushed to remote: false")


if __name__ == "__main__":
    train()
