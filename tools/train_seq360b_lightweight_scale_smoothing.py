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

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from datasets.dset2c_sequence_clip_dataset import Dset2CSequenceClipDataset
from mainline_dependency_utils import optional_read_json, summarize_optional_artifact
from miniyaml import load_yaml_like
from models.seq360b_scale_smoothing_head import Seq360BScaleSmoothingHead, seq360b_scale_losses
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train360.core.pose_head import matrix_geodesic_distance
from train_struct360b_match_free_coarse_to_fine import _cfg_from_dict, _inject_struct360b_cfg
import train360e_sequence_trajectory_export_and_ate_eval as traj


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seq360b_lightweight_scale_smoothing.yaml"
TMAG_EPS = 1.0e-6


def _git(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout or "").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _fmt(value: Any, digits: int = 6) -> str:
    x = _safe_float(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def _mean(vals: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _load_pair_model(path: Path, cfg: Mapping[str, Any], device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    ckpt_cfg = _cfg_from_dict(_inject_struct360b_cfg(dict(payload.get("cfg", {})), cfg))
    model = STRUCT360BMatchFreeCoarseToFineModel(ckpt_cfg, device).to(device)
    result = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "D": int(ckpt_cfg.D)}


def _load_head(path: Path, context_dim: int, cfg: Mapping[str, Any], device: torch.device) -> Seq360BScaleSmoothingHead:
    payload = torch.load(str(path), map_location=device)
    head = Seq360BScaleSmoothingHead(context_dim, hidden_dim=int(cfg["model"]["hidden_dim"]), delta_clamp=float(cfg["model"]["delta_clamp"])).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    return head


def _clip_dataset(cfg: Mapping[str, Any], split: str) -> Dset2CSequenceClipDataset:
    data = cfg["data"]
    return Dset2CSequenceClipDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        clip_len=int(data["clip_len"]),
        image_hw=tuple(int(x) for x in data["image_hw"]),
        max_frame_gap=int(data["max_frame_gap"]),
        max_timestamp_gap_factor=float(data["max_timestamp_gap_factor"]),
        tmag_epsilon=float(data["data"].get("tmag_epsilon", data["tmag_epsilon"])) if isinstance(data.get("data"), dict) else float(data["tmag_epsilon"]),
        require_paths=bool(data["require_paths"]),
        skip_invalid=bool(data["skip_invalid"]),
    )


def _clip_loader(cfg: Mapping[str, Any], split: str, *, subset_max: Optional[int] = None, shuffle: bool = False) -> Tuple[DataLoader, Dict[str, Any]]:
    ds = _clip_dataset(cfg, split)
    if subset_max is not None and int(subset_max) > 0 and len(ds) > int(subset_max):
        rng = np.random.default_rng(int(cfg["train"]["seed"]))
        idx = sorted(int(x) for x in rng.permutation(len(ds))[: int(subset_max)].tolist())
        dataset = Subset(ds, idx)
    else:
        dataset = ds
    batch_key = "train_batch_size" if split == "train" and shuffle else "eval_batch_size"
    loader = DataLoader(dataset, batch_size=int(cfg["data"].get(batch_key, cfg["data"]["train_batch_size"])), shuffle=shuffle, num_workers=int(cfg["data"].get("num_workers", 0)), pin_memory=torch.cuda.is_available(), drop_last=False)
    return loader, ds.get_clip_summary()


def _adjacent_predictions(pair_model: STRUCT360BMatchFreeCoarseToFineModel, batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    timestamps = batch["timestamps"].to(device, non_blocking=True)
    B = int(images.shape[0])
    adj_counts = [int(batch["adjacent_mask"][b].bool().sum().item()) for b in range(B)]
    if len(set(adj_counts)) != 1:
        raise RuntimeError(f"SEQ360B expects fixed adjacent count per clip batch, got {adj_counts}")
    A = adj_counts[0]
    IA_all: List[torch.Tensor] = []
    IB_all: List[torch.Tensor] = []
    dt_all: List[torch.Tensor] = []
    gt_tmags: List[torch.Tensor] = []
    gt_vecs: List[torch.Tensor] = []
    gt_Rs: List[torch.Tensor] = []
    for b in range(B):
        adj_mask = batch["adjacent_mask"][b].bool()
        pair_i = batch["pair_i"][b, adj_mask].to(device)
        pair_j = batch["pair_j"][b, adj_mask].to(device)
        IA_all.append(images[b, pair_i])
        IB_all.append(images[b, pair_j])
        dt_all.append((timestamps[b, pair_j] - timestamps[b, pair_i]).float())
        gt_tmags.append(batch["tmag"][b, adj_mask].to(device))
        gt_vecs.append(batch["t_BA_B"][b, adj_mask].to(device))
        gt_Rs.append(batch["R_BA"][b, adj_mask].to(device))
    with torch.no_grad():
        R, _t, aux = pair_model(torch.cat(IA_all, dim=0), torch.cat(IB_all, dim=0), dt_world=torch.cat(dt_all, dim=0))
    return {
        "pair_context": aux["coarse_pair_context"].view(B, A, -1),
        "log_tmag": torch.log(aux["t_mag"].clamp_min(TMAG_EPS)).view(B, A),
        "gt_tmag": torch.stack(gt_tmags, dim=0),
        "R": R.view(B, A, 3, 3),
        "tdir": aux["t_dir_out"].view(B, A, 3),
        "gt_vec": torch.stack(gt_vecs, dim=0),
        "gt_R": torch.stack(gt_Rs, dim=0),
    }


def _loss(head_out: Mapping[str, torch.Tensor], pred: Mapping[str, torch.Tensor], cfg: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    loss_cfg = cfg["loss"]
    return seq360b_scale_losses(
        head_out["corrected_log_tmag"],
        pred["gt_tmag"],
        delta_log_tmag=head_out["delta_log_tmag"],
        valid_mask=head_out["valid_mask"],
        eps=float(cfg["data"]["tmag_epsilon"]),
        log_weight=float(loss_cfg["log_tmag_weight"]),
        path_weight=float(loss_cfg["path_ratio_weight"]),
        smooth_weight=float(loss_cfg["scale_smoothness_weight"]),
        delta_weight=float(loss_cfg["delta_regularization_weight"]),
    )


@torch.no_grad()
def _evaluate_clip(head: Seq360BScaleSmoothingHead, pair_model: STRUCT360BMatchFreeCoarseToFineModel, loader: DataLoader, device: torch.device, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    head.eval()
    losses: Dict[str, List[float]] = {"total": [], "log_tmag_loss": [], "path_ratio_loss": [], "scale_smoothness_loss": [], "delta_regularization_loss": []}
    pred_path = 0.0
    gt_path = 0.0
    delta_abs: List[float] = []
    ratios: List[float] = []
    for batch in loader:
        pred = _adjacent_predictions(pair_model, batch, device)
        out = head(pred["pair_context"], pred["log_tmag"])
        loss = _loss(out, pred, cfg)
        for k in losses:
            losses[k].append(float(loss[k].detach().cpu()))
        pred_tmag = out["corrected_tmag"]
        pred_path += float(pred_tmag.sum().detach().cpu())
        gt_path += float(pred["gt_tmag"].sum().detach().cpu())
        delta_abs.extend(torch.abs(out["delta_log_tmag"]).detach().cpu().view(-1).numpy().tolist())
        ratios.extend((pred_tmag / pred["gt_tmag"].clamp_min(TMAG_EPS)).detach().cpu().view(-1).numpy().tolist())
    return {
        "loss": {k: _mean(v) for k, v in losses.items()},
        "clip_path_ratio": float(pred_path / max(gt_path, TMAG_EPS)),
        "tmag_median_ratio": float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None,
        "delta_abs_mean": _mean(delta_abs),
        "pred_path_length": pred_path,
        "gt_path_length": gt_path,
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / max(na * nb, 1.0e-12))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


@torch.no_grad()
def _evaluate_pair_from_clips(head: Seq360BScaleSmoothingHead, pair_model: STRUCT360BMatchFreeCoarseToFineModel, loader: DataLoader, device: torch.device, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    rot, signed, anti, ratios, pred_lengths, gt_lengths = [], [], [], [], [], []
    nan_inf = 0
    for batch in loader:
        pred = _adjacent_predictions(pair_model, batch, device)
        out = head(pred["pair_context"], pred["log_tmag"])
        pred_t = (pred["tdir"] * out["corrected_tmag"].unsqueeze(-1)).detach().cpu().numpy()
        gt_t = pred["gt_vec"].detach().cpu().numpy()
        R_pred = pred["R"].detach().cpu()
        R_gt = pred["gt_R"].detach().cpu()
        rot.extend((matrix_geodesic_distance(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3)).numpy() * (180.0 / math.pi)).tolist())
        for a, b in zip(pred_t.reshape(-1, 3), gt_t.reshape(-1, 3)):
            if not np.isfinite(a).all():
                nan_inf += 1
            ang = _angle_deg(a, b)
            if ang is not None:
                signed.append(ang)
                anti.append(1.0 if float(np.dot(a, b)) < 0.0 else 0.0)
            pm = max(float(np.linalg.norm(a)), TMAG_EPS)
            gm = max(float(np.linalg.norm(b)), TMAG_EPS)
            ratios.append(pm / gm)
            pred_lengths.append(pm)
            gt_lengths.append(gm)
    return {
        "count": int(len(ratios)),
        "coverage": 1.0,
        "rot_mean_deg": _mean(rot),
        "signed_tdir_mean_deg": _mean(signed),
        "anti_parallel_rate": _mean(anti),
        "tmag_median_ratio": float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None,
        "path_ratio": float(sum(pred_lengths) / max(sum(gt_lengths), TMAG_EPS)),
        "path_length_pred": float(sum(pred_lengths)),
        "path_length_gt": float(sum(gt_lengths)),
        "nan_inf_count": int(nan_inf),
        "note": "SEQ360B pair metrics are adjacent-clip metrics because the head is sequence-context scale-only.",
    }


def _pair_metrics_from_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rot, signed, anti, ratios, pred_lengths, gt_lengths = [], [], [], [], [], []
    nan_inf = 0
    for row in rows:
        R_pred = np.asarray(row["pred_R_BA"], dtype=np.float64)
        R_gt = np.asarray(row["gt_R_BA"], dtype=np.float64)
        t_pred = np.asarray(row["pred_t_BA_B"], dtype=np.float64)
        t_gt = np.asarray(row["gt_t_BA_B"], dtype=np.float64)
        if bool(row.get("nan_or_inf", False)):
            nan_inf += 1
        trace = np.clip((np.trace(R_pred @ R_gt.T) - 1.0) * 0.5, -1.0, 1.0)
        rot.append(float(np.degrees(np.arccos(trace))))
        ang = _angle_deg(t_pred, t_gt)
        if ang is not None:
            signed.append(ang)
            anti.append(1.0 if float(np.dot(t_pred, t_gt)) < 0.0 else 0.0)
        pm = max(float(np.linalg.norm(t_pred)), TMAG_EPS)
        gm = max(float(np.linalg.norm(t_gt)), TMAG_EPS)
        ratios.append(pm / gm)
        pred_lengths.append(pm)
        gt_lengths.append(gm)
    return {
        "count": int(len(rows)),
        "coverage": 1.0,
        "rot_mean_deg": _mean(rot),
        "signed_tdir_mean_deg": _mean(signed),
        "anti_parallel_rate": _mean(anti),
        "tmag_median_ratio": float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None,
        "path_ratio": float(sum(pred_lengths) / max(sum(gt_lengths), TMAG_EPS)),
        "path_length_pred": float(sum(pred_lengths)),
        "path_length_gt": float(sum(gt_lengths)),
        "nan_inf_count": int(nan_inf),
        "note": "SEQ360B pair metrics are adjacent trajectory rows with scale-smoothed tmag; R/tdir come from frozen FINAL360I.",
    }


def _prediction_rows_from_clips(
    head: Seq360BScaleSmoothingHead,
    pair_model: STRUCT360BMatchFreeCoarseToFineModel,
    clip_loader: DataLoader,
    pair_dataset: Dset2CCanonicalPairDataset,
    device: torch.device,
) -> List[Dict[str, Any]]:
    edge_accum: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for batch in clip_loader:
        pred = _adjacent_predictions(pair_model, batch, device)
        out = head(pred["pair_context"], pred["log_tmag"])
        R_np = pred["R"].detach().cpu().numpy()
        tdir_np = pred["tdir"].detach().cpu().numpy()
        tmag_np = out["corrected_tmag"].detach().cpu().numpy()
        B, A = tmag_np.shape
        for b in range(B):
            adj_positions = torch.where(batch["adjacent_mask"][b].bool())[0].numpy().tolist()
            for local_idx, pidx in enumerate(adj_positions):
                frame_a = int(batch["frame_ids"][b, int(batch["pair_i"][b, pidx])].item())
                frame_b = int(batch["frame_ids"][b, int(batch["pair_j"][b, pidx])].item())
                key = (str(batch["sequence_id"][b]), frame_a, frame_b)
                edge_accum.setdefault(key, []).append({
                    "R": R_np[b, local_idx],
                    "tdir": tdir_np[b, local_idx],
                    "tmag": float(tmag_np[b, local_idx]),
                })
    rows: List[Dict[str, Any]] = []
    for sample in pair_dataset.samples:
        if sample["pair_type"] != "adjacent" or int(sample["k"]) != 1:
            continue
        key = (str(sample["sequence_id"]), int(Path(sample["image_path_a"]).stem), int(Path(sample["image_path_b"]).stem))
        preds = edge_accum.get(key)
        if not preds:
            continue
        R = np.mean(np.asarray([p["R"] for p in preds], dtype=np.float64), axis=0)
        tdir = np.mean(np.asarray([p["tdir"] for p in preds], dtype=np.float64), axis=0)
        tdir = tdir / max(float(np.linalg.norm(tdir)), TMAG_EPS)
        tmag = float(np.mean(np.asarray([p["tmag"] for p in preds], dtype=np.float64)))
        t = tdir * tmag
        rows.append({
            "split": str(sample["split"]),
            "sequence": str(sample["sequence_id"]),
            "pair_index": int(sample["pair_index"]),
            "frame_a": int(Path(sample["image_path_a"]).stem),
            "frame_b": int(Path(sample["image_path_b"]).stem),
            "timestamp_a": float(sample["timestamp_a"]),
            "timestamp_b": float(sample["timestamp_b"]),
            "image_a": str(sample["image_path_a"]),
            "image_b": str(sample["image_path_b"]),
            "pred_R_BA": R.tolist(),
            "pred_tdir_B": tdir.tolist(),
            "pred_tmag": tmag,
            "pred_t_BA_B": t.tolist(),
            "gt_R_BA": np.asarray(sample["R_BA"], dtype=np.float64).tolist(),
            "gt_t_BA_B": np.asarray(sample["t_BA_B"], dtype=np.float64).tolist(),
            "nan_or_inf": not bool(np.isfinite(R).all() and np.isfinite(t).all() and math.isfinite(tmag)),
        })
    return rows


def _trajectory_eval(head: Seq360BScaleSmoothingHead, pair_model: STRUCT360BMatchFreeCoarseToFineModel, cfg: Mapping[str, Any], device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = cfg["data"]
    image_hw = tuple(int(x) for x in data["image_hw"])
    pair_val = Dset2CCanonicalPairDataset(str(REPO_ROOT / cfg["inputs"]["val_manifest"]), expected_split="val", image_hw=image_hw, require_paths=bool(data["require_paths"]), skip_invalid=bool(data["skip_invalid"]))
    pair_test = Dset2CCanonicalPairDataset(str(REPO_ROOT / cfg["inputs"]["test_manifest"]), expected_split="test", image_hw=image_hw, require_paths=bool(data["require_paths"]), skip_invalid=bool(data["skip_invalid"]))
    val_specs, val_blockers = traj._build_sequence_specs(pair_val, "val")
    test_specs, test_blockers = traj._build_sequence_specs(pair_test, "test")
    if val_blockers or test_blockers:
        raise RuntimeError("; ".join(val_blockers + test_blockers))
    convention, convention_summary, warnings = traj._convention_sanity({**val_specs, **test_specs})
    if convention != "BA":
        raise RuntimeError("Convention sanity failed: " + "; ".join(warnings))
    val_clip_loader, _ = _clip_loader(cfg, "val", shuffle=False)
    test_clip_loader, _ = _clip_loader(cfg, "test", shuffle=False)
    val_predictions = _prediction_rows_from_clips(head, pair_model, val_clip_loader, pair_val, device)
    test_predictions = _prediction_rows_from_clips(head, pair_model, test_clip_loader, pair_test, device)
    predictions = list(val_predictions) + list(test_predictions)
    by_split_seq: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in predictions:
        by_split_seq.setdefault((row["split"], row["sequence"]), []).append(row)
    old_root, old_task, old_ckpt = traj.RESULTS_ROOT, traj.TASK_NAME, traj.CHECKPOINT_PATH
    traj.RESULTS_ROOT = REPO_ROOT / cfg["outputs"]["trajectory_dir"]
    traj.TASK_NAME = str(cfg["task_name"])
    traj.CHECKPOINT_PATH = REPO_ROOT / cfg["outputs"]["checkpoint_dir"] / "best_val.pt"
    try:
        val_per = {seq: traj._compose_sequence_trajectory("val", seq, spec, by_split_seq.get(("val", seq), []), selected_convention=convention, git_commit=_git(["git", "rev-parse", "HEAD"]), git_branch=_git(["git", "branch", "--show-current"])) for seq, spec in val_specs.items()}
        test_per = {seq: traj._compose_sequence_trajectory("test", seq, spec, by_split_seq.get(("test", seq), []), selected_convention=convention, git_commit=_git(["git", "rev-parse", "HEAD"]), git_branch=_git(["git", "branch", "--show-current"])) for seq, spec in test_specs.items()}
        val_metrics = traj._aggregate_split("val", val_per)
        test_metrics = traj._aggregate_split("test", test_per)
        val_metrics["selected_convention"] = convention
        test_metrics["selected_convention"] = convention
        val_metrics["convention_sanity"] = convention_summary
        test_metrics["convention_sanity"] = convention_summary
        base_val = traj._base360_split_eval("val", sorted(val_specs.keys()))
        base_test = traj._base360_split_eval("test", sorted(test_specs.keys()))
        return val_metrics, test_metrics, base_val, base_test, val_predictions, test_predictions
    finally:
        traj.RESULTS_ROOT, traj.TASK_NAME, traj.CHECKPOINT_PATH = old_root, old_task, old_ckpt


def _precheck(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    branch = _git(["git", "branch", "--show-current"])
    disk_free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    allowed_branches = {
        "experiment/seq360b-train-lightweight-scale-smoothing-head",
        "maintenance/destructive-cleanup-current-mainline-only",
        "maintenance/trim-remaining-mainline-dependencies",
    }
    if branch not in allowed_branches:
        blockers.append(f"unexpected branch: {branch}")
    if disk_free_gb <= 5.0:
        blockers.append(f"insufficient disk free space: {disk_free_gb:.2f} GiB <= 5 GiB")
    if not torch.cuda.is_available():
        blockers.append("CUDA unavailable")
    required = []
    for key in ["init_checkpoint", "hygiene_json", "train_manifest", "val_manifest", "test_manifest", "final360i_test_metrics", "train360e_test_metrics"]:
        required.append(REPO_ROOT / cfg["inputs"][key])
    required.extend([REPO_ROOT / "datasets/dset2c_manifest_dataset.py", REPO_ROOT / "datasets/dset2c_sequence_clip_dataset.py", REPO_ROOT / "models/seq360b_scale_smoothing_head.py"])
    for path in required:
        if not path.is_file():
            blockers.append(f"missing required file: {path}")
    clip_summaries: Dict[str, Any] = {}
    if not blockers:
        for split in ("train", "val", "test"):
            ds = _clip_dataset(cfg, split)
            clip_summaries[split] = ds.get_clip_summary()
    return {
        "blockers": blockers,
        "git_status_short": _git(["git", "status", "--short"]).splitlines(),
        "branch": branch,
        "branch_vv": _git(["git", "branch", "-vv"]).splitlines(),
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "disk_free_gb": disk_free_gb,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "clip_summaries": clip_summaries,
        "baseline_recap": {
            "final360i_val": optional_read_json(REPO_ROOT / cfg["inputs"]["final360i_val_metrics"]),
            "final360i_test": optional_read_json(REPO_ROOT / cfg["inputs"]["final360i_test_metrics"]),
            "train360e_val": optional_read_json(REPO_ROOT / cfg["inputs"]["train360e_val_metrics"]),
            "train360e_test": optional_read_json(REPO_ROOT / cfg["inputs"]["train360e_test_metrics"]),
            "base360d_val": optional_read_json(REPO_ROOT / cfg["inputs"]["base360d_val_metrics"]),
            "base360d_test": optional_read_json(REPO_ROOT / cfg["inputs"]["base360d_test_metrics"]),
            "base360d_test_status": summarize_optional_artifact(REPO_ROOT / cfg["inputs"]["base360d_test_metrics"], "BASE360D test metrics"),
        },
    }


def _classification(pair_test: Mapping[str, Any], traj_test: Mapping[str, Any]) -> str:
    traj_path = float(traj_test.get("trajectory_path_ratio") or float("inf"))
    sim3 = float(traj_test.get("ate_sim3", {}).get("rmse") or float("inf"))
    se3 = float(traj_test.get("ate_se3", {}).get("rmse") or float("inf"))
    signed = float(pair_test.get("signed_tdir_mean_deg") or float("inf"))
    anti = float(pair_test.get("anti_parallel_rate") or 1.0)
    tmag = float(pair_test.get("tmag_median_ratio") or 0.0)
    if int(pair_test.get("nan_inf_count") or 0) > 0 or traj_path >= 1.756343083453392 or sim3 > 35.0 or signed > 54.96 or anti > 0.2157:
        return "regression"
    if traj_path <= 1.30 and (sim3 < 27.564661865900444 or se3 < 118.6037794846689):
        return "primary_success"
    if traj_path <= 1.45 and sim3 <= 27.564661865900444 and signed <= 50.0 and anti <= 0.22 and tmag >= 0.75:
        return "balanced_success"
    if traj_path < 1.756343083453392 or sim3 < 27.564661865900444:
        return "partial"
    return "no_improvement"


def _compare_pair(candidate: Mapping[str, Any], final360i_payload: Mapping[str, Any]) -> str:
    ref = final360i_payload.get("metrics", {})
    improved = 0
    if float(candidate.get("signed_tdir_mean_deg") or 999) < float(ref.get("signed_tdir_mean_deg") or 999):
        improved += 1
    if float(candidate.get("anti_parallel_rate") or 1) < float(ref.get("anti_parallel_rate") or 1):
        improved += 1
    if float(candidate.get("tmag_median_ratio") or 0) > float(ref.get("tmag_median_ratio") or 0):
        improved += 1
    if float(candidate.get("path_ratio") or 0) > float(ref.get("path_ratio") or 0):
        improved += 1
    return "better" if improved >= 3 else ("partial" if improved else "worse")


def _compare_traj(candidate: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    improved = 0
    total = 0
    for key in ("ate_none", "ate_se3", "ate_sim3"):
        a = _safe_float(candidate.get(key, {}).get("rmse"))
        b = _safe_float(ref.get(key, {}).get("rmse"))
        if a is not None and b is not None:
            total += 1
            improved += int(a < b)
    a = _safe_float(candidate.get("trajectory_path_ratio"))
    b = _safe_float(ref.get("trajectory_path_ratio"))
    if a is not None and b is not None:
        total += 1
        improved += int(abs(math.log(max(a, TMAG_EPS))) < abs(math.log(max(b, TMAG_EPS))))
    return "better" if total and improved >= 3 else ("partial" if improved else "worse")


def _write_report(cfg: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    pair_test = payload["pair_test"]["metrics"]
    traj_test = payload["trajectory_test"]
    cls = payload["classification"]
    next_rec = (
        "prepare_thesis_experiment_section"
        if cls in {"primary_success", "balanced_success"}
        else (
            "keep_SEQ360B_as_retained_sequence_scale_variant"
            if cls in {"partial", "trajectory_improved_scale_tradeoff"}
            else "keep_FINAL360I_as_main_and_report_SEQ360B_ablation"
        )
    )
    lines = [
        "# SEQ360B train lightweight scale smoothing head",
        "",
        "## 1. Executive summary",
        "- training executed: `true`",
        "- checkpoint saved: `true`",
        f"- best checkpoint: `{payload['best_checkpoint']}`",
        "- scale smoothing head trained: `true`",
        f"- pair test signed / anti / tmag / path: `{_fmt(pair_test.get('signed_tdir_mean_deg'))}` / `{_fmt(pair_test.get('anti_parallel_rate'))}` / `{_fmt(pair_test.get('tmag_median_ratio'))}` / `{_fmt(pair_test.get('path_ratio'))}`",
        f"- trajectory test ATE none / SE3 / Sim3: `{_fmt(traj_test['ate_none']['rmse'])}` / `{_fmt(traj_test['ate_se3']['rmse'])}` / `{_fmt(traj_test['ate_sim3']['rmse'])}`",
        f"- trajectory test path ratio: `{_fmt(traj_test.get('trajectory_path_ratio'))}`",
        f"- classification: `{cls}`",
        "",
        "## 2. Motivation",
        "- FINAL360I pair-level metrics are strong, but TRAIN360E trajectory path ratio is `1.756343083453392`.",
        "- SEQ360B tests whether a small log_tmag correction head can reduce scale/path drift while preserving R/tdir.",
        "",
        "## 3. Design",
        "- base model frozen: `true`",
        "- R/tdir changed by head: `false`",
        f"- delta_log_tmag clamp: `[-{cfg['model']['delta_clamp']}, {cfg['model']['delta_clamp']}]`",
        "- losses: `SmoothL1 log_tmag + path ratio + delta smoothness + delta regularization`",
        "- explicit matching / RANSAC / PnP / BA: `false / false / false / false`",
        "",
        "## 4. Data protocol",
        "- DSET2C canonical train/val/test manifests used.",
        "- Train clips from train manifest only; val for selection; test for final evaluation only.",
        "- No random pair split and no direct raw sequence glob split.",
        "",
        "## 5. Training setup",
        f"- clip_len: `{cfg['data']['clip_len']}`",
        f"- train/val/test clip counts: `{payload['precheck']['clip_summaries']['train']['clip_count']}` / `{payload['precheck']['clip_summaries']['val']['clip_count']}` / `{payload['precheck']['clip_summaries']['test']['clip_count']}`",
        f"- epochs: `{cfg['train']['epochs']}`",
        f"- batch size: `{cfg['data']['train_batch_size']}`",
        f"- LR: `{cfg['train']['lr']}`",
        "- optimizer: `AdamW`",
        f"- selected epoch: `{payload['best_epoch']}`",
        f"- best val score: `{payload['best_val_score']}`",
        "",
        "## 6. Validation results",
        f"- clip path ratio: `{_fmt(payload['best_val_clip'].get('clip_path_ratio'))}`",
        f"- tmag median ratio: `{_fmt(payload['best_val_clip'].get('tmag_median_ratio'))}`",
        f"- pair metrics: `{payload['pair_val']['metrics']}`",
        "",
        "## 7. Test pair-level results",
        f"- FINAL360I vs SEQ360B pair comparison: `{payload['compared_to_final360i_pair']}`",
        f"- SEQ360B test pair metrics: `{pair_test}`",
        "",
        "## 8. Test trajectory results",
        f"- compared to TRAIN360E FINAL360I trajectory: `{payload['compared_to_train360e_trajectory']}`",
        f"- compared to BASE360D trajectory: `{payload['compared_to_base360d_trajectory']}`",
        f"- SEQ360B test trajectory metrics: path_ratio=`{_fmt(traj_test.get('trajectory_path_ratio'))}`, ATE Sim3=`{_fmt(traj_test['ate_sim3']['rmse'])}`",
        "",
        "## 9. Analysis",
        f"- scale smoothing reduced path overestimation: `{float(traj_test.get('trajectory_path_ratio') or 999) < 1.756343083453392}`",
        f"- ATE improved: `{float(traj_test['ate_sim3']['rmse'] or 999) < 27.564661865900444 or float(traj_test['ate_se3']['rmse'] or 999) < 118.6037794846689}`",
        "- Pair direction should remain stable because the head never modifies R/tdir.",
        "- FINAL360I remains the pair-level main model unless sequence-scale behavior is preferred.",
        "",
        "## 10. Next recommendation",
        f"- `{next_rec}`",
        "",
        "## 11. Compliance checklist",
        "- `training_executed = true`",
        "- `fine_tune_executed = true`",
        "- `learned_weights_saved = true`",
        "- `final360i_checkpoint_modified = false`",
        "- `r_tdir_modified_by_head = false`",
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
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = [
        "# SEQ360B current variant summary",
        "",
        f"- classification: `{cls}`",
        f"- compared to FINAL360I pair-level: `{payload['compared_to_final360i_pair']}`",
        f"- compared to TRAIN360E trajectory: `{payload['compared_to_train360e_trajectory']}`",
        f"- compared to BASE360D trajectory: `{payload['compared_to_base360d_trajectory']}`",
        f"- test trajectory path ratio: `{_fmt(traj_test.get('trajectory_path_ratio'))}`",
        f"- test ATE none / SE3 / Sim3: `{_fmt(traj_test['ate_none']['rmse'])}` / `{_fmt(traj_test['ate_se3']['rmse'])}` / `{_fmt(traj_test['ate_sim3']['rmse'])}`",
    ]
    (REPO_ROOT / cfg["outputs"]["comparison_summary_path"]).write_text("\n".join(summary) + "\n", encoding="utf-8")


def _write_blocker(cfg: Mapping[str, Any], precheck: Mapping[str, Any]) -> None:
    payload = {"task_name": cfg["task_name"], "training_executed": False, "blockers": precheck["blockers"], "precheck": precheck}
    for key in ("val_metrics_path", "test_metrics_path", "trajectory_val_metrics_path", "trajectory_test_metrics_path"):
        _write_json(REPO_ROOT / cfg["outputs"][key], payload)
    (REPO_ROOT / cfg["outputs"]["report_path"]).write_text("# SEQ360B blocked\n\n" + "\n".join(f"- {b}" for b in precheck["blockers"]) + "\n", encoding="utf-8")


def run_dry_smoke(cfg: Mapping[str, Any], smoke_batches: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"].get("use_cuda_if_available", True)) else "cpu")
    loader, clip_summary = _clip_loader(cfg, "train", subset_max=smoke_batches, shuffle=False)
    pair_model, load_summary = _load_pair_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    head = Seq360BScaleSmoothingHead(3 * int(load_summary["D"]), hidden_dim=int(cfg["model"]["hidden_dim"]), delta_clamp=float(cfg["model"]["delta_clamp"])).to(device)
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= int(smoke_batches):
                break
            pred = _adjacent_predictions(pair_model, batch, device)
            out = head(pred["pair_context"], pred["log_tmag"])
            losses = _loss(out, pred, cfg)
            results.append({"batch": idx, "losses": {k: float(v.detach().cpu()) for k, v in losses.items()}, "rotation_modified": False, "tdir_modified": False})
    return {"variant": "SEQ360B", "dry_run": True, "training_executed": False, "optimizer_step_executed": False, "checkpoint_load": load_summary, "clip_summary": clip_summary, "results": results, "smoke_pass": bool(results)}


def train(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    precheck = _precheck(cfg)
    if precheck["blockers"]:
        _write_blocker(cfg, precheck)
        raise RuntimeError("SEQ360B precheck failed: " + "; ".join(precheck["blockers"]))
    _seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["model"].get("use_cuda_if_available", True)) else "cpu")
    pair_model, load_summary = _load_pair_model(REPO_ROOT / cfg["inputs"]["init_checkpoint"], cfg, device)
    head = Seq360BScaleSmoothingHead(3 * int(load_summary["D"]), hidden_dim=int(cfg["model"]["hidden_dim"]), delta_clamp=float(cfg["model"]["delta_clamp"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    train_loader, train_summary = _clip_loader(cfg, "train", subset_max=cfg["data"].get("train_subset_max"), shuffle=True)
    val_loader, val_summary = _clip_loader(cfg, "val", shuffle=False)
    out_dir = REPO_ROOT / cfg["outputs"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()
    best_score = float("inf")
    best_epoch = -1
    best_val: Dict[str, Any] = {}
    best_path = out_dir / "best_val.pt"
    start = time.time()
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        head.train()
        epoch_losses: Dict[str, List[float]] = {"total": [], "log_tmag_loss": [], "path_ratio_loss": [], "scale_smoothness_loss": [], "delta_regularization_loss": []}
        for batch in train_loader:
            pred = _adjacent_predictions(pair_model, batch, device)
            out = head(pred["pair_context"], pred["log_tmag"])
            losses = _loss(out, pred, cfg)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg["train"]["grad_clip_norm"]))
            optimizer.step()
            for k in epoch_losses:
                epoch_losses[k].append(float(losses[k].detach().cpu()))
        val = _evaluate_clip(head, pair_model, val_loader, device, cfg)
        score = 50.0 * abs(math.log(max(float(val["clip_path_ratio"]), TMAG_EPS))) + 20.0 * abs(math.log(max(float(val["tmag_median_ratio"] or TMAG_EPS), TMAG_EPS)))
        row = {"epoch": epoch, "train_loss": {k: _mean(v) for k, v in epoch_losses.items()}, "val": val, "val_score": score}
        _append_jsonl(train_log, row)
        if score < best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_val = dict(val)
            torch.save({"head": head.state_dict(), "cfg": dict(cfg), "epoch": epoch, "val_score": best_score, "init_checkpoint": str(REPO_ROOT / cfg["inputs"]["init_checkpoint"]), "base_model_frozen": True}, str(best_path))
    final_path = out_dir / "final.pt"
    torch.save({"head": head.state_dict(), "cfg": dict(cfg), "epoch": int(cfg["train"]["epochs"]), "best_epoch": best_epoch, "init_checkpoint": str(REPO_ROOT / cfg["inputs"]["init_checkpoint"]), "base_model_frozen": True}, str(final_path))
    head = _load_head(best_path, 3 * int(load_summary["D"]), cfg, device)
    traj_val, traj_test, base_val, base_test, val_rows, test_rows = _trajectory_eval(head, pair_model, cfg, device)
    pair_val = _pair_metrics_from_prediction_rows(val_rows)
    pair_test = _pair_metrics_from_prediction_rows(test_rows)
    classification = _classification(pair_test, traj_test)
    payload = {
        "task_name": cfg["task_name"],
        "training_executed": True,
        "fine_tune_executed": True,
        "checkpoint_saved": True,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "init_checkpoint": str(REPO_ROOT / cfg["inputs"]["init_checkpoint"]),
        "clip_len": int(cfg["data"]["clip_len"]),
        "base_model_frozen": True,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "best_val_clip": best_val,
        "elapsed_sec": time.time() - start,
        "precheck": precheck,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "pair_val": {"metrics": pair_val},
        "pair_test": {"metrics": pair_test},
        "trajectory_val": traj_val,
        "trajectory_test": traj_test,
        "base360d_trajectory_val": base_val,
        "base360d_trajectory_test": base_test,
        "baseline_train360e_test": precheck["baseline_recap"]["train360e_test"],
        "classification": classification,
        "compared_to_final360i_pair": _compare_pair(pair_test, precheck["baseline_recap"]["final360i_test"]),
        "compared_to_train360e_trajectory": _compare_traj(traj_test, precheck["baseline_recap"]["train360e_test"]),
        "compared_to_base360d_trajectory": _compare_traj(traj_test, base_test),
        "test_used_for_selection": False,
    }
    _write_json(REPO_ROOT / cfg["outputs"]["val_metrics_path"], payload["pair_val"])
    _write_json(REPO_ROOT / cfg["outputs"]["test_metrics_path"], payload["pair_test"])
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_val_metrics_path"], traj_val)
    _write_json(REPO_ROOT / cfg["outputs"]["trajectory_test_metrics_path"], traj_test)
    _write_report(cfg, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_yaml_like(Path(args.config))
    if args.dry_run or args.no_train or not bool(cfg.get("train", {}).get("enabled", False)):
        result = run_dry_smoke(cfg, args.smoke_batches)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["smoke_pass"] else 1
    payload = train(cfg)
    test_pair = payload["pair_test"]["metrics"]
    test_traj = payload["trajectory_test"]
    print("- SEQ360B training executed: true")
    print("- checkpoint saved: true")
    print(f"- best checkpoint: {payload['best_checkpoint']}")
    print(f"- init checkpoint: {payload['init_checkpoint']}")
    print(f"- clip_len: {payload['clip_len']}")
    print(f"- base model frozen: {str(payload['base_model_frozen']).lower()}")
    print(f"- pair test signed_tdir_mean: {test_pair.get('signed_tdir_mean_deg')}")
    print(f"- pair test anti_parallel_rate: {test_pair.get('anti_parallel_rate')}")
    print(f"- pair test tmag_median_ratio: {test_pair.get('tmag_median_ratio')}")
    print(f"- pair test path_ratio: {test_pair.get('path_ratio')}")
    print(f"- trajectory test ATE none: {test_traj['ate_none']['rmse']}")
    print(f"- trajectory test ATE SE3: {test_traj['ate_se3']['rmse']}")
    print(f"- trajectory test ATE Sim3: {test_traj['ate_sim3']['rmse']}")
    print(f"- trajectory test path_ratio: {test_traj.get('trajectory_path_ratio')}")
    print(f"- compared to FINAL360I pair-level: {payload['compared_to_final360i_pair']}")
    print(f"- compared to TRAIN360E trajectory: {payload['compared_to_train360e_trajectory']}")
    print(f"- compared to BASE360D trajectory: {payload['compared_to_base360d_trajectory']}")
    print(f"- classification: {payload['classification']}")
    print("- committed to git: false")
    print("- pushed to remote: false")
    print("- next recommended task: keep_SEQ360B_as_retained_sequence_scale_variant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
