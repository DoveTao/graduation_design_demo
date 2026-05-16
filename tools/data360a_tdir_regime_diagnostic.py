#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from datasets.dset2c_sequence_clip_dataset import Dset2CSequenceClipDataset
import yaml
from models.seq360b_scale_smoothing_head import Seq360BScaleSmoothingHead
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train360.core.config import Config
from train360.core.pose_head import matrix_geodesic_distance


TMAG_EPS = 1.0e-6
DATA_ROOT = REPO_ROOT / "external_baselines" / "results" / "data360a_tdir_regime_diagnostic"
FINAL360I_CKPT = REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "best_val.pt"
SEQ360B_CKPT = REPO_ROOT / "checkpoints" / "SEQ360B_lightweight_scale_smoothing" / "best_val.pt"
SEQ360B_CFG = REPO_ROOT / "configs" / "seq360b_lightweight_scale_smoothing.yaml"
MANIFESTS = {
    "val": REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_val.jsonl",
    "test": REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_test.jsonl",
}
METRIC_SOURCES = {
    "FINAL360I_test": REPO_ROOT / "reports" / "FINAL360I_metrics_test.json",
    "TRAIN360E_test": REPO_ROOT / "reports" / "TRAIN360E_metrics_test.json",
    "SEQ360B_test": REPO_ROOT / "reports" / "SEQ360B_metrics_test.json",
    "SEQ360B_trajectory_test": REPO_ROOT / "reports" / "SEQ360B_trajectory_metrics_test.json",
    "BASE360D_component_alignment": REPO_ROOT / "reports" / "BASE360D_component_metric_alignment.md",
    "CURRENT_MAINLINE": REPO_ROOT / "CURRENT_MAINLINE.md",
    "DELIVERY_HANDOFF": REPO_ROOT / "DELIVERY_HANDOFF.md",
    "RELEASE3_delivery_handoff": REPO_ROOT / "reports" / "RELEASE3_delivery_handoff.md",
    "THESIS361_claims": REPO_ROOT / "thesis" / "THESIS361_claims_and_caveats_checklist.md",
}
OUTPUTS = {
    "report": REPO_ROOT / "reports" / "DATA360A_translation_direction_regime_diagnostic_and_reweighting.md",
    "regime_buckets": REPO_ROOT / "reports" / "DATA360A_regime_buckets.json",
    "worst_scenes": REPO_ROOT / "reports" / "DATA360A_worst_scenes_sequences.json",
    "antiparallel": REPO_ROOT / "reports" / "DATA360A_antiparallel_analysis.json",
    "reweighting": REPO_ROOT / "reports" / "DATA360A_reweighting_recommendations.md",
    "source_manifest": REPO_ROOT / "reports" / "DATA360A_metric_source_manifest.json",
    "bucket_tables": REPO_ROOT / "reports" / "DATA360A_bucket_tables.md",
    "comparison": REPO_ROOT / "reports" / "DATA360A_final360i_vs_seq360b_bucket_comparison.json",
}


def _run(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), cwd=REPO_ROOT, text=True).strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cfg_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 4) -> str:
    x = _safe_float(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def _mean(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _median(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else None


def _percentile(values: Iterable[float], q: float) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else None


def _angle_deg_from_vectors(a: np.ndarray, b: np.ndarray, *, absolute: bool) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / max(na * nb, 1.0e-12))
    if absolute:
        c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _rot_angle_deg(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _frame_id_from_path(path: str) -> int:
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else -1


def _scene_from_sequence(sequence_id: str) -> str:
    seq = str(sequence_id)
    if "/" in seq:
        return seq.split("/", 1)[0]
    return seq


def _to_dt_tensor(meta_dt: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(meta_dt):
        return meta_dt.to(device=device, dtype=torch.float32).view(-1)
    return torch.tensor([float(x) for x in meta_dt], device=device, dtype=torch.float32)


def _load_pair_model(path: Path, device: torch.device) -> Tuple[STRUCT360BMatchFreeCoarseToFineModel, Config, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    cfg = _cfg_from_dict(payload["cfg"])
    model = STRUCT360BMatchFreeCoarseToFineModel(cfg, device).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg, {
        "checkpoint": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "epoch": int(payload.get("epoch", -1)),
    }


def _load_seq360b_head(path: Path, device: torch.device) -> Tuple[Seq360BScaleSmoothingHead, Dict[str, Any], Mapping[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    cfg = payload["cfg"]
    head = Seq360BScaleSmoothingHead(
        context_dim=3 * int(payload["cfg"]["model"]["struct360b"].get("residual_hidden_dim", 256)) if False else 3 * 256,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        delta_clamp=float(cfg["model"]["delta_clamp"]),
    ).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, {
        "checkpoint": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "val_score": _safe_float(payload.get("val_score")),
        "init_checkpoint": str(payload.get("init_checkpoint", "")),
    }, cfg


def _load_seq360b_head_with_context(path: Path, context_dim: int, device: torch.device) -> Tuple[Seq360BScaleSmoothingHead, Dict[str, Any], Mapping[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    cfg = payload["cfg"]
    head = Seq360BScaleSmoothingHead(
        context_dim=context_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        delta_clamp=float(cfg["model"]["delta_clamp"]),
    ).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, {
        "checkpoint": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "val_score": _safe_float(payload.get("val_score")),
        "init_checkpoint": str(payload.get("init_checkpoint", "")),
    }, cfg


def _dataset_for_split(split: str, image_hw: Tuple[int, int]) -> Dset2CCanonicalPairDataset:
    return Dset2CCanonicalPairDataset(
        str(MANIFESTS[split]),
        expected_split=split,
        image_hw=image_hw,
        require_paths=True,
        skip_invalid=True,
    )


def _final360i_rows_for_split(
    model: STRUCT360BMatchFreeCoarseToFineModel,
    *,
    split: str,
    image_hw: Tuple[int, int],
    device: torch.device,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    dataset = _dataset_for_split(split, image_hw)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    rows: List[Dict[str, Any]] = []
    print(f"[DATA360A] FINAL360I {split}: {len(dataset)} pairs, batch_size={batch_size}", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"[DATA360A] FINAL360I {split}: batch {batch_idx}/{len(loader)}", flush=True)
            IA = batch["IA"].to(device, non_blocking=True)
            IB = batch["IB"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)
            t_gt_vec = batch["t_gt_vec"].to(device, non_blocking=True)
            dt_world = _to_dt_tensor(batch["meta"]["dt_world"], device=device)
            R_pred, _t_local, aux = model(IA, IB, dt_world=dt_world)

            rot_err_deg = (
                matrix_geodesic_distance(R_pred.float(), R_gt.float()).detach().cpu().numpy() * (180.0 / math.pi)
            )
            R_pred_np = R_pred.detach().cpu().numpy()
            R_gt_np = R_gt.detach().cpu().numpy()
            tdir_pred_np = aux["t_dir_out"].detach().cpu().numpy()
            tmag_pred_np = aux["t_mag"].detach().view(-1).cpu().numpy()
            t_gt_np = t_gt_vec.detach().cpu().numpy()
            gt_tmag_np = torch.linalg.norm(t_gt_vec.float(), dim=-1).detach().cpu().numpy()
            meta = batch["meta"]
            batch_size_now = IA.shape[0]
            for i in range(batch_size_now):
                pred_tdir = np.asarray(tdir_pred_np[i], dtype=np.float64)
                gt_tvec = np.asarray(t_gt_np[i], dtype=np.float64)
                gt_tmag = max(float(gt_tmag_np[i]), TMAG_EPS)
                pred_tmag = max(float(tmag_pred_np[i]), TMAG_EPS)
                gt_tdir = gt_tvec / max(float(np.linalg.norm(gt_tvec)), TMAG_EPS)
                pred_tvec = pred_tdir * pred_tmag
                cos_signed = float(np.dot(pred_tdir, gt_tdir))
                signed = _angle_deg_from_vectors(pred_tdir, gt_tdir, absolute=False)
                unsigned = _angle_deg_from_vectors(pred_tdir, gt_tdir, absolute=True)
                finite = bool(
                    np.isfinite(R_pred_np[i]).all()
                    and np.isfinite(pred_tdir).all()
                    and math.isfinite(pred_tmag)
                    and np.isfinite(gt_tvec).all()
                )
                image_a = str(meta["image_path_a"][i])
                image_b = str(meta["image_path_b"][i])
                sequence = str(meta["sequence_id"][i])
                rows.append(
                    {
                        "model": "FINAL360I",
                        "split": split,
                        "scene": _scene_from_sequence(sequence),
                        "sequence": sequence,
                        "sample_id": f"{split}:{sequence}:{int(meta['pair_index'][i])}",
                        "pair_id": int(meta["pair_index"][i]),
                        "frame_i": _frame_id_from_path(image_a),
                        "frame_j": _frame_id_from_path(image_b),
                        "k": int(meta["k"][i]),
                        "pair_type": str(meta["pair_type"][i]),
                        "timestamp_a": float(meta["timestamp_a"][i]),
                        "timestamp_b": float(meta["timestamp_b"][i]),
                        "frame_distance": abs(_frame_id_from_path(image_b) - _frame_id_from_path(image_a)),
                        "gt_tmag": gt_tmag,
                        "gt_rot_angle_deg": _rot_angle_deg(np.asarray(R_gt_np[i], dtype=np.float64)),
                        "pred_rot_angle_error_deg": float(rot_err_deg[i]),
                        "signed_tdir_error_deg": signed,
                        "unsigned_tdir_error_deg": unsigned,
                        "cos_tdir_signed": cos_signed,
                        "anti_parallel": bool(cos_signed < 0.0),
                        "pred_tmag": pred_tmag,
                        "tmag_ratio": float(pred_tmag / gt_tmag),
                        "log_tmag_error": float(abs(math.log(pred_tmag) - math.log(gt_tmag))),
                        "finite": finite,
                        "scale_collapse": bool((pred_tmag / gt_tmag) < 0.1),
                        "scale_explosion": bool((pred_tmag / gt_tmag) > 10.0),
                        "pred_path_contribution": pred_tmag,
                        "gt_path_contribution": gt_tmag,
                    }
                )
    return rows


def _seq360b_clip_dataset(cfg: Mapping[str, Any], split: str) -> Dset2CSequenceClipDataset:
    data_cfg = cfg["data"]
    return Dset2CSequenceClipDataset(
        str(REPO_ROOT / cfg["inputs"][f"{split}_manifest"]),
        expected_split=split,
        clip_len=int(data_cfg["clip_len"]),
        image_hw=tuple(int(x) for x in data_cfg["image_hw"]),
        max_frame_gap=int(data_cfg["max_frame_gap"]),
        max_timestamp_gap_factor=float(data_cfg["max_timestamp_gap_factor"]),
        tmag_epsilon=float(data_cfg["tmag_epsilon"]),
        require_paths=bool(data_cfg["require_paths"]),
        skip_invalid=bool(data_cfg["skip_invalid"]),
    )


def _seq360b_adjacent_predictions(
    pair_model: STRUCT360BMatchFreeCoarseToFineModel,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    timestamps = batch["timestamps"].to(device, non_blocking=True)
    B = int(images.shape[0])
    adj_counts = [int(batch["adjacent_mask"][b].bool().sum().item()) for b in range(B)]
    if len(set(adj_counts)) != 1:
        raise RuntimeError(f"Expected fixed adjacent count per clip batch, got {adj_counts}")
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


def _seq360b_rows_for_split(
    pair_model: STRUCT360BMatchFreeCoarseToFineModel,
    head: Seq360BScaleSmoothingHead,
    *,
    split: str,
    seq_cfg: Mapping[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    clip_ds = _seq360b_clip_dataset(seq_cfg, split)
    clip_loader = DataLoader(
        clip_ds,
        batch_size=max(int(seq_cfg["data"].get("eval_batch_size", 8)), 12),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    pair_ds = _dataset_for_split(split, tuple(int(x) for x in seq_cfg["data"]["image_hw"]))
    edge_accum: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    print(f"[DATA360A] SEQ360B {split}: {len(clip_ds)} clips, batch_size={clip_loader.batch_size}", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(clip_loader):
            if batch_idx % 25 == 0:
                print(f"[DATA360A] SEQ360B {split}: batch {batch_idx}/{len(clip_loader)}", flush=True)
            pred = _seq360b_adjacent_predictions(pair_model, batch, device)
            out = head(pred["pair_context"], pred["log_tmag"])
            R_np = pred["R"].detach().cpu().numpy()
            tdir_np = pred["tdir"].detach().cpu().numpy()
            tmag_np = out["corrected_tmag"].detach().cpu().numpy()
            B, A = tmag_np.shape
            for b in range(B):
                adj_positions = torch.where(batch["adjacent_mask"][b].bool())[0].tolist()
                for local_idx, pidx in enumerate(adj_positions):
                    frame_a = int(batch["frame_ids"][b, int(batch["pair_i"][b, pidx])].item())
                    frame_b = int(batch["frame_ids"][b, int(batch["pair_j"][b, pidx])].item())
                    key = (str(batch["sequence_id"][b]), frame_a, frame_b)
                    edge_accum[key].append(
                        {
                            "R": np.asarray(R_np[b, local_idx], dtype=np.float64),
                            "tdir": np.asarray(tdir_np[b, local_idx], dtype=np.float64),
                            "tmag": float(tmag_np[b, local_idx]),
                        }
                    )

    rows: List[Dict[str, Any]] = []
    for sample in pair_ds.samples:
        if str(sample["pair_type"]) != "adjacent" or int(sample["k"]) != 1:
            continue
        frame_a = _frame_id_from_path(str(sample["image_path_a"]))
        frame_b = _frame_id_from_path(str(sample["image_path_b"]))
        preds = edge_accum.get((str(sample["sequence_id"]), frame_a, frame_b), [])
        if not preds:
            continue
        R_pred = np.mean(np.asarray([p["R"] for p in preds], dtype=np.float64), axis=0)
        tdir_pred = np.mean(np.asarray([p["tdir"] for p in preds], dtype=np.float64), axis=0)
        tdir_pred = tdir_pred / max(float(np.linalg.norm(tdir_pred)), TMAG_EPS)
        pred_tmag = float(np.mean(np.asarray([p["tmag"] for p in preds], dtype=np.float64)))
        gt_tvec = np.asarray(sample["t_BA_B"], dtype=np.float64)
        gt_tmag = max(float(sample["tmag"]), TMAG_EPS)
        gt_tdir = gt_tvec / max(float(np.linalg.norm(gt_tvec)), TMAG_EPS)
        cos_signed = float(np.dot(tdir_pred, gt_tdir))
        rows.append(
            {
                "model": "SEQ360B",
                "split": split,
                "scene": _scene_from_sequence(str(sample["sequence_id"])),
                "sequence": str(sample["sequence_id"]),
                "sample_id": f"{split}:{sample['sequence_id']}:{int(sample['pair_index'])}",
                "pair_id": int(sample["pair_index"]),
                "frame_i": frame_a,
                "frame_j": frame_b,
                "k": 1,
                "pair_type": "adjacent",
                "timestamp_a": float(sample["timestamp_a"]),
                "timestamp_b": float(sample["timestamp_b"]),
                "frame_distance": abs(frame_b - frame_a),
                "gt_tmag": gt_tmag,
                "gt_rot_angle_deg": _rot_angle_deg(np.asarray(sample["R_BA"], dtype=np.float64)),
                "pred_rot_angle_error_deg": _rot_angle_deg(R_pred @ np.asarray(sample["R_BA"], dtype=np.float64).T),
                "signed_tdir_error_deg": _angle_deg_from_vectors(tdir_pred, gt_tdir, absolute=False),
                "unsigned_tdir_error_deg": _angle_deg_from_vectors(tdir_pred, gt_tdir, absolute=True),
                "cos_tdir_signed": cos_signed,
                "anti_parallel": bool(cos_signed < 0.0),
                "pred_tmag": pred_tmag,
                "tmag_ratio": float(pred_tmag / gt_tmag),
                "log_tmag_error": float(abs(math.log(max(pred_tmag, TMAG_EPS)) - math.log(gt_tmag))),
                "finite": bool(np.isfinite(R_pred).all() and np.isfinite(tdir_pred).all() and math.isfinite(pred_tmag)),
                "scale_collapse": bool((pred_tmag / gt_tmag) < 0.1),
                "scale_explosion": bool((pred_tmag / gt_tmag) > 10.0),
                "pred_path_contribution": pred_tmag,
                "gt_path_contribution": gt_tmag,
            }
        )
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    signed = [float(r["signed_tdir_error_deg"]) for r in rows if _safe_float(r.get("signed_tdir_error_deg")) is not None]
    unsigned = [float(r["unsigned_tdir_error_deg"]) for r in rows if _safe_float(r.get("unsigned_tdir_error_deg")) is not None]
    rot = [float(r["pred_rot_angle_error_deg"]) for r in rows if _safe_float(r.get("pred_rot_angle_error_deg")) is not None]
    ratios = [float(r["tmag_ratio"]) for r in rows if _safe_float(r.get("tmag_ratio")) is not None]
    log_tmag = [float(r["log_tmag_error"]) for r in rows if _safe_float(r.get("log_tmag_error")) is not None]
    anti = [1.0 if bool(r.get("anti_parallel")) else 0.0 for r in rows if r.get("anti_parallel") is not None]
    pred_lengths = [float(r["pred_path_contribution"]) for r in rows if _safe_float(r.get("pred_path_contribution")) is not None]
    gt_lengths = [float(r["gt_path_contribution"]) for r in rows if _safe_float(r.get("gt_path_contribution")) is not None]
    finite_flags = [1.0 if bool(r.get("finite", False)) else 0.0 for r in rows]
    return {
        "count": int(len(rows)),
        "signed_tdir_mean_deg": _mean(signed),
        "signed_tdir_median_deg": _median(signed),
        "unsigned_tdir_mean_deg": _mean(unsigned),
        "anti_parallel_rate": _mean(anti),
        "rot_mean_deg": _mean(rot),
        "tmag_median_ratio": _median(ratios),
        "tmag_mean_ratio": _mean(ratios),
        "log_tmag_mae": _mean(log_tmag),
        "path_ratio": float(sum(pred_lengths) / max(sum(gt_lengths), TMAG_EPS)) if gt_lengths else None,
        "finite_rate": _mean(finite_flags),
        "scale_collapse_rate": _mean([1.0 if float(v) < 0.1 else 0.0 for v in ratios]) if ratios else None,
        "scale_explosion_rate": _mean([1.0 if float(v) > 10.0 else 0.0 for v in ratios]) if ratios else None,
    }


def _quantile_bucket_specs(values: Sequence[float], quantiles: Sequence[float]) -> List[Dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return []
    bounds = [float(np.percentile(arr, q)) for q in quantiles]
    specs = []
    for i in range(len(bounds) - 1):
        low = bounds[i]
        high = bounds[i + 1]
        specs.append(
            {
                "label": f"q{int(quantiles[i])}_{int(quantiles[i+1])}",
                "low": low,
                "high": high,
                "include_high": i == len(bounds) - 2,
            }
        )
    return specs


def _tmag_named_specs(values: Sequence[float]) -> List[Dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return []
    p10, p25, p50, p75 = [float(np.percentile(arr, q)) for q in (10, 25, 50, 75)]
    return [
        {"label": "very_small", "low": -float("inf"), "high": p10, "include_high": True},
        {"label": "small", "low": p10, "high": p25, "include_high": True},
        {"label": "medium", "low": p25, "high": p50, "include_high": True},
        {"label": "large", "low": p50, "high": p75, "include_high": True},
        {"label": "very_large", "low": p75, "high": float("inf"), "include_high": True},
    ]


def _rot_named_specs() -> List[Dict[str, Any]]:
    return [
        {"label": "small_rotation", "low": 0.0, "high": 15.0, "include_high": False},
        {"label": "medium_rotation", "low": 15.0, "high": 45.0, "include_high": False},
        {"label": "large_rotation", "low": 45.0, "high": 90.0, "include_high": False},
        {"label": "extreme_rotation", "low": 90.0, "high": float("inf"), "include_high": True},
    ]


def _spec_match(value: float, spec: Mapping[str, Any]) -> bool:
    low = float(spec["low"])
    high = float(spec["high"])
    include_high = bool(spec.get("include_high", False))
    if value < low:
        return False
    if include_high:
        return value <= high
    return value < high


def _bucketize(rows: Sequence[Mapping[str, Any]], field: str, specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    bucket_rows: List[Dict[str, Any]] = []
    for spec in specs:
        selected = [r for r in rows if _safe_float(r.get(field)) is not None and _spec_match(float(r[field]), spec)]
        bucket_rows.append(
            {
                "bucket": str(spec["label"]),
                "field": field,
                "low": spec["low"],
                "high": spec["high"],
                **_summarize_rows(selected),
            }
        )
    return bucket_rows


def _bucketize_2d(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    x_specs: Sequence[Mapping[str, Any]],
    y_field: str,
    y_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for x_spec in x_specs:
        for y_spec in y_specs:
            selected = [
                r
                for r in rows
                if _safe_float(r.get(x_field)) is not None
                and _safe_float(r.get(y_field)) is not None
                and _spec_match(float(r[x_field]), x_spec)
                and _spec_match(float(r[y_field]), y_spec)
            ]
            table.append(
                {
                    "tmag_bucket": str(x_spec["label"]),
                    "rotation_bucket": str(y_spec["label"]),
                    **_summarize_rows(selected),
                }
            )
    return table


def _group_stats(rows: Sequence[Mapping[str, Any]], group_field: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    results = []
    total_gt = sum(float(r.get("gt_path_contribution") or 0.0) for r in rows)
    for group_name, group_rows in grouped.items():
        summary = _summarize_rows(group_rows)
        summary[group_field] = group_name
        summary["path_contribution_gt_share"] = (
            float(sum(float(r.get("gt_path_contribution") or 0.0) for r in group_rows) / max(total_gt, TMAG_EPS))
            if total_gt > 0
            else None
        )
        results.append(summary)
    results.sort(key=lambda x: (-(x.get("count") or 0), x[group_field]))
    if limit is not None:
        return results[: int(limit)]
    return results


def _worst_rankings(rows: Sequence[Mapping[str, Any]], group_field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped = _group_stats(rows, group_field)
    by_signed = sorted(grouped, key=lambda x: (-(x.get("signed_tdir_mean_deg") or -1.0), -(x.get("count") or 0)))[:10]
    by_anti = sorted(grouped, key=lambda x: (-(x.get("anti_parallel_rate") or -1.0), -(x.get("count") or 0)))[:10]
    by_tmag = sorted(grouped, key=lambda x: (-abs(math.log(max(float(x.get("tmag_median_ratio") or 1.0), TMAG_EPS))), -(x.get("count") or 0)))[:10]
    return {
        "worst_by_signed_tdir_mean": by_signed,
        "worst_by_anti_parallel_rate": by_anti,
        "worst_by_tmag_ratio_deviation": by_tmag,
    }


def _anti_parallel_analysis(rows: Sequence[Mapping[str, Any]], tmag_specs: Sequence[Mapping[str, Any]], rot_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    anti_rows = [r for r in rows if bool(r.get("anti_parallel"))]
    total_anti = len(anti_rows)
    tmag_bucket_counts = []
    for spec in tmag_specs:
        selected = [r for r in anti_rows if _safe_float(r.get("gt_tmag")) is not None and _spec_match(float(r["gt_tmag"]), spec)]
        tmag_bucket_counts.append({"bucket": str(spec["label"]), "count": len(selected), "share": float(len(selected) / max(total_anti, 1))})
    rot_bucket_counts = []
    for spec in rot_specs:
        selected = [r for r in anti_rows if _safe_float(r.get("gt_rot_angle_deg")) is not None and _spec_match(float(r["gt_rot_angle_deg"]), spec)]
        rot_bucket_counts.append({"bucket": str(spec["label"]), "count": len(selected), "share": float(len(selected) / max(total_anti, 1))})
    by_scene = Counter(str(r["scene"]) for r in anti_rows).most_common(10)
    by_sequence = Counter(str(r["sequence"]) for r in anti_rows).most_common(10)
    return {
        "anti_parallel_count": int(total_anti),
        "anti_parallel_rate": float(total_anti / max(len(rows), 1)),
        "gt_tmag_stats": {
            "mean": _mean(float(r["gt_tmag"]) for r in anti_rows if _safe_float(r.get("gt_tmag")) is not None),
            "median": _median(float(r["gt_tmag"]) for r in anti_rows if _safe_float(r.get("gt_tmag")) is not None),
        },
        "gt_rot_angle_stats": {
            "mean": _mean(float(r["gt_rot_angle_deg"]) for r in anti_rows if _safe_float(r.get("gt_rot_angle_deg")) is not None),
            "median": _median(float(r["gt_rot_angle_deg"]) for r in anti_rows if _safe_float(r.get("gt_rot_angle_deg")) is not None),
        },
        "tmag_bucket_counts": tmag_bucket_counts,
        "rotation_bucket_counts": rot_bucket_counts,
        "top_scenes": [{"scene": k, "count": v} for k, v in by_scene],
        "top_sequences": [{"sequence": k, "count": v} for k, v in by_sequence],
        "pred_tmag_anomaly_share": {
            "collapse": _mean(1.0 if bool(r.get("scale_collapse")) else 0.0 for r in anti_rows),
            "explosion": _mean(1.0 if bool(r.get("scale_explosion")) else 0.0 for r in anti_rows),
        },
    }


def _compare_models_by_bucket(final_rows: Sequence[Mapping[str, Any]], seq_rows: Sequence[Mapping[str, Any]], tmag_specs: Sequence[Mapping[str, Any]], rot_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final_adj = [r for r in final_rows if str(r.get("pair_type")) == "adjacent" and int(r.get("k", 1)) == 1]
    seq_adj = list(seq_rows)
    comparison = {
        "overall_final360i_adjacent": _summarize_rows(final_adj),
        "overall_seq360b_adjacent": _summarize_rows(seq_adj),
        "tmag_buckets_final360i_adjacent": _bucketize(final_adj, "gt_tmag", tmag_specs),
        "tmag_buckets_seq360b_adjacent": _bucketize(seq_adj, "gt_tmag", tmag_specs),
        "rotation_buckets_final360i_adjacent": _bucketize(final_adj, "gt_rot_angle_deg", rot_specs),
        "rotation_buckets_seq360b_adjacent": _bucketize(seq_adj, "gt_rot_angle_deg", rot_specs),
    }
    final_overall = comparison["overall_final360i_adjacent"]
    seq_overall = comparison["overall_seq360b_adjacent"]
    tdir_delta = None
    anti_delta = None
    path_delta = None
    tmag_delta = None
    if _safe_float(final_overall.get("signed_tdir_mean_deg")) is not None and _safe_float(seq_overall.get("signed_tdir_mean_deg")) is not None:
        tdir_delta = float(seq_overall["signed_tdir_mean_deg"]) - float(final_overall["signed_tdir_mean_deg"])
    if _safe_float(final_overall.get("anti_parallel_rate")) is not None and _safe_float(seq_overall.get("anti_parallel_rate")) is not None:
        anti_delta = float(seq_overall["anti_parallel_rate"]) - float(final_overall["anti_parallel_rate"])
    if _safe_float(final_overall.get("path_ratio")) is not None and _safe_float(seq_overall.get("path_ratio")) is not None:
        path_delta = float(seq_overall["path_ratio"]) - float(final_overall["path_ratio"])
    if _safe_float(final_overall.get("tmag_median_ratio")) is not None and _safe_float(seq_overall.get("tmag_median_ratio")) is not None:
        tmag_delta = float(seq_overall["tmag_median_ratio"]) - float(final_overall["tmag_median_ratio"])
    comparison["delta_overall_seq360b_minus_final360i_adjacent"] = {
        "signed_tdir_mean_deg": tdir_delta,
        "anti_parallel_rate": anti_delta,
        "path_ratio": path_delta,
        "tmag_median_ratio": tmag_delta,
    }
    comparison["scale_path_only_interpretation"] = bool(
        path_delta is not None
        and abs(path_delta) > 0.1
        and tdir_delta is not None
        and abs(tdir_delta) < 0.5
        and anti_delta is not None
        and abs(anti_delta) < 0.01
    )
    return comparison


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _recommendation_from_analysis(
    final_tmag_named: Sequence[Mapping[str, Any]],
    final_rot_named: Sequence[Mapping[str, Any]],
    anti_payload: Mapping[str, Any],
    seq_compare: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    worst_tmag = max(final_tmag_named, key=lambda x: float(x.get("signed_tdir_mean_deg") or -1.0)) if final_tmag_named else {}
    worst_rot = max(final_rot_named, key=lambda x: float(x.get("signed_tdir_mean_deg") or -1.0)) if final_rot_named else {}
    anti_tmag_top = max(anti_payload.get("tmag_bucket_counts", []), key=lambda x: int(x.get("count", 0)), default={})
    anti_rot_top = max(anti_payload.get("rotation_bucket_counts", []), key=lambda x: int(x.get("count", 0)), default={})
    scale_path_only = bool(seq_compare and seq_compare.get("scale_path_only_interpretation"))

    conservative = [
        "Lightly down-weight very_small gt_tmag pairs in the tdir loss.",
        "Keep full supervision on medium/large gt_tmag pairs.",
        "Add a small anti-parallel penalty before changing the translation head.",
    ]
    aggressive = [
        "Strongly down-weight very_small gt_tmag plus extreme_rotation pairs for tdir supervision.",
        "Up-weight medium/large gt_tmag pairs with non-extreme rotation for tdir loss.",
        "Consider explicit observability weighting or a lightweight tdir confidence branch.",
    ]
    filtering = [
        "Do not filter by default; use filtering only if a tiny subset is clearly geometry-unobservable or corrupted.",
        "If filtering is used, report the exact scene/sequence subset and proportion removed.",
    ]

    if scale_path_only and anti_tmag_top.get("bucket") == "very_small":
        next_task = "proceed_to_FINAL360K_observability_weighting"
    elif anti_tmag_top.get("bucket") == "very_small" or anti_rot_top.get("bucket") == "extreme_rotation":
        next_task = "proceed_to_FINAL360K_observability_weighting"
    elif worst_tmag.get("bucket") == "medium" and worst_rot.get("bucket") in {"medium_rotation", "large_rotation"}:
        next_task = "proceed_to_FINAL360J_loss_reweight"
    elif scale_path_only:
        next_task = "proceed_to_FINAL360J_loss_reweight"
    else:
        next_task = "proceed_to_FINAL360L_head_factorization"

    return {
        "conservative_weighting": conservative,
        "aggressive_weighting": aggressive,
        "diagnostic_filtering": filtering,
        "recommended_next_experiment": next_task,
        "rationale": {
            "worst_tmag_bucket": worst_tmag.get("bucket"),
            "worst_rotation_bucket": worst_rot.get("bucket"),
            "anti_parallel_tmag_concentration": anti_tmag_top.get("bucket"),
            "anti_parallel_rotation_concentration": anti_rot_top.get("bucket"),
            "seq360b_scale_path_only": scale_path_only,
        },
    }


def _write_bucket_tables(regime_payload: Mapping[str, Any]) -> None:
    lines = ["# DATA360A bucket tables", ""]
    for model_name, model_payload in regime_payload.items():
        lines.append(f"## {model_name}")
        lines.append("")
        tmag_rows = [
            [
                row["bucket"],
                row["count"],
                _fmt(row.get("signed_tdir_mean_deg")),
                _fmt(row.get("anti_parallel_rate")),
                _fmt(row.get("rot_mean_deg")),
                _fmt(row.get("tmag_median_ratio")),
                _fmt(row.get("path_ratio")),
            ]
            for row in model_payload["tmag_named_buckets"]
        ]
        lines.append("### tmag named buckets")
        lines.append("")
        lines.append(_markdown_table(
            ["bucket", "count", "signed_tdir_mean", "anti_parallel", "rot_mean", "tmag_median_ratio", "path_ratio"],
            tmag_rows,
        ))
        lines.append("")
        rot_rows = [
            [
                row["bucket"],
                row["count"],
                _fmt(row.get("signed_tdir_mean_deg")),
                _fmt(row.get("anti_parallel_rate")),
                _fmt(row.get("rot_mean_deg")),
                _fmt(row.get("tmag_median_ratio")),
            ]
            for row in model_payload["rotation_named_buckets"]
        ]
        lines.append("### rotation named buckets")
        lines.append("")
        lines.append(_markdown_table(
            ["bucket", "count", "signed_tdir_mean", "anti_parallel", "rot_mean", "tmag_median_ratio"],
            rot_rows,
        ))
        lines.append("")
    OUTPUTS["bucket_tables"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reweighting_report(payload: Mapping[str, Any]) -> None:
    lines = [
        "# DATA360A reweighting recommendations",
        "",
        "## Conservative weighting",
        *[f"- {x}" for x in payload["conservative_weighting"]],
        "",
        "## Aggressive weighting",
        *[f"- {x}" for x in payload["aggressive_weighting"]],
        "",
        "## Diagnostic filtering",
        *[f"- {x}" for x in payload["diagnostic_filtering"]],
        "",
        "## Recommended next experiment",
        f"- `{payload['recommended_next_experiment']}`",
        "",
        "## Rationale",
        f"- worst tmag bucket: `{payload['rationale'].get('worst_tmag_bucket')}`",
        f"- worst rotation bucket: `{payload['rationale'].get('worst_rotation_bucket')}`",
        f"- anti-parallel tmag concentration: `{payload['rationale'].get('anti_parallel_tmag_concentration')}`",
        f"- anti-parallel rotation concentration: `{payload['rationale'].get('anti_parallel_rotation_concentration')}`",
        f"- SEQ360B scale/path only: `{payload['rationale'].get('seq360b_scale_path_only')}`",
    ]
    OUTPUTS["reweighting"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_report(
    *,
    source_manifest: Mapping[str, Any],
    overall_recap: Mapping[str, Any],
    regime_payload: Mapping[str, Any],
    worst_payload: Mapping[str, Any],
    anti_payload: Mapping[str, Any],
    seq_compare: Optional[Mapping[str, Any]],
    recommendations: Mapping[str, Any],
    prediction_dump_generated: bool,
) -> None:
    final_all = regime_payload["FINAL360I_all_pairs"]
    final_adj = regime_payload["FINAL360I_adjacent_only"]
    main_conclusion = (
        "Small-motion and/or high-rotation regimes dominate translation-direction failure, so observability-aware reweighting is a better next step than another scale-only sequence variant."
        if recommendations["recommended_next_experiment"] == "proceed_to_FINAL360K_observability_weighting"
        else "The dominant failure mode is broad pair-level translation-direction instability, so direct tdir loss reweighting should be tried before more complex sequence-level changes."
    )
    lines = [
        "# DATA360A translation direction regime diagnostic and reweighting",
        "",
        "## 1. Executive summary",
        "- diagnostic executed: `true`",
        "- training executed: `false`",
        "- checkpoint modified: `false`",
        "- model structure changed: `false`",
        "- splits analyzed: `val`, `test`",
        f"- models analyzed: `{', '.join(source_manifest['models_analyzed'])}`",
        f"- prediction dump generated: `{str(prediction_dump_generated).lower()}`",
        f"- main conclusion: {main_conclusion}",
        "",
        "## 2. Data sources",
        "- canonical manifests:",
        f"  - `{MANIFESTS['val']}`",
        f"  - `{MANIFESTS['test']}`",
        "- checkpoints:",
        f"  - `{FINAL360I_CKPT}`",
        f"  - `{SEQ360B_CKPT}`",
        "- metric JSON / report files:",
        *[f"  - `{path}`" for path in source_manifest["metric_source_files"]],
        "",
        "## 3. Overall metrics recap",
        f"- FINAL360I pair-level: signed_tdir_mean=`{_fmt(overall_recap['FINAL360I_test'].get('signed_tdir_mean_deg'))}`, anti_parallel=`{_fmt(overall_recap['FINAL360I_test'].get('anti_parallel_rate'))}`, tmag_median_ratio=`{_fmt(overall_recap['FINAL360I_test'].get('tmag_median_ratio'))}`, path_ratio=`{_fmt(overall_recap['FINAL360I_test'].get('path_ratio'))}`",
        f"- TRAIN360E trajectory: ATE none / SE3 / Sim3=`{_fmt(overall_recap['TRAIN360E_test'].get('ate_none', {}).get('rmse'))}` / `{_fmt(overall_recap['TRAIN360E_test'].get('ate_se3', {}).get('rmse'))}` / `{_fmt(overall_recap['TRAIN360E_test'].get('ate_sim3', {}).get('rmse'))}`, path_ratio=`{_fmt(overall_recap['TRAIN360E_test'].get('trajectory_path_ratio'))}`",
        f"- SEQ360B pair-level: signed_tdir_mean=`{_fmt(overall_recap['SEQ360B_test'].get('signed_tdir_mean_deg'))}`, anti_parallel=`{_fmt(overall_recap['SEQ360B_test'].get('anti_parallel_rate'))}`, tmag_median_ratio=`{_fmt(overall_recap['SEQ360B_test'].get('tmag_median_ratio'))}`, path_ratio=`{_fmt(overall_recap['SEQ360B_test'].get('path_ratio'))}`",
        f"- SEQ360B trajectory: ATE Sim3=`{_fmt(overall_recap['SEQ360B_trajectory_test'].get('ate_sim3', {}).get('rmse'))}`, trajectory_path_ratio=`{_fmt(overall_recap['SEQ360B_trajectory_test'].get('trajectory_path_ratio'))}`",
        "- BASE360D caveat: component metrics are trajectory-derived and are not identical to native pair-level predictions.",
        "",
        "## 4. tmag bucket analysis",
        "The following regime tables pool `val + test` samples for diagnostic density unless otherwise noted.",
        _markdown_table(
            ["bucket", "count", "signed_tdir_mean", "anti_parallel", "rot_mean", "tmag_median_ratio", "path_ratio"],
            [
                [
                    row["bucket"],
                    row["count"],
                    _fmt(row.get("signed_tdir_mean_deg")),
                    _fmt(row.get("anti_parallel_rate")),
                    _fmt(row.get("rot_mean_deg")),
                    _fmt(row.get("tmag_median_ratio")),
                    _fmt(row.get("path_ratio")),
                ]
                for row in final_all["tmag_named_buckets"]
            ],
        ),
        "",
        f"Conclusion: the worst gt_tmag regime by signed_tdir_mean is `{max(final_all['tmag_named_buckets'], key=lambda x: float(x.get('signed_tdir_mean_deg') or -1.0))['bucket']}`.",
        "",
        "## 5. rotation bucket analysis",
        _markdown_table(
            ["bucket", "count", "signed_tdir_mean", "anti_parallel", "rot_mean", "tmag_median_ratio"],
            [
                [
                    row["bucket"],
                    row["count"],
                    _fmt(row.get("signed_tdir_mean_deg")),
                    _fmt(row.get("anti_parallel_rate")),
                    _fmt(row.get("rot_mean_deg")),
                    _fmt(row.get("tmag_median_ratio")),
                ]
                for row in final_all["rotation_named_buckets"]
            ],
        ),
        "",
        f"Conclusion: the worst gt_rot_angle regime by signed_tdir_mean is `{max(final_all['rotation_named_buckets'], key=lambda x: float(x.get('signed_tdir_mean_deg') or -1.0))['bucket']}`.",
        "",
        "## 6. tmag × rotation regime analysis",
        _markdown_table(
            ["tmag_bucket", "rotation_bucket", "count", "signed_tdir_mean", "anti_parallel", "tmag_median_ratio"],
            [
                [
                    row["tmag_bucket"],
                    row["rotation_bucket"],
                    row["count"],
                    _fmt(row.get("signed_tdir_mean_deg")),
                    _fmt(row.get("anti_parallel_rate")),
                    _fmt(row.get("tmag_median_ratio")),
                ]
                for row in final_all["tmag_rotation_2d"]
            ],
        ),
        "",
        "## 7. scene / sequence analysis",
        "- these rankings are also pooled over `val + test` to expose stable bad regimes rather than a single-split artifact.",
        "- worst scenes by signed_tdir_mean:",
        *[
            f"  - `{row['scene']}`: signed_tdir_mean=`{_fmt(row.get('signed_tdir_mean_deg'))}`, anti_parallel=`{_fmt(row.get('anti_parallel_rate'))}`, count=`{row.get('count')}`"
            for row in worst_payload["scene"]["worst_by_signed_tdir_mean"][:10]
        ],
        "- worst sequences by anti_parallel_rate:",
        *[
            f"  - `{row['sequence']}`: anti_parallel=`{_fmt(row.get('anti_parallel_rate'))}`, signed_tdir_mean=`{_fmt(row.get('signed_tdir_mean_deg'))}`, count=`{row.get('count')}`"
            for row in worst_payload["sequence"]["worst_by_anti_parallel_rate"][:10]
        ],
        "",
        "## 8. anti-parallel analysis",
        f"- anti_parallel count: `{anti_payload['anti_parallel_count']}` / `{final_all['overall']['count']}`",
        f"- strongest tmag concentration: `{max(anti_payload['tmag_bucket_counts'], key=lambda x: int(x['count']))['bucket'] if anti_payload['tmag_bucket_counts'] else 'N/A'}`",
        f"- strongest rotation concentration: `{max(anti_payload['rotation_bucket_counts'], key=lambda x: int(x['count']))['bucket'] if anti_payload['rotation_bucket_counts'] else 'N/A'}`",
        f"- collapse share within anti_parallel: `{_fmt(anti_payload['pred_tmag_anomaly_share'].get('collapse'))}`",
        f"- explosion share within anti_parallel: `{_fmt(anti_payload['pred_tmag_anomaly_share'].get('explosion'))}`",
        "",
        "## 9. FINAL360I vs SEQ360B comparison",
        f"- FINAL360I adjacent-only: signed_tdir_mean=`{_fmt(final_adj['overall'].get('signed_tdir_mean_deg'))}`, anti_parallel=`{_fmt(final_adj['overall'].get('anti_parallel_rate'))}`, tmag_median_ratio=`{_fmt(final_adj['overall'].get('tmag_median_ratio'))}`, path_ratio=`{_fmt(final_adj['overall'].get('path_ratio'))}`",
        (
            f"- SEQ360B adjacent-only: signed_tdir_mean=`{_fmt(seq_compare['overall_seq360b_adjacent'].get('signed_tdir_mean_deg'))}`, anti_parallel=`{_fmt(seq_compare['overall_seq360b_adjacent'].get('anti_parallel_rate'))}`, tmag_median_ratio=`{_fmt(seq_compare['overall_seq360b_adjacent'].get('tmag_median_ratio'))}`, path_ratio=`{_fmt(seq_compare['overall_seq360b_adjacent'].get('path_ratio'))}`"
            if seq_compare
            else "- SEQ360B comparison not available."
        ),
        (
            f"- Interpretation: SEQ360B improves scale/path only = `{str(seq_compare.get('scale_path_only_interpretation')).lower()}`"
            if seq_compare
            else ""
        ),
        "",
        "## 10. Reweighting recommendation",
        "- Conservative:",
        *[f"  - {x}" for x in recommendations["conservative_weighting"]],
        "- Aggressive:",
        *[f"  - {x}" for x in recommendations["aggressive_weighting"]],
        "- Diagnostic filtering:",
        *[f"  - {x}" for x in recommendations["diagnostic_filtering"]],
        f"- Recommended next experiment: `{recommendations['recommended_next_experiment']}`",
        "",
        "## 11. Final recommendation",
        f"- `{recommendations['recommended_next_experiment']}`",
        "",
        "## 12. Compliance checklist",
        "- `training_executed = false`",
        "- `fine_tune_executed = false`",
        "- `model_structure_changed = false`",
        "- `metrics_modified = false`",
        "- `checkpoints_modified = false`",
        "- `raw_data_modified = false`",
        "- `explicit_matching_used = false`",
        "- `ransac_used = false`",
        "- `pnp_used = false`",
        "- `ba_used = false`",
        "- `large_prediction_dump_committed = false`",
    ]
    OUTPUTS["report"].write_text("\n".join([line for line in lines if line != ""]) + "\n", encoding="utf-8")


def _report_metrics_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, Mapping) else payload


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch = _run(["git", "branch", "--show-current"])
    if branch != "research/data360a-tdir-regime-diagnostic":
        raise RuntimeError(f"Expected branch research/data360a-tdir-regime-diagnostic, got {branch}")
    status_tracked = _run(["git", "status", "--short", "--untracked-files=no"])
    if status_tracked.strip():
        raise RuntimeError(f"Tracked/staged worktree is not clean:\n{status_tracked}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    pair_model, pair_cfg, pair_load = _load_pair_model(FINAL360I_CKPT, device)
    image_hw = (int(pair_cfg.H), int(pair_cfg.W))
    final_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("val", "test"):
        rows = _final360i_rows_for_split(pair_model, split=split, image_hw=image_hw, device=device)
        final_rows_by_split[split] = rows
        _write_jsonl(DATA_ROOT / f"final360i_{split}_pair_rows.jsonl", rows)

    final_all_rows = final_rows_by_split["val"] + final_rows_by_split["test"]
    final_adj_rows = [r for r in final_all_rows if str(r.get("pair_type")) == "adjacent" and int(r.get("k", 1)) == 1]

    seq_compare = None
    seq_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    seq_head_load: Optional[Dict[str, Any]] = None
    if SEQ360B_CKPT.is_file():
        seq_cfg = yaml.safe_load(SEQ360B_CFG.read_text(encoding="utf-8"))
        context_dim = 3 * int(getattr(pair_cfg, "D", 256))
        seq_head, seq_head_load, _ = _load_seq360b_head_with_context(SEQ360B_CKPT, context_dim, device)
        for split in ("val", "test"):
            rows = _seq360b_rows_for_split(pair_model, seq_head, split=split, seq_cfg=seq_cfg, device=device)
            seq_rows_by_split[split] = rows
            _write_jsonl(DATA_ROOT / f"seq360b_{split}_adjacent_rows.jsonl", rows)
        seq_compare = _compare_models_by_bucket(final_all_rows, seq_rows_by_split["val"] + seq_rows_by_split["test"], _tmag_named_specs([r["gt_tmag"] for r in final_all_rows]), _rot_named_specs())
        _write_json(OUTPUTS["comparison"], seq_compare)

    all_tmag_values = [float(r["gt_tmag"]) for r in final_all_rows if _safe_float(r.get("gt_tmag")) is not None]
    all_rot_values = [float(r["gt_rot_angle_deg"]) for r in final_all_rows if _safe_float(r.get("gt_rot_angle_deg")) is not None]
    tmag_quant_specs = _quantile_bucket_specs(all_tmag_values, [0, 10, 25, 50, 75, 90, 100])
    tmag_named_specs = _tmag_named_specs(all_tmag_values)
    rot_quant_specs = _quantile_bucket_specs(all_rot_values, [0, 25, 50, 75, 100])
    rot_named_specs = _rot_named_specs()

    regime_payload: Dict[str, Any] = {}
    for name, rows in {
        "FINAL360I_all_pairs": final_all_rows,
        "FINAL360I_adjacent_only": final_adj_rows,
        "SEQ360B_adjacent_only": seq_rows_by_split.get("val", []) + seq_rows_by_split.get("test", []),
    }.items():
        if not rows:
            continue
        regime_payload[name] = {
            "overall": _summarize_rows(rows),
            "tmag_quantile_buckets": _bucketize(rows, "gt_tmag", tmag_quant_specs),
            "tmag_named_buckets": _bucketize(rows, "gt_tmag", tmag_named_specs),
            "rotation_quantile_buckets": _bucketize(rows, "gt_rot_angle_deg", rot_quant_specs),
            "rotation_named_buckets": _bucketize(rows, "gt_rot_angle_deg", rot_named_specs),
            "tmag_rotation_2d": _bucketize_2d(rows, x_field="gt_tmag", x_specs=tmag_named_specs, y_field="gt_rot_angle_deg", y_specs=rot_named_specs),
            "frame_gap_buckets": _bucketize(rows, "k", [{"label": "adjacent", "low": 1.0, "high": 1.0, "include_high": True}, {"label": "short_gap", "low": 2.0, "high": 2.0, "include_high": True}, {"label": "medium_gap", "low": 3.0, "high": 4.0, "include_high": True}, {"label": "long_gap", "low": 5.0, "high": float('inf'), "include_high": True}]),
        }
    _write_json(OUTPUTS["regime_buckets"], regime_payload)

    worst_payload = {
        "scene": _worst_rankings(final_all_rows, "scene"),
        "sequence": _worst_rankings(final_all_rows, "sequence"),
    }
    _write_json(OUTPUTS["worst_scenes"], worst_payload)

    anti_payload = _anti_parallel_analysis(final_all_rows, tmag_named_specs, rot_named_specs)
    _write_json(OUTPUTS["antiparallel"], anti_payload)

    recommendations = _recommendation_from_analysis(
        regime_payload["FINAL360I_all_pairs"]["tmag_named_buckets"],
        regime_payload["FINAL360I_all_pairs"]["rotation_named_buckets"],
        anti_payload,
        seq_compare,
    )
    _write_reweighting_report(recommendations)
    _write_bucket_tables(regime_payload)

    overall_recap_raw = {key: _read_json(path) for key, path in METRIC_SOURCES.items() if path.suffix == ".json"}
    overall_recap = {key: _report_metrics_payload(value) for key, value in overall_recap_raw.items()}
    source_manifest = {
        "task_name": "DATA360A_translation_direction_regime_diagnostic_and_reweighting",
        "git_branch": branch,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "models_analyzed": ["FINAL360I"] + (["SEQ360B"] if seq_compare is not None else []),
        "splits_analyzed": ["val", "test"],
        "metric_source_files": [str(path) for path in METRIC_SOURCES.values()],
        "checkpoint_sources": [str(FINAL360I_CKPT)] + ([str(SEQ360B_CKPT)] if seq_compare is not None else []),
        "generated_prediction_dump_dir": str(DATA_ROOT),
        "generated_prediction_dump_files": sorted(str(p.relative_to(REPO_ROOT)) for p in DATA_ROOT.glob("*.jsonl")),
        "values_copied_unchanged_from_existing_reports": True,
        "normalized_metric_payloads": {key: overall_recap.get(key, {}) for key in overall_recap},
        "pair_model_load": pair_load,
        "seq360b_head_load": seq_head_load,
        "field_mapping_note": {
            "scene": "Derived from sequence_id. For current canonical manifests, scene == sequence_id because seq_id is a flat label such as snowmobile.",
            "frame_i_frame_j": "Derived from image file stems.",
        },
    }
    _write_json(OUTPUTS["source_manifest"], source_manifest)

    _write_main_report(
        source_manifest=source_manifest,
        overall_recap=overall_recap,
        regime_payload=regime_payload,
        worst_payload=worst_payload,
        anti_payload=anti_payload,
        seq_compare=seq_compare,
        recommendations=recommendations,
        prediction_dump_generated=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
