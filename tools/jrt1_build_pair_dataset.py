#!/usr/bin/env python3
"""Build the JRT1a train/val smoke cache from real S5 pair predictions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from eval_clean_policy import DtBucketScaledMagnitudeModel, _build_eval_dataset, _cfg_from_dict, _load_ckpt_cfg
from model import PanoramaRelPoseModel


FEATURE_NAMES = [
    "R_pred_00",
    "R_pred_01",
    "R_pred_02",
    "R_pred_10",
    "R_pred_11",
    "R_pred_12",
    "R_pred_20",
    "R_pred_21",
    "R_pred_22",
    "tdir_pred_x",
    "tdir_pred_y",
    "tdir_pred_z",
    "log_tmag_pred",
    "dt_world",
    "k",
]
SPLIT_TO_ID = {"train": 0, "val": 1}
LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


class PredTmagShrinkModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, q90: float, q95: float, mid_scale: float, high_scale: float) -> None:
        super().__init__()
        self.base = base
        self.q90 = float(q90)
        self.q95 = float(q95)
        self.mid_scale = float(mid_scale)
        self.high_scale = float(high_scale)
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu().item())
        scale = 1.0
        if pred_mag >= self.q95:
            scale = self.high_scale
        elif pred_mag >= self.q90:
            scale = self.mid_scale
        if abs(scale - 1.0) < 1.0e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(scale, device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["s5_tmag_scale_factor"] = fac
        return R_pred, t_pred, aux


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _validate_config(config: Dict[str, Any]) -> None:
    metrics = config.get("locked_s5_metrics", {})
    for key, expected in LOCKED.items():
        actual = float(metrics.get(key, float("nan")))
        if abs(actual - expected) > 1.0e-9:
            raise RuntimeError(f"JRT1 config changed locked S5 {key}: expected {expected}, got {actual}")
    policy = _read_json(_resolve(config["base_policy_path"]))
    manifest = _read_json(_resolve(config["manifest_path"]))
    for key, expected in LOCKED.items():
        pol = float(policy["expected_metrics"][key])
        man = float(manifest["final_metrics"][key])
        if abs(pol - expected) > 1.0e-9 or abs(man - expected) > 1.0e-9:
            raise RuntimeError(f"S5 locked metric mismatch for {key}: policy={pol} manifest={man} expected={expected}")
    if manifest.get("final_candidate_name") != "S5_clean_tmag_calibration_policy":
        raise RuntimeError("Final candidate manifest does not point to S5_clean_tmag_calibration_policy")


def _load_base_model(cfg: Config, ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_s5_model(config: Dict[str, Any], device: torch.device):
    s5_policy = _read_json(_resolve(config["base_policy_path"]))
    s2b_policy = _read_json(_resolve(s5_policy["base_policy_path"]))
    ckpt_path = _resolve(s5_policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(s5_policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(s5_policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(s5_policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(s5_policy.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    cfg.max_eval_batches = 0
    cfg.odom_eval_prefer_k = 1
    cfg.odom_eval_fallback_to_min_k = False
    cfg.eval_k_list = (1, 2, 3, 5, 10, 20)
    base_model, load_summary = _load_base_model(cfg, ckpt_path, device)
    s2b_factors = {str(k): float(v) for k, v in s2b_policy["effective_bucket_factors"].items()}
    s2b_model = DtBucketScaledMagnitudeModel(base_model, s2b_factors).to(device)
    q90 = float(s5_policy["thresholds"]["q90_value"])
    q95 = float(s5_policy["thresholds"]["q95_upper_tail_value"])
    s5_model = PredTmagShrinkModel(
        s2b_model,
        q90=q90,
        q95=q95,
        mid_scale=float(s5_policy["scales"]["mid_scale"]),
        high_scale=float(s5_policy["scales"]["high_scale"]),
    ).to(device)
    s5_model.eval()
    return s5_model, cfg, s5_policy, load_summary


def _select_indices(n: int, cap: int) -> List[int]:
    if cap <= 0 or n <= 0:
        return []
    if n <= cap:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=cap)
    out: List[int] = []
    seen = set()
    for v in raw:
        idx = int(round(float(v)))
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1.0e-12)).astype(np.float32)


def _extract_rows(model, ds, indices: Sequence[int], split_name: str, device: torch.device) -> List[Dict[str, Any]]:
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for offset, ds_idx in enumerate(indices):
            if offset == 0 or (offset + 1) % 64 == 0 or (offset + 1) == len(indices):
                print(f"[jrt1-dataset] {split_name} {offset + 1}/{len(indices)}", flush=True)
            sample = ds[ds_idx]
            meta = manifest[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_val = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
            R_pred_t, t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_world)
            R_pred = R_pred_t.detach().float().cpu().numpy()[0].astype(np.float32)
            tdir_pred_t = aux.get("t_dir_out", t_pred_t) if isinstance(aux, dict) else t_pred_t
            tdir_pred = _unit(tdir_pred_t.detach().float().cpu().numpy()[0])
            tmag_pred = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            log_tmag_pred = float(math.log(max(tmag_pred, 1.0e-12)))
            R_gt = sample["R_gt"].detach().float().cpu().numpy().astype(np.float32)
            tdir_gt = _unit(sample["t_gt_dir"].detach().float().cpu().numpy())
            tmag_gt = float(sample["t_gt_mag"])
            k = int(meta.get("k", sample.get("meta", {}).get("k", -1)))
            features = np.concatenate(
                [
                    R_pred.reshape(-1),
                    tdir_pred.reshape(-1),
                    np.asarray([log_tmag_pred, dt_val, float(k)], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
            rows.append(
                {
                    "features": features,
                    "R_pred": R_pred,
                    "tdir_pred": tdir_pred,
                    "log_tmag_pred": log_tmag_pred,
                    "tmag_pred": tmag_pred,
                    "R_gt": R_gt,
                    "tdir_gt": tdir_gt,
                    "tmag_gt": tmag_gt,
                    "split": split_name,
                    "split_id": SPLIT_TO_ID[split_name],
                    "sequence_id": f"{meta.get('scene')}::{meta.get('seq')}",
                    "pair_id": f"{meta.get('scene')}::{meta.get('seq')}::{meta.get('tsA')}->{meta.get('tsB')}::k={k}",
                    "dt": dt_val,
                    "k": k,
                }
            )
    return rows


def _stack(rows: List[Dict[str, Any]], key: str, dtype=np.float32):
    return np.asarray([r[key] for r in rows], dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build JRT1a pair-level smoke dataset from S5 predictions.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    _run_guard()
    config_path = _resolve(args.config)
    config = _read_json(config_path)
    _validate_config(config)

    out_path = _resolve(args.output)
    summary_path = _resolve(args.summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, s5_policy, load_summary = _build_s5_model(config, device)
    train_ds = _build_eval_dataset(cfg, split="train")
    test_ds = _build_eval_dataset(cfg, split="test")
    smoke = config["smoke"]
    max_train = int(smoke["max_train_pairs"])
    max_val = int(smoke["max_val_pairs"])
    selected = _select_indices(len(train_ds), max_train + max_val)
    train_indices = selected[:max_train]
    val_indices = selected[max_train : max_train + max_val]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError(f"JRT1a needs non-empty train and val smoke sets, got train={len(train_indices)} val={len(val_indices)}")

    rows = _extract_rows(model, train_ds, train_indices, "train", device)
    rows.extend(_extract_rows(model, train_ds, val_indices, "val", device))
    if not rows:
        raise RuntimeError("JRT1a dataset builder produced zero rows from real S5 predictions.")

    metadata = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "source_policy": str(config["base_policy_path"]),
        "source_policy_name": s5_policy["name"],
        "dataset_source": "RflyPanoPanoramaPairsEvalFixedKList train split with S5 policy predictions",
        "num_total_pairs": int(len(rows)),
        "num_train_pairs": int(sum(r["split"] == "train" for r in rows)),
        "num_val_pairs": int(sum(r["split"] == "val" for r in rows)),
        "num_test_pairs": 0,
        "num_available_test_pairs": int(len(test_ds)),
        "feature_dim": int(len(FEATURE_NAMES)),
        "label_dim": 13,
        "feature_fields": FEATURE_NAMES,
        "label_fields": ["R_gt_3x3", "tdir_gt_3", "tmag_gt"],
        "split_id_map": SPLIT_TO_ID,
        "sequence_id_field": "sequence_id",
        "pair_id_field": "pair_id",
        "no_test_gt_used_for_training": True,
        "test_gt_cached": False,
        "load_missing": int(len(load_summary["missing"])),
        "load_unexpected": int(len(load_summary["unexpected"])),
    }

    np.savez_compressed(
        out_path,
        features=_stack(rows, "features"),
        R_pred=_stack(rows, "R_pred"),
        tdir_pred=_stack(rows, "tdir_pred"),
        log_tmag_pred=_stack(rows, "log_tmag_pred"),
        tmag_pred=_stack(rows, "tmag_pred"),
        dt=_stack(rows, "dt"),
        k=_stack(rows, "k", dtype=np.int64),
        R_gt=_stack(rows, "R_gt"),
        tdir_gt=_stack(rows, "tdir_gt"),
        tmag_gt=_stack(rows, "tmag_gt"),
        split=_stack(rows, "split_id", dtype=np.int64),
        sequence_id=np.asarray([r["sequence_id"] for r in rows], dtype="U128"),
        pair_id=np.asarray([r["pair_id"] for r in rows], dtype="U256"),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, ensure_ascii=True)),
    )
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
