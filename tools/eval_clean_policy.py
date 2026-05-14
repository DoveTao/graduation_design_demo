#!/usr/bin/env python3
"""Evaluate a frozen clean policy artifact on top of a base checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a frozen clean dt-anchor policy.")
    p.add_argument("--policy", required=True, help="Path to policy json")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--eval-variant", default="default", choices=["default", "max_eval_batches_off", "explicit_selected_k"])
    return p.parse_args()


def _load_policy(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _q(arr: List[float]) -> Dict[str, float]:
    vals = np.asarray(list(arr), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "mean": float(vals.mean()),
    }


def _load_ckpt_cfg(ckpt_path: Path) -> Dict[str, Any]:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg = payload.get("cfg", {})
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported cfg type: {type(cfg)}")
    return cfg


def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _build_eval_dataset(cfg: Config, split: str) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=split,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
    )


class DtBucketScaledMagnitudeModel(torch.nn.Module):
    def __init__(self, base: PanoramaRelPoseModel, bucket_factors: Dict[str, float]) -> None:
        super().__init__()
        self.base = base
        self.bucket_factors = dict(bucket_factors)
        self.cfg = base.cfg

    @staticmethod
    def _bucket(dt: float) -> str | None:
        if 0.1 <= dt < 0.3:
            return "[0.1,0.3)"
        if 0.3 <= dt < 0.5:
            return "[0.3,0.5)"
        if 0.5 <= dt < 1.0:
            return "[0.5,1)"
        return None

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        if dt_world is None:
            return R_pred, t_pred, aux
        dt = float(dt_world.detach().float().view(-1)[0].cpu().item())
        bucket = self._bucket(dt)
        factor = float(self.bucket_factors.get(bucket, 1.0))
        if abs(factor - 1.0) < 1e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(factor, device=IA.device, dtype=torch.float32)
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
        aux["dt_bucket_scale_anchor_factor"] = fac
        return R_pred, t_pred, aux


def _build_loader(cfg: Config, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        drop_last=False,
    )


def _read_debug_summary(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "odom_trajectory_debug_latest.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = dict(obj.get("summary", {}))
    chains = obj.get("chains", [])
    if chains:
        vals = []
        for chain in chains:
            sd = chain.get("shape_direction_only", {})
            v = _safe_float(sd.get("path_length_ratio"))
            if math.isfinite(v):
                vals.append(v)
        if vals:
            summary["direction_only_mean_path_length_ratio"] = float(sum(vals) / max(len(vals), 1))
    return summary


def _load_fine_model(
    ckpt_path: Path,
    device: torch.device,
    *,
    fine_rot: float,
    fine_tdir: float = 0.0,
    fine_tmag: float = 0.0,
    max_eval_batches: int | None = None,
    explicit_selected_k: bool = False,
) -> Tuple[PanoramaRelPoseModel, Config, Dict[str, Any]]:
    cfg_dict = _load_ckpt_cfg(ckpt_path)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(fine_rot)
    cfg.fine_tdir_fuse_strength = float(fine_tdir)
    cfg.fine_tmag_fuse_strength = float(fine_tmag)
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    if max_eval_batches is not None:
        cfg.max_eval_batches = int(max_eval_batches)
    if explicit_selected_k:
        cfg.odom_eval_prefer_k = 1
        cfg.odom_eval_fallback_to_min_k = False
        cfg.eval_k_list = (1, 2, 3, 5, 10, 20)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _extract_pairs(
    model: PanoramaRelPoseModel,
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows = []
    manifest = ds.manifest()
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            try:
                if int(meta.get("k", -1)) != 1:
                    continue
            except Exception:
                continue
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_mag = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            gt_mag = float(sample["t_gt_mag"])
            rows.append(
                {
                    "dataset_index": int(ds_idx),
                    "dt_world": float(meta.get("dt_world", gt_mag)),
                    "tmag_gt": gt_mag,
                    "tmag_pred": pred_mag,
                }
            )
    return rows


def _run(policy: Dict[str, Any], policy_path: Path, out_dir: Path, eval_variant: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    max_eval_batches = None
    explicit_selected_k = False
    if eval_variant == "max_eval_batches_off":
        max_eval_batches = 0
    elif eval_variant == "explicit_selected_k":
        explicit_selected_k = True
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        max_eval_batches=max_eval_batches,
        explicit_selected_k=explicit_selected_k,
    )
    if load_summary["unexpected"]:
        raise RuntimeError(
            "Tmag head parameters were not loaded; eval result may be a wrapper-default cfg mismatch. "
            f"unexpected={load_summary['unexpected'][:8]}"
        )
    print(f"[S1d5-eval] checkpoint={ckpt_path}")
    print(f"[S1d5-eval] policy={out_dir}")
    print(f"[S1d5-eval] restored cfg tmag_head_mode={getattr(cfg, 'tmag_head_mode', 'unknown')}")
    print(f"[S1d5-eval] missing={len(load_summary['missing'])} unexpected={len(load_summary['unexpected'])}")
    print(f"[S1d5-eval] eval protocol={eval_variant} | fine_rot={cfg.fine_rot_fuse_strength} fine_tdir={cfg.fine_tdir_fuse_strength} fine_tmag={cfg.fine_tmag_fuse_strength}")

    ds = _build_eval_dataset(cfg, split="test")
    loader = _build_loader(cfg, ds)
    factors = {k: float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped = DtBucketScaledMagnitudeModel(model, factors)
    out_dir.mkdir(parents=True, exist_ok=True)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        wrapped, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(wrapped, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    debug_summary = _read_debug_summary(out_dir)
    rows = _extract_pairs(model, ds, device)
    scaled_rows = []
    for r in rows:
        rec = dict(r)
        dt = float(rec["dt_world"])
        if 0.1 <= dt < 0.3:
            bucket = "[0.1,0.3)"
        elif 0.3 <= dt < 0.5:
            bucket = "[0.3,0.5)"
        elif 0.5 <= dt < 1.0:
            bucket = "[0.5,1)"
        else:
            bucket = None
        factor = float(factors.get(bucket, 1.0))
        rec["tmag_pred"] = float(rec["tmag_pred"]) * factor
        scaled_rows.append(rec)
    q = _q([float(r["tmag_pred"]) for r in scaled_rows])
    payload = {
        "policy_path": str(policy_path),
        "base_checkpoint_path": str(ckpt_path),
        "eval_variant": eval_variant,
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "fine_rot_fuse_strength": float(cfg.fine_rot_fuse_strength),
        "fine_tdir_fuse_strength": float(cfg.fine_tdir_fuse_strength),
        "fine_tmag_fuse_strength": float(cfg.fine_tmag_fuse_strength),
        "rot": float(rot),
        "tdir_abs": float(tdir_abs),
        "tdir_local_A_abs": float(tdir_local_A_abs),
        "drift": _safe_float(odom.get("odom_metric_drift")),
        "ATE": _safe_float(odom.get("odom_metric_ATE")),
        "RPE_rot": _safe_float(odom.get("odom_metric_RPE_rot")),
        "RPE_trans_dir": _safe_float(odom.get("odom_metric_RPE_trans_dir")),
        "RPE_trans_mag": _safe_float(odom.get("odom_metric_RPE_trans_mag")),
        "metric_path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        "direction_only_path_ratio": _safe_float(debug_summary.get("direction_only_mean_path_length_ratio"), 1.0),
        "odom_selected_k": int(odom.get("odom_selected_k", -1)),
        "odom_available_k": list(odom.get("odom_available_k", [])),
        "num_pairs": int(odom.get("odom_num_pairs", 0)),
        "num_chains": int(odom.get("odom_num_chains", 0)),
        "tmag_p10": float(q["p10"]),
        "tmag_p50": float(q["p50"]),
        "tmag_p90": float(q["p90"]),
    }
    (out_dir / "s1d5_policy_eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    args = parse_args()
    policy_path = Path(args.policy)
    policy = _load_policy(policy_path)
    _run(policy, policy_path, Path(args.output_dir), args.eval_variant)
