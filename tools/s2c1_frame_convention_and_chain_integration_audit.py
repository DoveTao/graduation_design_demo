#!/usr/bin/env python3
"""Audit official odometry aggregation and frame/integration conventions for S2b."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.eval_clean_policy import (  # type: ignore
    DtBucketScaledMagnitudeModel,
    _build_eval_dataset,
    _load_fine_model,
    _load_policy,
)
from train_mvp import (  # type: ignore
    _build_odometry_chains,
    _camera_center_from_T_c0_np,
    _compose_rel_pose_np,
    _rot_geodesic_deg_np,
    _trajectory_shape_summary,
    _vec_angle_deg_np,
)


POLICY_PATH = REPO_ROOT / "checkpoints/S2b_clean_fine_rot_policy.json"
REPRO_DIR = REPO_ROOT / "checkpoints/S2b_final_repro"
REPORT_PATH = REPO_ROOT / "checkpoints/S2c1_frame_convention_and_chain_integration_audit_report.md"


@dataclass
class PairRecord:
    chain_id: int
    scene_seq: str
    step_idx: int
    ds_idx: int
    i: int
    j: int
    k: int
    dt_world: float
    R_gt: np.ndarray
    t_gt_vec: np.ndarray
    t_gt_dir: np.ndarray
    t_gt_mag: float
    R_pred: np.ndarray
    t_dir_out: np.ndarray
    t_dir_local: np.ndarray
    t_vec_out: np.ndarray
    t_vec_local: np.ndarray
    pred_tmag: float


@dataclass
class ChainResult:
    chain_id: int
    scene_seq: str
    num_steps: int
    gt_path_length: float
    pred_path_length: float
    path_ratio: float
    ATE: float
    drift: float
    mean_rot_error: float
    mean_tdir_error: float
    mean_tmag_ratio: float
    mean_turn_error: float
    max_step_error: float
    worst_step_index: int


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return f"{x:.{digits}f}" if math.isfinite(x) else "nan"


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    ax = np.asarray(xs, dtype=np.float64)
    ay = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(ax) & np.isfinite(ay)
    ax = ax[mask]
    ay = ay[mask]
    if ax.size < 2:
        return float("nan")
    sx = float(ax.std())
    sy = float(ay.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / n


def _compose_pose_variant(
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    R_cur0: np.ndarray,
    t_cur0: np.ndarray,
    *,
    order: str = "official",
) -> Tuple[np.ndarray, np.ndarray]:
    if order == "official":
        return _compose_rel_pose_np(R_rel, t_rel, R_cur0, t_cur0)
    if order == "inverse_order":
        R_next0 = R_cur0.astype(np.float64) @ R_rel.astype(np.float64)
        t_next0 = R_cur0.astype(np.float64) @ t_rel.astype(np.float64) + t_cur0.astype(np.float64)
        return R_next0, t_next0
    raise ValueError(f"Unknown composition order: {order}")


def _load_records() -> Tuple[List[List[PairRecord]], Dict[str, Any]]:
    policy = _load_policy(POLICY_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    ds = _build_eval_dataset(cfg, split="test")
    chains = _build_odometry_chains(ds.manifest(), 1, 0)
    bucket_factors = {str(k): float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped = DtBucketScaledMagnitudeModel(model, bucket_factors).to(device)
    wrapped.eval()

    out: List[List[PairRecord]] = []
    with torch.no_grad():
        for chain_id, chain in enumerate(chains):
            chain_rows: List[PairRecord] = []
            for step_idx, meta in enumerate(chain["pairs"]):
                ds_idx = int(meta["_ds_idx"])
                sample = ds[ds_idx]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                t_gt_vec = sample["t_gt_vec"].float().cpu().numpy()
                t_gt_mag = float(np.linalg.norm(t_gt_vec))
                dt_world = float(meta.get("dt_world", t_gt_mag))
                dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
                R_pred_t, _t_pred_t, aux = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                R_pred = R_pred_t.detach().float().cpu().numpy()[0]
                t_dir_out = _unit(aux["t_dir_out"].detach().float().cpu().numpy()[0])
                t_dir_local = _unit(aux["t_dir_local"].detach().float().cpu().numpy()[0])
                t_vec_out = aux["t_vec_out"].detach().float().cpu().numpy()[0]
                t_vec_local = aux["t_vec_local"].detach().float().cpu().numpy()[0]
                pred_tmag = float(np.linalg.norm(t_vec_out))
                chain_rows.append(
                    PairRecord(
                        chain_id=chain_id,
                        scene_seq=f"{chain['scene_seq'][0]}/{chain['scene_seq'][1]}",
                        step_idx=step_idx,
                        ds_idx=ds_idx,
                        i=int(meta.get("i", -1)),
                        j=int(meta.get("j", -1)),
                        k=int(meta.get("k", 1)),
                        dt_world=dt_world,
                        R_gt=sample["R_gt"].float().cpu().numpy(),
                        t_gt_vec=t_gt_vec,
                        t_gt_dir=_unit(sample["t_gt_dir"].float().cpu().numpy()),
                        t_gt_mag=t_gt_mag,
                        R_pred=R_pred,
                        t_dir_out=t_dir_out,
                        t_dir_local=t_dir_local,
                        t_vec_out=t_vec_out,
                        t_vec_local=t_vec_local,
                        pred_tmag=pred_tmag,
                    )
                )
            out.append(chain_rows)
    meta = {
        "load_missing": list(load_summary["missing"]),
        "load_unexpected": list(load_summary["unexpected"]),
        "selected_k": 1,
        "num_pairs": int(sum(len(c) for c in out)),
        "num_chains": int(len(out)),
        "policy": policy,
    }
    return out, meta


def _variant_pose(rec: PairRecord, name: str) -> Dict[str, Any]:
    R_pred = rec.R_pred
    R_gt = rec.R_gt
    tmag_pred = rec.pred_tmag
    tmag_gt = rec.t_gt_mag
    tdir_out = rec.t_dir_out
    tdir_local = rec.t_dir_local
    tdir_gt = rec.t_gt_dir

    if name == "official_current":
        return {"R": R_pred, "t": rec.t_vec_out, "tdir": _unit(rec.t_vec_out), "tmag": tmag_pred, "order": "official"}
    if name == "pred_R_pred_tdir":
        return {"R": R_pred, "t": tdir_out * tmag_pred, "tdir": tdir_out, "tmag": tmag_pred, "order": "official"}
    if name == "pred_RT_pred_tdir":
        return {"R": R_pred.T, "t": tdir_out * tmag_pred, "tdir": tdir_out, "tmag": tmag_pred, "order": "official"}
    if name == "pred_R_R_tdir":
        tdir = _unit(R_pred @ tdir_out)
        return {"R": R_pred, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "pred_R_RT_tdir":
        tdir = _unit(R_pred.T @ tdir_out)
        return {"R": R_pred, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "pred_RT_R_tdir":
        tdir = _unit(R_pred @ tdir_out)
        return {"R": R_pred.T, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "pred_RT_RT_tdir":
        tdir = _unit(R_pred.T @ tdir_out)
        return {"R": R_pred.T, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "invert_comp_order":
        return {"R": R_pred, "t": rec.t_vec_out, "tdir": _unit(rec.t_vec_out), "tmag": tmag_pred, "order": "inverse_order"}
    if name == "swap_AB_direction":
        R_use = R_pred.T
        t_use = -(R_pred.T @ rec.t_vec_out)
        return {"R": R_use, "t": t_use, "tdir": _unit(t_use), "tmag": float(np.linalg.norm(t_use)), "order": "official"}
    if name == "use_local_A_tdir":
        return {"R": R_pred, "t": tdir_local * tmag_pred, "tdir": tdir_local, "tmag": tmag_pred, "order": "official"}
    if name == "use_output_frame_tdir":
        return {"R": R_pred, "t": tdir_out * tmag_pred, "tdir": tdir_out, "tmag": tmag_pred, "order": "official"}
    if name == "oracle_R":
        return {"R": R_gt, "t": tdir_out * tmag_pred, "tdir": tdir_out, "tmag": tmag_pred, "order": "official"}
    if name == "oracle_tdir":
        return {"R": R_pred, "t": tdir_gt * tmag_pred, "tdir": tdir_gt, "tmag": tmag_pred, "order": "official"}
    if name == "oracle_R_tdir":
        return {"R": R_gt, "t": tdir_gt * tmag_pred, "tdir": tdir_gt, "tmag": tmag_pred, "order": "official"}
    if name == "oracle_tmag":
        return {"R": R_pred, "t": tdir_out * tmag_gt, "tdir": tdir_out, "tmag": tmag_gt, "order": "official"}
    if name == "oracle_all":
        return {"R": R_gt, "t": rec.t_gt_vec, "tdir": tdir_gt, "tmag": tmag_gt, "order": "official"}
    if name == "gtR_predtdir_gtRframe":
        tdir = _unit(R_gt @ tdir_local)
        return {"R": R_gt, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "predR_gttdir_predRframe":
        tdir = _unit(R_pred @ tdir_gt)
        return {"R": R_pred, "t": tdir * tmag_pred, "tdir": tdir, "tmag": tmag_pred, "order": "official"}
    if name == "gtR_gttdir_pred_order":
        return {"R": R_gt, "t": tdir_gt * tmag_gt, "tdir": tdir_gt, "tmag": tmag_gt, "order": "inverse_order"}
    if name == "predR_predtdir_gt_order":
        return {"R": R_pred, "t": tdir_out * tmag_pred, "tdir": tdir_out, "tmag": tmag_pred, "order": "inverse_order"}
    if name == "gtR_predtdir_A_local":
        return {"R": R_gt, "t": tdir_local * tmag_pred, "tdir": tdir_local, "tmag": tmag_pred, "order": "official"}
    raise KeyError(name)


def _evaluate_variant(
    chains: Sequence[Sequence[PairRecord]],
    variant: str,
    *,
    chain_subset: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    selected = set(chain_subset) if chain_subset is not None else None
    chain_results: List[ChainResult] = []
    step_rows: List[Dict[str, Any]] = []
    pos_err_sq: List[float] = []
    rpe_rot: List[float] = []
    rpe_tdir: List[float] = []
    rpe_tmag: List[float] = []
    endpoint_errs: List[float] = []

    for chain_id, chain in enumerate(chains):
        if selected is not None and chain_id not in selected:
            continue
        if not chain:
            continue
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pr_c0 = np.eye(3, dtype=np.float64)
        t_pr_c0 = np.zeros(3, dtype=np.float64)
        rows: List[Dict[str, Any]] = []
        chain_pos_errs: List[float] = []
        chain_rot_errs: List[float] = []
        chain_tdir_errs: List[float] = []
        chain_tmag_ratios: List[float] = []
        for rec in chain:
            use = _variant_pose(rec, variant)
            R_gt_c0, t_gt_c0 = _compose_pose_variant(rec.R_gt, rec.t_gt_vec, R_gt_c0, t_gt_c0, order="official")
            R_pr_c0, t_pr_c0 = _compose_pose_variant(use["R"], use["t"], R_pr_c0, t_pr_c0, order=use["order"])
            p_gt = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            p_pr = _camera_center_from_T_c0_np(R_pr_c0, t_pr_c0)
            pos_err = float(np.linalg.norm(p_pr - p_gt))
            rot_err = float(_rot_geodesic_deg_np(use["R"], rec.R_gt))
            tdir_err = float(_vec_angle_deg_np(use["tdir"], rec.t_gt_dir))
            tmag_ratio = float(use["tmag"] / max(rec.t_gt_mag, 1e-12))
            pos_err_sq.append(pos_err ** 2)
            rpe_rot.append(rot_err)
            rpe_tdir.append(tdir_err)
            rpe_tmag.append(abs(tmag_ratio - 1.0))
            chain_pos_errs.append(pos_err)
            chain_rot_errs.append(rot_err)
            chain_tdir_errs.append(tdir_err)
            chain_tmag_ratios.append(tmag_ratio)
            row = {
                "step_idx": int(rec.step_idx),
                "i": int(rec.i),
                "j": int(rec.j),
                "dt_gt": float(rec.t_gt_mag),
                "gt_x": float(p_gt[0]),
                "gt_y": float(p_gt[1]),
                "gt_z": float(p_gt[2]),
                "metric_x": float(p_pr[0]),
                "metric_y": float(p_pr[1]),
                "metric_z": float(p_pr[2]),
                "metric_pos_err": pos_err,
                "rot_err_deg": rot_err,
                "tdir_err_deg": tdir_err,
                "tmag_ratio": tmag_ratio,
                "tmag_rel_err": abs(tmag_ratio - 1.0),
            }
            rows.append(row)
            step_rows.append({"chain_id": chain_id, "scene_seq": rec.scene_seq, **row})
        shape = _trajectory_shape_summary(rows, "metric")
        endpoint_err = float(chain_pos_errs[-1]) if chain_pos_errs else float("nan")
        endpoint_errs.append(endpoint_err)
        chain_results.append(
            ChainResult(
                chain_id=chain_id,
                scene_seq=chain[0].scene_seq,
                num_steps=len(chain),
                gt_path_length=float(shape.get("gt_path_length", float("nan"))),
                pred_path_length=float(shape.get("pred_path_length", float("nan"))),
                path_ratio=float(shape.get("path_length_ratio", float("nan"))),
                ATE=float(math.sqrt(np.mean(np.square(chain_pos_errs)))) if chain_pos_errs else float("nan"),
                drift=endpoint_err,
                mean_rot_error=float(np.mean(chain_rot_errs)) if chain_rot_errs else float("nan"),
                mean_tdir_error=float(np.mean(chain_tdir_errs)) if chain_tdir_errs else float("nan"),
                mean_tmag_ratio=float(np.mean(chain_tmag_ratios)) if chain_tmag_ratios else float("nan"),
                mean_turn_error=float(shape.get("mean_turn_abs_err_deg", float("nan"))),
                max_step_error=float(max(chain_pos_errs)) if chain_pos_errs else float("nan"),
                worst_step_index=int(int(np.argmax(chain_pos_errs))) if chain_pos_errs else -1,
            )
        )
    gt_lens = [c.gt_path_length for c in chain_results if math.isfinite(c.gt_path_length)]
    weighted_ratio = float(
        sum(c.path_ratio * c.gt_path_length for c in chain_results if math.isfinite(c.path_ratio) and math.isfinite(c.gt_path_length))
        / max(sum(gt_lens), 1e-12)
    ) if gt_lens else float("nan")
    mean_ratio = float(np.mean([c.path_ratio for c in chain_results if math.isfinite(c.path_ratio)])) if chain_results else float("nan")
    return {
        "variant": variant,
        "ATE": float(math.sqrt(np.mean(pos_err_sq))) if pos_err_sq else float("nan"),
        "drift": float(np.mean(endpoint_errs)) if endpoint_errs else float("nan"),
        "path_ratio": weighted_ratio,
        "path_weighted_ratio": weighted_ratio,
        "mean_path_ratio": mean_ratio,
        "RPE_rot": float(np.mean(rpe_rot)) if rpe_rot else float("nan"),
        "RPE_trans_dir": float(np.mean(rpe_tdir)) if rpe_tdir else float("nan"),
        "RPE_trans_mag": float(np.mean(rpe_tmag)) if rpe_tmag else float("nan"),
        "selected_k": 1,
        "num_chains": len(chain_results),
        "num_pairs": len(step_rows),
        "chain_results": chain_results,
        "step_rows": step_rows,
        "mean_step_position_error": float(np.mean([math.sqrt(x) for x in pos_err_sq])) if pos_err_sq else float("nan"),
        "chain_ATE_mean": float(np.mean([c.ATE for c in chain_results])) if chain_results else float("nan"),
        "chain_drift_mean": float(np.mean([c.drift for c in chain_results])) if chain_results else float("nan"),
        "chain_path_ratio_mean": mean_ratio,
    }


def _classify(
    official_reproduced: bool,
    mismatch_reason: str,
    frame_results: Dict[str, Dict[str, Any]],
    oracle_results: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    if mismatch_reason:
        return "OFFICIAL/S2C-AGGREGATION-MISMATCH", "修正 S2c 诊断工具，不改模型，然后重跑 S2c。"
    best_non_oracle = min(
        (v for k, v in frame_results.items() if k != "official_current"),
        key=lambda x: _safe_float(x["ATE"], float("inf")),
        default=None,
    )
    base_ate = _safe_float(frame_results["official_current"]["ATE"], float("inf"))
    if best_non_oracle and _safe_float(best_non_oracle["ATE"], float("inf")) + 0.5 < base_ate:
        name = str(best_non_oracle["variant"])
        if "swap" in name:
            return "RELATIVE-POSE-DIRECTION-MISMATCH", "修正 A/B 相对位姿方向后，重新评估 S2b。"
        if "invert_comp_order" in name:
            return "COMPOSITION-ORDER-ISSUE", "修正 trajectory integration 的 pose composition 顺序后，重新评估 S2b。"
        if "local_A" in name or "output_frame" in name or "R_tdir" in name:
            return "TDIR-FRAME-MISMATCH", "修正 tdir frame conversion 或增加 frame-aware eval。"
    if _safe_float(oracle_results["oracle_R_tdir"]["ATE"], float("inf")) + 1.0 < min(
        _safe_float(oracle_results["oracle_R"]["ATE"], float("inf")),
        _safe_float(oracle_results["oracle_tdir"]["ATE"], float("inf")),
    ):
        return "TRUE-ROT-TDIR-COUPLED-ERROR", "进入 S2d joint fine_rot + fine_tdir eval-only sweep。"
    return "NO-CONVENTION-BUG-FOUND", "没有发现 convention bug；若仍要继续，进入 S2d joint fine_rot + fine_tdir eval-only sweep。"


def _render_chain_table(rows: Sequence[ChainResult], limit: int = 10, key: str = "ATE") -> str:
    sorted_rows = sorted(rows, key=lambda r: _safe_float(getattr(r, key), float("-inf")), reverse=True)
    lines = [
        "| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted_rows[:limit]:
        lines.append(
            f"| {r.chain_id} | {r.scene_seq} | {r.num_steps} | {_fmt(r.gt_path_length)} | {_fmt(r.pred_path_length)} | "
            f"{_fmt(r.path_ratio)} | {_fmt(r.ATE)} | {_fmt(r.drift)} | {_fmt(r.mean_rot_error)} | {_fmt(r.mean_tdir_error)} | "
            f"{_fmt(r.mean_tmag_ratio)} | {_fmt(r.mean_turn_error)} | {_fmt(r.max_step_error)} | {r.worst_step_index} |"
        )
    return "\n".join(lines)


def main() -> None:
    chains, meta = _load_records()
    official_summary = json.loads((REPRO_DIR / "s1d5_policy_eval_summary.json").read_text(encoding="utf-8"))
    debug_json = json.loads((REPRO_DIR / "odom_trajectory_debug_latest.json").read_text(encoding="utf-8"))

    official_field = "odom_shape_metric_mean_path_length_ratio"
    official_chain_count = int(debug_json.get("num_debug_chains", 0))
    official_chain_scene = (
        debug_json.get("chains", [{}])[0].get("scene_seq", "unknown") if debug_json.get("chains") else "unknown"
    )
    official_chain_ids = [0] if official_chain_count > 0 else []
    official_expected = _safe_float(official_summary.get("metric_path_ratio"))

    official_repro = _evaluate_variant(chains, "official_current", chain_subset=official_chain_ids)
    all_chain_current = _evaluate_variant(chains, "official_current")

    mismatch_reason = ""
    if official_chain_count != len(debug_json.get("chains", [])):
        mismatch_reason = "debug_json chain count mismatch"
    elif abs(_safe_float(official_repro["path_ratio"]) - official_expected) > 1e-6:
        mismatch_reason = "official debug-chain path ratio did not reproduce"
    elif abs(_safe_float(all_chain_current["path_ratio"]) - official_expected) > 1e-3:
        mismatch_reason = (
            "official path_ratio is computed from debug_chain_summaries only; "
            f"debug_max_chains={official_chain_count}, so only chain `{official_chain_scene}` contributes, "
            "while S2c all-chain aggregation used all 19 chains including many tiny 1-2 step residual chains."
        )

    frame_variant_names = [
        "official_current",
        "pred_R_pred_tdir",
        "pred_RT_pred_tdir",
        "pred_R_R_tdir",
        "pred_R_RT_tdir",
        "pred_RT_R_tdir",
        "pred_RT_RT_tdir",
        "invert_comp_order",
        "swap_AB_direction",
        "use_local_A_tdir",
        "use_output_frame_tdir",
    ]
    frame_results = {name: _evaluate_variant(chains, name) for name in frame_variant_names}

    oracle_names = [
        "oracle_R",
        "oracle_tdir",
        "oracle_R_tdir",
        "oracle_tmag",
        "oracle_all",
        "gtR_predtdir_gtRframe",
        "predR_gttdir_predRframe",
        "gtR_gttdir_pred_order",
        "predR_predtdir_gt_order",
        "gtR_predtdir_A_local",
    ]
    oracle_results = {name: _evaluate_variant(chains, name) for name in oracle_names}

    base_chain_rows = frame_results["official_current"]["chain_results"]
    base_step_rows = frame_results["official_current"]["step_rows"]
    corrs = {
        "corr(ATE, tdir error)": _corr([r.ATE for r in base_chain_rows], [r.mean_tdir_error for r in base_chain_rows]),
        "corr(ATE, rot error)": _corr([r.ATE for r in base_chain_rows], [r.mean_rot_error for r in base_chain_rows]),
        "corr(ATE, path_ratio_error)": _corr([r.ATE for r in base_chain_rows], [abs(r.path_ratio - 1.0) for r in base_chain_rows]),
        "corr(ATE, turn error)": _corr([r.ATE for r in base_chain_rows], [r.mean_turn_error for r in base_chain_rows]),
    }

    classification, next_step = _classify(not mismatch_reason or official_repro["path_ratio"] == official_expected, mismatch_reason, frame_results, oracle_results)

    worst_tdir = sorted(
        base_step_rows,
        key=lambda r: _safe_float(r.get("tdir_err_deg"), float("-inf")),
        reverse=True,
    )[:10]
    worst_rot = sorted(
        base_step_rows,
        key=lambda r: _safe_float(r.get("rot_err_deg"), float("-inf")),
        reverse=True,
    )[:10]

    lines: List[str] = []
    lines.append("# S2c1 Frame Convention And Chain Integration Audit")
    lines.append("")
    lines.append("## Official odom metric source")
    lines.append(f"- official reported path_ratio field: `{official_field}`")
    lines.append("- implementation source: `train_mvp.py::eval_odometry_sequence`")
    lines.append("- field value is assigned from `debug_metric_path_ratio`")
    lines.append("- `debug_metric_path_ratio = _finite_mean_nested(debug_chain_summaries, 'shape_metric', 'path_length_ratio')`")
    lines.append(f"- `debug_max_chains = {official_chain_count}` in this S2b repro output, so official path_ratio uses only the first debug chain")
    lines.append(f"- official debug chain scene_seq: `{official_chain_scene}`")
    lines.append("- selected_k filtering: `selected_k = 1`")
    lines.append("- aggregation mode: mean over debug chains, not all manifest chains; weighted alternative exists as `odom_shape_metric_path_weighted_path_length_ratio`")
    lines.append("")
    lines.append("## Official chain selection / aggregation")
    lines.append(f"- manifest chains at selected_k=1: `{meta['num_chains']}`")
    lines.append(f"- manifest pairs at selected_k=1: `{meta['num_pairs']}`")
    lines.append(f"- official debug chains serialized: `{official_chain_count}`")
    lines.append("- official payload drift/ATE come from all selected chains")
    lines.append("- official payload path_ratio comes only from debug_chain_summaries")
    lines.append("")
    lines.append("## Official vs S2c mismatch")
    lines.append(f"- official reported path_ratio = `{_fmt(official_expected)}`")
    lines.append(f"- reproduced official debug-chain path_ratio = `{_fmt(official_repro['path_ratio'])}`")
    lines.append(f"- all-chain weighted path_ratio = `{_fmt(all_chain_current['path_ratio'])}`")
    lines.append(f"- all-chain mean path_ratio = `{_fmt(all_chain_current['mean_path_ratio'])}`")
    lines.append(f"- mismatch reason: {mismatch_reason or 'none'}")
    lines.append("")
    lines.append("## Official integration reproduction")
    lines.append(f"- reproduced official drift = `{_fmt(all_chain_current['drift'])}` vs report `{_fmt(official_summary.get('drift'))}`")
    lines.append(f"- reproduced official ATE = `{_fmt(all_chain_current['ATE'])}` vs report `{_fmt(official_summary.get('ATE'))}`")
    lines.append(f"- reproduced official path_ratio (debug-chain) = `{_fmt(official_repro['path_ratio'])}` vs report `{_fmt(official_summary.get('metric_path_ratio'))}`")
    lines.append("- first divergence point between official and original S2c is the path_ratio aggregation scope, not pose composition math.")
    lines.append("")
    lines.append("## Frame convention variants")
    lines.append("| variant | drift | ATE | path_ratio | path_weighted_ratio | mean_path_ratio | RPE_rot | RPE_trans_dir | selected_k | num_chains | num_pairs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in frame_variant_names:
        res = frame_results[name]
        lines.append(
            f"| {name} | {_fmt(res['drift'])} | {_fmt(res['ATE'])} | {_fmt(res['path_ratio'])} | {_fmt(res['path_weighted_ratio'])} | "
            f"{_fmt(res['mean_path_ratio'])} | {_fmt(res['RPE_rot'])} | {_fmt(res['RPE_trans_dir'])} | {res['selected_k']} | {res['num_chains']} | {res['num_pairs']} |"
        )
    lines.append("")
    lines.append("## R / tdir coupling oracle")
    lines.append("| variant | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in oracle_names:
        res = oracle_results[name]
        lines.append(
            f"| {name} | {_fmt(res['drift'])} | {_fmt(res['ATE'])} | {_fmt(res['path_ratio'])} | {_fmt(res['RPE_rot'])} | {_fmt(res['RPE_trans_dir'])} | {_fmt(res['RPE_trans_mag'])} |"
        )
    lines.append("")
    lines.append("## Chain-level breakdown")
    lines.append(_render_chain_table(base_chain_rows, limit=len(base_chain_rows), key="ATE"))
    lines.append("")
    lines.append("## Worst chains by ATE")
    lines.append(_render_chain_table(base_chain_rows, limit=10, key="ATE"))
    lines.append("")
    lines.append("## Worst chains by drift")
    lines.append(_render_chain_table(base_chain_rows, limit=10, key="drift"))
    lines.append("")
    lines.append("## Top 10 highest tdir error steps")
    for row in worst_tdir:
        lines.append(
            f"- chain {row['chain_id']} `{row['scene_seq']}` step {row['step_idx']} (i={row['i']} j={row['j']}): "
            f"tdir_err={_fmt(row['tdir_err_deg'])}, rot_err={_fmt(row['rot_err_deg'])}, pos_err={_fmt(row['metric_pos_err'])}"
        )
    lines.append("")
    lines.append("## Top 10 highest rot error steps")
    for row in worst_rot:
        lines.append(
            f"- chain {row['chain_id']} `{row['scene_seq']}` step {row['step_idx']} (i={row['i']} j={row['j']}): "
            f"rot_err={_fmt(row['rot_err_deg'])}, tdir_err={_fmt(row['tdir_err_deg'])}, pos_err={_fmt(row['metric_pos_err'])}"
        )
    lines.append("")
    lines.append("## Correlation analysis")
    for k, v in corrs.items():
        lines.append(f"- {k} = `{_fmt(v)}`")
    lines.append("")
    lines.append("## Final classification")
    lines.append(f"- classification: `{classification}`")
    lines.append("")
    lines.append("## Next step")
    lines.append(f"- {next_step}")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[S2c1] wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
