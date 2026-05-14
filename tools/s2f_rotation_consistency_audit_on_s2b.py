#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import (  # type: ignore
    DtBucketScaledMagnitudeModel,
    _build_eval_dataset,
    _cfg_from_dict,
    _load_ckpt_cfg,
    _q,
    _safe_float,
)
from model import PanoramaRelPoseModel
from train_mvp import (  # type: ignore
    _build_odometry_chains,
    _camera_center_from_T_c0_np,
    _compose_rel_pose_np,
    _rot_geodesic_deg_np,
    _trajectory_shape_summary,
    _vec_angle_deg_np,
    eval_model,
    eval_odometry_sequence,
)


CKPT_PATH = REPO_ROOT / "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
POLICY_PATH = REPO_ROOT / "checkpoints/S2b_clean_fine_rot_policy.json"
REPORT_PATH = REPO_ROOT / "checkpoints/S2f_rotation_consistency_audit_on_s2b_report.md"
OUT_ROOT = REPO_ROOT / "checkpoints/S2f_rotation_consistency_audit_on_s2b"

ROT_VALUES = [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55]
DT_BUCKETS = ("[0.1,0.3)", "[0.3,0.5)", "[0.5,1.0)")
DT_POLICIES: Dict[str, Dict[str, float]] = {
    "global_s2b_baseline": {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.45, "[0.5,1.0)": 0.45},
    "conservative_small_dt": {"[0.1,0.3)": 0.40, "[0.3,0.5)": 0.45, "[0.5,1.0)": 0.50},
    "aggressive_large_dt": {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.50, "[0.5,1.0)": 0.55},
    "conservative_large_dt": {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.45, "[0.5,1.0)": 0.40},
}


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
    dt_bucket: str
    R_gt: np.ndarray
    t_gt_vec: np.ndarray
    t_gt_dir: np.ndarray
    t_gt_mag: float
    R_pred: np.ndarray
    t_dir_out: np.ndarray
    t_dir_local: np.ndarray
    t_vec_out: np.ndarray
    pred_tmag: float


@dataclass
class ChainSummary:
    chain_id: int
    scene_seq: str
    num_steps: int
    ate: float
    drift: float
    path_ratio: float
    mean_rot_error: float
    mean_turn_error: float
    mean_step_position_error: float
    max_cumulative_error: float
    mean_tdir_error: float


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
    if float(ax.std()) <= 1e-12 or float(ay.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


def _dt_bucket(dt: float) -> str:
    if 0.1 <= dt < 0.3:
        return "[0.1,0.3)"
    if 0.3 <= dt < 0.5:
        return "[0.3,0.5)"
    if 0.5 <= dt < 1.0:
        return "[0.5,1.0)"
    return "other"


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / n


class RotPolicyWrapper(torch.nn.Module):
    def __init__(self, base: DtBucketScaledMagnitudeModel, rot_policy: Callable[[float], float]) -> None:
        super().__init__()
        self.base = base
        self.rot_policy = rot_policy
        self.cfg = base.cfg

    def forward(self, IA, IB, *, enable_depth_fusion=None, dt_world=None):
        old_rot = float(getattr(self.base.base.cfg, "fine_rot_fuse_strength", 0.45))
        dt_val = float(dt_world.detach().float().view(-1)[0].cpu().item()) if dt_world is not None else 0.0
        new_rot = float(self.rot_policy(dt_val))
        self.base.base.cfg.fine_rot_fuse_strength = new_rot
        try:
            return self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        finally:
            self.base.base.cfg.fine_rot_fuse_strength = old_rot


def _load_base_cfg():
    cfg_dict = _load_ckpt_cfg(CKPT_PATH)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    cfg.batch_size = 1
    return cfg


def _load_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    cfg = _load_base_cfg()
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(CKPT_PATH), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_wrapped_model(device: torch.device, rot_policy: Callable[[float], float]) -> Tuple[RotPolicyWrapper, Any, Dict[str, Any], Any]:
    base_model, load_summary = _load_model(device)
    cfg = base_model.cfg
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    factors = {str(k): float(v) for k, v in policy["effective_bucket_factors"].items()}
    scaled = DtBucketScaledMagnitudeModel(base_model, factors).to(device)
    wrapped = RotPolicyWrapper(scaled, rot_policy).to(device)
    ds = _build_eval_dataset(cfg, split="test")
    return wrapped, cfg, load_summary, ds


def _collect_pair_records(
    wrapped: RotPolicyWrapper,
    ds,
    device: torch.device,
    *,
    selected_k: int = 1,
) -> List[List[PairRecord]]:
    manifest = ds.manifest()
    chains = _build_odometry_chains(manifest, selected_k, 0)
    out: List[List[PairRecord]] = []
    with torch.no_grad():
        for chain_id, chain in enumerate(chains):
            recs: List[PairRecord] = []
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
                recs.append(
                    PairRecord(
                        chain_id=chain_id,
                        scene_seq=f"{chain['scene_seq'][0]}/{chain['scene_seq'][1]}",
                        step_idx=step_idx,
                        ds_idx=ds_idx,
                        i=int(meta.get("i", -1)),
                        j=int(meta.get("j", -1)),
                        k=int(meta.get("k", selected_k)),
                        dt_world=dt_world,
                        dt_bucket=_dt_bucket(dt_world),
                        R_gt=sample["R_gt"].float().cpu().numpy(),
                        t_gt_vec=t_gt_vec,
                        t_gt_dir=_unit(sample["t_gt_dir"].float().cpu().numpy()),
                        t_gt_mag=t_gt_mag,
                        R_pred=R_pred,
                        t_dir_out=t_dir_out,
                        t_dir_local=t_dir_local,
                        t_vec_out=t_vec_out,
                        pred_tmag=float(np.linalg.norm(t_vec_out)),
                    )
                )
            out.append(recs)
    return out


def _evaluate_chains(chains: Sequence[Sequence[PairRecord]]) -> Tuple[Dict[str, Any], List[ChainSummary], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pos_err_sq: List[float] = []
    endpoint_errs: List[float] = []
    rpe_rot: List[float] = []
    rpe_tdir: List[float] = []
    rpe_tmag: List[float] = []
    chain_summaries: List[ChainSummary] = []
    step_rows: List[Dict[str, Any]] = []
    dir_only_ratios: List[float] = []
    metric_path_ratios: List[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    tmag_preds: List[float] = []

    for chain in chains:
        if not chain:
            continue
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pr_c0 = np.eye(3, dtype=np.float64)
        t_pr_c0 = np.zeros(3, dtype=np.float64)
        R_dir_c0 = np.eye(3, dtype=np.float64)
        t_dir_c0 = np.zeros(3, dtype=np.float64)
        rows_metric: List[Dict[str, Any]] = []
        rows_dir: List[Dict[str, Any]] = []
        pos_errs: List[float] = []
        rot_errs: List[float] = []
        turn_errs: List[float] = []
        tdir_errs: List[float] = []
        cum_max = 0.0
        prev_gt_seg = None
        prev_pr_seg = None
        prev_dir_seg = None
        for rec in chain:
            R_gt_c0, t_gt_c0 = _compose_rel_pose_np(rec.R_gt, rec.t_gt_vec, R_gt_c0, t_gt_c0)
            R_pr_c0, t_pr_c0 = _compose_rel_pose_np(rec.R_pred, rec.t_vec_out, R_pr_c0, t_pr_c0)
            t_dir_only = rec.t_dir_out * rec.t_gt_mag
            R_dir_c0, t_dir_c0 = _compose_rel_pose_np(rec.R_pred, t_dir_only, R_dir_c0, t_dir_c0)
            p_gt = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            p_pr = _camera_center_from_T_c0_np(R_pr_c0, t_pr_c0)
            p_dir = _camera_center_from_T_c0_np(R_dir_c0, t_dir_c0)
            pos_err = float(np.linalg.norm(p_pr - p_gt))
            rot_err = float(_rot_geodesic_deg_np(rec.R_pred, rec.R_gt))
            tdir_err = float(_vec_angle_deg_np(rec.t_dir_out, rec.t_gt_dir))
            tmag_ratio = float(rec.pred_tmag / max(rec.t_gt_mag, 1e-12))
            pos_err_sq.append(pos_err ** 2)
            pos_errs.append(pos_err)
            cum_max = max(cum_max, pos_err)
            rot_errs.append(rot_err)
            tdir_errs.append(tdir_err)
            rpe_rot.append(rot_err)
            rpe_tdir.append(tdir_err)
            rpe_tmag.append(abs(tmag_ratio - 1.0))
            tmag_preds.append(rec.pred_tmag)
            gt_seg = rec.t_gt_vec
            pr_seg = rec.t_vec_out
            dir_seg = t_dir_only
            turn_err = float("nan")
            if prev_gt_seg is not None:
                gt_turn = float(_vec_angle_deg_np(gt_seg, prev_gt_seg))
                pr_turn = float(_vec_angle_deg_np(pr_seg, prev_pr_seg))
                turn_err = abs(pr_turn - gt_turn)
                if math.isfinite(turn_err):
                    turn_errs.append(turn_err)
            prev_gt_seg = gt_seg
            prev_pr_seg = pr_seg
            prev_dir_seg = dir_seg
            rows_metric.append({
                "step_idx": rec.step_idx,
                "i": rec.i,
                "j": rec.j,
                "dt_gt": rec.t_gt_mag,
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
                "turn_err_deg": turn_err,
                "dt_bucket": rec.dt_bucket,
                "k": rec.k,
                "chain_id": rec.chain_id,
                "scene_seq": rec.scene_seq,
            })
            rows_dir.append({
                "step_idx": rec.step_idx,
                "i": rec.i,
                "j": rec.j,
                "dt_gt": rec.t_gt_mag,
                "gt_x": float(p_gt[0]),
                "gt_y": float(p_gt[1]),
                "gt_z": float(p_gt[2]),
                "metric_x": float(p_dir[0]),
                "metric_y": float(p_dir[1]),
                "metric_z": float(p_dir[2]),
                "metric_pos_err": float(np.linalg.norm(p_dir - p_gt)),
            })
        metric_shape = _trajectory_shape_summary(rows_metric, "metric")
        dir_shape = _trajectory_shape_summary(rows_dir, "metric")
        metric_ratio = float(metric_shape.get("path_length_ratio", float("nan")))
        dir_ratio = float(dir_shape.get("path_length_ratio", float("nan")))
        metric_path_ratios.append(metric_ratio)
        dir_only_ratios.append(dir_ratio)
        gt_path_length = float(metric_shape.get("gt_path_length", float("nan")))
        if math.isfinite(metric_ratio) and math.isfinite(gt_path_length):
            weighted_num += metric_ratio * gt_path_length
            weighted_den += gt_path_length
        endpoint_err = float(pos_errs[-1]) if pos_errs else float("nan")
        endpoint_errs.append(endpoint_err)
        chain_summaries.append(
            ChainSummary(
                chain_id=int(chain[0].chain_id),
                scene_seq=str(chain[0].scene_seq),
                num_steps=len(chain),
                ate=float(math.sqrt(np.mean(np.square(pos_errs)))) if pos_errs else float("nan"),
                drift=endpoint_err,
                path_ratio=metric_ratio,
                mean_rot_error=float(np.mean(rot_errs)) if rot_errs else float("nan"),
                mean_turn_error=float(np.mean(turn_errs)) if turn_errs else float("nan"),
                mean_step_position_error=float(np.mean(pos_errs)) if pos_errs else float("nan"),
                max_cumulative_error=cum_max,
                mean_tdir_error=float(np.mean(tdir_errs)) if tdir_errs else float("nan"),
            )
        )
        step_rows.extend(rows_metric)

    summary = {
        "drift": float(np.mean(endpoint_errs)) if endpoint_errs else float("nan"),
        "ATE": float(math.sqrt(np.mean(pos_err_sq))) if pos_err_sq else float("nan"),
        "metric_path_ratio": metric_path_ratios[0] if metric_path_ratios else float("nan"),
        "direction_only_path_ratio": dir_only_ratios[0] if dir_only_ratios else float("nan"),
        "path_weighted_ratio": float(weighted_num / max(weighted_den, 1e-12)) if weighted_den > 0 else float("nan"),
        "RPE_rot": float(np.mean(rpe_rot)) if rpe_rot else float("nan"),
        "RPE_trans_dir": float(np.mean(rpe_tdir)) if rpe_tdir else float("nan"),
        "RPE_trans_mag": float(np.mean(rpe_tmag)) if rpe_tmag else float("nan"),
        "tmag_p10": float(_q(tmag_preds)["p10"]) if tmag_preds else float("nan"),
        "tmag_p50": float(_q(tmag_preds)["p50"]) if tmag_preds else float("nan"),
        "tmag_p90": float(_q(tmag_preds)["p90"]) if tmag_preds else float("nan"),
        "selected_k": 1,
        "num_pairs": len(step_rows),
        "num_chains": len(chain_summaries),
    }
    return summary, chain_summaries, step_rows, rows_metric


def _aggregate_rows(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    out = []
    for label in sorted(groups.keys()):
        rs = groups[label]
        pos = np.asarray([_safe_float(r.get("metric_pos_err")) for r in rs], dtype=np.float64)
        pos = pos[np.isfinite(pos)]
        rot = np.asarray([_safe_float(r.get("rot_err_deg")) for r in rs], dtype=np.float64)
        rot = rot[np.isfinite(rot)]
        turn = np.asarray([_safe_float(r.get("turn_err_deg")) for r in rs], dtype=np.float64)
        turn = turn[np.isfinite(turn)]
        out.append({
            "bucket": label,
            "count": len(rs),
            "rot_error": float(rot.mean()) if rot.size else float("nan"),
            "turn_error": float(turn.mean()) if turn.size else float("nan"),
            "ATE_contribution": float(math.sqrt(float(np.mean(np.square(pos))))) if pos.size else float("nan"),
            "drift_contribution": float(pos.mean()) if pos.size else float("nan"),
            "step_position_error": float(pos.mean()) if pos.size else float("nan"),
        })
    return out


def _rot_policy_global(value: float) -> Callable[[float], float]:
    return lambda _dt: float(value)


def _rot_policy_dt(bucket_map: Dict[str, float]) -> Callable[[float], float]:
    def fn(dt: float) -> float:
        return float(bucket_map.get(_dt_bucket(dt), 0.45))
    return fn


def _evaluate_policy(name: str, rot_policy: Callable[[float], float], device: torch.device) -> Dict[str, Any]:
    wrapped, cfg, load_summary, ds = _build_wrapped_model(device, rot_policy)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        wrapped, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(wrapped, ds, device, cfg, output_dir=str(run_dir), step=0, upd=0)
    chains = _collect_pair_records(wrapped, ds, device, selected_k=1)
    chain_diag, chain_summaries, step_rows, _ = _evaluate_chains(chains)
    return {
        "name": name,
        "rot": float(rot),
        "tdir_abs": float(tdir_abs),
        "tdir_local_A_abs": float(tdir_local_A_abs),
        "drift": float(_safe_float(odom.get("odom_metric_drift"))),
        "ATE": float(_safe_float(odom.get("odom_metric_ATE"))),
        "metric_path_ratio": float(_safe_float(odom.get("odom_shape_metric_mean_path_length_ratio"))),
        "direction_only_path_ratio": float(_safe_float(odom.get("odom_shape_direction_only_mean_path_length_ratio"), chain_diag["direction_only_path_ratio"])),
        "RPE_rot": float(_safe_float(odom.get("odom_metric_RPE_rot"))),
        "RPE_trans_dir": float(_safe_float(odom.get("odom_metric_RPE_trans_dir"))),
        "RPE_trans_mag": float(_safe_float(odom.get("odom_metric_RPE_trans_mag"))),
        "tmag_p10": float(chain_diag["tmag_p10"]),
        "tmag_p50": float(chain_diag["tmag_p50"]),
        "tmag_p90": float(chain_diag["tmag_p90"]),
        "selected_k": int(odom.get("odom_selected_k", chain_diag["selected_k"])),
        "num_pairs": int(odom.get("odom_num_pairs", chain_diag["num_pairs"])),
        "num_chains": int(odom.get("odom_num_chains", chain_diag["num_chains"])),
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "dt_bucket_breakdown": _aggregate_rows(step_rows, "dt_bucket"),
        "k_bucket_breakdown": _aggregate_rows(step_rows, "k"),
        "chain_summaries": chain_summaries,
        "step_rows": step_rows,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


def _render_bucket_table(rows: Sequence[Dict[str, Any]], title: str) -> List[str]:
    lines = [f"### {title}\n", "| bucket | count | rot_error | turn_error | ATE_contribution | drift_contribution | step_position_error |\n", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"]
    for r in rows:
        lines.append(
            f"| {r['bucket']} | {r['count']} | {_fmt(r['rot_error'])} | {_fmt(r['turn_error'])} | {_fmt(r['ATE_contribution'])} | {_fmt(r['drift_contribution'])} | {_fmt(r['step_position_error'])} |\n"
        )
    return lines


def _render_chain_table(chains: Sequence[ChainSummary], title: str, limit: int = 10) -> List[str]:
    rows = sorted(chains, key=lambda c: c.ate, reverse=True)[:limit]
    lines = [f"### {title}\n", "| chain_id | scene_seq | num_steps | ATE | drift | path_ratio | mean_rot_error | mean_turn_error | mean_step_position_error | max_cumulative_error |\n", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"]
    for c in rows:
        lines.append(
            f"| {c.chain_id} | {c.scene_seq} | {c.num_steps} | {_fmt(c.ate)} | {_fmt(c.drift)} | {_fmt(c.path_ratio)} | {_fmt(c.mean_rot_error)} | {_fmt(c.mean_turn_error)} | {_fmt(c.mean_step_position_error)} | {_fmt(c.max_cumulative_error)} |\n"
        )
    return lines


def _render_step_list(rows: Sequence[Dict[str, Any]], key: str, title: str, limit: int = 10) -> List[str]:
    sorted_rows = sorted(rows, key=lambda r: _safe_float(r.get(key), float("-inf")), reverse=True)[:limit]
    lines = [f"### {title}\n"]
    for r in sorted_rows:
        lines.append(
            f"- chain {int(r['chain_id'])} step {int(r['step_idx'])} (i={int(r['i'])}, j={int(r['j'])}): "
            f"rot_err={_fmt(r.get('rot_err_deg'))}, turn_err={_fmt(r.get('turn_err_deg'))}, "
            f"tdir_err={_fmt(r.get('tdir_err_deg'))}, pos_err={_fmt(r.get('metric_pos_err'))}\n"
        )
    return lines


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = _evaluate_policy("S2b_baseline", _rot_policy_global(0.45), device)

    fine_rows: List[Dict[str, Any]] = []
    for rot in ROT_VALUES:
        row = _evaluate_policy(f"global_rot_{rot:.3f}", _rot_policy_global(rot), device)
        row["fine_rot"] = rot
        fine_rows.append(row)

    dt_rows: List[Dict[str, Any]] = []
    for name, mapping in DT_POLICIES.items():
        row = _evaluate_policy(name, _rot_policy_dt(mapping), device)
        row["policy_map"] = mapping
        dt_rows.append(row)

    # k-aware is not meaningful under official selected_k=1 odom scope.
    k_aware_note = "Not feasible as an odom policy under official selected_k=1 scope; all evaluated odom chains use k=1."

    # best diagnostic among dt-aware only
    dt_positive = [
        r for r in dt_rows
        if r["name"] != "global_s2b_baseline"
        and r["metric_path_ratio"] >= 0.90
        and r["ATE"] < 7.352371
        and r["drift"] <= 1.35
    ]
    best_dt = min(dt_positive, key=lambda r: (r["ATE"], r["drift"])) if dt_positive else min(
        [r for r in dt_rows if r["name"] != "global_s2b_baseline"], key=lambda r: (r["ATE"], r["drift"])
    )

    best_global = min(fine_rows, key=lambda r: (r["ATE"], r["drift"]))

    if dt_positive:
        verdict = "DT-AWARE-ROT-POSITIVE-DIAGNOSTIC"
        best_diag = best_dt
    elif best_global["fine_rot"] == 0.45:
        verdict = "GLOBAL-ROT-STILL-BEST"
        best_diag = baseline
    elif best_global["ATE"] < baseline["ATE"] and best_global["drift"] <= 1.35 and best_global["metric_path_ratio"] >= 0.90:
        verdict = "INCONCLUSIVE"
        best_diag = best_global
    else:
        verdict = "ROTATION-NOT-MAIN-BOTTLENECK"
        best_diag = best_global

    corr_base = {
        "corr(ATE, mean_rot_error)": _corr([c.ate for c in baseline["chain_summaries"]], [c.mean_rot_error for c in baseline["chain_summaries"]]),
        "corr(ATE, mean_turn_error)": _corr([c.ate for c in baseline["chain_summaries"]], [c.mean_turn_error for c in baseline["chain_summaries"]]),
        "corr(ATE, path_ratio_error)": _corr([c.ate for c in baseline["chain_summaries"]], [abs(c.path_ratio - 1.0) for c in baseline["chain_summaries"]]),
        "corr(drift, mean_rot_error)": _corr([c.drift for c in baseline["chain_summaries"]], [c.mean_rot_error for c in baseline["chain_summaries"]]),
        "corr(drift, mean_turn_error)": _corr([c.drift for c in baseline["chain_summaries"]], [c.mean_turn_error for c in baseline["chain_summaries"]]),
    }

    lines: List[str] = []
    lines.append("# S2f Rotation Consistency Audit On S2b\n\n")
    lines.append("## S2b baseline\n")
    lines.append(f"- policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`\n")
    lines.append(f"- drift = {_fmt(baseline['drift'])}\n")
    lines.append(f"- ATE = {_fmt(baseline['ATE'])}\n")
    lines.append(f"- path_ratio = {_fmt(baseline['metric_path_ratio'])}\n")
    lines.append(f"- RPE_rot = {_fmt(baseline['RPE_rot'])}\n")
    lines.append(f"- RPE_trans_dir = {_fmt(baseline['RPE_trans_dir'])}\n")
    lines.append(f"- RPE_trans_mag = {_fmt(baseline['RPE_trans_mag'])}\n")
    lines.append(f"- rot = {_fmt(baseline['rot'])}\n")
    lines.append(f"- tdir_abs = {_fmt(baseline['tdir_abs'])}\n")
    lines.append(f"- tdir_local_A_abs = {_fmt(baseline['tdir_local_A_abs'])}\n")
    lines.append(f"- selected_k / num_pairs / num_chains = {baseline['selected_k']} / {baseline['num_pairs']} / {baseline['num_chains']}\n")
    lines.append(f"- missing / unexpected = {baseline['load_missing']} / {baseline['load_unexpected']}\n\n")

    lines.append("## Fine-grained global fine_rot sweep\n")
    lines.append("| fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | direction_only_path_ratio | tmag_p10 | tmag_p50 | tmag_p90 | selected_k | num_pairs | num_chains | missing | unexpected |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in fine_rows:
        lines.append(
            f"| {_fmt(r['fine_rot'],3)} | {_fmt(r['drift'])} | {_fmt(r['ATE'])} | {_fmt(r['metric_path_ratio'])} | {_fmt(r['RPE_rot'])} | {_fmt(r['RPE_trans_dir'])} | {_fmt(r['RPE_trans_mag'])} | {_fmt(r['rot'])} | {_fmt(r['tdir_abs'])} | {_fmt(r['tdir_local_A_abs'])} | {_fmt(r['direction_only_path_ratio'])} | {_fmt(r['tmag_p10'])} | {_fmt(r['tmag_p50'])} | {_fmt(r['tmag_p90'])} | {r['selected_k']} | {r['num_pairs']} | {r['num_chains']} | {r['load_missing']} | {r['load_unexpected']} |\n"
        )
    lines.append("\n")

    lines.append("## DT-aware / K-aware diagnostic policy table\n")
    lines.append("| policy | mapping | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | selected_k | num_pairs | num_chains | missing | unexpected | notes |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for r in dt_rows:
        lines.append(
            f"| {r['name']} | `{r.get('policy_map')}` | {_fmt(r['drift'])} | {_fmt(r['ATE'])} | {_fmt(r['metric_path_ratio'])} | {_fmt(r['RPE_rot'])} | {_fmt(r['RPE_trans_dir'])} | {_fmt(r['RPE_trans_mag'])} | {r['selected_k']} | {r['num_pairs']} | {r['num_chains']} | {r['load_missing']} | {r['load_unexpected']} | dt-aware diagnostic |\n"
        )
    lines.append(f"| k_aware_diagnostic | n/a | nan | nan | nan | nan | nan | nan | {baseline['selected_k']} | {baseline['num_pairs']} | {baseline['num_chains']} | 2 | 0 | {k_aware_note} |\n")
    lines.append("\n")

    lines.append("## Bucket-wise error breakdown\n")
    lines.extend(_render_bucket_table(baseline["dt_bucket_breakdown"], "S2b baseline dt bucket"))
    lines.append("\n")
    lines.extend(_render_bucket_table(baseline["k_bucket_breakdown"], "S2b baseline k bucket"))
    lines.append("\n")
    if best_diag is not baseline:
        lines.extend(_render_bucket_table(best_diag["dt_bucket_breakdown"], "Best diagnostic dt bucket"))
        lines.append("\n")
        lines.extend(_render_bucket_table(best_diag["k_bucket_breakdown"], "Best diagnostic k bucket"))
        lines.append("\n")

    lines.append("## Chain-level worst cases\n")
    lines.extend(_render_chain_table(baseline["chain_summaries"], "S2b baseline top chains by ATE"))
    lines.append("\n")
    if best_diag is not baseline:
        lines.extend(_render_chain_table(best_diag["chain_summaries"], "Best diagnostic top chains by ATE"))
        lines.append("\n")

    lines.append("## Step-level worst cases\n")
    lines.extend(_render_step_list(baseline["step_rows"], "rot_err_deg", "Top 10 rot error steps"))
    lines.append("\n")
    lines.extend(_render_step_list(baseline["step_rows"], "turn_err_deg", "Top 10 turn error steps"))
    lines.append("\n")
    lines.extend(_render_step_list(baseline["step_rows"], "metric_pos_err", "Top 10 cumulative position error steps"))
    lines.append("\n")

    lines.append("## Correlation analysis\n")
    for k, v in corr_base.items():
        lines.append(f"- {k} = `{_fmt(v)}`\n")
    lines.append("\n")

    better_than_s2b = (
        best_diag is not baseline
        and best_diag["metric_path_ratio"] >= 0.90
        and best_diag["ATE"] < 7.352371
        and best_diag["drift"] <= 1.35
    )
    lines.append("## Interpretation\n")
    lines.append(f"- best global fine_rot in this sweep = `{_fmt(best_global['fine_rot'],3)}` with ATE={_fmt(best_global['ATE'])}, drift={_fmt(best_global['drift'])}\n")
    lines.append(f"- best dt-aware diagnostic policy = `{best_dt['name']}` with ATE={_fmt(best_dt['ATE'])}, drift={_fmt(best_dt['drift'])}, path_ratio={_fmt(best_dt['metric_path_ratio'])}\n")
    lines.append(f"- exists diagnostic policy better than S2b under the positive threshold = {better_than_s2b}\n")
    lines.append(f"- k-aware diagnostic feasibility = False (`selected_k=1` official odom scope)\n\n")

    lines.append("## Final verdict\n")
    lines.append(f"- `{verdict}`\n\n")

    lines.append("## Next step\n")
    if verdict == "DT-AWARE-ROT-POSITIVE-DIAGNOSTIC":
        lines.append("- Next: `S2g_train_cv_dt_aware_fine_rot_policy_selection_on_s2b`.\n")
    elif verdict == "K-AWARE-ROT-POSITIVE-DIAGNOSTIC":
        lines.append("- Next: `S2g_train_cv_k_aware_fine_rot_policy_selection_on_s2b`.\n")
    elif verdict == "GLOBAL-ROT-STILL-BEST":
        lines.append("- Keep `S2b` as the current clean fine-rot candidate and consider final reporting or deeper representation/model-design diagnosis.\n")
    elif verdict == "ROTATION-NOT-MAIN-BOTTLENECK":
        lines.append("- Do not continue fine_rot policy tuning; move to trajectory-shape or representation diagnosis.\n")
    else:
        lines.append("- Signal is mixed; do not upgrade the mainline without train-CV clean selection.\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
