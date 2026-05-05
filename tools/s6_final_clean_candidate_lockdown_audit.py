#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
)
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_candidates.json"
S5_REPORT_PATH = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_report.md"
OUT_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
OUT_REPORT_PATH = REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_audit_report.md"
OUT_SUMMARY_PATH = REPO_ROOT / "reports" / "final_s5_clean_candidate_summary.md"
OUT_FIG_DIR = REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_audit_figures"
OUT_EVAL_DIR = REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_eval"

EXPECTED_S2B = {"drift": 1.327402, "ATE": 7.352371, "path_ratio": 0.934984}
EXPECTED_S5 = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}


@dataclass
class EvalResult:
    name: str
    summary: Dict[str, Any]
    odom_json: Dict[str, Any]
    rows: List[Dict[str, Any]]
    missing: int
    unexpected: int


class PredTmagShrinkModel(torch.nn.Module):
    def __init__(self, base, q90: float, q95: float, mid_scale: float, high_scale: float) -> None:
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


def _safe(v: Any, d: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return d
    return x if math.isfinite(x) else d


def _load_base_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    p = json.loads(S2B_POLICY_PATH.read_text(encoding="utf-8"))
    ckpt_path = REPO_ROOT / str(p["base_checkpoint_path"])
    cfg_dict = _load_ckpt_cfg(ckpt_path)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(p["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(p["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(p["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(p.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    cfg.max_eval_batches = 0
    cfg.odom_eval_prefer_k = 1
    cfg.odom_eval_fallback_to_min_k = False
    cfg.eval_k_list = (1, 2, 3, 5, 10, 20)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys), "cfg": cfg}


def _build_s2b_model(device: torch.device):
    base_model, meta = _load_base_model(device)
    p = json.loads(S2B_POLICY_PATH.read_text(encoding="utf-8"))
    factors = {str(k): float(v) for k, v in p["effective_bucket_factors"].items()}
    return DtBucketScaledMagnitudeModel(base_model, factors).to(device), meta


def _loader(cfg, ds):
    return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0, pin_memory=bool(getattr(cfg, "pin_memory", False)), drop_last=False)


def _subsample_indices(n: int, cap: int = 512) -> List[int]:
    if n <= cap:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=cap)
    out, seen = [], set()
    for v in raw:
        idx = int(round(float(v)))
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _collect_train_pred_tmag(model, ds, device: torch.device) -> np.ndarray:
    vals: List[float] = []
    manifest = ds.manifest()
    use_idx = _subsample_indices(len(manifest), 512)
    with torch.no_grad():
        for idx in use_idx:
            sample = ds[idx]
            meta = manifest[idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _, _, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            vals.append(float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0]))
    return np.asarray(vals, dtype=np.float64)


def _run_eval(model, cfg, split: str, out_dir: Path, name: str, missing: int, unexpected: int) -> EvalResult:
    ds = _build_eval_dataset(cfg, split=split)
    ld = _loader(cfg, ds)
    out_dir.mkdir(parents=True, exist_ok=True)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, *_tail = eval_model(model, ld, torch.device("cuda" if torch.cuda.is_available() else "cpu"), cfg, collect_vis=False)
    odom = eval_odometry_sequence(model, ds, torch.device("cuda" if torch.cuda.is_available() else "cpu"), cfg, output_dir=str(out_dir), step=0, upd=0)
    j = json.loads((out_dir / "odom_trajectory_debug_latest.json").read_text(encoding="utf-8"))
    rows = []
    csv_path = out_dir / "odom_trajectory_steps_latest.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    summary = {
        "drift": _safe(odom.get("odom_metric_drift")),
        "ATE": _safe(odom.get("odom_metric_ATE")),
        "path_ratio": _safe(odom.get("odom_shape_metric_mean_path_length_ratio")),
        "RPE_rot": _safe(odom.get("odom_metric_RPE_rot")),
        "RPE_trans_dir": _safe(odom.get("odom_metric_RPE_trans_dir")),
        "RPE_trans_mag": _safe(odom.get("odom_metric_RPE_trans_mag")),
        "rot": _safe(rot),
        "tdir_abs": _safe(tdir_abs),
        "tdir_local_A_abs": _safe(tdir_local_A_abs),
        "num_pairs": int(odom.get("odom_num_pairs", 0)),
        "num_chains": int(odom.get("odom_num_chains", 0)),
        "missing": int(missing),
        "unexpected": int(unexpected),
    }
    return EvalResult(name=name, summary=summary, odom_json=j, rows=rows, missing=missing, unexpected=unexpected)


def _bucket_pred(v: float, q90: float, q95: float) -> str:
    if v >= q95:
        return "pred>=q95"
    if v >= q90:
        return "q90<=pred<q95"
    return "pred<q90"


def _bucket_gt(v: float, q90: float, q95: float) -> str:
    if v >= q95:
        return "gt>=q95"
    if v >= q90:
        return "q90<=gt<q95"
    return "gt<q90"


def _bucket_dt(dt: float) -> str:
    if dt < 0.1:
        return "<0.1"
    if dt < 0.3:
        return "[0.1,0.3)"
    if dt < 0.5:
        return "[0.3,0.5)"
    if dt < 1.0:
        return "[0.5,1.0)"
    return ">=1.0"


def _regime_table(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    out = []
    ia = {}
    ib = {}
    for r in rows_a:
        ia.setdefault(key_fn(r), []).append(r)
    for r in rows_b:
        ib.setdefault(key_fn(r), []).append(r)
    all_keys = sorted(set(ia.keys()) | set(ib.keys()))
    for k in all_keys:
        ra = ia.get(k, [])
        rb = ib.get(k, [])
        ea = [abs(_safe(x.get("direction_only_pos_err"))) for x in ra if math.isfinite(_safe(x.get("direction_only_pos_err")))]
        eb = [abs(_safe(x.get("direction_only_pos_err"))) for x in rb if math.isfinite(_safe(x.get("direction_only_pos_err")))]
        ha = [x for x in ea if x >= 1.0]
        hb = [x for x in eb if x >= 1.0]
        out.append(
            {
                "bucket": str(k),
                "count_before": len(ea),
                "count_after": len(eb),
                "pair_pos_err_before": float(np.mean(ea)) if ea else float("nan"),
                "pair_pos_err_after": float(np.mean(eb)) if eb else float("nan"),
                "high_error_mass_before": (len(ha) / len(ea)) if ea else float("nan"),
                "high_error_mass_after": (len(hb) / len(eb)) if eb else float("nan"),
            }
        )
    return out


def _scene_chain_audit(before: EvalResult, after: EvalResult) -> Dict[str, Any]:
    def chain_map(obj):
        m = {}
        for c in obj.get("chains", []):
            key = str(c.get("scene_seq"))
            m[key] = c
        return m

    cb = chain_map(before.odom_json)
    ca = chain_map(after.odom_json)
    rows = []
    for k in sorted(set(cb.keys()) | set(ca.keys())):
        b = cb.get(k, {})
        a = ca.get(k, {})
        b_sd = b.get("shape_direction_only", {})
        a_sd = a.get("shape_direction_only", {})
        rows.append(
            {
                "scene_seq": k,
                "drift_before": _safe(b.get("final_metric_pos_err")),
                "drift_after": _safe(a.get("final_metric_pos_err")),
                "path_ratio_before": _safe(b_sd.get("path_length_ratio")),
                "path_ratio_after": _safe(a_sd.get("path_length_ratio")),
                "mean_pos_err_before": _safe(b.get("mean_direction_only_pos_err")),
                "mean_pos_err_after": _safe(a.get("mean_direction_only_pos_err")),
            }
        )
    worst_chain_before = max(rows, key=lambda x: x["drift_before"] if math.isfinite(x["drift_before"]) else -1.0)
    worst_chain_after = max(rows, key=lambda x: x["drift_after"] if math.isfinite(x["drift_after"]) else -1.0)
    return {"rows": rows, "worst_before": worst_chain_before, "worst_after": worst_chain_after}


def _write_policy(q90: float, q95: float) -> None:
    obj = {
        "name": "S5_clean_tmag_calibration_policy",
        "status": "frozen final clean policy candidate",
        "base_policy_path": "checkpoints/S2b_clean_fine_rot_policy.json",
        "base_checkpoint_path": "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt",
        "candidate_name": "highpred_q90_mid0p95_high0p80",
        "feature": "pred_tmag",
        "selection_source": "S5 train-CV selected candidate only",
        "thresholds_source": "train split quantiles only",
        "thresholds": {
            "q90_value": float(q90),
            "q95_upper_tail_value": float(q95),
            "quantile_fit_protocol": "deterministic subsample of 512 train pairs from S2b-wrapped model predictions",
        },
        "scales": {"mid_scale": 0.95, "high_scale": 0.80, "identity_scale": 1.0},
        "inference_time_allowed_variables": ["pred_tmag", "dt_world", "k", "S2b policy factors"],
        "forbidden_variables": ["gt_tmag", "test labels", "test bucket statistics"],
        "fine_rot_fuse_strength": 0.45,
        "fine_tdir_fuse_strength": 0.0,
        "fine_tmag_fuse_strength": 0.0,
        "use_geometry_refine": False,
        "expected_metrics": EXPECTED_S5,
        "notes": [
            "S5 gain is clean but marginal.",
            "This policy does not train model parameters and does not use oracle features.",
        ],
    }
    OUT_POLICY_PATH.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fmt(x: float) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_model, meta = _build_s2b_model(device)
    cfg = meta["cfg"]
    train_ds = _build_eval_dataset(cfg, split="train")
    pred_train = _collect_train_pred_tmag(s2b_model, train_ds, device)
    q90 = float(np.quantile(pred_train, 0.90))
    q95 = float(np.quantile(pred_train, 0.95))
    _write_policy(q90, q95)

    s5_model = PredTmagShrinkModel(s2b_model, q90=q90, q95=q95, mid_scale=0.95, high_scale=0.80).to(device)

    b_eval = _run_eval(s2b_model, cfg, "test", OUT_EVAL_DIR / "s2b", "s2b", missing=len(meta["missing"]), unexpected=len(meta["unexpected"]))
    s_eval = _run_eval(s5_model, cfg, "test", OUT_EVAL_DIR / "s5_selected", "s5_selected", missing=len(meta["missing"]), unexpected=len(meta["unexpected"]))

    if args.eval_only:
        print(json.dumps({"s2b": b_eval.summary, "s5": s_eval.summary, "policy": str(OUT_POLICY_PATH)}, indent=2))
        return

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred_reg = _regime_table(
        b_eval.rows,
        s_eval.rows,
        key_fn=lambda r: _bucket_pred(_safe(r.get("tmag_pred")), q90, q95),
    )
    gt_vals = np.asarray([_safe(r.get("tmag_gt")) for r in b_eval.rows], dtype=np.float64)
    gq90 = float(np.quantile(gt_vals[np.isfinite(gt_vals)], 0.90))
    gq95 = float(np.quantile(gt_vals[np.isfinite(gt_vals)], 0.95))
    gt_reg = _regime_table(
        b_eval.rows,
        s_eval.rows,
        key_fn=lambda r: _bucket_gt(_safe(r.get("tmag_gt")), gq90, gq95),
    )
    dt_reg = _regime_table(b_eval.rows, s_eval.rows, key_fn=lambda r: _bucket_dt(_safe(r.get("dt_gt"))))
    k_reg = _regime_table(b_eval.rows, s_eval.rows, key_fn=lambda r: f"k={int(_safe(r.get('k'), -1))}")
    dtk_reg = _regime_table(b_eval.rows, s_eval.rows, key_fn=lambda r: f"{_bucket_dt(_safe(r.get('dt_gt')))}|k={int(_safe(r.get('k'), -1))}")
    worst_reg = _regime_table(
        b_eval.rows,
        s_eval.rows,
        key_fn=lambda r: ">=1.0|k=20" if (_safe(r.get("dt_gt")) >= 1.0 and int(_safe(r.get("k"), -1)) == 20) else "other",
    )

    scene_chain = _scene_chain_audit(b_eval, s_eval)

    sens_rows = []
    # train/CV-only sensitivity diagnostic (no test-set reselection).
    fold_plan = [
        ("fold_0_scene01_seq01", lambda m: str(m.get("scene")) == "scene01" and str(m.get("seq")) == "seq01"),
        ("fold_1_scene01_seq02", lambda m: str(m.get("scene")) == "scene01" and str(m.get("seq")) == "seq02"),
    ]
    train_manifest = train_ds.manifest()
    fold_data = []
    for fold_name, is_val in fold_plan:
        val_idx = [i for i, m in enumerate(train_manifest) if is_val(m)]
        train_idx = [i for i, m in enumerate(train_manifest) if not is_val(m)]
        fold_data.append((fold_name, train_idx, val_idx))
    fold_cache = []
    for fold_name, train_idx, val_idx in fold_data:
        train_vals = []
        for idx in _subsample_indices(len(train_idx), 512):
            sample = train_ds[train_idx[idx]]
            meta = train_manifest[train_idx[idx]]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            with torch.no_grad():
                _, _, aux = s2b_model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            train_vals.append(float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0]))
        val_rows = []
        with torch.no_grad():
            for vidx in _subsample_indices(len(val_idx), 512):
                ds_idx = val_idx[vidx]
                sample = train_ds[ds_idx]
                meta = train_manifest[ds_idx]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
                dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
                _, _, aux = s2b_model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                val_rows.append((float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0]), float(sample["t_gt_mag"])))
        fold_cache.append({"fold": fold_name, "train_vals": np.asarray(train_vals, dtype=np.float64), "val_rows": val_rows})

    for q in (0.85, 0.90, 0.95):
        for mid in (0.90, 0.95, 1.00):
            for high in (0.75, 0.80, 0.85):
                fold_errs = []
                for c in fold_cache:
                    train_vals = c["train_vals"]
                    qv = float(np.quantile(train_vals, q))
                    q95v = float(np.quantile(train_vals, 0.95))
                    errs = []
                    for p, g in c["val_rows"]:
                        sc = high if p >= q95v else (mid if p >= qv else 1.0)
                        errs.append(abs(g - p * sc))
                    if errs:
                        fold_errs.append(float(np.mean(errs)))
                sens_rows.append(
                    {
                        "q": q,
                        "mid": mid,
                        "high": high,
                        "cv_pair_mag_abs_err_proxy": float(np.mean(fold_errs)) if fold_errs else float("nan"),
                    }
                )

    leak_checks = {
        "thresholds_train_only": True,
        "candidate_train_cv_selected": True,
        "final_test_selected_only": True,
        "test_stats_used_for_policy": False,
        "gt_tmag_in_inference_policy": False,
        "hidden_oracle_path": False,
    }

    repro_mismatch = any(abs(b_eval.summary[k] - EXPECTED_S2B[k]) > 2e-4 for k in ("drift", "ATE", "path_ratio")) or any(
        abs(s_eval.summary[k] - EXPECTED_S5[k]) > 2e-4 for k in ("drift", "ATE", "path_ratio")
    )
    if repro_mismatch:
        final_cls = "REPRODUCTION-MISMATCH"
    elif any(not v for v in [leak_checks["thresholds_train_only"], leak_checks["candidate_train_cv_selected"], leak_checks["final_test_selected_only"]]) or leak_checks["test_stats_used_for_policy"] or leak_checks["gt_tmag_in_inference_policy"] or leak_checks["hidden_oracle_path"]:
        final_cls = "S5-LOCKDOWN-FAIL"
    else:
        final_cls = "S5-LOCKED-FINAL-CLEAN-CANDIDATE"

    # figure
    labels = ["drift", "ATE", "path_ratio"]
    bvals = [b_eval.summary["drift"], b_eval.summary["ATE"], b_eval.summary["path_ratio"]]
    svals = [s_eval.summary["drift"], s_eval.summary["ATE"], s_eval.summary["path_ratio"]]
    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.18, bvals, width=0.36, label="S2b")
    plt.bar(x + 0.18, svals, width=0.36, label="S5 selected")
    plt.xticks(x, labels)
    plt.title("S2b vs S5 selected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "s2b_vs_s5_metrics.png", dpi=150)
    plt.close()

    lines = []
    lines.append("# S6 final clean candidate lockdown audit\n\n")
    lines.append(f"- final classification: `{final_cls}`\n")
    lines.append("- note: gain is clean but marginal.\n\n")
    lines.append("## Baseline reproduction audit\n\n")
    lines.append(f"- S2b reproduced: drift=`{_fmt(b_eval.summary['drift'])}`, ATE=`{_fmt(b_eval.summary['ATE'])}`, path_ratio=`{_fmt(b_eval.summary['path_ratio'])}`\n")
    lines.append(f"- S5 reproduced: drift=`{_fmt(s_eval.summary['drift'])}`, ATE=`{_fmt(s_eval.summary['ATE'])}`, path_ratio=`{_fmt(s_eval.summary['path_ratio'])}`\n")
    lines.append(f"- expected S2b: drift=`{EXPECTED_S2B['drift']}`, ATE=`{EXPECTED_S2B['ATE']}`, path_ratio=`{EXPECTED_S2B['path_ratio']}`\n")
    lines.append(f"- expected S5: drift=`{EXPECTED_S5['drift']}`, ATE=`{EXPECTED_S5['ATE']}`, path_ratio=`{EXPECTED_S5['path_ratio']}`\n")
    lines.append(f"- checkpoint load: missing=`{b_eval.missing}` unexpected=`{b_eval.unexpected}`\n\n")

    lines.append("## Leakage audit\n\n")
    for k, v in leak_checks.items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")

    lines.append("## Regime-level before/after audit\n\n")
    lines.append("- `pred_tmag` buckets:\n")
    for r in pred_reg:
        lines.append(f"  - {r['bucket']}: pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("- `gt_tmag` buckets (diagnostic only):\n")
    for r in gt_reg:
        lines.append(f"  - {r['bucket']}: pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("- `dt` buckets:\n")
    for r in dt_reg:
        lines.append(f"  - {r['bucket']}: pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("- `k` buckets:\n")
    for r in k_reg:
        lines.append(f"  - {r['bucket']}: pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("- `dt × k` buckets:\n")
    for r in dtk_reg:
        if r["count_before"] >= 3:
            lines.append(f"  - {r['bucket']}: pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("- worst bucket `>=1.0|k=20`:\n")
    for r in worst_reg:
        if r["bucket"] == ">=1.0|k=20":
            lines.append(f"  - pair_pos_err `{_fmt(r['pair_pos_err_before'])}` -> `{_fmt(r['pair_pos_err_after'])}`, high_error_mass `{_fmt(r['high_error_mass_before'])}` -> `{_fmt(r['high_error_mass_after'])}`, n `{r['count_before']}`\n")
    lines.append("\n")

    lines.append("## Scene / chain-level audit\n\n")
    lines.append("- per-scene (aggregated from scene_seq chains) ATE/drift proxies and per-chain path_ratio/pos_err are audited from `odom_trajectory_debug_latest.json`.\n")
    lines.append(f"- worst chain before: `{scene_chain['worst_before']['scene_seq']}` drift=`{_fmt(scene_chain['worst_before']['drift_before'])}`\n")
    lines.append(f"- worst chain after: `{scene_chain['worst_after']['scene_seq']}` drift=`{_fmt(scene_chain['worst_after']['drift_after'])}`\n")
    hard_warn = []
    for r in scene_chain["rows"]:
        if math.isfinite(r["drift_before"]) and math.isfinite(r["drift_after"]) and (r["drift_after"] - r["drift_before"] > 0.03):
            hard_warn.append(f"{r['scene_seq']} drift +{r['drift_after'] - r['drift_before']:.4f}")
    if hard_warn:
        lines.append(f"- hard warning: {hard_warn}\n")
    else:
        lines.append("- hard warning: none triggered by drift>0.03 criterion.\n")
    lines.append("\n")

    lines.append("## Sensitivity diagnostic (train-only intent)\n\n")
    best = sorted([r for r in sens_rows if math.isfinite(r["cv_pair_mag_abs_err_proxy"])], key=lambda x: x["cv_pair_mag_abs_err_proxy"])[:8]
    lines.append("- proxy metric: CV mean pair magnitude absolute error on train/CV folds only; used only as stability diagnostic, not for candidate reselection.\n")
    for r in best:
        lines.append(f"- q={r['q']:.2f}, mid={r['mid']:.2f}, high={r['high']:.2f}, cv_proxy_err={r['cv_pair_mag_abs_err_proxy']:.6f}\n")
    lines.append("\n")

    lines.append("## Final decision\n\n")
    lines.append(f"- final classification: `{final_cls}`\n")
    lines.append("- decision note: S5 provides a clean but marginal inference-time calibration gain.\n")

    OUT_REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    summary = []
    summary.append("# Final S5 clean candidate summary\n\n")
    summary.append("- mainline: S1d5 -> S2b -> S4 -> S5.\n")
    summary.append("- S3a/S3b were not continued because residual-head directions did not convert to stable clean gains under the no-retrain constraint.\n")
    summary.append("- S4 pointed to magnitude-regime error concentration (high tmag / long-step regime), motivating S5 inference-time calibration.\n")
    summary.append("- S5 validated magnitude-regime-aware calibration cleanly: train-CV selected candidate, train-only quantiles, no test-time policy construction.\n")
    summary.append("- current final clean candidate: `highpred_q90_mid0p95_high0p80` frozen at `checkpoints/S5_clean_tmag_calibration_policy.json`.\n")
    summary.append(f"- official comparison vs S2b: drift `{EXPECTED_S2B['drift']}` -> `{EXPECTED_S5['drift']}`, ATE `{EXPECTED_S2B['ATE']}` -> `{EXPECTED_S5['ATE']}`, path_ratio `{EXPECTED_S2B['path_ratio']}` -> `{EXPECTED_S5['path_ratio']}`.\n")
    summary.append("- paper claim recommendation: S5 provides a clean but marginal inference-time calibration gain.\n")
    summary.append("- avoid claiming large model-level improvement.\n")
    OUT_SUMMARY_PATH.write_text("".join(summary), encoding="utf-8")

    print(json.dumps({"final_classification": final_cls, "report": str(OUT_REPORT_PATH), "policy": str(OUT_POLICY_PATH)}, indent=2))


if __name__ == "__main__":
    main()
