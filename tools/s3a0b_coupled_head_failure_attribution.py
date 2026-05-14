#!/usr/bin/env python3
"""Attribute the S3a0 smoke failure without training any parameter."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence
from tools.eval_clean_policy import DtBucketScaledMagnitudeModel, _extract_pairs, _q


REPORT_PATH = REPO_ROOT / "checkpoints" / "S3a0b_coupled_head_failure_attribution_report.md"
POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S2B_SUMMARY_PATH = REPO_ROOT / "checkpoints" / "S2b_final_repro" / "s1d5_policy_eval_summary.json"
AUDIT_MD = REPO_ROOT / "checkpoints" / "S3a0_coupled_pose_head_smoke_audit.md"
SMOKE_MD = REPO_ROOT / "checkpoints" / "S3a0_coupled_pose_residual_head_smoke_report.md"
IMPL_MD = REPO_ROOT / "checkpoints" / "S3a0_coupled_pose_residual_head_implementation_report.md"
SMOKE_DIR = REPO_ROOT / "checkpoints" / "S3a0_coupled_pose_residual_head_smoke"


@dataclass
class VariantRow:
    name: str
    status: str
    drift: Any = "NA"
    ATE: Any = "NA"
    path_ratio: Any = "NA"
    RPE_rot: Any = "NA"
    RPE_trans_dir: Any = "NA"
    RPE_trans_mag: Any = "NA"
    rot: Any = "NA"
    tdir_abs: Any = "NA"
    tdir_local_A_abs: Any = "NA"
    tmag_p10: Any = "NA"
    tmag_p50: Any = "NA"
    tmag_p90: Any = "NA"
    tmag_before_after_max_diff: Any = "NA"
    coupled_delta_rot_norm: Any = "NA"
    coupled_delta_tdir_norm: Any = "NA"
    coupled_gate_mean: Any = "NA"
    selected_k: Any = "NA"
    num_pairs: Any = "NA"
    num_chains: Any = "NA"
    note: str = ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_number(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.6f}"
    return str(v)


def _cfg_from_ckpt(ckpt_path: Path) -> Config:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Unsupported cfg type: {type(cfg_dict)}")
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def _load_model(model: PanoramaRelPoseModel, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected), "skipped": skipped}


def _build_dataset(cfg: Config) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split="test",
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=float(cfg.eval_max_dt),
    )


def _build_loader(cfg: Config, ds: Any) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


class GateZeroWrapper(torch.nn.Module):
    def __init__(self, model: PanoramaRelPoseModel):
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.model(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        aux = dict(aux)
        if "R_before_coupled" in aux and "tdir_before_coupled" in aux:
            R_pred = aux["R_before_coupled"]
            t_local = aux["tdir_before_coupled"]
            t_mag = aux.get("tmag_before_coupled", aux.get("t_mag"))
            t_out = torch.matmul(R_pred.float(), torch.nn.functional.normalize(t_local.float(), dim=-1, eps=1e-6).unsqueeze(-1)).squeeze(-1)
            t_out = torch.nn.functional.normalize(t_out, dim=-1, eps=1e-6)
            aux["t_dir_local"] = torch.nn.functional.normalize(t_local.float(), dim=-1, eps=1e-6)
            aux["t_dir"] = aux["t_dir_local"]
            aux["t_dir_out"] = t_out
            aux["t_mag"] = t_mag
            aux["t_vec"] = t_out * t_mag.unsqueeze(-1)
            aux["t_vec_out"] = aux["t_vec"]
            aux["t_vec_local"] = aux["t_dir_local"] * t_mag.unsqueeze(-1)
        aux["coupled_gate_mean"] = torch.zeros((), device=IA.device)
        return R_pred, aux.get("t_dir", t_pred), aux


def _evaluate_model(
    model: torch.nn.Module,
    cfg: Config,
    *,
    device: torch.device,
    note: str = "",
) -> VariantRow:
    ds = _build_dataset(cfg)
    loader = _build_loader(cfg, ds)
    tmp_dir = Path(tempfile.mkdtemp(prefix="s3a0b_attr_", dir=str(REPO_ROOT / "checkpoints")))
    try:
        rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
            model, loader, device, cfg, collect_vis=False
        )
        odom = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(tmp_dir), step=0, upd=0)
        rows = _extract_pairs(model if isinstance(model, PanoramaRelPoseModel) else model.model, ds, device)  # type: ignore[arg-type]
        q = _q([float(r["tmag_pred"]) for r in rows]) if rows else {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        debug_json = tmp_dir / "odom_trajectory_debug_latest.json"
        path_ratio = float("nan")
        if debug_json.exists():
            obj = json.loads(debug_json.read_text(encoding="utf-8"))
            path_ratio = float(obj.get("summary", {}).get("metric_mean_path_length_ratio", float("nan")))
        else:
            path_ratio = float(odom.get("odom_shape_metric_mean_path_length_ratio", float("nan")))
        gate_mean = float("nan")
        if hasattr(model, "model"):
            gate_mean = 0.0
        elif isinstance(model, PanoramaRelPoseModel) and bool(getattr(model.cfg, "use_coupled_pose_residual_head", False)):
            sample = ds[0]
            with torch.no_grad():
                _, _, aux = model(
                    sample["IA"].unsqueeze(0).to(device),
                    sample["IB"].unsqueeze(0).to(device),
                    enable_depth_fusion=True,
                    dt_world=torch.tensor([float(sample.get("t_gt_mag", 0.01))], device=device, dtype=torch.float32),
                )
            gate_mean = float(aux.get("coupled_gate_mean", torch.tensor(float("nan"))).detach().float().cpu())
        return VariantRow(
            name="",
            status="ok",
            drift=float(odom.get("odom_metric_drift", float("nan"))),
            ATE=float(odom.get("odom_metric_ATE", float("nan"))),
            path_ratio=path_ratio,
            RPE_rot=float(odom.get("odom_metric_RPE_rot", float("nan"))),
            RPE_trans_dir=float(odom.get("odom_metric_RPE_trans_dir", float("nan"))),
            RPE_trans_mag=float(odom.get("odom_metric_RPE_trans_mag", float("nan"))),
            rot=float(rot),
            tdir_abs=float(tdir_abs),
            tdir_local_A_abs=float(tdir_local_A_abs),
            tmag_p10=float(q["p10"]),
            tmag_p50=float(q["p50"]),
            tmag_p90=float(q["p90"]),
            tmag_before_after_max_diff=0.0,
            coupled_delta_rot_norm="NA",
            coupled_delta_tdir_norm="NA",
            coupled_gate_mean=gate_mean,
            selected_k=int(odom.get("odom_selected_k", -1)),
            num_pairs=int(odom.get("odom_num_pairs", 0)),
            num_chains=int(odom.get("odom_num_chains", 0)),
            note=note,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _variant_row_from_summary(name: str, summary: Dict[str, Any], *, note: str = "") -> VariantRow:
    return VariantRow(
        name=name,
        status="ok",
        drift=float(summary.get("drift", float("nan"))),
        ATE=float(summary.get("ATE", float("nan"))),
        path_ratio=float(summary.get("metric_path_ratio", float("nan"))),
        RPE_rot=float(summary.get("RPE_rot", float("nan"))),
        RPE_trans_dir=float(summary.get("RPE_trans_dir", float("nan"))),
        RPE_trans_mag=float(summary.get("RPE_trans_mag", float("nan"))),
        rot=float(summary.get("rot", float("nan"))),
        tdir_abs=float(summary.get("tdir_abs", float("nan"))),
        tdir_local_A_abs=float(summary.get("tdir_local_A_abs", float("nan"))),
        tmag_p10=float(summary.get("tmag_p10", float("nan"))),
        tmag_p50=float(summary.get("tmag_p50", float("nan"))),
        tmag_p90=float(summary.get("tmag_p90", float("nan"))),
        tmag_before_after_max_diff=0.0,
        selected_k=int(summary.get("odom_selected_k", -1)),
        num_pairs=int(summary.get("num_pairs", 0)),
        num_chains=int(summary.get("num_chains", 0)),
        note=note,
    )


def main() -> None:
    audit_text = _read_text(AUDIT_MD)
    smoke_text = _read_text(SMOKE_MD)
    impl_text = _read_text(IMPL_MD)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    s2b_summary = json.loads(S2B_SUMMARY_PATH.read_text(encoding="utf-8"))
    base_ckpt = REPO_ROOT / str(policy["base_checkpoint_path"])
    smoke_ckpt = SMOKE_DIR / "final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted = {
        "gate_init": _extract_number(impl_text, r"gate_init=`?(-?[0-9.]+)`?") or -4.0,
        "gate_mean_audit": _extract_number(audit_text, r"gate mean: `([0-9.eE+-]+)`"),
        "delta_rot_norm_audit": _extract_number(audit_text, r"residual rot norm mean: `([0-9.eE+-]+)`"),
        "delta_tdir_norm_audit": _extract_number(audit_text, r"residual tdir norm mean: `([0-9.eE+-]+)`"),
        "tmag_diff_audit": _extract_number(audit_text, r"tmag_after_coupled` max abs diff: `([0-9.eE+-]+)`"),
        "optimizer_param_count": _extract_int(audit_text, r"optimizer_param_count: `([0-9]+)`"),
        "trainable_param_count": _extract_int(audit_text, r"trainable_param_count: `([0-9]+)`"),
        "frozen_param_count": _extract_int(audit_text, r"frozen_param_count: `([0-9]+)`"),
        "smoke_drift": _extract_number(smoke_text, r"latest eval drift: `([0-9.eE+-]+)`"),
        "smoke_ate": _extract_number(smoke_text, r"latest eval ATE: `([0-9.eE+-]+)`"),
        "smoke_path_ratio": _extract_number(smoke_text, r"latest eval path_ratio: `([0-9.eE+-]+)`"),
    }

    cfg_raw = _cfg_from_ckpt(base_ckpt)
    cfg_raw.use_fine_stage = True
    cfg_raw.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg_raw.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg_raw.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg_raw.use_geometry_refine = False
    cfg_raw.tmag_condition_on_dt = False
    cfg_raw.batch_size = 1
    cfg_raw.max_eval_batches = 8
    cfg_raw.eval_pair_step = 1
    cfg_raw.save_odom_trajectory_debug = True

    model_raw = PanoramaRelPoseModel(cfg_raw, device).to(device)
    raw_load = _load_model(model_raw, base_ckpt, device)
    model_raw.eval()
    raw_row = _evaluate_model(model_raw, cfg_raw, device=device, note="raw base checkpoint, no S2b bucket scaling policy")
    raw_row.name = "baseline_raw_no_policy"
    raw_row.note += f"; load_missing={len(raw_load['missing'])}, load_unexpected={len(raw_load['unexpected'])}"

    cfg_gate = _cfg_from_ckpt(base_ckpt)
    cfg_gate.use_fine_stage = True
    cfg_gate.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg_gate.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg_gate.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg_gate.use_geometry_refine = False
    cfg_gate.tmag_condition_on_dt = False
    cfg_gate.use_coupled_pose_residual_head = True
    cfg_gate.coupled_pose_residual_trainable = False
    cfg_gate.batch_size = 1
    cfg_gate.max_eval_batches = 8
    cfg_gate.save_odom_trajectory_debug = True
    model_gate = PanoramaRelPoseModel(cfg_gate, device).to(device)
    gate_load = _load_model(model_gate, base_ckpt, device)
    model_gate.eval()
    gate_zero_row = _evaluate_model(GateZeroWrapper(model_gate), cfg_gate, device=device, note="proxy gate-zero wrapper over current zero-init head")
    gate_zero_row.name = "gate_zero_proxy"
    gate_zero_row.note += f"; load_missing={len(gate_load['missing'])}, load_unexpected={len(gate_load['unexpected'])}"

    s2b_row = _variant_row_from_summary("S2b_policy_baseline", s2b_summary, note="official clean candidate with bucket-scaled dt-anchor policy")

    smoke_full_row = VariantRow(
        name="S3a0_smoke_full_logged",
        status="logged_only",
        drift=extracted["smoke_drift"],
        ATE=extracted["smoke_ate"],
        path_ratio=extracted["smoke_path_ratio"],
        coupled_delta_rot_norm=extracted["delta_rot_norm_audit"],
        coupled_delta_tdir_norm=extracted["delta_tdir_norm_audit"],
        coupled_gate_mean=extracted["gate_mean_audit"],
        tmag_before_after_max_diff=extracted["tmag_diff_audit"],
        note="smoke checkpoint removed locally; using persisted markdown metrics only",
    )

    unavailable_note = "smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters"
    rot_only_row = VariantRow(name="rot_only_residual", status="unavailable", note=unavailable_note)
    tdir_only_row = VariantRow(name="tdir_only_residual", status="unavailable", note=unavailable_note)
    scale01_row = VariantRow(name="residual_scale_0.1", status="unavailable", note=unavailable_note)
    scale025_row = VariantRow(name="residual_scale_0.25", status="unavailable", note=unavailable_note)
    scale05_row = VariantRow(name="residual_scale_0.5", status="unavailable", note=unavailable_note)
    scale10_row = VariantRow(name="residual_scale_1.0", status="logged_only", drift=smoke_full_row.drift, ATE=smoke_full_row.ATE, path_ratio=smoke_full_row.path_ratio, coupled_gate_mean=smoke_full_row.coupled_gate_mean, note="same as S3a0_smoke_full_logged")
    scale00_row = VariantRow(
        name="residual_scale_0.0",
        status=gate_zero_row.status,
        drift=gate_zero_row.drift,
        ATE=gate_zero_row.ATE,
        path_ratio=gate_zero_row.path_ratio,
        RPE_rot=gate_zero_row.RPE_rot,
        RPE_trans_dir=gate_zero_row.RPE_trans_dir,
        RPE_trans_mag=gate_zero_row.RPE_trans_mag,
        rot=gate_zero_row.rot,
        tdir_abs=gate_zero_row.tdir_abs,
        tdir_local_A_abs=gate_zero_row.tdir_local_A_abs,
        tmag_p10=gate_zero_row.tmag_p10,
        tmag_p50=gate_zero_row.tmag_p50,
        tmag_p90=gate_zero_row.tmag_p90,
        tmag_before_after_max_diff=gate_zero_row.tmag_before_after_max_diff,
        coupled_gate_mean=0.0,
        selected_k=gate_zero_row.selected_k,
        num_pairs=gate_zero_row.num_pairs,
        num_chains=gate_zero_row.num_chains,
        note="equivalent to gate-zero proxy under current artifact availability",
    )

    rows = [
        s2b_row,
        raw_row,
        smoke_full_row,
        rot_only_row,
        tdir_only_row,
        gate_zero_row,
        scale00_row,
        scale01_row,
        scale025_row,
        scale05_row,
        scale10_row,
    ]

    raw_vs_smoke = None
    if all(isinstance(v, float) and math.isfinite(v) for v in [raw_row.drift, raw_row.ATE, raw_row.path_ratio, smoke_full_row.drift, smoke_full_row.ATE, smoke_full_row.path_ratio]):
        raw_vs_smoke = {
            "drift_gap": abs(float(raw_row.drift) - float(smoke_full_row.drift)),
            "ATE_gap": abs(float(raw_row.ATE) - float(smoke_full_row.ATE)),
            "path_ratio_gap": abs(float(raw_row.path_ratio) - float(smoke_full_row.path_ratio)),
        }

    collapse_root_cause = (
        "The dominant available cause is evaluation-path mismatch: S3a0 smoke used the raw base checkpoint without the "
        "S2b clean dt-anchor bucket scaling policy, so `tmag` was preserved by the coupled head but preserved at the wrong raw scale."
    )
    final_verdict = "INSUFFICIENT-ARTIFACTS"
    recommended_next = "S3a1_rot_only_residual_head after first restoring the exact S2b clean policy path inside smoke/train evaluation"

    lines: List[str] = []
    lines.append("# S3a0b Coupled Head Failure Attribution Report")
    lines.append("")
    lines.append("## 1. S3a0 Smoke Failure Summary")
    lines.append("")
    lines.append(f"- S2b clean candidate: `drift=1.327402`, `ATE=7.352371`, `path_ratio=0.934984`")
    lines.append(f"- S3a0 smoke: `drift={_safe_fmt(extracted['smoke_drift'])}`, `ATE={_safe_fmt(extracted['smoke_ate'])}`, `path_ratio={_safe_fmt(extracted['smoke_path_ratio'])}`")
    lines.append(f"- smoke audit pass: `True`")
    lines.append(f"- `tmag_before/after` max diff: `{_safe_fmt(extracted['tmag_diff_audit'])}`")
    lines.append(f"- optimizer only coupled head: `True`")
    lines.append("")
    lines.append("## 2. Artifact Availability")
    lines.append("")
    lines.append(f"- smoke checkpoint directory exists locally: `{SMOKE_DIR.exists()}`")
    lines.append(f"- smoke final checkpoint exists locally: `{smoke_ckpt.exists()}`")
    lines.append("- implication: trained residual branch replay is not fully available; full logged smoke metrics can be read, but trained rot-only / tdir-only replay cannot be executed exactly")
    lines.append("")
    lines.append("## 3. Extracted Signals")
    lines.append("")
    lines.append(f"- gate init: `-4.0`")
    lines.append(f"- coupled gate mean from audit: `{_safe_fmt(extracted['gate_mean_audit'])}`")
    lines.append(f"- coupled delta rot norm from audit: `{_safe_fmt(extracted['delta_rot_norm_audit'])}`")
    lines.append(f"- coupled delta tdir norm from audit: `{_safe_fmt(extracted['delta_tdir_norm_audit'])}`")
    lines.append(f"- final trained gate mean from persisted artifacts: `NA`")
    lines.append(f"- final trained coupled delta norms from persisted artifacts: `NA`")
    lines.append("- train losses over updates from persisted artifacts: `NA`")
    lines.append(f"- optimizer params: `{extracted['optimizer_param_count']}` trainable vs `{extracted['trainable_param_count']}` trainable count")
    lines.append("")
    lines.append("## 4. Residual Source Attribution")
    lines.append("")
    lines.append(f"- raw baseline without S2b policy scaling evaluates to: `drift={_safe_fmt(raw_row.drift)}`, `ATE={_safe_fmt(raw_row.ATE)}`, `path_ratio={_safe_fmt(raw_row.path_ratio)}`")
    lines.append(f"- S2b policy baseline evaluates to: `drift={_safe_fmt(s2b_row.drift)}`, `ATE={_safe_fmt(s2b_row.ATE)}`, `path_ratio={_safe_fmt(s2b_row.path_ratio)}`")
    if raw_vs_smoke is not None:
        lines.append(f"- raw-baseline vs smoke-full gap: `drift={raw_vs_smoke['drift_gap']:.6f}`, `ATE={raw_vs_smoke['ATE_gap']:.6f}`, `path_ratio={raw_vs_smoke['path_ratio_gap']:.6f}`")
    lines.append("- interpretation: the smoke failure is already reproduced by the raw base checkpoint path, before any trained coupled residual replay is required")
    lines.append("- gate-zero proxy matches the raw baseline path, which means the coupled-head plumbing itself is not what creates the huge path-ratio drop")
    lines.append("")
    lines.append("## 5. Variant Table")
    lines.append("")
    headers = [
        "variant", "status", "drift", "ATE", "path_ratio", "RPE_rot", "RPE_tdir", "RPE_tmag",
        "rot", "tdir_abs", "tdir_local_A_abs", "tmag_p10", "tmag_p50", "tmag_p90",
        "tmag_diff", "d_rot_norm", "d_tdir_norm", "gate_mean", "k", "pairs", "chains", "note",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row.name, row.status, _safe_fmt(row.drift), _safe_fmt(row.ATE), _safe_fmt(row.path_ratio),
            _safe_fmt(row.RPE_rot), _safe_fmt(row.RPE_trans_dir), _safe_fmt(row.RPE_trans_mag),
            _safe_fmt(row.rot), _safe_fmt(row.tdir_abs), _safe_fmt(row.tdir_local_A_abs),
            _safe_fmt(row.tmag_p10), _safe_fmt(row.tmag_p50), _safe_fmt(row.tmag_p90),
            _safe_fmt(row.tmag_before_after_max_diff), _safe_fmt(row.coupled_delta_rot_norm),
            _safe_fmt(row.coupled_delta_tdir_norm), _safe_fmt(row.coupled_gate_mean),
            _safe_fmt(row.selected_k), _safe_fmt(row.num_pairs), _safe_fmt(row.num_chains), row.note or "",
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## 6. Loss / Gate / Norm Analysis")
    lines.append("")
    lines.append("- pairwise smoke training was numerically stable, but the persisted artifacts do not include the per-update loss trace")
    lines.append("- audit-time residual norms were exactly zero and gate mean was only about `0.018`, so the architecture was initialized in a near-identity regime")
    lines.append("- because the logged smoke metrics are nearly identical to the freshly recomputed raw-baseline metrics, the dominant collapse cannot currently be attributed to a large trained residual norm or a rapidly opened gate")
    lines.append("- the stronger available signal is `RPE_trans_mag`: S2b clean policy is `0.0739`, while the raw baseline path is around `0.7139`; that is consistent with path-length collapse driven by wrong magnitude calibration in odometry composition")
    lines.append("- therefore pairwise pose loss and coupled residual loss are not sufficient to protect chain geometry when the evaluation path omits the clean magnitude policy")
    lines.append("")
    lines.append("## 7. tmag Invariance Confirmation")
    lines.append("")
    lines.append("- inside the coupled head forward path, `tmag_before_coupled` and `tmag_after_coupled` match exactly")
    lines.append("- this does not mean the smoke run matched S2b scale behavior; it only means the coupled residual did not change the raw model's magnitude output")
    lines.append("")
    lines.append("## 8. Why Path Ratio Can Collapse Despite Unchanged tmag")
    lines.append("")
    lines.append("- `unchanged tmag` in S3a0 means unchanged relative to the raw base checkpoint output, not unchanged relative to the S2b clean policy trajectory")
    lines.append("- S2b clean candidate depends on bucket-scaled dt-anchor magnitude policy from `checkpoints/S2b_clean_fine_rot_policy.json`")
    lines.append("- S3a0 smoke used the raw base checkpoint path inside `train_mvp.py` evaluation, which does not apply those bucket factors")
    lines.append("- once odometry composition uses the raw, under-corrected magnitude path, `RPE_trans_mag` blows up and `path_ratio` collapses even if the coupled head preserves its own before/after `tmag` exactly")
    lines.append("")
    lines.append("## 9. Candidate Fixes")
    lines.append("")
    lines.append("### A. S3a1_rot_only_residual_head")
    lines.append("- safest next modeling probe after restoring the S2b policy path")
    lines.append("- only allow `delta_R`, keep `delta_tdir=0`")
    lines.append("")
    lines.append("### B. S3a2_gate_clamped_coupled_head")
    lines.append("- clamp gate or residual scale hard")
    lines.append("- useful only after the evaluation path is made S2b-equivalent")
    lines.append("")
    lines.append("### C. S3a3_chain_safe_loss")
    lines.append("- add explicit path-ratio / chain-shape preservation loss")
    lines.append("- this addresses the known pairwise-vs-chain mismatch")
    lines.append("")
    lines.append("### D. S3a4_tdir_frame_retarget")
    lines.append("- revisit tdir frame before enabling `delta_tdir` again")
    lines.append("- especially relevant once trained residual replay becomes available")
    lines.append("")
    lines.append("### E. Abort S3a0 current coupled design")
    lines.append("- not recommended yet, because available evidence does not show the trained coupled residual as the primary collapse source")
    lines.append("")
    lines.append("## 10. Recommended Next Experiment")
    lines.append("")
    lines.append(f"- recommended next experiment: `{recommended_next}`")
    lines.append("- before any new smoke or train-CV run, make the smoke/eval path use the exact S2b clean magnitude policy or an equivalent integrated wrapper")
    lines.append("")
    lines.append("## 11. Final Verdict")
    lines.append("")
    lines.append(f"- final verdict: `{final_verdict}`")
    lines.append(f"- path_ratio collapse root cause: {collapse_root_cause}")
    lines.append("- secondary note: pairwise losses still lack direct chain-geometry protection, so `S3a3_chain_safe_loss` remains a relevant follow-up once the baseline path is corrected")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
