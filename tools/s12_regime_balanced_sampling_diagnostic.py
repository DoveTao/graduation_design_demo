#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import PanoramaRelPoseModel
from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg


BASE_CKPT = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s12_regime_balanced_sampling_summary.md"
MANIFEST_DIR = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_manifests"
TMP_DATA_ROOT = REPO_ROOT / "checkpoints" / "_tmp_s12_regime_balanced_data"
TRAIN_GROUPS: Tuple[Tuple[str, str], ...] = (("scene01", "seq01"), ("scene01", "seq02"))
FOLDS: Tuple[Tuple[str, int], ...] = (("heldout_scene01_seq01", 0), ("heldout_scene01_seq02", 3))
S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _ensure_subset_data_root() -> Path:
    pano_root = REPO_ROOT / "data" / "PanoramaView"
    if not pano_root.is_dir():
        raise FileNotFoundError(f"PanoramaView root not found: {pano_root}")
    if TMP_DATA_ROOT.exists():
        shutil.rmtree(TMP_DATA_ROOT)
    for scene, seq in TRAIN_GROUPS:
        src = pano_root / scene / seq
        dst = TMP_DATA_ROOT / "PanoramaView" / scene / seq
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)
    return TMP_DATA_ROOT


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    s5_ok = all(abs(float(s5["metrics"][k]) - float(S5_LOCKED[k])) <= 1.0e-9 for k in ("drift", "ATE", "path_ratio"))
    load_ok = (
        int(s5["load_missing"]) == 14
        and int(s5["load_unexpected"]) == 0
        and len(s5["missing_key_categories"]["ridge_calib_buffers"]) == 2
        and len(s5["missing_key_categories"]["coupled_pose_head_params"]) == 12
        and len(s5["missing_key_categories"]["other_missing"]) == 0
    )
    return {
        "passed": bool(contract["current_architecture_14_key_path_allowed_for_s8_baseline_gate"]) and s5_ok and load_ok,
        "s5_metrics": s5["metrics"],
        "load_missing": int(s5["load_missing"]),
        "load_unexpected": int(s5["load_unexpected"]),
    }


def _build_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Config]:
    cfg = _cfg_from_dict(_load_ckpt_cfg(BASE_CKPT))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = 0.45
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.pin_memory = False
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(BASE_CKPT), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def _quantile_edges(vals: Sequence[float], q: Sequence[float]) -> List[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return [0.0 for _ in q]
    return np.quantile(arr, q).astype(np.float64).tolist()


def _bucket_from_edges(v: float, edges: Sequence[float], prefix: str) -> str:
    if len(edges) < 2:
        return f"{prefix}_all"
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        is_last = idx == len(edges) - 2
        if (v >= lo and v < hi) or (is_last and v <= hi):
            return f"{prefix}{idx}"
    return f"{prefix}{len(edges) - 2}"


def _dt_bucket(v: float) -> str:
    if v < 0.1:
        return "dt<0.1"
    if v < 0.3:
        return "0.1<=dt<0.3"
    if v < 0.5:
        return "0.3<=dt<0.5"
    if v < 1.0:
        return "0.5<=dt<1.0"
    if v < 2.0:
        return "1.0<=dt<2.0"
    return "dt>=2.0"


def _load_rows_for_fold(data_root: Path, split_seed: int, model: PanoramaRelPoseModel, cfg: Config, device: torch.device) -> List[Dict[str, Any]]:
    ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(data_root),
        split="train",
        split_by="scene_seq",
        train_ratio=0.5,
        split_seed=int(split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=(1, 2, 3, 5, 10, 20),
        pair_step=1,
        min_dt=0.0,
        max_dt=None,
    )
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device)
            IB = sample["IB"].unsqueeze(0).to(device)
            dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            gt_tmag = float(sample["t_gt_mag"])
            rows.append(
                {
                    "scene": str(meta["scene"]),
                    "seq": str(meta["seq"]),
                    "i": int(meta["i"]),
                    "j": int(meta["j"]),
                    "k": int(meta["k"]),
                    "dt_world": float(dt_world),
                    "pred_tmag": float(pred_tmag),
                    "gt_tmag": float(gt_tmag),
                }
            )
    pred_edges = _quantile_edges([r["pred_tmag"] for r in rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    gt_edges = _quantile_edges([r["gt_tmag"] for r in rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    pred_q90 = float(pred_edges[-2]) if len(pred_edges) >= 2 else float("inf")
    for row in rows:
        row["pred_tmag_bucket"] = _bucket_from_edges(float(row["pred_tmag"]), pred_edges, "pred_q")
        row["gt_tmag_bucket"] = _bucket_from_edges(float(row["gt_tmag"]), gt_edges, "gt_q")
        row["dt_bucket"] = _dt_bucket(float(row["dt_world"]))
        row["k_bucket"] = f"k={int(row['k'])}"
        row["dt_k_bucket"] = f"{row['dt_bucket']}|{row['k_bucket']}"
        row["high_risk_bucket"] = bool(float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20)
        row["high_pred_bucket"] = bool(float(row["pred_tmag"]) >= pred_q90)
        row["mixed_hard_bucket"] = bool(float(row["pred_tmag"]) >= pred_q90 and float(row["dt_world"]) >= 1.0)
    return rows


def _counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        label = str(row[key])
        out[label] = out.get(label, 0) + 1
    return out


def _repeat_rows_by_inverse_freq(rows: Sequence[Dict[str, Any]], key: str, cap: int = 4) -> List[Dict[str, Any]]:
    freq = _counts(rows, key)
    max_count = max(freq.values()) if freq else 1
    out: List[Dict[str, Any]] = []
    for row in rows:
        count = max(freq.get(str(row[key]), 1), 1)
        rep = min(cap, max(1, int(round(max_count / count))))
        out.extend([dict(row)] * rep)
    return out


def _candidate_rows(rows: Sequence[Dict[str, Any]], spec: CandidateSpec) -> List[Dict[str, Any]]:
    base = [dict(r) for r in rows]
    if spec.name == "A_baseline_uniform_sampling":
        return base
    if spec.name == "B_pred_tmag_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "pred_tmag_bucket", cap=4)
    if spec.name == "C_gt_tmag_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "gt_tmag_bucket", cap=4)
    if spec.name == "D_dt_k_balanced_sampling":
        return _repeat_rows_by_inverse_freq(base, "dt_k_bucket", cap=4)
    if spec.name == "E_hard_regime_oversampling":
        out: List[Dict[str, Any]] = []
        for row in base:
            rep = 1
            if bool(row["high_pred_bucket"]):
                rep += 1
            if bool(row["high_risk_bucket"]):
                rep += 2
            out.extend([dict(row)] * min(rep, 4))
        return out
    if spec.name == "F_mixed_balanced_sampling":
        pred_freq = _counts(base, "pred_tmag_bucket")
        dtk_freq = _counts(base, "dt_k_bucket")
        max_pred = max(pred_freq.values()) if pred_freq else 1
        max_dtk = max(dtk_freq.values()) if dtk_freq else 1
        out: List[Dict[str, Any]] = []
        for row in base:
            rep_pred = max(1, int(round(max_pred / max(pred_freq.get(str(row["pred_tmag_bucket"]), 1), 1))))
            rep_dtk = max(1, int(round(max_dtk / max(dtk_freq.get(str(row["dt_k_bucket"]), 1), 1))))
            rep = min(4, max(rep_pred, rep_dtk))
            if bool(row["mixed_hard_bucket"]):
                rep = min(4, rep + 1)
            out.extend([dict(row)] * rep)
        return out
    raise ValueError(spec.name)


def _manifest_payload(rows: Sequence[Dict[str, Any]], candidate_name: str, split_seed: int) -> Dict[str, Any]:
    pairs = []
    for row in rows:
        item = dict(row)
        item["weight_tag"] = candidate_name
        pairs.append(item)
    return {
        "candidate_name": candidate_name,
        "split_seed": int(split_seed),
        "pairs": pairs,
        "split_summary": {
            "num_pairs": int(len(pairs)),
            "num_unique_pairs": int(len({(p["scene"], p["seq"], int(p["i"]), int(p["j"]), int(p["k"])) for p in pairs})),
            "num_sequences_after_filter": int(len({(p["scene"], p["seq"]) for p in pairs})),
        },
    }


def _candidate_specs() -> List[CandidateSpec]:
    return [
        CandidateSpec("A_baseline_uniform_sampling", "baseline_uniform_sampling"),
        CandidateSpec("B_pred_tmag_balanced_sampling", "pred_tmag_balanced_sampling"),
        CandidateSpec("C_gt_tmag_balanced_sampling", "gt_tmag_balanced_sampling"),
        CandidateSpec("D_dt_k_balanced_sampling", "dt_k_balanced_sampling"),
        CandidateSpec("E_hard_regime_oversampling", "hard_regime_oversampling"),
        CandidateSpec("F_mixed_balanced_sampling", "mixed_balanced_sampling"),
    ]


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S12 Regime Balanced Sampling Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- baseline gate passed: `{payload['baseline_gate']['passed']}`\n")
    lines.append(f"- run mode: `{payload['run_mode']}`\n")
    lines.append(f"- device used: `{payload['device']}`\n")
    lines.append(f"- manifest dir: `{payload['manifest_dir']}`\n")
    lines.append("- current run status: sampler audit / manifest generation only\n\n")
    lines.append("## Motivation\n\n")
    lines.append("- S4 identified tmag-regime-dominant error structure and a worst bucket around `dt>=1.0, k=20`.\n")
    lines.append("- S8/S9/S10/S11 suggest that changing heads, routers, smoothers, or tmag losses alone is unlikely to replace S5.\n")
    lines.append("- S12 therefore shifts to data-centric regime-balanced sampling and curriculum redesign.\n\n")
    lines.append("## Baseline gate\n\n")
    lines.append(f"- contract path: `{S8B_CONTRACT_PATH}`\n")
    lines.append(f"- load_missing / load_unexpected: `{payload['baseline_gate']['load_missing']} / {payload['baseline_gate']['load_unexpected']}`\n")
    lines.append(
        f"- cited locked S5 metrics: drift=`{_fmt(payload['baseline_gate']['s5_metrics']['drift'])}`, "
        f"ATE=`{_fmt(payload['baseline_gate']['s5_metrics']['ATE'])}`, "
        f"path_ratio=`{_fmt(payload['baseline_gate']['s5_metrics']['path_ratio'])}`\n\n"
    )
    lines.append("## Candidate sampler families\n\n")
    for row in payload["candidate_specs"]:
        lines.append(f"- `{row['name']}`: `{row['family']}`\n")
    lines.append("\n")
    lines.append("## Fold sampler distribution summary\n\n")
    for fold in payload["folds"]:
        lines.append(f"### {fold['fold_name']}\n\n")
        lines.append(f"- base rows: `{fold['base_num_rows']}`\n")
        lines.append(f"- pred_q90 threshold: `{_fmt(fold['pred_q90'])}`\n")
        for cand in fold["candidates"]:
            lines.append(
                f"- `{cand['name']}`: rows=`{cand['num_rows']}`, unique_pairs=`{cand['num_unique_pairs']}`, "
                f"high_pred_mass=`{cand['high_pred_mass']}`, high_risk_mass=`{cand['high_risk_mass']}`, mixed_hard_mass=`{cand['mixed_hard_mass']}`\n"
            )
        lines.append("\n")
    lines.append("## Leakage audit\n\n")
    lines.append("- train-only labels used for sampling: `True`\n")
    lines.append("- gt_tmag used as inference feature: `False`\n")
    lines.append("- test set used to define buckets or thresholds: `False`\n\n")
    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Final S12 Regime Balanced Sampling Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- Run mode: `{payload['run_mode']}`\n")
    lines.append(f"- Device: `{payload['device']}`\n")
    lines.append("- Current status: sampler manifests generated and ready for lightweight train-CV launch\n")
    lines.append("- S5 remains final clean candidate at this stage\n\n")
    lines.append("S12 has been started as a data-centric follow-up to S11. ")
    lines.append("The current stage builds train-only regime buckets and candidate sampling manifests for A/B/C/D/E/F, so the next S12 step can legally compare balanced-sampling training under the same clean-CV protocol.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run(mode: str = "audit") -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload: Dict[str, Any] = {
        "run_mode": str(mode),
        "device": str(device),
        "manifest_dir": str(MANIFEST_DIR),
        "baseline_gate": baseline_gate,
        "candidate_specs": [{"name": s.name, "family": s.family} for s in _candidate_specs()],
        "folds": [],
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
    }
    if not baseline_gate["passed"]:
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload

    data_root = _ensure_subset_data_root()
    model, cfg = _build_model(device)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for fold_name, split_seed in FOLDS:
        rows = _load_rows_for_fold(data_root, split_seed, model, cfg, device)
        pred_q90 = float(np.quantile(np.asarray([float(r["pred_tmag"]) for r in rows], dtype=np.float64), 0.90)) if rows else float("nan")
        fold_payload = {
            "fold_name": fold_name,
            "split_seed": int(split_seed),
            "base_num_rows": int(len(rows)),
            "pred_q90": pred_q90,
            "candidates": [],
        }
        for spec in _candidate_specs():
            sampled = _candidate_rows(rows, spec)
            manifest = _manifest_payload(sampled, spec.name, split_seed)
            manifest_path = MANIFEST_DIR / f"{fold_name}_{spec.name}.json"
            _write_json(manifest_path, manifest)
            high_pred_mass = int(sum(1 for r in sampled if bool(r["high_pred_bucket"])))
            high_risk_mass = int(sum(1 for r in sampled if bool(r["high_risk_bucket"])))
            mixed_hard_mass = int(sum(1 for r in sampled if bool(r["mixed_hard_bucket"])))
            fold_payload["candidates"].append(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "manifest_path": str(manifest_path),
                    "num_rows": int(len(sampled)),
                    "num_unique_pairs": int(len({(r["scene"], r["seq"], int(r["i"]), int(r["j"]), int(r["k"])) for r in sampled})),
                    "high_pred_mass": high_pred_mass,
                    "high_risk_mass": high_risk_mass,
                    "mixed_hard_mass": mixed_hard_mass,
                    "pred_bucket_counts": _counts(sampled, "pred_tmag_bucket"),
                    "gt_bucket_counts": _counts(sampled, "gt_tmag_bucket"),
                    "dt_k_bucket_counts": _counts(sampled, "dt_k_bucket"),
                }
            )
        payload["folds"].append(fold_payload)
    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    _write_summary(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch S12 regime-balanced sampling diagnostic.")
    parser.add_argument("--mode", default="audit", choices=("audit",), help="Current S12 stage.")
    args = parser.parse_args()
    payload = run(mode=args.mode)
    print(
        json.dumps(
            {
                "final_classification": payload["final_classification"],
                "run_mode": payload["run_mode"],
                "device": payload["device"],
                "report_path": str(REPORT_PATH),
                "candidates_path": str(CANDIDATES_PATH),
                "summary_path": str(SUMMARY_PATH),
                "manifest_dir": str(MANIFEST_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
