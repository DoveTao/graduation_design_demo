#!/usr/bin/env python3
"""JRT1c train-CV trajectory proxy gate for JRT1b refiner variants."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from jrt1_train_cv_joint_rtdir_refiner import _make_folds
from jrt1_train_joint_rtdir_refiner import JointRefinerMLP, exp_so3, geodesic_rad


EVAL_VARIANTS = [
    "A_s5_no_refiner_reference",
    "B_rot_only_residual",
    "C_tdir_residual_cosine",
    "D_joint_rtdir_residual",
    "E_joint_rtdir_tdir_weighted",
    "F_joint_rtdir_tdir_hardcase_weighted",
]
LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _resolve(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.6f}"


def _check_locked(config: Dict[str, Any], results: Dict[str, Any]) -> None:
    for source, obj in [("config", config), ("JRT1b results", results)]:
        metrics = obj.get("locked_s5_metrics", {})
        for key, expected in LOCKED.items():
            actual = float(metrics.get(key, float("nan")))
            if abs(actual - expected) > 1.0e-9:
                raise RuntimeError(f"{source} changed locked S5 {key}: expected {expected}, got {actual}")


def _load_cache(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    return {
        "features": torch.tensor(arr["features"], dtype=torch.float32),
        "R_pred": torch.tensor(arr["R_pred"], dtype=torch.float32),
        "tdir_pred": torch.tensor(arr["tdir_pred"], dtype=torch.float32),
        "log_tmag_pred": torch.tensor(arr["log_tmag_pred"], dtype=torch.float32),
        "tmag_pred": torch.tensor(arr["tmag_pred"], dtype=torch.float32),
        "R_gt": torch.tensor(arr["R_gt"], dtype=torch.float32),
        "tdir_gt": torch.tensor(arr["tdir_gt"], dtype=torch.float32),
        "tmag_gt": torch.tensor(arr["tmag_gt"], dtype=torch.float32),
        "dataset_index": np.asarray(arr["dataset_index"], dtype=np.int64),
        "dt": np.asarray(arr["dt"], dtype=np.float64),
        "k": np.asarray(arr["k"], dtype=np.int64),
        "sequence_id": np.asarray(arr["sequence_id"]).astype(str),
        "pair_id": np.asarray(arr["pair_id"]).astype(str),
        "metadata": json.loads(str(arr["metadata_json"].item())),
    }


def _pair_times(pair_id: str) -> Tuple[float, float]:
    m = re.search(r"::([0-9.]+)->([0-9.]+)::k=", pair_id)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def _build_chain_sets(cache: Dict[str, Any], folds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index_to_local = {int(v): i for i, v in enumerate(cache["dataset_index"].tolist())}
    out: List[Dict[str, Any]] = []
    for fold in folds:
        val_local = [index_to_local[int(i)] for i in fold["val_indices"] if int(i) in index_to_local]
        val_k1 = [i for i in val_local if int(cache["k"][i]) == 1]
        grouped: Dict[str, List[int]] = {}
        for i in val_k1:
            grouped.setdefault(str(cache["sequence_id"][i]), []).append(i)
        chains: List[List[int]] = []
        for seq, items in sorted(grouped.items()):
            items = sorted(items, key=lambda i: (_pair_times(str(cache["pair_id"][i]))[0], _pair_times(str(cache["pair_id"][i]))[1]))
            cur: List[int] = []
            prev_b = None
            for i in items:
                ts_a, ts_b = _pair_times(str(cache["pair_id"][i]))
                if not cur:
                    cur = [i]
                elif prev_b is not None and math.isfinite(ts_a) and abs(ts_a - prev_b) <= 1.0e-4:
                    cur.append(i)
                else:
                    chains.append(cur)
                    cur = [i]
                prev_b = ts_b
            if cur:
                chains.append(cur)
        out.append(
            {
                "fold_id": int(fold["fold_id"]),
                "val_pairs": val_local,
                "val_k1_pairs": val_k1,
                "chains": chains,
                "num_val_pairs": int(len(val_local)),
                "num_val_k1_pairs": int(len(val_k1)),
                "num_chains": int(len(chains)),
                "num_pairs_used": int(sum(len(c) for c in chains)),
                "chain_coverage": float(sum(len(c) for c in chains) / max(len(val_k1), 1)),
            }
        )
    return out


def _load_refiner(results: Dict[str, Any], fold_id: int, variant: str) -> JointRefinerMLP | None:
    if variant == "A_s5_no_refiner_reference":
        return None
    for row in results.get("train_logs", []):
        if int(row.get("fold_id", -1)) == int(fold_id) and row.get("variant") == variant:
            path = row.get("model_state_path")
            if not path:
                raise RuntimeError(f"Missing model_state_path for fold={fold_id} variant={variant}")
            ckpt = torch.load(str(_resolve(path)), map_location="cpu")
            model = JointRefinerMLP(int(ckpt["input_dim"]))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            return model
    raise RuntimeError(f"No JRT1b train log for fold={fold_id} variant={variant}")


def _refine(model: JointRefinerMLP | None, cache: Dict[str, Any], idx: torch.Tensor, variant: str):
    x = cache["features"][idx]
    if model is None:
        delta = torch.zeros((idx.numel(), 6), dtype=torch.float32)
    else:
        delta = model(x.float())
    delta_rot = delta[:, :3]
    delta_tdir = delta[:, 3:6]
    if variant == "B_rot_only_residual":
        delta_tdir = torch.zeros_like(delta_tdir)
    if variant == "C_tdir_residual_cosine":
        delta_rot = torch.zeros_like(delta_rot)
    if variant == "A_s5_no_refiner_reference":
        delta_rot = torch.zeros_like(delta_rot)
        delta_tdir = torch.zeros_like(delta_tdir)
    R_ref = torch.matmul(exp_so3(delta_rot), cache["R_pred"][idx].float())
    tdir_ref = F.normalize(cache["tdir_pred"][idx].float() + delta_tdir.float(), dim=-1, eps=1.0e-6)
    return R_ref, tdir_ref


def _compose_rel_pose(R_rel: np.ndarray, t_rel: np.ndarray, R_cur0: np.ndarray, t_cur0: np.ndarray):
    R_next0 = R_rel.astype(np.float64) @ R_cur0.astype(np.float64)
    t_next0 = R_rel.astype(np.float64) @ t_cur0.astype(np.float64) + t_rel.astype(np.float64)
    return R_next0, t_next0


def _camera_center(R_c0: np.ndarray, t_c0: np.ndarray) -> np.ndarray:
    return -(R_c0.astype(np.float64).T @ t_c0.astype(np.float64))


def _component_metrics(cache: Dict[str, Any], idx_list: Sequence[int], model: JointRefinerMLP | None, variant: str) -> Dict[str, float]:
    idx = torch.tensor(list(idx_list), dtype=torch.long)
    if idx.numel() == 0:
        raise RuntimeError("component metrics require non-empty pair set")
    with torch.no_grad():
        R_ref, tdir_ref = _refine(model, cache, idx, variant)
        rot = geodesic_rad(R_ref, cache["R_gt"][idx]) * (180.0 / math.pi)
        cos = (tdir_ref * F.normalize(cache["tdir_gt"][idx], dim=-1, eps=1.0e-6)).sum(dim=-1).clamp(-1.0, 1.0)
        tdir = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)) * (180.0 / math.pi)
        tmag = torch.abs(cache["log_tmag_pred"][idx] - torch.log(cache["tmag_gt"][idx].clamp_min(1.0e-12)))
    return {
        "rot_mean_deg": float(rot.mean().item()),
        "rot_p90_deg": float(torch.quantile(rot.float().cpu(), 0.9).item()),
        "tdir_mean_deg": float(tdir.mean().item()),
        "tdir_p90_deg": float(torch.quantile(tdir.float().cpu(), 0.9).item()),
        "tdir_mean_cosine": float(cos.mean().item()),
        "tmag_mean_log_error": float(tmag.mean().item()),
    }


def _trajectory_metrics(cache: Dict[str, Any], chains: Sequence[Sequence[int]], model: JointRefinerMLP | None, variant: str) -> Dict[str, float]:
    all_gt_pts: List[np.ndarray] = []
    all_pr_pts: List[np.ndarray] = []
    endpoint_errs: List[float] = []
    gt_path_sum = 0.0
    pr_path_sum = 0.0
    pairs_used = 0
    for chain in chains:
        if not chain:
            continue
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pr_c0 = np.eye(3, dtype=np.float64)
        t_pr_c0 = np.zeros(3, dtype=np.float64)
        gt_pts = [_camera_center(R_gt_c0, t_gt_c0)]
        pr_pts = [_camera_center(R_pr_c0, t_pr_c0)]
        idx_t = torch.tensor(list(chain), dtype=torch.long)
        with torch.no_grad():
            R_ref, tdir_ref = _refine(model, cache, idx_t, variant)
        for local_pos, i in enumerate(chain):
            R_gt = cache["R_gt"][i].numpy()
            t_gt = cache["tdir_gt"][i].numpy() * float(cache["tmag_gt"][i])
            R_pr = R_ref[local_pos].detach().float().numpy()
            t_pr = tdir_ref[local_pos].detach().float().numpy() * float(cache["tmag_pred"][i])
            R_gt_c0, t_gt_c0 = _compose_rel_pose(R_gt, t_gt, R_gt_c0, t_gt_c0)
            R_pr_c0, t_pr_c0 = _compose_rel_pose(R_pr, t_pr, R_pr_c0, t_pr_c0)
            gt_pts.append(_camera_center(R_gt_c0, t_gt_c0))
            pr_pts.append(_camera_center(R_pr_c0, t_pr_c0))
            pairs_used += 1
        gt_arr = np.asarray(gt_pts, dtype=np.float64)
        pr_arr = np.asarray(pr_pts, dtype=np.float64)
        all_gt_pts.extend(list(gt_arr))
        all_pr_pts.extend(list(pr_arr))
        endpoint_errs.append(float(np.linalg.norm(pr_arr[-1] - gt_arr[-1])))
        if len(gt_arr) >= 2:
            gt_path_sum += float(np.linalg.norm(np.diff(gt_arr, axis=0), axis=1).sum())
            pr_path_sum += float(np.linalg.norm(np.diff(pr_arr, axis=0), axis=1).sum())
    if pairs_used == 0:
        raise RuntimeError("trajectory proxy has zero k=1 pairs")
    gt_all = np.asarray(all_gt_pts, dtype=np.float64)
    pr_all = np.asarray(all_pr_pts, dtype=np.float64)
    ate = float(np.sqrt(np.mean(np.sum((pr_all - gt_all) ** 2, axis=1))))
    drift = float(np.mean(np.asarray(endpoint_errs, dtype=np.float64)))
    path_ratio = float(pr_path_sum / max(gt_path_sum, 1.0e-12))
    return {
        "ATE_proxy": ate,
        "drift_proxy": drift,
        "path_ratio_proxy": path_ratio,
        "num_pairs_used": int(pairs_used),
        "num_chains": int(len([c for c in chains if c])),
    }


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _summarize(per_fold: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    keys = [
        "ATE_proxy",
        "drift_proxy",
        "path_ratio_proxy",
        "rot_mean_deg",
        "rot_p90_deg",
        "tdir_mean_deg",
        "tdir_p90_deg",
        "tdir_mean_cosine",
        "tmag_mean_log_error",
        "num_chains",
        "num_pairs_used",
        "chain_coverage",
    ]
    for variant in EVAL_VARIANTS:
        rows = [r for r in per_fold if r["variant"] == variant and r["status"] == "ok"]
        out[variant] = {key: _mean_std([row[key] for row in rows]) for key in keys}
        out[variant]["num_folds_ok"] = len(rows)
    return out


def _add_deltas(per_fold: List[Dict[str, Any]]) -> None:
    by_fold: Dict[int, Dict[str, Any]] = {}
    for row in per_fold:
        if row["variant"] == "A_s5_no_refiner_reference":
            by_fold[int(row["fold_id"])] = row
    for row in per_fold:
        ref = by_fold[int(row["fold_id"])]
        row["delta_ATE_proxy"] = float(row["ATE_proxy"] - ref["ATE_proxy"])
        row["delta_drift_proxy"] = float(row["drift_proxy"] - ref["drift_proxy"])
        row["delta_path_ratio_proxy"] = float(row["path_ratio_proxy"] - ref["path_ratio_proxy"])
        row["delta_rot_mean"] = float(row["rot_mean_deg"] - ref["rot_mean_deg"])
        row["delta_tdir_mean"] = float(row["tdir_mean_deg"] - ref["tdir_mean_deg"])
        row["delta_tdir_cos"] = float(row["tdir_mean_cosine"] - ref["tdir_mean_cosine"])


def _gate(config: Dict[str, Any], mean_cv: Dict[str, Dict[str, Any]], per_fold: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, str | None]:
    gate_cfg = config["gate"]
    ref = mean_cv["A_s5_no_refiner_reference"]
    ref_tdir = float(ref["tdir_mean_deg"]["mean"])
    ref_cos = float(ref["tdir_mean_cosine"]["mean"])
    ref_rot = float(ref["rot_mean_deg"]["mean"])
    ref_ate = float(ref["ATE_proxy"]["mean"])
    ref_drift = float(ref["drift_proxy"]["mean"])
    decisions: Dict[str, Any] = {}
    passers: List[str] = []
    component_positive = False
    low_coverage = False
    for variant, row in mean_cv.items():
        if variant == "A_s5_no_refiner_reference":
            decisions[variant] = {"gate_status": "REFERENCE", "folds_passing_direction": 0}
            continue
        tdir_improve = ref_tdir - float(row["tdir_mean_deg"]["mean"])
        rot_worsen = float(row["rot_mean_deg"]["mean"]) - ref_rot
        cos_improve = float(row["tdir_mean_cosine"]["mean"]) > ref_cos
        ate_delta = float(row["ATE_proxy"]["mean"]) - ref_ate
        drift_delta = float(row["drift_proxy"]["mean"]) - ref_drift
        path_ratio = float(row["path_ratio_proxy"]["mean"])
        rows = [r for r in per_fold if r["variant"] == variant]
        fold_pass_count = 0
        for r in rows:
            fr = next(x for x in per_fold if x["fold_id"] == r["fold_id"] and x["variant"] == "A_s5_no_refiner_reference")
            if (
                (fr["tdir_mean_deg"] - r["tdir_mean_deg"]) >= float(gate_cfg["min_tdir_mean_improvement_deg"])
                and r["tdir_mean_cosine"] > fr["tdir_mean_cosine"]
                and (r["rot_mean_deg"] - fr["rot_mean_deg"]) <= float(gate_cfg["max_rot_mean_worsening_deg"])
                and float(gate_cfg["safe_path_ratio_min"]) <= r["path_ratio_proxy"] <= float(gate_cfg["safe_path_ratio_max"])
                and (r["ATE_proxy"] - fr["ATE_proxy"]) <= float(gate_cfg["max_ATE_worsening"])
                and (r["drift_proxy"] - fr["drift_proxy"]) <= 0.05
            ):
                fold_pass_count += 1
        no_collapse = int(row["num_folds_ok"]) == int(config["train_cv"]["num_folds"])
        coverage_ok = float(row["chain_coverage"]["mean"]) >= 0.50 and float(row["num_pairs_used"]["mean"]) >= 20.0
        if not coverage_ok:
            low_coverage = True
        component_ok = (
            tdir_improve >= float(gate_cfg["min_tdir_mean_improvement_deg"])
            and cos_improve
            and rot_worsen <= float(gate_cfg["max_rot_mean_worsening_deg"])
            and no_collapse
        )
        if component_ok:
            component_positive = True
        traj_ok = (
            component_ok
            and coverage_ok
            and float(gate_cfg["safe_path_ratio_min"]) <= path_ratio <= float(gate_cfg["safe_path_ratio_max"])
            and ate_delta <= float(gate_cfg["max_ATE_worsening"])
            and drift_delta <= 0.05
            and fold_pass_count >= 2
        )
        status = "PASS" if traj_ok else "FAIL"
        if traj_ok:
            passers.append(variant)
        decisions[variant] = {
            "tdir_improvement_deg": float(tdir_improve),
            "tdir_cos_improved": bool(cos_improve),
            "rot_worsening_deg": float(rot_worsen),
            "delta_ATE_proxy": float(ate_delta),
            "delta_drift_proxy": float(drift_delta),
            "path_ratio_proxy": float(path_ratio),
            "coverage_ok": bool(coverage_ok),
            "no_fold_collapse": bool(no_collapse),
            "folds_passing_direction": int(fold_pass_count),
            "gate_status": status,
        }
    if passers:
        best = min(passers, key=lambda v: float(mean_cv[v]["ATE_proxy"]["mean"]))
        return decisions, "JRT1C-TRAIN-CV-TRAJECTORY-GATE-PASS", best
    if low_coverage:
        return decisions, "TRAJECTORY_PROXY_LOW_COVERAGE", None
    if component_positive:
        return decisions, "COMPONENT_SIGNAL_NO_TRAJECTORY_GAIN", None
    return decisions, "NO_STABLE_JRT1_TRAJECTORY_GAIN", None


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    per_fold_table = [
        "| fold | variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean | tdir_mean | tdir_cos | num_chains | num_pairs_used |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["per_fold_results"]:
        per_fold_table.append(
            f"| {row['fold_id']} | {row['variant']} | {_fmt(row['ATE_proxy'])} | {_fmt(row['drift_proxy'])} | {_fmt(row['path_ratio_proxy'])} | {_fmt(row['rot_mean_deg'])} | {_fmt(row['tdir_mean_deg'])} | {_fmt(row['tdir_mean_cosine'])} | {row['num_chains']} | {row['num_pairs_used']} |"
        )
    mean_table = [
        "| variant | mean_ATE_proxy | mean_drift_proxy | mean_path_ratio_proxy | mean_rot | mean_tdir | mean_tdir_cos | gate_status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant, row in payload["mean_cv_results"].items():
        status = payload["gate_decision"].get(variant, {}).get("gate_status", "")
        mean_table.append(
            f"| {variant} | {_fmt(row['ATE_proxy']['mean'])} | {_fmt(row['drift_proxy']['mean'])} | {_fmt(row['path_ratio_proxy']['mean'])} | {_fmt(row['rot_mean_deg']['mean'])} | {_fmt(row['tdir_mean_deg']['mean'])} | {_fmt(row['tdir_mean_cosine']['mean'])} | {status} |"
        )
    gate_lines = []
    for variant, row in payload["gate_decision"].items():
        gate_lines.append(f"- `{variant}`: `{row.get('gate_status')}`")
    selected = payload.get("selected_candidate_for_next_stage")
    lines = [
        "# JRT1c Train-CV Trajectory Proxy Gate",
        "",
        "## Scope",
        "",
        "Train-CV trajectory proxy only. No final test is run. S5 remains the final clean candidate.",
        "",
        "## JRT1b Recap",
        "",
        "JRT1b produced component improvements: C was the strongest translation-direction-only variant, while E/F were balanced joint candidates. JRT1b left trajectory gate unavailable.",
        "",
        "## Trajectory Proxy Construction",
        "",
        "Option 2 was used. The chain source is the JRT1b train-split cache and reproduced train-CV fold assignment. The proxy filters validation pairs to `k=1`, groups by sequence, orders by timestamp, and composes local odometry chains. Prediction translation uses `tmag_s5 * tdir_refined`; S5 predicted magnitude is preserved and GT magnitude is not used for prediction.",
        "",
        f"- chain source: `{payload['trajectory_proxy_construction']['chain_source']}`",
        f"- k filtering: `{payload['trajectory_proxy_construction']['k_filtering']}`",
        f"- tmag policy: `{payload['trajectory_proxy_construction']['tmag_policy']}`",
        f"- no GT tmag used for prediction: `{payload['trajectory_proxy_construction']['no_gt_tmag_used_for_prediction']}`",
        "",
        "## Per-Fold Results",
        "",
        *per_fold_table,
        "",
        "## Mean CV Results",
        "",
        *mean_table,
        "",
        "## Gate Decision",
        "",
        *gate_lines,
        f"- selected candidate for next stage: `{selected}`",
        "",
        "## Interpretation",
        "",
        "The report compares every evaluated variant against A on the same fold and chain set. Component gains are considered only diagnostic unless the trajectory proxy gate also passes. If C improves tdir but worsens or fails trajectory proxy, it is not a next-stage candidate. E/F must keep balanced rotation behavior and satisfy ATE, drift, and path-ratio proxy criteria.",
        "",
        "## Final Classification",
        "",
        f"`{payload['final_classification']}`",
        "",
        "## Next Step",
        "",
        "If the gate passes, prepare a separate JRT1d final-test evaluation plan. Otherwise stop JRT1 as a negative or diagnostic result.",
        "",
        "## Caveats",
        "",
        "- train-CV only",
        "- no final candidate selected",
        "- no final test",
        "- S5 locked metrics unchanged",
        "- proxy trajectory is not final official test",
        "- small sequence protocol",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate JRT1c train-CV trajectory proxy gate.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--jrt1b-results", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    _run_guard()
    config = _read_json(_resolve(args.config))
    jrt1b = _read_json(_resolve(args.jrt1b_results))
    _check_locked(config, jrt1b)
    cache_path = REPO_ROOT / "outputs" / "jrt1" / "train_cv" / "JRT1b_train_cv_dataset_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"JRT1b cache not found: {cache_path}")
    cache = _load_cache(cache_path)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cv = config["train_cv"]
    folds = _make_folds(
        int(jrt1b["dataset_summary"]["num_available_train_pairs"]),
        int(cv["num_folds"]),
        int(cv["seed"]),
        int(cv["max_train_pairs_per_fold"]),
        int(cv["max_val_pairs_per_fold"]),
    )
    chain_sets = _build_chain_sets(cache, folds)
    per_fold: List[Dict[str, Any]] = []
    for fold in chain_sets:
        pair_set = [i for chain in fold["chains"] for i in chain]
        if not pair_set:
            raise RuntimeError(f"fold {fold['fold_id']} has no k=1 validation pairs for trajectory proxy")
        for variant in EVAL_VARIANTS:
            model = _load_refiner(jrt1b, int(fold["fold_id"]), variant)
            comp = _component_metrics(cache, pair_set, model, variant)
            traj = _trajectory_metrics(cache, fold["chains"], model, variant)
            row = {
                "fold_id": int(fold["fold_id"]),
                "variant": variant,
                "status": "ok",
                "chain_coverage": float(fold["chain_coverage"]),
                **traj,
                **comp,
            }
            per_fold.append(row)
    _add_deltas(per_fold)
    mean_cv = _summarize(per_fold)
    decisions, classification, selected = _gate(config, mean_cv, per_fold)
    payload = {
        "experiment_name": "JRT1c_train_cv_trajectory_proxy_gate",
        "base_candidate": config["base_candidate"],
        "locked_s5_metrics": config["locked_s5_metrics"],
        "variants": EVAL_VARIANTS,
        "trajectory_proxy_construction": {
            "option": "Option 2 deterministic validation odometry chains",
            "chain_source": "JRT1b train-CV validation folds from train split cache",
            "k_filtering": "k=1 only",
            "sequence_grouping": "sequence_id",
            "timestamp_ordering": "pair_id tsA then tsB",
            "tmag_policy": "S5 predicted tmag reused for prediction",
            "no_gt_tmag_used_for_prediction": True,
            "same_chain_set_as_reference": True,
        },
        "fold_chain_summary": [
            {
                "fold_id": f["fold_id"],
                "num_val_pairs": f["num_val_pairs"],
                "num_val_k1_pairs": f["num_val_k1_pairs"],
                "num_chains": f["num_chains"],
                "num_pairs_used": f["num_pairs_used"],
                "chain_coverage": f["chain_coverage"],
            }
            for f in chain_sets
        ],
        "per_fold_results": per_fold,
        "mean_cv_results": mean_cv,
        "gate_decision": decisions,
        "selected_candidate_for_next_stage": selected,
        "final_classification": classification,
        "no_final_test": True,
        "caveats": [
            "train-CV only",
            "no final candidate selected",
            "no final test",
            "S5 locked metrics unchanged",
            "proxy trajectory is not final official test",
            "small sequence protocol",
        ],
    }
    out_json = _resolve(args.output_json)
    out_md = _resolve(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(out_md, payload)
    (output_root / "JRT1c_train_cv_trajectory_proxy_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
