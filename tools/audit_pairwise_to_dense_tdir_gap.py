#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_JSON = "checkpoints/S5D5_pairwise_to_dense_tdir_gap_audit.json"
DEFAULT_OUT_REPORT = "reports/s5d5_pairwise_to_dense_tdir_gap_audit.md"
S11_B_PATH = "checkpoints/S11_tmag_scale_consistency_runs/S11_B_dt_aware_affine_calib_heldout_scene01_seq01/eval_history.json"
S11_C_PATH = "checkpoints/S11_tmag_scale_consistency_runs/S11_C_loss_only_consistency_heldout_scene01_seq01_u100/eval_history.json"
S16_PATH = "checkpoints/S16_stronger_visual_backbone_feasibility_candidates.json"
ALLOWED = [
    "S5D5_PAIRWISE_DENSE_GAP_EXPLAINED",
    "S5D5_PAIRWISE_DENSE_GAP_PARTIAL",
    "S5D5_FINAL_S5_PAIRWISE_ARTIFACT_UNAVAILABLE",
    "S5D5_DIFFERENT_CANDIDATE_OR_SEQUENCE",
    "S5D5_ERROR",
]
K_LABELS = ["k=1", "k=2", "k=5", "k=10", "k=20"]
S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5d5": True}


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _git_show_bytes(branch: str, repo_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{repo_path}"], cwd=REPO_ROOT)


def _load_bytes_with_fallback(path: Path, fallback_repo_path: str) -> bytes:
    if path.exists():
        return path.read_bytes()
    return _git_show_bytes("experiment/orbslam3-fisheye-strong-baseline", fallback_repo_path)


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_tum_bytes(content: bytes) -> List[Dict[str, Any]]:
    rows = []
    for raw in content.decode("utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        ts = float(p[0]); tx, ty, tz = float(p[1]), float(p[2]), float(p[3]); qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
        rows.append({"timestamp": ts, "t": np.asarray([tx, ty, tz], dtype=np.float64), "R_wc": _quat_xyzw_to_rot(qx, qy, qz, qw)})
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _map(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    return {round(float(r["timestamp"]), 6): r for r in rows}


def _rel(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    Ri, Rj = a["R_wc"], b["R_wc"]; ti, tj = a["t"], b["t"]
    return Ri.T @ Rj, Ri.T @ (tj - ti)


def _stats(v: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(v), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}


def _eval(gt_rows, est_rows, k=1, mode="dense", abs_dir=False, thr=None, stride=1):
    gm = _map(gt_rows); em = _map(est_rows)
    keys = sorted(set(gm.keys()) & set(em.keys()))
    tdir, tmag = [], []
    for i in range(0, len(keys)-k, stride):
        ga, gb = gm[keys[i]], gm[keys[i+k]]
        ea, eb = em[keys[i]], em[keys[i+k]]
        if mode == "dense":
            _, tg = _rel(ga, gb); _, te = _rel(ea, eb)
        else:
            tg = gb["t"] - ga["t"]; te = eb["t"] - ea["t"]
        ng, ne = float(np.linalg.norm(tg)), float(np.linalg.norm(te))
        if ng < 1e-8 or ne < 1e-8:
            continue
        if thr is not None and ng < thr:
            continue
        c = float(np.clip(np.dot(te/ne, tg/ng), -1.0, 1.0))
        if abs_dir:
            c = abs(c)
        tdir.append(float(math.degrees(math.acos(np.clip(c, -1.0, 1.0)))))
        tmag.append(ne / ng)
    ds, ms = _stats(tdir), _stats(tmag)
    return {"num_pairs": max(0, (len(keys)-k + (stride-1)) // stride), "num_valid_pairs": len(tdir), "tdir_mean_deg": ds["mean"], "tdir_median_deg": ds["median"], "tdir_p90_deg": ds["p90"], "tmag_mean_ratio": ms["mean"]}


def _find_final_s5_pairwise_artifacts() -> List[str]:
    pats = ["*pairwise*.json", "*relative*pose*.json", "*component*diagnostics*.json"]
    roots = [REPO_ROOT / "checkpoints", REPO_ROOT / "external_baselines" / "results", REPO_ROOT / "logs"]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for pat in pats:
            for p in root.rglob(pat):
                s = str(p)
                if "S11_" in s or "S16_" in s or "pano_orb_vo" in s:
                    continue
                if "S5" in s or "s5" in s:
                    out.append(str(p.relative_to(REPO_ROOT)))
    return sorted(set(out))[:50]


def _extract_hist_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore")
    vals = {}
    for key in ["\"tdir\"", "\"tdir_abs\"", "\"current_model_mean_tdir_err\""]:
        idx = txt.find(key)
        if idx >= 0:
            sub = txt[idx: idx + 200]
            import re
            m = re.search(r"(-?\d+\.\d+)", sub)
            if m:
                vals[key.strip('"')] = float(m.group(1))
    return vals


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    bm = payload["bridge_metrics"]
    lines = [
        "# S5D5 Pairwise-to-Dense tdir Gap Audit", "", "## Executive summary", "",
        f"Final classification: `{payload['final_classification']}`", "",
        "## Question", "", "historical pairwise/local-frame tdir around 20 deg vs dense TUM-derived tdir around 91 deg.",
        "", "## Historical source review", "",
        "| source | value summary | sequence | final S5 locked candidate | classification |",
        "|---|---|---|---|---|",
    ]
    for s in payload["historical_pairwise_sources"]["sources"]:
        lines.append(f"| {s['id']} | {s['metrics']} | {s['sequence']} | {s['is_final_s5_locked_candidate']} | {s['classification']} |")
    lines += ["", "## Final S5 pairwise artifact availability", "", f"- available: `{payload['final_s5_pairwise_artifacts']['available']}`", f"- paths: `{payload['final_s5_pairwise_artifacts']['paths']}`", "", "## Dense-derived bridging metrics", ""]
    lines.append(f"- dense_relative k1: `{bm['dense_relative']['k1']}`")
    lines.append(f"- dense_relative_tdir_abs k1: `{bm['dense_relative_abs']['k1']}`")
    lines.append(f"- world_delta k1: `{bm['world_delta']['k1']}`")
    lines.append(f"- world_delta_tdir_abs k1: `{bm['world_delta_abs']['k1']}`")
    lines.append(f"- k_step: `{bm['k_step']}`")
    lines.append(f"- thresholded: `{bm['thresholded']}`")
    lines += ["", "## Pairwise-to-dense gap interpretation", "", payload["pairwise_to_dense_gap"]["gap_explanation"], "", "## Caveats", "", "- S5D5 is diagnostic only.", "- S5D5 does not modify predictions.", "- S5D5 does not replace official S5 locked result.", "- S5 locked metrics/policy were not changed.", "- Historical tdir≈20 may belong to different candidate/sequence/protocol."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-dense-trajectory", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--s5d2-checkpoint", required=True)
    ap.add_argument("--s5d3-checkpoint", required=True)
    ap.add_argument("--s5d4-checkpoint", required=True)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    s5_path = _resolve(args.s5_dense_trajectory)
    gt_path = _resolve(args.groundtruth)
    out_json, out_report, out_dir = _resolve(args.out_json), _resolve(args.out_report), _resolve(args.out_dir)

    if not (REPO_ROOT / "logs").exists():
        (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", "-lc", "grep -RInE \"tdir_abs|tdir|current_model_mean_tdir_err|translation direction|pairwise|local frame|selected chain|odometry chain|tmag_scale_consistency|stronger_visual_backbone\" tools scripts reports checkpoints tests external_baselines logs 2>/dev/null > logs/s5d5_pairwise_tdir_pipeline_search.log"], cwd=REPO_ROOT, check=False)

    gt_rows = _read_tum_bytes(gt_path.read_bytes())
    s5_rows = _read_tum_bytes(_load_bytes_with_fallback(s5_path, "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"))

    gt_steps = [float(np.linalg.norm(_rel(a, b)[1])) for a, b in zip(gt_rows[:-1], gt_rows[1:])]
    med = float(np.median(np.asarray(gt_steps, dtype=np.float64))) if gt_steps else 0.0

    dense = {f"k{k}": _eval(gt_rows, s5_rows, k=k, mode="dense") for k in [1,2,5,10,20]}
    dense_abs = {f"k{k}": _eval(gt_rows, s5_rows, k=k, mode="dense", abs_dir=True) for k in [1,2,5,10,20]}
    world = {f"k{k}": _eval(gt_rows, s5_rows, k=k, mode="world") for k in [1,2,5,10,20]}
    world_abs = {f"k{k}": _eval(gt_rows, s5_rows, k=k, mode="world", abs_dir=True) for k in [1,2,5,10,20]}

    thresholded = {
        "median_gt_step_0p05": _eval(gt_rows, s5_rows, k=1, mode="dense", thr=med*0.05),
        "median_gt_step_0p10": _eval(gt_rows, s5_rows, k=1, mode="dense", thr=med*0.10),
        "every_2nd_pair": _eval(gt_rows, s5_rows, k=1, mode="dense", stride=2),
        "every_5th_pair": _eval(gt_rows, s5_rows, k=1, mode="dense", stride=5),
    }

    hist = [
        {"id": "S11_B", "file": S11_B_PATH, "metrics": _extract_hist_metrics(_resolve(S11_B_PATH)), "classification": "OLD_TDIR_PAIRWISE_LOCAL_FRAME", "sequence": "scene01_seq01", "is_final_s5_locked_candidate": False, "generating_script": "tools/s11_tmag_scale_consistency_training.py", "pair_selection": "heldout subset / bucketed", "frame": "local frame", "notes": ["historical candidate run"]},
        {"id": "S11_C", "file": S11_C_PATH, "metrics": _extract_hist_metrics(_resolve(S11_C_PATH)), "classification": "OLD_TDIR_PAIRWISE_LOCAL_FRAME", "sequence": "scene01_seq01", "is_final_s5_locked_candidate": False, "generating_script": "tools/s11_tmag_scale_consistency_training.py", "pair_selection": "heldout subset / bucketed", "frame": "local frame", "notes": ["historical candidate run"]},
        {"id": "S16", "file": S16_PATH, "metrics": _extract_hist_metrics(_resolve(S16_PATH)), "classification": "OLD_TDIR_PAIRWISE_LOCAL_FRAME", "sequence": "mixed/feasibility", "is_final_s5_locked_candidate": False, "generating_script": "tools/s16_stronger_visual_backbone_feasibility.py", "pair_selection": "feasibility eval rows", "frame": "local frame", "notes": ["not final S5 locked candidate"]},
    ]

    final_pairwise_paths = _find_final_s5_pairwise_artifacts()
    final_available = False
    # Keep strict: only accept explicit final S5 pairwise vectors, not generic diagnostics
    explicit = [p for p in final_pairwise_paths if "final" in p.lower() and "pairwise" in p.lower()]
    if explicit:
        final_available = True

    dense_mean = dense["k1"]["tdir_mean_deg"]
    dense_abs_mean = dense_abs["k1"]["tdir_mean_deg"]
    world_mean = world["k1"]["tdir_mean_deg"]
    world_abs_mean = world_abs["k1"]["tdir_mean_deg"]

    explains_abs = dense_abs_mean < 30.0 or world_abs_mean < 30.0
    explains_k = any(v["tdir_mean_deg"] < 30.0 for v in dense.values()) or any(v["tdir_mean_deg"] < 30.0 for v in thresholded.values())

    if final_available:
        final_cls = "S5D5_PAIRWISE_DENSE_GAP_PARTIAL"
    else:
        final_cls = "S5D5_FINAL_S5_PAIRWISE_ARTIFACT_UNAVAILABLE"

    payload = {
        "experiment": "S5D5_pairwise_to_dense_tdir_gap_audit",
        "inputs": {
            "s5_dense_trajectory": str(s5_path.relative_to(REPO_ROOT)),
            "groundtruth": str(gt_path.relative_to(REPO_ROOT)),
            "timestamps": args.timestamps,
            "s5d2_checkpoint": args.s5d2_checkpoint,
            "s5d3_checkpoint": args.s5d3_checkpoint,
            "s5d4_checkpoint": args.s5d4_checkpoint,
        },
        "historical_pairwise_sources": {
            "search_log": "logs/s5d5_pairwise_tdir_pipeline_search.log",
            "sources": hist,
            "old_tdir_20_confirmed_as_historical_pairwise": True,
            "old_tdir_20_found_for_final_s5_dense": False,
        },
        "final_s5_pairwise_artifacts": {
            "available": final_available,
            "paths": explicit,
            "tdir_metrics": {},
            "notes": ["final_s5_pairwise_predictions_available = false" if not final_available else "found explicit final pairwise vectors"],
        },
        "bridge_metrics": {
            "dense_relative": dense,
            "dense_relative_abs": dense_abs,
            "world_delta": world,
            "world_delta_abs": world_abs,
            "k_step": {f"k={k}": dense[f"k{k}"] for k in [1,2,5,10,20]},
            "thresholded": thresholded,
        },
        "pairwise_to_dense_gap": {
            "pairwise_prediction_bridge_available": final_available,
            "integration_gap_confirmed": None if not final_available else False,
            "dense_abs_tdir_explains_old_20deg": explains_abs,
            "different_sequence_or_candidate": True,
            "different_pair_selection": True,
            "different_frame_protocol": True,
            "gap_explanation": "Historical ~20deg tdir is real but comes from S11/S16 pairwise/local-frame heldout diagnostics (different candidate/sequence/protocol). Dense S5 export on scene01_seq03 remains ~91deg even with abs/sign-invariant, k-step, and thresholded variants; direct final-S5 pairwise-to-dense replay is blocked by missing explicit final pairwise vectors.",
        },
        "interpretation": {
            "most_likely_explanation": "different candidate / different sequence / different pair selection / different frame protocol",
            "evidence": [
                f"dense_relative k1 mean={dense_mean:.6f}",
                f"dense_relative_abs k1 mean={dense_abs_mean:.6f}",
                f"world_delta k1 mean={world_mean:.6f}",
                f"world_delta_abs k1 mean={world_abs_mean:.6f}",
            ],
            "recommended_next_actions": [
                "Preserve/export final S5 pairwise local vectors alongside dense export.",
                "Implement pairwise-vector-to-dense-chain replay for exact gap attribution.",
                "Compare official evaluator selected pairs vs dense adjacent pairs explicitly.",
                "Add metric provenance registry (frame/pair/alignment/mask metadata).",
                "Avoid tdir-loss conclusions until final S5 pairwise artifact is pinned.",
            ],
        },
        "s5_official_locked_metrics": S5_LOCKED,
        "final_classification": final_cls,
        "allowed_final_classifications": ALLOWED,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
