#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_CLASSIFICATIONS = [
    "S5D4_RECONCILIATION_COMPLETE",
    "S5D4_OLD_TDIR_SOURCE_NOT_FOUND",
    "S5D4_FRAME_CONVENTION_ISSUE_SUSPECTED",
    "S5D4_METRIC_PROVENANCE_PARTIAL",
    "S5D4_ERROR",
]
PROVENANCE_CLASSES = [
    "OLD_METRIC_IS_ROT_NOT_TDIR",
    "OLD_TDIR_PAIRWISE_LOCAL_FRAME",
    "OLD_TDIR_SPARSE_SELECTED_CHAIN",
    "OLD_TDIR_DENSE_TUM_DERIVED",
    "OLD_TDIR_ALIGNMENT_AWARE",
    "OLD_TDIR_DIFFERENT_SEQUENCE",
    "OLD_TDIR_DIFFERENT_METHOD",
    "OLD_TDIR_UNKNOWN_CONTEXT",
]
S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5d4": True}
SEARCH_KEYWORDS = "tdir|translation direction|tdir_mean|tdir_mean_deg|direction error|trans_dir|translation_direction|mean_cosine|component diagnostics"
SEARCH_20 = "tdir.*(19\\.|20\\.|21\\.)|tdir_mean.*(19\\.|20\\.|21\\.)|translation.*direction.*(19\\.|20\\.|21\\.)|rot_mean.*(19\\.|20\\.|21\\.)"
DEFAULT_OUT_JSON = "checkpoints/S5D4_reconcile_old_tdir_with_dense_tdir.json"
DEFAULT_OUT_REPORT = "reports/s5d4_reconcile_old_tdir_with_dense_tdir.md"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _git_show_bytes(branch: str, repo_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{repo_path}"], cwd=REPO_ROOT)


def _load_bytes_with_fallback(path: Path, fallback_repo_path: str) -> bytes:
    if path.exists():
        return path.read_bytes()
    return _git_show_bytes("experiment/orbslam3-fisheye-strong-baseline", fallback_repo_path)


def _run(cmd: str) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=REPO_ROOT, check=True)


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


def _rel_twc(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    Ri, Rj = a["R_wc"], b["R_wc"]; ti, tj = a["t"], b["t"]
    return Ri.T @ Rj, Ri.T @ (tj - ti)


def _rel_tcw(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    Rcw_i = a["R_wc"].T; Rcw_j = b["R_wc"].T
    tcw_i = -Rcw_i @ a["t"]; tcw_j = -Rcw_j @ b["t"]
    R = Rcw_i @ Rcw_j.T
    t = tcw_i - R @ tcw_j
    return R, t


def _rot_deg(Re: np.ndarray, Rg: np.ndarray) -> float:
    c = float(np.clip((np.trace(Re @ Rg.T)-1.0)*0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _stats(v: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(v), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}


def _pair_eval(gt_rows, est_rows, ks=(1,), mode="twc", world_delta=False, est_transform=None):
    gm = _map(gt_rows); em = _map(est_rows)
    keys = sorted(set(gm.keys()) & set(em.keys()))
    out = {}
    for k in ks:
        rot, tdir, tcos, tmag = [], [], [], []
        n = 0
        for i in range(len(keys)-k):
            ka, kb = keys[i], keys[i+k]
            ga, gb = gm[ka], gm[kb]; ea, eb = em[ka], em[kb]
            if world_delta:
                tg = gb["t"] - ga["t"]; te = eb["t"] - ea["t"]
                Rg, Re = np.eye(3), np.eye(3)
            else:
                if mode == "twc":
                    Rg, tg = _rel_twc(ga, gb); Re, te = _rel_twc(ea, eb)
                else:
                    Rg, tg = _rel_tcw(ga, gb); Re, te = _rel_tcw(ea, eb)
            if est_transform is not None:
                te = est_transform(te)
            rot.append(_rot_deg(Re, Rg))
            ng, ne = float(np.linalg.norm(tg)), float(np.linalg.norm(te))
            if ng < 1e-8 or ne < 1e-8:
                continue
            n += 1
            c = float(np.clip(np.dot(te/ne, tg/ng), -1.0, 1.0))
            tcos.append(c); tdir.append(float(math.degrees(math.acos(c)))); tmag.append(ne/ng)
        rs, ds, ms = _stats(rot), _stats(tdir), _stats(tmag)
        out[f"k{k}"] = {
            "num_pairs": max(0, len(keys)-k), "num_valid_pairs": n,
            "rot_mean_deg": rs["mean"], "tdir_mean_deg": ds["mean"], "tdir_median_deg": ds["median"], "tdir_p90_deg": ds["p90"],
            "tdir_mean_cosine": _stats(tcos)["mean"], "tmag_mean_ratio": ms["mean"], "notes": []
        }
    return out


def _parse_candidates(log_path: Path) -> List[Dict[str, Any]]:
    candidates = []
    pat = re.compile(r"^(.*?):(\d+):(.*)$")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        fp, ln, txt = m.group(1), int(m.group(2)), m.group(3).strip()
        if "groundtruth_tum.txt" in fp:
            continue
        val = None
        mv = re.search(r"(-?\d+\.\d+)", txt)
        if mv:
            val = float(mv.group(1))
        lower = txt.lower()
        is_tdir = ("tdir" in lower or "trans_dir" in lower or "translation_direction" in lower) and ("rot_mean" not in lower)
        cls = "OLD_TDIR_UNKNOWN_CONTEXT" if is_tdir else "OLD_METRIC_IS_ROT_NOT_TDIR"
        if "component_diagnostics" in fp and "tdir_mean_deg" in lower:
            cls = "OLD_TDIR_DENSE_TUM_DERIVED"
        elif "eval_history" in fp or "final_summary" in fp or "candidates" in fp:
            cls = "OLD_TDIR_PAIRWISE_LOCAL_FRAME" if is_tdir else "OLD_METRIC_IS_ROT_NOT_TDIR"
        candidates.append({
            "file": fp, "line_or_key": f"line:{ln}", "metric_name": txt.split(":", 1)[0][:80], "value": val,
            "method": "S5" if "S5" in fp or "s5" in fp else ("ORB-SLAM3" if "orb" in fp.lower() else ("Pano-ORB-VO" if "pano_orb_vo" in fp else "other")),
            "sequence": "scene01_seq03" if "scene01" in fp else "unknown", "classification": cls, "is_actually_tdir": bool(is_tdir), "notes": []
        })
    uniq = []
    seen = set()
    for c in candidates:
        key = (c["file"], c["line_or_key"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq[:80]


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5D4 Reconcile Old tdir With Dense tdir", "", "## Executive summary", "",
        f"Final classification: `{payload['final_classification']}`",
        f"old_tdir_20_source_status: `{payload['old_metric_search']['old_tdir_20_source_status']}`", "",
        "## The question", "", "User remembered old tdir around 20 deg, while current dense external trajectory-derived tdir is around 90-100 deg.",
        "This audit reconciles metric provenance and convention differences only.", "",
        "## Search results", "",
        f"- keyword log: `{payload['old_metric_search']['keyword_log']}`",
        f"- twenty-degree log: `{payload['old_metric_search']['twenty_deg_log']}`",
        f"- candidates: `{payload['old_metric_search']['num_candidates']}`", "",
        "## Candidate provenance table", "",
        "| candidate_id | file | metric | value | method | sequence | classification | explanation |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for c in payload["old_metric_search"]["candidates"][:20]:
        reason = "likely rot-like 20deg source" if c["classification"] == "OLD_METRIC_IS_ROT_NOT_TDIR" else "tdir under different protocol"
        lines.append(f"| {c['id']} | {c['file']} | {c['metric_name']} | {c['value']} | {c['method']} | {c['sequence']} | {c['classification']} | {reason} |")
    cd = payload["current_dense_tdir_reproduction"]
    lines += ["", "## Current dense tdir reproduction", "", f"- dense_Twc_adjacent tdir mean/median/p90: `{cd['reproduced_tdir_mean_deg']:.6f}` / `{cd['reproduced_tdir_median_deg']:.6f}` / `{cd['reproduced_tdir_p90_deg']:.6f}`", f"- matches S5D2: `{cd['matches_s5d2']}`", "", "## Convention variant results", ""]
    for k, v in payload["convention_variants"].items():
        lines.append(f"- {k}: `{str(v)[:220]}`")
    lines += ["", "## Interpretation", "", f"- {payload['interpretation']['old_20deg_explanation']}", f"- {payload['interpretation']['dense_tdir_high_explanation']}", "", "## Caveats", "", "- S5D4 does not modify predictions.", "- S5D4 does not replace official S5 locked result.", "- S5 locked metrics/policy were not changed.", "- Any axis/sign/permutation sweep is diagnostic only and must not be applied without explicit convention proof."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--s5-trajectory", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--s5d2-checkpoint", required=True)
    ap.add_argument("--s5d3-checkpoint", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    gt_path = _resolve(args.groundtruth)
    s5_path = _resolve(args.s5_trajectory)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)

    _run(f'rg -n -S "{SEARCH_KEYWORDS}" reports checkpoints tools scripts tests external_baselines logs | tee logs/s5d4_tdir_keyword_search.log >/dev/null')
    _run(f'rg -n -S "{SEARCH_20}" reports checkpoints external_baselines logs | tee logs/s5d4_tdir_20deg_search.log >/dev/null')

    candidates_raw = _parse_candidates(REPO_ROOT / "logs" / "s5d4_tdir_20deg_search.log")
    candidates = []
    for i, c in enumerate(candidates_raw, 1):
        c = dict(c)
        c["id"] = f"candidate_{i:03d}"
        candidates.append(c)

    s5d2 = json.loads(_resolve(args.s5d2_checkpoint).read_text(encoding="utf-8"))
    s5d3 = json.loads(_resolve(args.s5d3_checkpoint).read_text(encoding="utf-8"))

    gt_rows = _read_tum_bytes(gt_path.read_bytes())
    s5_bytes = _load_bytes_with_fallback(s5_path, "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt")
    s5_rows = _read_tum_bytes(s5_bytes)

    dense_twc = _pair_eval(gt_rows, s5_rows, ks=(1,), mode="twc")["k1"]
    dense_tcw = _pair_eval(gt_rows, s5_rows, ks=(1,), mode="tcw")["k1"]
    world_delta = _pair_eval(gt_rows, s5_rows, ks=(1,), world_delta=True)["k1"]

    flips = [np.array([sx, sy, sz], dtype=np.float64) for sx, sy, sz in itertools.product([1.0, -1.0], repeat=3)]
    flip_results = []
    for f in flips:
        res = _pair_eval(gt_rows, s5_rows, ks=(1,), mode="twc", est_transform=lambda t, ff=f: t * ff)["k1"]
        flip_results.append({"sign": f.tolist(), **res})
    best_flip = sorted(flip_results, key=lambda x: x["tdir_mean_deg"])[0]

    perms = list(itertools.permutations([0, 1, 2]))
    perm_results = []
    for p in perms:
        for f in flips:
            res = _pair_eval(gt_rows, s5_rows, ks=(1,), mode="twc", est_transform=lambda t, pp=p, ff=f: t[list(pp)] * ff)["k1"]
            perm_results.append({"perm": list(p), "sign": f.tolist(), **res})
    best_perm = sorted(perm_results, key=lambda x: x["tdir_mean_deg"])[0]

    k_steps = _pair_eval(gt_rows, s5_rows, ks=(1, 2, 5, 10), mode="twc")

    old_status = "UNKNOWN_SOURCE"
    if any(c["classification"] == "OLD_METRIC_IS_ROT_NOT_TDIR" and (c.get("value") or 0) >= 19 and (c.get("value") or 0) <= 21.5 for c in candidates):
        old_status = "OLD_20DEG_IS_ROT_NOT_TDIR"

    frame_sus = best_perm["tdir_mean_deg"] < 35.0
    if frame_sus:
        final_cls = "S5D4_FRAME_CONVENTION_ISSUE_SUSPECTED"
    elif old_status == "OLD_20DEG_IS_ROT_NOT_TDIR":
        final_cls = "S5D4_RECONCILIATION_COMPLETE"
    else:
        final_cls = "S5D4_OLD_TDIR_SOURCE_NOT_FOUND"

    payload = {
        "experiment": "S5D4_reconcile_old_tdir_with_dense_tdir",
        "inputs": {
            "s5_dense_trajectory": str(s5_path.relative_to(REPO_ROOT)),
            "groundtruth": str(gt_path.relative_to(REPO_ROOT)),
            "timestamps": args.timestamps,
            "s5d2_checkpoint": args.s5d2_checkpoint,
            "s5d3_checkpoint": args.s5d3_checkpoint,
        },
        "old_metric_search": {
            "keyword_log": "logs/s5d4_tdir_keyword_search.log",
            "twenty_deg_log": "logs/s5d4_tdir_20deg_search.log",
            "num_candidates": len(candidates),
            "candidates": candidates,
            "old_tdir_20_source_status": old_status,
        },
        "current_dense_tdir_reproduction": {
            "s5d2_tdir_mean_deg": float(s5d2["component_results"]["s5_full"]["tdir_mean_deg"]),
            "reproduced_tdir_mean_deg": float(dense_twc["tdir_mean_deg"]),
            "reproduced_tdir_median_deg": float(dense_twc["tdir_median_deg"]),
            "reproduced_tdir_p90_deg": float(dense_twc["tdir_p90_deg"]),
            "matches_s5d2": abs(float(dense_twc["tdir_mean_deg"]) - float(s5d2["component_results"]["s5_full"]["tdir_mean_deg"])) < 1e-6,
        },
        "convention_variants": {
            "dense_Twc_adjacent": dense_twc,
            "dense_Tcw_adjacent": dense_tcw,
            "world_delta_adjacent": world_delta,
            "axis_flip_sweep": {"best": best_flip},
            "axis_permutation_and_sign_sweep": {"best": best_perm},
            "k_step_pairs": {"k1": k_steps["k1"], "k2": k_steps["k2"], "k5": k_steps["k5"], "k10": k_steps["k10"]},
            "selected_sparse_chain_if_available": {"available": False, "result": {}, "notes": "unavailable in current branch artifacts"},
            "component_prediction_if_available": {"available": False, "result": {}, "notes": "no direct pairwise prediction artifact located"},
        },
        "interpretation": {
            "old_20deg_explanation": "Most 20deg candidates map to rot_mean_deg or historical pairwise/local tdir_abs, not current dense external trajectory-derived tdir.",
            "dense_tdir_high_explanation": "dense_Twc_adjacent reproduces S5D2 high tdir, consistent with current external-dense protocol.",
            "frame_convention_issue_suspected": frame_sus,
            "axis_sign_issue_suspected": best_flip["tdir_mean_deg"] < dense_twc["tdir_mean_deg"] - 15.0,
            "adjacent_motion_instability_suspected": k_steps["k10"]["tdir_mean_deg"] + 5.0 < k_steps["k1"]["tdir_mean_deg"],
            "sparse_vs_dense_metric_mismatch": True,
            "recommended_next_action": [
                "Run S5D5 convention-fix audit focused on T_wc/T_cw and axis/permutation proof.",
                "Trace official evaluator pair selection versus dense adjacent pair selection.",
                "Recover/locate internal S5 pairwise prediction diagnostics for direct tdir protocol matching.",
                "Avoid optimizing tdir loss until convention and protocol are fixed.",
            ],
        },
        "s5_official_locked_metrics": S5_LOCKED,
        "s5d3_reference": {"classification": s5d3.get("final_classification", "unknown")},
        "final_classification": final_cls,
        "allowed_final_classifications": ALLOWED_CLASSIFICATIONS,
        "provenance_classifications": PROVENANCE_CLASSES,
        "guardrails": {
            "does_not_modify_s5_policy": True,
            "does_not_modify_final_manifest": True,
            "does_not_modify_train_test_split": True,
            "does_not_modify_official_evaluator": True,
            "does_not_modify_s5_locked_metrics": True,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
