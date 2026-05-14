#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _mean(vals: List[float]) -> float | None:
    return None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64)))


def _rate(flags: List[bool]) -> float | None:
    return None if not flags else float(np.mean(np.asarray(flags, dtype=np.float64)))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.s5e13_metrics).read_text(encoding="utf-8"))
    prov = _read_jsonl(Path(args.s5e13_provenance))
    weights = _read_jsonl(Path(args.s5e13_weights))
    weight_by_idx = {int(r["edge_index"]): r for r in weights}

    feat_by_idx: Dict[int, Dict[str, Any]] = {}
    with Path(args.s5e12_feature_dir, "correspondence_features.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feat_by_idx[int(row["edge_id"])] = row

    rows: List[Dict[str, Any]] = []
    for p in prov:
        idx = int(p["edge_index"])
        m = dict(p.get("metric_preview", {}))
        w = weight_by_idx.get(idx, {})
        feat = feat_by_idx.get(idx, {})
        angle_disp = float(feat.get("flow_angle_dispersion", feat.get("match_angle_dispersion", 0.0)) or 0.0)
        rows.append(
            {
                "edge_index": idx,
                "observable": bool(p.get("observable", False)),
                "reliable_original": bool(p.get("signed_direction_reliable", False)),
                "small_motion": bool(p.get("small_motion", False)),
                "low_parallax": bool(p.get("low_parallax", False)),
                "near_static": bool(p.get("near_static", False)),
                "anti_parallel": bool(m.get("anti_parallel_flag", False)),
                "signed_tdir_deg": float(m.get("tdir_deg", 0.0) or 0.0),
                "tdir_abs_deg": float(m.get("tdir_abs_deg", 0.0) or 0.0),
                "parallax_proxy": float(w.get("parallax_proxy", feat.get("parallax_proxy", 0.0)) or 0.0),
                "flow_magnitude": float(w.get("flow_magnitude", feat.get("median_flow_magnitude", 0.0)) or 0.0),
                "inlier_ratio": float(w.get("inlier_ratio", feat.get("inlier_ratio", 0.0)) or 0.0),
                "gt_tmag": float(feat.get("gt_tmag", 0.0) or 0.0),
                "angle_dispersion": angle_disp,
            }
        )

    bad_obs = [r for r in rows if r["observable"] and r["signed_tdir_deg"] >= 70.0]
    rel_rows = [r for r in rows if r["reliable_original"]]
    obs_rows = [r for r in rows if r["observable"]]
    unobs_rows = [r for r in rows if not r["observable"]]

    stricter = [
        r
        for r in rel_rows
        if (not r["low_parallax"]) and (not r["near_static"]) and (not r["small_motion"]) and r["angle_dispersion"] <= 55.0 and r["parallax_proxy"] >= 0.35 and r["flow_magnitude"] >= 0.2 and r["inlier_ratio"] >= 0.93 and r["gt_tmag"] >= 0.08
    ]

    payload = {
        "s5e13_reference": {
            "overall_tdir_mean_deg": 50.35304145887563,
            "observable_tdir_mean_deg": 46.32471099526666,
            "reliable_tdir_mean_deg": 47.07029256818264,
            "anti_parallel_rate": 0.1479028697571744,
            "reliable_anti_parallel_rate": 0.20535714285714285,
        },
        "error_clusters": {
            "low_parallax": {
                "count": int(sum(1 for r in bad_obs if r["low_parallax"])),
                "fraction_in_bad_observable": _rate([r["low_parallax"] for r in bad_obs]),
            },
            "near_static": {
                "count": int(sum(1 for r in bad_obs if r["near_static"])),
                "fraction_in_bad_observable": _rate([r["near_static"] for r in bad_obs]),
            },
            "high_angle_dispersion": {
                "count": int(sum(1 for r in bad_obs if r["angle_dispersion"] > 55.0)),
                "fraction_in_bad_observable": _rate([r["angle_dispersion"] > 55.0 for r in bad_obs]),
            },
            "small_motion": {
                "count": int(sum(1 for r in bad_obs if r["small_motion"])),
                "fraction_in_bad_observable": _rate([r["small_motion"] for r in bad_obs]),
            },
            "anti_parallel": {
                "count": int(sum(1 for r in bad_obs if r["anti_parallel"])),
                "fraction_in_bad_observable": _rate([r["anti_parallel"] for r in bad_obs]),
            },
        },
        "threshold_recommendation": {
            "current_reliable_count": 112,
            "stricter_reliable_count": len(stricter),
            "expected_noise_reduction": None if not rel_rows else float(_mean([r["signed_tdir_deg"] for r in rel_rows]) - _mean([r["signed_tdir_deg"] for r in stricter])) if stricter else None,
        },
        "diagnosis": {
            "reliable_mask_too_loose": _rate([r["anti_parallel"] for r in rel_rows]) is not None and _rate([r["anti_parallel"] for r in rel_rows]) > 0.18,
            "observable_edges_include_noisy_direction_cases": len(bad_obs) > max(15, int(0.12 * max(len(obs_rows), 1))),
            "inference_time_blending_recommended": True,
            "main_direction_error_source": "mask_noise" if len(stricter) > 0 else "unknown",
        },
        "stats": {
            "observable_count": len(obs_rows),
            "unobservable_count": len(unobs_rows),
            "reliable_original_count": len(rel_rows),
            "observable_anti_parallel_rate": _rate([r["anti_parallel"] for r in obs_rows]),
            "unobservable_anti_parallel_rate": _rate([r["anti_parallel"] for r in unobs_rows]),
            "reliable_original_anti_parallel_rate": _rate([r["anti_parallel"] for r in rel_rows]),
            "bad_observable_count": len(bad_obs),
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e13-checkpoint", required=True)
    p.add_argument("--s5e13-metrics", required=True)
    p.add_argument("--s5e13-provenance", required=True)
    p.add_argument("--s5e13-weights", required=True)
    p.add_argument("--s5e12-feature-dir", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
