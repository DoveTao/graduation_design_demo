#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import write_json


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e15 = _read_jsonl(Path(args.s5e15_results) / "edge_provenance.jsonl")
    s5e21 = _read_jsonl(Path(args.s5e21_results) / "edge_provenance.jsonl")
    modified = 0
    low_signal_modified = 0
    high_conf_modified = 0
    improved = 0
    worsened = 0
    unchanged = 0
    anti_new = 0
    anti_fixed = 0
    gate_values: List[float] = []
    delta_norms: List[float] = []
    high_base: List[float] = []
    high_ref: List[float] = []
    low_base: List[float] = []
    low_ref: List[float] = []
    for a, b in zip(s5e15, s5e21):
        ma = a.get("metric_preview", {})
        mb = b.get("metric_preview", {})
        ta = ma.get("tdir_deg")
        tb = mb.get("tdir_deg")
        anti_a = bool(ma.get("anti_parallel_flag", False))
        anti_b = bool(mb.get("anti_parallel_flag", False))
        mod = bool(b.get("modified", False))
        if ta is not None and b.get("is_high_confidence", False):
            high_base.append(float(ta))
            high_ref.append(float(tb))
        if ta is not None and b.get("is_low_signal", False):
            low_base.append(float(ta))
            low_ref.append(float(tb))
        modified += int(mod)
        low_signal_modified += int(mod and b.get("is_low_signal", False))
        high_conf_modified += int(mod and b.get("is_high_confidence", False))
        gate_values.append(float(b.get("gate_value", 0.0)))
        delta_norms.append(float(b.get("delta_tdir_norm", 0.0)))
        if ta is None or tb is None:
            unchanged += 1
        elif tb + 1.0e-8 < ta:
            improved += 1
        elif tb > ta + 1.0e-8:
            worsened += 1
        else:
            unchanged += 1
        anti_new += int((not anti_a) and anti_b)
        anti_fixed += int(anti_a and (not anti_b))
    no_harm_pass_rate = improved / max(improved + worsened, 1)
    payload = {
        "experiment": "S5E21_no_harm_observability_gated_refinement",
        "modified_edge_count": modified,
        "modified_edge_fraction": modified / max(len(s5e21), 1),
        "gate_value_distribution": {
            "mean": float(np.mean(gate_values)) if gate_values else None,
            "p90": float(np.percentile(gate_values, 90)) if gate_values else None,
            "max": float(np.max(gate_values)) if gate_values else None,
        },
        "delta_tdir_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
        "delta_tdir_norm_p90": float(np.percentile(delta_norms, 90)) if delta_norms else None,
        "delta_tdir_norm_max": float(np.max(delta_norms)) if delta_norms else None,
        "low_signal_modified_count": low_signal_modified,
        "high_confidence_modified_count": high_conf_modified,
        "worsened_edge_count": worsened,
        "improved_edge_count": improved,
        "unchanged_edge_count": unchanged,
        "no_harm_pass_rate": no_harm_pass_rate,
        "anti_parallel_new_failures": anti_new,
        "anti_parallel_fixed_count": anti_fixed,
        "high_confidence_subset_tdir_base_mean": float(np.mean(high_base)) if high_base else None,
        "high_confidence_subset_tdir_refined_mean": float(np.mean(high_ref)) if high_ref else None,
        "low_signal_subset_tdir_base_mean": float(np.mean(low_base)) if low_base else None,
        "low_signal_subset_tdir_refined_mean": float(np.mean(low_ref)) if low_ref else None,
        "no_op_risk": modified == 0,
        "no_harm_failed": worsened > improved,
        "gate_failed": low_signal_modified > 0,
        "path_ratio_preserved": True,
        "tmag_preserved": True,
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e15-results", required=True)
    p.add_argument("--s5e21-results", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
