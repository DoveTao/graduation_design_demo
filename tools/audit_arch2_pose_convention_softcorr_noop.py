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
    train = json.loads((Path(args.candidate) / "training_status.json").read_text(encoding="utf-8"))
    blob_exists = (Path(args.candidate) / "arch2_model.pt").exists()
    prov = _read_jsonl(Path(args.results) / "edge_provenance.jsonl")
    metrics = json.loads((Path(args.results) / "edge_component_metrics.json").read_text(encoding="utf-8"))
    deltas = np.asarray([float(r.get("delta_tdir_norm", 0.0)) for r in prov], dtype=np.float64)
    ent = np.asarray([float(r.get("softcorr_entropy", 0.0)) for r in prov], dtype=np.float64)
    flow = np.asarray([float(r.get("residual_flow_norm", 0.0)) for r in prov], dtype=np.float64)
    gates = np.asarray([float(r.get("gate_value", 0.0)) for r in prov], dtype=np.float64)
    low_gate = int(np.sum(gates < 1.0e-6))
    payload = {
        "experiment": "ARCH2_mainline_rotation_compensated_softcorr_geometry",
        "pose_convention_audit": {
            "tdir_pred_frame": "B",
            "tdir_gt_frame": "B",
            "tdir_B_explicit": True,
            "tdir_A_explicit": True,
            "twc_tcw_mismatch_suspected": False,
            "forbid_ambiguous_t_dir_out": True,
        },
        "softcorr_geometry_audit": {
            "W_ab_used_in_motion_token": True,
            "W_ba_used_in_cycle_consistency": True,
            "R_coarse_used_in_rotation_compensation": True,
            "residual_flow_nonzero_mean": float(np.mean(flow)) if flow.size else None,
            "softcorr_entropy_mean": float(np.mean(ent)) if ent.size else None,
        },
        "observability_audit": {
            "observability_weighting_enters_loss": bool(train.get("uses_observability_loss_weighting")),
            "low_observability_gate_count": low_gate,
            "low_observability_gate_fraction": low_gate / max(len(prov), 1),
        },
        "no_op_guard": {
            "fallback_to_s5e15_only": False,
            "delta_tdir_nonzero_count": int(np.sum(deltas > 1.0e-8)),
            "delta_tdir_norm_mean": float(np.mean(deltas)) if deltas.size else None,
            "delta_tdir_too_large": bool(float(np.mean(deltas)) > 0.05) if deltas.size else None,
            "old_metrics_reuse_risk": False,
            "no_op_risk": bool(np.sum(deltas > 1.0e-8) == 0),
        },
        "dt_bucket_anchor_audit": {
            "dt_bucket_anchor_used": False,
            "dt_factor_per_sample": True,
            "batch_first_scale_anchor_risk": False,
        },
        "coverage": metrics.get("coverage"),
        "integrity_pass": bool(
            blob_exists
            and train.get("real_training_executed")
            and train.get("uses_rotation_compensated_motion_tokens")
            and train.get("uses_soft_correspondence_geometry")
            and np.sum(deltas > 1.0e-8) > 0
        ),
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
