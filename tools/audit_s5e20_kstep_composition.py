#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import read_tum, read_timestamps, vector_angle_deg, write_json


S5E19C_DELTA_REF = 0.6029416022053354


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Tuple[np.ndarray, np.ndarray] | None:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _pred_rel(pred: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Tuple[np.ndarray, np.ndarray] | None:
    if a not in pred or b not in pred:
        return None
    pi, pj = pred[a], pred[b]
    return pj["R"].T @ pi["R"], pj["R"].T @ (pi["t"] - pj["t"])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    candidate_dir = Path(args.candidate)
    train_status = json.loads((candidate_dir / "training_status.json").read_text(encoding="utf-8"))
    metrics = json.loads(Path(args.results, "edge_component_metrics.json").read_text(encoding="utf-8"))
    provenance = _read_jsonl(Path(args.results, "edge_provenance.jsonl"))
    gt = read_tum(Path("external_baselines/dataset/scene01_seq03/groundtruth_tum.txt"))
    pred = read_tum(Path(args.results, "scene01_seq03_s5e20_traceable_dense_tum.txt"))
    timestamps = read_timestamps(Path("external_baselines/dataset/scene01_seq03/timestamps.txt"))

    k_vals = [1, 2, 3, 5]
    k_scores: Dict[int, List[float]] = {k: [] for k in k_vals}
    comp_errors: List[float] = []
    path_errors: List[float] = []
    for k in k_vals:
        for i in range(0, len(timestamps) - k):
            gt_rel = _gt_rel(gt, timestamps[i], timestamps[i + k])
            pred_rel = _pred_rel(pred, timestamps[i], timestamps[i + k])
            if gt_rel is None or pred_rel is None:
                continue
            _, t_gt = gt_rel
            _, t_pr = pred_rel
            ang = vector_angle_deg(t_pr, t_gt, absolute=False)
            if ang is not None:
                k_scores[k].append(float(ang))
                if k > 1:
                    comp_errors.append(float(ang))
            path_errors.append(abs(float(np.linalg.norm(t_pr)) - float(np.linalg.norm(t_gt))))

    delta_norms = [float(x.get("delta_tdir_norm", 0.0)) for x in provenance]
    payload = {
        "experiment": "S5E20_true_kstep_composition_supervision",
        "pairs_contiguous": bool(pairs.get("all_windows_contiguous")),
        "train_has_eval_scene": bool(pairs.get("uses_eval_scene")),
        "uses_true_contiguous_kstep_windows": bool(train_status.get("uses_true_contiguous_kstep_windows")),
        "uses_random_batch_proxy": bool(train_status.get("uses_random_batch_proxy")),
        "t_chain_contiguous_composition": True,
        "delta_tdir_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
        "delta_tdir_norm_p90": float(np.percentile(delta_norms, 90)) if delta_norms else None,
        "delta_tdir_too_aggressive": bool(delta_norms and np.mean(delta_norms) > 0.20),
        "more_conservative_than_s5e19c": bool(delta_norms and np.mean(delta_norms) < S5E19C_DELTA_REF),
        "k1_tdir": float(np.mean(k_scores[1])) if k_scores[1] else None,
        "k2_tdir": float(np.mean(k_scores[2])) if k_scores[2] else None,
        "k3_tdir": float(np.mean(k_scores[3])) if k_scores[3] else None,
        "k5_tdir": float(np.mean(k_scores[5])) if k_scores[5] else None,
        "kstep_composition_error": float(np.mean(comp_errors)) if comp_errors else None,
        "path_length_consistency_error": float(np.mean(path_errors)) if path_errors else None,
        "coverage": metrics.get("coverage", {}),
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
