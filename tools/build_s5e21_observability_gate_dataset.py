#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import adjacent_pairs, relative_pose_A_to_B_in_B, scan_frames, write_json


def _parse_cfg(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            return [] if not inner else [parse_scalar(x.strip()) for x in inner.split(",")]
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text.strip('"')

    cfg: Dict[str, Any] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, value = raw.split(":", 1)
            key, value = key.strip(), value.strip()
            if value:
                cfg[key] = parse_scalar(value)
                current = None
            else:
                cfg[key] = {}
                current = key
        elif current is not None and ":" in raw:
            key, value = raw.strip().split(":", 1)
            cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def _corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "inlier_ratio": float(orb["inlier_ratio"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "low_parallax": bool(orb["low_parallax_flag"]),
    }


def _obs_score(corr: Dict[str, float], motion_mag: float) -> Tuple[float, float, float]:
    direction_conf = (
        0.45 * min(1.0, corr["inlier_ratio"])
        + 0.35 * min(1.0, corr["parallax_proxy"] / 0.30)
        + 0.20 * min(1.0, corr["median_flow_magnitude"] / 8.0)
    )
    base_proxy = 1.0 - direction_conf
    observability = direction_conf
    if corr["low_parallax"]:
        observability *= 0.55
    if motion_mag < 0.01:
        observability *= 0.35
    return float(observability), float(direction_conf), float(base_proxy)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    frames = scan_frames(Path("data"), scene="scene01")
    train_scenes = [str(x) for x in cfg["training"]["train_scenes"]]
    records: List[Dict[str, Any]] = []
    scores: List[float] = []
    for scene_seq in train_scenes:
        scene, seq = scene_seq.split("/")
        for idx, (a, b) in enumerate(adjacent_pairs(frames[(scene, seq)])):
            _, t = relative_pose_A_to_B_in_B(a, b)
            motion_mag = float(np.linalg.norm(t))
            corr = _corr(a, b)
            observability, direction_conf, base_proxy = _obs_score(corr, motion_mag)
            scores.append(observability)
            records.append(
                {
                    "scene": scene,
                    "seq": seq,
                    "edge_index": idx,
                    "timestamp_i": float(a.timestamp),
                    "timestamp_j": float(b.timestamp),
                    "observability_score": observability,
                    "direction_confidence": direction_conf,
                    "motion_magnitude": motion_mag,
                    "base_tdir_error_proxy": base_proxy,
                    "is_high_confidence": False,
                    "is_low_signal": False,
                    "gate_value": 0.0,
                    "uses_eval_scene": False,
                }
            )
    warnings: List[str] = []
    if not scores:
        payload = {
            "experiment": "S5E21_no_harm_observability_gated_refinement",
            "num_edges": 0,
            "num_high_confidence": 0,
            "num_low_signal": 0,
            "high_confidence_fraction": 0.0,
            "threshold_source": "fallback_proxy",
            "uses_eval_gt_for_gate": False,
            "gate_dataset_ready": False,
            "warnings": ["no_train_edges"],
            "records": [],
        }
        write_json(Path(args.out_json), payload)
        return payload

    high_q = float(cfg["gate_thresholds"]["high_confidence_quantile"])
    low_q = float(cfg["gate_thresholds"]["low_signal_quantile"])
    high_thr = float(np.quantile(np.asarray(scores, dtype=np.float64), high_q))
    low_thr = float(np.quantile(np.asarray(scores, dtype=np.float64), low_q))
    threshold_source = "train_split_quantile"
    if not math.isfinite(high_thr) or not math.isfinite(low_thr):
        high_thr = float(cfg["gate_thresholds"]["fallback_high_confidence_threshold"])
        low_thr = float(cfg["gate_thresholds"]["fallback_low_signal_threshold"])
        threshold_source = "fallback_proxy"
        warnings.append("used_fallback_thresholds")

    num_high = 0
    num_low = 0
    for row in records:
        score = float(row["observability_score"])
        is_high = score >= high_thr
        is_low = score <= low_thr or row["motion_magnitude"] < 0.01
        gate_value = 0.0 if is_low else min(1.0, max(0.0, (score - low_thr) / max(high_thr - low_thr, 1.0e-6)))
        row["is_high_confidence"] = bool(is_high)
        row["is_low_signal"] = bool(is_low)
        row["gate_value"] = float(gate_value)
        num_high += int(is_high)
        num_low += int(is_low)
    if num_high < max(8, int(0.05 * len(records))):
        warnings.append("gate_dataset_low_coverage")

    payload = {
        "experiment": "S5E21_no_harm_observability_gated_refinement",
        "num_edges": len(records),
        "num_high_confidence": num_high,
        "num_low_signal": num_low,
        "high_confidence_fraction": num_high / max(len(records), 1),
        "threshold_source": threshold_source,
        "high_confidence_threshold": high_thr,
        "low_signal_threshold": low_thr,
        "uses_eval_gt_for_gate": False,
        "gate_dataset_ready": len(records) > 0,
        "warnings": warnings,
        "records": records,
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
