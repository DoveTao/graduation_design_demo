#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import adjacent_pairs, relative_pose_A_to_B_in_B, scan_frames, split_sequence_keys
from s5e7_direction_scale_lib import rotation_matrix_to_rotvec


@dataclass
class S5E9Sample:
    frame_i: Any
    frame_j: Any
    edge_index: int
    total_edges: int
    gt_direction: np.ndarray
    gt_logmag: float
    gt_magnitude: float
    gt_rotvec: np.ndarray
    bucket_name: str
    bucket_index: int


def compute_bucket_stats(magnitudes: Sequence[float]) -> Dict[str, Any]:
    mags = np.asarray(list(magnitudes), dtype=np.float64)
    near_thr = float(np.percentile(mags, 10))
    small_thr = float(np.percentile(mags, 25))
    large_thr = float(np.percentile(mags, 75))

    def _stats(sub: np.ndarray) -> Dict[str, float]:
        if sub.size == 0:
            return {"count": 0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "clip_lo": 1e-6, "clip_hi": 1e-6}
        return {
            "count": int(sub.size),
            "median": float(np.percentile(sub, 50)),
            "p90": float(np.percentile(sub, 90)),
            "p95": float(np.percentile(sub, 95)),
            "p99": float(np.percentile(sub, 99)),
            "max": float(np.max(sub)),
            "clip_lo": float(max(np.percentile(sub, 5), 1e-6)),
            "clip_hi": float(np.percentile(sub, 95)),
        }

    near = mags <= near_thr
    small = (mags > near_thr) & (mags <= small_thr)
    normal = (mags > small_thr) & (mags < large_thr)
    large = mags >= large_thr
    return {
        "near_static_threshold": near_thr,
        "small_motion_threshold": small_thr,
        "large_motion_threshold": large_thr,
        "global": _stats(mags),
        "near_static": _stats(mags[near]),
        "small_motion": _stats(mags[small]),
        "normal_motion": _stats(mags[normal]),
        "large_motion": _stats(mags[large]),
    }


def assign_bucket(mag: float, stats: Dict[str, Any]) -> Tuple[str, int]:
    if mag <= stats["near_static_threshold"]:
        return "near_static", 0
    if mag <= stats["small_motion_threshold"]:
        return "small_motion", 1
    if mag >= stats["large_motion_threshold"]:
        return "large_motion", 3
    return "normal_motion", 2


def make_samples() -> Tuple[List[S5E9Sample], List[S5E9Sample], Dict[str, Any], Dict[str, Any]]:
    frames = scan_frames(Path("data"))
    train_keys, _test_keys, split_summary = split_sequence_keys(list(frames.keys()))
    train_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        train_pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(train_pairs) * 0.1))) if train_pairs else 0
    core_train, core_val = train_pairs[:-val_count], train_pairs[-val_count:]

    def _basic(pairs):
        out = []
        for a, b in pairs:
            R, t = relative_pose_A_to_B_in_B(a, b)
            mag = float(np.linalg.norm(t))
            direction = t / max(mag, 1e-12)
            out.append((a, b, direction.astype(np.float32), mag, rotation_matrix_to_rotvec(R).astype(np.float32)))
        return out

    train_basic = _basic(core_train)
    val_basic = _basic(core_val)
    stats = compute_bucket_stats([r[3] for r in train_basic + val_basic])

    def _pack(rows) -> List[S5E9Sample]:
        out: List[S5E9Sample] = []
        total = max(len(rows), 1)
        for idx, (a, b, direction, mag, rotvec) in enumerate(rows):
            bucket_name, bucket_index = assign_bucket(mag, stats)
            out.append(S5E9Sample(a, b, idx, total, direction, float(np.log(max(mag, 1e-12))), mag, rotvec, bucket_name, bucket_index))
        return out

    return _pack(train_basic), _pack(val_basic), split_summary, stats


def build_feature_lookups(samples: Sequence[S5E9Sample]) -> Dict[str, Any]:
    mags = [s.gt_magnitude for s in samples]
    return compute_bucket_stats(mags)
