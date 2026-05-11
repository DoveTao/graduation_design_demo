#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image

from s5e2_adjacent_dense_lib import read_timestamps, read_tum, write_json
from s5e7_direction_scale_lib import build_eval_seq_frames


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def load_config(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    cfg = load_simple_yaml(path)
    # structured defaults without yaml dependency
    cfg["_raw_text"] = txt
    return cfg


def parse_section_value(text: str, section: str, key: str, default: Any) -> Any:
    lines = text.splitlines()
    inside = False
    prefix = section + ":"
    for raw in lines:
        if not raw.strip():
            continue
        if not raw.startswith(" ") and raw.strip().endswith(":"):
            inside = raw.strip() == prefix
            continue
        if inside and raw.startswith("  ") and ":" in raw:
            k, v = raw.strip().split(":", 1)
            if k == key:
                val = v.strip()
                if val.lower() in {"true", "false"}:
                    return val.lower() == "true"
                try:
                    if "." in val:
                        return float(val)
                    return int(val)
                except Exception:
                    return val
    return default


def nearest(frames: List[Any], ts: float, tol: float = 1e-5) -> Optional[Any]:
    best = None
    for rec in frames:
        dt = abs(float(rec.timestamp) - float(ts))
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, rec)
    return best[1] if best else None


def gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[np.ndarray]:
    if a not in gt or b not in gt:
        return None
    return gt[b]["R"].T @ (gt[a]["t"] - gt[b]["t"])


def load_gray(path: str, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    img = Image.open(path).convert("L").resize((w, h))
    return np.asarray(img, dtype=np.uint8)


def flow_angle_stats(vecs: np.ndarray) -> Tuple[float, float]:
    if vecs.size == 0:
        return 0.0, 180.0
    ang = np.rad2deg(np.arctan2(vecs[:, 1], vecs[:, 0]))
    median = float(np.percentile(ang, 50))
    disp = float(np.mean(np.abs(((ang - median + 180.0) % 360.0) - 180.0)))
    return median, disp


def extract_orb_matches(img_i: np.ndarray, img_j: np.ndarray, nfeatures: int, fast_threshold: int, ratio_test: float, cross_check: bool) -> Dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=int(nfeatures), fastThreshold=int(fast_threshold))
    kp_i, des_i = orb.detectAndCompute(img_i, None)
    kp_j, des_j = orb.detectAndCompute(img_j, None)
    out: Dict[str, Any] = {
        "detector_type": "ORB",
        "keypoint_count_i": len(kp_i) if kp_i is not None else 0,
        "keypoint_count_j": len(kp_j) if kp_j is not None else 0,
        "raw_match_count": 0,
        "filtered_match_count": 0,
        "inlier_match_count": 0,
        "inlier_ratio": 0.0,
        "median_match_displacement": 0.0,
        "p90_match_displacement": 0.0,
        "match_angle_median": 0.0,
        "match_angle_dispersion": 180.0,
        "parallax_proxy": 0.0,
        "low_parallax_flag": True,
        "correspondence_success": False,
        "correspondence_failure_reason": "no_descriptors",
        "match_points_i": [],
        "match_points_j": [],
    }
    if des_i is None or des_j is None or len(kp_i) == 0 or len(kp_j) == 0:
        return out

    if cross_check:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_i, des_j)
        good = sorted(matches, key=lambda m: m.distance)
        out["raw_match_count"] = len(good)
        filtered = good
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(des_i, des_j, k=2)
        out["raw_match_count"] = len(knn)
        filtered = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < float(ratio_test) * n.distance:
                filtered.append(m)

    out["filtered_match_count"] = len(filtered)
    if len(filtered) < 4:
        out["correspondence_failure_reason"] = "too_few_filtered_matches"
        return out

    pts_i = np.float32([kp_i[m.queryIdx].pt for m in filtered])
    pts_j = np.float32([kp_j[m.trainIdx].pt for m in filtered])
    disp = pts_j - pts_i
    mags = np.linalg.norm(disp, axis=1)
    ang_med, ang_disp = flow_angle_stats(disp)
    out["median_match_displacement"] = float(np.percentile(mags, 50))
    out["p90_match_displacement"] = float(np.percentile(mags, 90))
    out["match_angle_median"] = ang_med
    out["match_angle_dispersion"] = ang_disp
    out["parallax_proxy"] = float(np.percentile(mags, 50) * max(1e-6, 1.0 - min(ang_disp / 180.0, 1.0)))

    inlier_mask = None
    if len(filtered) >= 8:
        try:
            _F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.5, 0.99)
            if mask is not None:
                inlier_mask = mask.reshape(-1).astype(bool)
        except cv2.error:
            inlier_mask = None
    if inlier_mask is None:
        inlier_mask = np.ones(len(filtered), dtype=bool)

    out["inlier_match_count"] = int(inlier_mask.sum())
    out["inlier_ratio"] = float(inlier_mask.sum() / max(len(filtered), 1))
    out["correspondence_success"] = bool(out["inlier_match_count"] >= 8)
    out["correspondence_failure_reason"] = "" if out["correspondence_success"] else "insufficient_inliers"
    out["match_points_i"] = pts_i.tolist()
    out["match_points_j"] = pts_j.tolist()
    return out


def extract_sparse_flow(img_i: np.ndarray, img_j: np.ndarray, max_corners: int, quality_level: float, min_distance: float, win_size: int, max_level: int) -> Dict[str, Any]:
    pts = cv2.goodFeaturesToTrack(img_i, maxCorners=int(max_corners), qualityLevel=float(quality_level), minDistance=float(min_distance))
    out = {
        "flow_success": False,
        "flow_failure_reason": "no_corners",
        "tracked_point_count": 0,
        "valid_flow_count": 0,
        "median_flow_magnitude": 0.0,
        "p90_flow_magnitude": 0.0,
        "flow_angle_median": 0.0,
        "flow_angle_dispersion": 180.0,
        "forward_backward_flow_consistency": None,
    }
    if pts is None or len(pts) == 0:
        return out
    p1, st, _err = cv2.calcOpticalFlowPyrLK(img_i, img_j, pts, None, winSize=(int(win_size), int(win_size)), maxLevel=int(max_level))
    p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(img_j, img_i, p1, None, winSize=(int(win_size), int(win_size)), maxLevel=int(max_level))
    good = (st.reshape(-1) > 0) & (st_back.reshape(-1) > 0)
    if not np.any(good):
        out["tracked_point_count"] = int(len(pts))
        out["flow_failure_reason"] = "no_valid_tracks"
        return out
    q0 = pts.reshape(-1, 2)[good]
    q1 = p1.reshape(-1, 2)[good]
    q0r = p0r.reshape(-1, 2)[good]
    vec = q1 - q0
    mags = np.linalg.norm(vec, axis=1)
    fb = np.linalg.norm(q0r - q0, axis=1)
    ang_med, ang_disp = flow_angle_stats(vec)
    out.update(
        {
            "flow_success": True,
            "flow_failure_reason": "",
            "tracked_point_count": int(len(pts)),
            "valid_flow_count": int(np.sum(good)),
            "median_flow_magnitude": float(np.percentile(mags, 50)),
            "p90_flow_magnitude": float(np.percentile(mags, 90)),
            "flow_angle_median": ang_med,
            "flow_angle_dispersion": ang_disp,
            "forward_backward_flow_consistency": float(np.mean(fb < 1.5)),
        }
    )
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    text = cfg["_raw_text"]
    scene = cfg.get("scene", "scene01")
    seq = cfg.get("seq", "seq03")
    timestamps = read_timestamps(Path(cfg.get("timestamps", "")))
    gt = read_tum(Path(cfg.get("groundtruth", "")))
    frames = build_eval_seq_frames(scene, seq)
    out_dir = Path(cfg.get("output_dir", "external_baselines/results/s5e12_feature_gate"))
    out_json = Path(cfg.get("correspondence_json", out_dir / "correspondence_features.json"))
    out_csv = Path(cfg.get("correspondence_csv", out_dir / "correspondence_features.csv"))

    w = int(parse_section_value(text, "image", "resized_width", 640))
    h = int(parse_section_value(text, "image", "resized_height", 320))
    nfeatures = int(parse_section_value(text, "detector", "nfeatures", 1200))
    fast_threshold = int(parse_section_value(text, "detector", "fastThreshold", 12))
    ratio_test = float(parse_section_value(text, "matcher", "ratio_test", 0.75))
    cross_check = bool(parse_section_value(text, "matcher", "cross_check", False))
    max_corners = int(parse_section_value(text, "flow", "max_corners", 600))
    quality_level = float(parse_section_value(text, "flow", "quality_level", 0.01))
    min_distance = float(parse_section_value(text, "flow", "min_distance", 7))
    win_size = int(parse_section_value(text, "flow", "win_size", 21))
    max_level = int(parse_section_value(text, "flow", "max_level", 3))
    low_parallax_thr = float(parse_section_value(text, "observability", "low_parallax_displacement", 2.0))
    near_static_disp_thr = float(parse_section_value(text, "observability", "near_static_displacement", 0.75))
    near_static_diff_thr = float(parse_section_value(text, "observability", "near_static_image_diff_mean", 0.02))
    min_filtered = int(parse_section_value(text, "observability", "min_filtered_matches", 12))
    min_inliers = int(parse_section_value(text, "observability", "min_inlier_matches", 8))

    records: List[Dict[str, Any]] = []
    for edge_id in range(len(timestamps) - 1):
        fi = nearest(frames, timestamps[edge_id])
        fj = nearest(frames, timestamps[edge_id + 1])
        if fi is None or fj is None:
            records.append(
                {
                    "edge_id": edge_id,
                    "image_i": None,
                    "image_j": None,
                    "correspondence_success": False,
                    "correspondence_failure_reason": "frame_missing",
                    "flow_success": False,
                    "flow_failure_reason": "frame_missing",
                }
            )
            continue
        img_i = load_gray(fi.image_path, (w, h))
        img_j = load_gray(fj.image_path, (w, h))
        match = extract_orb_matches(img_i, img_j, nfeatures, fast_threshold, ratio_test, cross_check)
        flow = extract_sparse_flow(img_i, img_j, max_corners, quality_level, min_distance, win_size, max_level)
        diff = np.abs(img_j.astype(np.float32) - img_i.astype(np.float32)) / 255.0
        image_difference_mean = float(np.mean(diff))
        image_difference_p90 = float(np.percentile(diff, 90))
        median_disp = max(match["median_match_displacement"], flow["median_flow_magnitude"])
        p90_disp = max(match["p90_match_displacement"], flow["p90_flow_magnitude"])
        low_parallax_flag = bool(p90_disp < low_parallax_thr)
        near_static_flag = bool(median_disp < near_static_disp_thr and image_difference_mean < near_static_diff_thr)
        success = bool(
            (match["filtered_match_count"] >= min_filtered and match["inlier_match_count"] >= min_inliers)
            or (flow["flow_success"] and flow["valid_flow_count"] >= min_inliers)
        )
        failure_reason = ""
        if not success:
            if match["filtered_match_count"] < min_filtered:
                failure_reason = "filtered_matches_too_few"
            elif match["inlier_match_count"] < min_inliers:
                failure_reason = "inlier_matches_too_few"
            elif not flow["flow_success"]:
                failure_reason = flow["flow_failure_reason"]
            else:
                failure_reason = "correspondence_quality_too_low"
        gt_t = gt_rel(gt, timestamps[edge_id], timestamps[edge_id + 1])
        gt_tmag = None if gt_t is None else float(np.linalg.norm(gt_t))
        records.append(
            {
                "edge_id": edge_id,
                "timestamp_i": timestamps[edge_id],
                "timestamp_j": timestamps[edge_id + 1],
                "image_i": fi.image_path,
                "image_j": fj.image_path,
                "correspondence_success": success,
                "correspondence_failure_reason": failure_reason,
                "detector_type": match["detector_type"],
                "keypoint_count_i": match["keypoint_count_i"],
                "keypoint_count_j": match["keypoint_count_j"],
                "raw_match_count": match["raw_match_count"],
                "filtered_match_count": match["filtered_match_count"],
                "inlier_match_count": match["inlier_match_count"],
                "inlier_ratio": match["inlier_ratio"],
                "median_match_displacement": match["median_match_displacement"],
                "p90_match_displacement": match["p90_match_displacement"],
                "match_angle_median": match["match_angle_median"],
                "match_angle_dispersion": match["match_angle_dispersion"],
                "parallax_proxy": max(match["parallax_proxy"], flow["median_flow_magnitude"] * ((flow["forward_backward_flow_consistency"] or 0.0) if flow["flow_success"] else 0.0)),
                "low_parallax_flag": low_parallax_flag,
                "near_static_flag": near_static_flag,
                "image_difference_mean": image_difference_mean,
                "image_difference_p90": image_difference_p90,
                "flow_success": flow["flow_success"],
                "flow_failure_reason": flow["flow_failure_reason"],
                "tracked_point_count": flow["tracked_point_count"],
                "valid_flow_count": flow["valid_flow_count"],
                "median_flow_magnitude": flow["median_flow_magnitude"],
                "p90_flow_magnitude": flow["p90_flow_magnitude"],
                "flow_angle_median": flow["flow_angle_median"],
                "flow_angle_dispersion": flow["flow_angle_dispersion"],
                "forward_backward_flow_consistency": flow["forward_backward_flow_consistency"],
                "gt_tmag": gt_tmag,
            }
        )

    payload = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "scene": scene,
        "seq": seq,
        "edge_count": len(records),
        "records": records,
        "real_correspondence_features_used": True,
        "proxy_features_reused_as_real": False,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "edge_id",
                "timestamp_i",
                "timestamp_j",
                "correspondence_success",
                "correspondence_failure_reason",
                "detector_type",
                "keypoint_count_i",
                "keypoint_count_j",
                "raw_match_count",
                "filtered_match_count",
                "inlier_match_count",
                "inlier_ratio",
                "median_match_displacement",
                "p90_match_displacement",
                "match_angle_median",
                "match_angle_dispersion",
                "parallax_proxy",
                "low_parallax_flag",
                "near_static_flag",
                "image_difference_mean",
                "image_difference_p90",
                "flow_success",
                "flow_failure_reason",
                "tracked_point_count",
                "valid_flow_count",
                "median_flow_magnitude",
                "p90_flow_magnitude",
                "flow_angle_median",
                "flow_angle_dispersion",
                "forward_backward_flow_consistency",
                "gt_tmag",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(json.dumps({"edge_count": len(records), "success_count": sum(1 for r in records if r["correspondence_success"])}, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
