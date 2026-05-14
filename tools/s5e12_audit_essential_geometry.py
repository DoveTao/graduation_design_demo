#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import read_json, write_json


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dep = read_json(Path(args.dependency_json))
    camera = load_simple_yaml(Path(args.camera_yaml))
    camera_model = camera.get("camera_model", "unknown")
    strict = bool(dep.get("camera_intrinsics_available")) and camera_model not in {"equirectangular_panorama"}
    out = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "camera_intrinsics_available": bool(dep.get("camera_intrinsics_available")),
        "camera_model": camera_model,
        "strict_essential_geometry_available": strict,
        "essential_geometry_blocker": None if strict else "pinhole_intrinsics_missing_or_non_pinhole_camera_model",
        "findEssentialMat_api_available": bool(dep.get("essential_matrix_available")),
        "recoverPose_api_available": bool(dep.get("recover_pose_available")),
        "essential_axis_gt_dir_mean_angle": "unavailable",
        "essential_axis_gt_dir_p90_angle": "unavailable",
        "essential_axis_sign_ambiguity_rate": "unavailable",
        "reason": None if strict else "pinhole_intrinsics_missing_or_non_pinhole_camera_model",
    }
    write_json(Path(args.out_json), out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dependency-json", required=True)
    p.add_argument("--camera-yaml", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
