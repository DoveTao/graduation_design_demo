#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


DATA_NOT_FOUND = "DATA_NOT_FOUND"
DOWNLOAD_REQUIRED = "DOWNLOAD_REQUIRED"
POSE_CONVENTION_UNKNOWN = "POSE_CONVENTION_UNKNOWN"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSE_HINTS = ("pose", "traj", "trajectory", "odom", "gt", "groundtruth")
TIME_HINTS = ("time", "timestamp")


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob("*")


def _looks_numeric_stem(path: Path) -> bool:
    return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", path.stem))


def _find_sequence_dirs(root: Path) -> List[Path]:
    seq_dirs: List[Tuple[int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if images:
            seq_dirs.append((len(images), path))
    seq_dirs.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in seq_dirs]


def _sample_resolution(images: List[Path]) -> Optional[List[int]]:
    for img_path in images[:3]:
        try:
            with Image.open(img_path) as img:
                return [int(img.width), int(img.height)]
        except Exception:
            continue
    return None


def _numeric_columns(line: str) -> List[float]:
    parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return [float(x) for x in parts]


def _detect_pose_format(path: Path) -> Tuple[str, str, str]:
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    except Exception:
        return "unreadable", "unknown", "unknown"
    if not lines:
        return "empty", "unknown", "unknown"
    cols = [_numeric_columns(ln) for ln in lines[:5]]
    widths = {len(c) for c in cols if c}
    if 8 in widths:
        return "tum_like_timestamp_tx_ty_tz_qx_qy_qz_qw", "xyzw", "T_w_c_or_T_c_w_unknown_until_verified"
    if 7 in widths:
        return "seven_column_pose_or_quaternion_unknown", "unknown", "unknown"
    if 12 in widths:
        return "3x4_matrix_rows_flattened", "not_applicable", "unknown"
    if 16 in widths:
        return "4x4_matrix_rows_flattened", "not_applicable", "unknown"
    return "unknown", "unknown", "unknown"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(Path(args.config))
    dataset_cfg = config.get("dataset", {})
    root = Path(str(dataset_cfg.get("local_root", ""))).expanduser()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "dataset": "360DVO",
        "dataset_found": root.exists(),
        "local_root": str(root),
        "sequences_found": [],
        "image_count_by_sequence": {},
        "pose_files_found": [],
        "timestamp_files_found": [],
        "sample_image_resolution": None,
        "pose_format_detected": "unknown",
        "quaternion_order_detected": "unknown",
        "inspection_ready": False,
        "blockers": [],
    }

    if not root.exists():
        payload["blockers"] = [DATA_NOT_FOUND, DOWNLOAD_REQUIRED]
        write_json(out_path, payload)
        return payload

    seq_dirs = _find_sequence_dirs(root)
    images_by_seq: Dict[str, List[Path]] = {}
    pose_files: List[Path] = []
    time_files: List[Path] = []
    for path in _iter_files(root):
        if not isinstance(path, Path):
            continue
        if path.is_file():
            suffix = path.suffix.lower()
            name_l = path.name.lower()
            if suffix in IMAGE_EXTS:
                continue
            if suffix in {".txt", ".csv", ".json", ".tum", ".log"} and any(h in name_l for h in POSE_HINTS):
                pose_files.append(path)
            if suffix in {".txt", ".csv", ".json"} and any(h in name_l for h in TIME_HINTS):
                time_files.append(path)

    for seq_dir in seq_dirs:
        seq_id = str(seq_dir.relative_to(root))
        images = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        images_by_seq[seq_id] = images
        payload["sequences_found"].append(seq_id)
        payload["image_count_by_sequence"][seq_id] = len(images)

    flat_images = [img for imgs in images_by_seq.values() for img in imgs]
    payload["sample_image_resolution"] = _sample_resolution(flat_images)
    payload["pose_files_found"] = [str(p) for p in sorted(pose_files)[:50]]
    payload["timestamp_files_found"] = [str(p) for p in sorted(time_files)[:50]]

    pose_fmt = "unknown"
    quat_order = "unknown"
    pose_conv = "unknown"
    if pose_files:
        pose_fmt, quat_order, pose_conv = _detect_pose_format(sorted(pose_files)[0])
    payload["pose_format_detected"] = pose_fmt
    payload["quaternion_order_detected"] = quat_order
    payload["source_pose_convention_detected"] = pose_conv

    if not payload["sequences_found"]:
        payload["blockers"].append("NO_SEQUENCE_WITH_IMAGES")
    if payload["sample_image_resolution"] is None:
        payload["blockers"].append("IMAGE_RESOLUTION_NOT_DETECTED")
    if not pose_files:
        payload["blockers"].append("POSE_FILE_NOT_FOUND")
    if pose_fmt == "unknown":
        payload["blockers"].append(POSE_CONVENTION_UNKNOWN)
    if not time_files:
        payload["blockers"].append("TIMESTAMP_FILE_NOT_FOUND")

    payload["inspection_ready"] = bool(
        payload["dataset_found"]
        and payload["sequences_found"]
        and payload["sample_image_resolution"] is not None
        and pose_fmt != "unknown"
    )
    write_json(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
