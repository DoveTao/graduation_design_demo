#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


K_VALUES = [1, 2, 3, 5]
STATUSES = [
    "USABLE_FOR_TRAIN360",
    "USABLE_PARTIAL_HIGH_YIELD",
    "QUARANTINE_EMPTY_IMAGE_DIR",
    "QUARANTINE_IMAGE_MISSING",
    "QUARANTINE_POSE_MISSING",
    "QUARANTINE_TIMESTAMP_MISSING",
    "QUARANTINE_LOW_PAIR_COUNT",
    "QUARANTINE_UNKNOWN_FORMAT",
]


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg")))


def _collect_all_sequences(local_root: Path) -> List[str]:
    names = set()
    for path in (local_root / "Sequences").glob("*"):
        if path.is_dir():
            names.add(path.name)
    for path in (local_root / "GroundTruth").glob("*.txt"):
        names.add(path.stem)
    for path in (local_root / "Timestamps").glob("*.txt"):
        names.add(path.name.replace("_timestamps.txt", "").replace(".txt", ""))
    return sorted(names)


def _load_dset2b_usage(manifest_dir: Path) -> tuple[Dict[str, int], Dict[str, List[str]]]:
    counts: Dict[str, int] = {}
    used_by: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        for row in _read_jsonl(manifest_dir / f"pair_manifest_{split}.jsonl"):
            seq = str(row.get("seq_id", ""))
            counts[seq] = counts.get(seq, 0) + 1
            used_by.setdefault(seq, [])
            if split not in used_by[seq]:
                used_by[seq].append(split)
    return counts, used_by


def _recommend_status(
    image_dir_exists: bool,
    image_count: int,
    pose_file_exists: bool,
    pose_row_count: int,
    timestamp_file_exists: bool,
    timestamp_count: int,
    valid_adjacent_pairs_possible: int,
    valid_kstep_pairs_possible: int,
    hygiene: Dict[str, Any],
) -> str:
    if not image_dir_exists and not pose_file_exists and not timestamp_file_exists:
        return "QUARANTINE_UNKNOWN_FORMAT"
    if image_dir_exists and image_count == 0:
        return "QUARANTINE_EMPTY_IMAGE_DIR"
    if image_count == 0:
        return "QUARANTINE_IMAGE_MISSING"
    if not pose_file_exists or pose_row_count == 0:
        return "QUARANTINE_POSE_MISSING"
    if not timestamp_file_exists or timestamp_count == 0:
        return "QUARANTINE_TIMESTAMP_MISSING"

    min_images = int(hygiene.get("min_images_for_train_sequence", 80))
    min_pose_rows = int(hygiene.get("min_pose_rows_for_train_sequence", 80))
    min_adj = int(hygiene.get("min_valid_adjacent_pairs", 50))
    min_total = int(hygiene.get("min_valid_total_pairs", 200))
    valid_total = valid_adjacent_pairs_possible + valid_kstep_pairs_possible
    usable = min(image_count, pose_row_count, timestamp_count)

    if (
        image_count >= min_images
        and pose_row_count >= min_pose_rows
        and timestamp_count >= min_pose_rows
        and valid_adjacent_pairs_possible >= min_adj
        and valid_total >= min_total
        and usable >= min_images
    ):
        return "USABLE_FOR_TRAIN360"

    if (
        bool(hygiene.get("allow_partial_if_valid_pairs_high", True))
        and valid_adjacent_pairs_possible >= min_adj
        and valid_total >= min_total
    ):
        return "USABLE_PARTIAL_HIGH_YIELD"

    return "QUARANTINE_LOW_PAIR_COUNT"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    manifest_dir = Path(str(cfg.get("source", {}).get("dset2b_manifest_dir", "")))
    dset2b_checkpoint = _read_json(Path(str(cfg.get("source", {}).get("dset2b_checkpoint", ""))))
    pair_counts, used_by = _load_dset2b_usage(manifest_dir)

    sequences: Dict[str, Any] = {}
    status_counts = {status: 0 for status in STATUSES}
    for seq in _collect_all_sequences(local_root):
        image_dir = local_root / "Sequences" / seq
        pose_file = local_root / "GroundTruth" / f"{seq}.txt"
        timestamp_file = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        image_dir_exists = image_dir.exists() and image_dir.is_dir()
        image_count = _image_count(image_dir)
        pose_file_exists = pose_file.exists()
        pose_row_count = _line_count(pose_file)
        timestamp_file_exists = timestamp_file.exists()
        timestamp_count = _line_count(timestamp_file)
        usable = min(image_count, pose_row_count, timestamp_count)
        valid_adjacent = max(usable - 1, 0)
        valid_kstep = int(sum(max(usable - k, 0) for k in K_VALUES if k > 1))
        status = _recommend_status(
            image_dir_exists=image_dir_exists,
            image_count=image_count,
            pose_file_exists=pose_file_exists,
            pose_row_count=pose_row_count,
            timestamp_file_exists=timestamp_file_exists,
            timestamp_count=timestamp_count,
            valid_adjacent_pairs_possible=valid_adjacent,
            valid_kstep_pairs_possible=valid_kstep,
            hygiene=cfg.get("hygiene", {}),
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        sequences[seq] = {
            "sequence_name": seq,
            "image_dir_exists": image_dir_exists,
            "image_count": image_count,
            "pose_file_exists": pose_file_exists,
            "pose_row_count": pose_row_count,
            "timestamp_file_exists": timestamp_file_exists,
            "timestamp_count": timestamp_count,
            "has_empty_image_dir": image_dir_exists and image_count == 0,
            "has_images_no_pose": image_count > 0 and not pose_file_exists,
            "has_pose_no_images": pose_file_exists and image_count == 0,
            "has_timestamps_no_images": timestamp_file_exists and image_count == 0,
            "image_pose_count_ratio": float(image_count / max(pose_row_count, 1)),
            "valid_adjacent_pairs_possible": valid_adjacent,
            "valid_kstep_pairs_possible": valid_kstep,
            "current_dset2b_manifest_pairs": int(pair_counts.get(seq, 0)),
            "used_by_dset2b_train_val_test": used_by.get(seq, []),
            "recommended_status": status,
        }

    payload = {
        "experiment": "DSET2C_360DVO_dataset_hygiene",
        "dataset": {
            "name": cfg.get("dataset", {}).get("name", "360DVO"),
            "local_root": str(local_root),
        },
        "dset2b_summary": {
            "final_classification": dset2b_checkpoint.get("final_classification"),
            "manifest_sequences": dset2b_checkpoint.get("manifest", {}).get("sequences_all", []),
        },
        "status_counts": status_counts,
        "sequences": dict(sorted(sequences.items())),
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
