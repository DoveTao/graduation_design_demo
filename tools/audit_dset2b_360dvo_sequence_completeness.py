#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


K_VALUES = [1, 2, 3, 5]


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


def _frame_idx(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _valid_images(img_dir: Path) -> List[Path]:
    best: Dict[int, Path] = {}
    for path in sorted(img_dir.glob("*.jpg")):
        idx = _frame_idx(path)
        if idx is None or not _is_valid_image(path):
            continue
        prev = best.get(idx)
        if prev is None or len(path.stem) > len(prev.stem):
            best[idx] = path
    return [best[idx] for idx in sorted(best)]


def _missing_frame_gaps(imgs: Sequence[Path]) -> bool:
    indices = [idx for idx in (_frame_idx(path) for path in imgs) if idx is not None]
    return any(b - a > 1 for a, b in zip(indices, indices[1:]))


def _load_current_manifest_pair_counts(manifest_dir: Path) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for split_name in ["all", "train", "val", "test"]:
        rows = _read_jsonl(manifest_dir / f"pair_manifest_{split_name}.jsonl")
        for row in rows:
            seq = str(row.get("seq_id", ""))
            rec = counts.setdefault(seq, {"all": 0, "train": 0, "val": 0, "test": 0})
            rec[split_name] += 1
    return counts


def _all_local_sequences(local_root: Path) -> List[str]:
    names = set()
    for path in (local_root / "Sequences").glob("*"):
        if path.is_dir():
            names.add(path.name)
    for path in (local_root / "GroundTruth").glob("*.txt"):
        names.add(path.stem)
    for path in (local_root / "Timestamps").glob("*.txt"):
        names.add(path.name.replace("_timestamps.txt", "").replace(".txt", ""))
    return sorted(names)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    download_summary = _read_json(Path(args.download_summary))
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    manifest_dir = Path(str(cfg.get("current_dset2", {}).get("manifest_dir", "")))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    current_manifest_counts = _load_current_manifest_pair_counts(manifest_dir) if manifest_dir else {}
    selected_from_download = set(download_summary.get("selected_sequences_total", []))

    audit: Dict[str, Any] = {}
    for seq in _all_local_sequences(local_root):
        img_dir = local_root / "Sequences" / seq
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        imgs = _valid_images(img_dir) if img_dir.exists() else []
        image_count = len(imgs)
        pose_count = _line_count(gt_path)
        ts_count = _line_count(ts_path)
        usable = min(image_count, pose_count, ts_count)
        valid_adjacent = max(usable - 1, 0)
        valid_kstep = int(sum(max(usable - k, 0) for k in K_VALUES if k > 1))
        total_possible = valid_adjacent + valid_kstep
        current_manifest = current_manifest_counts.get(seq)
        completeness = float(usable / max(max(image_count, pose_count, ts_count), 1))
        missing_frames = _missing_frame_gaps(imgs)
        pose_limited = pose_count < image_count
        ts_limited = ts_count < min(image_count, pose_count) or ts_count == 0
        adapter_limited = bool(
            current_manifest
            and current_manifest.get("all", 0) < total_possible
            and not pose_limited
            and not ts_limited
            and not missing_frames
        )
        audit[seq] = {
            "selected_for_dset2b_download": seq in selected_from_download,
            "image_count": image_count,
            "pose_row_count": pose_count,
            "timestamp_count": ts_count,
            "first_image": imgs[0].name if imgs else None,
            "last_image": imgs[-1].name if imgs else None,
            "has_groundtruth": gt_path.exists(),
            "has_timestamps": ts_path.exists(),
            "valid_pose_fraction": completeness,
            "valid_adjacent_pairs_possible": valid_adjacent,
            "valid_kstep_pairs_possible": valid_kstep,
            "current_manifest_pairs_if_present": current_manifest,
            "missing_frames_suspected": missing_frames or image_count < pose_count,
            "pose_limited_suspected": pose_limited,
            "timestamp_limited_suspected": ts_limited,
            "adapter_limit_suspected": adapter_limited,
            "usable_frame_count": usable,
            "completeness_score": total_possible + completeness,
        }

    payload = {
        "local_root": str(local_root),
        "num_sequences_scanned": len(audit),
        "selected_sequences_total": list(download_summary.get("selected_sequences_total", [])),
        "sequences": dict(sorted(audit.items())),
    }
    write_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download-summary", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
