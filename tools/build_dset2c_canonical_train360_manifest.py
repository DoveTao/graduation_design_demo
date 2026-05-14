#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import quat_xyzw_to_rot, write_json


ALLOWED_STATUSES = {"USABLE_FOR_TRAIN360", "USABLE_PARTIAL_HIGH_YIELD"}


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _numeric_columns(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _frame_idx(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def _collect_images(img_dir: Path) -> List[Path]:
    keyed = {}
    for path in sorted(img_dir.glob("*.jpg")):
        idx = _frame_idx(path)
        if idx is not None:
            keyed[idx] = path
    return [keyed[idx] for idx in sorted(keyed)]


def _read_rows(gt_path: Path, ts_path: Path, images: Sequence[Path]) -> List[Dict[str, Any]]:
    gt_lines = [line.strip() for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ts_lines = [line.strip() for line in ts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = min(len(gt_lines), len(ts_lines), len(images))
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        cols = _numeric_columns(gt_lines[idx])
        if len(cols) < 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = cols[:7]
        rows.append(
            {
                "timestamp": float(ts_lines[idx]),
                "R_w_c": quat_xyzw_to_rot(qx, qy, qz, qw),
                "t_w_c": np.asarray([tx, ty, tz], dtype=np.float64),
                "image_path": str(images[idx]),
            }
        )
    return rows


def _relative_pose(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R_BA = b["R_w_c"].T @ a["R_w_c"]
    t_BA_B = b["R_w_c"].T @ (a["t_w_c"] - b["t_w_c"])
    return R_BA, t_BA_B


def _rot_to_list(R: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in R.tolist()]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _load_dset2b_split(manifest_dir: Path) -> Dict[str, List[str]]:
    out = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        seqs = []
        for row in _read_jsonl(manifest_dir / f"pair_manifest_{split}.jsonl"):
            seq = str(row.get("seq_id", ""))
            if seq and seq not in seqs:
                seqs.append(seq)
        out[split] = seqs
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    audit = _read_json(Path(args.local_tree_audit))
    sequences = audit.get("sequences", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    manifest_dir = Path(str(cfg.get("source", {}).get("dset2b_manifest_dir", "")))
    dset2b_split = _load_dset2b_split(manifest_dir)
    k_values = [int(x) for x in cfg.get("canonical_manifest", {}).get("k_values", [1, 2, 3, 5])]

    usable = {seq for seq, rec in sequences.items() if rec.get("recommended_status") in ALLOWED_STATUSES}
    canonical_split = {
        split: [seq for seq in dset2b_split.get(split, []) if seq in usable]
        for split in ["train", "val", "test"]
    }
    excluded = {}
    canonical_set = set(canonical_split["train"] + canonical_split["val"] + canonical_split["test"])
    for seq, rec in sequences.items():
        if seq in canonical_set:
            continue
        status = str(rec.get("recommended_status"))
        if status in ALLOWED_STATUSES:
            reason = "NOT_IN_DSET2B_SPLIT"
        else:
            reason = status
        excluded[seq] = {
            "recommended_status": status,
            "exclude_reason": reason,
            "used_by_dset2b_train_val_test": rec.get("used_by_dset2b_train_val_test", []),
            "current_dset2b_manifest_pairs": rec.get("current_dset2b_manifest_pairs", 0),
        }

    all_rows: List[Dict[str, Any]] = []
    pair_index = 0
    counts = {
        "train": {"adjacent": 0, "kstep": 0},
        "val": {"adjacent": 0, "kstep": 0},
        "test": {"adjacent": 0, "kstep": 0},
    }
    for split in ["train", "val", "test"]:
        for seq in canonical_split[split]:
            images = _collect_images(local_root / "Sequences" / seq)
            gt_path = local_root / "GroundTruth" / f"{seq}.txt"
            ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
            rows = _read_rows(gt_path=gt_path, ts_path=ts_path, images=images)
            for idx in range(len(rows) - 1):
                for k in k_values:
                    if idx + k >= len(rows):
                        continue
                    a = rows[idx]
                    b = rows[idx + k]
                    R_BA, t_BA_B = _relative_pose(a, b)
                    tmag = float(np.linalg.norm(t_BA_B))
                    tdir = (t_BA_B / max(tmag, 1.0e-12)).tolist()
                    pair_type = "adjacent" if k == 1 else "kstep"
                    all_rows.append(
                        {
                            "dataset": "360DVO",
                            "seq_id": seq,
                            "split": split,
                            "pair_index": pair_index,
                            "pair_type": pair_type,
                            "k": int(k),
                            "image_path_a": a["image_path"],
                            "image_path_b": b["image_path"],
                            "timestamp_a": float(a["timestamp"]),
                            "timestamp_b": float(b["timestamp"]),
                            "R_BA": _rot_to_list(R_BA),
                            "t_BA_B": [float(x) for x in t_BA_B.tolist()],
                            "tdir_B": [float(x) for x in tdir],
                            "tmag": tmag,
                            "valid_pose": True,
                            "valid_timestamp": True,
                        }
                    )
                    pair_index += 1
                    counts[split][pair_type] += 1

    split_rows = {split: [row for row in all_rows if row["split"] == split] for split in ["train", "val", "test"]}
    _write_jsonl(out_dir / "pair_manifest_all.jsonl", all_rows)
    _write_jsonl(out_dir / "pair_manifest_train.jsonl", split_rows["train"])
    _write_jsonl(out_dir / "pair_manifest_val.jsonl", split_rows["val"])
    _write_jsonl(out_dir / "pair_manifest_test.jsonl", split_rows["test"])

    summary = {
        "num_sequences": len(canonical_split["train"] + canonical_split["val"] + canonical_split["test"]),
        "sequences_all": canonical_split["train"] + canonical_split["val"] + canonical_split["test"],
        "train_sequences": canonical_split["train"],
        "val_sequences": canonical_split["val"],
        "test_sequences": canonical_split["test"],
        "num_pairs_all": len(all_rows),
        "num_pairs_train": len(split_rows["train"]),
        "num_pairs_val": len(split_rows["val"]),
        "num_pairs_test": len(split_rows["test"]),
        "num_adjacent_pairs_train": counts["train"]["adjacent"],
        "num_adjacent_pairs_val": counts["val"]["adjacent"],
        "num_adjacent_pairs_test": counts["test"]["adjacent"],
        "num_kstep_pairs_train": counts["train"]["kstep"],
        "num_kstep_pairs_val": counts["val"]["kstep"],
        "num_kstep_pairs_test": counts["test"]["kstep"],
        "split_by_sequence": True,
        "forbid_random_pair_split": True,
        "pose_convention": "R_BA_t_BA_B_tdir_B",
        "manifest_ready": bool(all_rows),
        "reused_dset2b_split": True,
    }
    write_json(out_dir / "canonical_manifest_summary.json", summary)
    write_json(out_dir / "excluded_sequences.json", excluded)
    return {"summary": summary, "excluded": excluded}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-tree-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
