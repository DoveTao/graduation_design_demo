#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import quat_xyzw_to_rot, write_json


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _numeric_columns(line: str) -> List[float]:
    import re
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _read_gt_and_ts(gt_path: Path, ts_path: Path) -> List[Dict[str, Any]]:
    gt_lines = [ln.strip() for ln in gt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ts_lines = [ln.strip() for ln in ts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows: List[Dict[str, Any]] = []
    for idx, (g, ts) in enumerate(zip(gt_lines, ts_lines)):
        cols = _numeric_columns(g)
        if len(cols) < 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = cols[:7]
        rows.append(
            {
                "timestamp": float(ts),
                "R_w_c": quat_xyzw_to_rot(qx, qy, qz, qw),
                "t_w_c": np.asarray([tx, ty, tz], dtype=np.float64),
                "frame_index": idx + 1,
            }
        )
    return rows


def _relative_pose(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R_BA = b["R_w_c"].T @ a["R_w_c"]
    t_BA_B = b["R_w_c"].T @ (a["t_w_c"] - b["t_w_c"])
    return R_BA, t_BA_B


def _rot_to_list(R: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in R.tolist()]


def _auto_split(seqs: Sequence[str]) -> Dict[str, List[str]]:
    seqs = sorted(seqs)
    if len(seqs) >= 5:
        return {"train": seqs[:3], "val": [seqs[3]], "test": [seqs[4]]}
    if len(seqs) == 4:
        return {"train": seqs[:2], "val": [seqs[2]], "test": [seqs[3]]}
    if len(seqs) == 3:
        return {"train": [seqs[0]], "val": [seqs[1]], "test": [seqs[2]]}
    if len(seqs) == 2:
        return {"train": [seqs[0]], "val": [], "test": [seqs[1]]}
    return {"train": [], "val": [], "test": list(seqs)}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    # DSET1 builds a multi sequence manifest with split-by-sequence semantics.
    cfg = _load_cfg(Path(args.config))
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted([p.name for p in (local_root / "Sequences").glob("*") if p.is_dir()])
    split_cfg = cfg.get("split", {})
    if split_cfg.get("train_sequences") or split_cfg.get("val_sequences") or split_cfg.get("test_sequences"):
        split = {
            "train": list(split_cfg.get("train_sequences", [])),
            "val": list(split_cfg.get("val_sequences", [])),
            "test": list(split_cfg.get("test_sequences", [])),
        }
    else:
        split = _auto_split(sequences)

    max_pairs_per_sequence = int(cfg.get("pair_manifest", {}).get("max_pairs_per_sequence", 1000))
    k_values = [int(x) for x in cfg.get("pair_manifest", {}).get("k_values", [1, 2, 3, 5])]
    all_rows: List[Dict[str, Any]] = []
    sequence_summaries: List[Dict[str, Any]] = []
    pair_index = 0
    for seq in sequences:
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        img_dir = local_root / "Sequences" / seq
        images = sorted(img_dir.glob("*.jpg"))
        pose_rows = _read_gt_and_ts(gt_path, ts_path) if gt_path.exists() and ts_path.exists() else []
        seq_split = "smoke"
        for name in ["train", "val", "test"]:
            if seq in split[name]:
                seq_split = name
                break
        built = 0
        for i in range(len(pose_rows) - 1):
            for k in k_values:
                if i + k >= len(pose_rows):
                    continue
                if built >= max_pairs_per_sequence:
                    break
                a = pose_rows[i]
                b = pose_rows[i + k]
                R_BA, t_BA_B = _relative_pose(a, b)
                tmag = float(np.linalg.norm(t_BA_B))
                tdir = (t_BA_B / max(tmag, 1.0e-12)).tolist()
                pair_type = "adjacent" if k == 1 else "kstep"
                img_a = images[i] if i < len(images) else None
                img_b = images[i + k] if (i + k) < len(images) else None
                if img_a is None or img_b is None:
                    continue
                all_rows.append(
                    {
                        "dataset": "360DVO",
                        "seq_id": seq,
                        "split": seq_split,
                        "pair_index": pair_index,
                        "pair_type": pair_type,
                        "k": int(k),
                        "image_path_a": str(img_a),
                        "image_path_b": str(img_b),
                        "timestamp_a": float(a["timestamp"]),
                        "timestamp_b": float(b["timestamp"]),
                        "T_w_a": {"R": _rot_to_list(a["R_w_c"]), "t": [float(x) for x in a["t_w_c"].tolist()]},
                        "T_w_b": {"R": _rot_to_list(b["R_w_c"]), "t": [float(x) for x in b["t_w_c"].tolist()]},
                        "R_BA": _rot_to_list(R_BA),
                        "t_BA_B": [float(x) for x in t_BA_B.tolist()],
                        "tdir_B": [float(x) for x in tdir],
                        "tmag": tmag,
                        "valid_pose": True,
                        "valid_timestamp": True,
                    }
                )
                pair_index += 1
                built += 1
            if built >= max_pairs_per_sequence:
                break
        sequence_summaries.append(
            {
                "seq_id": seq,
                "split": seq_split,
                "num_images": len(images),
                "num_pose_rows": len(pose_rows),
                "pairs_built": built,
            }
        )

    split_rows = {
        "train": [r for r in all_rows if r["split"] == "train"],
        "val": [r for r in all_rows if r["split"] == "val"],
        "test": [r for r in all_rows if r["split"] == "test"],
    }
    _write_jsonl(out_dir / "pair_manifest_all.jsonl", all_rows)
    _write_jsonl(out_dir / "pair_manifest_train.jsonl", split_rows["train"])
    _write_jsonl(out_dir / "pair_manifest_val.jsonl", split_rows["val"])
    _write_jsonl(out_dir / "pair_manifest_test.jsonl", split_rows["test"])

    summary = {
        "dataset": "360DVO",
        "local_root": str(local_root),
        "selected_sequences": sequences,
        "num_sequences": len(sequences),
        "split": split,
        "split_by_sequence": bool(split_cfg.get("split_by_sequence", True)),
        "forbid_random_pair_split": bool(split_cfg.get("forbid_random_pair_split", True)),
        "num_pairs_all": len(all_rows),
        "num_pairs_train": len(split_rows["train"]),
        "num_pairs_val": len(split_rows["val"]),
        "num_pairs_test": len(split_rows["test"]),
        "sequence_summaries": sequence_summaries,
        "output_convention": cfg.get("pair_manifest", {}).get("output_convention", {}),
        "insufficient_sequences": len(sequences) < int(cfg.get("sequence_selection", {}).get("min_sequences_required", 3)),
    }
    write_json(out_dir / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
