#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import quat_xyzw_to_rot, write_json


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _numeric_columns(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _frame_index_from_name(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _collect_valid_images(img_dir: Path) -> List[Path]:
    best: Dict[int, Path] = {}
    for path in sorted(img_dir.glob("*.jpg")):
        idx = _frame_index_from_name(path)
        if idx is None or not _is_valid_image(path):
            continue
        prev = best.get(idx)
        if prev is None or len(path.stem) > len(prev.stem):
            best[idx] = path
    return [best[idx] for idx in sorted(best)]


def _read_gt_and_ts(gt_path: Path, ts_path: Path, images: Sequence[Path]) -> List[Dict[str, Any]]:
    gt_lines = [ln.strip() for ln in gt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ts_lines = [ln.strip() for ln in ts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
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
                "frame_index": idx + 1,
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


def _auto_split(seqs: Sequence[str], split_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    seqs = sorted(seqs)
    train_n = min(int(split_cfg.get("target_train_sequences", 7)), max(len(seqs) - 2, 0))
    val_n = min(int(split_cfg.get("target_val_sequences", 2)), max(len(seqs) - train_n - 1, 0))
    test_n = min(int(split_cfg.get("target_test_sequences", 2)), max(len(seqs) - train_n - val_n, 0))
    if len(seqs) >= 11:
        train = seqs[:train_n]
        val = seqs[train_n:train_n + val_n]
        test = seqs[train_n + val_n:train_n + val_n + test_n]
    else:
        train = seqs[: max(int(split_cfg.get("min_train_sequences", 5)), min(train_n, len(seqs) - 2))]
        remain = seqs[len(train):]
        val = remain[: max(int(split_cfg.get("min_val_sequences", 1)), min(val_n, max(len(remain) - 1, 0)))]
        test = remain[len(val):]
    return {"train": train, "val": val, "test": test}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    download = _read_json(Path(args.download_summary))
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = list(download.get("selected_sequences_total", []))
    k_values = [int(x) for x in cfg.get("pair_manifest", {}).get("k_values", [1, 2, 3, 5])]
    max_pairs_per_sequence = int(cfg.get("pair_manifest", {}).get("max_pairs_per_sequence", 1500))

    usable_sequences: List[str] = []
    usable_rows_by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for seq in seqs:
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        img_dir = local_root / "Sequences" / seq
        images = _collect_valid_images(img_dir)
        rows = _read_gt_and_ts(gt_path, ts_path, images) if gt_path.exists() and ts_path.exists() else []
        if len(rows) >= 2:
            usable_sequences.append(seq)
            usable_rows_by_seq[seq] = rows
    split = _auto_split(usable_sequences, cfg.get("split", {}))

    all_rows: List[Dict[str, Any]] = []
    pair_index = 0
    num_adj = 0
    num_kstep = 0
    for seq in usable_sequences:
        rows = usable_rows_by_seq[seq]
        seq_split = "train" if seq in split["train"] else ("val" if seq in split["val"] else "test")
        built = 0
        for i in range(len(rows) - 1):
            for k in k_values:
                if i + k >= len(rows) or built >= max_pairs_per_sequence:
                    continue
                a = rows[i]
                b = rows[i + k]
                R_BA, t_BA_B = _relative_pose(a, b)
                tmag = float(np.linalg.norm(t_BA_B))
                tdir = (t_BA_B / max(tmag, 1.0e-12)).tolist()
                pair_type = "adjacent" if k == 1 else "kstep"
                all_rows.append(
                    {
                        "dataset": "360DVO",
                        "seq_id": seq,
                        "split": seq_split,
                        "pair_index": pair_index,
                        "pair_type": pair_type,
                        "k": int(k),
                        "image_path_a": a["image_path"],
                        "image_path_b": b["image_path"],
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
                if pair_type == "adjacent":
                    num_adj += 1
                else:
                    num_kstep += 1
            if built >= max_pairs_per_sequence:
                break

    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    _write_jsonl(out_dir / "pair_manifest_all.jsonl", all_rows)
    _write_jsonl(out_dir / "pair_manifest_train.jsonl", train_rows)
    _write_jsonl(out_dir / "pair_manifest_val.jsonl", val_rows)
    _write_jsonl(out_dir / "pair_manifest_test.jsonl", test_rows)

    summary = {
        "num_sequences": len(usable_sequences),
        "sequences_all": usable_sequences,
        "train_sequences": split["train"],
        "val_sequences": split["val"],
        "test_sequences": split["test"],
        "num_pairs_all": len(all_rows),
        "num_pairs_train": len(train_rows),
        "num_pairs_val": len(val_rows),
        "num_pairs_test": len(test_rows),
        "num_adjacent_pairs": num_adj,
        "num_kstep_pairs": num_kstep,
        "split_by_sequence": True,
        "forbid_random_pair_split": True,
        "pose_convention": "R_BA_t_BA_B_tdir_B",
        "manifest_ready": len(seqs) > 0 and len(all_rows) > 0,
    }
    write_json(out_dir / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
