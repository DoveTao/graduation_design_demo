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


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _numeric_columns(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _frame_idx(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def _collect_images(img_dir: Path) -> List[Path]:
    items = [path for path in sorted(img_dir.glob("*.jpg")) if _frame_idx(path) is not None]
    keyed = {int(_frame_idx(path)): path for path in items if _frame_idx(path) is not None}
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


def _candidate_sequences(audit: Dict[str, Dict[str, Any]], min_total_sequences: int) -> List[str]:
    ranked = sorted(
        audit.items(),
        key=lambda item: (
            int(item[1].get("valid_adjacent_pairs_possible", 0)) + int(item[1].get("valid_kstep_pairs_possible", 0)),
            float(item[1].get("valid_pose_fraction", 0.0)),
            int(item[1].get("image_count", 0)),
        ),
        reverse=True,
    )
    eligible = [
        seq for seq, rec in ranked
        if int(rec.get("valid_adjacent_pairs_possible", 0)) >= 80
        and float(rec.get("valid_pose_fraction", 0.0)) >= 0.95
    ]
    if len(eligible) < min_total_sequences:
        eligible = [
            seq for seq, rec in ranked
            if int(rec.get("valid_adjacent_pairs_possible", 0)) >= 40
            and float(rec.get("valid_pose_fraction", 0.0)) >= 0.8
        ]
    if len(eligible) < min_total_sequences:
        eligible = [
            seq for seq, rec in ranked
            if int(rec.get("valid_adjacent_pairs_possible", 0)) >= 40
            and int(rec.get("image_count", 0)) >= 50
        ]
    return eligible


def _split_counts(num_sequences: int, split_cfg: Dict[str, Any]) -> Dict[str, int]:
    min_train = int(split_cfg.get("min_train_sequences", 7))
    min_val = int(split_cfg.get("min_val_sequences", 2))
    min_test = int(split_cfg.get("min_test_sequences", 2))
    train_n = max(min_train, int(round(num_sequences * float(split_cfg.get("train_ratio_by_sequence", 0.7)))))
    val_n = max(min_val, int(round(num_sequences * float(split_cfg.get("val_ratio_by_sequence", 0.15)))))
    test_n = max(min_test, num_sequences - train_n - val_n)
    while train_n + val_n + test_n > num_sequences:
        if train_n > min_train:
            train_n -= 1
        elif val_n > min_val:
            val_n -= 1
        elif test_n > min_test:
            test_n -= 1
        else:
            break
    return {"train": train_n, "val": val_n, "test": test_n}


def _assign_splits(candidate_seqs: Sequence[str], audit: Dict[str, Dict[str, Any]], split_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    quotas = _split_counts(len(candidate_seqs), split_cfg)
    ordered = sorted(
        candidate_seqs,
        key=lambda seq: (
            int(audit[seq].get("valid_adjacent_pairs_possible", 0)) + int(audit[seq].get("valid_kstep_pairs_possible", 0)),
            int(audit[seq].get("image_count", 0)),
        ),
        reverse=True,
    )
    split_order = ["train", "val", "test"]
    splits = {"train": [], "val": [], "test": []}
    cursor = 0
    for seq in ordered:
        for _ in range(3):
            name = split_order[cursor % 3]
            cursor += 1
            if len(splits[name]) < quotas[name]:
                splits[name].append(seq)
                break
        else:
            for name in split_order:
                if len(splits[name]) < quotas[name]:
                    splits[name].append(seq)
                    break
    return splits


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    audit_payload = _read_json(Path(args.sequence_audit))
    audit = audit_payload.get("sequences", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_root = Path(str(cfg.get("dataset", {}).get("local_root", "data/360DVO")))
    min_total = int(cfg.get("download", {}).get("min_total_sequences", 10))
    k_values = [int(x) for x in cfg.get("pair_manifest", {}).get("k_values", [1, 2, 3, 5])]

    candidate_seqs = _candidate_sequences(audit=audit, min_total_sequences=min_total)
    splits = _assign_splits(candidate_seqs=candidate_seqs, audit=audit, split_cfg=cfg.get("split", {}))

    all_rows: List[Dict[str, Any]] = []
    pair_index = 0
    adj_counts = {"train": 0, "val": 0, "test": 0}
    kstep_counts = {"train": 0, "val": 0, "test": 0}
    for seq in candidate_seqs:
        images = _collect_images(local_root / "Sequences" / seq)
        gt_path = local_root / "GroundTruth" / f"{seq}.txt"
        ts_path = local_root / "Timestamps" / f"{seq}_timestamps.txt"
        rows = _read_rows(gt_path=gt_path, ts_path=ts_path, images=images)
        if seq in splits["train"]:
            split_name = "train"
        elif seq in splits["val"]:
            split_name = "val"
        else:
            split_name = "test"
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
                        "split": split_name,
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
                if pair_type == "adjacent":
                    adj_counts[split_name] += 1
                else:
                    kstep_counts[split_name] += 1

    split_rows = {name: [row for row in all_rows if row["split"] == name] for name in ["train", "val", "test"]}
    _write_jsonl(out_dir / "pair_manifest_all.jsonl", all_rows)
    _write_jsonl(out_dir / "pair_manifest_train.jsonl", split_rows["train"])
    _write_jsonl(out_dir / "pair_manifest_val.jsonl", split_rows["val"])
    _write_jsonl(out_dir / "pair_manifest_test.jsonl", split_rows["test"])

    summary = {
        "num_sequences": len(candidate_seqs),
        "sequences_all": candidate_seqs,
        "train_sequences": splits["train"],
        "val_sequences": splits["val"],
        "test_sequences": splits["test"],
        "num_pairs_all": len(all_rows),
        "num_pairs_train": len(split_rows["train"]),
        "num_pairs_val": len(split_rows["val"]),
        "num_pairs_test": len(split_rows["test"]),
        "num_adjacent_pairs_train": adj_counts["train"],
        "num_adjacent_pairs_val": adj_counts["val"],
        "num_adjacent_pairs_test": adj_counts["test"],
        "num_kstep_pairs_train": kstep_counts["train"],
        "num_kstep_pairs_val": kstep_counts["val"],
        "num_kstep_pairs_test": kstep_counts["test"],
        "split_by_sequence": True,
        "forbid_random_pair_split": True,
        "pose_convention": "R_BA_t_BA_B_tdir_B",
        "manifest_ready": bool(all_rows and splits["train"] and splits["val"] and splits["test"]),
    }
    write_json(out_dir / "manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
