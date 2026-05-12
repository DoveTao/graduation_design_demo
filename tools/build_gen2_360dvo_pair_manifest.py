#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import quat_xyzw_to_rot, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_columns(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _rot_to_list(R: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in R.tolist()]


def _collect_images(seq_root: Path) -> List[Path]:
    return sorted([p for p in seq_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _guess_pose_file(seq_root: Path) -> Optional[Path]:
    candidates = sorted(
        [
            p
            for p in seq_root.parent.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".tum", ".log"} and any(h in p.name.lower() for h in ("pose", "traj", "trajectory", "gt"))
        ]
    )
    for cand in candidates:
        if seq_root.name.lower() in cand.as_posix().lower():
            return cand
    return candidates[0] if candidates else None


def _guess_time_file(seq_root: Path) -> Optional[Path]:
    candidates = sorted(
        [
            p
            for p in seq_root.parent.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".json"} and any(h in p.name.lower() for h in ("time", "timestamp"))
        ]
    )
    for cand in candidates:
        if seq_root.name.lower() in cand.as_posix().lower():
            return cand
    return candidates[0] if candidates else None


def _load_timestamps(path: Optional[Path], images: Sequence[Path]) -> List[float]:
    if path and path.exists():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        vals = []
        for ln in lines:
            cols = _numeric_columns(ln)
            if cols:
                vals.append(float(cols[0]))
        if len(vals) >= len(images):
            return vals[: len(images)]
    out: List[float] = []
    for i, img in enumerate(images):
        try:
            out.append(float(img.stem))
        except Exception:
            out.append(float(i))
    return out


def _load_tum_poses(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in lines:
        cols = _numeric_columns(ln)
        if len(cols) < 8:
            continue
        ts = float(cols[0])
        tx, ty, tz = cols[1:4]
        qx, qy, qz, qw = cols[4:8]
        rows.append(
            {
                "timestamp": ts,
                "R_w_c": quat_xyzw_to_rot(qx, qy, qz, qw),
                "t_w_c": np.asarray([tx, ty, tz], dtype=np.float64),
            }
        )
    return rows


def _match_pose_rows(images: Sequence[Path], timestamps: Sequence[float], pose_rows: Sequence[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    if not pose_rows:
        return [None for _ in images]
    pose_ts = np.asarray([float(r["timestamp"]) for r in pose_rows], dtype=np.float64)
    matched: List[Optional[Dict[str, Any]]] = []
    for ts in timestamps:
        idx = int(np.argmin(np.abs(pose_ts - float(ts))))
        row = pose_rows[idx]
        if abs(float(row["timestamp"]) - float(ts)) <= 1.0 or len(pose_rows) == len(images):
            matched.append(row)
        else:
            matched.append(None)
    return matched


def _relative_pose(R_w_a: np.ndarray, t_w_a: np.ndarray, R_w_b: np.ndarray, t_w_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_BA = R_w_b.T @ R_w_a
    t_BA_B = R_w_b.T @ (t_w_a - t_w_b)
    return R_BA, t_BA_B


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(Path(args.config))
    inspection = _read_json(Path(args.inspection))
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(str(config.get("dataset", {}).get("local_root", ""))).expanduser()
    pair_cfg = config.get("pair_manifest", {})
    max_pairs = int(pair_cfg.get("max_smoke_pairs", 100))
    k_values = [int(x) for x in pair_cfg.get("k_values", [1, 2, 3, 5])]
    rows: List[Dict[str, Any]] = []
    blockers: List[str] = []

    if not inspection.get("dataset_found"):
        blockers.append("DATA_NOT_FOUND")
    if not inspection.get("inspection_ready"):
        blockers.append("INSPECTION_NOT_READY")
    if "POSE_CONVENTION_UNKNOWN" in inspection.get("blockers", []):
        blockers.append("POSE_CONVENTION_BLOCKED")

    sequence_summaries: List[Dict[str, Any]] = []
    if not blockers:
        for seq_id in inspection.get("sequences_found", []):
            seq_root = dataset_root / seq_id
            if not seq_root.exists():
                continue
            images = _collect_images(seq_root)
            pose_file = _guess_pose_file(seq_root)
            time_file = _guess_time_file(seq_root)
            pose_rows = _load_tum_poses(pose_file) if pose_file and inspection.get("pose_format_detected", "").startswith("tum_like") else []
            timestamps = _load_timestamps(time_file, images)
            matched = _match_pose_rows(images, timestamps, pose_rows)
            built_for_seq = 0
            for k in k_values:
                for i in range(0, max(0, len(images) - k)):
                    if len(rows) >= max_pairs:
                        break
                    pose_a = matched[i]
                    pose_b = matched[i + k]
                    valid_pose = pose_a is not None and pose_b is not None
                    valid_timestamp = i < len(timestamps) and (i + k) < len(timestamps)
                    if valid_pose:
                        R_BA, t_BA_B = _relative_pose(
                            pose_a["R_w_c"],
                            pose_a["t_w_c"],
                            pose_b["R_w_c"],
                            pose_b["t_w_c"],
                        )
                        tmag = float(np.linalg.norm(t_BA_B))
                        tdir = (t_BA_B / max(tmag, 1.0e-12)).tolist()
                        R_list = _rot_to_list(R_BA)
                        t_list = [float(x) for x in t_BA_B.tolist()]
                        T_w_a = {
                            "R": _rot_to_list(pose_a["R_w_c"]),
                            "t": [float(x) for x in pose_a["t_w_c"].tolist()],
                        }
                        T_w_b = {
                            "R": _rot_to_list(pose_b["R_w_c"]),
                            "t": [float(x) for x in pose_b["t_w_c"].tolist()],
                        }
                    else:
                        tmag = 0.0
                        tdir = [0.0, 0.0, 0.0]
                        R_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                        t_list = [0.0, 0.0, 0.0]
                        T_w_a = None
                        T_w_b = None
                    pair_type = "adjacent" if k == 1 else "kstep"
                    rows.append(
                        {
                            "dataset": "360DVO",
                            "seq_id": seq_id,
                            "pair_index": len(rows),
                            "pair_type": pair_type,
                            "k": int(k),
                            "image_path_a": str(images[i]),
                            "image_path_b": str(images[i + k]),
                            "timestamp_a": float(timestamps[i]),
                            "timestamp_b": float(timestamps[i + k]),
                            "T_w_a": T_w_a,
                            "T_w_b": T_w_b,
                            "R_BA": R_list,
                            "t_BA_B": t_list,
                            "tdir_B": [float(x) for x in tdir],
                            "tmag": float(tmag),
                            "valid_pose": bool(valid_pose),
                            "valid_timestamp": bool(valid_timestamp),
                        }
                    )
                    built_for_seq += 1
                if len(rows) >= max_pairs:
                    break
            sequence_summaries.append(
                {
                    "seq_id": seq_id,
                    "num_images": len(images),
                    "pose_file": str(pose_file) if pose_file else None,
                    "timestamp_file": str(time_file) if time_file else None,
                    "pose_rows": len(pose_rows),
                    "pairs_built": built_for_seq,
                }
            )
            if len(rows) >= max_pairs:
                break

    if not rows:
        out_jsonl.write_text("", encoding="utf-8")
    else:
        _write_jsonl(out_jsonl, rows)

    valid_pose_count = sum(1 for r in rows if r["valid_pose"])
    summary = {
        "dataset": "360DVO",
        "manifest_ready": bool(rows and valid_pose_count > 0 and not blockers),
        "pose_convention_ready": bool(not blockers and inspection.get("inspection_ready")),
        "split_by_sequence": bool(pair_cfg.get("split_by_sequence", True)),
        "forbid_random_pair_split": bool(pair_cfg.get("forbid_random_pair_split", True)),
        "max_smoke_pairs": max_pairs,
        "num_pairs": len(rows),
        "num_adjacent_pairs": sum(1 for r in rows if r["pair_type"] == "adjacent"),
        "num_kstep_pairs": sum(1 for r in rows if r["pair_type"] == "kstep"),
        "valid_pose_fraction": float(valid_pose_count / max(len(rows), 1)),
        "k_values": k_values,
        "sequence_summaries": sequence_summaries,
        "blockers": blockers,
        "target_schema_terms": ["R_BA", "t_BA_B", "tdir_B", "T_w_c", "adjacent", "kstep"],
    }
    write_json(out_summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
