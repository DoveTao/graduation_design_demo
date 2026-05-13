from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dset2c_manifest_dataset import discover_manifest_field_mapping


def _as_path(value: Any) -> Path:
    return Path(os.path.expanduser(str(value)))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_rgb(path: Path, hw: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    h, w = hw
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _frame_index(path: str) -> int:
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise
        return int(digits)


@dataclass(frozen=True)
class _Frame:
    image_path: str
    timestamp: float
    frame_index: int


class Dset2CSequenceClipDataset(Dataset):
    """Manifest-native contiguous clip dataset for DSET2C.

    Clips are constructed only from canonical manifest rows. The dataset never
    scans raw sequence directories to define membership or split boundaries.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        expected_split: Optional[str],
        clip_len: int = 3,
        image_hw: Tuple[int, int] = (384, 768),
        max_frame_gap: int = 1,
        max_timestamp_gap_factor: float = 1.75,
        tmag_epsilon: float = 1.0e-6,
        skip_invalid: bool = True,
        require_paths: bool = True,
    ) -> None:
        if int(clip_len) < 3:
            raise ValueError("clip_len must be >= 3 for sequence consistency.")
        self.manifest_path = _as_path(manifest_path)
        self.expected_split = None if expected_split is None else str(expected_split)
        self.clip_len = int(clip_len)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.max_frame_gap = int(max_frame_gap)
        self.max_timestamp_gap_factor = float(max_timestamp_gap_factor)
        self.tmag_epsilon = float(tmag_epsilon)
        self.skip_invalid = bool(skip_invalid)
        self.require_paths = bool(require_paths)

        rows = _read_jsonl(self.manifest_path)
        self.field_mapping = discover_manifest_field_mapping(rows)
        self.stats: Dict[str, Any] = {
            "manifest_path": str(self.manifest_path),
            "expected_split": self.expected_split,
            "clip_len": self.clip_len,
            "raw_row_count": len(rows),
            "kept_pose_row_count": 0,
            "clip_count": 0,
            "skipped_clip_count": 0,
            "skip_reasons": {},
            "sequence_clip_counts": {},
            "sequence_ids": [],
            "direct_glob_data_360dvo_sequences": False,
        }

        normalized_rows: List[Dict[str, Any]] = []
        for row in rows:
            norm, reason = self._normalize_row(row)
            if reason is not None:
                self._bump_skip(f"row:{reason}")
                if not self.skip_invalid:
                    raise ValueError(f"Invalid manifest row: {reason} | row={row}")
                continue
            assert norm is not None
            normalized_rows.append(norm)
        self.stats["kept_pose_row_count"] = len(normalized_rows)
        self.samples = self._build_clips(normalized_rows)
        self.stats["clip_count"] = len(self.samples)
        self.stats["sequence_ids"] = sorted({str(s["sequence_id"]) for s in self.samples})
        if not self.samples:
            raise RuntimeError(f"No contiguous sequence clips found in manifest: {self.manifest_path}")

    def _bump_skip(self, reason: str) -> None:
        bucket = self.stats["skip_reasons"]
        bucket[reason] = int(bucket.get(reason, 0)) + 1

    def _normalize_row(self, row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        fmap = self.field_mapping
        split = str(row.get(fmap.split, ""))
        if self.expected_split is not None and split != self.expected_split:
            return None, f"split_mismatch:{split}"
        if not bool(row.get(fmap.valid_pose, False)):
            return None, "invalid_pose_flag"
        if not bool(row.get(fmap.valid_timestamp, False)):
            return None, "invalid_timestamp_flag"
        path_a = _as_path(row[fmap.image_path_a])
        path_b = _as_path(row[fmap.image_path_b])
        if self.require_paths and (not path_a.is_file() or not path_b.is_file()):
            return None, "missing_image_path"
        try:
            R = np.asarray(row[fmap.rotation], dtype=np.float32)
            t = np.asarray(row[fmap.translation_vec], dtype=np.float32)
            tdir = np.asarray(row[fmap.translation_dir], dtype=np.float32)
            tmag = float(row[fmap.translation_mag])
        except Exception as exc:
            return None, f"bad_pose_fields:{type(exc).__name__}"
        if R.shape != (3, 3) or t.shape != (3,) or tdir.shape != (3,):
            return None, "bad_pose_shape"
        if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(tdir).all() or not math.isfinite(tmag):
            return None, "pose_nonfinite"
        vec_norm = float(np.linalg.norm(t))
        tmag = max(tmag, vec_norm, self.tmag_epsilon)
        dir_norm = float(np.linalg.norm(tdir))
        if dir_norm <= self.tmag_epsilon and vec_norm > self.tmag_epsilon:
            tdir = t / vec_norm
            dir_norm = float(np.linalg.norm(tdir))
        if dir_norm <= self.tmag_epsilon:
            return None, "translation_dir_zero"
        return {
            "dataset_name": str(row.get(fmap.dataset_name, "")),
            "split": split,
            "sequence_id": str(row[fmap.sequence_id]),
            "pair_index": int(row[fmap.pair_index]),
            "pair_type": str(row[fmap.pair_type]),
            "k": int(row[fmap.k]),
            "timestamp_a": float(row[fmap.timestamp_a]),
            "timestamp_b": float(row[fmap.timestamp_b]),
            "image_path_a": str(path_a),
            "image_path_b": str(path_b),
            "frame_a": _frame_index(str(path_a)),
            "frame_b": _frame_index(str(path_b)),
            "R_BA": R.astype(np.float32),
            "t_BA_B": t.astype(np.float32),
            "tdir_B": (tdir / max(dir_norm, self.tmag_epsilon)).astype(np.float32),
            "tmag": float(tmag),
            "log_tmag": float(np.log(max(tmag, self.tmag_epsilon))),
        }, None

    def _build_clips(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        pair_lookup: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        for row in rows:
            by_seq[str(row["sequence_id"])].append(row)
            pair_lookup[(str(row["sequence_id"]), int(row["frame_a"]), int(row["frame_b"]))] = row

        clips: List[Dict[str, Any]] = []
        sequence_clip_counts: Dict[str, int] = {}
        for seq_id, seq_rows in sorted(by_seq.items()):
            adj = [
                r for r in seq_rows
                if str(r["pair_type"]) == "adjacent" and int(r["k"]) == 1
            ]
            adj.sort(key=lambda r: (int(r["frame_a"]), int(r["frame_b"]), float(r["timestamp_a"])))
            if len(adj) < self.clip_len - 1:
                self._bump_skip("clip:not_enough_adjacent_rows")
                continue
            ts_diffs = [float(r["timestamp_b"]) - float(r["timestamp_a"]) for r in adj if float(r["timestamp_b"]) > float(r["timestamp_a"])]
            median_dt = float(np.median(np.asarray(ts_diffs, dtype=np.float64))) if ts_diffs else 0.0
            for start in range(0, len(adj) - (self.clip_len - 1) + 1):
                window = adj[start:start + self.clip_len - 1]
                frames = [_Frame(window[0]["image_path_a"], float(window[0]["timestamp_a"]), int(window[0]["frame_a"]))]
                valid = True
                reason = ""
                prev_b = int(window[0]["frame_a"])
                prev_path_b = str(window[0]["image_path_a"])
                for edge in window:
                    if str(edge["image_path_a"]) != prev_path_b or int(edge["frame_a"]) != prev_b:
                        valid, reason = False, "clip:chain_discontinuity"
                        break
                    if int(edge["frame_b"]) - int(edge["frame_a"]) > self.max_frame_gap:
                        valid, reason = False, "clip:large_frame_gap"
                        break
                    dt = float(edge["timestamp_b"]) - float(edge["timestamp_a"])
                    if median_dt > 0.0 and dt > median_dt * self.max_timestamp_gap_factor:
                        valid, reason = False, "clip:large_timestamp_gap"
                        break
                    frames.append(_Frame(edge["image_path_b"], float(edge["timestamp_b"]), int(edge["frame_b"])))
                    prev_b = int(edge["frame_b"])
                    prev_path_b = str(edge["image_path_b"])
                if not valid:
                    self._bump_skip(reason)
                    continue
                pair_rows: List[Dict[str, Any]] = []
                missing_pair = False
                for i in range(self.clip_len):
                    for j in range(i + 1, self.clip_len):
                        row = pair_lookup.get((seq_id, frames[i].frame_index, frames[j].frame_index))
                        if row is None:
                            missing_pair = True
                            break
                        pair_rows.append(row)
                    if missing_pair:
                        break
                if missing_pair:
                    self._bump_skip("clip:missing_kstep_pair")
                    continue
                clips.append(
                    {
                        "sequence_id": seq_id,
                        "split": str(window[0]["split"]),
                        "dataset_name": str(window[0]["dataset_name"]),
                        "frames": frames,
                        "pairs": pair_rows,
                    }
                )
                sequence_clip_counts[seq_id] = int(sequence_clip_counts.get(seq_id, 0)) + 1
        self.stats["sequence_clip_counts"] = sequence_clip_counts
        self.stats["skipped_clip_count"] = sum(
            int(v) for k, v in self.stats["skip_reasons"].items() if k.startswith("clip:")
        )
        return clips

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        frames: Sequence[_Frame] = sample["frames"]
        images = torch.stack([_read_rgb(Path(frame.image_path), self.image_hw) for frame in frames], dim=0)
        pair_i: List[int] = []
        pair_j: List[int] = []
        pair_k: List[int] = []
        pair_type: List[int] = []
        R: List[torch.Tensor] = []
        t: List[torch.Tensor] = []
        tdir: List[torch.Tensor] = []
        tmag: List[float] = []
        log_tmag: List[float] = []
        frame_to_pos = {int(frame.frame_index): pos for pos, frame in enumerate(frames)}
        for row in sample["pairs"]:
            i = frame_to_pos[int(row["frame_a"])]
            j = frame_to_pos[int(row["frame_b"])]
            pair_i.append(i)
            pair_j.append(j)
            pair_k.append(j - i)
            pair_type.append(1 if (j - i) == 1 else 0)
            R.append(torch.from_numpy(row["R_BA"]))
            t.append(torch.from_numpy(row["t_BA_B"]))
            tdir.append(torch.from_numpy(row["tdir_B"]))
            tmag.append(float(row["tmag"]))
            log_tmag.append(float(row["log_tmag"]))
        return {
            "images": images,
            "frame_ids": torch.tensor([int(f.frame_index) for f in frames], dtype=torch.int64),
            "timestamps": torch.tensor([float(f.timestamp) for f in frames], dtype=torch.float32),
            "sequence_id": str(sample["sequence_id"]),
            "split": str(sample["split"]),
            "manifest_path": str(self.manifest_path),
            "pair_i": torch.tensor(pair_i, dtype=torch.int64),
            "pair_j": torch.tensor(pair_j, dtype=torch.int64),
            "pair_k": torch.tensor(pair_k, dtype=torch.int64),
            "adjacent_mask": torch.tensor(pair_type, dtype=torch.bool),
            "R_BA": torch.stack(R, dim=0),
            "t_BA_B": torch.stack(t, dim=0),
            "tdir_B": torch.stack(tdir, dim=0),
            "tmag": torch.tensor(tmag, dtype=torch.float32),
            "log_tmag": torch.tensor(log_tmag, dtype=torch.float32),
            "valid_mask": torch.ones(len(pair_i), dtype=torch.bool),
            "provenance": {
                "manifest_path": str(self.manifest_path),
                "sequence_id": str(sample["sequence_id"]),
                "frame_ids": [int(f.frame_index) for f in frames],
            },
        }

    def get_clip_summary(self) -> Dict[str, Any]:
        return dict(self.stats)
