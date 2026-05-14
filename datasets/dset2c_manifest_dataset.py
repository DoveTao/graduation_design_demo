from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestFieldMapping:
    image_path_a: str = "image_path_a"
    image_path_b: str = "image_path_b"
    rotation: str = "R_BA"
    translation_vec: str = "t_BA_B"
    translation_dir: str = "tdir_B"
    translation_mag: str = "tmag"
    sequence_id: str = "seq_id"
    pair_index: str = "pair_index"
    pair_type: str = "pair_type"
    k: str = "k"
    split: str = "split"
    timestamp_a: str = "timestamp_a"
    timestamp_b: str = "timestamp_b"
    valid_pose: str = "valid_pose"
    valid_timestamp: str = "valid_timestamp"
    dataset_name: str = "dataset"


def discover_manifest_field_mapping(rows: Sequence[Dict[str, Any]]) -> ManifestFieldMapping:
    if not rows:
        raise ValueError("Cannot discover manifest schema from empty rows.")
    keys = set(rows[0].keys())
    expected = ManifestFieldMapping()
    missing = [value for value in expected.__dict__.values() if value not in keys]
    if missing:
        raise KeyError(
            "Canonical DSET2C manifest is missing expected fields: "
            + ", ".join(missing)
        )
    return expected


def _as_path(value: Any) -> Path:
    return Path(os.path.expanduser(str(value)))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_rgb(path: Path, hw: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    h, w = hw
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class Dset2CCanonicalPairDataset(Dataset):
    """Manifest-native DSET2C pair dataset.

    This dataset only reads samples from a jsonl manifest and never scans raw
    sequence directories. It is intended to be the TRAIN360 entry point.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        expected_split: Optional[str] = None,
        image_hw: Tuple[int, int] = (1024, 2048),
        tmag_epsilon: float = 1.0e-6,
        skip_invalid: bool = True,
        require_paths: bool = True,
    ) -> None:
        self.manifest_path = _as_path(manifest_path)
        self.expected_split = None if expected_split is None else str(expected_split)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.tmag_epsilon = float(tmag_epsilon)
        self.skip_invalid = bool(skip_invalid)
        self.require_paths = bool(require_paths)

        rows = _load_jsonl(self.manifest_path)
        self.field_mapping = discover_manifest_field_mapping(rows)
        self.schema_fields = sorted(rows[0].keys()) if rows else []
        self.sequence_ids_all = sorted({str(r[self.field_mapping.sequence_id]) for r in rows})
        self.stats: Dict[str, Any] = {
            "manifest_path": str(self.manifest_path),
            "expected_split": self.expected_split,
            "raw_row_count": len(rows),
            "kept_row_count": 0,
            "skipped_row_count": 0,
            "skip_reasons": {},
            "schema_fields": list(self.schema_fields),
            "sequence_ids": [],
        }
        self.samples: List[Dict[str, Any]] = []

        for row in rows:
            normalized, reason = self._normalize_row(row)
            if reason is not None:
                self._bump_skip(reason)
                if not self.skip_invalid:
                    raise ValueError(
                        f"Invalid manifest row in {self.manifest_path}: {reason} | row={row}"
                    )
                continue
            assert normalized is not None
            self.samples.append(normalized)

        self.stats["kept_row_count"] = len(self.samples)
        self.stats["skipped_row_count"] = sum(self.stats["skip_reasons"].values())
        self.stats["sequence_ids"] = sorted({s["sequence_id"] for s in self.samples})
        if not self.samples:
            raise RuntimeError(f"No valid samples found in manifest: {self.manifest_path}")

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
            R_BA = np.asarray(row[fmap.rotation], dtype=np.float32)
            t_BA_B = np.asarray(row[fmap.translation_vec], dtype=np.float32)
            tdir_B = np.asarray(row[fmap.translation_dir], dtype=np.float32)
            tmag = float(row[fmap.translation_mag])
        except Exception as exc:
            return None, f"bad_pose_fields:{type(exc).__name__}"

        if R_BA.shape != (3, 3):
            return None, f"rotation_shape:{tuple(R_BA.shape)}"
        if t_BA_B.shape != (3,):
            return None, f"translation_vec_shape:{tuple(t_BA_B.shape)}"
        if tdir_B.shape != (3,):
            return None, f"translation_dir_shape:{tuple(tdir_B.shape)}"
        if not np.isfinite(R_BA).all():
            return None, "rotation_nonfinite"
        if not np.isfinite(t_BA_B).all():
            return None, "translation_vec_nonfinite"
        if not np.isfinite(tdir_B).all():
            return None, "translation_dir_nonfinite"
        if not math.isfinite(tmag):
            return None, "translation_mag_nonfinite"

        vec_norm = float(np.linalg.norm(t_BA_B))
        if tmag <= self.tmag_epsilon and vec_norm > self.tmag_epsilon:
            tmag = vec_norm
        tmag = max(float(tmag), float(self.tmag_epsilon))

        dir_norm = float(np.linalg.norm(tdir_B))
        if dir_norm <= self.tmag_epsilon and vec_norm > self.tmag_epsilon:
            tdir_B = t_BA_B / vec_norm
            dir_norm = float(np.linalg.norm(tdir_B))
        if dir_norm <= self.tmag_epsilon:
            return None, "translation_dir_zero"
        tdir_B = (tdir_B / max(dir_norm, self.tmag_epsilon)).astype(np.float32)

        timestamp_a = float(row[fmap.timestamp_a])
        timestamp_b = float(row[fmap.timestamp_b])
        dt_world = max(timestamp_b - timestamp_a, 0.0)
        if dt_world <= 0.0:
            dt_world = tmag

        return {
            "dataset_name": str(row.get(fmap.dataset_name, "")),
            "split": split,
            "sequence_id": str(row[fmap.sequence_id]),
            "pair_index": int(row[fmap.pair_index]),
            "pair_type": str(row[fmap.pair_type]),
            "k": int(row[fmap.k]),
            "timestamp_a": timestamp_a,
            "timestamp_b": timestamp_b,
            "dt_world": float(dt_world),
            "image_path_a": str(path_a),
            "image_path_b": str(path_b),
            "R_BA": R_BA.astype(np.float32),
            "t_BA_B": t_BA_B.astype(np.float32),
            "tdir_B": tdir_B.astype(np.float32),
            "tmag": float(tmag),
            "log_tmag": float(np.log(max(tmag, self.tmag_epsilon))),
        }, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        path_a = Path(sample["image_path_a"])
        path_b = Path(sample["image_path_b"])

        try:
            image_a = _read_rgb(path_a, self.image_hw)
            image_b = _read_rgb(path_b, self.image_hw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load manifest-native sample idx={idx} from "
                f"{self.manifest_path}: {type(exc).__name__}: {exc}"
            ) from exc

        R_BA = torch.from_numpy(sample["R_BA"])
        t_BA_B = torch.from_numpy(sample["t_BA_B"])
        tdir_B = torch.from_numpy(sample["tdir_B"])
        tmag = torch.tensor(float(sample["tmag"]), dtype=torch.float32)
        log_tmag = torch.tensor(float(sample["log_tmag"]), dtype=torch.float32)

        meta = {
            "dataset_name": sample["dataset_name"],
            "manifest_path": str(self.manifest_path),
            "sequence_id": sample["sequence_id"],
            "pair_index": sample["pair_index"],
            "pair_type": sample["pair_type"],
            "k": sample["k"],
            "split": sample["split"],
            "timestamp_a": sample["timestamp_a"],
            "timestamp_b": sample["timestamp_b"],
            "dt_world": sample["dt_world"],
            "image_path_a": sample["image_path_a"],
            "image_path_b": sample["image_path_b"],
        }

        return {
            "image_a": image_a,
            "image_b": image_b,
            "IA": image_a,
            "IB": image_b,
            "R_BA": R_BA,
            "R_gt": R_BA,
            "t_BA_B": t_BA_B,
            "t_gt_vec": t_BA_B,
            "tdir_B": tdir_B,
            "t_gt_dir": tdir_B,
            "tmag": tmag,
            "t_gt_mag": tmag,
            "log_tmag": log_tmag,
            "sequence_id": sample["sequence_id"],
            "pair_index": torch.tensor(sample["pair_index"], dtype=torch.int64),
            "pair_type": sample["pair_type"],
            "k": torch.tensor(sample["k"], dtype=torch.int64),
            "split": sample["split"],
            "meta": meta,
        }

    def get_schema_summary(self) -> Dict[str, Any]:
        return dict(self.stats)


def summarize_manifest_group(datasets: Iterable[Dset2CCanonicalPairDataset]) -> Dict[str, Any]:
    ds_list = list(datasets)
    seq_sets = {
        Path(ds.manifest_path).stem: set(ds.get_schema_summary().get("sequence_ids", []))
        for ds in ds_list
    }
    overlap = {}
    names = list(seq_sets.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            shared = sorted(seq_sets[name_a] & seq_sets[name_b])
            if shared:
                overlap[f"{name_a}__{name_b}"] = shared
    return {
        "sequence_sets": {k: sorted(v) for k, v in seq_sets.items()},
        "sequence_overlap": overlap,
        "has_overlap": bool(overlap),
    }
