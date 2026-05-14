"""
File: dataset_pano_only.py
Description:
    Dataset definitions for loading panoramic image pairs and relative pose
    labels from the RflyPano-style data layout. This file supports mixed-k
    training pairs, fixed evaluation pairs, and pose metadata parsing.

Main Components:
    - Frame and pair metadata structures
    - Label parsing and relative pose computation
    - Training dataset with mixed temporal offsets and optional color augmentation
    - Fixed-k evaluation dataset for deterministic validation

Usage / Role:
    Provides the primary dataset loading layer for training and evaluation.

Notes:
    The dataset is used by the graduation design MVP to evaluate coarse
    matching, translation direction/scale supervision, k buckets, and dt-world
    buckets for odometry-oriented relative transforms.
"""

import json
import os
import re
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, get_worker_info


def _interleave_sorted_groups(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item.get(key), []).append(item)
    out: List[Dict[str, Any]] = []
    keys = sorted(groups.keys(), key=str)
    max_len = max((len(v) for v in groups.values()), default=0)
    for i in range(max_len):
        for k in keys:
            group = groups[k]
            if i < len(group):
                out.append(group[i])
    return out


@dataclass
class FrameRec:
    scene: str
    seq: str
    ts_str: str
    ts_val: float
    pano_path: str
    label_path: str


def _read_pano_rgb(path: str, hw: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    H, W = hw
    if img.size != (W, H):
        img = img.resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _rpy_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _parse_label_13(path: str) -> Tuple[np.ndarray, np.ndarray]:
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    v = np.array([float(x) for x in nums], dtype=np.float32)
    if v.size < 7:
        raise ValueError(f"Bad label file (need >=7 numbers): {path}")

    roll, pitch, yaw = float(v[1]), float(v[2]), float(v[3])
    x, y, z = float(v[4]), float(v[5]), float(v[6])
    R_wb = _rpy_to_R_zyx(roll, pitch, yaw)
    t_w = np.array([x, y, z], dtype=np.float32)
    return R_wb, t_w


def _relative_pose_A_to_B_in_B(R_wA: np.ndarray, t_wA: np.ndarray, R_wB: np.ndarray, t_wB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_BA = (R_wB.T @ R_wA).astype(np.float32)
    t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
    return R_BA, t_BA


def _list_pano_paths(data_root: str) -> Tuple[str, List[str]]:
    cand = os.path.join(data_root, "PanoramaView")
    pano_root = cand if os.path.isdir(cand) else data_root
    patterns = [
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.jpg"),
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.JPG"),
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.jpeg"),
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.JPEG"),
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.png"),
        os.path.join(pano_root, "scene*", "seq*", "panorama_*.PNG"),
        os.path.join(pano_root, "scene*", "seq*", "panorama*.jpg"),
        os.path.join(pano_root, "scene*", "seq*", "panorama*.JPG"),
        os.path.join(pano_root, "scene*", "seq*", "panorama*.png"),
        os.path.join(pano_root, "scene*", "seq*", "panorama*.PNG"),
    ]
    pano_paths: List[str] = []
    for pat in patterns:
        pano_paths.extend(glob.glob(pat))
    pano_paths = sorted(set(pano_paths))
    if len(pano_paths) == 0:
        raise FileNotFoundError(f"No panorama images found under pano_root={pano_root}")
    return pano_root, pano_paths


def _scan_seq_frames(data_root: str, scenes: Optional[List[str]] = None, seqs: Optional[List[str]] = None) -> Dict[Tuple[str, str], List[FrameRec]]:
    _pano_root, pano_paths = _list_pano_paths(data_root)
    seq_frames: Dict[Tuple[str, str], List[FrameRec]] = {}
    for pano_path in pano_paths:
        seq_dir = os.path.dirname(pano_path)
        seq = os.path.basename(seq_dir)
        scene = os.path.basename(os.path.dirname(seq_dir))
        if scenes is not None and scene not in scenes:
            continue
        if seqs is not None and seq not in seqs:
            continue
        stem, _ext = os.path.splitext(os.path.basename(pano_path))
        if stem.startswith("panorama_"):
            ts_str = stem[len("panorama_"):]
        elif stem.startswith("panorama"):
            ts_str = stem[len("panorama"):].lstrip("_")
        else:
            continue
        try:
            ts_val = float(ts_str)
        except Exception:
            continue
        label_path = os.path.join(seq_dir, f"label_{ts_str}.txt")
        if not os.path.exists(label_path):
            continue
        rec = FrameRec(scene=scene, seq=seq, ts_str=ts_str, ts_val=ts_val, pano_path=pano_path, label_path=label_path)
        seq_frames.setdefault((scene, seq), []).append(rec)
    for key in list(seq_frames.keys()):
        seq_frames[key].sort(key=lambda r: r.ts_val)
    return seq_frames


def _normalize_split_mode(split_by: str) -> str:
    split_by = str(split_by).strip().lower()
    if split_by in {"scene_seq", "scene/seq", "scene-seq", "scene+seq"}:
        return "scene_seq"
    if split_by == "scene":
        return "scene"
    raise ValueError(f"Unsupported split_by={split_by!r}. Use 'scene_seq' or 'scene'.")


def _group_key(scene: str, seq: str, split_by: str) -> Any:
    mode = _normalize_split_mode(split_by)
    if mode == "scene":
        return scene
    return (scene, seq)


def _partition_seq_keys(seq_keys: List[Tuple[str, str]], split: Optional[str], split_by: str, train_ratio: float, split_seed: int) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    seq_keys = sorted(set(seq_keys))
    mode = _normalize_split_mode(split_by)
    if split is None:
        return seq_keys, {
            "enabled": False,
            "split": None,
            "split_by": mode,
            "num_total_seq": len(seq_keys),
            "num_selected_seq": len(seq_keys),
            "num_total_groups": len({_group_key(s, q, mode) for s, q in seq_keys}),
            "num_selected_groups": len({_group_key(s, q, mode) for s, q in seq_keys}),
            "selected_seq_keys": list(seq_keys),
            "selected_groups": sorted({_group_key(s, q, mode) for s, q in seq_keys}, key=str),
        }

    split = str(split).strip().lower()
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split={split!r}. Use 'train', 'test', or None.")
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    group_to_seqs: Dict[Any, List[Tuple[str, str]]] = {}
    for scene, seq in seq_keys:
        g = _group_key(scene, seq, mode)
        group_to_seqs.setdefault(g, []).append((scene, seq))
    all_groups = sorted(group_to_seqs.keys(), key=str)
    n_groups = len(all_groups)
    if n_groups < 2:
        raise RuntimeError(f"Need at least 2 groups to split train/test, but only found {n_groups} group(s) with split_by={mode}.")

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n_groups)
    n_train = int(round(n_groups * float(train_ratio)))
    n_train = max(1, min(n_groups - 1, n_train))
    train_groups = {all_groups[i] for i in perm[:n_train]}
    test_groups = {all_groups[i] for i in perm[n_train:]}
    selected_groups = train_groups if split == "train" else test_groups

    selected_seq_keys: List[Tuple[str, str]] = []
    for g in sorted(selected_groups, key=str):
        selected_seq_keys.extend(sorted(group_to_seqs[g]))

    summary = {
        "enabled": True,
        "split": split,
        "split_by": mode,
        "train_ratio": float(train_ratio),
        "split_seed": int(split_seed),
        "num_total_seq": len(seq_keys),
        "num_selected_seq": len(selected_seq_keys),
        "num_total_groups": n_groups,
        "num_selected_groups": len(selected_groups),
        "selected_seq_keys": list(selected_seq_keys),
        "selected_groups": sorted(selected_groups, key=str),
        "all_groups": all_groups,
    }
    return selected_seq_keys, summary


class RflyPanoPanoramaPairs(Dataset):
    def __init__(self, data_root: str, split: str | None = None, hw: Tuple[int, int] = (1024, 2048), k_stride: int = 5,
                 pair_step: int = 1, scenes: Optional[List[str]] = None, seqs: Optional[List[str]] = None,
                 split_by: str = "scene_seq", train_ratio: float = 0.8, split_seed: int = 3407):
        self.data_root = data_root
        self.hw = hw
        self.k_stride = int(k_stride)
        self.pair_step = int(pair_step)
        self.split = split
        self.split_by = _normalize_split_mode(split_by)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        seq_frames = _scan_seq_frames(data_root=data_root, scenes=scenes, seqs=seqs)
        selected_seq_keys, self.split_summary = _partition_seq_keys(list(seq_frames.keys()), split, self.split_by, self.train_ratio, self.split_seed)
        selected_seq_key_set = set(selected_seq_keys)
        self.sequence_keys = sorted(selected_seq_key_set)
        self.pairs: List[Tuple[FrameRec, FrameRec]] = []
        for (scene, seq), recs in sorted(seq_frames.items()):
            if (scene, seq) not in selected_seq_key_set:
                continue
            n = len(recs)
            if n <= self.k_stride:
                continue
            for i in range(0, n - self.k_stride, self.pair_step):
                self.pairs.append((recs[i], recs[i + self.k_stride]))
        self.split_summary["num_pairs"] = len(self.pairs)
        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid pairs built for split={split!r}. Check split ratio, k_stride, and label existence.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        A, B = self.pairs[idx]
        IA = _read_pano_rgb(A.pano_path, self.hw)
        IB = _read_pano_rgb(B.pano_path, self.hw)
        R_wA, t_wA = _parse_label_13(A.label_path)
        R_wB, t_wB = _parse_label_13(B.label_path)
        R_BA, t_BA = _relative_pose_A_to_B_in_B(R_wA, t_wA, R_wB, t_wB)
        t_mag = float(np.linalg.norm(t_BA))
        t_dir = t_BA / (t_mag + 1e-8)
        return {
            "IA": IA,
            "IB": IB,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_vec": torch.from_numpy(t_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "t_gt_mag": torch.tensor(t_mag, dtype=torch.float32),
            "meta": {
                "scene": A.scene, "seq": A.seq, "tsA": A.ts_str, "tsB": B.ts_str,
                "k": int(self.k_stride), "dt_world": t_mag,
            },
        }


class RflyPanoPanoramaPairsMixedK(Dataset):
    def __init__(self, data_root: str, hw: Tuple[int, int] = (1024, 2048), k_choices: List[int] = [5, 10],
                 k_probs: Optional[List[float]] = None, pair_step: int = 1, scenes: Optional[List[str]] = None,
                 seqs: Optional[List[str]] = None, min_dt: float = 0.0, max_tries: int = 10, seed: int = 1234,
                 split: str | None = None, H: int | None = None, W: int | None = None, strict_dt: bool | None = None,
                 split_by: str = "scene_seq", train_ratio: float = 0.8, split_seed: int = 3407,
                 color_aug: bool = False, color_aug_strength: float = 1.0,
                 return_seq_turn_triplet: bool = False, seq_turn_only_k: int = 1):
        self.data_root = data_root
        if (H is not None) and (W is not None):
            hw = (int(H), int(W))
        self.hw = hw
        self.pair_step = int(pair_step)
        self.return_seq_turn_triplet = bool(return_seq_turn_triplet)
        self.seq_turn_only_k = int(seq_turn_only_k)
        self.split = split
        self.strict_dt = strict_dt
        self.split_by = _normalize_split_mode(split_by)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.color_aug = bool(color_aug)
        self.color_aug_strength = float(color_aug_strength)
        self.k_choices = [int(k) for k in k_choices]
        assert len(self.k_choices) > 0
        self.max_k = max(self.k_choices)
        if k_probs is None:
            self.k_probs = None
        else:
            assert len(k_probs) == len(self.k_choices)
            p = np.asarray(k_probs, dtype=np.float64)
            p = p / (p.sum() + 1e-12)
            self.k_probs = p.tolist()
        self.min_dt = float(min_dt)
        self.max_tries = int(max_tries)
        self.seed = int(seed)
        seq_frames = _scan_seq_frames(data_root=data_root, scenes=scenes, seqs=seqs)
        selected_seq_keys, self.split_summary = _partition_seq_keys(list(seq_frames.keys()), split, self.split_by, self.train_ratio, self.split_seed)
        selected_seq_key_set = set(selected_seq_keys)
        self.sequence_keys = sorted(selected_seq_key_set)
        self.seqs_data: List[Dict[str, Any]] = []
        for (scene, seq), frames in sorted(seq_frames.items()):
            if (scene, seq) not in selected_seq_key_set:
                continue
            n = len(frames)
            if n <= self.max_k:
                continue
            R_w = np.zeros((n, 3, 3), dtype=np.float32)
            t_w = np.zeros((n, 3), dtype=np.float32)
            pano_list: List[str] = []
            ts_list: List[str] = []
            for i, fr in enumerate(frames):
                R_i, t_i = _parse_label_13(fr.label_path)
                R_w[i] = R_i
                t_w[i] = t_i
                pano_list.append(fr.pano_path)
                ts_list.append(fr.ts_str)
            valid_i = list(range(0, n - self.max_k, self.pair_step))
            if len(valid_i) == 0:
                continue
            self.seqs_data.append({
                "scene": scene, "seq": seq, "n": n, "R_w": R_w, "t_w": t_w,
                "pano": pano_list, "ts": ts_list, "valid_i": valid_i,
            })
        if len(self.seqs_data) == 0:
            raise RuntimeError(f"No valid sequences for split={split!r} after group split.")
        self.index_map: List[Tuple[int, int]] = []
        for sidx, sd in enumerate(self.seqs_data):
            for li in sd["valid_i"]:
                self.index_map.append((sidx, li))
        self.split_summary["num_pairs"] = len(self.index_map)
        self.split_summary["num_sequences_after_filter"] = len(self.seqs_data)
        if len(self.index_map) == 0:
            raise RuntimeError(f"No valid base indices for split={split!r}.")
        self._rng = np.random.default_rng(self.seed)

    def _apply_color_aug(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if (not self.color_aug) or self.color_aug_strength <= 0.0:
            return img
        s = float(self.color_aug_strength)
        brightness = float(rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s))
        contrast = float(rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s))
        gamma = float(rng.uniform(1.0 - 0.08 * s, 1.0 + 0.08 * s))
        noise_std = 0.008 * s
        out = img.float()
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * contrast + mean
        out = out * brightness
        out = out.clamp(0.0, 1.0).pow(gamma)
        if noise_std > 0.0:
            noise = torch.from_numpy(rng.normal(0.0, noise_std, size=tuple(out.shape)).astype(np.float32))
            out = out + noise
        return out.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_rng(self):
        info = get_worker_info()
        if info is None:
            return self._rng
        if not hasattr(self, "_worker_rng"):
            wseed = (self.seed + 1000003 * info.id) % (2**32)
            self._worker_rng = np.random.default_rng(wseed)
        return self._worker_rng

    def __getitem__(self, idx: int):
        rng = self._get_rng()
        max_dt = getattr(self, "max_dt", None)
        if max_dt is not None:
            max_dt = float(max_dt)
            if max_dt <= 0:
                max_dt = None
        strict = (self.min_dt > 0.0) or (max_dt is not None)
        def _ok_dt(dt: float) -> bool:
            if (self.min_dt > 0.0) and (dt < self.min_dt):
                return False
            if (max_dt is not None) and (dt > max_dt):
                return False
            return True
        GLOBAL_TRIES = max(50, self.max_tries * 50)
        base_seq_idx, base_i = self.index_map[idx]
        chosen = None
        tries_global = 0
        while chosen is None and tries_global < GLOBAL_TRIES:
            tries_global += 1
            if tries_global == 1:
                seq_idx, i = base_seq_idx, base_i
            else:
                if tries_global < (GLOBAL_TRIES // 3):
                    seq_idx = base_seq_idx
                    sd0 = self.seqs_data[seq_idx]
                    i = int(rng.choice(sd0["valid_i"]))
                else:
                    seq_idx, i = self.index_map[int(rng.integers(0, len(self.index_map)))]
            sd = self.seqs_data[seq_idx]
            n = sd["n"]
            for _ in range(max(1, self.max_tries)):
                k = int(rng.choice(self.k_choices, p=self.k_probs))
                j = i + k
                if j >= n:
                    continue
                dt_world = float(np.linalg.norm(sd["t_w"][j] - sd["t_w"][i]))
                if (not strict) or _ok_dt(dt_world):
                    chosen = (seq_idx, i, j, k, dt_world)
                    break
        if chosen is None:
            raise RuntimeError(
                f"MixedK strict dt sampling failed after {GLOBAL_TRIES} attempts. split={self.split!r} split_by={self.split_by} min_dt={self.min_dt} max_dt={max_dt}."
            )
        seq_idx, i, j, k, dt_world = chosen
        sd = self.seqs_data[seq_idx]
        IA = _read_pano_rgb(sd["pano"][i], self.hw)
        IB = _read_pano_rgb(sd["pano"][j], self.hw)
        IA = self._apply_color_aug(IA, rng)
        IB = self._apply_color_aug(IB, rng)
        R_wA = sd["R_w"][i]
        t_wA = sd["t_w"][i]
        R_wB = sd["R_w"][j]
        t_wB = sd["t_w"][j]
        R_BA = (R_wB.T @ R_wA).astype(np.float32)
        t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
        t_mag = float(np.linalg.norm(t_BA))
        t_dir = t_BA / (t_mag + 1e-8)
        if self.return_seq_turn_triplet and int(k) == int(self.seq_turn_only_k) and (j + 1) < n:
            R_wC = sd["R_w"][j + 1]
            t_wC = sd["t_w"][j + 1]
            R_BC = (R_wC.T @ R_wB).astype(np.float32)
            t_BC = (R_wC.T @ (t_wB - t_wC)).astype(np.float32)
            t_BC_mag = float(np.linalg.norm(t_BC))
            t_BC_dir = t_BC / (t_BC_mag + 1e-8)
            IC = _read_pano_rgb(sd["pano"][j + 1], self.hw)
            IC = self._apply_color_aug(IC, rng)
            has_seq_turn_triplet = True
        else:
            IC = torch.zeros_like(IA)
            R_BC = np.eye(3, dtype=np.float32)
            t_BC = np.zeros(3, dtype=np.float32)
            t_BC_mag = 0.0
            t_BC_dir = np.zeros(3, dtype=np.float32)
            has_seq_turn_triplet = False
        return {
            "IA": IA, "IB": IB,
            "IC": IC,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_vec": torch.from_numpy(t_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "t_gt_mag": torch.tensor(t_mag, dtype=torch.float32),
            "R_gt_bc": torch.from_numpy(R_BC),
            "t_gt_bc_vec": torch.from_numpy(t_BC),
            "t_gt_bc_dir": torch.from_numpy(t_BC_dir),
            "t_gt_bc_mag": torch.tensor(t_BC_mag, dtype=torch.float32),
            "meta": {
                "scene": sd["scene"], "seq": sd["seq"],
                "tsA": sd["ts"][i], "tsB": sd["ts"][j],
                "i": int(i), "j": int(j),
                "k": int(k), "dt_world": float(dt_world),
                "dt_world_bc": float(t_BC_mag),
                "has_seq_turn_triplet": bool(has_seq_turn_triplet),
                "seq_turn_only_k": int(self.seq_turn_only_k),
            },
        }


class RflyPanoPanoramaPairsEvalFixedKList(Dataset):
    """Deterministic evaluation set built from a fixed list of (i, i+k) pairs.

    Unlike MixedK, this dataset precomputes all valid evaluation pairs once and then
    iterates over them in a fixed order, making evaluation fully reproducible.
    """
    def __init__(self, data_root: str, hw: Tuple[int, int] = (1024, 2048), k_list: Tuple[int, ...] = (10, 20),
                 pair_step: int = 1, scenes: Optional[List[str]] = None, seqs: Optional[List[str]] = None,
                 min_dt: float = 0.0, max_dt: Optional[float] = None, split: str | None = "test",
                 split_by: str = "scene_seq", train_ratio: float = 0.8, split_seed: int = 3407,
                 H: int | None = None, W: int | None = None):
        self.data_root = data_root
        if H is not None and W is not None:
            hw = (int(H), int(W))
        self.hw = hw
        self.k_list = tuple(sorted({int(k) for k in k_list}))
        self.pair_step = int(pair_step)
        self.min_dt = float(min_dt)
        self.max_dt = None if max_dt is None else float(max_dt)
        self.split = split
        self.split_by = _normalize_split_mode(split_by)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)

        seq_frames = _scan_seq_frames(data_root=data_root, scenes=scenes, seqs=seqs)
        selected_seq_keys, self.split_summary = _partition_seq_keys(list(seq_frames.keys()), split, self.split_by, self.train_ratio, self.split_seed)
        selected_seq_key_set = set(selected_seq_keys)
        self.sequence_keys = sorted(selected_seq_key_set)

        self.seqs_data: List[Dict[str, Any]] = []
        self.pairs_meta: List[Dict[str, Any]] = []
        for (scene, seq), frames in sorted(seq_frames.items()):
            if (scene, seq) not in selected_seq_key_set:
                continue
            n = len(frames)
            if n <= 1:
                continue
            R_w = np.zeros((n, 3, 3), dtype=np.float32)
            t_w = np.zeros((n, 3), dtype=np.float32)
            pano_list: List[str] = []
            ts_list: List[str] = []
            for i, fr in enumerate(frames):
                R_i, t_i = _parse_label_13(fr.label_path)
                R_w[i] = R_i
                t_w[i] = t_i
                pano_list.append(fr.pano_path)
                ts_list.append(fr.ts_str)
            sd_idx = len(self.seqs_data)
            self.seqs_data.append({
                "scene": scene, "seq": seq, "n": n, "R_w": R_w, "t_w": t_w, "pano": pano_list, "ts": ts_list,
            })
            for k in self.k_list:
                if n <= k:
                    continue
                for i in range(0, n - k, self.pair_step):
                    j = i + k
                    dt_world = float(np.linalg.norm(t_w[j] - t_w[i]))
                    if self.min_dt > 0.0 and dt_world < self.min_dt:
                        continue
                    if self.max_dt is not None and self.max_dt > 0.0 and dt_world > self.max_dt:
                        continue
                    self.pairs_meta.append({
                        "seq_idx": sd_idx, "scene": scene, "seq": seq, "i": i, "j": j, "k": int(k),
                        "tsA": ts_list[i], "tsB": ts_list[j], "dt_world": dt_world,
                    })
        self.pairs_meta.sort(key=lambda m: (m["scene"], m["seq"], m["k"], m["i"], m["j"]))
        # Interleave k groups so capped evaluation prefixes remain representative.
        # Full evaluation is unchanged as a set, but train_eval with max batches no
        # longer reports only the first k bucket.
        self.pairs_meta = _interleave_sorted_groups(self.pairs_meta, "k")
        self.split_summary["num_pairs"] = len(self.pairs_meta)
        self.split_summary["k_list"] = list(self.k_list)
        if len(self.pairs_meta) == 0:
            raise RuntimeError(
                f"No valid deterministic eval pairs for split={split!r}. Check k_list={self.k_list}, min_dt={self.min_dt}, max_dt={self.max_dt}."
            )

    def __len__(self) -> int:
        return len(self.pairs_meta)

    def manifest(self) -> List[Dict[str, Any]]:
        return list(self.pairs_meta)

    def __getitem__(self, idx: int):
        meta = self.pairs_meta[idx]
        sd = self.seqs_data[meta["seq_idx"]]
        i, j = int(meta["i"]), int(meta["j"])
        IA = _read_pano_rgb(sd["pano"][i], self.hw)
        IB = _read_pano_rgb(sd["pano"][j], self.hw)
        R_wA = sd["R_w"][i]
        t_wA = sd["t_w"][i]
        R_wB = sd["R_w"][j]
        t_wB = sd["t_w"][j]
        R_BA, t_BA = _relative_pose_A_to_B_in_B(R_wA, t_wA, R_wB, t_wB)
        t_mag = float(np.linalg.norm(t_BA))
        t_dir = t_BA / (t_mag + 1e-8)
        return {
            "IA": IA,
            "IB": IB,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_vec": torch.from_numpy(t_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "t_gt_mag": torch.tensor(t_mag, dtype=torch.float32),
            "meta": {
                "scene": meta["scene"], "seq": meta["seq"], "tsA": meta["tsA"], "tsB": meta["tsB"],
                "k": meta["k"], "dt_world": meta["dt_world"], "eval_fixed": True,
            },
        }


class RflyPanoPanoramaPairsTrainFixedList(Dataset):
    """Training dataset driven by an explicit pair manifest.

    This is intended for train-only curriculum / balancing experiments where
    pair selection must be controlled externally rather than sampled online.
    """

    def __init__(
        self,
        data_root: str,
        manifest_json: str,
        hw: Tuple[int, int] = (1024, 2048),
        *,
        color_aug: bool = False,
        color_aug_strength: float = 1.0,
        return_seq_turn_triplet: bool = False,
        seq_turn_only_k: int = 1,
        seed: int = 1234,
    ):
        self.data_root = data_root
        self.hw = hw
        self.color_aug = bool(color_aug)
        self.color_aug_strength = float(color_aug_strength)
        self.return_seq_turn_triplet = bool(return_seq_turn_triplet)
        self.seq_turn_only_k = int(seq_turn_only_k)
        self.seed = int(seed)
        path = os.path.expanduser(str(manifest_json))
        payload = json.loads(open(path, "r", encoding="utf-8").read())
        if isinstance(payload, dict):
            items = payload.get("pairs", [])
            self.split_summary = dict(payload.get("split_summary", {}))
        elif isinstance(payload, list):
            items = payload
            self.split_summary = {}
        else:
            raise TypeError(f"Unsupported fixed-pair manifest payload: {type(payload)}")
        self.pairs_meta: List[Dict[str, Any]] = []
        self.sequence_keys = sorted({(str(x["scene"]), str(x["seq"])) for x in items})
        seq_frames = _scan_seq_frames(data_root=data_root, scenes=None, seqs=None)
        self.seqs_data: List[Dict[str, Any]] = []
        seq_to_idx: Dict[Tuple[str, str], int] = {}
        for scene, seq in self.sequence_keys:
            frames = seq_frames.get((scene, seq), None)
            if not frames:
                raise FileNotFoundError(f"Missing frames for fixed-pair sequence: {(scene, seq)}")
            n = len(frames)
            R_w = np.zeros((n, 3, 3), dtype=np.float32)
            t_w = np.zeros((n, 3), dtype=np.float32)
            pano_list: List[str] = []
            ts_list: List[str] = []
            for i, fr in enumerate(frames):
                R_i, t_i = _parse_label_13(fr.label_path)
                R_w[i] = R_i
                t_w[i] = t_i
                pano_list.append(fr.pano_path)
                ts_list.append(fr.ts_str)
            seq_to_idx[(scene, seq)] = len(self.seqs_data)
            self.seqs_data.append({
                "scene": scene,
                "seq": seq,
                "n": n,
                "R_w": R_w,
                "t_w": t_w,
                "pano": pano_list,
                "ts": ts_list,
            })
        for item in items:
            scene = str(item["scene"])
            seq = str(item["seq"])
            seq_idx = seq_to_idx[(scene, seq)]
            meta = dict(item)
            meta["seq_idx"] = int(seq_idx)
            self.pairs_meta.append(meta)
        if not self.pairs_meta:
            raise RuntimeError("Empty fixed-pair training manifest.")
        self.split_summary.setdefault("num_pairs", len(self.pairs_meta))
        self.split_summary.setdefault("num_sequences_after_filter", len(self.sequence_keys))
        self._rng = np.random.default_rng(self.seed)

    def _get_rng(self):
        info = get_worker_info()
        if info is None:
            return self._rng
        if not hasattr(self, "_worker_rng"):
            wseed = (self.seed + 1000003 * info.id) % (2**32)
            self._worker_rng = np.random.default_rng(wseed)
        return self._worker_rng

    def _apply_color_aug(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if (not self.color_aug) or self.color_aug_strength <= 0.0:
            return img
        s = float(self.color_aug_strength)
        brightness = float(rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s))
        contrast = float(rng.uniform(1.0 - 0.12 * s, 1.0 + 0.12 * s))
        gamma = float(rng.uniform(1.0 - 0.08 * s, 1.0 + 0.08 * s))
        noise_std = 0.008 * s
        out = img.float()
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * contrast + mean
        out = out * brightness
        out = out.clamp(0.0, 1.0).pow(gamma)
        if noise_std > 0.0:
            noise = torch.from_numpy(rng.normal(0.0, noise_std, size=tuple(out.shape)).astype(np.float32))
            out = out + noise
        return out.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.pairs_meta)

    def manifest(self) -> List[Dict[str, Any]]:
        return list(self.pairs_meta)

    def __getitem__(self, idx: int):
        rng = self._get_rng()
        meta = self.pairs_meta[idx]
        sd = self.seqs_data[int(meta["seq_idx"])]
        i = int(meta["i"])
        j = int(meta["j"])
        k = int(meta["k"])
        IA = _read_pano_rgb(sd["pano"][i], self.hw)
        IB = _read_pano_rgb(sd["pano"][j], self.hw)
        IA = self._apply_color_aug(IA, rng)
        IB = self._apply_color_aug(IB, rng)
        R_wA = sd["R_w"][i]
        t_wA = sd["t_w"][i]
        R_wB = sd["R_w"][j]
        t_wB = sd["t_w"][j]
        R_BA = (R_wB.T @ R_wA).astype(np.float32)
        t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
        t_mag = float(np.linalg.norm(t_BA))
        t_dir = t_BA / (t_mag + 1e-8)
        if self.return_seq_turn_triplet and int(k) == int(self.seq_turn_only_k) and (j + 1) < int(sd["n"]):
            R_wC = sd["R_w"][j + 1]
            t_wC = sd["t_w"][j + 1]
            R_BC = (R_wC.T @ R_wB).astype(np.float32)
            t_BC = (R_wC.T @ (t_wB - t_wC)).astype(np.float32)
            t_BC_mag = float(np.linalg.norm(t_BC))
            t_BC_dir = t_BC / (t_BC_mag + 1e-8)
            IC = _read_pano_rgb(sd["pano"][j + 1], self.hw)
            IC = self._apply_color_aug(IC, rng)
            has_seq_turn_triplet = True
        else:
            IC = torch.zeros_like(IA)
            R_BC = np.eye(3, dtype=np.float32)
            t_BC = np.zeros(3, dtype=np.float32)
            t_BC_mag = 0.0
            t_BC_dir = np.zeros(3, dtype=np.float32)
            has_seq_turn_triplet = False
        meta_out = {
            "scene": str(meta["scene"]),
            "seq": str(meta["seq"]),
            "tsA": sd["ts"][i],
            "tsB": sd["ts"][j],
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "dt_world": float(meta.get("dt_world", t_mag)),
            "dt_world_bc": float(t_BC_mag),
            "has_seq_turn_triplet": bool(has_seq_turn_triplet),
            "seq_turn_only_k": int(self.seq_turn_only_k),
        }
        for extra_key in (
            "pred_tmag",
            "gt_tmag",
            "pred_tmag_bucket",
            "gt_tmag_bucket",
            "dt_bucket",
            "k_bucket",
            "dt_k_bucket",
            "hard_bucket",
            "weight_tag",
        ):
            if extra_key in meta:
                meta_out[extra_key] = meta[extra_key]
        return {
            "IA": IA,
            "IB": IB,
            "IC": IC,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_vec": torch.from_numpy(t_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "t_gt_mag": torch.tensor(t_mag, dtype=torch.float32),
            "R_gt_bc": torch.from_numpy(R_BC),
            "t_gt_bc_vec": torch.from_numpy(t_BC),
            "t_gt_bc_dir": torch.from_numpy(t_BC_dir),
            "t_gt_bc_mag": torch.tensor(t_BC_mag, dtype=torch.float32),
            "meta": meta_out,
        }
