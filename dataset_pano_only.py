# dataset_pano_only.py
import os
import re
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, get_worker_info


@dataclass
class FrameRec:
    scene: str
    seq: str
    ts_str: str
    ts_val: float
    pano_path: str
    label_path: str


def _read_pano_rgb(path: str, hw: Tuple[int, int]) -> torch.Tensor:
    """Return float32 tensor [3,H,W] in [0,1]. Keep original resolution if matches."""
    img = Image.open(path).convert("RGB")
    H, W = hw
    if img.size != (W, H):
        img = img.resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))               # [3,H,W]
    return torch.from_numpy(arr)


def _rpy_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Assume degrees. R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _parse_label_13(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    label 一行 13 个数：
    [ts, roll, pitch, yaw, x, y, z, vx, vy, vz, wx, wy, wz]
    只取 rpy(度) 与 xyz(米)。
    """
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    v = np.array([float(x) for x in nums], dtype=np.float32)
    if v.size < 7:
        raise ValueError(f"Bad label file (need >=7 numbers): {path}")

    roll, pitch, yaw = float(v[1]), float(v[2]), float(v[3])  # degrees
    x, y, z = float(v[4]), float(v[5]), float(v[6])           # meters

    # 根据说明：BCS relative to NED => body->world
    R_wb = _rpy_to_R_zyx(roll, pitch, yaw)                    # NED<-BCS
    t_w = np.array([x, y, z], dtype=np.float32)
    return R_wb, t_w


def _relative_pose_A_to_B_in_B(
    R_wA: np.ndarray,
    t_wA: np.ndarray,
    R_wB: np.ndarray,
    t_wB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If label gives body->world: p_w = R_wX p_X + t_wX
    Then relative (A->B) in B frame:
      R_BA = R_wB^T R_wA
      t_BA = R_wB^T (t_wA - t_wB)
    """
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


def _scan_seq_frames(
    data_root: str,
    scenes: Optional[List[str]] = None,
    seqs: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], List[FrameRec]]:
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


def _partition_seq_keys(
    seq_keys: List[Tuple[str, str]],
    split: Optional[str],
    split_by: str,
    train_ratio: float,
    split_seed: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Deterministically split complete groups into train/test.

    split_by='scene_seq' => group is a full (scene, seq) folder.
    split_by='scene'     => all seq under one scene are kept together.
    """
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
        raise RuntimeError(
            f"Need at least 2 groups to split train/test, but only found {n_groups} group(s) with split_by={mode}."
        )

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
    """
    只使用 PanoramaView 下的 panorama_*.jpg + label_*.txt（同目录同 ts）。
    配对规则：同一 scene/seq 内按 ts 排序，取 (i, i+k_stride)。

    split='train'/'test' 时，会按完整的 (scene, seq) 或完整 scene 作真正划分，
    不会把同一个组里的 pair 同时放进 train/test。
    """
    def __init__(
        self,
        data_root: str,
        split: str | None = None,
        hw: Tuple[int, int] = (1024, 2048),
        k_stride: int = 5,
        pair_step: int = 1,
        scenes: Optional[List[str]] = None,
        seqs: Optional[List[str]] = None,
        split_by: str = "scene_seq",
        train_ratio: float = 0.8,
        split_seed: int = 3407,
    ):
        self.data_root = data_root
        self.hw = hw
        self.k_stride = int(k_stride)
        self.pair_step = int(pair_step)
        self.split = split
        self.split_by = _normalize_split_mode(split_by)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)

        seq_frames = _scan_seq_frames(data_root=data_root, scenes=scenes, seqs=seqs)
        selected_seq_keys, self.split_summary = _partition_seq_keys(
            seq_keys=list(seq_frames.keys()),
            split=split,
            split_by=self.split_by,
            train_ratio=self.train_ratio,
            split_seed=self.split_seed,
        )
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
            raise RuntimeError(
                f"No valid pairs built for split={split!r}. Check split ratio, k_stride, and label existence."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        A, B = self.pairs[idx]

        IA = _read_pano_rgb(A.pano_path, self.hw)  # [3,H,W]
        IB = _read_pano_rgb(B.pano_path, self.hw)

        R_wA, t_wA = _parse_label_13(A.label_path)
        R_wB, t_wB = _parse_label_13(B.label_path)

        R_BA, t_BA = _relative_pose_A_to_B_in_B(R_wA, t_wA, R_wB, t_wB)
        t_dir = t_BA / (np.linalg.norm(t_BA) + 1e-8)

        return {
            "IA": IA,
            "IB": IB,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "meta": {"scene": A.scene, "seq": A.seq, "tsA": A.ts_str, "tsB": B.ts_str},
        }


class RflyPanoPanoramaPairsMixedK(Dataset):
    """
    PanoramaView-only dataset with mixed-k sampling.

    - Pre-scan frames in (scene,seq), parse label once for each frame (R_wb, t_w).
    - Build base indices i where i+max_k is valid.
    - __getitem__: sample k ~ categorical(k_choices, k_probs), return pair (i, i+k).
    - Optional: reject pairs with too small / too large world displacement.

    关键点：split='train'/'test' 时，先按完整 group 划分，再在各自 split 内构造 pairs，
    从而避免同一 (scene, seq) 泄漏到 train/test 两边。
    """
    def __init__(
        self,
        data_root: str,
        hw: Tuple[int, int] = (1024, 2048),
        k_choices: List[int] = [5, 10],
        k_probs: Optional[List[float]] = None,
        pair_step: int = 1,
        scenes: Optional[List[str]] = None,
        seqs: Optional[List[str]] = None,
        min_dt: float = 0.0,
        max_tries: int = 10,
        seed: int = 1234,
        # ---- compatibility args used by train_mvp.py ----
        split: str | None = None,
        H: int | None = None,
        W: int | None = None,
        strict_dt: bool | None = None,
        # ---- real group split args ----
        split_by: str = "scene_seq",
        train_ratio: float = 0.8,
        split_seed: int = 3407,
    ):
        self.data_root = data_root
        if (H is not None) and (W is not None):
            hw = (int(H), int(W))
        self.hw = hw
        self.pair_step = int(pair_step)
        self.split = split
        self.strict_dt = strict_dt
        self.split_by = _normalize_split_mode(split_by)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)

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
        selected_seq_keys, self.split_summary = _partition_seq_keys(
            seq_keys=list(seq_frames.keys()),
            split=split,
            split_by=self.split_by,
            train_ratio=self.train_ratio,
            split_seed=self.split_seed,
        )
        selected_seq_key_set = set(selected_seq_keys)
        self.sequence_keys = sorted(selected_seq_key_set)

        # --- pack sequences & pre-parse labels once ---
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
                "scene": scene,
                "seq": seq,
                "n": n,
                "R_w": R_w,
                "t_w": t_w,
                "pano": pano_list,
                "ts": ts_list,
                "valid_i": valid_i,
            })

        if len(self.seqs_data) == 0:
            raise RuntimeError(
                f"No valid sequences for split={split!r} after group split. Check train_ratio, split_by, scenes/seqs, or max_k."
            )

        self.index_map: List[Tuple[int, int]] = []
        for sidx, sd in enumerate(self.seqs_data):
            for li in sd["valid_i"]:
                self.index_map.append((sidx, li))

        self.split_summary["num_pairs"] = len(self.index_map)
        self.split_summary["num_sequences_after_filter"] = len(self.seqs_data)
        if len(self.index_map) == 0:
            raise RuntimeError(
                f"No valid base indices for split={split!r}. Check k_choices/max_k, pair_step, and sequence lengths."
            )

        self._rng = np.random.default_rng(self.seed)

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
        """
        Strict min_dt (and optional max_dt) sampling.
        - Never falls back to dt < min_dt when min_dt > 0.
        - If a given (seq, i) cannot find valid k, it resamples within/beyond the split only.
        """
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
                f"MixedK strict dt sampling failed after {GLOBAL_TRIES} attempts. "
                f"split={self.split!r} split_by={self.split_by} min_dt={self.min_dt} max_dt={max_dt}. "
                f"Try lowering min_dt, disabling max_dt, or using a less aggressive split."
            )

        seq_idx, i, j, k, dt_world = chosen
        sd = self.seqs_data[seq_idx]

        IA = _read_pano_rgb(sd["pano"][i], self.hw)
        IB = _read_pano_rgb(sd["pano"][j], self.hw)

        R_wA = sd["R_w"][i]
        t_wA = sd["t_w"][i]
        R_wB = sd["R_w"][j]
        t_wB = sd["t_w"][j]

        R_BA = (R_wB.T @ R_wA).astype(np.float32)
        t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
        t_dir = t_BA / (np.linalg.norm(t_BA) + 1e-8)

        return {
            "IA": IA,
            "IB": IB,
            "R_gt": torch.from_numpy(R_BA),
            "t_gt_dir": torch.from_numpy(t_dir),
            "meta": {
                "scene": sd["scene"],
                "seq": sd["seq"],
                "tsA": sd["ts"][i],
                "tsB": sd["ts"][j],
                "k": int(k),
                "dt_world": float(dt_world),
            },
        }
