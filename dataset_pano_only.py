# dataset_pano_only.py
import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


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

    # 根据你给的说明：BCS relative to NED => body->world
    R_wb = _rpy_to_R_zyx(roll, pitch, yaw)                    # NED<-BCS
    t_w = np.array([x, y, z], dtype=np.float32)
    return R_wb, t_w


def _relative_pose_A_to_B_in_B(R_wA: np.ndarray, t_wA: np.ndarray,
                              R_wB: np.ndarray, t_wB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If label gives body->world: p_w = R_wX p_X + t_wX
    Then relative (A->B) in B frame:
      R_BA = R_wB^T R_wA
      t_BA = R_wB^T (t_wA - t_wB)
    """
    R_BA = (R_wB.T @ R_wA).astype(np.float32)
    t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
    return R_BA, t_BA


class RflyPanoPanoramaPairs(Dataset):
    """
    只使用 PanoramaView 下的 panorama_*.jpg + label_*.txt（同目录同 ts）。
    配对规则：同一 scene/seq 内按 ts 排序，取 (i, i+k_stride)。
    """
    def __init__(
        self,
        data_root: str,
        hw: Tuple[int, int] = (1024, 2048),
        k_stride: int = 5,
        pair_step: int = 1,
        scenes: Optional[List[str]] = None,
        seqs: Optional[List[str]] = None,
    ):
        self.data_root = data_root
        self.hw = hw
        self.k_stride = int(k_stride)
        self.pair_step = int(pair_step)

        pano_root = os.path.join(data_root, "PanoramaView")
        pattern = os.path.join(pano_root, "scene*", "seq*", "panorama_*.jpg")
        pano_paths = sorted(glob.glob(pattern))
        if len(pano_paths) == 0:
            raise FileNotFoundError(f"No panorama_*.jpg found under: {pattern}")

        # index frames per seq
        seq_frames: Dict[Tuple[str, str], List[FrameRec]] = {}

        for pano_path in pano_paths:
            seq_dir = os.path.dirname(pano_path)
            seq = os.path.basename(seq_dir)
            scene = os.path.basename(os.path.dirname(seq_dir))

            if scenes is not None and scene not in scenes:
                continue
            if seqs is not None and seq not in seqs:
                continue

            base = os.path.basename(pano_path)
            ts_str = base[len("panorama_"):-len(".jpg")]
            try:
                ts_val = float(ts_str)
            except Exception:
                continue

            # label 与 pano 同目录
            label_path = os.path.join(seq_dir, f"label_{ts_str}.txt")
            if not os.path.exists(label_path):
                continue

            rec = FrameRec(scene, seq, ts_str, ts_val, pano_path, label_path)
            seq_frames.setdefault((scene, seq), []).append(rec)

        # sort by timestamp
        for k in list(seq_frames.keys()):
            seq_frames[k].sort(key=lambda r: r.ts_val)

        # build pairs
        self.pairs: List[Tuple[FrameRec, FrameRec]] = []
        for (scene, seq), recs in seq_frames.items():
            n = len(recs)
            if n <= self.k_stride:
                continue
            for i in range(0, n - self.k_stride, self.pair_step):
                self.pairs.append((recs[i], recs[i + self.k_stride]))

        if len(self.pairs) == 0:
            raise RuntimeError("No valid pairs built. Check k_stride/pair_step and label existence.")

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
            "R_gt": torch.from_numpy(R_BA),        # [3,3]
            "t_gt_dir": torch.from_numpy(t_dir),   # [3]
            "meta": {"scene": A.scene, "seq": A.seq, "tsA": A.ts_str, "tsB": B.ts_str},
        }




from torch.utils.data import get_worker_info

class RflyPanoPanoramaPairsMixedK(Dataset):
    """
    PanoramaView-only dataset with mixed-k sampling.

    - Pre-scan frames in (scene,seq), parse label once for each frame (R_wb, t_w).
    - Build base indices i where i+max_k is valid.
    - __getitem__: sample k ~ categorical(k_choices, k_probs), return pair (i, i+k).
    - Optional: reject pairs with too small world displacement (min_dt meters).

    Return dict:
      IA: [3,H,W], IB: [3,H,W]
      R_gt: [3,3]  (relative rotation A->B expressed in B frame)
      t_gt_dir: [3] (unit translation direction A->B expressed in B frame)
      meta: includes scene/seq/ts/k/dt_world
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
        min_dt: float = 0.0,            # world displacement threshold (meters), 0 disables
        max_tries: int = 10,            # resample k up to this times to satisfy min_dt
        seed: int = 1234,
    ):
        self.data_root = data_root
        self.hw = hw
        self.pair_step = int(pair_step)
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

        # --- robust pano_root + patterns (jpg/jpeg/png) ---
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
        pano_paths = []
        for pat in patterns:
            pano_paths.extend(glob.glob(pat))
        pano_paths = sorted(set(pano_paths))
        if len(pano_paths) == 0:
            raise FileNotFoundError(f"No panorama images found under pano_root={pano_root}")

        # --- build seq -> frames list ---
        seq_frames: Dict[Tuple[str, str], List[FrameRec]] = {}

        for pano_path in pano_paths:
            seq_dir = os.path.dirname(pano_path)
            seq = os.path.basename(seq_dir)
            scene = os.path.basename(os.path.dirname(seq_dir))

            if scenes is not None and scene not in scenes:
                continue
            if seqs is not None and seq not in seqs:
                continue

            base = os.path.basename(pano_path)
            stem, _ext = os.path.splitext(base)

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

            rec = FrameRec(scene, seq, ts_str, ts_val, pano_path, label_path)
            seq_frames.setdefault((scene, seq), []).append(rec)

        for k in list(seq_frames.keys()):
            seq_frames[k].sort(key=lambda r: r.ts_val)

        # --- pack sequences & pre-parse labels once ---
        self.seqs_data = []  # list of dict
        for (scene, seq), frames in seq_frames.items():
            n = len(frames)
            if n <= self.max_k:
                continue

            R_w = np.zeros((n, 3, 3), dtype=np.float32)
            t_w = np.zeros((n, 3), dtype=np.float32)
            pano_list = []
            ts_list = []

            for i, fr in enumerate(frames):
                R_i, t_i = _parse_label_13(fr.label_path)  # R_wb, t_w
                R_w[i] = R_i
                t_w[i] = t_i
                pano_list.append(fr.pano_path)
                ts_list.append(fr.ts_str)

            # valid base indices i where i+max_k exists
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
            raise RuntimeError("No valid sequences for mixed-k dataset (check scenes/seqs/max_k).")

        # create a flat index map (seq_idx, local_i_index)
        self.index_map = []
        for sidx, sd in enumerate(self.seqs_data):
            for li in sd["valid_i"]:
                self.index_map.append((sidx, li))

        # RNG placeholder (per-process and per-worker)
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_rng(self):
        info = get_worker_info()
        if info is None:
            return self._rng
        # Create one RNG per worker
        if not hasattr(self, "_worker_rng"):
            wseed = (self.seed + 1000003 * info.id) % (2**32)
            self._worker_rng = np.random.default_rng(wseed)
        return self._worker_rng

    def __getitem__(self, idx: int):
        rng = self._get_rng()

        seq_idx, i = self.index_map[idx]
        sd = self.seqs_data[seq_idx]
        n = sd["n"]

        # sample k (and optionally enforce min_dt)
        k = int(rng.choice(self.k_choices, p=self.k_probs))
        j = i + k

        # ensure valid (should always hold because i <= n-max_k-1 and k<=max_k)
        if j >= n:
            j = n - 1
            k = j - i

        # optional: reject too small dt by resampling k
        dt_world = float(np.linalg.norm(sd["t_w"][j] - sd["t_w"][i]))
        if self.min_dt > 0.0:
            tries = 0
            while dt_world < self.min_dt and tries < self.max_tries:
                k = int(rng.choice(self.k_choices, p=self.k_probs))
                j = i + k
                if j >= n:
                    tries += 1
                    continue
                dt_world = float(np.linalg.norm(sd["t_w"][j] - sd["t_w"][i]))
                tries += 1

        # load images
        IA = _read_pano_rgb(sd["pano"][i], self.hw)  # [3,H,W]
        IB = _read_pano_rgb(sd["pano"][j], self.hw)

        # relative pose A->B expressed in B frame
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
            "R_gt": torch.from_numpy(R_BA),        # [3,3]
            "t_gt_dir": torch.from_numpy(t_dir),   # [3]
            "meta": {
                "scene": sd["scene"],
                "seq": sd["seq"],
                "tsA": sd["ts"][i],
                "tsB": sd["ts"][j],
                "k": int(k),
                "dt_world": float(dt_world),
            },
        }
