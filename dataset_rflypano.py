# dataset_rflypano.py
import os
import re
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def _read_image_rgb(path: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Return float tensor in [0,1], shape [3,H,W]."""
    img = Image.open(path).convert("RGB")
    H, W = size_hw
    if (img.height, img.width) != (H, W):
        img = img.resize((W, H), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))  # [3,H,W]
    return torch.from_numpy(arr)


def _rpy_to_R_zyx(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """ZYX order: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
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


def parse_label_pose(path: str, rpy_in_degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    尝试从 label_<timestamp>.txt 解析出 (R_w, t_w).
    支持若干常见格式（按数字总数/特征做启发式判断）：
      - 7 个数：可能是 (qw,qx,qy,qz, tx,ty,tz)
      - 6 个数：可能是 (roll,pitch,yaw, tx,ty,tz)
      - 12 个数：3x4 [R|t]
      - 16 个数：4x4 T
    如果你的 label 不符合这些，需要你在这里按实际格式改一下映射。
    """
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    vals = np.array([float(x) for x in nums], dtype=np.float32)

    if vals.size == 16:
        T = vals.reshape(4, 4)
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        return R, t

    if vals.size == 12:
        M = vals.reshape(3, 4)
        R = M[:3, :3].astype(np.float32)
        t = M[:3, 3].astype(np.float32)
        return R, t

    if vals.size >= 7:
        q = vals[:4]
        qn = np.linalg.norm(q)
        # 近似单位四元数判定
        if 0.8 < qn < 1.2:
            q = q / (qn + 1e-8)
            qw, qx, qy, qz = q.tolist()
            # quaternion -> R (w,x,y,z)
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
            ], dtype=np.float32)
            t = vals[4:7].astype(np.float32)
            return R, t

    if vals.size >= 6:
        roll, pitch, yaw = vals[:3].tolist()
        t = vals[3:6].astype(np.float32)
        R = _rpy_to_R_zyx(roll, pitch, yaw, degrees=rpy_in_degrees)
        return R, t

    raise ValueError(f"Unrecognized label format: {path}, got {vals.size} numbers.")


def relative_pose_from_world(
    R_wA: np.ndarray, t_wA: np.ndarray,
    R_wB: np.ndarray, t_wB: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R_BA, t_BA) where x_B = R_BA x_A + t_BA
    Assuming x_w = R_wX x_X + t_wX
    """
    R_BA = (R_wB.T @ R_wA).astype(np.float32)
    t_BA = (R_wB.T @ (t_wA - t_wB)).astype(np.float32)
    return R_BA, t_BA


@dataclass
class SampleRecord:
    scene: str
    seq: str
    ts_str: str
    pano_path: str
    label_path: str
    ts_val: float


class RflyPanoPairDataset(Dataset):
    """
    生成 (IA, IB, R_gt, t_gt_dir)。
    pairs 默认取相邻帧，也支持随机间隔。
    """
    def __init__(
        self,
        root_dir: str,
        split_scenes: List[str],
        image_hw: Tuple[int, int],
        pair_mode: str = "adjacent",   # "adjacent" or "random"
        max_delta: int = 3,
        rpy_in_degrees: bool = True,
    ):
        self.root_dir = root_dir
        self.image_hw = image_hw
        self.pair_mode = pair_mode
        self.max_delta = max_delta
        self.rpy_in_degrees = rpy_in_degrees

        self.records_by_seq: Dict[Tuple[str, str], List[SampleRecord]] = {}
        self.pairs: List[Tuple[SampleRecord, SampleRecord]] = []

        self._index(split_scenes)
        self._build_pairs()

    def _index(self, split_scenes: List[str]) -> None:
        pano_glob = os.path.join(self.root_dir, "PanoramaView", "scene*", "seq*", "panorama_*.jpg")
        pano_paths = sorted(glob.glob(pano_glob))

        for pano_path in pano_paths:
            # .../PanoramaView/sceneXX/seqYY/panorama_<ts>.jpg
            seq_dir = os.path.dirname(pano_path)
            seq = os.path.basename(seq_dir)
            scene = os.path.basename(os.path.dirname(seq_dir))

            if scene not in split_scenes:
                continue

            name = os.path.basename(pano_path)
            ts_str = name[len("panorama_"):-len(".jpg")]
            try:
                ts_val = float(ts_str)
            except Exception:
                continue

            label_path = os.path.join(self.root_dir, "FisheyeView", scene, seq, f"label_{ts_str}.txt")
            if not os.path.exists(label_path):
                # 如果你的本地 label 命名/格式稍有不同，可以在这里做“最近时间戳匹配”扩展
                continue

            rec = SampleRecord(scene=scene, seq=seq, ts_str=ts_str, pano_path=pano_path, label_path=label_path, ts_val=ts_val)
            key = (scene, seq)
            self.records_by_seq.setdefault(key, []).append(rec)

        # sort by timestamp within each seq
        for k in list(self.records_by_seq.keys()):
            self.records_by_seq[k].sort(key=lambda r: r.ts_val)

    def _build_pairs(self) -> None:
        rng = random.Random(1234)
        for key, recs in self.records_by_seq.items():
            if len(recs) < 2:
                continue
            if self.pair_mode == "adjacent":
                for i in range(len(recs) - 1):
                    self.pairs.append((recs[i], recs[i + 1]))
            elif self.pair_mode == "random":
                for i in range(len(recs) - 1):
                    delta = rng.randint(1, min(self.max_delta, len(recs) - 1 - i))
                    self.pairs.append((recs[i], recs[i + delta]))
            else:
                raise ValueError(f"Unknown pair_mode: {self.pair_mode}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        A, B = self.pairs[idx]

        IA = _read_image_rgb(A.pano_path, self.image_hw)  # [3,H,W]
        IB = _read_image_rgb(B.pano_path, self.image_hw)

        R_wA, t_wA = parse_label_pose(A.label_path, rpy_in_degrees=self.rpy_in_degrees)
        R_wB, t_wB = parse_label_pose(B.label_path, rpy_in_degrees=self.rpy_in_degrees)

        R_BA, t_BA = relative_pose_from_world(R_wA, t_wA, R_wB, t_wB)
        t_dir = t_BA / (np.linalg.norm(t_BA) + 1e-8)

        return {
            "IA": IA,  # [3,H,W]
            "IB": IB,
            "R_gt": torch.from_numpy(R_BA),      # [3,3]
            "t_gt_dir": torch.from_numpy(t_dir), # [3]
            "meta": {"scene": A.scene, "seq": A.seq, "tsA": A.ts_str, "tsB": B.ts_str},
        }
