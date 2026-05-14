#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


OFFICIAL_S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}
ORBSLAM3_REFERENCE = {
    "coverage": "273/454",
    "tracking_success_rate": 0.6013215859030837,
    "none": {
        "ate": 11.46685113827757,
        "drift": 0.03737053832593786,
        "path_ratio": 0.2998258665660257,
    },
    "se3": {
        "ate": 0.30854441069248173,
        "drift": 0.0374760642069601,
        "path_ratio": 0.2998258665660257,
    },
    "sim3": {
        "ate": 0.224292165986624,
        "drift": 0.06452708470276013,
        "path_ratio": 0.2998258665660257,
    },
}
RESTORED_S5_DENSE_REFERENCE = {
    "se3_ate": 8.231468716451547,
    "sim3_ate": 4.07912293550008,
    "path_ratio": 2.777267572676944,
}
S5E1_SELECTED_REFERENCE = {
    "rot_mean_deg": 20.835182098696297,
    "tdir_mean_deg": 99.54956140857581,
    "tdir_abs_mean_deg": 41.117886773689875,
    "tmag_median_ratio": 0.9819414718338607,
    "tmag_p90_ratio": 5.680931529985728,
}
DEFAULT_CONFIG_PATH = "configs/s5e2_adjacent_dense_candidate.yaml"
DEFAULT_CANDIDATE_REPORT = "reports/s5e2_adjacent_dense_candidate_report.md"
DEFAULT_COMPARISON_REPORT = "reports/s5e2_orbslam3_external_comparison.md"
FINAL_CLASSIFICATIONS = [
    "S5E2_TRACEABLE_DENSE_IMPROVED",
    "S5E2_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E2_TRACEABLE_DENSE_EXPORTED_FAILED_METRICS",
    "S5E2_ADJACENT_DATASET_BLOCKED",
    "S5E2_TRAINING_BLOCKED",
    "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED",
    "S5E2_ERROR",
]
DATASET_CLASSIFICATIONS = [
    "S5E2_ADJACENT_DATASET_READY",
    "S5E2_ADJACENT_DATASET_BLOCKED",
    "S5E2_SPLIT_RISK_BLOCKED",
    "S5E2_DATASET_ERROR",
]
TRAINING_CLASSIFICATIONS = [
    "S5E2_TRAINING_COMPLETE",
    "S5E2_TRAINING_SMOKE_ONLY",
    "S5E2_TRAINING_BLOCKED",
    "S5E2_TRAINING_ERROR",
]


@dataclass(frozen=True)
class FrameRec:
    scene: str
    seq: str
    ts_str: str
    timestamp: float
    image_path: str
    label_path: str


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_timestamps(path: Path) -> List[float]:
    values: List[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            values.append(float(line.split()[0]))
    return values


def rpy_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def parse_label_pose(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    v = np.asarray([float(x) for x in nums], dtype=np.float64)
    if v.size < 7:
        raise ValueError(f"Bad label file: {path}")
    R_wb = rpy_to_R_zyx(float(v[1]), float(v[2]), float(v[3]))
    t_w = np.asarray([float(v[4]), float(v[5]), float(v[6])], dtype=np.float64)
    return R_wb, t_w


def relative_pose_A_to_B_in_B(a: FrameRec, b: FrameRec) -> Tuple[np.ndarray, np.ndarray]:
    R_wA, t_wA = parse_label_pose(Path(a.label_path))
    R_wB, t_wB = parse_label_pose(Path(b.label_path))
    R_BA = R_wB.T @ R_wA
    t_BA = R_wB.T @ (t_wA - t_wB)
    return R_BA, t_BA


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_tum(path: Path) -> Dict[float, Dict[str, np.ndarray]]:
    rows: Dict[float, Dict[str, np.ndarray]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        rows[ts] = {
            "t": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
            "R": quat_xyzw_to_rot(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        }
    return rows


def scan_frames(data_root: Path, scene: Optional[str] = None, seq: Optional[str] = None) -> Dict[Tuple[str, str], List[FrameRec]]:
    root = data_root / "PanoramaView"
    out: Dict[Tuple[str, str], List[FrameRec]] = {}
    for img_path in sorted(root.glob("scene*/seq*/panorama_*.jpg")):
        seq_name = img_path.parent.name
        scene_name = img_path.parent.parent.name
        if scene is not None and scene_name != scene:
            continue
        if seq is not None and seq_name != seq:
            continue
        ts_str = img_path.stem[len("panorama_") :]
        try:
            timestamp = float(ts_str)
        except ValueError:
            continue
        label_path = img_path.parent / f"label_{ts_str}.txt"
        if not label_path.exists():
            continue
        out.setdefault((scene_name, seq_name), []).append(
            FrameRec(scene_name, seq_name, ts_str, timestamp, str(img_path), str(label_path))
        )
    for key in list(out):
        out[key].sort(key=lambda rec: rec.timestamp)
    return out


def split_sequence_keys(seq_keys: Sequence[Tuple[str, str]], train_ratio: float = 0.8, split_seed: int = 3407) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    keys = sorted(set(seq_keys))
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(keys))
    n_train = int(round(len(keys) * float(train_ratio)))
    n_train = max(1, min(len(keys) - 1, n_train))
    train = sorted(keys[int(i)] for i in perm[:n_train])
    test = sorted(keys[int(i)] for i in perm[n_train:])
    return train, test, {
        "split_by": "scene_seq",
        "train_ratio": float(train_ratio),
        "split_seed": int(split_seed),
        "train_sequence_keys": train,
        "test_sequence_keys": test,
        "all_sequence_keys": keys,
    }


def adjacent_pairs(frames: Sequence[FrameRec]) -> List[Tuple[FrameRec, FrameRec]]:
    return [(frames[i], frames[i + 1]) for i in range(max(0, len(frames) - 1))]


def image_stats(path: str, size: Tuple[int, int] = (32, 64)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size[1], size[0]), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    chans = [arr[:, :, c].reshape(-1) for c in range(3)]
    feats: List[float] = []
    for c in chans:
        feats.extend([float(c.mean()), float(c.std()), float(np.percentile(c, 10)), float(np.percentile(c, 50)), float(np.percentile(c, 90))])
    h, w, _ = arr.shape
    patches = [
        arr[: h // 2, : w // 2],
        arr[: h // 2, w // 2 :],
        arr[h // 2 :, : w // 2],
        arr[h // 2 :, w // 2 :],
    ]
    for p in patches:
        feats.extend([float(x) for x in p.mean(axis=(0, 1))])
    return np.asarray(feats, dtype=np.float64)


def pair_features(a: FrameRec, b: FrameRec, edge_index: int, total_edges: int, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if a.image_path not in cache:
        cache[a.image_path] = image_stats(a.image_path)
    if b.image_path not in cache:
        cache[b.image_path] = image_stats(b.image_path)
    fa = cache[a.image_path]
    fb = cache[b.image_path]
    dt = float(b.timestamp - a.timestamp)
    return np.concatenate(
        [
            fa,
            fb,
            fb - fa,
            np.asarray([dt, edge_index / max(float(total_edges), 1.0), 1.0], dtype=np.float64),
        ]
    )


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    if theta < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    v = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return theta / (2.0 * math.sin(theta)) * v


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1.0e-10:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    K = np.asarray([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rot_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    return tuple(float(x) for x in q)


def vector_angle_deg(a: np.ndarray, b: np.ndarray, absolute: bool = False) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    c = float(np.dot(a, b) / (na * nb))
    if absolute:
        c = abs(c)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def angle_deg_from_rot(delta: np.ndarray) -> float:
    c = (float(np.trace(delta)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def summarize_values(values: Iterable[Optional[float]], prefix: str) -> Dict[str, Optional[float]]:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_p90": None}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_median": float(np.percentile(vals, 50)),
        f"{prefix}_p90": float(np.percentile(vals, 90)),
    }


def base_checkpoint() -> Dict[str, Any]:
    return {
        "experiment": "S5E2_real_adjacent_dense_candidate",
        "status": {
            "experimental_candidate": True,
            "official_s5_unchanged": True,
            "not_official_replacement": True,
        },
        "s5e1_blocker_recap": {},
        "dataset": {
            "adjacent_dataset_ready": None,
            "split_respected": None,
            "test_gt_used_for_training": False,
            "num_train_pairs": None,
            "num_val_pairs": None,
        },
        "training": {
            "attempted": False,
            "classification": None,
            "checkpoint_dir": "checkpoints/S5E2_adjacent_dense_candidate",
            "training_status": "checkpoints/S5E2_adjacent_dense_candidate/training_status.json",
        },
        "adjacent_dense_export": {
            "available": False,
            "trajectory_path": "external_baselines/results/s5e2_traceable_dense/scene01_seq03_s5e2_traceable_dense_tum.txt",
            "edge_provenance": "external_baselines/results/s5e2_traceable_dense/edge_provenance.jsonl",
            "num_poses": None,
            "num_edges": None,
            "direct_adjacent_prediction_edges": None,
            "coverage": None,
            "all_edges_traceable": False,
        },
        "component_metrics": {
            "rot_mean_deg": None,
            "tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "tmag_median_ratio": None,
            "tmag_p90_ratio": None,
            "path_ratio": None,
        },
        "external_eval": {
            "none": {"ate": None, "drift": None, "path_ratio": None},
            "se3": {"ate": None, "drift": None, "path_ratio": None},
            "sim3": {"ate": None, "drift": None, "path_ratio": None},
        },
        "comparison_to_orbslam3": {
            "orbslam3_coverage": "273/454",
            "s5e2_coverage_advantage": None,
            "aligned_ate_gap_to_orbslam3": None,
            "path_ratio_comparison": "",
            "summary": "",
        },
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5e2": True,
        },
        "validation": validation_from_logs(),
        "allowed_final_classifications": FINAL_CLASSIFICATIONS,
        "final_classification": "S5E2_ERROR",
    }


def load_or_base_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        ckpt = read_json(path)
        base = base_checkpoint()
        base.update(ckpt)
        return base
    return base_checkpoint()


def validation_from_logs() -> Dict[str, str]:
    paths = {
        "verify_final_candidate": Path("logs/s5d11_verify_final_candidate_serial.log"),
        "project_health_check": Path("logs/s5d11_project_health_check_serial.log"),
        "s6_eval_only": Path("logs/s5d11_s6_eval_only_serial.log"),
        "unittest": Path("logs/s5d11_unittest.log"),
    }
    out: Dict[str, str] = {}
    for key, path in paths.items():
        if not path.exists():
            out[key] = "not_run"
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        out[key] = "FAIL" if "FAILED" in text or "Traceback" in text or "ERROR" in text else "PASS"
    return out
