#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from evaluate_external_baseline_trajectory import evaluate_external_baseline_trajectory


DEFAULT_DATASET = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "external_baselines" / "results" / "pano_orb_vo"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "SB1_pano_orb_vo_baseline_results.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "pano_orb_vo_baseline_report.md"
FINAL_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


@dataclass
class FrameRec:
    scene: str
    seq: str
    ts_str: str
    ts_val: float
    pano_path: Path
    label_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _parse_label_13(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import re

    txt = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    vals = np.asarray([float(x) for x in nums], dtype=np.float64)
    if vals.size < 7:
        raise ValueError(f"Bad label file (need >= 7 numbers): {path}")
    roll, pitch, yaw = vals[1], vals[2], vals[3]
    x, y, z = vals[4], vals[5], vals[6]
    rr, pp, yy = np.deg2rad([roll, pitch, yaw])
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pp), math.sin(pp)
    cy, sy = math.cos(yy), math.sin(yy)
    Rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    t = np.asarray([x, y, z], dtype=np.float64)
    return R, t


def _scan_seq_frames(data_root: Path, scene: str | None = None, seq: str | None = None) -> Dict[Tuple[str, str], List[FrameRec]]:
    pano_root = data_root / "PanoramaView" if (data_root / "PanoramaView").is_dir() else data_root
    out: Dict[Tuple[str, str], List[FrameRec]] = {}
    for pano_path in sorted(pano_root.glob("scene*/seq*/panorama_*")):
        seq_name = pano_path.parent.name
        scene_name = pano_path.parent.parent.name
        if scene is not None and scene_name != scene:
            continue
        if seq is not None and seq_name != seq:
            continue
        ts_str = pano_path.stem[len("panorama_") :]
        try:
            ts_val = float(ts_str)
        except Exception:
            continue
        label_path = pano_path.parent / f"label_{ts_str}.txt"
        if not label_path.is_file():
            continue
        out.setdefault((scene_name, seq_name), []).append(
            FrameRec(scene_name, seq_name, ts_str, ts_val, pano_path, label_path)
        )
    for key in list(out.keys()):
        out[key].sort(key=lambda fr: fr.ts_val)
    return out


def _partition_seq_keys(
    seq_keys: Sequence[Tuple[str, str]],
    split: str,
    split_by: str,
    train_ratio: float,
    split_seed: int,
) -> List[Tuple[str, str]]:
    keys = sorted(set((str(a), str(b)) for a, b in seq_keys))
    if split_by != "scene_seq":
        raise ValueError(f"Unsupported split_by for this tool: {split_by!r}")
    groups = list(keys)
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(len(groups))
    n_train = int(round(len(groups) * float(train_ratio)))
    n_train = max(1, min(len(groups) - 1, n_train))
    train_idx = set(int(x) for x in perm[:n_train])
    if split == "train":
        return [groups[i] for i in range(len(groups)) if i in train_idx]
    if split == "test":
        return [groups[i] for i in range(len(groups)) if i not in train_idx]
    raise ValueError(f"Unsupported split={split!r}")


def _load_dataset_records(dataset_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    image_list_path = dataset_dir / "image_list.txt"
    gt_path = dataset_dir / "groundtruth_tum.txt"
    metadata_path = dataset_dir / "metadata.json"
    if not image_list_path.is_file():
        raise FileNotFoundError(f"missing image_list: {image_list_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing groundtruth_tum: {gt_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    rows: List[Dict[str, Any]] = []
    for raw in image_list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        ts_str, path_str = line.split(maxsplit=1)
        path = Path(path_str)
        rows.append({"timestamp": float(ts_str), "ts_str": ts_str, "image_path": path})
    return rows, metadata


def _rotation_matrix_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_yaw = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    R_pitch = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return R_yaw @ R_pitch


def _build_view_grid(
    pano_w: int,
    pano_h: int,
    view_w: int,
    view_h: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = 0.5 * float(view_w) / math.tan(math.radians(float(fov_deg)) * 0.5)
    fy = fx
    cx = 0.5 * float(view_w)
    cy = 0.5 * float(view_h)
    uu, vv = np.meshgrid(np.arange(view_w, dtype=np.float32), np.arange(view_h, dtype=np.float32))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    dirs = np.stack([x, y, z], axis=-1).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1.0e-12
    R = _rotation_matrix_yaw_pitch(yaw_deg, pitch_deg)
    dirs_world = dirs @ R.T
    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))
    map_x = ((lon / (2.0 * math.pi)) + 0.5) * float(pano_w)
    map_y = (0.5 - (lat / math.pi)) * float(pano_h)
    return map_x.astype(np.float32), map_y.astype(np.float32), np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_xyzw_from_rot(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class PanoViewCache:
    def __init__(self, view_specs: Sequence[Tuple[float, float]], view_w: int, view_h: int, fov_deg: float) -> None:
        self.view_specs = list(view_specs)
        self.view_w = int(view_w)
        self.view_h = int(view_h)
        self.fov_deg = float(fov_deg)
        self._grids: Dict[Tuple[int, int, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cache: Dict[str, Tuple[List[np.ndarray], np.ndarray]] = {}

    def _grid(self, pano_shape: Tuple[int, int], yaw_deg: float, pitch_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (int(pano_shape[0]), int(pano_shape[1]), float(yaw_deg), float(pitch_deg))
        if key not in self._grids:
            self._grids[key] = _build_view_grid(
                pano_w=int(pano_shape[1]),
                pano_h=int(pano_shape[0]),
                view_w=self.view_w,
                view_h=self.view_h,
                fov_deg=self.fov_deg,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
            )
        return self._grids[key]

    def get_views(self, image_path: Path) -> Tuple[List[np.ndarray], np.ndarray]:
        key = str(image_path.resolve())
        if key in self._cache:
            return self._cache[key]
        pano = np.asarray(Image.open(image_path).convert("RGB"))
        pano_bgr = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
        views: List[np.ndarray] = []
        K_ref = None
        for yaw_deg, pitch_deg in self.view_specs:
            map_x, map_y, K = self._grid((pano_bgr.shape[0], pano_bgr.shape[1]), yaw_deg, pitch_deg)
            view = cv2.remap(pano_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            views.append(gray)
            K_ref = K
        while len(self._cache) >= 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (views, K_ref)
        return views, K_ref


def _traj_rows_to_tum(rows: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        qx, qy, qz, qw = _quat_xyzw_from_rot(np.asarray(row["R"], dtype=np.float64))
        t = np.asarray(row["t"], dtype=np.float64).reshape(3)
        lines.append(
            f"{float(row['timestamp']):.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    return "\n".join(lines) + "\n"


def _compute_train_median_scale(metadata: Dict[str, Any]) -> Dict[str, Any]:
    data_root = _resolve_path(metadata["data_root"], REPO_ROOT / metadata["data_root"])
    split_cfg = metadata["historical_split"]
    seq_frames = _scan_seq_frames(data_root)
    train_keys = _partition_seq_keys(
        seq_frames.keys(),
        "train",
        str(split_cfg["split_by"]),
        float(split_cfg["train_ratio"]),
        int(split_cfg["split_seed"]),
    )
    dists: List[float] = []
    for key in train_keys:
        frames = seq_frames.get(tuple(key), [])
        if len(frames) < 2:
            continue
        prev_R, prev_t = _parse_label_13(frames[0].label_path)
        for fr in frames[1:]:
            R, t = _parse_label_13(fr.label_path)
            dist = float(np.linalg.norm(t - prev_t))
            if math.isfinite(dist) and dist > 0.0:
                dists.append(dist)
            prev_R, prev_t = R, t
    if not dists:
        raise RuntimeError("Could not compute train-only median scale from historical train sequences.")
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "policy_name": "train_median_k1_step_scale",
        "value": float(np.median(arr)),
        "num_train_steps": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _run_orb_pose_on_pair(
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    views_a: Sequence[np.ndarray],
    views_b: Sequence[np.ndarray],
    view_specs: Sequence[Tuple[float, float]],
    K: np.ndarray,
    prev_tdir: np.ndarray,
) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for view_idx, ((yaw_deg, pitch_deg), img_a, img_b) in enumerate(zip(view_specs, views_a, views_b)):
        kp_a, desc_a = orb.detectAndCompute(img_a, None)
        kp_b, desc_b = orb.detectAndCompute(img_b, None)
        if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            cand = {
                "view_idx": int(view_idx),
                "selected_view_yaw": float(yaw_deg),
                "selected_view_pitch": float(pitch_deg),
                "success": False,
                "reason": "insufficient_keypoints",
                "match_count": 0,
                "inlier_count": 0,
            }
        else:
            matches_knn = matcher.knnMatch(desc_a, desc_b, k=2)
            good = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) < 8:
                cand = {
                    "view_idx": int(view_idx),
                    "selected_view_yaw": float(yaw_deg),
                    "selected_view_pitch": float(pitch_deg),
                    "success": False,
                    "reason": "insufficient_matches",
                    "match_count": int(len(good)),
                    "inlier_count": 0,
                }
            else:
                pts_a = np.asarray([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
                pts_b = np.asarray([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)
                E, mask = cv2.findEssentialMat(
                    pts_a,
                    pts_b,
                    cameraMatrix=K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )
                if E is None:
                    cand = {
                        "view_idx": int(view_idx),
                        "selected_view_yaw": float(yaw_deg),
                        "selected_view_pitch": float(pitch_deg),
                        "success": False,
                        "reason": "essential_matrix_failed",
                        "match_count": int(len(good)),
                        "inlier_count": 0,
                    }
                else:
                    try:
                        inliers, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, cameraMatrix=K)
                        tdir = np.asarray(t, dtype=np.float64).reshape(3)
                        tnorm = float(np.linalg.norm(tdir))
                        if tnorm <= 1.0e-12:
                            raise ValueError("recoverPose returned near-zero translation")
                        tdir /= tnorm
                        cand = {
                            "view_idx": int(view_idx),
                            "selected_view_yaw": float(yaw_deg),
                            "selected_view_pitch": float(pitch_deg),
                            "success": True,
                            "reason": "ok",
                            "match_count": int(len(good)),
                            "inlier_count": int(inliers),
                            "R": np.asarray(R, dtype=np.float64),
                            "tdir": tdir,
                        }
                    except Exception as exc:
                        cand = {
                            "view_idx": int(view_idx),
                            "selected_view_yaw": float(yaw_deg),
                            "selected_view_pitch": float(pitch_deg),
                            "success": False,
                            "reason": f"recover_pose_failed:{type(exc).__name__}",
                            "match_count": int(len(good)),
                            "inlier_count": 0,
                        }
        if best is None or int(cand["inlier_count"]) > int(best["inlier_count"]):
            best = cand
    assert best is not None
    if best["success"]:
        return best
    return {
        **best,
        "fallback_used": True,
        "R": np.eye(3, dtype=np.float64),
        "tdir": np.asarray(prev_tdir, dtype=np.float64).reshape(3),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a protocol-compatible panorama-derived virtual-pinhole ORB VO baseline.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset export directory.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output result directory.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output json path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="Output markdown path.")
    p.add_argument("--fov-deg", type=float, default=90.0, help="Virtual pinhole horizontal FOV.")
    p.add_argument("--view-yaws", default="0,90,180,270", help="Comma-separated yaw angles in degrees.")
    p.add_argument("--view-pitch", type=float, default=0.0, help="Shared pitch angle in degrees.")
    p.add_argument("--view-width", type=int, default=640, help="Virtual view width.")
    p.add_argument("--view-height", type=int, default=480, help="Virtual view height.")
    return p.parse_args()


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for row in payload["result_rows"]:
        rows.append(
            {
                "method": row["method"],
                "alignment": row["alignment"],
                "ATE": row["ATE"],
                "drift": row["drift"],
                "path_ratio": row["path_ratio"],
                "tracking_success_rate": row["tracking_success_rate"],
                "mean_inliers": row["mean_inliers"],
                "median_inliers": row["median_inliers"],
            }
        )
    s5 = payload["s5_locked_metrics"]
    comp_rows = [
        {
            "method": "S5_clean_tmag_calibration_policy",
            "ATE": float(s5["ATE"]),
            "drift": float(s5["drift"]),
            "path_ratio": float(s5["path_ratio"]),
            "notes": "Final locked clean candidate.",
        }
    ]
    for summary in payload["comparison_with_s5"]:
        comp_rows.append(summary)

    def table(rows_: List[Dict[str, Any]], cols: List[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows_:
            vals = []
            for col in cols:
                val = row.get(col, "")
                vals.append(_fmt(val) if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Pano-ORB-VO Strong Baseline",
        "",
        "## Scope",
        "This is a protocol-compatible strong baseline built on panorama-derived virtual pinhole views.",
        "It is not ORB-SLAM3, and it is not an original fisheye-camera baseline.",
        "",
        "## Input",
        f"- sequence: `{payload['dataset_summary']['sequence']}`",
        f"- number of frames: `{payload['dataset_summary']['num_frames']}`",
        f"- virtual camera: `{payload['virtual_camera']['view_width']}x{payload['virtual_camera']['view_height']}`, `fov={payload['virtual_camera']['fov_deg']}` deg, yaws=`{payload['virtual_camera']['view_yaws_deg']}` pitch=`{payload['virtual_camera']['view_pitch_deg']}`",
        f"- virtual intrinsics: `fx=fy={_fmt(payload['virtual_camera']['fx'])}`, `cx={_fmt(payload['virtual_camera']['cx'])}`, `cy={_fmt(payload['virtual_camera']['cy'])}`",
        "- note: these are derived virtual pinhole intrinsics, not original camera parameters.",
        "",
        "## Method",
        "- panorama -> virtual pinhole views",
        "- ORB matching",
        "- essential matrix",
        "- recoverPose",
        "- trajectory composition",
        f"- main scale policy: `{payload['scale_policy']['policy_name']}` (train-only, non-test-GT tuned)",
        "",
        "## Results",
        table(rows, ["method", "alignment", "ATE", "drift", "path_ratio", "tracking_success_rate", "mean_inliers", "median_inliers"]),
        "",
        "## Comparison with S5",
        table(comp_rows, ["method", "ATE", "drift", "path_ratio", "notes"]),
        "",
        "## Interpretation",
        payload["interpretation"],
        "",
        "## Caveats",
        "- derived virtual pinhole input",
        "- not direct original fisheye baseline",
        "- no loop closure",
        "- monocular scale ambiguity",
        "- scale policy not test-GT tuned unless a diagnostic oracle-scale row is explicitly marked",
        "- Sim(3)-aligned ATE evaluates shape after scale alignment, while path_ratio and none alignment remain scale-sensitive",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = _resolve_path(args.dataset, DEFAULT_DATASET)
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, metadata = _load_dataset_records(dataset_dir)
    if not records:
        raise RuntimeError("No panorama records found in dataset export.")
    if not all(Path(r["image_path"]).is_file() for r in records):
        raise RuntimeError("image source unavailable")

    view_yaws = [float(x.strip()) for x in str(args.view_yaws).split(",") if x.strip()]
    view_pitch = float(args.view_pitch)
    view_specs = [(yaw, view_pitch) for yaw in view_yaws]
    view_cache = PanoViewCache(view_specs=view_specs, view_w=int(args.view_width), view_h=int(args.view_height), fov_deg=float(args.fov_deg))
    scale_policy = _compute_train_median_scale(metadata)
    step_scale = float(scale_policy["value"])

    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    prev_tdir = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    pair_diags: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = [{"timestamp": records[0]["timestamp"], "R": np.eye(3), "t": np.zeros(3)}]
    oracle_rows: List[Dict[str, Any]] = [{"timestamp": records[0]["timestamp"], "R": np.eye(3), "t": np.zeros(3)}]
    world_R = np.eye(3, dtype=np.float64)
    world_t = np.zeros(3, dtype=np.float64)
    world_R_oracle = np.eye(3, dtype=np.float64)
    world_t_oracle = np.zeros(3, dtype=np.float64)

    seq_scene = str(metadata["sequence"]["scene"])
    seq_name = str(metadata["sequence"]["seq"])
    data_root = _resolve_path(metadata["data_root"], REPO_ROOT / metadata["data_root"])
    seq_frames = _scan_seq_frames(data_root, scene=seq_scene, seq=seq_name).get((seq_scene, seq_name), [])
    frame_by_ts = {fr.ts_str: fr for fr in seq_frames}

    all_inliers: List[int] = []
    success_count = 0
    fail_count = 0

    K_virtual = None
    for idx in range(len(records) - 1):
        rec_a = records[idx]
        rec_b = records[idx + 1]
        views_a, K_virtual = view_cache.get_views(Path(rec_a["image_path"]))
        views_b, _K_virtual = view_cache.get_views(Path(rec_b["image_path"]))
        pair = _run_orb_pose_on_pair(
            orb=orb,
            matcher=matcher,
            views_a=views_a,
            views_b=views_b,
            view_specs=view_specs,
            K=K_virtual,
            prev_tdir=prev_tdir,
        )
        pair["timestamp_A"] = float(rec_a["timestamp"])
        pair["timestamp_B"] = float(rec_b["timestamp"])
        pair["tsA"] = str(rec_a["ts_str"])
        pair["tsB"] = str(rec_b["ts_str"])
        pair["selected_view_yaw"] = float(pair["selected_view_yaw"])
        pair["match_count"] = int(pair["match_count"])
        pair["inlier_count"] = int(pair["inlier_count"])
        pair["R"] = np.asarray(pair["R"], dtype=np.float64)
        pair["tdir"] = np.asarray(pair["tdir"], dtype=np.float64)
        all_inliers.append(int(pair["inlier_count"]))
        if pair["success"]:
            prev_tdir = np.asarray(pair["tdir"], dtype=np.float64)
            success_count += 1
        else:
            fail_count += 1

        step_t = pair["tdir"] * step_scale
        world_R = pair["R"] @ world_R
        world_t = pair["R"] @ world_t + step_t
        main_rows.append({"timestamp": rec_b["timestamp"], "R": world_R.copy(), "t": world_t.copy()})

        oracle_step_mag = float("nan")
        fr_a = frame_by_ts.get(str(rec_a["ts_str"]))
        fr_b = frame_by_ts.get(str(rec_b["ts_str"]))
        if fr_a is not None and fr_b is not None:
            _R_a, t_a = _parse_label_13(fr_a.label_path)
            _R_b, t_b = _parse_label_13(fr_b.label_path)
            oracle_step_mag = float(np.linalg.norm(t_b - t_a))
        oracle_step_t = pair["tdir"] * (oracle_step_mag if math.isfinite(oracle_step_mag) else step_scale)
        world_R_oracle = pair["R"] @ world_R_oracle
        world_t_oracle = pair["R"] @ world_t_oracle + oracle_step_t
        oracle_rows.append({"timestamp": rec_b["timestamp"], "R": world_R_oracle.copy(), "t": world_t_oracle.copy()})

        pair_diags.append(
            {
                "pair_index": int(idx),
                "timestamp_A": float(rec_a["timestamp"]),
                "timestamp_B": float(rec_b["timestamp"]),
                "tsA": str(rec_a["ts_str"]),
                "tsB": str(rec_b["ts_str"]),
                "success": bool(pair["success"]),
                "fallback_used": bool(pair.get("fallback_used", False)),
                "reason": str(pair["reason"]),
                "match_count": int(pair["match_count"]),
                "inlier_count": int(pair["inlier_count"]),
                "selected_view_yaw": float(pair["selected_view_yaw"]),
                "selected_view_pitch": float(pair["selected_view_pitch"]),
                "tdir": [float(x) for x in pair["tdir"].tolist()],
                "oracle_step_mag": oracle_step_mag,
            }
        )

    traj_path = output_dir / "scene01_seq03_est_tum.txt"
    oracle_traj_path = output_dir / "scene01_seq03_est_oracle_scale_tum.txt"
    pair_diag_path = output_dir / "pair_diagnostics.json"
    traj_path.write_text(_traj_rows_to_tum(main_rows), encoding="utf-8")
    oracle_traj_path.write_text(_traj_rows_to_tum(oracle_rows), encoding="utf-8")
    pair_diag_path.write_text(json.dumps(pair_diags, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    gt_path = dataset_dir / "groundtruth_tum.txt"
    main_eval = {
        align: evaluate_external_baseline_trajectory(gt_path=gt_path, est_path=traj_path, alignment=align)
        for align in ("none", "se3", "sim3")
    }
    oracle_eval = {
        align: evaluate_external_baseline_trajectory(gt_path=gt_path, est_path=oracle_traj_path, alignment=align)
        for align in ("none", "se3", "sim3")
    }

    manifest = _read_json(FINAL_MANIFEST)
    inlier_arr = np.asarray(all_inliers, dtype=np.float64)
    mean_inliers = float(inlier_arr.mean()) if inlier_arr.size else float("nan")
    median_inliers = float(np.median(inlier_arr)) if inlier_arr.size else float("nan")
    success_rate = float(success_count / max(len(pair_diags), 1))

    result_rows: List[Dict[str, Any]] = []
    for method_name, eval_map, notes in [
        (
            "Pano-ORB-VO-train-scale",
            main_eval,
            "main fair baseline with train-only median step scale",
        ),
        (
            "Pano-ORB-VO-oracle-scale",
            oracle_eval,
            "diagnostic_only=true; uses test GT step magnitudes and is not deployable",
        ),
    ]:
        for align in ("none", "se3", "sim3"):
            res = eval_map[align]
            result_rows.append(
                {
                    "method": method_name,
                    "alignment": align,
                    "ATE": float(res.get("ATE", float("nan"))),
                    "drift": float(res.get("drift", float("nan"))),
                    "path_ratio": float(res.get("path_ratio", float("nan"))),
                    "tracking_success_rate": float(res.get("tracking_success_rate", success_rate)),
                    "mean_inliers": mean_inliers,
                    "median_inliers": median_inliers,
                    "notes": notes,
                    "diagnostic_only": method_name.endswith("oracle-scale"),
                }
            )

    s5 = dict(manifest["final_metrics"])
    comparison_with_s5 = [
        {
            "method": "Pano-ORB-VO-train-scale",
            "ATE": float(main_eval["se3"].get("ATE", float("nan"))),
            "drift": float(main_eval["se3"].get("drift", float("nan"))),
            "path_ratio": float(main_eval["none"].get("path_ratio", float("nan"))),
            "notes": "Protocol-compatible classical panorama VO baseline using derived virtual pinhole views.",
        }
    ]

    main_se3_ate = float(main_eval["se3"].get("ATE", float("nan")))
    main_none_ratio = float(main_eval["none"].get("path_ratio", float("nan")))
    if math.isfinite(main_se3_ate) and main_se3_ate > float(s5["ATE"]):
        interpretation = "S5 outperforms a protocol-compatible classical panorama VO baseline on this sequence."
    elif math.isfinite(main_se3_ate):
        interpretation = "Pano-ORB-VO is competitive in at least one trajectory metric, but scale/path stability and failure rate still need to be considered together."
    else:
        interpretation = "Pano-ORB-VO did not produce a stable enough trajectory for a strong numerical claim."

    fx = 0.5 * float(args.view_width) / math.tan(math.radians(float(args.fov_deg)) * 0.5)
    payload = {
        "name": "SB1_pano_orb_vo_baseline",
        "dataset_summary": {
            "dataset_dir": str(dataset_dir),
            "sequence": f"{metadata['sequence']['scene']}/{metadata['sequence']['seq']}",
            "num_frames": int(len(records)),
        },
        "virtual_camera": {
            "fov_deg": float(args.fov_deg),
            "view_yaws_deg": view_yaws,
            "view_pitch_deg": float(view_pitch),
            "view_width": int(args.view_width),
            "view_height": int(args.view_height),
            "fx": float(fx),
            "fy": float(fx),
            "cx": float(args.view_width) * 0.5,
            "cy": float(args.view_height) * 0.5,
            "derived_virtual_pinhole": True,
        },
        "scale_policy": scale_policy,
        "pair_diagnostics_path": str(pair_diag_path),
        "trajectory_path": str(traj_path),
        "oracle_trajectory_path": str(oracle_traj_path),
        "num_pairs": int(len(pair_diags)),
        "success_pairs": int(success_count),
        "failed_pairs": int(fail_count),
        "tracking_success_rate": success_rate,
        "mean_inliers": mean_inliers,
        "median_inliers": median_inliers,
        "main_evaluation": main_eval,
        "oracle_scale_evaluation": oracle_eval,
        "result_rows": result_rows,
        "s5_locked_metrics": s5,
        "comparison_with_s5": comparison_with_s5,
        "interpretation": interpretation,
        "diagnostic_only_oracle_scale": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(output_md, payload)
    print(f"[SB1] trajectory={traj_path}")
    print(f"[SB1] json={output_json}")
    print(f"[SB1] md={output_md}")
    print("[SB1] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
