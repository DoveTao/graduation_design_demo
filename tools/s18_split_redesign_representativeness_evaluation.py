#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList, _parse_label_13, _relative_pose_A_to_B_in_B, _scan_seq_frames
from eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S2B_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s2b_wrapper_result.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
S1D5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S1d5_clean_dt_anchor_policy.json"
S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"

REPORT_PATH = REPO_ROOT / "checkpoints" / "S18_split_redesign_representativeness_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S18_split_redesign_representativeness_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s18_split_redesign_representativeness_summary.md"
FIGURE_DIR = REPO_ROOT / "checkpoints" / "S18_split_redesign_representativeness_figures"

FINAL_CLASSES = {
    "SPLIT-REDESIGN-RECOMMENDED",
    "CURRENT-SPLIT-ACCEPTABLE-WITH-CAVEAT",
    "MORE-DATA-REQUIRED",
    "REPRODUCTION-MISMATCH",
    "INCONCLUSIVE",
}

SEQUENCE_LABELS = ["scene01/seq01", "scene01/seq02", "scene01/seq03"]
TRAIN_CV_SEQS = ["scene01/seq01", "scene01/seq02"]
FINAL_TEST_SEQ = "scene01/seq03"
HIGH_RISK_DT = 1.0
HIGH_RISK_K = 20
NUMERIC_HIST_BINS = 16
IMAGE_STAT_MAX_SAMPLES = 16
PRED_AUDIT_MAX_SAMPLES_PER_SEQ = 240


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = float(np.trace(R))
    c = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(np.degrees(np.arccos(c)))


def _tdir_dispersion_deg(vectors: np.ndarray) -> Dict[str, float]:
    vecs = np.asarray(vectors, dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        return {"mean_resultant_norm": float("nan"), "angle_p50": float("nan"), "angle_p90": float("nan")}
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    valid = norms[:, 0] > 1.0e-8
    if not np.any(valid):
        return {"mean_resultant_norm": float("nan"), "angle_p50": float("nan"), "angle_p90": float("nan")}
    unit = vecs[valid] / norms[valid]
    mean_vec = unit.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    if mean_norm <= 1.0e-8:
        ang = np.full((unit.shape[0],), 90.0, dtype=np.float64)
    else:
        mean_unit = mean_vec / mean_norm
        dots = np.clip(unit @ mean_unit, -1.0, 1.0)
        ang = np.degrees(np.arccos(dots))
    return {
        "mean_resultant_norm": mean_norm,
        "angle_p50": float(np.percentile(ang, 50)),
        "angle_p90": float(np.percentile(ang, 90)),
    }


def _hist_counts(values: Sequence[float], bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros((len(bins) - 1,), dtype=np.float64)
    counts, _ = np.histogram(arr, bins=bins)
    return counts.astype(np.float64)


def _normalized_hist_l1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.sum() <= 0 and b.sum() <= 0:
        return 0.0
    pa = a / max(a.sum(), 1.0)
    pb = b / max(b.sum(), 1.0)
    return float(np.abs(pa - pb).sum())


def _discrete_l1(values_a: Sequence[int], values_b: Sequence[int]) -> float:
    keys = sorted(set(int(x) for x in values_a) | set(int(x) for x in values_b))
    if not keys:
        return 0.0
    a = np.asarray([sum(1 for x in values_a if int(x) == k) for k in keys], dtype=np.float64)
    b = np.asarray([sum(1 for x in values_b if int(x) == k) for k in keys], dtype=np.float64)
    return _normalized_hist_l1(a, b)


def _all_pairs_distances(items: Sequence[str], fn) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for a in items:
        out[a] = {}
        for b in items:
            out[a][b] = float(fn(a, b)) if a != b else 0.0
    return out


def _sequence_sort_key(name: str) -> Tuple[str, str]:
    scene, seq = name.split("/")
    return scene, seq


@dataclass
class EvalBundle:
    cfg: Config
    ckpt_path: Path
    device: torch.device
    load_summary: Dict[str, Any]
    base_model: PanoramaRelPoseModel


class DtBucketScaledMagnitudeModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, bucket_factors: Dict[str, float]) -> None:
        super().__init__()
        self.base = base
        self.bucket_factors = dict(bucket_factors)
        self.cfg = base.cfg

    @staticmethod
    def _bucket(dt: float) -> Optional[str]:
        if 0.1 <= dt < 0.3:
            return "[0.1,0.3)"
        if 0.3 <= dt < 0.5:
            return "[0.3,0.5)"
        if 0.5 <= dt < 1.0:
            return "[0.5,1)"
        return None

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        if dt_world is None:
            return R_pred, t_pred, aux
        dt = float(dt_world.detach().float().view(-1)[0].cpu())
        factor = float(self.bucket_factors.get(self._bucket(dt), 1.0))
        if abs(factor - 1.0) < 1.0e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(factor, device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["dt_bucket_scale_anchor_factor"] = fac
        return R_pred, t_pred, aux


class PredTmagShrinkModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, q90: float, q95: float, mid_scale: float, high_scale: float) -> None:
        super().__init__()
        self.base = base
        self.q90 = float(q90)
        self.q95 = float(q95)
        self.mid_scale = float(mid_scale)
        self.high_scale = float(high_scale)
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        scale = 1.0
        if pred_mag >= self.q95:
            scale = self.high_scale
        elif pred_mag >= self.q90:
            scale = self.mid_scale
        if abs(scale - 1.0) < 1.0e-12:
            return R_pred, t_pred, aux
        aux = dict(aux)
        fac = torch.tensor(scale, device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["s5_tmag_scale_factor"] = fac
        return R_pred, t_pred, aux


def _load_policy(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cfg_from_policy(policy: Dict[str, Any]) -> Tuple[Config, Path]:
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    cfg.tmag_condition_on_dt = False
    return cfg, ckpt_path


def _load_current_arch_model(cfg: Config, ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    categories = {
        "ridge_calib_buffers": [k for k in missing if k.endswith("ridge_calib_raw_center")],
        "coupled_pose_head_params": [k for k in missing if k.startswith("coupled_pose_head.")],
    }
    categories["other_missing"] = [k for k in missing if k not in set(categories["ridge_calib_buffers"]) | set(categories["coupled_pose_head_params"])]
    return model, {"missing": missing, "unexpected": unexpected, "categories": categories}


def _build_dataset(cfg: Config, seqs: Optional[List[str]] = None) -> RflyPanoPanoramaPairsEvalFixedKList:
    seq_names = None
    if seqs is not None:
        seq_names = [s.split("/")[1] for s in seqs]
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=None if seqs is not None else "test",
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
        seqs=seq_names,
        scenes=["scene01"] if seqs is not None else None,
    )


def _build_loader(cfg: Config, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        drop_last=False,
    )


def _build_eval_bundle() -> EvalBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_policy = _load_policy(S2B_POLICY_PATH)
    cfg, ckpt_path = _build_cfg_from_policy(s2b_policy)
    base_model, load_summary = _load_current_arch_model(cfg, ckpt_path, device)
    return EvalBundle(cfg=cfg, ckpt_path=ckpt_path, device=device, load_summary=load_summary, base_model=base_model)


def _wrap_model(bundle: EvalBundle, policy_name: str) -> torch.nn.Module:
    s1d5 = _load_policy(S1D5_POLICY_PATH)
    s2b = _load_policy(S2B_POLICY_PATH)
    s5 = _load_policy(S5_POLICY_PATH)
    wrapped: torch.nn.Module
    if policy_name == "S1d5":
        wrapped = DtBucketScaledMagnitudeModel(bundle.base_model, {str(k): float(v) for k, v in s1d5["effective_bucket_factors"].items()})
    elif policy_name == "S2b":
        wrapped = DtBucketScaledMagnitudeModel(bundle.base_model, {str(k): float(v) for k, v in s2b["effective_bucket_factors"].items()})
    elif policy_name == "S5":
        s2b_wrapped = DtBucketScaledMagnitudeModel(bundle.base_model, {str(k): float(v) for k, v in s2b["effective_bucket_factors"].items()})
        wrapped = PredTmagShrinkModel(
            s2b_wrapped,
            q90=float(s5["thresholds"]["q90_value"]),
            q95=float(s5["thresholds"]["q95_upper_tail_value"]),
            mid_scale=float(s5["scales"]["mid_scale"]),
            high_scale=float(s5["scales"]["high_scale"]),
        )
    else:
        raise ValueError(policy_name)
    return wrapped.to(bundle.device)


def _read_debug_summary(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "odom_trajectory_debug_latest.json"
    if not path.exists():
        return {}
    obj = _read_json(path)
    summary = dict(obj.get("summary", {}))
    chains = obj.get("chains", [])
    if chains:
        vals = []
        for chain in chains:
            shape = chain.get("shape_direction_only", {})
            v = _safe_float(shape.get("path_length_ratio"))
            if math.isfinite(v):
                vals.append(v)
        if vals:
            summary["direction_only_mean_path_length_ratio"] = float(np.mean(vals))
    return summary


def _run_policy_eval(bundle: EvalBundle, policy_name: str, seq: str) -> Dict[str, Any]:
    model = _wrap_model(bundle, policy_name)
    ds = _build_dataset(bundle.cfg, seqs=[seq])
    out_dir = Path(tempfile.mkdtemp(prefix=f"s18_{policy_name}_{seq.replace('/', '_')}_", dir=str(REPO_ROOT / "checkpoints")))
    try:
        odom = eval_odometry_sequence(model, ds, bundle.device, bundle.cfg, output_dir=str(out_dir), step=0, upd=0)
        dbg = _read_debug_summary(out_dir)
        return {
            "seq": seq,
            "policy": policy_name,
            "drift": _safe_float(odom.get("odom_metric_drift")),
            "ATE": _safe_float(odom.get("odom_metric_ATE")),
            "path_ratio": _safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
            "RPE_rot": _safe_float(odom.get("odom_metric_RPE_rot")),
            "RPE_trans_dir": _safe_float(odom.get("odom_metric_RPE_trans_dir")),
            "RPE_trans_mag": _safe_float(odom.get("odom_metric_RPE_trans_mag")),
            "rot": float("nan"),
            "tdir_abs": float("nan"),
            "tdir_local_A_abs": float("nan"),
            "num_pairs": int(odom.get("odom_num_pairs", 0)),
            "num_chains": int(odom.get("odom_num_chains", 0)),
            "selected_k": int(odom.get("odom_selected_k", -1)),
            "available_k": [int(x) for x in odom.get("odom_available_k", [])],
            "direction_only_path_ratio": _safe_float(dbg.get("direction_only_mean_path_length_ratio")),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _sample_manifest_indices(manifest: Sequence[Dict[str, Any]], max_samples: int) -> List[int]:
    by_seq: Dict[str, List[int]] = {}
    for idx, meta in enumerate(manifest):
        by_seq.setdefault(f"{meta['scene']}/{meta['seq']}", []).append(idx)
    out: List[int] = []
    for seq in sorted(by_seq.keys(), key=_sequence_sort_key):
        idxs = by_seq[seq]
        if len(idxs) <= max_samples:
            out.extend(idxs)
            continue
        picked = np.linspace(0, len(idxs) - 1, max_samples, dtype=int)
        out.extend([idxs[int(i)] for i in picked])
    return sorted(set(out))


def _extract_sequence_rows(bundle: EvalBundle, seqs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ds = _build_dataset(bundle.cfg, seqs=list(seqs))
    manifest = ds.manifest()
    rows: List[Dict[str, Any]] = []
    for meta in manifest:
        seq_idx = int(meta["seq_idx"])
        sd = ds.seqs_data[seq_idx]
        i = int(meta["i"])
        j = int(meta["j"])
        R_gt, t_gt_vec = _relative_pose_A_to_B_in_B(sd["R_w"][i], sd["t_w"][i], sd["R_w"][j], sd["t_w"][j])
        gt_tmag = float(np.linalg.norm(t_gt_vec))
        scene_seq = f"{meta['scene']}/{meta['seq']}"
        rows.append(
            {
                "scene_seq": scene_seq,
                "i": i,
                "j": j,
                "k": int(meta["k"]),
                "dt_world": float(meta["dt_world"]),
                "gt_tmag": gt_tmag,
                "dt_k": float(meta["dt_world"]) * int(meta["k"]),
                "rot_deg": _rotation_angle_deg(R_gt),
                "tdir_vec": [float(x) for x in t_gt_vec],
            }
        )

    model = _wrap_model(bundle, "S2b")
    pred_rows: List[Dict[str, Any]] = []
    sampled_indices = _sample_manifest_indices(manifest, PRED_AUDIT_MAX_SAMPLES_PER_SEQ)
    manifest = ds.manifest()
    with torch.no_grad():
        for idx in sampled_indices:
            meta = manifest[idx]
            sample = ds[idx]
            IA = sample["IA"].unsqueeze(0).to(bundle.device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(bundle.device, non_blocking=True)
            dt = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_tensor = torch.tensor([dt], device=bundle.device, dtype=torch.float32)
            _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            scene_seq = f"{meta['scene']}/{meta['seq']}"
            pred_rows.append(
                {
                    "scene_seq": scene_seq,
                    "i": int(meta["i"]),
                    "j": int(meta["j"]),
                    "k": int(meta["k"]),
                    "dt_world": dt,
                    "gt_tmag": float(sample["t_gt_mag"]),
                    "pred_tmag": pred_tmag,
                    "dt_k": dt * int(meta["k"]),
                }
            )
    return rows, pred_rows


def _seq_frame_path_stats(data_root: str, seq: str) -> Dict[str, float]:
    scene, seq_name = seq.split("/")
    seq_frames = _scan_seq_frames(data_root, scenes=[scene], seqs=[seq_name])
    frames = seq_frames.get((scene, seq_name), [])
    if not frames:
        return {"sampled_frames": 0, "brightness_mean": float("nan"), "contrast_mean": float("nan"), "blur_proxy_mean": float("nan"), "texture_proxy_mean": float("nan"), "seq_abs_path_length": float("nan")}
    idxs = np.linspace(0, len(frames) - 1, min(IMAGE_STAT_MAX_SAMPLES, len(frames)), dtype=int)
    brightness: List[float] = []
    contrast: List[float] = []
    blur: List[float] = []
    texture: List[float] = []
    for idx in idxs:
        img = Image.open(frames[int(idx)].pano_path).convert("L").resize((256, 128), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        brightness.append(float(arr.mean()))
        contrast.append(float(arr.std()))
        dx = np.abs(np.diff(arr, axis=1))
        dy = np.abs(np.diff(arr, axis=0))
        blur.append(float((dx.mean() + dy.mean()) * 0.5))
        lap = (
            -4.0 * arr[1:-1, 1:-1]
            + arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
        )
        texture.append(float(np.var(lap)))
    traj = []
    for fr in frames:
        _R, t = _parse_label_13(fr.label_path)
        traj.append(np.asarray(t, dtype=np.float64))
    seq_abs_path_length = float(np.sum(np.linalg.norm(np.diff(np.asarray(traj), axis=0), axis=1))) if len(traj) >= 2 else float("nan")
    return {
        "sampled_frames": int(len(idxs)),
        "brightness_mean": float(np.mean(brightness)),
        "contrast_mean": float(np.mean(contrast)),
        "blur_proxy_mean": float(np.mean(blur)),
        "texture_proxy_mean": float(np.mean(texture)),
        "seq_abs_path_length": seq_abs_path_length,
    }


def _audit_sequences(bundle: EvalBundle) -> Dict[str, Any]:
    all_rows, pred_rows = _extract_sequence_rows(bundle, SEQUENCE_LABELS)
    by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_seq.setdefault(str(row["scene_seq"]), []).append(row)
    pred_by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for row in pred_rows:
        pred_by_seq.setdefault(str(row["scene_seq"]), []).append(row)

    s5_policy = _load_policy(S5_POLICY_PATH)
    q90 = float(s5_policy["thresholds"]["q90_value"])
    q95 = float(s5_policy["thresholds"]["q95_upper_tail_value"])

    seq_stats: Dict[str, Any] = {}
    for seq in sorted(by_seq.keys(), key=_sequence_sort_key):
        rows = by_seq[seq]
        ks = [int(r["k"]) for r in rows]
        dt = [float(r["dt_world"]) for r in rows]
        gt = [float(r["gt_tmag"]) for r in rows]
        dtk = [float(r["dt_k"]) for r in rows]
        rot = [float(r["rot_deg"]) for r in rows]
        tdir = np.asarray([r["tdir_vec"] for r in rows], dtype=np.float64)
        pred = [float(r["pred_tmag"]) for r in pred_by_seq.get(seq, [])]
        img_stats = _seq_frame_path_stats(str(bundle.cfg.data_root), seq)
        high_risk_mass = float(np.mean([float(r["dt_world"]) >= HIGH_RISK_DT and int(r["k"]) == HIGH_RISK_K for r in rows]))
        high_pred_mass = float(np.mean([float(r["pred_tmag"]) >= q90 for r in pred_by_seq.get(seq, [])])) if pred else float("nan")
        upper_tail_pred_mass = float(np.mean([float(r["pred_tmag"]) >= q95 for r in pred_by_seq.get(seq, [])])) if pred else float("nan")
        pair_tmag_sum = float(np.sum(gt))
        seq_path_len = _safe_float(img_stats.get("seq_abs_path_length"))
        seq_stats[seq] = {
            "pair_count": int(len(rows)),
            "k_distribution": {str(k): int(sum(1 for x in ks if x == k)) for k in sorted(set(ks))},
            "dt_world_summary": _percentiles(dt),
            "gt_tmag_summary": _percentiles(gt),
            "pred_tmag_summary_s2b_base_eval": _percentiles(pred),
            "dt_k_summary": _percentiles(dtk),
            "rot_deg_summary": _percentiles(rot),
            "tdir_dispersion_deg": _tdir_dispersion_deg(tdir),
            "pair_tmag_sum": pair_tmag_sum,
            "pair_tmag_sum_over_seq_path_length": float(pair_tmag_sum / seq_path_len) if math.isfinite(seq_path_len) and seq_path_len > 1.0e-8 else float("nan"),
            "high_risk_bucket_mass": high_risk_mass,
            "high_pred_bucket_mass_q90": high_pred_mass,
            "high_pred_bucket_mass_q95": upper_tail_pred_mass,
            "image_stats": img_stats,
            "pred_tmag_sample_count": int(len(pred)),
        }

    all_gt = np.asarray([r["gt_tmag"] for r in all_rows], dtype=np.float64)
    all_pred = np.asarray([r["pred_tmag"] for r in pred_rows], dtype=np.float64)
    all_dtk = np.asarray([r["dt_k"] for r in all_rows], dtype=np.float64)
    gt_bins = np.linspace(float(all_gt.min()), float(all_gt.max()) + 1.0e-12, NUMERIC_HIST_BINS + 1)
    pred_bins = np.linspace(float(all_pred.min()), float(all_pred.max()) + 1.0e-12, NUMERIC_HIST_BINS + 1)
    dtk_bins = np.linspace(float(all_dtk.min()), float(all_dtk.max()) + 1.0e-12, NUMERIC_HIST_BINS + 1)

    k_dist = _all_pairs_distances(SEQUENCE_LABELS, lambda a, b: _discrete_l1(
        [int(r["k"]) for r in by_seq[a]],
        [int(r["k"]) for r in by_seq[b]],
    ))
    gt_dist = _all_pairs_distances(SEQUENCE_LABELS, lambda a, b: _normalized_hist_l1(
        _hist_counts([float(r["gt_tmag"]) for r in by_seq[a]], gt_bins),
        _hist_counts([float(r["gt_tmag"]) for r in by_seq[b]], gt_bins),
    ))
    pred_dist = _all_pairs_distances(SEQUENCE_LABELS, lambda a, b: _normalized_hist_l1(
        _hist_counts([float(r["pred_tmag"]) for r in pred_by_seq[a]], pred_bins),
        _hist_counts([float(r["pred_tmag"]) for r in pred_by_seq[b]], pred_bins),
    ))
    dtk_dist = _all_pairs_distances(SEQUENCE_LABELS, lambda a, b: _normalized_hist_l1(
        _hist_counts([float(r["dt_k"]) for r in by_seq[a]], dtk_bins),
        _hist_counts([float(r["dt_k"]) for r in by_seq[b]], dtk_bins),
    ))

    combined = _all_pairs_distances(
        SEQUENCE_LABELS,
        lambda a, b: 0.2 * k_dist[a][b] + 0.3 * gt_dist[a][b] + 0.3 * pred_dist[a][b] + 0.2 * dtk_dist[a][b],
    )

    train_pool = [r for r in all_rows if r["scene_seq"] in TRAIN_CV_SEQS]
    seq03 = [r for r in all_rows if r["scene_seq"] == FINAL_TEST_SEQ]
    pred_train_pool = [r for r in pred_rows if r["scene_seq"] in TRAIN_CV_SEQS]
    pred_seq03 = [r for r in pred_rows if r["scene_seq"] == FINAL_TEST_SEQ]
    train_pool_vs_seq03 = {
        "k_l1": _discrete_l1([int(r["k"]) for r in train_pool], [int(r["k"]) for r in seq03]),
        "gt_tmag_l1": _normalized_hist_l1(_hist_counts([float(r["gt_tmag"]) for r in train_pool], gt_bins), _hist_counts([float(r["gt_tmag"]) for r in seq03], gt_bins)),
        "pred_tmag_l1": _normalized_hist_l1(_hist_counts([float(r["pred_tmag"]) for r in pred_train_pool], pred_bins), _hist_counts([float(r["pred_tmag"]) for r in pred_seq03], pred_bins)),
        "dt_k_l1": _normalized_hist_l1(_hist_counts([float(r["dt_k"]) for r in train_pool], dtk_bins), _hist_counts([float(r["dt_k"]) for r in seq03], dtk_bins)),
    }

    return {
        "base_eval_pred_source": "current-architecture S2b-wrapped deterministic eval",
        "threshold_reference": {"q90": q90, "q95": q95},
        "per_sequence": seq_stats,
        "distance_matrices": {
            "k_l1": k_dist,
            "gt_tmag_l1": gt_dist,
            "pred_tmag_l1": pred_dist,
            "dt_k_l1": dtk_dist,
            "combined_representativeness": combined,
        },
        "train_pool_vs_seq03": train_pool_vs_seq03,
        "raw_pair_count_total": int(len(all_rows)),
        "pred_tmag_sample_count_total": int(len(pred_rows)),
    }


def _fixed_policy_eval(bundle: EvalBundle) -> Dict[str, Any]:
    per_seq: Dict[str, Dict[str, Any]] = {}
    for seq in SEQUENCE_LABELS:
        per_seq[seq] = {}
        for policy in ("S1d5", "S2b", "S5"):
            per_seq[seq][policy] = _run_policy_eval(bundle, policy, seq)
    deltas: Dict[str, Any] = {}
    for seq in SEQUENCE_LABELS:
        s2b = per_seq[seq]["S2b"]
        s5 = per_seq[seq]["S5"]
        deltas[seq] = {
            "delta_drift": _safe_float(s5["drift"] - s2b["drift"]),
            "delta_ATE": _safe_float(s5["ATE"] - s2b["ATE"]),
            "delta_path_ratio": _safe_float(s5["path_ratio"] - s2b["path_ratio"]),
        }
    ranking = {}
    for seq in SEQUENCE_LABELS:
        rows = [(policy, per_seq[seq][policy]["ATE"]) for policy in ("S1d5", "S2b", "S5")]
        ranking[seq] = [name for name, _ate in sorted(rows, key=lambda item: item[1])]
    return {"per_sequence": per_seq, "s5_vs_s2b_delta": deltas, "ate_ranking_per_sequence": ranking}


def _decide_classification(audit: Dict[str, Any], fixed_eval: Dict[str, Any]) -> Tuple[str, List[str]]:
    dist = audit["distance_matrices"]["combined_representativeness"]
    seq03_far = float(dist["scene01/seq01"]["scene01/seq03"]) > float(dist["scene01/seq01"]["scene01/seq02"]) and float(dist["scene01/seq02"]["scene01/seq03"]) > float(dist["scene01/seq01"]["scene01/seq02"])
    deltas = fixed_eval["s5_vs_s2b_delta"]
    delta_ates = [float(deltas[s]["delta_ATE"]) for s in SEQUENCE_LABELS]
    unstable = (max(delta_ates) - min(delta_ates)) > 0.5 or any(fixed_eval["ate_ranking_per_sequence"][s][0] != "S5" for s in SEQUENCE_LABELS)
    seq_count_small = len(SEQUENCE_LABELS) < 4
    reasons: List[str] = []
    if seq03_far:
        reasons.append("seq03 is farther from seq01/seq02 than the within-train seq01-seq02 gap under the combined representativeness score")
    if unstable:
        reasons.append("fixed-policy performance is not stable across heldout sequences")
    if seq03_far and unstable:
        return "SPLIT-REDESIGN-RECOMMENDED", reasons
    if not unstable:
        if not seq03_far:
            reasons.append("seq03 is not the most isolated sequence in the pairwise representativeness matrices")
        reasons.append("S5 remains the best clean policy on all three sequences and the gain is not seq03-only")
        if seq_count_small:
            reasons.append("only three sequences are available, so acceptance still needs an explicit representativeness caveat")
        return "CURRENT-SPLIT-ACCEPTABLE-WITH-CAVEAT", reasons
    if seq_count_small:
        reasons.append("only three sequences are available, so any redesigned split still has low statistical confidence")
        return "MORE-DATA-REQUIRED", reasons
    reasons.append("available evidence does not cleanly separate redesign need from sample scarcity")
    return "INCONCLUSIVE", reasons


def _protocol_proposal(classification: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": "A. current protocol",
            "definition": "train/CV on seq01+seq02, final test on seq03",
            "pros": ["historically locked and directly comparable to S5"],
            "cons": ["single-sequence final test", "representativeness caveat from S17/S18"],
            "leakage_risk": "low",
            "thesis_main_result_fit": "historical-baseline-only",
            "future_work_fit": "limited",
        },
        {
            "name": "B. leave-one-sequence-out over seq01/seq02/seq03",
            "definition": "held out one full sequence at a time; evaluation diagnostic only",
            "pros": ["sequence-level robustness visibility", "no pair-level leakage"],
            "cons": ["still only three folds", "cannot replace historical S5 lock"],
            "leakage_risk": "low",
            "thesis_main_result_fit": "recommended caveat protocol" if classification == "SPLIT-REDESIGN-RECOMMENDED" else "supplementary",
            "future_work_fit": "strong",
        },
        {
            "name": "C. stratified pair-level CV by regime",
            "definition": "stratify by k / gt_tmag / pred_tmag regimes",
            "pros": ["better regime balancing in principle"],
            "cons": ["sequence leakage risk", "temporal correlation can invalidate optimistic CV"],
            "leakage_risk": "high",
            "thesis_main_result_fit": "not recommended",
            "future_work_fit": "research-only with explicit leakage controls",
        },
        {
            "name": "D. collect-new-seq protocol",
            "definition": "keep S5 as historical best clean candidate; expand data, then redesign train/CV/test",
            "pros": ["best route to representative split", "supports stronger thesis claims"],
            "cons": ["requires new data collection effort"],
            "leakage_risk": "low",
            "thesis_main_result_fit": "recommended future-work direction",
            "future_work_fit": "strongest",
        },
    ]


def _render_matrix_table(matrix: Dict[str, Dict[str, float]]) -> str:
    cols = SEQUENCE_LABELS
    lines = ["| seq | " + " | ".join(cols) + " |\n", "| --- | " + " | ".join(["---"] * len(cols)) + " |\n"]
    for row in cols:
        lines.append("| " + row + " | " + " | ".join(_fmt(matrix[row][col], 4) for col in cols) + " |\n")
    return "".join(lines)


def _plot_heatmap(matrix: Dict[str, Dict[str, float]], title: str, path: Path) -> None:
    labels = SEQUENCE_LABELS
    arr = np.asarray([[float(matrix[a][b]) for b in labels] for a in labels], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="magma")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_seq_metric(fixed_eval: Dict[str, Any], metric: str, path: Path) -> None:
    labels = SEQUENCE_LABELS
    policies = ["S1d5", "S2b", "S5"]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, policy in enumerate(policies):
        vals = [float(fixed_eval["per_sequence"][seq][policy][metric]) for seq in labels]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=policy)
    ax.set_xticks(x, labels=labels, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-sequence {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pred_tmag_quantiles(audit: Dict[str, Any], path: Path) -> None:
    labels = SEQUENCE_LABELS
    p50 = [float(audit["per_sequence"][seq]["pred_tmag_summary_s2b_base_eval"]["p50"]) for seq in labels]
    p90 = [float(audit["per_sequence"][seq]["pred_tmag_summary_s2b_base_eval"]["p90"]) for seq in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.plot(x, p50, marker="o", label="p50")
    ax.plot(x, p90, marker="o", label="p90")
    ax.axhline(float(audit["threshold_reference"]["q90"]), color="tab:red", linestyle="--", linewidth=1.2, label="S5 q90")
    ax.axhline(float(audit["threshold_reference"]["q95"]), color="tab:orange", linestyle=":", linewidth=1.2, label="S5 q95")
    ax.set_xticks(x, labels=labels, rotation=15, ha="right")
    ax.set_ylabel("pred_tmag")
    ax.set_title("Per-sequence pred_tmag quantiles under S2b base eval")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(payload: Dict[str, Any]) -> None:
    gate = payload["baseline_gate"]
    lines: List[str] = []
    lines.append("# S18 Split Redesign And Representativeness Evaluation\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{payload['s5_remains_final_clean_candidate']}`\n")
    lines.append(f"- baseline gate pass: `{gate['pass']}`\n")
    if payload.get("decision_reasons"):
        lines.append(f"- classification rationale: `{' ; '.join(payload['decision_reasons'])}`\n")
    lines.append("\n")

    lines.append("## Motivation from S17\n\n")
    lines.append("- S17 concluded `CV-SPLIT-NOT-REPRESENTATIVE` while keeping dataset/supervision/convention audits clean.\n")
    lines.append("- S18 therefore re-audits split representativeness without retraining, without changing S5, and without replacing locked historical metrics.\n\n")

    lines.append("## Baseline gate\n\n")
    lines.append(f"- contract path: `{S8B_CONTRACT_PATH}`\n")
    lines.append(f"- current-architecture load_missing/unexpected accepted: `{gate['accepted_loading']}` with observed `{gate['observed_load_missing']} / {gate['observed_load_unexpected']}`\n")
    lines.append(f"- S5 locked metrics preserved: `{gate['s5_locked_metrics_preserved']}`\n")
    lines.append(f"- final manifest overwritten: `{gate['final_manifest_overwritten']}`\n\n")

    if payload["final_classification"] == "REPRODUCTION-MISMATCH":
        REPORT_PATH.write_text("".join(lines), encoding="utf-8")
        return

    audit = payload["sequence_distribution_audit"]
    lines.append("## Sequence-level distribution audit\n\n")
    for seq in SEQUENCE_LABELS:
        row = audit["per_sequence"][seq]
        lines.append(f"### {seq}\n\n")
        lines.append(f"- pair count: `{row['pair_count']}`\n")
        lines.append(f"- k distribution: `{json.dumps(row['k_distribution'], ensure_ascii=True)}`\n")
        lines.append(f"- dt p50/p90: `{_fmt(row['dt_world_summary']['p50'])}` / `{_fmt(row['dt_world_summary']['p90'])}`\n")
        lines.append(f"- gt_tmag p50/p90: `{_fmt(row['gt_tmag_summary']['p50'])}` / `{_fmt(row['gt_tmag_summary']['p90'])}`\n")
        lines.append(f"- pred_tmag p50/p90: `{_fmt(row['pred_tmag_summary_s2b_base_eval']['p50'])}` / `{_fmt(row['pred_tmag_summary_s2b_base_eval']['p90'])}`\n")
        lines.append(f"- rot target p50/p90: `{_fmt(row['rot_deg_summary']['p50'])}` / `{_fmt(row['rot_deg_summary']['p90'])}`\n")
        lines.append(f"- tdir dispersion p50/p90: `{_fmt(row['tdir_dispersion_deg']['angle_p50'])}` / `{_fmt(row['tdir_dispersion_deg']['angle_p90'])}`\n")
        lines.append(f"- pair_tmag_sum / seq_path_length: `{_fmt(row['pair_tmag_sum_over_seq_path_length'])}`\n")
        lines.append(f"- high-risk bucket mass: `{_fmt(row['high_risk_bucket_mass'], 4)}`\n")
        lines.append(f"- high-pred bucket mass q90/q95: `{_fmt(row['high_pred_bucket_mass_q90'], 4)}` / `{_fmt(row['high_pred_bucket_mass_q95'], 4)}`\n")
        img = row["image_stats"]
        lines.append(f"- image stats brightness/contrast/blur/texture: `{_fmt(img['brightness_mean'], 4)}` / `{_fmt(img['contrast_mean'], 4)}` / `{_fmt(img['blur_proxy_mean'], 4)}` / `{_fmt(img['texture_proxy_mean'], 4)}`\n\n")

    lines.append("## Sequence distance matrix\n\n")
    for name, matrix in audit["distance_matrices"].items():
        lines.append(f"### {name}\n\n")
        lines.append(_render_matrix_table(matrix))
        lines.append("\n")
    lines.append(f"- pooled train(seq01+seq02) vs seq03: `{json.dumps(audit['train_pool_vs_seq03'], ensure_ascii=True)}`\n\n")

    fixed_eval = payload["fixed_policy_per_sequence_evaluation"]
    lines.append("## Fixed-policy per-sequence evaluation\n\n")
    lines.append("| seq | policy | drift | ATE | path_ratio | rot | tdir_abs |\n")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for seq in SEQUENCE_LABELS:
        for policy in ("S1d5", "S2b", "S5"):
            row = fixed_eval["per_sequence"][seq][policy]
            lines.append(
                f"| {seq} | {policy} | {_fmt(row['drift'])} | {_fmt(row['ATE'])} | {_fmt(row['path_ratio'])} | {_fmt(row['rot'])} | {_fmt(row['tdir_abs'])} |\n"
            )
    lines.append("\n")
    lines.append("## S5/S2b/S1d5 stability across sequences\n\n")
    for seq in SEQUENCE_LABELS:
        delta = fixed_eval["s5_vs_s2b_delta"][seq]
        lines.append(
            f"- {seq}: S5-S2b delta drift=`{_fmt(delta['delta_drift'])}`, ATE=`{_fmt(delta['delta_ATE'])}`, path_ratio=`{_fmt(delta['delta_path_ratio'])}`, ranking=`{fixed_eval['ate_ranking_per_sequence'][seq]}`\n"
        )
    lines.append("\n")

    lines.append("## Alternative CV protocol proposal\n\n")
    for row in payload["alternative_protocols"]:
        lines.append(f"### {row['name']}\n\n")
        lines.append(f"- definition: `{row['definition']}`\n")
        lines.append(f"- pros: `{'; '.join(row['pros'])}`\n")
        lines.append(f"- cons: `{'; '.join(row['cons'])}`\n")
        lines.append(f"- leakage risk: `{row['leakage_risk']}`\n")
        lines.append(f"- thesis main result fit: `{row['thesis_main_result_fit']}`\n")
        lines.append(f"- future work fit: `{row['future_work_fit']}`\n\n")

    lines.append("## Impact on thesis claims\n\n")
    lines.append("- `S5_clean_tmag_calibration_policy` remains the best clean candidate under the historical protocol.\n")
    lines.append("- Current historical CV/test should be presented with an explicit representativeness caveat rather than as practical-ready evidence.\n")
    lines.append("- Future improvements should prefer leave-one-sequence-out diagnostics and, ideally, additional sequences before promoting stronger claims.\n\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    audit = payload.get("sequence_distribution_audit", {})
    fixed_eval = payload.get("fixed_policy_per_sequence_evaluation", {})
    lines: List[str] = []
    lines.append("# Final S18 Split Redesign Representativeness Summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append("- S5 remains final clean candidate under the historical protocol.\n")
    if audit:
        combined = audit["distance_matrices"]["combined_representativeness"]
        lines.append(
            f"- combined distance seq01-seq02 / seq01-seq03 / seq02-seq03: `{_fmt(combined['scene01/seq01']['scene01/seq02'], 4)}` / `{_fmt(combined['scene01/seq01']['scene01/seq03'], 4)}` / `{_fmt(combined['scene01/seq02']['scene01/seq03'], 4)}`\n"
        )
    if fixed_eval:
        for seq in SEQUENCE_LABELS:
            s5 = fixed_eval["per_sequence"][seq]["S5"]
            s2b = fixed_eval["per_sequence"][seq]["S2b"]
            lines.append(
                f"- {seq}: S5 ATE/drift/path_ratio=`{_fmt(s5['ATE'])}` / `{_fmt(s5['drift'])}` / `{_fmt(s5['path_ratio'])}`, delta_vs_S2b_ATE=`{_fmt(s5['ATE'] - s2b['ATE'])}`\n"
            )
    lines.append("- recommended protocol: leave-one-sequence-out diagnostics for current thesis caveat; collect more sequences for future final split redesign.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def _make_figures(payload: Dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    audit = payload["sequence_distribution_audit"]
    fixed_eval = payload["fixed_policy_per_sequence_evaluation"]
    _plot_heatmap(audit["distance_matrices"]["combined_representativeness"], "Combined representativeness distance", FIGURE_DIR / "combined_distance_heatmap.png")
    _plot_pred_tmag_quantiles(audit, FIGURE_DIR / "pred_tmag_quantiles_by_sequence.png")
    _plot_per_seq_metric(fixed_eval, "ATE", FIGURE_DIR / "per_sequence_ate.png")
    _plot_per_seq_metric(fixed_eval, "drift", FIGURE_DIR / "per_sequence_drift.png")


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s2b = _read_json(S8B_S2B_RESULT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    expected = contract["current_architecture_expected_benign_loading"]
    locked = contract["s5_locked_metrics"]
    accepted_loading = (
        int(s5["load_missing"]) == int(expected["load_missing"])
        and int(s5["load_unexpected"]) == int(expected["load_unexpected"])
        and len(s5["missing_key_categories"]["ridge_calib_buffers"]) == int(expected["missing_key_categories"]["ridge_calib_buffers"])
        and len(s5["missing_key_categories"]["coupled_pose_head_params"]) == int(expected["missing_key_categories"]["coupled_pose_head_params"])
        and len(s5["missing_key_categories"]["other_missing"]) == 0
        and int(s2b["load_missing"]) == int(expected["load_missing"])
        and int(s2b["load_unexpected"]) == int(expected["load_unexpected"])
    )
    s5_locked_metrics_preserved = (
        abs(float(s5["metrics"]["drift"]) - float(locked["drift"])) < 1.0e-9
        and abs(float(s5["metrics"]["ATE"]) - float(locked["ATE"])) < 1.0e-9
        and abs(float(s5["metrics"]["path_ratio"]) - float(locked["path_ratio"])) < 1.0e-9
    )
    return {
        "pass": bool(accepted_loading and s5_locked_metrics_preserved and contract["tolerance_policy"]["locked_metrics_must_not_be_overwritten"]),
        "accepted_loading": bool(accepted_loading),
        "observed_load_missing": int(s5["load_missing"]),
        "observed_load_unexpected": int(s5["load_unexpected"]),
        "s5_locked_metrics_preserved": bool(s5_locked_metrics_preserved),
        "final_manifest_overwritten": False,
        "contract": contract,
    }


def main() -> None:
    gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "task": "S18_split_redesign_and_representativeness_evaluation",
        "baseline_gate": gate,
        "s5_remains_final_clean_candidate": True,
    }
    if not gate["pass"]:
        payload["final_classification"] = "REPRODUCTION-MISMATCH"
        payload["decision_reasons"] = ["S8b baseline gate did not pass under the accepted current-architecture benign loading contract"]
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return

    bundle = _build_eval_bundle()
    payload["current_architecture_loading_audit"] = {
        "load_missing": int(len(bundle.load_summary["missing"])),
        "load_unexpected": int(len(bundle.load_summary["unexpected"])),
        "missing_key_categories": {
            "ridge_calib_buffers": list(bundle.load_summary["categories"]["ridge_calib_buffers"]),
            "coupled_pose_head_params": list(bundle.load_summary["categories"]["coupled_pose_head_params"]),
            "other_missing": list(bundle.load_summary["categories"]["other_missing"]),
        },
    }
    payload["sequence_distribution_audit"] = _audit_sequences(bundle)
    payload["fixed_policy_per_sequence_evaluation"] = _fixed_policy_eval(bundle)
    classification, reasons = _decide_classification(payload["sequence_distribution_audit"], payload["fixed_policy_per_sequence_evaluation"])
    if classification not in FINAL_CLASSES:
        raise RuntimeError(f"Unexpected final classification: {classification}")
    payload["final_classification"] = classification
    payload["decision_reasons"] = reasons
    payload["alternative_protocols"] = _protocol_proposal(classification)
    payload["evaluation_protocol_caveat"] = {
        "current_final_test_is_single_sequence": True,
        "sequence": FINAL_TEST_SEQ,
        "s17_primary_classification": "CV-SPLIT-NOT-REPRESENTATIVE",
    }
    _make_figures(payload)
    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    _write_summary(payload)


if __name__ == "__main__":
    main()
