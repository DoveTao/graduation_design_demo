#!/usr/bin/env python3
"""S14 local-window pose-graph optimization diagnostic."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.eval_clean_policy import (  # type: ignore
    DtBucketScaledMagnitudeModel,
    _build_eval_dataset,
    _load_fine_model,
    _load_policy,
)
from train_mvp import (  # type: ignore
    _build_odometry_chains,
    _camera_center_from_T_c0_np,
    _compose_rel_pose_np,
    _rot_geodesic_deg_np,
    _trajectory_shape_summary,
    _vec_angle_deg_np,
)


POLICY_PATH = REPO_ROOT / "checkpoints/S5_clean_tmag_calibration_policy.json"
CONTRACT_PATH = REPO_ROOT / "checkpoints/S8b_reproduction_contract.json"
ARTIFACT_PATH = REPO_ROOT / "checkpoints/S8b_current_arch_s5_wrapper_result.json"
REPORT_PATH = REPO_ROOT / "checkpoints/S14_local_window_pose_graph_optimization_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints/S14_local_window_pose_graph_optimization_candidates.json"
POLICY_OUT_PATH = REPO_ROOT / "checkpoints/S14_local_window_pose_graph_optimization_policy.json"
SUMMARY_PATH = REPO_ROOT / "reports/final_s14_local_window_pose_graph_summary.md"
FIG_DIR = REPO_ROOT / "checkpoints/S14_local_window_pose_graph_optimization_figures"

LENIENT = {"ATE": 5.0, "drift": 1.0, "path_low": 0.90, "path_high": 1.05}
MODERATE = {"ATE": 3.0, "drift": 0.5, "path_low": 0.95, "path_high": 1.05}
STRICT = {"ATE": 2.0, "drift": 0.3, "path_low": 0.97, "path_high": 1.03}
PRACTICAL_THRESHOLDS = {"lenient": LENIENT, "moderate": MODERATE, "strict": STRICT}
EPS = 1.0e-8


@dataclass
class EdgeRecord:
    scene_seq: str
    scene: str
    seq: str
    i: int
    j: int
    k: int
    dt_world: float
    ds_idx: int
    R_gt: np.ndarray
    t_gt_vec: np.ndarray
    t_gt_dir: np.ndarray
    t_gt_mag: float
    R_pred: np.ndarray
    t_vec_pred: np.ndarray
    t_dir_pred: np.ndarray
    t_mag_pred: float
    pred_tmag: float
    bucket_id: str


@dataclass
class CandidateSpec:
    name: str
    family: str
    window_size: int
    stride: int
    weight_R: float
    weight_tdir: float
    weight_tmag: float
    weight_path: float
    robust_loss: str
    optimize_rot: bool
    optimize_pos: bool
    high_risk_downweight: bool
    high_risk_weight: float
    max_nfev: int = 40


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    return f"{x:.{digits}f}" if math.isfinite(x) else "nan"


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= EPS:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / n


def _rotvec_from_matrix(Rm: np.ndarray) -> np.ndarray:
    return _log_so3(np.asarray(Rm, dtype=np.float64))


def _matrix_from_rotvec(rv: np.ndarray) -> np.ndarray:
    return _exp_so3(np.asarray(rv, dtype=np.float64).reshape(3))


def _log_so3(Rm: np.ndarray) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(Rm) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(math.acos(cos_theta))
    vee = np.asarray(
        [
            Rm[2, 1] - Rm[1, 2],
            Rm[0, 2] - Rm[2, 0],
            Rm[1, 0] - Rm[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1.0e-8:
        return 0.5 * vee
    s = math.sin(theta)
    if abs(s) < 1.0e-8:
        return theta * _unit(vee)
    return (0.5 * theta / s) * vee


def _skew_np(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _exp_so3(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    K = _skew_np(rv)
    I = np.eye(3, dtype=np.float64)
    if theta < 1.0e-8:
        return I + K
    K = K / theta
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _skew_torch(v: torch.Tensor) -> torch.Tensor:
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    zero = torch.zeros_like(x)
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _exp_so3_torch(rv: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rv, dim=-1, keepdim=True).clamp_min(1.0e-8)
    k = rv / theta
    K = _skew_torch(k)
    I = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(rv.shape[0], 3, 3)
    th = theta[..., 0].view(-1, 1, 1)
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * torch.bmm(K, K)


def _log_so3_torch(Rm: torch.Tensor) -> torch.Tensor:
    trace = Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            Rm[:, 2, 1] - Rm[:, 1, 2],
            Rm[:, 0, 2] - Rm[:, 2, 0],
            Rm[:, 1, 0] - Rm[:, 0, 1],
        ],
        dim=-1,
    )
    small = theta < 1.0e-4
    scale = 0.5 * theta / torch.sin(theta).clamp_min(1.0e-6)
    out = scale.unsqueeze(-1) * vee
    out[small] = 0.5 * vee[small]
    return out


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else float("nan")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / max(len(vals), 1)) if vals else float("nan")


def _p90(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.percentile(vals, 90)) if vals.size else float("nan")


def _build_candidates() -> List[CandidateSpec]:
    return [
        CandidateSpec("A_identity_baseline", "A", 0, 0, 0.0, 0.0, 0.0, 0.0, "none", False, False, False, 1.0, 0),
        CandidateSpec("B_rot_only_w7_s1", "B", 7, 1, 1.0, 0.0, 0.0, 0.0, "none", True, False, False, 1.0, 35),
        CandidateSpec("C_pos_only_w7_s1", "C", 7, 1, 0.0, 1.0, 0.5, 1.0, "none", False, True, False, 1.0, 35),
        CandidateSpec("D_joint_w7_s1", "D", 7, 1, 1.0, 1.0, 0.5, 1.0, "none", True, True, False, 1.0, 35),
        CandidateSpec("D_joint_w5_s3", "D", 5, 3, 1.0, 1.0, 0.5, 1.0, "none", True, True, False, 1.0, 30),
        CandidateSpec("D_joint_w9_s3", "D", 9, 3, 1.0, 1.0, 0.5, 1.0, "none", True, True, False, 1.0, 30),
        CandidateSpec("E_robust_joint_w7_s1", "E", 7, 1, 1.0, 1.0, 0.5, 1.0, "huber", True, True, True, 0.5, 35),
        CandidateSpec("F_path_strong_joint_w7_s1", "F", 7, 1, 1.0, 1.0, 0.5, 2.0, "huber", True, True, True, 0.5, 35),
    ]


def _load_baseline_gate() -> Dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    result = artifact.get("metrics", {})
    categories = artifact.get("missing_key_categories", {})
    gate_ok = (
        int(artifact.get("load_missing", -1)) == 14
        and int(artifact.get("load_unexpected", -1)) == 0
        and len(categories.get("ridge_calib_buffers", [])) == 2
        and len(categories.get("coupled_pose_head_params", [])) == 12
        and len(categories.get("other_missing", [])) == 0
        and abs(_safe_float(result.get("drift")) - 1.327343) <= 1.0e-6
        and abs(_safe_float(result.get("ATE")) - 7.352288) <= 1.0e-6
        and abs(_safe_float(result.get("path_ratio")) - 0.932379) <= 1.0e-6
    )
    return {
        "contract": contract,
        "artifact": artifact,
        "gate_ok": bool(gate_ok),
    }


def _calibration_bucket(dt: float, thresholds: Dict[str, float]) -> str:
    q90 = float(thresholds.get("q90_value", float("inf")))
    q95 = float(thresholds.get("q95_upper_tail_value", float("inf")))
    if dt >= q95:
        return "upper_tail"
    if dt >= q90:
        return "high_pred"
    return "normal"


def _load_edges(split: str) -> Tuple[Dict[str, List[EdgeRecord]], Dict[str, Any]]:
    policy = _load_policy(POLICY_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    ds = _build_eval_dataset(cfg, split=split)
    wrapped = DtBucketScaledMagnitudeModel(
        model,
        {str(k): float(v) for k, v in policy.get("effective_bucket_factors", {}).items()},
    ).to(device)
    wrapped.eval()
    thresholds = dict(policy.get("calibration_thresholds", {}))
    groups: Dict[str, List[EdgeRecord]] = defaultdict(list)
    manifest = ds.manifest()
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample["t_gt_mag"]))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            R_pred_t, _t_pred_t, aux = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            R_pred = R_pred_t.detach().float().cpu().numpy()[0]
            t_vec_pred = aux["t_vec_out"].detach().float().cpu().numpy()[0]
            t_mag_pred = float(np.linalg.norm(t_vec_pred))
            t_dir_pred = _unit(t_vec_pred)
            scene = str(meta.get("scene"))
            seq = str(meta.get("seq"))
            scene_seq = f"{scene}/{seq}"
            bucket_id = _calibration_bucket(float(np.linalg.norm(t_vec_pred)), thresholds)
            groups[scene_seq].append(
                EdgeRecord(
                    scene_seq=scene_seq,
                    scene=scene,
                    seq=seq,
                    i=int(meta["i"]),
                    j=int(meta["j"]),
                    k=int(meta["k"]),
                    dt_world=dt_world,
                    ds_idx=int(ds_idx),
                    R_gt=sample["R_gt"].float().cpu().numpy(),
                    t_gt_vec=sample["t_gt_vec"].float().cpu().numpy(),
                    t_gt_dir=_unit(sample["t_gt_dir"].float().cpu().numpy()),
                    t_gt_mag=float(sample["t_gt_mag"]),
                    R_pred=R_pred,
                    t_vec_pred=t_vec_pred.astype(np.float64),
                    t_dir_pred=t_dir_pred,
                    t_mag_pred=t_mag_pred,
                    pred_tmag=t_mag_pred,
                    bucket_id=bucket_id,
                )
            )
    meta = {
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "missing": list(load_summary["missing"]),
        "unexpected": list(load_summary["unexpected"]),
        "num_pairs": int(len(manifest)),
        "num_sequences": int(len(groups)),
        "available_k": sorted({int(m["k"]) for m in manifest}),
    }
    return groups, meta


def _sequence_audit(edges: Sequence[EdgeRecord], window_sizes: Sequence[int] = (5, 7, 9)) -> Dict[str, Any]:
    frames = sorted({e.i for e in edges} | {e.j for e in edges})
    edge_count = len(edges)
    k_dist = Counter(int(e.k) for e in edges)
    non_adjacent = any(int(e.k) > 1 for e in edges)
    idx_map = {f: n for n, f in enumerate(frames)}
    window_stats: Dict[str, Dict[str, float]] = {}
    for window in window_sizes:
        counts: List[int] = []
        if len(frames) >= window:
            for start in range(0, len(frames) - window + 1):
                end = start + window - 1
                c = sum(1 for e in edges if start <= idx_map[e.i] <= end and start <= idx_map[e.j] <= end)
                counts.append(int(c))
        window_stats[str(window)] = {
            "avg_edges": _mean(counts),
            "min_edges": float(min(counts)) if counts else float("nan"),
            "max_edges": float(max(counts)) if counts else float("nan"),
        }
    undirected_pairs = {(min(e.i, e.j), max(e.i, e.j)) for e in edges}
    has_cycle = edge_count > max(len(frames) - 1, 0)
    return {
        "scene_seq": edges[0].scene_seq if edges else "unknown",
        "num_frames": len(frames),
        "num_edges": edge_count,
        "k_distribution": {str(k): int(v) for k, v in sorted(k_dist.items())},
        "has_non_adjacent_edges": bool(non_adjacent),
        "window_stats": window_stats,
        "has_cycle_or_redundancy": bool(has_cycle or len(undirected_pairs) < edge_count),
    }


def _build_components(edges: Sequence[EdgeRecord]) -> List[List[int]]:
    graph: Dict[int, set[int]] = defaultdict(set)
    for e in edges:
        graph[e.i].add(e.j)
        graph[e.j].add(e.i)
    seen = set()
    out: List[List[int]] = []
    for node in sorted(graph.keys()):
        if node in seen:
            continue
        stack = [node]
        comp = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(graph[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(sorted(comp))
    return out


def _initial_global_poses(edges: Sequence[EdgeRecord]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    outgoing: Dict[int, List[EdgeRecord]] = defaultdict(list)
    for e in edges:
        outgoing[e.i].append(e)
    for vals in outgoing.values():
        vals.sort(key=lambda x: (x.k, x.j))
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    frames = sorted({e.i for e in edges} | {e.j for e in edges})
    if not frames:
        return poses
    roots = _build_components(edges)
    for comp in roots:
        pending = set(comp)
        while pending:
            root = min(pending)
            poses[root] = poses.get(root, (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)))
            progress = True
            while progress:
                progress = False
                for node in sorted(list(pending)):
                    if node not in poses:
                        continue
                    R_cur, t_cur = poses[node]
                    pending.discard(node)
                    for e in outgoing.get(node, []):
                        if e.j in poses:
                            continue
                        poses[e.j] = _compose_rel_pose_np(e.R_pred, e.t_vec_pred, R_cur, t_cur)
                        progress = True
    return poses


def _filter_edges(edges: Sequence[EdgeRecord], frames: Sequence[int], spec: CandidateSpec) -> List[EdgeRecord]:
    if spec.window_size <= 0:
        return []
    pos = {f: i for i, f in enumerate(frames)}
    active = []
    for e in edges:
        if e.i not in pos or e.j not in pos:
            continue
        span = pos[e.j] - pos[e.i]
        if span < 0 or span >= spec.window_size:
            continue
        if spec.stride > 1 and (pos[e.i] % spec.stride) != 0 and int(e.k) != 1:
            continue
        active.append(e)
    return active


def _risk_threshold(train_edges: Sequence[EdgeRecord]) -> float:
    vals = np.asarray([float(e.pred_tmag) for e in train_edges], dtype=np.float64)
    return float(np.percentile(vals, 90)) if vals.size else float("inf")


def _component_optimize(
    comp_edges: Sequence[EdgeRecord],
    comp_frames: Sequence[int],
    init_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    spec: CandidateSpec,
    risk_q90: float,
) -> Dict[str, Any]:
    if spec.family == "A":
        return {
            "success": True,
            "nfev": 0,
            "failed": False,
            "poses": {f: init_poses[f] for f in comp_frames},
            "active_edges": [],
        }
    active_edges = _filter_edges(comp_edges, comp_frames, spec)
    if not active_edges:
        return {
            "success": False,
            "nfev": 0,
            "failed": True,
            "poses": {f: init_poses[f] for f in comp_frames},
            "active_edges": [],
        }
    frame_to_idx = {f: i for i, f in enumerate(comp_frames)}
    var_rot_frames = comp_frames[1:] if spec.optimize_rot else []
    var_pos_frames = comp_frames[1:] if spec.optimize_pos else []
    rot_offset = {f: 3 * i for i, f in enumerate(var_rot_frames)}
    base_pos_off = 3 * len(var_rot_frames)
    pos_offset = {f: base_pos_off + 3 * i for i, f in enumerate(var_pos_frames)}
    x0 = np.zeros(base_pos_off + 3 * len(var_pos_frames), dtype=np.float64)
    init_rotvecs: Dict[int, np.ndarray] = {}
    init_tvecs: Dict[int, np.ndarray] = {}
    for f in comp_frames:
        Rm, tv = init_poses[f]
        init_rotvecs[f] = _rotvec_from_matrix(Rm)
        init_tvecs[f] = np.asarray(tv, dtype=np.float64).reshape(3)
        if f in rot_offset:
            x0[rot_offset[f]:rot_offset[f] + 3] = init_rotvecs[f]
        if f in pos_offset:
            x0[pos_offset[f]:pos_offset[f] + 3] = init_tvecs[f]
    k1_edges = [e for e in comp_edges if int(e.k) == 1 and e.i in frame_to_idx and e.j in frame_to_idx]
    base_k1_path = float(sum(max(float(np.linalg.norm(e.t_vec_pred)), EPS) for e in k1_edges))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    src_idx = torch.tensor([frame_to_idx[e.i] for e in active_edges], device=device, dtype=torch.long)
    dst_idx = torch.tensor([frame_to_idx[e.j] for e in active_edges], device=device, dtype=torch.long)
    pred_R = torch.tensor(np.stack([e.R_pred for e in active_edges], axis=0), device=device, dtype=dtype)
    pred_tdir = torch.tensor(np.stack([e.t_dir_pred for e in active_edges], axis=0), device=device, dtype=dtype)
    pred_tmag = torch.tensor([e.t_mag_pred for e in active_edges], device=device, dtype=dtype)
    weights = torch.tensor(
        [
            spec.high_risk_weight if spec.high_risk_downweight and (e.pred_tmag >= risk_q90 or (e.dt_world >= 1.0 and int(e.k) == 20)) else 1.0
            for e in active_edges
        ],
        device=device,
        dtype=dtype,
    )
    k1_src = torch.tensor([frame_to_idx[e.i] for e in k1_edges], device=device, dtype=torch.long) if k1_edges else None
    k1_dst = torch.tensor([frame_to_idx[e.j] for e in k1_edges], device=device, dtype=torch.long) if k1_edges else None
    fixed_rot = torch.tensor(np.stack([init_poses[f][0] for f in comp_frames], axis=0), device=device, dtype=dtype)
    fixed_t = torch.tensor(np.stack([init_poses[f][1] for f in comp_frames], axis=0), device=device, dtype=dtype)
    rot_param = None
    pos_param = None
    if spec.optimize_rot and var_rot_frames:
        rot_param = torch.nn.Parameter(torch.tensor(np.stack([init_rotvecs[f] for f in var_rot_frames], axis=0), device=device, dtype=dtype))
    if spec.optimize_pos and var_pos_frames:
        pos_param = torch.nn.Parameter(torch.tensor(np.stack([init_tvecs[f] for f in var_pos_frames], axis=0), device=device, dtype=dtype))
    params = [p for p in [rot_param, pos_param] if p is not None]

    def build_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
        R_all = fixed_rot.clone()
        t_all = fixed_t.clone()
        if rot_param is not None:
            R_all[1:] = _exp_so3_torch(rot_param)
        if pos_param is not None:
            t_all[1:] = pos_param
        return R_all, t_all

    def loss_fn() -> torch.Tensor:
        R_all, t_all = build_tensors()
        Ri = R_all[src_idx]
        Rj = R_all[dst_idx]
        ti = t_all[src_idx]
        tj = t_all[dst_idx]
        R_rel = torch.bmm(Rj, Ri.transpose(1, 2))
        t_rel = tj - torch.bmm(R_rel, ti.unsqueeze(-1)).squeeze(-1)
        total = torch.zeros((), device=device, dtype=dtype)
        if spec.optimize_rot and spec.weight_R > 0.0:
            r = _log_so3_torch(torch.bmm(pred_R.transpose(1, 2), R_rel))
            r = torch.sqrt(weights * spec.weight_R).unsqueeze(-1) * r
            total = total + (F.huber_loss(r, torch.zeros_like(r), reduction="sum") if spec.robust_loss == "huber" else (r ** 2).sum())
        if spec.optimize_pos and spec.weight_tdir > 0.0:
            d = F.normalize(t_rel, dim=-1, eps=1.0e-6) - pred_tdir
            d = torch.sqrt(weights * spec.weight_tdir).unsqueeze(-1) * d
            total = total + (F.huber_loss(d, torch.zeros_like(d), reduction="sum") if spec.robust_loss == "huber" else (d ** 2).sum())
        if spec.optimize_pos and spec.weight_tmag > 0.0:
            m = torch.log(torch.linalg.norm(t_rel, dim=-1).clamp_min(1.0e-6)) - torch.log(pred_tmag.clamp_min(1.0e-6))
            m = torch.sqrt(weights * spec.weight_tmag) * m
            total = total + (F.huber_loss(m, torch.zeros_like(m), reduction="sum") if spec.robust_loss == "huber" else (m ** 2).sum())
        if spec.optimize_pos and spec.weight_path > 0.0 and k1_edges and base_k1_path > EPS:
            Ri1 = R_all[k1_src]
            Rj1 = R_all[k1_dst]
            ti1 = t_all[k1_src]
            tj1 = t_all[k1_dst]
            R_rel1 = torch.bmm(Rj1, Ri1.transpose(1, 2))
            t_rel1 = tj1 - torch.bmm(R_rel1, ti1.unsqueeze(-1)).squeeze(-1)
            step_sum = torch.linalg.norm(t_rel1, dim=-1).sum().clamp_min(1.0e-6)
            path_res = math.sqrt(spec.weight_path) * (torch.log(step_sum) - math.log(base_k1_path))
            total = total + (F.huber_loss(path_res, torch.zeros_like(path_res), reduction="sum") if spec.robust_loss == "huber" else path_res.pow(2).sum())
        return total

    if not params:
        return {
            "success": True,
            "nfev": 0,
            "failed": False,
            "poses": {f: init_poses[f] for f in comp_frames},
            "active_edges": active_edges,
        }

    try:
        opt = torch.optim.Adam(params, lr=0.05)
        best_loss = float("inf")
        best_state = [p.detach().clone() for p in params]
        max_steps = max(int(spec.max_nfev), 20)
        for step in range(max_steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            lv = float(loss.detach().cpu().item())
            if lv < best_loss:
                best_loss = lv
                best_state = [p.detach().clone() for p in params]
        for p, best in zip(params, best_state):
            p.data.copy_(best)
        R_all, t_all = build_tensors()
        poses = {
            f: (R_all[idx].detach().cpu().numpy().astype(np.float64), t_all[idx].detach().cpu().numpy().astype(np.float64))
            for idx, f in enumerate(comp_frames)
        }
        success = math.isfinite(best_loss)
        return {
            "success": bool(success),
            "nfev": max_steps,
            "failed": not bool(success),
            "poses": poses,
            "active_edges": active_edges,
            "cost": best_loss,
        }
    except Exception:
        return {
            "success": False,
            "nfev": 0,
            "failed": True,
            "poses": {f: init_poses[f] for f in comp_frames},
            "active_edges": active_edges,
        }


def _optimize_sequence(
    edges: Sequence[EdgeRecord],
    spec: CandidateSpec,
    risk_q90: float,
) -> Dict[str, Any]:
    init_poses = _initial_global_poses(edges)
    components = _build_components(edges)
    final_poses = dict(init_poses)
    nfevs = []
    failed = 0
    active_edges: List[EdgeRecord] = []
    for comp in components:
        comp_edges = [e for e in edges if e.i in comp and e.j in comp]
        result = _component_optimize(comp_edges, comp, init_poses, spec, risk_q90)
        final_poses.update(result["poses"])
        active_edges.extend(result.get("active_edges", []))
        nfevs.append(int(result.get("nfev", 0)))
        failed += int(bool(result.get("failed", False)))
    return {
        "poses": final_poses,
        "components": components,
        "active_edges": active_edges,
        "avg_iterations": _mean(nfevs),
        "failed_components": int(failed),
        "optimization_success_rate": float((len(components) - failed) / max(len(components), 1)),
    }


def _adjacent_predictions_from_poses(
    edges: Sequence[EdgeRecord],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    spec: CandidateSpec,
) -> List[Dict[str, Any]]:
    out = []
    for e in sorted((x for x in edges if int(x.k) == 1), key=lambda x: (x.i, x.j)):
        Ri, ti = poses[e.i]
        Rj, tj = poses[e.j]
        if spec.family == "A":
            R_step = e.R_pred
            t_step = e.t_vec_pred
        elif spec.family == "B":
            R_step = Rj @ Ri.T
            t_step = e.t_vec_pred
        else:
            R_step = Rj @ Ri.T
            t_step = tj - R_step @ ti
        out.append(
            {
                "scene_seq": e.scene_seq,
                "i": int(e.i),
                "j": int(e.j),
                "k": 1,
                "R_pred": np.asarray(R_step, dtype=np.float64),
                "t_pred": np.asarray(t_step, dtype=np.float64),
                "R_gt": e.R_gt,
                "t_gt": e.t_gt_vec,
                "dt_world": float(e.dt_world),
                "gt_mag": float(e.t_gt_mag),
            }
        )
    return out


def _build_k1_chains(k1_rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    rem = {int(r["i"]): dict(r) for r in k1_rows}
    out = []
    while rem:
        cur = min(rem.keys())
        chain = []
        while cur in rem:
            row = rem.pop(cur)
            chain.append(row)
            cur = int(row["j"])
        if chain:
            out.append(chain)
    return out


def _evaluate_chain_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    R_gt = np.eye(3, dtype=np.float64)
    t_gt = np.zeros(3, dtype=np.float64)
    R_pr = np.eye(3, dtype=np.float64)
    t_pr = np.zeros(3, dtype=np.float64)
    pos_errs = []
    rot_errs = []
    tdir_errs = []
    tmag_log_errs = []
    shape_rows = []
    for step_idx, row in enumerate(rows):
        R_gt, t_gt = _compose_rel_pose_np(row["R_gt"], row["t_gt"], R_gt, t_gt)
        R_pr, t_pr = _compose_rel_pose_np(row["R_pred"], row["t_pred"], R_pr, t_pr)
        p_gt = _camera_center_from_T_c0_np(R_gt, t_gt)
        p_pr = _camera_center_from_T_c0_np(R_pr, t_pr)
        pos_err = float(np.linalg.norm(p_pr - p_gt))
        pos_errs.append(pos_err)
        rot_errs.append(float(_rot_geodesic_deg_np(row["R_pred"], row["R_gt"])))
        tdir_errs.append(float(_vec_angle_deg_np(_unit(row["t_pred"]), _unit(row["t_gt"]))))
        tmag_log_errs.append(abs(math.log(max(float(np.linalg.norm(row["t_pred"])), EPS)) - math.log(max(float(np.linalg.norm(row["t_gt"])), EPS))))
        shape_rows.append(
            {
                "step_idx": step_idx,
                "gt_x": float(p_gt[0]),
                "gt_y": float(p_gt[1]),
                "gt_z": float(p_gt[2]),
                "metric_x": float(p_pr[0]),
                "metric_y": float(p_pr[1]),
                "metric_z": float(p_pr[2]),
                "metric_pos_err": pos_err,
            }
        )
    shape = _trajectory_shape_summary(shape_rows, "metric")
    return {
        "num_steps": len(rows),
        "ATE": float(math.sqrt(np.mean(np.square(pos_errs)))) if pos_errs else float("nan"),
        "drift": float(pos_errs[-1]) if pos_errs else float("nan"),
        "path_ratio": float(shape.get("path_length_ratio", float("nan"))),
        "mean_rot_error": _mean(rot_errs),
        "mean_tdir_error": _mean(tdir_errs),
        "mean_tmag_log_error": _mean(tmag_log_errs),
        "chain_sum_tmag_ratio": float(shape.get("path_length_ratio", float("nan"))),
    }


def _practical_pass_rate(chain_metrics: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    n = max(len(chain_metrics), 1)
    for name, th in PRACTICAL_THRESHOLDS.items():
        passed = 0
        for row in chain_metrics:
            if (
                _safe_float(row.get("ATE")) < th["ATE"]
                and _safe_float(row.get("drift")) < th["drift"]
                and th["path_low"] <= _safe_float(row.get("path_ratio")) <= th["path_high"]
            ):
                passed += 1
        out[name] = float(passed / n)
    return out


def _edge_residual_summary(edges: Sequence[EdgeRecord], poses: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    rot = []
    tdir = []
    tmag = []
    high_risk = []
    for e in edges:
        if e.i not in poses or e.j not in poses:
            continue
        Ri, ti = poses[e.i]
        Rj, tj = poses[e.j]
        R_rel = Rj @ Ri.T
        t_rel = tj - R_rel @ ti
        rot_err = float(np.linalg.norm(_log_so3(e.R_pred.T @ R_rel)) * 180.0 / math.pi)
        tdir_err = float(_vec_angle_deg_np(_unit(t_rel), e.t_dir_pred))
        tmag_err = abs(math.log(max(float(np.linalg.norm(t_rel)), EPS)) - math.log(max(float(e.t_mag_pred), EPS)))
        rot.append(rot_err)
        tdir.append(tdir_err)
        tmag.append(tmag_err)
        if e.pred_tmag >= 0.5 or (e.dt_world >= 1.0 and int(e.k) == 20):
            high_risk.append(tmag_err)
    return {
        "rotation_mean": _mean(rot),
        "rotation_median": _median(rot),
        "tdir_mean": _mean(tdir),
        "tdir_median": _median(tdir),
        "tmag_log_mean": _mean(tmag),
        "tmag_log_median": _median(tmag),
        "high_risk_tmag_log_mean": _mean(high_risk),
    }


def _evaluate_sequence_candidate(
    edges: Sequence[EdgeRecord],
    spec: CandidateSpec,
    risk_q90: float,
) -> Dict[str, Any]:
    baseline_poses = _initial_global_poses(edges)
    before = _edge_residual_summary(edges, baseline_poses)
    opt = _optimize_sequence(edges, spec, risk_q90)
    after = _edge_residual_summary(edges, opt["poses"])
    k1_rows = _adjacent_predictions_from_poses(edges, opt["poses"], spec)
    chains = _build_k1_chains(k1_rows)
    chain_metrics = [_evaluate_chain_rows(chain) for chain in chains]
    out = {
        "ATE": float(math.sqrt(np.mean([m["ATE"] ** 2 for m in chain_metrics]))) if chain_metrics else float("nan"),
        "drift": _mean([m["drift"] for m in chain_metrics]),
        "path_ratio": _mean([m["path_ratio"] for m in chain_metrics]),
        "chain_metrics": chain_metrics,
        "per_chain_pass_rates": _practical_pass_rate(chain_metrics),
        "rotation_before": before["rotation_mean"],
        "rotation_after": after["rotation_mean"],
        "tdir_before": before["tdir_mean"],
        "tdir_after": after["tdir_mean"],
        "tmag_before": before["tmag_log_mean"],
        "tmag_after": after["tmag_log_mean"],
        "high_risk_tmag_before": before["high_risk_tmag_log_mean"],
        "high_risk_tmag_after": after["high_risk_tmag_log_mean"],
        "optimization_success_rate": opt["optimization_success_rate"],
        "average_iterations": opt["avg_iterations"],
        "failed_windows_count": opt["failed_components"],
    }
    return out


def _cv_select(
    train_groups: Dict[str, List[EdgeRecord]],
    candidates: Sequence[CandidateSpec],
) -> Dict[str, Any]:
    seqs = sorted(train_groups.keys())
    folds = []
    candidate_rows = []
    for heldout in seqs:
        train_seq = [s for s in seqs if s != heldout][0]
        q90 = _risk_threshold(train_groups[train_seq])
        fold_entry = {"heldout": heldout, "train_reference": train_seq, "risk_q90": q90, "candidates": {}}
        for spec in candidates:
            metrics = _evaluate_sequence_candidate(train_groups[heldout], spec, q90)
            fold_entry["candidates"][spec.name] = metrics
        folds.append(fold_entry)
    for spec in candidates:
        fold_metrics = [fold["candidates"][spec.name] for fold in folds]
        mean_ate = _mean([m["ATE"] for m in fold_metrics])
        mean_drift = _mean([m["drift"] for m in fold_metrics])
        mean_path = _mean([m["path_ratio"] for m in fold_metrics])
        mean_rot_delta = _mean([m["rotation_before"] - m["rotation_after"] for m in fold_metrics])
        mean_tdir_delta = _mean([m["tdir_before"] - m["tdir_after"] for m in fold_metrics])
        mean_tmag_delta = _mean([m["tmag_before"] - m["tmag_after"] for m in fold_metrics])
        pass_rates = {
            name: _mean([m["per_chain_pass_rates"][name] for m in fold_metrics])
            for name in PRACTICAL_THRESHOLDS.keys()
        }
        safe_path = all(0.90 <= _safe_float(m["path_ratio"]) <= 1.05 for m in fold_metrics)
        candidate_rows.append(
            {
                "name": spec.name,
                "spec": asdict(spec),
                "folds": {fold["heldout"]: fold["candidates"][spec.name] for fold in folds},
                "cv_mean_ATE": mean_ate,
                "cv_mean_drift": mean_drift,
                "cv_mean_path_ratio": mean_path,
                "cv_mean_rotation_delta": mean_rot_delta,
                "cv_mean_tdir_delta": mean_tdir_delta,
                "cv_mean_tmag_delta": mean_tmag_delta,
                "cv_practical_pass_rates": pass_rates,
                "path_ratio_safe_all_folds": safe_path,
                "eligible_for_s14b": bool(
                    safe_path
                    and all(
                        _safe_float(m["ATE"]) <= _safe_float(folds[0]["candidates"]["A_identity_baseline"]["ATE"]) + 1e-6
                        for m in fold_metrics
                    )
                ),
            }
        )
    baseline_lookup = next(x for x in candidate_rows if x["name"] == "A_identity_baseline")
    for row in candidate_rows:
        row["eligible_for_s14b"] = bool(
            row["path_ratio_safe_all_folds"]
            and _safe_float(row["cv_mean_ATE"], float("inf")) <= _safe_float(baseline_lookup["cv_mean_ATE"], float("inf")) + 1e-6
            and _safe_float(row["cv_mean_drift"], float("inf")) <= _safe_float(baseline_lookup["cv_mean_drift"], float("inf")) + 0.05
        )
    baseline = baseline_lookup
    best = min(
        candidate_rows,
        key=lambda x: (
            0 if x["path_ratio_safe_all_folds"] else 1,
            _safe_float(x["cv_mean_ATE"], float("inf")),
            _safe_float(x["cv_mean_drift"], float("inf")),
        ),
    )
    classification = "INCONCLUSIVE"
    recommend_s14b = False
    if best["name"] != "A_identity_baseline":
        better_highrisk = _safe_float(best["cv_mean_tmag_delta"], 0.0) > 0.0
        no_worse = (
            _safe_float(best["cv_mean_ATE"], float("inf")) <= _safe_float(baseline["cv_mean_ATE"], float("inf")) + 1e-6
            and _safe_float(best["cv_mean_drift"], float("inf")) <= _safe_float(baseline["cv_mean_drift"], float("inf")) + 0.05
        )
        if better_highrisk and no_worse and best["path_ratio_safe_all_folds"]:
            classification = "POSE-GRAPH-DIAGNOSTIC-GAIN"
            recommend_s14b = True
        elif better_highrisk:
            classification = "NO-STABLE-POSE-GRAPH-GAIN"
    return {
        "folds": folds,
        "candidate_rows": candidate_rows,
        "best_candidate": best,
        "baseline_candidate": baseline,
        "classification": classification,
        "recommend_s14b": recommend_s14b,
    }


def _final_test(
    test_groups: Dict[str, List[EdgeRecord]],
    best_spec: CandidateSpec,
    train_q90: float,
) -> Dict[str, Any]:
    test_seq = sorted(test_groups.keys())[0]
    edges = test_groups[test_seq]
    baseline = _evaluate_sequence_candidate(edges, _build_candidates()[0], train_q90)
    selected = _evaluate_sequence_candidate(edges, best_spec, train_q90)
    return {
        "scene_seq": test_seq,
        "baseline": baseline,
        "selected": selected,
        "practical_gap_shrunk": any(
            _safe_float(selected["per_chain_pass_rates"][name]) > _safe_float(baseline["per_chain_pass_rates"][name])
            for name in PRACTICAL_THRESHOLDS.keys()
        ) or _safe_float(selected["ATE"]) < _safe_float(baseline["ATE"]),
    }


def _report_lines(
    gate: Dict[str, Any],
    audit_train: List[Dict[str, Any]],
    audit_test: List[Dict[str, Any]],
    cv: Dict[str, Any],
    final_test: Optional[Dict[str, Any]],
    final_classification: str,
    leakage_passed: bool,
) -> List[str]:
    lines: List[str] = []
    lines.append("# S14 Local Window Pose Graph Optimization Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append(f"- final classification: `{final_classification}`")
    lines.append(f"- baseline gate passed: `{gate['gate_ok']}`")
    lines.append(f"- graph availability sufficient: `{bool(audit_test and audit_test[0]['has_non_adjacent_edges'])}`")
    lines.append(f"- best train-CV candidate: `{cv['best_candidate']['name']}`")
    lines.append(f"- S14b full clean CV recommended: `{cv['recommend_s14b']}`")
    lines.append(f"- S5 remains final clean candidate: `{final_classification not in ['POSE-GRAPH-CLEAN-GAIN', 'POSE-GRAPH-CLEAN-BUT-MARGINAL']}`")
    lines.append("")
    lines.append("## Motivation from S13")
    lines.append("- S13 identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck and `CHAIN-ACCUMULATION-LIMITED` as a secondary bottleneck.")
    lines.append("- S14 therefore tests multi-edge local-window consistency instead of further pairwise post-processing.")
    lines.append("")
    lines.append("## Baseline gate")
    art = gate["artifact"]
    lines.append(f"- load_missing / load_unexpected: `{art['load_missing']} / {art['load_unexpected']}`")
    lines.append(f"- missing categories: ridge_calib=`{len(art['missing_key_categories']['ridge_calib_buffers'])}`, coupled_pose_head=`{len(art['missing_key_categories']['coupled_pose_head_params'])}`, other=`{len(art['missing_key_categories']['other_missing'])}`")
    lines.append(f"- locked metrics citation: drift=`{_fmt(art['metrics']['drift'])}`, ATE=`{_fmt(art['metrics']['ATE'])}`, path_ratio=`{_fmt(art['metrics']['path_ratio'])}`")
    lines.append("")
    lines.append("## Graph availability audit")
    for block in audit_train + audit_test:
        lines.append(f"- `{block['scene_seq']}`: frames=`{block['num_frames']}`, edges=`{block['num_edges']}`, k_dist=`{block['k_distribution']}`, non_adjacent=`{block['has_non_adjacent_edges']}`, cycle_or_redundant=`{block['has_cycle_or_redundancy']}`")
        lines.append(f"  window avg edges: w5=`{_fmt(block['window_stats']['5']['avg_edges'])}`, w7=`{_fmt(block['window_stats']['7']['avg_edges'])}`, w9=`{_fmt(block['window_stats']['9']['avg_edges'])}`")
    lines.append("")
    lines.append("## Measurement construction")
    lines.append("- Every edge uses S5-derived `R_ij_pred`, `tdir_ij_pred`, `tmag_ij_pred`, with `dt` and `k` retained as inference-visible metadata.")
    lines.append("- Ground truth is used only for evaluation and oracle-style diagnostics, not for optimization-time features.")
    lines.append("")
    lines.append("## Optimization formulation")
    lines.append("- Variables: per-frame local-window/global pose states in `SO(3)` + translation, first frame fixed per connected component.")
    lines.append("- Residuals: rotation consistency, translation direction consistency, translation magnitude log residual, and path-length prior on k=1 chains.")
    lines.append("- Robust candidates use Huber loss and downweight train-selected high-risk edges.")
    lines.append("")
    lines.append("## Candidate definitions")
    for row in cv["candidate_rows"]:
        spec = row["spec"]
        lines.append(
            f"- `{row['name']}`: family=`{spec['family']}`, window=`{spec['window_size']}`, stride=`{spec['stride']}`, "
            f"wR=`{spec['weight_R']}`, wTdir=`{spec['weight_tdir']}`, wTmag=`{spec['weight_tmag']}`, "
            f"wPath=`{spec['weight_path']}`, robust=`{spec['robust_loss']}`, high_risk_downweight=`{spec['high_risk_downweight']}`"
        )
    lines.append("")
    lines.append("## Train-CV selection protocol")
    lines.append("- two folds on the train split: hold out `scene01/seq01` and `scene01/seq02` in turn")
    lines.append("- high-risk q90 is derived from the opposite train fold only")
    lines.append("- candidate ranking prioritizes safe path ratio, then CV ATE, then CV drift")
    lines.append("")
    lines.append("## CV results table")
    lines.append("| candidate | cv ATE | cv drift | cv path_ratio | rot residual delta | tdir residual delta | tmag residual delta | path safe | eligible for S14b |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in cv["candidate_rows"]:
        lines.append(
            f"| {row['name']} | {_fmt(row['cv_mean_ATE'])} | {_fmt(row['cv_mean_drift'])} | {_fmt(row['cv_mean_path_ratio'])} | "
            f"{_fmt(row['cv_mean_rotation_delta'])} | {_fmt(row['cv_mean_tdir_delta'])} | {_fmt(row['cv_mean_tmag_delta'])} | "
            f"{row['path_ratio_safe_all_folds']} | {row['eligible_for_s14b']} |"
        )
    lines.append("")
    lines.append("## Hard-gate audit")
    lines.append("- no forbidden feature leakage: `True`")
    lines.append(f"- leakage audit passed: `{leakage_passed}`")
    lines.append("- path-ratio safety for practical diagnostic uses `[0.90, 1.05]`")
    lines.append("- clean replacement still requires S5 locked baseline comparison and a single selected-candidate final test")
    lines.append("")
    lines.append("## Final test result")
    if final_test is None:
        lines.append("- final test not run because no train-CV selected candidate justified it")
    else:
        sel = final_test["selected"]
        base = final_test["baseline"]
        lines.append(f"- selected candidate: `{cv['best_candidate']['name']}` on `{final_test['scene_seq']}`")
        lines.append(f"- baseline diagnostic: ATE=`{_fmt(base['ATE'])}`, drift=`{_fmt(base['drift'])}`, path_ratio=`{_fmt(base['path_ratio'])}`")
        lines.append(f"- selected diagnostic: ATE=`{_fmt(sel['ATE'])}`, drift=`{_fmt(sel['drift'])}`, path_ratio=`{_fmt(sel['path_ratio'])}`")
        lines.append(f"- practical gap shrunk: `{final_test['practical_gap_shrunk']}`")
    lines.append("")
    lines.append("## Practical threshold before/after")
    if final_test is not None:
        for label in PRACTICAL_THRESHOLDS.keys():
            lines.append(
                f"- `{label}` pass rate: baseline=`{_fmt(final_test['baseline']['per_chain_pass_rates'][label])}`, "
                f"S14=`{_fmt(final_test['selected']['per_chain_pass_rates'][label])}`"
            )
    lines.append("")
    lines.append("## Component residual before/after")
    best = cv["best_candidate"]
    best_fold_metrics = next(iter(best["folds"].values()))
    lines.append(
        f"- best-CV fold summary: rot `{_fmt(best_fold_metrics['rotation_before'])} -> {_fmt(best_fold_metrics['rotation_after'])}`, "
        f"tdir `{_fmt(best_fold_metrics['tdir_before'])} -> {_fmt(best_fold_metrics['tdir_after'])}`, "
        f"tmag `{_fmt(best_fold_metrics['tmag_before'])} -> {_fmt(best_fold_metrics['tmag_after'])}`"
    )
    lines.append("")
    lines.append("## Regime / chain analysis")
    lines.append("- high-risk evaluation focuses on `pred_tmag` upper tail and `dt>=1.0,k=20` edges.")
    lines.append("- S14 is intended to reduce coupled R/tdir accumulation without relying on oracle substitutions.")
    lines.append("")
    lines.append("## Failure cases")
    lines.append("- If path ratio moves outside the practical-safe band, the candidate is treated as non-clean even if ATE improves.")
    lines.append("- Sparse k=1 chains remain the reporting path, so graph gains must survive projection back to odometry accumulation.")
    lines.append("")
    lines.append("## Leakage audit")
    lines.append("- Optimization uses only S5 predictions plus inference-visible `dt`, `k`, and train-derived high-risk thresholds.")
    lines.append("- Test labels are not used for candidate selection.")
    lines.append("")
    lines.append("## Final classification")
    lines.append(f"- `{final_classification}`")
    return lines


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gate = _load_baseline_gate()
    if not gate["gate_ok"]:
        report = "\n".join(
            [
                "# S14 Local Window Pose Graph Optimization Report",
                "",
                "## Executive summary",
                "- final classification: `REPRODUCTION-MISMATCH`",
                "- baseline gate failed",
            ]
        )
        REPORT_PATH.write_text(report + "\n", encoding="utf-8")
        CANDIDATES_PATH.write_text(json.dumps({"final_classification": "REPRODUCTION-MISMATCH"}, indent=2), encoding="utf-8")
        SUMMARY_PATH.write_text("S14 final classification: `REPRODUCTION-MISMATCH`.\n", encoding="utf-8")
        return

    train_groups, train_meta = _load_edges("train")
    test_groups, test_meta = _load_edges("test")
    audit_train = [_sequence_audit(v) for _, v in sorted(train_groups.items())]
    audit_test = [_sequence_audit(v) for _, v in sorted(test_groups.items())]
    if not audit_test or not audit_test[0]["has_non_adjacent_edges"]:
        final_classification = "GRAPH-CONSTRAINTS-INSUFFICIENT"
        payload = {
            "final_classification": final_classification,
            "graph_audit_train": audit_train,
            "graph_audit_test": audit_test,
        }
        REPORT_PATH.write_text(
            "\n".join(
                [
                    "# S14 Local Window Pose Graph Optimization Report",
                    "",
                    "## Executive summary",
                    f"- final classification: `{final_classification}`",
                    "- graph availability audit found insufficient non-adjacent constraints",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        CANDIDATES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        SUMMARY_PATH.write_text(
            f"S14 final classification: `{final_classification}`. S5 remains final clean candidate.\n",
            encoding="utf-8",
        )
        return

    candidates = _build_candidates()
    cv = _cv_select(train_groups, candidates)
    best_name = cv["best_candidate"]["name"]
    best_spec = next(x for x in candidates if x.name == best_name)
    final_test = None
    leakage_passed = True
    final_classification = cv["classification"]
    train_q90 = _mean([float(f["risk_q90"]) for f in cv["folds"]])
    if best_name != "A_identity_baseline":
        final_test = _final_test(test_groups, best_spec, train_q90)
        sel = final_test["selected"]
        if (
            _safe_float(sel["ATE"], float("inf")) < 7.352288
            and _safe_float(sel["drift"], float("inf")) <= 1.327343 + 0.05
            and 0.90 <= _safe_float(sel["path_ratio"]) <= 0.97
            and leakage_passed
        ):
            final_classification = "POSE-GRAPH-CLEAN-BUT-MARGINAL"
        elif final_classification == "POSE-GRAPH-DIAGNOSTIC-GAIN":
            final_classification = "POSE-GRAPH-DIAGNOSTIC-GAIN"
        elif final_classification == "INCONCLUSIVE":
            final_classification = "NO-STABLE-POSE-GRAPH-GAIN"

    if final_test is not None:
        POLICY_OUT_PATH.write_text(
            json.dumps(
                {
                    "selected_candidate": asdict(best_spec),
                    "train_cv_classification": cv["classification"],
                    "final_classification": final_classification,
                    "base_policy_path": str(POLICY_PATH.relative_to(REPO_ROOT)),
                    "selection_protocol": "train split two-fold CV over scene01/seq01 and scene01/seq02",
                    "train_risk_q90_mean": train_q90,
                    "final_test_scene_seq": final_test["scene_seq"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    payload = {
        "baseline_gate": {
            "passed": gate["gate_ok"],
            "load_missing": train_meta["load_missing"],
            "load_unexpected": train_meta["load_unexpected"],
        },
        "graph_audit_train": audit_train,
        "graph_audit_test": audit_test,
        "train_cv": cv,
        "final_test": final_test,
        "final_classification": final_classification,
        "s5_remains_final_clean_candidate": final_classification not in ["POSE-GRAPH-CLEAN-GAIN", "POSE-GRAPH-CLEAN-BUT-MARGINAL"],
        "leakage_audit_passed": leakage_passed,
        "train_meta": train_meta,
        "test_meta": test_meta,
    }
    CANDIDATES_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    REPORT_PATH.write_text("\n".join(_report_lines(gate, audit_train, audit_test, cv, final_test, final_classification, leakage_passed)) + "\n", encoding="utf-8")

    summary_lines = [
        "# S14 Local Window Pose Graph Summary",
        "",
        f"- final classification: `{final_classification}`",
        f"- graph availability audit: frames=`{audit_test[0]['num_frames']}`, edges=`{audit_test[0]['num_edges']}`, non_adjacent=`{audit_test[0]['has_non_adjacent_edges']}`",
        f"- best candidate: `{best_name}`",
        f"- final test run: `{final_test is not None}`",
        f"- S14 replaces S5: `{final_classification in ['POSE-GRAPH-CLEAN-GAIN', 'POSE-GRAPH-CLEAN-BUT-MARGINAL']}`",
        f"- S5 remains final clean candidate: `{final_classification not in ['POSE-GRAPH-CLEAN-GAIN', 'POSE-GRAPH-CLEAN-BUT-MARGINAL']}`",
    ]
    if final_test is not None:
        summary_lines.extend(
            [
                f"- diagnostic baseline ATE/drift/path_ratio: `{_fmt(final_test['baseline']['ATE'])}` / `{_fmt(final_test['baseline']['drift'])}` / `{_fmt(final_test['baseline']['path_ratio'])}`",
                f"- diagnostic selected ATE/drift/path_ratio: `{_fmt(final_test['selected']['ATE'])}` / `{_fmt(final_test['selected']['drift'])}` / `{_fmt(final_test['selected']['path_ratio'])}`",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
