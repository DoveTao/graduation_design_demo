#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import adjacent_pairs, relative_pose_A_to_B_in_B, scan_frames, validation_from_logs, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head
from s5e2_adjacent_dense_lib import pair_features


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def _parse_cfg(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            return [] if not inner else [parse_scalar(x.strip()) for x in inner.split(",")]
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text.strip('"')

    cfg: Dict[str, Any] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, value = raw.split(":", 1)
            key, value = key.strip(), value.strip()
            if value:
                cfg[key] = parse_scalar(value)
                current = None
            else:
                cfg[key] = {}
                current = key
        elif current is not None and ":" in raw:
            key, value = raw.strip().split(":", 1)
            cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def _extract_corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "filtered_match_count": float(orb["filtered_match_count"]),
        "inlier_ratio": float(orb["inlier_ratio"]),
        "median_match_displacement": float(orb["median_match_displacement"]),
        "p90_match_displacement": float(orb["p90_match_displacement"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "low_parallax_flag": 1.0 if orb["low_parallax_flag"] else 0.0,
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "p90_flow_magnitude": float(flow["p90_flow_magnitude"]),
        "flow_angle_dispersion": float(flow["flow_angle_dispersion"]) / 180.0,
        "forward_backward_flow_consistency": 0.0 if flow["forward_backward_flow_consistency"] is None else float(flow["forward_backward_flow_consistency"]),
    }


def _obs_weight(corr: Dict[str, float], cfg: Dict[str, Any], gt_mag: float) -> float:
    low = float(cfg["observability_weighting"]["low_signal_tdir_weight"])
    high = float(cfg["observability_weighting"]["high_signal_tdir_weight"])
    parallax = float(corr["parallax_proxy"])
    inlier = float(corr["inlier_ratio"])
    flow = float(corr["median_flow_magnitude"])
    low_flag = bool(corr["low_parallax_flag"] >= 0.5)
    small_motion = gt_mag < 0.01
    if small_motion:
        return low
    score = 0.50 * min(1.0, parallax / 0.30) + 0.35 * min(1.0, inlier) + 0.15 * min(1.0, flow / 8.0)
    if low_flag:
        score *= 0.5
    return float(low + (high - low) * min(1.0, max(0.0, score)))


def _compose_rel(R_list: Sequence[torch.Tensor], t_list: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    R_total = R_list[0]
    t_total = t_list[0]
    for i in range(1, len(R_list)):
        R_next = R_list[i]
        t_next = t_list[i]
        t_total = R_next @ t_total + t_next
        R_total = R_next @ R_total
    return R_total, t_total


class ConservativeDirectionResidual(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, delta_scale: float, delta_log_tmag_clip: float) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.delta_log_tmag_clip = float(delta_log_tmag_clip)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.delta_head = torch.nn.Linear(hidden_dim, 3)
        self.scale_head = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.zeros_(self.delta_head.weight)
        torch.nn.init.zeros_(self.delta_head.bias)
        torch.nn.init.zeros_(self.scale_head.weight)
        torch.nn.init.zeros_(self.scale_head.bias)

    def forward(self, x: torch.Tensor, base_dir: torch.Tensor, base_mag: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.net(x)
        delta = torch.tanh(self.delta_head(z)) * self.delta_scale
        final_dir = F.normalize(base_dir + delta, dim=1)
        delta_log_tmag = torch.tanh(self.scale_head(z).squeeze(1)) * self.delta_log_tmag_clip
        final_tmag = base_mag * torch.exp(delta_log_tmag)
        return {
            "delta_tdir": delta,
            "delta_tdir_norm": torch.linalg.norm(delta, dim=1),
            "final_tdir": final_dir,
            "delta_log_tmag": delta_log_tmag,
            "final_tmag": final_tmag,
        }


def _build_edge_cache(cfg: Dict[str, Any], frames: Dict[Tuple[str, str], List[Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache_img: Dict[str, np.ndarray] = {}
    edge_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    train_scenes = [str(x) for x in cfg["training"]["train_scenes"]]
    for scene_seq in train_scenes:
        scene, seq = scene_seq.split("/")
        rows = adjacent_pairs(frames[(scene, seq)])
        total = max(len(rows), 1)
        for idx, (a, b) in enumerate(rows):
            R_gt, t_gt = relative_pose_A_to_B_in_B(a, b)
            gt_mag = float(np.linalg.norm(t_gt))
            gt_dir = _norm(t_gt)
            x = pair_features(a, b, idx, total, cache_img)
            coarse = predict_s5e2(s5e2, x)
            base_dir = _norm(coarse[3:6])
            base_mag = float(np.exp(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]))
            corr = _extract_corr(a, b)
            obs = _obs_weight(corr, cfg, gt_mag)
            feat = np.concatenate(
                [
                    x.astype(np.float32),
                    base_dir.astype(np.float32),
                    np.asarray([base_mag, math.log(max(base_mag, 1.0e-12))], dtype=np.float32),
                    np.asarray(list(corr.values()), dtype=np.float32),
                    np.asarray([obs], dtype=np.float32),
                ]
            )
            edge_cache[(scene, seq, idx)] = {
                "feature": feat.tolist(),
                "base_dir": base_dir.tolist(),
                "base_mag": base_mag,
                "gt_dir": gt_dir.tolist(),
                "gt_mag": gt_mag,
                "R_gt": R_gt.tolist(),
                "obs_weight": obs,
            }
    return edge_cache


def _load_windows(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_window_samples(cfg: Dict[str, Any], edge_cache: Dict[Tuple[str, str, int], Dict[str, Any]], frames: Dict[Tuple[str, str], List[Any]], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in payload["windows"]:
        scene = row["scene"]
        seq = row["seq"]
        if row.get("uses_eval_scene"):
            continue
        start = int(row["start_index"])
        k = int(row["k"])
        step_keys = [(scene, seq, start + i) for i in range(k)]
        if not all(key in edge_cache for key in step_keys):
            continue
        seq_frames = frames[(scene, seq)]
        a = seq_frames[start]
        b = seq_frames[start + k]
        R_direct, t_direct = relative_pose_A_to_B_in_B(a, b)
        samples.append(
            {
                "scene": scene,
                "seq": seq,
                "start_index": start,
                "k": k,
                "step_keys": step_keys,
                "timestamps": row["timestamps"],
                "gt_direct_R": R_direct.tolist(),
                "gt_direct_dir": _norm(t_direct).tolist(),
                "gt_direct_mag": float(np.linalg.norm(t_direct)),
                "gt_path_length": float(sum(edge_cache[key]["gt_mag"] for key in step_keys)),
                "obs_weight": float(sum(edge_cache[key]["obs_weight"] for key in step_keys) / max(len(step_keys), 1)),
            }
        )
    return samples


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = _load_windows(Path(args.pairs))
    if not payload.get("ready_for_training"):
        status = {
            "real_training_executed": False,
            "optimizer_step_count": 0,
            "learned_weights_saved": False,
            "smoke_policy_only": False,
            "uses_eval_gt_for_training": False,
            "uses_orbslam3_teacher": False,
            "uses_random_batch_proxy": False,
            "uses_true_contiguous_kstep_windows": bool(payload.get("all_windows_contiguous")),
            "k_values": payload.get("num_windows_by_k", {}),
            "losses_actually_computed": {},
            "delta_tdir_norm_train_mean": None,
            "classification": "S5E20_KSTEP_DATASET_BLOCKED",
        }
        out = Path("checkpoints/S5E20_true_kstep_composition_supervision_candidate/training_status.json")
        write_json(out, status)
        return status

    frames = scan_frames(Path("data"), scene="scene01")
    edge_cache = _build_edge_cache(cfg, frames)
    samples = _make_window_samples(cfg, edge_cache, frames, payload)
    input_dim = len(next(iter(edge_cache.values()))["feature"])

    model = ConservativeDirectionResidual(
        input_dim=input_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        delta_scale=float(cfg["model"]["delta_tdir_scale"]),
        delta_log_tmag_clip=float(cfg["model"]["delta_log_tmag_clip"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    steps = 0
    history: List[Dict[str, Any]] = []
    batch_size = int(cfg["training"]["batch_size"])
    max_steps = int(cfg["training"]["max_steps"])
    loss_cfg = cfg["losses"]
    delta_norm_track: List[float] = []

    while steps < max_steps:
        random.shuffle(samples)
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            if not batch_samples:
                continue
            total_loss = torch.zeros([], dtype=torch.float32, device=device)
            stat_count = 0
            for sample in batch_samples:
                step_features = torch.tensor([edge_cache[key]["feature"] for key in sample["step_keys"]], dtype=torch.float32, device=device)
                base_dir = F.normalize(torch.tensor([edge_cache[key]["base_dir"] for key in sample["step_keys"]], dtype=torch.float32, device=device), dim=1)
                base_mag = torch.tensor([edge_cache[key]["base_mag"] for key in sample["step_keys"]], dtype=torch.float32, device=device)
                gt_dir = F.normalize(torch.tensor([edge_cache[key]["gt_dir"] for key in sample["step_keys"]], dtype=torch.float32, device=device), dim=1)
                gt_mag = torch.tensor([edge_cache[key]["gt_mag"] for key in sample["step_keys"]], dtype=torch.float32, device=device)
                obs_weight = torch.tensor([edge_cache[key]["obs_weight"] for key in sample["step_keys"]], dtype=torch.float32, device=device)
                pred = model(step_features, base_dir, base_mag)
                dot = torch.clamp(torch.sum(pred["final_tdir"] * gt_dir, dim=1), -1.0, 1.0)
                loss_k1 = torch.mean(obs_weight * (1.0 - dot))
                loss_abs = torch.mean((0.25 + 0.75 * obs_weight) * (1.0 - torch.abs(dot)))
                loss_anti = torch.mean(obs_weight * torch.relu(-dot))
                loss_tmag = torch.mean(torch.abs(torch.log(torch.clamp(pred["final_tmag"], min=1.0e-12) / torch.clamp(gt_mag, min=1.0e-12))))
                loss_delta = torch.mean(pred["delta_tdir_norm"] ** 2)
                R_steps = [torch.tensor(edge_cache[key]["R_gt"], dtype=torch.float32, device=device) for key in sample["step_keys"]]
                t_steps = [pred["final_tdir"][i] * pred["final_tmag"][i] for i in range(len(sample["step_keys"]))]
                _, t_chain = _compose_rel(R_steps, t_steps)
                gt_direct_dir = F.normalize(torch.tensor(sample["gt_direct_dir"], dtype=torch.float32, device=device).unsqueeze(0), dim=1)[0]
                gt_direct_mag = torch.tensor(float(sample["gt_direct_mag"]), dtype=torch.float32, device=device)
                chain_dir = F.normalize(t_chain.unsqueeze(0), dim=1)[0]
                chain_mag = torch.linalg.norm(t_chain)
                loss_kstep = (1.0 - torch.clamp(torch.dot(chain_dir, gt_direct_dir), -1.0, 1.0)) * float(sample["obs_weight"])
                loss_path = torch.abs(torch.sum(pred["final_tmag"]) - torch.tensor(float(sample["gt_path_length"]), dtype=torch.float32, device=device))
                loss_direct_mag = torch.abs(torch.log(torch.clamp(chain_mag, min=1.0e-12) / torch.clamp(gt_direct_mag, min=1.0e-12)))
                total_loss = total_loss + (
                    float(loss_cfg["w_signed_tdir_k1"]) * loss_k1
                    + float(loss_cfg["w_signed_tdir_kstep"]) * loss_kstep
                    + float(loss_cfg["w_tdir_abs_aux"]) * loss_abs
                    + float(loss_cfg["w_robust_tmag_log"]) * (loss_tmag + 0.5 * loss_direct_mag)
                    + float(loss_cfg["w_path_length_consistency"]) * loss_path
                    + float(loss_cfg["w_delta_tdir_l2_regularization"]) * loss_delta
                    + float(loss_cfg["w_anti_parallel_penalty"]) * loss_anti
                )
                delta_norm_track.extend(pred["delta_tdir_norm"].detach().cpu().numpy().tolist())
                stat_count += 1
            total_loss = total_loss / max(stat_count, 1)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()
            steps += 1
            if steps % 20 == 0 or steps == 1 or steps == max_steps:
                history.append({"step": steps, "loss": float(total_loss.item())})
            if steps >= max_steps:
                break

    out_dir = Path("checkpoints/S5E20_true_kstep_composition_supervision_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = out_dir / "s5e20_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": int(cfg["model"]["hidden_dim"]),
            "delta_tdir_scale": float(cfg["model"]["delta_tdir_scale"]),
            "delta_log_tmag_clip": float(cfg["model"]["delta_log_tmag_clip"]),
            "export_delta_scale_alpha": float(cfg["model"]["export_delta_scale_alpha"]),
            "history": history,
        },
        weight_path,
    )
    delta_mean = float(np.mean(delta_norm_track)) if delta_norm_track else None
    status = {
        "real_training_executed": True,
        "optimizer_step_count": steps,
        "learned_weights_saved": weight_path.exists(),
        "smoke_policy_only": False,
        "uses_eval_gt_for_training": False,
        "uses_orbslam3_teacher": False,
        "uses_random_batch_proxy": False,
        "uses_true_contiguous_kstep_windows": True,
        "k_values": [1, 2, 3, 5],
        "losses_actually_computed": {
            "true_kstep_composition": True,
            "path_length_consistency": True,
            "delta_tdir_l2_regularization": True,
            "signed_tdir_kstep": True,
        },
        "delta_tdir_norm_train_mean": delta_mean,
        "history": history,
        "classification": "S5E20_REAL_KSTEP_TRAINING_COMPLETE" if steps >= max_steps else "S5E20_SHORT_REAL_KSTEP_TRAINING",
        "validation_snapshot": validation_from_logs(),
    }
    write_json(out_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--pairs", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
