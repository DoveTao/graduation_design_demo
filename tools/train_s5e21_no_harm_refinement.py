#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import adjacent_pairs, pair_features, relative_pose_A_to_B_in_B, scan_frames, validation_from_logs, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E15_POLICY = Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/s5e15_refinement_policy.json")


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


class NoHarmRefiner(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.delta_head = torch.nn.Linear(hidden_dim, 3)
        torch.nn.init.zeros_(self.delta_head.weight)
        torch.nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.delta_head(self.net(x)))


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def _corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "inlier_ratio": float(orb["inlier_ratio"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "low_parallax_flag": 1.0 if orb["low_parallax_flag"] else 0.0,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    gate = json.loads(Path(args.gate_dataset).read_text(encoding="utf-8"))
    if not gate.get("gate_dataset_ready"):
        status = {
            "real_training_executed": False,
            "optimizer_step_count": 0,
            "learned_weights_saved": False,
            "smoke_policy_only": False,
            "uses_eval_gt_for_training": False,
            "uses_eval_gt_for_gate": False,
            "uses_orbslam3_teacher": False,
            "uses_external_router": False,
            "modified_train_edge_fraction": 0.0,
            "delta_tdir_norm_train_mean": None,
            "gate_dataset_ready": False,
            "classification": "S5E21_GATE_DATASET_BLOCKED",
        }
        write_json(Path("checkpoints/S5E21_no_harm_observability_gated_refinement_candidate/training_status.json"), status)
        return status

    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale_factor = float(json.loads(S5E15_POLICY.read_text(encoding="utf-8"))["scale_factor"])
    frames = scan_frames(Path("data"), scene="scene01")
    gate_by_key = {(r["scene"], r["seq"], int(r["edge_index"])): r for r in gate["records"]}
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache_img: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for scene_seq in [str(x) for x in cfg["training"]["train_scenes"]]:
        scene, seq = scene_seq.split("/")
        pairs = adjacent_pairs(frames[(scene, seq)])
        total = max(len(pairs), 1)
        for idx, (a, b) in enumerate(pairs):
            gate_row = gate_by_key[(scene, seq, idx)]
            R_gt, t_gt = relative_pose_A_to_B_in_B(a, b)
            gt_dir = _norm(t_gt)
            gt_mag = float(np.linalg.norm(t_gt))
            x = pair_features(a, b, idx, total, cache_img)
            coarse = predict_s5e2(s5e2, x)
            base_dir = _norm(coarse[3:6])
            base_mag = float(np.exp(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0])) * scale_factor
            corr = _corr(a, b)
            feat = np.concatenate(
                [
                    x.astype(np.float32),
                    base_dir.astype(np.float32),
                    np.asarray([base_mag], dtype=np.float32),
                    np.asarray(list(corr.values()), dtype=np.float32),
                    np.asarray([float(gate_row["observability_score"]), float(gate_row["gate_value"])], dtype=np.float32),
                ]
            )
            rows.append(
                {
                    "feature": feat.tolist(),
                    "base_dir": base_dir.tolist(),
                    "base_mag": base_mag,
                    "gt_dir": gt_dir.tolist(),
                    "gt_mag": gt_mag,
                    "gate_value": float(gate_row["gate_value"]),
                    "is_high_confidence": bool(gate_row["is_high_confidence"]),
                    "is_low_signal": bool(gate_row["is_low_signal"]),
                }
            )

    feats = torch.tensor([r["feature"] for r in rows], dtype=torch.float32, device=device)
    base_dir = F.normalize(torch.tensor([r["base_dir"] for r in rows], dtype=torch.float32, device=device), dim=1)
    base_mag = torch.tensor([r["base_mag"] for r in rows], dtype=torch.float32, device=device)
    gt_dir = F.normalize(torch.tensor([r["gt_dir"] for r in rows], dtype=torch.float32, device=device), dim=1)
    gt_mag = torch.tensor([r["gt_mag"] for r in rows], dtype=torch.float32, device=device)
    gate_val = torch.tensor([r["gate_value"] for r in rows], dtype=torch.float32, device=device)
    high_mask = torch.tensor([1.0 if r["is_high_confidence"] else 0.0 for r in rows], dtype=torch.float32, device=device)
    low_mask = torch.tensor([1.0 if r["is_low_signal"] else 0.0 for r in rows], dtype=torch.float32, device=device)

    model = NoHarmRefiner(feats.shape[1], int(cfg["model"]["hidden_dim"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_alpha = float(cfg["delta_tdir"]["max_alpha"])
    hard_clip = float(cfg["delta_tdir"]["hard_clip_norm"])
    optimizer_steps = 0
    all_delta_norms: List[float] = []

    idx_all = torch.arange(feats.shape[0], device=device)
    batch_size = int(cfg["training"]["batch_size"])
    max_steps = int(cfg["training"]["max_steps"])
    for _ in range(max_steps):
        perm = idx_all[torch.randperm(idx_all.numel(), device=device)]
        for start in range(0, len(perm), batch_size):
            bi = perm[start : start + batch_size]
            f = feats[bi]
            bdir = base_dir[bi]
            bmag = base_mag[bi]
            gdir = gt_dir[bi]
            gmag = gt_mag[bi]
            gv = gate_val[bi]
            hm = high_mask[bi]
            lm = low_mask[bi]
            raw = model(f)
            raw_norm = torch.linalg.norm(raw, dim=1, keepdim=True)
            clipped = raw * torch.clamp(hard_clip / torch.clamp(raw_norm, min=1.0e-8), max=1.0)
            effective = clipped * gv.unsqueeze(1) * max_alpha
            final_dir = F.normalize(bdir + effective, dim=1)
            base_dot = torch.clamp(torch.sum(bdir * gdir, dim=1), -1.0, 1.0)
            pred_dot = torch.clamp(torch.sum(final_dir * gdir, dim=1), -1.0, 1.0)
            loss_tdir = torch.sum(hm * (1.0 - pred_dot)) / torch.clamp(hm.sum(), min=1.0)
            loss_anti = torch.sum(hm * torch.relu(-pred_dot)) / torch.clamp(hm.sum(), min=1.0)
            loss_abs = torch.mean((0.2 + 0.8 * hm) * (1.0 - torch.abs(pred_dot)))
            loss_no_harm = torch.mean(torch.relu(base_dot - pred_dot))
            loss_low = torch.mean(lm * torch.linalg.norm(effective, dim=1))
            loss_delta = torch.mean(torch.linalg.norm(effective, dim=1) ** 2)
            loss_tmag = torch.mean(torch.abs(torch.log(torch.clamp(bmag, min=1.0e-12) / torch.clamp(gmag, min=1.0e-12))))
            loss = (
                float(cfg["losses"]["w_signed_tdir"]) * loss_tdir
                + float(cfg["losses"]["w_anti_parallel"]) * loss_anti
                + float(cfg["losses"]["w_tdir_abs_aux"]) * loss_abs
                + float(cfg["losses"]["w_no_harm_margin"]) * (loss_no_harm + 2.0 * loss_low)
                + float(cfg["losses"]["w_delta_tdir_l2_regularization"]) * loss_delta
                + float(cfg["losses"]["w_path_ratio_preservation"]) * 0.0
                + float(cfg["losses"]["w_tmag_preservation"]) * loss_tmag
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            optimizer_steps += 1
            all_delta_norms.extend(torch.linalg.norm(effective.detach(), dim=1).cpu().numpy().tolist())
            if optimizer_steps >= max_steps:
                break
        if optimizer_steps >= max_steps:
            break

    out_dir = Path("checkpoints/S5E21_no_harm_observability_gated_refinement_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = out_dir / "s5e21_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(feats.shape[1]),
            "hidden_dim": int(cfg["model"]["hidden_dim"]),
            "max_alpha": max_alpha,
            "hard_clip_norm": hard_clip,
            "scale_factor": scale_factor,
            "gate_threshold_source": gate["threshold_source"],
            "high_threshold": gate["high_confidence_threshold"],
            "low_threshold": gate["low_signal_threshold"],
        },
        weight_path,
    )
    modified_frac = float(sum(1 for x in all_delta_norms if x > 1.0e-8) / max(len(all_delta_norms), 1))
    status = {
        "real_training_executed": True,
        "optimizer_step_count": optimizer_steps,
        "learned_weights_saved": weight_path.exists(),
        "smoke_policy_only": False,
        "uses_eval_gt_for_training": False,
        "uses_eval_gt_for_gate": False,
        "uses_orbslam3_teacher": False,
        "uses_external_router": False,
        "modified_train_edge_fraction": modified_frac,
        "delta_tdir_norm_train_mean": float(np.mean(all_delta_norms)) if all_delta_norms else None,
        "gate_dataset_ready": bool(gate.get("gate_dataset_ready")),
        "classification": "S5E21_REAL_NO_HARM_TRAINING_COMPLETE" if optimizer_steps >= max_steps else "S5E21_SHORT_REAL_NO_HARM_TRAINING",
        "validation_snapshot": validation_from_logs(),
    }
    write_json(out_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gate-dataset", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
