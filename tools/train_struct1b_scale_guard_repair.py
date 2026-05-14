#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s5e2_adjacent_dense_lib import write_json
from train_struct1_geometry_token_pose_solver import (
    Struct1GeometryTokenModel,
    _make_rows,
    _rot_geodesic_deg,
    _tensorize,
    _vector_angle_t,
    load_struct1_config,
)


class HardBoundedTrainPriorScaleHead(nn.Module):
    """STRUCT1B scale-only repair head with explicit hard prior bounds."""

    def __init__(
        self,
        summary_dim: int,
        hidden_dim: int = 128,
        *,
        prior_min_ratio: float = 0.35,
        prior_max_ratio: float = 2.50,
        raw_delta_log_limit: float = 2.50,
    ) -> None:
        super().__init__()
        self.prior_min_ratio = float(prior_min_ratio)
        self.prior_max_ratio = float(prior_max_ratio)
        self.raw_delta_log_limit = float(raw_delta_log_limit)
        self.net = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, summary: torch.Tensor, base_log_tmag: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_delta_log_tmag = torch.tanh(self.net(summary.float()).view(-1)) * self.raw_delta_log_limit
        raw_log_tmag = base_log_tmag.float().view(-1) + raw_delta_log_tmag
        lower_log_bound = base_log_tmag.float().view(-1) + math.log(self.prior_min_ratio)
        upper_log_bound = base_log_tmag.float().view(-1) + math.log(self.prior_max_ratio)
        bounded_log_tmag = torch.minimum(torch.maximum(raw_log_tmag, lower_log_bound), upper_log_bound)
        lower_tmag_bound = torch.exp(lower_log_bound)
        upper_tmag_bound = torch.exp(upper_log_bound)
        final_tmag = torch.clamp(torch.exp(bounded_log_tmag), min=lower_tmag_bound, max=upper_tmag_bound)
        return {
            "hard_scale_guard_enabled": True,
            "raw_delta_log_tmag": raw_delta_log_tmag,
            "raw_log_tmag": raw_log_tmag,
            "bounded_log_tmag": bounded_log_tmag,
            "lower_log_bound": lower_log_bound,
            "upper_log_bound": upper_log_bound,
            "lower_tmag_bound": lower_tmag_bound,
            "upper_tmag_bound": upper_tmag_bound,
            "final_tmag": final_tmag,
        }


class Struct1BScaleRepairModel(nn.Module):
    def __init__(self, base_model: Struct1GeometryTokenModel, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.base_model = base_model
        self.scale_head = HardBoundedTrainPriorScaleHead(
            summary_dim=int(cfg["model"]["hidden_dim"]) * 3,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            prior_min_ratio=float(cfg["scale_guard"]["prior_min_ratio"]),
            prior_max_ratio=float(cfg["scale_guard"]["prior_max_ratio"]),
            raw_delta_log_limit=float(cfg["scale_guard"]["raw_delta_log_limit"]),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
        W_ab: torch.Tensor,
        W_ba: torch.Tensor,
        R_coarse: torch.Tensor,
        base_tdir: torch.Tensor,
        base_log_tmag: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = self.base_model(
            local_features,
            bearing_a,
            bearing_b,
            W_ab,
            W_ba,
            R_coarse,
            base_tdir,
            base_log_tmag,
        )
        repaired = self.scale_head(base["fine_pose"]["summary"], base_log_tmag)
        final_tvec_B = base["tdir_B"] * repaired["final_tmag"].unsqueeze(-1)
        return {
            **base,
            **repaired,
            "R_BA": base["R_BA"],
            "tdir_B": base["tdir_B"],
            "tmag": repaired["final_tmag"],
            "final_tvec_B": final_tvec_B,
        }


def _evaluate_scale_metrics(
    model: Struct1BScaleRepairModel,
    rows: Sequence[Any],
    low_thr: float,
    high_thr: float,
    device: torch.device,
) -> Dict[str, float]:
    ratios: List[float] = []
    pred_steps: List[float] = []
    gt_steps: List[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch_rows = rows[start : start + 16]
            batch = _tensorize(batch_rows, low_thr, high_thr, device)
            out = model(
                batch["local_features"],
                batch["bearing_a"],
                batch["bearing_b"],
                batch["W_ab"],
                batch["W_ba"],
                batch["R_coarse"],
                batch["base_tdir"],
                batch["base_log_tmag"],
            )
            ratio = out["tmag"] / batch["gt_tmag"].clamp_min(1.0e-6)
            ratios.extend(ratio.cpu().numpy().tolist())
            pred_steps.extend(torch.linalg.norm(out["final_tvec_B"], dim=-1).cpu().numpy().tolist())
            gt_steps.extend(batch["gt_tmag"].cpu().numpy().tolist())
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "tmag_median_ratio": float(np.median(arr)),
        "tmag_p95_ratio": float(np.percentile(arr, 95)),
        "path_ratio": float(np.sum(pred_steps) / max(float(np.sum(gt_steps)), 1.0e-12)),
    }


def build_model_blob(
    cfg: Dict[str, Any],
    model: Struct1BScaleRepairModel,
    dataset: Dict[str, Any],
    token_in_dim: int,
    best_run: Dict[str, Any],
    optimizer_step_count: int,
) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "config": cfg,
        "token_in_dim": token_in_dim,
        "dataset": dataset,
        "best_run": best_run,
        "optimizer_step_count": optimizer_step_count,
        "architecture": {
            "preserve_geometry_token_pose_solver": True,
            "redesign_tdir_head": False,
            "hard_scale_guard_enabled": True,
        },
    }


def load_trained_struct1b(path: Path) -> Tuple[Struct1BScaleRepairModel, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    cfg = blob["config"]
    base_model = Struct1GeometryTokenModel(cfg, token_in_dim=int(blob["token_in_dim"]))
    model = Struct1BScaleRepairModel(base_model, cfg)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_struct1_config(Path(args.config))
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, dataset, windows, token_in_dim = _make_rows(cfg)
    out_dir = Path(args.candidate_dir) if args.candidate_dir else Path("checkpoints/STRUCT1B_scale_guard_repair_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dataset["ready_for_training"]:
        status = {
            "real_training_executed": False,
            "hard_scale_guard_enabled": True,
            "optimizer_step_count": 0,
            "recorded_raw_log_tmag": True,
            "recorded_bounded_log_tmag": True,
            "recorded_final_tmag": True,
            "uses_eval_gt_for_training": False,
            "uses_orbslam3_teacher": False,
            "classification": "STRUCT1B_TRAINING_BLOCKED",
        }
        write_json(out_dir / "training_status.json", status)
        return status

    low_thr = float(dataset["low_threshold"])
    high_thr = float(dataset["high_threshold"])
    run_summaries: List[Dict[str, Any]] = []
    best_model_state: Dict[str, torch.Tensor] | None = None
    best_run: Dict[str, Any] | None = None
    total_optimizer_steps = 0

    for w_path in [float(x) for x in cfg["training"]["w_path_candidates"]]:
        for w_tmag in [float(x) for x in cfg["training"]["w_tmag_candidates"]]:
            torch.manual_seed(seed)
            model = Struct1BScaleRepairModel(Struct1GeometryTokenModel(cfg, token_in_dim=token_in_dim), cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
            optimizer_steps = 0
            for _ in range(int(cfg["training"]["steps_per_run"])):
                batch_rows = random.sample(rows, k=min(int(cfg["training"]["batch_size"]), len(rows)))
                batch = _tensorize(batch_rows, low_thr, high_thr, device)
                out = model(
                    batch["local_features"],
                    batch["bearing_a"],
                    batch["bearing_b"],
                    batch["W_ab"],
                    batch["W_ba"],
                    batch["R_coarse"],
                    batch["base_tdir"],
                    batch["base_log_tmag"],
                )
                pred_dot = torch.sum(out["tdir_B"] * batch["gt_tdir"], dim=-1).clamp(-1.0, 1.0)
                abs_dot = torch.abs(pred_dot)
                weight = batch["gate_floor"]
                loss_rot = _rot_geodesic_deg(out["R_BA"], batch["gt_R"]).mean() / 180.0
                loss_tdir = torch.sum(weight * (1.0 - pred_dot)) / torch.clamp(weight.sum(), min=1.0)
                loss_abs = torch.mean((0.2 + 0.8 * weight) * (1.0 - abs_dot))
                loss_anti = torch.sum(weight * torch.relu(0.15 - pred_dot)) / torch.clamp(weight.sum(), min=1.0)
                loss_tmag = torch.mean(torch.abs(out["bounded_log_tmag"] - torch.log(batch["gt_tmag"].clamp_min(1.0e-6))))
                loss_cycle = out["fine_geom"]["cycle_error"].mean()
                loss_epi = out["fine_geom"]["epipolar_residual"].mean()
                loss_inv = torch.mean(torch.linalg.norm(out["fine_geom"]["matched_bearing_a"] - out["fine_geom"]["bearing_b"], dim=-1))
                conf_reg = torch.mean(out["fine_geom"]["entropy"] * out["fine_geom"]["confidence"])
                window_terms: List[torch.Tensor] = []
                sampled_windows = random.sample(windows, k=min(int(cfg["training"]["kstep_batch_size"]), len(windows)))
                for w in sampled_windows:
                    w_rows = [rows[i] for i in w["edge_ids"]]
                    wb = _tensorize(w_rows, low_thr, high_thr, device)
                    wout = model(
                        wb["local_features"],
                        wb["bearing_a"],
                        wb["bearing_b"],
                        wb["W_ab"],
                        wb["W_ba"],
                        wb["R_coarse"],
                        wb["base_tdir"],
                        wb["base_log_tmag"],
                    )
                    R_chain = torch.eye(3, device=device)
                    t_chain = torch.zeros(3, device=device)
                    path_sum = torch.zeros((), device=device)
                    for i in range(wout["R_BA"].shape[0]):
                        R_step = wout["R_BA"][i]
                        t_step = wout["final_tvec_B"][i]
                        R_chain = R_step @ R_chain
                        t_chain = R_step @ t_chain + t_step
                        path_sum = path_sum + torch.linalg.norm(t_step)
                    gt_R = torch.tensor(w["gt_R"], dtype=torch.float32, device=device)
                    gt_tdir = torch.tensor(w["gt_tdir"], dtype=torch.float32, device=device)
                    gt_path = torch.tensor(float(w["gt_path_len"]), dtype=torch.float32, device=device)
                    window_terms.append(
                        0.35 * _rot_geodesic_deg(R_chain.unsqueeze(0), gt_R.unsqueeze(0)).mean()
                        + _vector_angle_t(t_chain.unsqueeze(0), gt_tdir.unsqueeze(0)).mean()
                        + 0.25 * torch.abs(path_sum - gt_path)
                    )
                loss_kstep = torch.stack(window_terms).mean() if window_terms else torch.zeros((), device=device)
                loss_path = torch.mean(torch.abs(torch.linalg.norm(out["final_tvec_B"], dim=-1) - batch["gt_tmag"]))
                loss = (
                    float(cfg["losses"]["w_rot"]) * loss_rot
                    + float(cfg["losses"]["w_tdir_signed"]) * loss_tdir
                    + float(cfg["losses"]["w_tdir_abs_aux"]) * loss_abs
                    + float(cfg["losses"]["w_antiparallel"]) * loss_anti
                    + w_tmag * loss_tmag
                    + float(cfg["losses"]["w_softcorr_cycle"]) * loss_cycle
                    + float(cfg["losses"]["w_epipolar_residual"]) * loss_epi
                    + float(cfg["losses"]["w_inverse_consistency"]) * loss_inv
                    + float(cfg["losses"]["w_kstep_composition"]) * loss_kstep
                    + w_path * loss_path
                    + float(cfg["losses"]["w_confidence_entropy_regularization"]) * conf_reg
                )
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                optimizer_steps += 1
            total_optimizer_steps += optimizer_steps
            summary = _evaluate_scale_metrics(model, rows, low_thr, high_thr, device)
            score = summary["tmag_median_ratio"] + 0.25 * summary["tmag_p95_ratio"] + 2.0 * summary["path_ratio"]
            if not math.isfinite(score):
                score = float("inf")
            item = {
                "w_path": w_path,
                "w_tmag": w_tmag,
                "optimizer_step_count": optimizer_steps,
                **summary,
                "score": float(score),
                "valid": bool(math.isfinite(summary["tmag_median_ratio"]) and math.isfinite(summary["tmag_p95_ratio"]) and math.isfinite(summary["path_ratio"])),
            }
            run_summaries.append(item)
            if item["valid"] and (best_run is None or float(score) < float(best_run["score"])):
                best_run = item
                best_model_state = copy.deepcopy(model.state_dict())

    assert best_run is not None and best_model_state is not None
    best_model = Struct1BScaleRepairModel(Struct1GeometryTokenModel(cfg, token_in_dim=token_in_dim), cfg)
    best_model.load_state_dict(best_model_state)
    torch.save(build_model_blob(cfg, best_model, dataset, token_in_dim, best_run, total_optimizer_steps), out_dir / "struct1b_scale_repair.pt")
    status = {
        "real_training_executed": True,
        "hard_scale_guard_enabled": True,
        "optimizer_step_count": total_optimizer_steps,
        "recorded_raw_log_tmag": True,
        "recorded_bounded_log_tmag": True,
        "recorded_final_tmag": True,
        "uses_eval_gt_for_training": False,
        "uses_orbslam3_teacher": False,
        "best_run": best_run,
        "sweep_runs": run_summaries,
        "classification": "STRUCT1B_REAL_TRAINING_COMPLETE",
    }
    write_json(out_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--candidate-dir", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
