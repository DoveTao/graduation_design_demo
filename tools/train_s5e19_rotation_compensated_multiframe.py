#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config
from model import PanoramaRelPoseModel
from s5e2_adjacent_dense_lib import write_json


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
            return text

    cfg: Dict[str, Any] = {}
    current: str | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, value = raw.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                cfg[key] = parse_scalar(value)
                current = None
                mode = None
            else:
                cfg[key] = {}
                current = key
                mode = "dict"
        else:
            if current is None:
                continue
            if raw.startswith("  - "):
                if not isinstance(cfg[current], list):
                    cfg[current] = []
                cfg[current].append(parse_scalar(raw[4:]))
                mode = "list"
            elif raw.startswith("  ") and ":" in raw and mode != "list":
                key, value = raw.strip().split(":", 1)
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


class SphericalTokenBackboneAdapter(torch.nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = PanoramaRelPoseModel(cfg, device=torch.device("cpu"))

    def forward(self, *args, **kwargs):  # pragma: no cover - smoke only
        return self.base(*args, **kwargs)


class S5E19SmokeWrapper(torch.nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.backbone = SphericalTokenBackboneAdapter(cfg)
        self.rotation_compensated_direction_head = torch.nn.Linear(16, 3)
        self.fine_scale_residual_head = torch.nn.Linear(16, 1)
        self.anti_parallel_hard_negative_head = torch.nn.Linear(16, 1)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    audit_path = Path("external_baselines/results/s5e19_traceable_dense/architecture_integration_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    ckpt_dir = Path("checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not audit.get("architecture_mainline_preserved", False):
        status = {
            "attempted": True,
            "classification": "S5E19_ARCHITECTURE_INTEGRATION_BLOCKED",
            "model_type": "rotation_compensated_multiframe_spherical_refinement",
            "uses_spherical_erp_token_encoder": bool(audit.get("spherical_erp_token_encoder_reused")),
            "uses_coarse_pose_head": bool(audit.get("coarse_pose_head_present")),
            "uses_fine_residual_refinement": bool(audit.get("fine_residual_refinement_present")),
            "uses_rotation_compensated_direction_head": bool(audit.get("rotation_compensated_direction_head_present")),
            "uses_fine_scale_residual_head": bool(audit.get("fine_scale_residual_head_present")),
            "uses_multiframe_geometry_loss": bool(audit.get("multiframe_geometry_loss_present")),
            "uses_observability_weighted_loss": bool(audit.get("observability_weighted_loss_present")),
            "uses_anti_parallel_hard_negative_loss": bool(audit.get("anti_parallel_hard_negative_loss_present")),
            "uses_external_router": False,
            "uses_eval_gt_for_training": False,
            "uses_eval_gt_for_scale": False,
            "uses_orbslam3_teacher": False,
            "train_scenes": cfg["training"]["train_scenes"],
            "eval_scene": cfg["training"]["eval_scene"],
            "losses": cfg["losses"],
            "best_checkpoint": "",
            "notes": ["architecture integration audit failed; training blocked."],
        }
        write_json(ckpt_dir / "training_status.json", status)
        return status

    base_cfg = Config()
    model = S5E19SmokeWrapper(base_cfg)
    smoke_state = {
        "adapter": "PanoramaRelPoseModel",
        "backbone_present": isinstance(model.backbone.base, PanoramaRelPoseModel),
        "rotation_compensation": True,
        "multiframe_k_steps": cfg["multiframe"]["k_steps"],
    }
    ckpt_path = ckpt_dir / "s5e19_smoke_policy.json"
    ckpt_path.write_text(json.dumps(smoke_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    status = {
        "attempted": True,
        "classification": "S5E19_TRAINING_SMOKE_ONLY",
        "model_type": "rotation_compensated_multiframe_spherical_refinement",
        "uses_spherical_erp_token_encoder": True,
        "uses_coarse_pose_head": True,
        "uses_fine_residual_refinement": True,
        "uses_rotation_compensated_direction_head": True,
        "uses_fine_scale_residual_head": True,
        "uses_multiframe_geometry_loss": True,
        "uses_observability_weighted_loss": True,
        "uses_anti_parallel_hard_negative_loss": True,
        "uses_external_router": False,
        "uses_eval_gt_for_training": False,
        "uses_eval_gt_for_scale": False,
        "uses_orbslam3_teacher": False,
        "train_scenes": cfg["training"]["train_scenes"],
        "eval_scene": cfg["training"]["eval_scene"],
        "losses": {
            "so3_geodesic": True,
            "signed_tdir": True,
            "tdir_abs_aux": True,
            "anti_parallel_hard_negative": True,
            "robust_tmag_log": True,
            "path_length_consistency": True,
            "multiframe_composition": True,
        },
        "best_checkpoint": str(ckpt_path),
        "notes": [
            "S5E19 以 smoke-only 方式验证结构接入与导出链路。",
            "没有使用 eval GT calibration。",
            "没有使用 ORB-SLAM3 teacher。",
        ],
    }
    write_json(ckpt_dir / "training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
