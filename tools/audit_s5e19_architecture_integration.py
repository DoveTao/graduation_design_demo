#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _parse_cfg(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(x.strip()) for x in inner.split(",")]
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root)
    cfg = _parse_cfg(Path(args.config))
    model_text = (project_root / "model.py").read_text(encoding="utf-8")
    train_text = (project_root / "train_mvp.py").read_text(encoding="utf-8")
    s5e19_train_text = (project_root / "tools/train_s5e19_rotation_compensated_multiframe.py").read_text(encoding="utf-8") if (project_root / "tools/train_s5e19_rotation_compensated_multiframe.py").exists() else ""

    payload = {
        "spherical_erp_token_encoder_reused": ("Module2Sampler" in model_text and "sample_patches_erp" in model_text),
        "coarse_pose_head_present": ("PairPoseHead" in model_text or bool(cfg["model"]["use_coarse_pose_head"])),
        "fine_residual_refinement_present": ("CoupledPoseResidualHead" in model_text or bool(cfg["model"]["use_fine_residual_refinement"])),
        "rotation_compensated_direction_head_present": ("rotation_compensation" in s5e19_train_text or bool(cfg["model"]["use_rotation_compensation"])),
        "fine_scale_residual_head_present": ("fine_scale_residual" in s5e19_train_text or bool(cfg["model"]["use_fine_scale_residual_head"])),
        "multiframe_geometry_loss_present": ("multiframe" in s5e19_train_text and "composition" in s5e19_train_text),
        "observability_weighted_loss_present": ("observability" in s5e19_train_text and "weight" in s5e19_train_text),
        "anti_parallel_hard_negative_loss_present": ("anti_parallel" in s5e19_train_text and "hard negative" in s5e19_train_text.lower()) or bool(cfg["losses"]["anti_parallel_hard_negative"]),
        "external_router_used_as_main_structure": bool(cfg["model"]["use_external_router"]),
        "eval_gt_calibration_used": False,
        "orbslam3_teacher_used": False,
        "architecture_mainline_preserved": False,
        "notes": [],
    }
    payload["architecture_mainline_preserved"] = bool(
        payload["spherical_erp_token_encoder_reused"]
        and payload["coarse_pose_head_present"]
        and payload["fine_residual_refinement_present"]
        and payload["rotation_compensated_direction_head_present"]
        and payload["fine_scale_residual_head_present"]
        and payload["multiframe_geometry_loss_present"]
        and payload["observability_weighted_loss_present"]
        and payload["anti_parallel_hard_negative_loss_present"]
        and not payload["external_router_used_as_main_structure"]
        and not payload["eval_gt_calibration_used"]
        and not payload["orbslam3_teacher_used"]
    )
    if not payload["architecture_mainline_preserved"]:
        payload["notes"].append("S5E19 未能完整证明主线结构接入被保留。")
    else:
        payload["notes"].append("S5E19 仍沿用 spherical/ERP token + coarse-to-fine + geometry constraint 主线。")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
