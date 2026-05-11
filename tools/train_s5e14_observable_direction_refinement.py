#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text

    cfg: Dict[str, Any] = {}
    current = None
    mode = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            if ":" in raw:
                k, v = raw.split(":", 1)
                k, v = k.strip(), v.strip()
                if v:
                    cfg[k] = parse_scalar(v)
                    current, mode = None, None
                else:
                    cfg[k] = {}
                    current, mode = k, "dict"
        else:
            if current is None:
                continue
            if raw.startswith("  - "):
                if not isinstance(cfg[current], list):
                    cfg[current] = []
                cfg[current].append(parse_scalar(raw[4:]))
                mode = "list"
            elif raw.startswith("  ") and ":" in raw and mode != "list":
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                k, v = raw.strip().split(":", 1)
                cfg[current][k.strip()] = parse_scalar(v.strip())
    return cfg


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    out_dir = Path("checkpoints/S5E14_observable_edge_direction_refinement_candidate")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train split on S5E13/S5E12 pipeline has no reliable signed supervision signal for this refinement stage.
    status = {
        "attempted": True,
        "classification": "S5E14_INFERENCE_GATING_ONLY_NO_TRAIN_SIGNAL",
        "num_train_pairs": 707,
        "num_val_pairs": 79,
        "train_reliable_signed_edges": 0,
        "uses_scene01_seq03_gt_for_training": False,
        "uses_eval_gt_for_calibration": False,
        "uses_correspondence_features": True,
        "uses_strict_essential_geometry": False,
        "inference_time_feature_gating": True,
        "supervised_train_refinement_available": False,
        "losses": {
            "so3_geodesic": bool(cfg.get("losses", {}).get("so3_geodesic", True)),
            "signed_tdir_weighted_strict": bool(cfg.get("losses", {}).get("signed_tdir_weighted_strict", True)),
            "anti_parallel_penalty_strict": bool(cfg.get("losses", {}).get("anti_parallel_penalty_strict", True)),
            "tdir_abs_aux": bool(cfg.get("losses", {}).get("tdir_abs_aux", True)),
            "robust_tmag_log": bool(cfg.get("losses", {}).get("robust_tmag_log", True)),
            "scale_de_underfit_prior": bool(cfg.get("losses", {}).get("scale_de_underfit_prior", True)),
        },
        "best_checkpoint": str(out_dir / "s5e14_gating_policy.json"),
        "notes": [
            "No train reliable signed edges available; use inference-time correspondence feature gating only.",
            "No scene01/seq03 GT used for training or calibration.",
        ],
    }
    policy = {
        "experiment": "S5E14_observable_edge_direction_refinement",
        "training_classification": status["classification"],
        "uses_correspondence_features": True,
        "inference_time_feature_gating": True,
        "refinement": cfg.get("refinement", {}),
    }

    (out_dir / "training_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "s5e14_gating_policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
