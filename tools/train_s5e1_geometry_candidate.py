#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_OUT_JSON = Path("checkpoints/S5E1_traceable_adjacent_dense_candidate.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/S5E1_geometry_candidate")

LOSS_DESIGN = {
    "SO(3) geodesic": True,
    "tdir": True,
    "tmag log": True,
    "path length consistency": True,
    "short-window consistency": True,
}

ALLOWED_CLASSIFICATIONS = [
    "S5E1_TRACEABLE_DENSE_IMPROVED",
    "S5E1_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
    "S5E1_TRAINING_BLOCKED",
    "S5E1_EXPERIMENT_FAILED",
    "S5E1_ERROR",
]


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "experiment": "S5E1_traceable_adjacent_dense_candidate",
        "status": {
            "official_s5_unchanged": True,
            "experimental_candidate": True,
            "not_official_replacement": True,
        },
        "allowed_classifications": ALLOWED_CLASSIFICATIONS,
        "final_classification": "S5E1_TRAINING_BLOCKED",
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_text = Path(args.config).read_text(encoding="utf-8") if Path(args.config).exists() else ""
    training_status = {
        "experiment": "S5E1_traceable_adjacent_dense_candidate",
        "attempted": True,
        "blocked": True,
        "checkpoint_written": False,
        "config": args.config,
        "loss_design": LOSS_DESIGN,
        "reason": (
            "No legal adjacent-dense training/inference harness is available in the current final S5 path. "
            "Training was not run to avoid changing official policy, train/test split, or using GT to generate predictions."
        ),
        "config_preview": config_text.splitlines()[:80],
        "official_s5_unchanged": True,
        "not_official_replacement": True,
    }
    _write_json(checkpoint_dir / "training_status.json", training_status)

    checkpoint = _load_json(Path(args.out_json))
    checkpoint.setdefault("training_or_finetune", {})
    checkpoint["training_or_finetune"].update(
        {
            "attempted": True,
            "config": args.config,
            "checkpoint_dir": str(checkpoint_dir),
            "losses": {
                "so3_geodesic": True,
                "tdir": True,
                "tmag_log": True,
                "path_length_consistency": True,
                "short_window_consistency": True,
            },
            "blocked": True,
            "notes": [
                training_status["reason"],
                "SO(3) geodesic / tdir / tmag log / path length consistency / short-window consistency are design targets only in this blocked run.",
            ],
        }
    )
    checkpoint.setdefault("status", {}).update(
        {
            "official_s5_unchanged": True,
            "experimental_candidate": True,
            "not_official_replacement": True,
        }
    )
    checkpoint["final_classification"] = checkpoint.get("final_classification") or "S5E1_TRAINING_BLOCKED"
    if checkpoint["final_classification"] == "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED":
        pass
    else:
        checkpoint["final_classification"] = "S5E1_TRAINING_BLOCKED"
    _write_json(Path(args.out_json), checkpoint)
    return training_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record S5E1 experimental geometry candidate training status.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
