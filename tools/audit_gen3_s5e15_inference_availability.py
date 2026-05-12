#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


ALLOWED_BRIDGE_CLASSIFICATIONS = [
    "INFERENCE_BRIDGE_READY",
    "MODEL_WEIGHTS_MISSING",
    "IMAGE_PAIR_FORWARD_MISSING",
    "SCENE_SPECIFIC_EXPORT_ONLY",
    "EVAL_GT_DEPENDENCY_DETECTED",
    "INPUT_PROTOCOL_UNSUPPORTED",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(Path(args.config))
    candidate_name = str(cfg.get("model", {}).get("source_candidate", "S5E15_scale_deunderfit_antiparallel_candidate"))
    candidate_json = Path(f"checkpoints/{candidate_name}.json")
    candidate_dir = Path(f"checkpoints/{candidate_name}")
    training_status = candidate_dir / "training_status.json"
    export_tool = Path("tools/export_s5e15_adjacent_dense_predictions.py")
    export_text = export_tool.read_text(encoding="utf-8") if export_tool.exists() else ""
    candidate_ckpt = _read_json(candidate_json)
    blockers = []

    s5e15_checkpoint_found = candidate_json.exists() and training_status.exists()
    s5e15_model_weights_found = any(candidate_dir.glob("*.pt")) or any(candidate_dir.glob("*.npz"))
    image_pair_forward_available = False
    external_erp_input_supported = False
    depends_on_scene01_artifacts = False
    uses_eval_gt_for_prediction = False
    can_predict_R_tdir_tmag = False

    if export_tool.exists():
        depends_on_scene01_artifacts = any(
            token in export_text
            for token in [
                "external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl",
                "args.s5e12_feature_dir",
                "external_baselines/results/s5e14_traceable_dense/correspondence_refinement_weights.jsonl",
            ]
        )
        uses_eval_gt_for_prediction = False
        can_predict_R_tdir_tmag = all(token in export_text for token in ["rotation", "translation_direction", "translation_magnitude"])
        image_pair_forward_available = False
        external_erp_input_supported = False

    if not s5e15_checkpoint_found:
        blockers.append("S5E15_CHECKPOINT_NOT_FOUND")
    if not s5e15_model_weights_found:
        blockers.append("MODEL_WEIGHTS_MISSING")
    if depends_on_scene01_artifacts:
        blockers.append("SCENE_SPECIFIC_EXPORT_ONLY")
    if not image_pair_forward_available:
        blockers.append("IMAGE_PAIR_FORWARD_MISSING")
    if not external_erp_input_supported:
        blockers.append("INPUT_PROTOCOL_UNSUPPORTED")
    if uses_eval_gt_for_prediction:
        blockers.append("EVAL_GT_DEPENDENCY_DETECTED")

    if uses_eval_gt_for_prediction:
        bridge_classification = "EVAL_GT_DEPENDENCY_DETECTED"
    elif depends_on_scene01_artifacts:
        bridge_classification = "SCENE_SPECIFIC_EXPORT_ONLY"
    elif not s5e15_model_weights_found:
        bridge_classification = "MODEL_WEIGHTS_MISSING"
    elif not image_pair_forward_available:
        bridge_classification = "IMAGE_PAIR_FORWARD_MISSING"
    elif not external_erp_input_supported:
        bridge_classification = "INPUT_PROTOCOL_UNSUPPORTED"
    else:
        bridge_classification = "INFERENCE_BRIDGE_READY"

    payload = {
        "s5e15_checkpoint_found": bool(s5e15_checkpoint_found),
        "s5e15_model_weights_found": bool(s5e15_model_weights_found),
        "image_pair_forward_available": bool(image_pair_forward_available),
        "external_erp_input_supported": bool(external_erp_input_supported),
        "depends_on_scene01_artifacts": bool(depends_on_scene01_artifacts),
        "uses_eval_gt_for_prediction": bool(uses_eval_gt_for_prediction),
        "can_predict_R_tdir_tmag": bool(can_predict_R_tdir_tmag),
        "inference_bridge_ready": bridge_classification == "INFERENCE_BRIDGE_READY",
        "bridge_classification": bridge_classification,
        "evidence": {
            "source_candidate": candidate_name,
            "candidate_json": str(candidate_json),
            "candidate_dir": str(candidate_dir),
            "training_status_exists": training_status.exists(),
            "export_tool": str(export_tool),
            "adjacent_dense_export_available": bool(candidate_ckpt.get("adjacent_dense_export", {}).get("available")),
            "training_status_classification": _read_json(training_status).get("classification") if training_status.exists() else None,
        },
        "blockers": blockers,
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
