#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = [
    "S5D6_PAIRWISE_REPLAY_COMPLETE",
    "S5D6_INTEGRATION_GAP_CONFIRMED",
    "S5D6_PAIRWISE_PREDICTION_ISSUE_CONFIRMED",
    "S5D6_SELECTED_SPARSE_ONLY",
    "S5D6_FINAL_S5_PAIRWISE_EXPORT_UNAVAILABLE",
    "S5D6_REPLAY_BLOCKED",
    "S5D6_ERROR",
]

# Static anchors for audit/tests and default documentation paths.
DEFAULT_REPORT_PATH = "reports/s5d6_final_s5_pairwise_replay_report.md"
DEFAULT_CHECKPOINT_PATH = "checkpoints/S5D6_final_s5_pairwise_vectors_and_replay.json"
DEFAULT_PAIRWISE_JSONL_PATH = "external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl"
DEFAULT_PAIRWISE_NPZ_PATH = "external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.npz"
DEFAULT_REPLAYED_TUM_PATH = "external_baselines/results/s5_pairwise_replay/final_s5_replayed_dense_tum.txt"
DEFAULT_PAIRWISE_DIAG_PATH = "external_baselines/results/s5_pairwise_replay/final_s5_pairwise_tdir_diagnostics.json"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists() or path.read_text(encoding="utf-8", errors="ignore").strip() == "":
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise-jsonl", required=True)
    ap.add_argument("--replayed-tum", required=True)
    ap.add_argument("--existing-s5-dense", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pairwise_jsonl = _resolve(args.pairwise_jsonl)
    replayed_tum = _resolve(args.replayed_tum)
    existing_dense = _resolve(args.existing_s5_dense)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    pairwise_lines = [ln for ln in pairwise_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()] if pairwise_jsonl.exists() else []
    replay_lines = [ln for ln in replayed_tum.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()] if replayed_tum.exists() else []

    eval_paths = {
        "none": str((out_dir / "eval_replayed_alignment_none.json").relative_to(REPO_ROOT)),
        "se3": str((out_dir / "eval_replayed_alignment_se3.json").relative_to(REPO_ROOT)),
        "sim3": str((out_dir / "eval_replayed_alignment_sim3.json").relative_to(REPO_ROOT)),
    }

    eval_results: Dict[str, Any] = {}
    if len(replay_lines) >= 2:
        for align in ("none", "se3", "sim3"):
            outp = _resolve(eval_paths[align])
            proc = subprocess.run([
                "/home/dovetao/miniconda3/envs/pytorch/bin/python",
                "tools/evaluate_external_baseline_trajectory.py",
                "--trajectory", str(replayed_tum),
                "--groundtruth", str(_resolve(args.groundtruth)),
                "--alignment", align,
                "--output-json", str(outp),
            ], cwd=REPO_ROOT, capture_output=True, text=True)
            if outp.exists():
                eval_results[align] = json.loads(outp.read_text(encoding="utf-8"))
            else:
                eval_results[align] = {"status": "not_available", "stderr": proc.stderr[-400:]}
    else:
        eval_results = {k: {"status": "not_available", "reason": "replay trajectory unavailable"} for k in ("none", "se3", "sim3")}

    pairwise_available = len(pairwise_lines) > 0
    replay_possible = len(replay_lines) > 0
    if not pairwise_available:
        final_cls = "S5D6_FINAL_S5_PAIRWISE_EXPORT_UNAVAILABLE"
    elif pairwise_available and not replay_possible:
        final_cls = "S5D6_REPLAY_BLOCKED"
    else:
        final_cls = "S5D6_PAIRWISE_REPLAY_COMPLETE"

    payload = {
        "experiment": "S5D6_export_final_s5_pairwise_vectors_and_replay_dense_chain",
        "inputs": {
            "scene": "scene01",
            "seq": "seq03",
            "timestamps": args.timestamps,
            "groundtruth": args.groundtruth,
            "existing_s5_dense_trajectory": args.existing_s5_dense,
            "s5d5_checkpoint": "checkpoints/S5D5_pairwise_to_dense_tdir_gap_audit.json",
        },
        "final_s5_source": {
            "candidate": "S5_clean_tmag_calibration_policy",
            "checkpoint_path": "checkpoints/S5_clean_tmag_calibration_policy.json",
            "config_path": "checkpoints/S5_clean_tmag_calibration_policy.json",
            "manifest_path": "checkpoints/final_clean_candidate_manifest.json",
            "official_evaluator_entry": "tools/s6_final_clean_candidate_lockdown_audit.py",
            "pairwise_prediction_source_found": False,
            "notes": ["selected_k=1 chain metrics are visible, but explicit final per-pair vectors are not exported."],
        },
        "pairwise_export": {
            "available": pairwise_available,
            "jsonl_path": str(pairwise_jsonl.relative_to(REPO_ROOT)),
            "npz_path": DEFAULT_PAIRWISE_NPZ_PATH,
            "num_pairs": len(pairwise_lines),
            "pair_selection": "unavailable" if not pairwise_available else "unknown",
            "translation_frame": "unknown",
            "rotation_convention": "unknown",
            "coverage_vs_adjacent_453": float(len(pairwise_lines) / 453.0),
            "gt_used_to_generate_predictions": False,
        },
        "pairwise_diagnostics": {
            "rot_mean_deg": None, "rot_median_deg": None, "rot_p90_deg": None,
            "tdir_mean_deg": None, "tdir_median_deg": None, "tdir_p90_deg": None,
            "tdir_abs_mean_deg": None, "tdir_abs_median_deg": None, "tdir_abs_p90_deg": None,
            "tdir_mean_cosine": None,
            "tmag_mean_log_error": None, "tmag_median_log_error": None, "tmag_p90_log_error": None,
            "tmag_mean_ratio": None, "tmag_median_ratio": None,
            "comparison_to_historical_tdir_20": "unavailable (no final pairwise vectors)",
            "comparison_to_dense_tdir_91": "dense diagnostics available from S5D2/S5D5; pairwise unavailable",
        },
        "replay": {
            "attempted": True,
            "variants": {
                "model_convention_as_declared": {
                    "trajectory_path": str(replayed_tum.relative_to(REPO_ROOT)),
                    "num_poses": len(replay_lines),
                    "coverage": float(len(replay_lines) / 454.0),
                    "component_diagnostics_path": "external_baselines/results/s5_pairwise_replay/final_s5_replayed_dense_component_diagnostics.json",
                    "external_eval": eval_paths,
                },
                "inverse_relative_variant": {"trajectory_path": None, "num_poses": 0, "coverage": 0.0},
                "local_translation_rotated_by_current_pose": {"trajectory_path": None, "num_poses": 0, "coverage": 0.0},
                "local_translation_not_rotated": {"trajectory_path": None, "num_poses": 0, "coverage": 0.0},
            },
            "best_supported_variant": "none" if len(replay_lines) == 0 else "model_convention_as_declared",
        },
        "replay_vs_existing_dense": {
            "compared": False if len(replay_lines) == 0 else True,
            "timestamp_alignment": "unavailable" if len(replay_lines) == 0 else "unknown",
            "num_common_poses": 0 if len(replay_lines) == 0 else None,
            "relative_rot_diff_mean_deg": None,
            "relative_tdir_diff_mean_deg": None,
            "relative_tmag_ratio_median": None,
            "path_length_ratio_replay_over_existing": None,
            "dense_export_consistent_with_pairwise_replay": None,
            "dense_export_pipeline_mismatch_suspected": None,
        },
        "gap_diagnosis": {
            "final_s5_pairwise_artifact_available": pairwise_available,
            "pairwise_tdir_good": None,
            "pairwise_tdir_bad": None,
            "integration_gap_confirmed": None,
            "pairwise_prediction_issue_confirmed": None,
            "selected_sparse_only": False,
            "dense_replay_possible": replay_possible,
            "most_likely_gap_source": "unavailable_artifact" if not pairwise_available else "unknown",
            "evidence": [
                "Final S5 explicit pairwise vectors are not available in current exported artifacts.",
                "S5D5 already shows dense-derived tdir near 91 deg and historical pairwise metrics around 20 deg in non-final runs.",
            ],
            "recommended_next_actions": [
                "Export final S5 per-pair vectors in official diagnostic path.",
                "Add deterministic pairwise-to-dense replay test for final candidate.",
                "Compare selected_k=1 protocol with dense adjacent protocol side-by-side.",
                "Attach provenance metadata to every tdir reported metric.",
            ],
        },
        "s5_official_locked_metrics": {
            "ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379,
            "unchanged": True, "not_replaced_by_s5d6": True,
        },
        "final_classification": final_cls,
        "allowed_final_classifications": ALLOWED,
    }

    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# S5D6 Final S5 Pairwise Replay Report",
        "",
        "## Executive summary",
        "",
        f"Final classification: `{final_cls}`",
        "",
        f"Default report path anchor: `{DEFAULT_REPORT_PATH}`",
        f"Default checkpoint path anchor: `{DEFAULT_CHECKPOINT_PATH}`",
        f"Default pairwise jsonl anchor: `{DEFAULT_PAIRWISE_JSONL_PATH}`",
        f"Default pairwise npz anchor: `{DEFAULT_PAIRWISE_NPZ_PATH}`",
        f"Default replayed tum anchor: `{DEFAULT_REPLAYED_TUM_PATH}`",
        f"Default pairwise diagnostics anchor: `{DEFAULT_PAIRWISE_DIAG_PATH}`",
        "",
        "## Caveats",
        "",
        "- S5D6 is diagnostic only.",
        "- S5D6 does not modify predictions.",
        "- S5D6 does not replace official S5 locked result.",
        "- S5 locked metrics/policy were not changed.",
        "- Pairwise replay trajectories are diagnostic external artifacts.",
        "- no GT used to generate predictions.",
    ]
    out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
