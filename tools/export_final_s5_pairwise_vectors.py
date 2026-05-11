#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _find_candidates() -> List[Path]:
    pats = ["*pairwise*.json", "*pairwise*.jsonl", "*relative*pose*.json", "*component*diagnostics*.json"]
    roots = [REPO_ROOT / "checkpoints", REPO_ROOT / "external_baselines" / "results", REPO_ROOT / "logs"]
    found = []
    for r in roots:
        if not r.exists():
            continue
        for pat in pats:
            for p in r.rglob(pat):
                s = str(p).lower()
                if "s11_" in s or "s16_" in s or "pano_orb_vo" in s:
                    continue
                if "s5" in s and "final" in s and "pairwise" in s:
                    found.append(p)
    return sorted(set(found))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-diagnostics", required=True)
    ap.add_argument("--out-checkpoint", default="checkpoints/S5D6_final_s5_pairwise_vectors_and_replay.json")
    ap.add_argument("--dry-run", default="false")
    args = ap.parse_args()

    out_jsonl = _resolve(args.out_jsonl)
    out_npz = _resolve(args.out_npz)
    out_diag = _resolve(args.out_diagnostics)
    out_ckpt = _resolve(args.out_checkpoint)
    for p in [out_jsonl, out_npz, out_diag, out_ckpt]:
        p.parent.mkdir(parents=True, exist_ok=True)

    found = _find_candidates()
    available = len(found) > 0

    if not available:
        out_jsonl.write_text("", encoding="utf-8")
        np.savez(out_npz, available=np.array([False]), num_pairs=np.array([0]))
        diag = {
            "available": False,
            "num_pairs": 0,
            "pair_selection": "unavailable",
            "translation_frame": "unknown",
            "rotation_convention": "unknown",
            "coverage_vs_adjacent_453": 0.0,
            "gt_used_to_generate_predictions": False,
            "notes": [
                "No explicit final S5 pairwise vector artifact found in current workspace.",
                "S5 official evaluator appears to expose selected_k=1 chain metrics but not exported per-pair final vectors.",
            ],
            "source_audit": {
                "candidate": "S5_clean_tmag_calibration_policy",
                "policy_path": "checkpoints/S5_clean_tmag_calibration_policy.json",
                "manifest_path": "checkpoints/final_clean_candidate_manifest.json",
                "official_evaluator_entry": "tools/s6_final_clean_candidate_lockdown_audit.py",
                "pairwise_prediction_source_found": False,
                "found_candidate_paths": [str(p.relative_to(REPO_ROOT)) for p in found],
            },
        }
    else:
        # strict mode: we still do not parse unknown schema as final vectors automatically
        out_jsonl.write_text("", encoding="utf-8")
        np.savez(out_npz, available=np.array([False]), num_pairs=np.array([0]))
        diag = {
            "available": False,
            "num_pairs": 0,
            "pair_selection": "unavailable",
            "translation_frame": "unknown",
            "rotation_convention": "unknown",
            "coverage_vs_adjacent_453": 0.0,
            "gt_used_to_generate_predictions": False,
            "notes": ["Potential paths found but schema not verified as final S5 pairwise vectors; kept unavailable to avoid fabrication."],
            "source_audit": {
                "candidate": "S5_clean_tmag_calibration_policy",
                "policy_path": "checkpoints/S5_clean_tmag_calibration_policy.json",
                "manifest_path": "checkpoints/final_clean_candidate_manifest.json",
                "official_evaluator_entry": "tools/s6_final_clean_candidate_lockdown_audit.py",
                "pairwise_prediction_source_found": False,
                "found_candidate_paths": [str(p.relative_to(REPO_ROOT)) for p in found],
            },
        }

    out_diag.write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_ckpt.write_text(json.dumps({"pairwise_export": diag}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(diag, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
