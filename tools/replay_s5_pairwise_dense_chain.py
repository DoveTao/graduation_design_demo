#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise-jsonl", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--out-tum", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--initial-pose", default="identity")
    args = ap.parse_args()

    pairwise_jsonl = _resolve(args.pairwise_jsonl)
    out_tum = _resolve(args.out_tum)
    out_json = _resolve(args.out_json)
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    text = pairwise_jsonl.read_text(encoding="utf-8", errors="ignore") if pairwise_jsonl.exists() else ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) == 0:
        out_tum.write_text("", encoding="utf-8")
        meta = {
            "attempted": True,
            "replay_blocked": True,
            "reason": "pairwise vectors unavailable or empty",
            "variant_results": {
                "model_convention_as_declared": {"num_poses": 0, "coverage": 0.0},
                "inverse_relative_variant": {"num_poses": 0, "coverage": 0.0},
                "local_translation_rotated_by_current_pose": {"num_poses": 0, "coverage": 0.0},
                "local_translation_not_rotated": {"num_poses": 0, "coverage": 0.0},
            },
        }
        out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    # Strict no-fabrication mode: parsing/replay requires verified schema, which we do not have in this run.
    out_tum.write_text("", encoding="utf-8")
    meta = {
        "attempted": True,
        "replay_blocked": True,
        "reason": "unverified pairwise schema for final S5 vectors",
        "variant_results": {
            "model_convention_as_declared": {"num_poses": 0, "coverage": 0.0},
            "inverse_relative_variant": {"num_poses": 0, "coverage": 0.0},
            "local_translation_rotated_by_current_pose": {"num_poses": 0, "coverage": 0.0},
            "local_translation_not_rotated": {"num_poses": 0, "coverage": 0.0},
        },
    }
    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
