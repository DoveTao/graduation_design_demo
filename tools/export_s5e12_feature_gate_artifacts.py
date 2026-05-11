#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import read_json, write_json


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dep = read_json(Path(args.dependency_json))
    corr = read_json(Path(args.features_json))
    qual = read_json(Path(args.quality_json))
    ess = read_json(Path(args.essential_json))
    obs = read_json(Path(args.observability_json))
    payload = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "dependency_check": dep,
        "correspondence_feature_availability": {
            "edge_count": corr.get("edge_count"),
            "success_count": qual.get("correspondence_success_count"),
            "success_fraction": qual.get("correspondence_success_fraction"),
        },
        "essential_geometry": ess,
        "observability_gate": {
            "observable_edge_fraction": obs.get("observable_edge_fraction"),
            "signed_direction_supervision_reliable_fraction": obs.get("signed_direction_supervision_reliable_fraction"),
            "small_motion_observable_fraction": obs.get("small_motion_observable_fraction"),
        },
        "artifacts": {
            "dependency_json": args.dependency_json,
            "correspondence_json": args.features_json,
            "quality_json": args.quality_json,
            "essential_json": args.essential_json,
            "observability_json": args.observability_json,
        },
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dependency-json", required=True)
    p.add_argument("--features-json", required=True)
    p.add_argument("--quality-json", required=True)
    p.add_argument("--essential-json", required=True)
    p.add_argument("--observability-json", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
