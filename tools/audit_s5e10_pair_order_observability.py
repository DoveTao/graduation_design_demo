#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from s5e2_adjacent_dense_lib import write_json
from s5e10_order_geometry_lib import architecture_audit


def run(args):
    payload = {
        "experiment": "S5E10_order_aware_signed_direction_scale_recalibration",
        **architecture_audit(),
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
