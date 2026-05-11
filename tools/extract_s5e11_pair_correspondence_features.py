#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from s5e11_correspondence_geometry_lib import RESULT_DIR, load_config, make_feature_records


def run(args: argparse.Namespace) -> None:
    load_config(Path(args.config))
    train_rows, val_rows, _split, _stats = make_feature_records()
    rows = train_rows + val_rows
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        payload.append(
            {
                "scene": row["sample"].frame_i.scene,
                "seq": row["sample"].frame_i.seq,
                "edge_index": row["sample"].edge_index,
                "timestamp_i": row["sample"].frame_i.timestamp,
                "timestamp_j": row["sample"].frame_j.timestamp,
                "observability_bucket": row["observability_bucket"],
                "signed_direction_loss_weight": row["signed_direction_loss_weight"],
                "features": row["features"],
            }
        )
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in payload) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/s5e11_correspondence_parallax_translation_geometry.yaml")
    p.add_argument("--out-jsonl", default=str(RESULT_DIR / "pair_correspondence_features.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
