#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import scan_frames, write_json


def _parse_cfg(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            return [] if not inner else [parse_scalar(x.strip()) for x in inner.split(",")]
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text.strip('"')

    cfg: Dict[str, Any] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, value = raw.split(":", 1)
            key, value = key.strip(), value.strip()
            if value:
                cfg[key] = parse_scalar(value)
                current = None
            else:
                cfg[key] = {}
                current = key
        elif current is not None and ":" in raw:
            key, value = raw.strip().split(":", 1)
            cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _parse_cfg(Path(args.config))
    frames = scan_frames(Path("data"), scene="scene01")
    k_values = [int(x) for x in cfg["kstep_supervision"]["k_values"]]
    train_scenes = [str(x) for x in cfg["training"]["train_scenes"]]
    eval_scene = str(cfg["training"]["eval_scene"])
    windows: List[Dict[str, Any]] = []
    num_windows_by_k: Dict[str, int] = {str(k): 0 for k in k_values}
    all_contiguous = True
    uses_eval_scene = False

    for scene_seq in train_scenes:
        scene, seq = scene_seq.split("/")
        seq_frames = frames[(scene, seq)]
        for k in k_values:
            for start in range(0, max(0, len(seq_frames) - (k + 1) + 1)):
                chunk = seq_frames[start : start + k + 1]
                valid = len(chunk) == (k + 1) and all(chunk[i + 1].timestamp > chunk[i].timestamp for i in range(len(chunk) - 1))
                all_contiguous = all_contiguous and valid
                uses_eval_scene = uses_eval_scene or (f"{scene}/{seq}" == eval_scene)
                row = {
                    "scene": scene,
                    "seq": seq,
                    "start_index": start,
                    "k": k,
                    "frame_indices": list(range(start, start + k + 1)),
                    "timestamps": [float(x.timestamp) for x in chunk],
                    "valid_contiguous": bool(valid),
                    "uses_eval_scene": False,
                }
                windows.append(row)
                num_windows_by_k[str(k)] += 1

    payload = {
        "experiment": "S5E20_true_kstep_composition_supervision",
        "windows": windows,
        "num_windows_by_k": num_windows_by_k,
        "all_windows_contiguous": all_contiguous,
        "uses_eval_scene": uses_eval_scene,
        "ready_for_training": bool(windows) and all_contiguous and not uses_eval_scene,
        "train_scenes": train_scenes,
        "eval_scene": eval_scene,
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
