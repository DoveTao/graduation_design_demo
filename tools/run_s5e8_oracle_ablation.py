#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import read_json, read_tum, repo_rel, rot_to_quat_xyzw, write_json


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    Rw = np.eye(3)
    tw = np.zeros(3)
    rows = []
    qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
    rows.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for i, (R_BA, t_BA) in enumerate(rels):
        Rb = Rw @ R_BA.T
        tb = tw - Rb @ t_BA
        Rw, tw = Rb, tb
        qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
        rows.append(f"{timestamps[i+1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _ext(traj: Path, gt: Path, out_json: Path, mode: str) -> Dict[str, Any]:
    subprocess.run(
        [
            "/home/dovetao/miniconda3/envs/pytorch/bin/python",
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            str(traj),
            "--groundtruth",
            str(gt),
            "--alignment",
            mode,
            "--output-json",
            str(out_json),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    data = read_json(out_json)
    return {"ate": data.get("ATE"), "drift": data.get("drift"), "path_ratio": data.get("path_ratio")}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    gt = read_tum(Path(args.groundtruth))
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamps = [rows[0]["timestamp_i"]] + [r["timestamp_j"] for r in rows]
    oracle_dir_rels = []
    oracle_scale_rels = []
    for r in rows:
        a, b = float(r["timestamp_i"]), float(r["timestamp_j"])
        gi, gj = gt[a], gt[b]
        R = np.asarray(r["rotation"]["value"], dtype=np.float64)
        gt_t = gj["R"].T @ (gi["t"] - gj["t"])
        gt_dir = gt_t / max(float(np.linalg.norm(gt_t)), 1.0e-12)
        pred_dir = np.asarray(r["pred_dir_unit"], dtype=np.float64)
        raw_tmag = float(r["raw_tmag_from_prior"])
        gt_mag = float(np.linalg.norm(gt_t))
        oracle_dir_rels.append((R, gt_dir * raw_tmag))
        oracle_scale_rels.append((R, pred_dir * gt_mag))
    out_dir = Path(args.out_dir)
    oracle_dir_tum = out_dir / "scene01_seq03_s5e8_oracle_direction_tum.txt"
    oracle_scale_tum = out_dir / "scene01_seq03_s5e8_oracle_scale_tum.txt"
    _write_tum(oracle_dir_tum, timestamps, oracle_dir_rels)
    _write_tum(oracle_scale_tum, timestamps, oracle_scale_rels)
    payload = {
        "oracle_direction": {
            "trajectory": repo_rel(oracle_dir_tum),
            "none": _ext(oracle_dir_tum, Path(args.groundtruth), out_dir / "oracle_direction_none.json", "none"),
            "se3": _ext(oracle_dir_tum, Path(args.groundtruth), out_dir / "oracle_direction_se3.json", "se3"),
            "sim3": _ext(oracle_dir_tum, Path(args.groundtruth), out_dir / "oracle_direction_sim3.json", "sim3"),
        },
        "oracle_scale": {
            "trajectory": repo_rel(oracle_scale_tum),
            "none": _ext(oracle_scale_tum, Path(args.groundtruth), out_dir / "oracle_scale_none.json", "none"),
            "se3": _ext(oracle_scale_tum, Path(args.groundtruth), out_dir / "oracle_scale_se3.json", "se3"),
            "sim3": _ext(oracle_scale_tum, Path(args.groundtruth), out_dir / "oracle_scale_sim3.json", "sim3"),
        },
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provenance", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
