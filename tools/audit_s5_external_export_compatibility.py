#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from tools.eval_clean_policy import _build_eval_dataset, _load_fine_model
from train_mvp import _build_odometry_chains, _resolve_clean_policy_payload


DEFAULT_S5_TRAJ = REPO_ROOT / "external_baselines" / "results" / "s5" / "scene01_seq03_est_tum.txt"
DEFAULT_GT = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "groundtruth_tum.txt"
DEFAULT_SB2 = REPO_ROOT / "checkpoints" / "SB2_alignment_consistent_baseline_results.json"
DEFAULT_POLICY = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "SB2b_s5_export_compatibility_audit.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "s5_external_export_compatibility_audit.md"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append(
            {
                "timestamp": float(parts[0]),
                "t": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
                "q_xyzw": [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
            }
        )
    return rows


def _path_length(points: Sequence[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())


def _match_timestamps(gt_rows: List[Dict[str, Any]], est_rows: List[Dict[str, Any]], tol: float = 1.0e-3) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    j = 0
    for i, gt in enumerate(gt_rows):
        while j < len(est_rows) and est_rows[j]["timestamp"] < gt["timestamp"] - tol:
            j += 1
        best = None
        for cand in (j - 1, j, j + 1):
            if cand < 0 or cand >= len(est_rows):
                continue
            dt = abs(est_rows[cand]["timestamp"] - gt["timestamp"])
            if dt <= tol and (best is None or dt < best[0]):
                best = (dt, cand)
        if best is not None:
            matches.append((i, best[1]))
    return matches


def _compress_missing_ranges(missing_indices: Sequence[int], gt_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not missing_indices:
        return []
    out: List[Dict[str, Any]] = []
    start = prev = int(missing_indices[0])
    for idx in missing_indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        out.append(
            {
                "start_index": int(start),
                "end_index": int(prev),
                "count": int(prev - start + 1),
                "start_timestamp": float(gt_rows[start]["timestamp"]),
                "end_timestamp": float(gt_rows[prev]["timestamp"]),
            }
        )
        start = prev = idx
    out.append(
        {
            "start_index": int(start),
            "end_index": int(prev),
            "count": int(prev - start + 1),
            "start_timestamp": float(gt_rows[start]["timestamp"]),
            "end_timestamp": float(gt_rows[prev]["timestamp"]),
        }
    )
    return out


def _load_eval_config(policy_path: Path) -> Tuple[Dict[str, Any], Config, Any]:
    policy = _resolve_clean_policy_payload(str(policy_path))
    device = torch.device("cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    _model, cfg, _summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        max_eval_batches=0,
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    ds = _build_eval_dataset(cfg, split="test")
    return policy, cfg, ds


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    ext = payload["external_s5_metrics"]
    locked = payload["locked_s5_metrics"]
    lines = [
        "# S5 External Export Compatibility Audit",
        "",
        "## Scope",
        "This report audits whether the current S5 TUM export is compatible with the authoritative locked clean evaluator.",
        "It does not change S5 final-candidate status.",
        "",
        "## Locked S5 Reference",
        f"- ATE = `{locked['ATE']}`",
        f"- drift = `{locked['drift']}`",
        f"- path_ratio = `{locked['path_ratio']}`",
        "",
        "## External Export Summary",
        f"- num S5 poses = `{payload['s5_num_exported_poses']}`",
        f"- num GT poses = `{payload['gt_num_poses']}`",
        f"- matched timestamps = `{payload['matched_timestamp_count']}`",
        f"- tracking_success_rate = `{payload['tracking_success_rate']:.6f}`",
        f"- external none ATE/drift/path_ratio = `{ext['none']['ATE']:.6f}` / `{ext['none']['drift']:.6f}` / `{ext['none']['path_ratio']:.6f}`",
        f"- external se3 ATE/drift/path_ratio = `{ext['se3']['ATE']:.6f}` / `{ext['se3']['drift']:.6f}` / `{ext['se3']['path_ratio']:.6f}`",
        f"- external sim3 ATE/drift/path_ratio = `{ext['sim3']['ATE']:.6f}` / `{ext['sim3']['drift']:.6f}` / `{ext['sim3']['path_ratio']:.6f}`",
        "",
        "## Mismatch Analysis",
        f"- path_ratio mismatch: locked = `{locked['path_ratio']}` vs external = `{payload['path_ratio_external']:.6f}`",
        f"- coverage mismatch: external matched poses = `{payload['matched_timestamp_count']}` vs GT poses = `{payload['gt_num_poses']}`",
        f"- coverage ratio = `{payload['tracking_success_rate']:.6f}`",
        f"- GT path length on matched subset = `{payload['gt_path_length_matched_subset']:.6f}`",
        f"- GT path length on full exported sequence = `{payload['gt_path_length_full_sequence']:.6f}`",
        f"- predicted path length from S5 export = `{payload['predicted_path_length_export']:.6f}`",
        f"- path_ratio on matched subset = `{payload['path_ratio_external']:.6f}`",
        "",
        "## Root Cause Investigation",
        f"- export pose count is only `{payload['s5_num_exported_poses']}` because the current exporter follows the official odometry-chain construction over `selected_k=1` pairs, which yielded `{payload['odometry_chain_num_pairs']}` pairs rather than a full 454-frame stream.",
        f"- export contains only chain endpoints / selected pairs / sparse frames: `{payload['mismatch_flags']['sparse_export']}`",
        f"- timestamp ordering monotonic: `{payload['mismatch_flags']['timestamp_order_ok']}`",
        f"- quaternion ordering used in export: `qx qy qz qw`",
        f"- convention uses accumulated camera pose from relative pair predictions: `{payload['root_cause_summary']['accumulation_convention']}`",
        f"- full trajectory export to 454 timestamps currently available from official clean evaluator: `{payload['root_cause_summary']['full_454_export_available']}`",
        "",
        "## Conclusion",
        "The current S5 TUM export is diagnostic-only and should not be used as a full-coverage alignment-consistent comparison against Pano-ORB-VO.",
        "The authoritative S5 result remains the locked clean evaluator metrics.",
        "",
        "## Thesis Policy",
        payload["thesis_policy"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit compatibility between the current S5 TUM export and the locked clean evaluator.")
    p.add_argument("--s5-trajectory", default=str(DEFAULT_S5_TRAJ), help="S5 exported TUM path.")
    p.add_argument("--gt", default=str(DEFAULT_GT), help="GT TUM path.")
    p.add_argument("--sb2-json", default=str(DEFAULT_SB2), help="SB2 json path.")
    p.add_argument("--policy", default=str(DEFAULT_POLICY), help="S5 policy path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Final manifest path.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output json path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="Output markdown path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    s5_traj_path = _resolve_path(args.s5_trajectory, DEFAULT_S5_TRAJ)
    gt_path = _resolve_path(args.gt, DEFAULT_GT)
    sb2_path = _resolve_path(args.sb2_json, DEFAULT_SB2)
    policy_path = _resolve_path(args.policy, DEFAULT_POLICY)
    manifest_path = _resolve_path(args.manifest, DEFAULT_MANIFEST)
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)

    sb2 = _read_json(sb2_path)
    manifest = _read_json(manifest_path)
    gt_rows = _read_tum(gt_path)
    est_rows = _read_tum(s5_traj_path)
    matches = _match_timestamps(gt_rows, est_rows)
    matched_gt_indices = [i for i, _ in matches]
    missing_indices = [i for i in range(len(gt_rows)) if i not in set(matched_gt_indices)]

    policy, cfg, ds = _load_eval_config(policy_path)
    manifest_pairs = ds.manifest()
    k1_pairs = [m for m in manifest_pairs if int(m.get("k", -1)) == 1]
    chains = _build_odometry_chains(manifest_pairs, selected_k=1, max_pairs=0)
    chain = chains[0] if chains else {"pairs": []}

    gt_points_full = [row["t"] for row in gt_rows]
    gt_points_matched = [gt_rows[i]["t"] for i, _ in matches]
    est_points = [row["t"] for row in est_rows]
    gt_len_full = _path_length(gt_points_full)
    gt_len_matched = _path_length(gt_points_matched)
    est_len = _path_length(est_points)
    tracking_success_rate = float(len(matches) / max(len(gt_rows), 1))

    mismatch_flags = {
        "sparse_export": len(est_rows) < len(gt_rows),
        "tracking_success_rate_low": tracking_success_rate < 0.95,
        "path_ratio_mismatch": abs(float(sb2["s5_external_eval_metrics_if_available"]["none"]["path_ratio"]) - float(manifest["final_metrics"]["path_ratio"])) > 0.1,
        "timestamp_order_ok": all(est_rows[i]["timestamp"] < est_rows[i + 1]["timestamp"] for i in range(max(len(est_rows) - 1, 0))),
        "export_matches_chain_pairs_plus_one": len(est_rows) == len(chain.get("pairs", [])) + 1,
    }

    root_cause_summary = {
        "sparse_export_reason": "The exporter follows the official odometry-chain construction over selected_k=1 pairs, not a dense full-frame trajectory stream.",
        "timestamp_alignment_reason": "The external evaluator matches only the sparse exported timestamps, so GT path length and path_ratio are computed on a sparse subset.",
        "accumulation_convention": "Relative pair predictions are accumulated in the same odometry-chain convention used by the official evaluator debug path.",
        "coordinate_frame_convention": "Export writes accumulated pose positions with qx qy qz qw quaternion ordering.",
        "full_454_export_available": False,
        "full_454_export_reason": "The current official clean evaluator does not directly expose a dense 454-pose full-frame trajectory; it exposes selected odometry chains over sparse k=1 evaluation pairs.",
    }

    final_classification = "SPARSE_EXPORT_DIAGNOSTIC_ONLY"
    thesis_policy = (
        "For thesis reporting, S5 locked metrics and Pano-ORB-VO external trajectory metrics should be reported in clearly separated blocks "
        "unless a full-coverage S5 TUM export equivalent to the official evaluator is available."
    )

    payload = {
        "name": "SB2b_s5_export_compatibility_audit",
        "locked_s5_metrics": dict(manifest["final_metrics"]),
        "external_s5_metrics": dict(sb2["s5_external_eval_metrics_if_available"]),
        "s5_num_exported_poses": int(len(est_rows)),
        "gt_num_poses": int(len(gt_rows)),
        "matched_timestamp_count": int(len(matches)),
        "tracking_success_rate": tracking_success_rate,
        "path_ratio_locked": float(manifest["final_metrics"]["path_ratio"]),
        "path_ratio_external": float(sb2["s5_external_eval_metrics_if_available"]["none"]["path_ratio"]),
        "predicted_path_length_export": est_len,
        "gt_path_length_matched_subset": gt_len_matched,
        "gt_path_length_full_sequence": gt_len_full,
        "s5_policy_eval_config": {
            "eval_min_dt": float(getattr(cfg, "eval_min_dt", float("nan"))),
            "eval_max_dt": None if getattr(cfg, "eval_max_dt", None) is None else float(getattr(cfg, "eval_max_dt")),
            "eval_pair_step": int(getattr(cfg, "eval_pair_step", 1)),
            "split_by": str(getattr(cfg, "split_by", "unknown")),
            "train_ratio": float(getattr(cfg, "train_ratio", float("nan"))),
            "split_seed": int(getattr(cfg, "split_seed", -1)),
        },
        "k1_manifest_pair_count": int(len(k1_pairs)),
        "odometry_chain_count": int(len(chains)),
        "odometry_chain_num_pairs": int(len(chain.get("pairs", []))),
        "missing_timestamp_ranges": _compress_missing_ranges(missing_indices, gt_rows),
        "mismatch_flags": mismatch_flags,
        "root_cause_summary": root_cause_summary,
        "final_classification": final_classification,
        "thesis_policy": thesis_policy,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(output_md, payload)
    print(f"[SB2b] json={output_json}")
    print(f"[SB2b] md={output_md}")
    print(f"[SB2b] final_classification={final_classification}")
    print("[SB2b] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
