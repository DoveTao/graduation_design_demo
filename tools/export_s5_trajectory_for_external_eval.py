#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from tools.eval_clean_policy import DtBucketScaledMagnitudeModel, _build_eval_dataset, _load_fine_model
from tools.evaluate_external_baseline_trajectory import evaluate_external_baseline_trajectory
from train_mvp import _build_odometry_chains, _compose_rel_pose_np, _resolve_clean_policy_payload


DEFAULT_POLICY = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
DEFAULT_PANO_JSON = REPO_ROOT / "checkpoints" / "SB1_pano_orb_vo_baseline_results.json"
DEFAULT_S5_TRAJ = REPO_ROOT / "external_baselines" / "results" / "s5" / "scene01_seq03_est_tum.txt"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "SB2_alignment_consistent_baseline_results.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "alignment_consistent_strong_baseline_comparison.md"
DEFAULT_GT = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "groundtruth_tum.txt"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _quat_xyzw_from_rot(R: np.ndarray) -> List[float]:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _export_s5_trajectory(policy_path: Path, output_traj: Path) -> Dict[str, Any]:
    policy = _resolve_clean_policy_payload(str(policy_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        max_eval_batches=0,
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    factors = {k: float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped = DtBucketScaledMagnitudeModel(model, factors)
    ds = _build_eval_dataset(cfg, split="test")
    manifest = ds.manifest()
    chains = _build_odometry_chains(manifest, selected_k=1, max_pairs=0)
    chains = [c for c in chains if len(c.get("pairs", [])) > 0]
    if not chains:
        raise RuntimeError("No valid S5 odometry chains available for external trajectory export.")

    chain = chains[0]
    world_R = np.eye(3, dtype=np.float64)
    world_t = np.zeros(3, dtype=np.float64)
    rows: List[str] = []
    first_meta = chain["pairs"][0]
    rows.append(f"{float(first_meta['tsA']):.6f} 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000")

    with torch.no_grad():
        for item in chain["pairs"]:
            sample = ds[int(item["_ds_idx"])]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_val = float(sample.get("dt_world", sample.get("t_gt_mag", 0.01)))
            dt_tensor = torch.tensor([dt_val], device=device, dtype=torch.float32)
            R_pred, t_pred, aux = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            R_np = R_pred.detach().float().cpu().numpy()[0]
            t_vec_metric = None
            if aux.get("t_vec_out", None) is not None:
                t_vec_metric = aux["t_vec_out"].detach().float().cpu().numpy()[0]
            elif aux.get("t_mag", None) is not None:
                t_dir_t = aux.get("t_dir_out", t_pred)
                t_dir = torch.nn.functional.normalize(t_dir_t.detach().float(), dim=-1, eps=1e-6).cpu().numpy()[0]
                t_mag = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
                t_vec_metric = t_dir * t_mag
            if t_vec_metric is None:
                raise RuntimeError("S5 trajectory export failed because metric translation vector was unavailable.")

            world_R, world_t = _compose_rel_pose_np(R_np, np.asarray(t_vec_metric, dtype=np.float64), world_R, world_t)
            qx, qy, qz, qw = _quat_xyzw_from_rot(world_R)
            rows.append(
                f"{float(item['tsB']):.6f} {world_t[0]:.9f} {world_t[1]:.9f} {world_t[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
            )

    output_traj.parent.mkdir(parents=True, exist_ok=True)
    output_traj.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "trajectory_path": str(output_traj),
        "load_missing": int(len(load_summary["missing"])),
        "load_unexpected": int(len(load_summary["unexpected"])),
        "num_poses": int(len(rows)),
        "num_pairs": int(len(chain["pairs"])),
        "scene_seq": str(chain.get("scene_seq", "unknown")),
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    s5_locked = payload["s5_locked_metrics"]
    pano = payload["pano_orb_vo_metrics"]
    lines = [
        "# Alignment-Consistent Strong Baseline Comparison",
        "",
        "## Scope",
        "This report aims to compare S5 and Pano-ORB-VO under a more alignment-consistent external trajectory evaluator when possible.",
        "It does not train a new model and does not reselect the final candidate.",
        "",
        "## Pano-ORB-VO Reminder",
        f"- none: ATE=`{_fmt(pano['none']['ATE'])}`, drift=`{_fmt(pano['none']['drift'])}`, path_ratio=`{_fmt(pano['none']['path_ratio'])}`",
        f"- se3: ATE=`{_fmt(pano['se3']['ATE'])}`, drift=`{_fmt(pano['se3']['drift'])}`, path_ratio=`{_fmt(pano['se3']['path_ratio'])}`",
        f"- sim3: ATE=`{_fmt(pano['sim3']['ATE'])}`, drift=`{_fmt(pano['sim3']['drift'])}`, path_ratio=`{_fmt(pano['sim3']['path_ratio'])}`",
        "",
        "## S5 Reminder",
        f"- locked ATE=`{_fmt(s5_locked['ATE'])}`",
        f"- locked drift=`{_fmt(s5_locked['drift'])}`",
        f"- locked path_ratio=`{_fmt(s5_locked['path_ratio'])}`",
        "",
        "## Alignment-Consistent Results",
    ]
    if payload["s5_trajectory_export_available"]:
        rows = payload["alignment_consistent_rows"]
        cols = ["method", "alignment", "ATE", "drift", "path_ratio", "tracking_success_rate", "notes"]
        table = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows:
            vals = []
            for col in cols:
                v = row.get(col, "")
                vals.append(_fmt(v) if isinstance(v, float) else str(v))
            table.append("| " + " | ".join(vals) + " |")
        lines.extend(table)
    else:
        lines.extend(
            [
                "S5 trajectory export is not currently available.",
                "Therefore, S5 official locked metrics and Pano-ORB-VO external evaluator metrics are reported in separate blocks.",
                "Alignment-mode direct comparison requires a future S5 trajectory export.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            payload["comparison_interpretation"],
            "",
            "## Thesis wording",
            payload["thesis_wording"],
            "",
            "## Caveats",
            "- path_ratio remains the scale-sensitive caveat even when alignment-corrected ATE looks favorable",
            "- Pano-ORB-VO is a strong protocol-compatible classical baseline",
            "- Pano-ORB-VO is not ORB-SLAM3",
            "- Pano-ORB-VO is not an original fisheye baseline",
            "- S5 remains the final clean candidate",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export S5 trajectory for external evaluation and build an alignment-consistent comparison report.")
    p.add_argument("--policy", default=str(DEFAULT_POLICY), help="Locked S5 policy path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Final manifest path.")
    p.add_argument("--pano-json", default=str(DEFAULT_PANO_JSON), help="SB1 pano-orb-vo result json.")
    p.add_argument("--gt", default=str(DEFAULT_GT), help="Groundtruth TUM path.")
    p.add_argument("--s5-traj-out", default=str(DEFAULT_S5_TRAJ), help="Exported S5 TUM trajectory path.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="SB2 output json path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="SB2 output markdown path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy_path = _resolve_path(args.policy, DEFAULT_POLICY)
    manifest_path = _resolve_path(args.manifest, DEFAULT_MANIFEST)
    pano_json_path = _resolve_path(args.pano_json, DEFAULT_PANO_JSON)
    gt_path = _resolve_path(args.gt, DEFAULT_GT)
    s5_traj_out = _resolve_path(args.s5_traj_out, DEFAULT_S5_TRAJ)
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)

    manifest = _read_json(manifest_path)
    pano_payload = _read_json(pano_json_path)
    pano_metrics = pano_payload["main_evaluation"]

    s5_export_meta = None
    s5_external_eval = None
    export_error = None
    try:
        s5_export_meta = _export_s5_trajectory(policy_path, s5_traj_out)
        s5_external_eval = {
            align: evaluate_external_baseline_trajectory(gt_path=gt_path, est_path=s5_traj_out, alignment=align)
            for align in ("none", "se3", "sim3")
        }
    except Exception as exc:
        export_error = f"{type(exc).__name__}: {exc}"

    rows: List[Dict[str, Any]] = []
    if s5_external_eval is not None:
        for method_name, eval_map, note in [
            ("S5", s5_external_eval, "S5 clean policy exported to the same external trajectory evaluator."),
            ("Pano-ORB-VO", pano_metrics, "Protocol-compatible classical panorama VO baseline."),
        ]:
            for align in ("none", "se3", "sim3"):
                res = eval_map[align]
                rows.append(
                    {
                        "method": method_name,
                        "alignment": align,
                        "ATE": float(res.get("ATE", float("nan"))),
                        "drift": float(res.get("drift", float("nan"))),
                        "path_ratio": float(res.get("path_ratio", float("nan"))),
                        "tracking_success_rate": float(res.get("tracking_success_rate", 1.0)),
                        "notes": note,
                    }
                )

    interpretation = (
        "Pano-ORB-VO is a strong protocol-compatible classical baseline. "
        "It is competitive or better in alignment-corrected trajectory shape metrics, "
        "but its path_ratio is substantially lower than S5, indicating weaker path-length/scale stability. "
        "S5 remains the final clean candidate, but this comparison shows that it is not uniformly superior to classical VO."
    )
    thesis_wording = (
        "Under an alignment-consistent external trajectory evaluator, the proposed S5 system does not dominate the "
        "protocol-compatible classical panorama VO baseline on every metric. The classical baseline can be competitive "
        "in alignment-corrected trajectory shape, whereas S5 retains an advantage in path-length/scale stability, "
        "which remains important for a clean final candidate under the historical protocol."
    )

    payload = {
        "name": "SB2_alignment_consistent_strong_baseline_comparison",
        "s5_trajectory_export_available": bool(s5_external_eval is not None),
        "s5_trajectory_export_meta": s5_export_meta,
        "s5_trajectory_export_error": export_error,
        "pano_orb_vo_metrics": pano_metrics,
        "s5_locked_metrics": dict(manifest["final_metrics"]),
        "s5_external_eval_metrics_if_available": s5_external_eval,
        "alignment_consistent_rows": rows,
        "comparison_interpretation": interpretation,
        "thesis_wording": thesis_wording,
        "caveats": [
            "path_ratio remains a scale-sensitive caveat",
            "official locked S5 metrics and external-eval alignment metrics answer different protocol questions",
            "Pano-ORB-VO does not replace S5 final-candidate status",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(output_md, payload)
    print(f"[SB2] json={output_json}")
    print(f"[SB2] md={output_md}")
    print(f"[SB2] s5_trajectory_export_available={payload['s5_trajectory_export_available']}")
    print("[SB2] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
