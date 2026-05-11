#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E4 = {
    "rot": 0.9134395040767528,
    "tdir": 51.47429479273258,
    "tdir_abs": 46.07968707914154,
    "anti_parallel_rate": 0.1368653421633554,
    "tmag": 12.109093390318423,
    "path_ratio": 2.1345479454214416,
    "sim3": 4.0211631491566155,
}


def _ext(traj: str, gt: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for mode in ["none", "se3", "sim3"]:
        p = out_dir / f"eval_alignment_{mode}.json"
        subprocess.run(
            ["/home/dovetao/miniconda3/envs/pytorch/bin/python", "tools/evaluate_external_baseline_trajectory.py", "--trajectory", traj, "--groundtruth", gt, "--alignment", mode, "--output-json", str(p)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        r = read_json(p)
        out[mode] = {"ate": r.get("ATE"), "drift": r.get("drift"), "path_ratio": r.get("path_ratio"), "status": r.get("status"), "num_matched_poses": r.get("num_matched_poses"), "tracking_success_rate": r.get("tracking_success_rate")}
    return out


def _worst(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            m = row.get("metric_preview") or {}
            vals.append({"edge_index": row["edge_index"], "timestamp_i": row["timestamp_i"], "timestamp_j": row["timestamp_j"], **m})
    return {
        "by_tdir": sorted(vals, key=lambda x: x.get("tdir_deg") or -1, reverse=True)[:20],
        "by_tmag_ratio": sorted(vals, key=lambda x: x.get("tmag_ratio") or -1, reverse=True)[:20],
        "by_rot": sorted(vals, key=lambda x: x.get("rot_deg") or -1, reverse=True)[:20],
    }


def _report(path: Path, ckpt: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E5 temporal visual backbone report",
        "",
        "## 执行摘要",
        f"S5E5 引入真正的 ordered image pair temporal visual backbone，并与 S5E4/S5E3 numeric prior 融合。`final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## S5E4 failure recap",
        "S5E4 修复了 signed tdir 的反向问题，但 `tdir_abs` 回退到 S5E2 水平，`tmag` 与 `path_ratio` 也没有继续改善，说明仅靠 signed prior 还不够。",
        "",
        "## temporal visual backbone 设计",
        "S5E5 使用 ordered image pair CNN，输入 `Ii, Ij, Ij-Ii, abs(Ij-Ii)`，并与 S5E4/S5E3 numeric prior 融合。",
        "",
        "## 为什么需要 ordered image pair，而不是 symmetric statistics",
        "symmetric statistics 会抹掉时间方向；ordered image pair 保留 `i->j` 的方向性，有助于同时优化 signed tdir 和 tdir_abs。",
        "",
        "## rot / signed tdir / tdir_abs / tmag gating 说明",
        "S5E5 不以 ATE 单独判定成功；rot、signed tdir、tdir_abs、anti_parallel、tmag、path_ratio 都必须同时报告。",
        "",
        "## training 结果",
        f"{ckpt.get('training')}",
        "",
        "## traceable dense export coverage",
        f"{ckpt.get('adjacent_dense_export')}",
        "",
        "## component metrics，重点讨论 rot 和 tdir",
        f"{metrics}",
        "",
        "## external evaluator none/se3/sim3",
        f"{ckpt.get('external_eval')}",
        "",
        "## 与 S5E4 / S5E3 / S5E2 比较",
        f"{ckpt.get('improvement_vs_s5e4')}",
        "",
        "## 与 ORB-SLAM3 比较",
        f"{ckpt.get('comparison_to_orbslam3')}",
        "",
        "## 是否更接近 ORB-SLAM3",
        "如果 signed tdir、tdir_abs、tmag 和 path_ratio 同时改善，才算真正更接近 ORB-SLAM3；否则只是局部修补。",
        "",
        "## rot / tdir / tmag 哪个仍是主要差距",
        "rot 已较稳定；tdir 和 tmag/path_ratio 通常仍是决定 trajectory 误差的主差距。",
        "",
        "## 下一步建议",
        "下一步可以把 temporal visual backbone 做得更深，或进入 S5E6 的 ORB-aware distillation，但那需要新的实验边界。",
        "",
        "## caveats",
        "- S5E5 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ext = _ext(args.trajectory, args.groundtruth, out_dir) if metrics.get("available") else {"none": {}, "se3": {}, "sim3": {}}
    write_json(out_dir / "worst_edges.json", _worst(Path(args.provenance)))
    ckpt["external_eval"] = ext
    ckpt["component_metrics"].update({k: metrics.get(k) for k in ckpt["component_metrics"] if k in metrics})
    imp = {
        "rot_preserved": (metrics.get("rot_mean_deg") or 999) <= S5E4["rot"] + 1.0,
        "signed_tdir_improved": (metrics.get("tdir_mean_deg") or 999) < S5E4["tdir"],
        "tdir_abs_improved": (metrics.get("tdir_abs_mean_deg") or 999) < S5E4["tdir_abs"],
        "anti_parallel_rate_reduced_or_preserved": (metrics.get("anti_parallel_rate") or 999) <= S5E4["anti_parallel_rate"] + 1e-6,
        "tmag_improved": (metrics.get("tmag_median_ratio") or 999) < S5E4["tmag"],
        "path_ratio_improved": (metrics.get("path_ratio") or 999) < S5E4["path_ratio"],
        "sim3_ate_improved": (ext.get("sim3", {}).get("ate") or 999) < S5E4["sim3"],
    }
    imp["overall_geometry_improved"] = bool(imp["rot_preserved"] and imp["signed_tdir_improved"] and imp["tdir_abs_improved"] and (imp["tmag_improved"] or imp["path_ratio_improved"]))
    ckpt["improvement_vs_s5e4"] = imp
    se3 = ext.get("se3", {}).get("ate")
    ckpt["comparison_to_orbslam3"] = {
        "coverage_advantage": True,
        "rot_close_to_orbslam3": (metrics.get("rot_mean_deg") or 999) <= 2.0,
        "tdir_gap_remaining": (metrics.get("tdir_mean_deg") or 999) >= 35.0,
        "tmag_gap_remaining": (metrics.get("tmag_median_ratio") or 999) >= 5.0,
        "aligned_ate_gap_to_orbslam3": None if se3 is None else se3 - ORBSLAM3_REFERENCE["se3"]["ate"],
        "summary": "S5E5 tests whether temporal visual features improve direction and magnitude jointly while keeping full coverage.",
    }
    if imp["overall_geometry_improved"]:
        final = "S5E5_GEOMETRY_IMPROVED"
    elif imp["signed_tdir_improved"]:
        final = "S5E5_TDIR_IMPROVED_TMAG_STILL_BAD"
    elif imp["tmag_improved"]:
        final = "S5E5_TMAG_IMPROVED_TDIR_STILL_BAD"
    else:
        final = "S5E5_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt, metrics)
    return ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
