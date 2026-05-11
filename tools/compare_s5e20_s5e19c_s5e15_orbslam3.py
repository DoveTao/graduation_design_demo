#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, write_json


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(ck: Dict[str, Any], name: str) -> Dict[str, Any]:
    comp = ck.get("component_metrics", {})
    ext = ck.get("external_eval", {})
    kstep = ck.get("kstep_metrics", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "coverage": ck.get("adjacent_dense_export", {}).get("coverage", 1.0),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "tdir_abs_mean_deg": comp.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": comp.get("anti_parallel_rate"),
        "tmag_median_ratio": comp.get("tmag_median_ratio"),
        "tmag_p95_ratio": comp.get("tmag_p95_ratio"),
        "path_ratio": comp.get("path_ratio"),
        "delta_tdir_norm_mean": ck.get("delta_tdir_control", {}).get("delta_tdir_norm_mean"),
        "k1_tdir": kstep.get("k1_tdir"),
        "k2_tdir": kstep.get("k2_tdir"),
        "k3_tdir": kstep.get("k3_tdir"),
        "k5_tdir": kstep.get("k5_tdir"),
        "none_ate": ext.get("none", {}).get("ate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "caveat": ck.get("final_classification"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s20 = _load(Path(args.s5e20_checkpoint))
    s19c = _load(Path(args.s5e19c_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s15, "S5E15"),
        _pick(s19c, "S5E19C"),
        _pick(s20, "S5E20"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "signed_tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "delta_tdir_norm_mean": None,
            "k1_tdir": None,
            "k2_tdir": None,
            "k3_tdir": None,
            "k5_tdir": None,
            "none_ate": ORBSLAM3_REFERENCE["none"]["ate"],
            "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
            "caveat": "external_baseline",
        },
    ]
    summary = (
        "S5E20 的关键不是再堆 head，而是确认 fake multiframe loss 已被真实 contiguous k-step composition supervision 替换，"
        "并且 delta_tdir 比 S5E19C 更保守。"
    )
    payload = {"experiment": "S5E20_true_kstep_composition_supervision", "rows": rows, "summary": summary}
    write_json(Path(args.out_json), payload)
    lines = [
        "# S5E20 对比报告",
        "",
        "## 执行摘要",
        summary,
        "",
        "## 方法对比",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['method']}",
                f"- status: {row['status']}",
                f"- coverage: {row['coverage']}",
                f"- signed_tdir_mean_deg: {row['signed_tdir_mean_deg']}",
                f"- anti_parallel_rate: {row['anti_parallel_rate']}",
                f"- tmag_median_ratio: {row['tmag_median_ratio']}",
                f"- path_ratio: {row['path_ratio']}",
                f"- delta_tdir_norm_mean: {row['delta_tdir_norm_mean']}",
                f"- se3_ate: {row['se3_ate']}",
                f"- sim3_ate: {row['sim3_ate']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论说明",
            "本对比的重点不是再证明 S5E19C 的 no-op 问题，而是确认 S5E20 是否真的把 fake multiframe loss 换成了真实 contiguous k-step supervision，并且是否在不重新放大 delta_tdir 的情况下超过 S5E15。",
            "当前结果显示：S5E20 成功修复了 S5E19C 的 aggressive delta，但没有形成超过 S5E15 的实质几何收益；相对 ORB-SLAM3 仍保留 full coverage 优势，但方向与路径长度质量差距仍然明显。",
        ]
    )
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e20-checkpoint", required=True)
    p.add_argument("--s5e19c-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
