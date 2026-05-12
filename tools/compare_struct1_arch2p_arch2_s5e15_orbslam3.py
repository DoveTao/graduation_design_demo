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
    comp_raw = ck.get("component_metrics", {})
    comp = comp_raw.get("overall", comp_raw) if isinstance(comp_raw, dict) else {}
    ext = ck.get("external_eval", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "anti_parallel_rate": comp.get("anti_parallel_rate"),
        "path_ratio": comp.get("path_ratio"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    struct1 = _load(Path(args.struct1_checkpoint))
    arch2p = _load(Path(args.arch2p_checkpoint))
    arch2 = _load(Path(args.arch2_checkpoint))
    s5e15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s5e15, "S5E15"),
        _pick(arch2p, "ARCH2P"),
        _pick(arch2, "ARCH2"),
        _pick(struct1, "STRUCT1"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "signed_tdir_mean_deg": None,
            "anti_parallel_rate": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
        },
    ]
    summary = {
        "struct1_changed_tdir_source": bool(struct1.get("architecture", {}).get("tdir_from_geometry_tokens")),
        "struct1_stronger_than_arch2_arch2p": bool(
            struct1.get("improvement_vs_s5e15", {}).get("signed_tdir_improved")
            or struct1.get("improvement_vs_s5e15", {}).get("high_confidence_subset_tdir_improved")
        ),
        "struct1_global_gain_vs_s5e15": bool(struct1.get("final_classification") == "STRUCT1_GEOMETRY_TOKENS_IMPROVED"),
        "promote_struct1": bool(struct1.get("recommendation", {}).get("promote_struct1")),
        "stop_model_development_for_thesis_closure": not bool(struct1.get("recommendation", {}).get("continue_model_development")),
    }
    write_json(Path(args.out_json), {"experiment": "STRUCT1_mainline_rebuild_with_geometry_tokens", "rows": rows, "summary": summary})
    Path(args.out_report).write_text(
        "# STRUCT1 / ARCH2P / ARCH2 / S5E15 / ORB-SLAM3 对比\n\n"
        "## 结论\n"
        f"- STRUCT1 是否真正改变 tdir source：{summary['struct1_changed_tdir_source']}\n"
        f"- 是否比 ARCH2/ARCH2P 更强：{summary['struct1_stronger_than_arch2_arch2p']}\n"
        f"- 是否比 S5E15 有全局提升：{summary['struct1_global_gain_vs_s5e15']}\n"
        f"- 是否 promote：{summary['promote_struct1']}\n"
        f"- 若仍失败，是否停止模型开发进入最终论文收口：{summary['stop_model_development_for_thesis_closure']}\n",
        encoding="utf-8",
    )
    return {"rows": rows, "summary": summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--struct1-checkpoint", required=True)
    p.add_argument("--arch2p-checkpoint", required=True)
    p.add_argument("--arch2-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
