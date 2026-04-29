#!/usr/bin/env python
"""Summarize experiment final_summary.json files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRICS = [
    "rot",
    "tdir",
    "tdir_abs",
    "tdir_local_A_abs",
    "tmag_abs_err",
    "tmag_rel_err",
    "trans_vec_l2",
    "stable_rot",
    "stable_tdir_abs",
    "epi_mass",
    "top1",
    "top5",
    "entropy",
    "cycle_error",
    "odom_ATE",
    "odom_drift",
    "geom_rot",
    "geom_tdir_abs",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Warn] skip unreadable json: {path} ({exc})")
        return None


def _as_float(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if not math.isfinite(value):
            return ""
        return value
    return value


def _first_metric(*vals: Any) -> Any:
    for val in vals:
        if val is not None:
            return _as_float(val)
    return ""


def _group_for_exp(exp_name: str) -> str:
    lower = exp_name.lower()
    if lower.startswith("e0") or "baseline" in lower:
        return "E0/Baseline"
    if lower.startswith("c"):
        return "C coarse"
    if lower.startswith("f"):
        return "F fine"
    if lower.startswith("g") or "geom" in lower:
        return "G geometry"
    if lower.startswith("o") or "odom" in lower:
        return "O odometry"
    return "Other"


def _read_matching_diag(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "matching_diag_latest.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not payload:
        return {}
    test = payload.get("test", {})
    return {
        "epi_mass": test.get("epi_mass_in_gt_band"),
        "top1": test.get("top1_in_gt_band"),
        "top5": test.get("top5_in_gt_band"),
        "entropy": test.get("matching_entropy"),
        "cycle_error": test.get("cycle_error"),
    }


def _summary_row(path: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(path)
    if not data:
        return None
    exp_dir = path.parent
    exp_name = exp_dir.name
    last = data.get("last_eval", {}) if isinstance(data.get("last_eval", {}), dict) else {}
    matching = _read_matching_diag(exp_dir)
    row: Dict[str, Any] = {
        "experiment": exp_name,
        "group": _group_for_exp(exp_name),
        "path": str(path),
    }
    row["rot"] = _first_metric(last.get("rot"), data.get("best_rot"))
    row["tdir"] = _first_metric(last.get("tdir"), data.get("best_tdir_raw"))
    row["tdir_abs"] = _first_metric(last.get("tdir_abs"), data.get("best_tdir_abs"))
    row["tdir_local_A_abs"] = _first_metric(last.get("tdir_local_A_abs"), data.get("best_tdir_local_A_abs"))
    row["tmag_abs_err"] = _first_metric(last.get("tmag_abs_err"))
    row["tmag_rel_err"] = _first_metric(last.get("tmag_rel_err"))
    row["trans_vec_l2"] = _first_metric(last.get("trans_vec_l2"))
    row["stable_rot"] = _first_metric(last.get("stable_rot"))
    row["stable_tdir_abs"] = _first_metric(last.get("stable_tdir_abs"))
    row["epi_mass"] = _first_metric(matching.get("epi_mass"), last.get("epi_mass_in_gt_band"))
    row["top1"] = _first_metric(matching.get("top1"), last.get("top1_in_gt_band"))
    row["top5"] = _first_metric(matching.get("top5"), last.get("top5_in_gt_band"))
    row["entropy"] = _first_metric(matching.get("entropy"), last.get("matching_entropy"))
    row["cycle_error"] = _first_metric(matching.get("cycle_error"), last.get("cycle_error"))
    row["odom_ATE"] = _first_metric(last.get("odom_metric_ATE"), last.get("odom_direction_only_ATE"))
    row["odom_drift"] = _first_metric(last.get("odom_metric_drift"), last.get("odom_direction_only_drift"))
    row["geom_rot"] = _first_metric(last.get("geom_rot"))
    row["geom_tdir_abs"] = _first_metric(last.get("geom_tdir_abs"))
    return row


def _find_summaries(roots: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        if root.is_file() and root.name == "final_summary.json":
            paths.append(root)
        elif root.exists():
            paths.extend(root.glob("*/final_summary.json"))
        else:
            print(f"[Warn] missing root: {root}")
    return sorted(set(paths))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "group", *METRICS, "path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _fmt_md(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "group", "rot", "tdir_abs", "tdir_local_A_abs", "tmag_rel_err", "trans_vec_l2", "odom_ATE", "odom_drift", "geom_tdir_abs"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_md(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["checkpoints"], help="Experiment roots or final_summary.json paths.")
    parser.add_argument("--out-dir", default=".", help="Directory for summary_all_experiments.csv and summary_main_table.md.")
    args = parser.parse_args()

    paths = _find_summaries(Path(p) for p in args.roots)
    rows = [row for row in (_summary_row(p) for p in paths) if row is not None]
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "summary_all_experiments.csv", rows)
    _write_md(out_dir / "summary_main_table.md", rows)
    print(f"[OK] summarized {len(rows)} experiments -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
