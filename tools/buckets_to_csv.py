#!/usr/bin/env python
"""Convert eval_buckets_latest.json files into flat CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COLUMNS = [
    "experiment", "bucket_type", "bucket_label", "count", "rot", "tdir", "tdir_abs",
    "local_A_abs", "tmag_rel_err", "epi_mass", "top1", "top5", "entropy", "cycle_error",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Warn] skip unreadable json: {path} ({exc})")
        return None


def _metric(rec: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in rec:
            return rec[key]
    return ""


def _rows_for_bucket(experiment: str, bucket_type: str, bucket: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for label in sorted(bucket.keys()):
        rec = bucket[label]
        rows.append({
            "experiment": experiment,
            "bucket_type": bucket_type,
            "bucket_label": label,
            "count": rec.get("count", 0),
            "rot": _metric(rec, "rot"),
            "tdir": _metric(rec, "tdir"),
            "tdir_abs": _metric(rec, "tdir_abs"),
            "local_A_abs": _metric(rec, "tdir_local_A_abs", "local_A_abs"),
            "tmag_rel_err": _metric(rec, "tmag_rel_err"),
            "epi_mass": _metric(rec, "epi_mass_in_gt_band", "epi_mass"),
            "top1": _metric(rec, "top1_in_gt_band", "top1"),
            "top5": _metric(rec, "top5_in_gt_band", "top5"),
            "entropy": _metric(rec, "matching_entropy", "entropy"),
            "cycle_error": _metric(rec, "cycle_error"),
        })
    return rows


def _find_bucket_files(paths: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        if path.is_file() and path.name.endswith(".json"):
            found.append(path)
        elif path.exists():
            found.extend(path.glob("*/eval_buckets_latest.json"))
        else:
            print(f"[Warn] missing path: {path}")
    return sorted(set(found))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["checkpoints"], help="Experiment roots or eval_buckets_latest.json files.")
    parser.add_argument("--out-dir", default=".", help="Output directory for bucket CSV files.")
    args = parser.parse_args()

    all_k: List[Dict[str, Any]] = []
    all_dt: List[Dict[str, Any]] = []
    all_kdt: List[Dict[str, Any]] = []
    for path in _find_bucket_files(Path(p) for p in args.paths):
        data = _load_json(path)
        if not data:
            continue
        exp = path.parent.name
        all_k.extend(_rows_for_bucket(exp, "k", data.get("bucket_k", {})))
        all_dt.extend(_rows_for_bucket(exp, "dt", data.get("bucket_dt", {})))
        all_kdt.extend(_rows_for_bucket(exp, "kdt", data.get("bucket_k_dt", {})))

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "bucket_k.csv", all_k)
    _write_csv(out_dir / "bucket_dt.csv", all_dt)
    _write_csv(out_dir / "bucket_kdt.csv", all_kdt)
    print(f"[OK] wrote {len(all_k)} k rows, {len(all_dt)} dt rows, {len(all_kdt)} kdt rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
