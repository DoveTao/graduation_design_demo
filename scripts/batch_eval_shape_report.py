#!/usr/bin/env python3
"""Batch offline evaluation and shape-aware reporting for checkpoint candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PYTHON_BIN = "/home/dovetao/miniconda3/envs/pytorch/bin/python"
ITEM_META_FILENAME = "batch_shape_item_meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run batched checkpoint eval and shape-aware report pipeline."
    )
    p.add_argument(
        "--item",
        action="append",
        required=True,
        help=(
            "Candidate item, format label=ckpt_path. "
            "Example: --item T51a3b_upd400=checkpoints/.../eval_upd_0400.pt"
        ),
    )
    p.add_argument(
        "--out-root",
        default="checkpoints/BATCH_SHAPE_REPORT_T51",
        help="Directory to store batch artifacts.",
    )
    p.add_argument(
        "--python-bin",
        default=DEFAULT_PYTHON_BIN,
        help="Python binary for helper scripts.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval for candidate if output dir already exists.",
    )
    p.add_argument(
        "--scripts-dir",
        default="scripts",
        help="Directory containing helper scripts.",
    )
    p.add_argument(
        "--selection-only",
        action="store_true",
        help="Skip eval and plot, only read existing repro dirs and run selection+report.",
    )
    return p.parse_args()


def parse_item(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --item format: {raw}. Expected label=ckpt_path")
    label, ckpt = raw.split("=", 1)
    label = label.strip()
    ckpt = ckpt.strip()
    if not label:
        raise ValueError(f"Empty label in item: {raw}")
    if not ckpt:
        raise ValueError(f"Empty ckpt path in item: {raw}")
    return label, ckpt


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def _safe_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def parse_upd_from_text(text: str) -> Optional[int]:
    m = re.search(r"(?:^|[._-])upd(\d+)(?:$|[._-])", text, flags=re.IGNORECASE)
    if not m:
        return None
    return _safe_int(m.group(1))


def parse_step_upd_from_checkpoint(ckpt_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        import torch
    except Exception:
        return None, parse_upd_from_text(ckpt_path.name)

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return None, parse_upd_from_text(ckpt_path.name)

    step = _safe_int(ckpt.get("step")) if isinstance(ckpt, dict) else None
    upd = _safe_int(ckpt.get("upd")) if isinstance(ckpt, dict) else None
    if upd is None:
        upd = parse_upd_from_text(ckpt_path.name)
    return step, upd


def write_item_metadata(
    repro_dir: Path, *, label: str, ckpt: str, step: Optional[int], upd: Optional[int]
) -> None:
    if not repro_dir.exists():
        return
    payload = {
        "label": label,
        "ckpt": str(ckpt),
        "step": step,
        "upd": upd,
    }
    try:
        with open(repro_dir / ITEM_META_FILENAME, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def run_cmd(cmd: List[str], dry_run: bool, description: str) -> int:
    print(f"[CMD] {description}: {' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def load_repro_last_eval(repro_dir: Path) -> Dict[str, float]:
    final_summary = repro_dir / "final_summary.json"
    out: Dict[str, float] = {}
    if not final_summary.exists():
        return out
    try:
        with final_summary.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out

    last_eval = data.get("last_eval", {}) if isinstance(data, dict) else {}
    if not isinstance(last_eval, dict):
        return out

    def _safe_num(v: object) -> Optional[float]:
        try:
            f = float(v)
        except Exception:
            return None
        return f if f == f else None

    out_map = [
        ("drift", "odom_metric_drift"),
        ("length_norm_drift", "odom_metric_length_normalized_drift"),
        ("ate", "odom_metric_ATE"),
        ("tmag_rel_err", "tmag_rel_err"),
        ("tdir_abs", "tdir_abs"),
        ("path_ratio", "odom_shape_metric_mean_path_length_ratio"),
        ("step_dir_err", "odom_shape_metric_mean_step_dir_err_deg"),
        ("mean_tmag_ratio", "odom_debug_mean_tmag_ratio"),
    ]
    for out_key, src_key in out_map:
        val = _safe_num(last_eval.get(src_key))
        if val is not None:
            out[out_key] = val
    return out


def is_eligible_shape(metrics: Dict[str, float]) -> bool:
    drift = metrics.get("drift")
    path_ratio = metrics.get("path_ratio")
    tdir_abs = metrics.get("tdir_abs")
    tmag_rel_err = metrics.get("tmag_rel_err")
    return (
        drift is not None
        and path_ratio is not None
        and tdir_abs is not None
        and tmag_rel_err is not None
        and path_ratio >= 0.35
        and tdir_abs <= 25.0
        and tmag_rel_err <= 1.0
        and drift <= 1.6
    )


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("inf")


def _infer_family(exp_name: str) -> str:
    for p in ("checkpoints/", "TRAJ_REPRO_", "BATCH_"):
        if exp_name.startswith(p):
            exp_name = exp_name[len(p) :]
    if "_upd" in exp_name:
        return exp_name.split("_upd", 1)[0]
    return exp_name


def _format_row_link(row: Dict[str, str]) -> str:
    exp = row.get("exp", "-")
    upd = row.get("upd", "-")
    checkpoint = row.get("checkpoint", "-").strip() or "-"
    if checkpoint == "-":
        return f"`{exp}` upd={upd}"
    return f"`{exp}` upd={upd} (`{Path(checkpoint).name}`)"


def summarize_item(items: List[Dict[str, object]], out_root: Path) -> None:
    md_path = out_root / "batch_shape_report.md"
    lines: List[str] = []
    lines.append("# Batch Shape Report")
    lines.append("")

    lines.append("## Item list")
    lines.append("| label | checkpoint | ckpt step | ckpt upd | repro_dir | status | notes |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for it in items:
        lines.append(
            "| {label} | `{ckpt}` | {step} | {upd} | `{dir}` | {status} | {notes} |".format(
                label=it["label"],
                ckpt=it["ckpt"],
                step=it.get("ckpt_step", "-") if it.get("ckpt_step") is not None else "-",
                upd=it.get("ckpt_upd", "-") if it.get("ckpt_upd") is not None else "-",
                dir=it["repro_dir"],
                status=it["status"],
                notes=it.get("notes", ""),
            )
        )
    lines.append("")

    lines.append("## Selection conclusion")
    rows = read_shape_selection_for_csv(out_root)
    if rows:
        rows_sorted_drift = sorted(rows, key=lambda r: _safe_float(r.get("drift", "")))
        raw_lowest = rows_sorted_drift[0]
        rows_shape = [r for r in rows if r.get("score_shape055_l2", "")]
        shape_sorted = sorted(rows_shape, key=lambda r: _safe_float(r.get("score_shape055_l2", "")))
        shape_best = shape_sorted[0] if shape_sorted else raw_lowest

        family = _infer_family(raw_lowest.get("exp", ""))
        same_family = [r for r in rows if _infer_family(r.get("exp", "")) == family]
        gated_ref = None
        if same_family:
            same_family = [r for r in same_family if r.get("upd", "").strip()]
            same_family = sorted(same_family, key=lambda r: _safe_float(r.get("upd", "")), reverse=True)
            gated_ref = same_family[0] if same_family else None

        lines.append("- raw lowest drift（原始最小 drift）：" + _format_row_link(raw_lowest))
        lines.append("- shape-aware recommended（形状约束推荐）：" + _format_row_link(shape_best))
        if gated_ref is not None:
            lines.append("- original gated best-odom/best-drift 对照：" + _format_row_link(gated_ref))
        else:
            lines.append("- original gated best-odom/best-drift 对照：未从路径标签或 checkpoint label 解析到对照更新点")
        lines.append("")
    else:
        sel_md = out_root / "shape_selection.md"
        if sel_md.exists():
            with sel_md.open("r", encoding="utf-8") as f:
                text = f.read()
            if "## 推荐" in text:
                start = False
                for ln in text.splitlines():
                    if ln.startswith("## 推荐"):
                        start = True
                        lines.append(ln)
                        continue
                    if start:
                        if ln.startswith("## ") and ln != "## 推荐":
                            break
                        lines.append(ln)
                lines.append("")
            else:
                lines.append("- shape_selection.md 未返回推荐段落。")
        else:
            lines.append("- selection 未生成。")
    lines.append("")

    lines.append("## Per-item shape metrics (latest_eval)")
    lines.append("| label | drift | length_norm_drift | ATE | tdir_abs | path_ratio | step_dir_err | shape_eligible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for it in items:
        m = it.get("metrics", {})
        if not isinstance(m, dict):
            m = {}
        metrics = m  # type: ignore[assignment]
        lines.append(
            "| {label} | {drift:.6g} | {length_norm_drift:.6g} | {ate:.6g} | {tdir_abs:.6g} | "
            "{path_ratio:.6g} | {step_dir_err:.6g} | {eligible} |".format(
                label=it["label"],
                drift=metrics.get("drift", float("nan")),
                length_norm_drift=metrics.get("length_norm_drift", float("nan")),
                ate=metrics.get("ate", float("nan")),
                tdir_abs=metrics.get("tdir_abs", float("nan")),
                path_ratio=metrics.get("path_ratio", float("nan")),
                step_dir_err=metrics.get("step_dir_err", float("nan")),
                eligible=bool(it.get("eligible", False)),
            )
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_shape_selection_for_csv(out_root: Path) -> List[Dict[str, str]]:
    csv_path = out_root / "shape_selection.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()

    out_root = Path(args.out_root)
    scripts_dir = Path(args.scripts_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scripts = {
        "eval": str(scripts_dir / "eval_checkpoint_trajectory_repro.py"),
        "select": str(scripts_dir / "select_shape_aware_checkpoint.py"),
        "plot": str(scripts_dir / "plot_odom_trajectory_debug.py"),
    }

    print(f"[info] out-root: {out_root}")
    print(f"[info] python-bin: {args.python_bin}")

    items: List[Dict[str, object]] = []
    for raw in args.item:
        label_raw, ckpt_raw = parse_item(raw)
        label = safe_label(label_raw)
        ckpt_path = Path(ckpt_raw)
        if not ckpt_path.exists():
            print(f"[warn] checkpoint not found: {ckpt_path}")
            return 2

        ckpt_step, ckpt_upd = parse_step_upd_from_checkpoint(ckpt_path)
        if ckpt_upd is None:
            ckpt_upd = parse_upd_from_text(label_raw) or parse_upd_from_text(ckpt_raw)

        exp_name = f"BATCH_{label}"
        repro_dir = Path("checkpoints") / exp_name
        status = "pending"
        note = ""

        if not args.selection_only:
            if args.skip_eval and repro_dir.exists():
                status = "skipped"
                note = "reuse existing"
            else:
                cmd = [
                    args.python_bin,
                    scripts["eval"],
                    "--ckpt",
                    str(ckpt_path),
                    "--exp-name",
                    exp_name,
                    "--python-bin",
                    args.python_bin,
                ]
                if args.dry_run:
                    cmd.append("--dry-run")
                rc = run_cmd(cmd, dry_run=args.dry_run, description=f"eval repro: {label}")
                if rc != 0:
                    return rc
                status = "done"

        else:
            status = "selection-only"

        if not args.dry_run:
            write_item_metadata(
                repro_dir,
                label=label_raw,
                ckpt=str(ckpt_path),
                step=ckpt_step,
                upd=ckpt_upd,
            )

        if not repro_dir.exists():
            status = "missing-repro-dir"
            note = note or "repro dir not found"
        elif args.skip_eval and status == "pending":
            status = "ready"
            note = note or "ready"

        metrics = load_repro_last_eval(repro_dir)
        row: Dict[str, object] = {
            "label": label_raw,
            "ckpt": str(ckpt_path),
            "ckpt_step": ckpt_step,
            "ckpt_upd": ckpt_upd,
            "repro_dir": str(repro_dir),
            "status": status,
            "notes": note,
            "metrics": metrics,
            "eligible": is_eligible_shape(metrics),
        }
        items.append(row)

    if args.dry_run:
        print("[done] dry-run completed")
        for it in items:
            print(f"  label={it['label']} repro_dir={it['repro_dir']} status={it['status']}")
        return 0

    if args.selection_only and args.skip_eval is False:
        print("[warn] --selection-only requested; implied no-eval mode")

    valid_dirs = [it["repro_dir"] for it in items if Path(str(it["repro_dir"])).exists()]
    if not valid_dirs:
        print("[error] no valid repro dirs for selection/plot")
        return 3

    # Selection step writes to out-root by explicit args.
    cmd_select = [
        args.python_bin,
        scripts["select"],
        "--csv",
        str(out_root / "shape_selection.csv"),
        "--md",
        str(out_root / "shape_selection.md"),
    ]
    cmd_select.extend(str(d) for d in valid_dirs)
    rc = run_cmd(cmd_select, dry_run=False, description="shape-aware selection")
    if rc != 0:
        return rc

    cmd_plot = [
        args.python_bin,
        scripts["plot"],
        "--compare",
        "--out",
        str(out_root / "shape_compare.png"),
        "--exp-dir",
    ]
    cmd_plot.extend(str(d) for d in valid_dirs)
    rc = run_cmd(cmd_plot, dry_run=False, description="trajectory compare plot")
    if rc != 0:
        return rc

    summarize_item(items, out_root)

    print(f"[done] wrote {out_root / 'shape_selection.csv'}")
    print(f"[done] wrote {out_root / 'shape_selection.md'}")
    print(f"[done] wrote {out_root / 'shape_compare.png'}")
    print(f"[done] wrote {out_root / 'batch_shape_report.md'}")

    rows = read_shape_selection_for_csv(out_root)
    print(f"[done] selection rows: {len(rows)}")
    if rows:
        print("[sample] top rows:")
        for row in rows[:3]:
            print(f"  {row.get('exp', '-')} upd={row.get('upd', '-')}, drift={row.get('drift', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
