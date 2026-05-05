#!/usr/bin/env python3
"""
Read checkpoint config and run a reproducible eval-only trajectory debug pass.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


FIELD_RESTORE_LIST = [
    "tmag_condition_on_dt",
    "tmag_head_mode",
    "tmag_multiscale_num_bins",
    "tmag_multiscale_log_centers",
    "tmag_multiscale_residual_scale",
    "tmag_multiscale_cls_w",
    "tmag_min",
    "log_tmag_clamp_min",
    "log_tmag_clamp_max",
    "tmag_pred_source",
    "tmag_detach_features",
    "tmag_ridge_head_path",
    "tmag_ridge_head_trainable",
    "tmag_ridge_head_scale",
    "tmag_ridge_calib_init_path",
    "tmag_ridge_calib_gamma_max",
    "tmag_ridge_calib_gamma_init",
    "tmag_ridge_calib_train_gamma",
    "tmag_ridge_calib_train_bias",
    "tmag_ridge_calib_raw_center",
    "tmag_ridge_calib_log_base",
    "tmag_ridge_calib_blend_init",
    "tmag_ridge_calib_train_blend",
    "use_translation_magnitude_head",
    "use_tmag_global_bias",
    "use_tmag_affine_calib",
    "use_seq_turn_loss",
    "seq_turn_loss_w",
    "seq_turn_max_dt",
    "seq_turn_acos_eps",
    "use_seq_turn_chain_loss",
    "use_tdir_anchor_loss",
    "w_tdir_anchor",
    "tdir_anchor_checkpoint",
    "k_choices",
    "k_probs",
    "eval_k_list",
    "min_dt",
    "max_dt",
    "eval_min_dt",
    "eval_max_dt",
    "max_eval_batches",
    "max_train_eval_batches",
    "batch_size",
    "num_workers",
    "persistent_workers",
    "odom_eval_smooth_tmag_window",
    "odom_eval_scale_fit",
    "odom_eval_dtcalib",
]


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (list, tuple)):
        return f"({', '.join(_format_value(x) if not isinstance(x, str) else x for x in v)})"
    if isinstance(v, str):
        return v
    return str(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval-only trajectory debug reproducibly from a checkpoint."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--exp-name", required=True, help="Experiment/output name")
    parser.add_argument(
        "--python-bin",
        default="/home/dovetao/miniconda3/envs/pytorch/bin/python",
        help="Python binary for train_mvp.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    parser.add_argument(
        "--extra-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra train_mvp.py override, can be repeated",
    )
    return parser.parse_args()


def parse_extra_set(raw_items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-set item: {item}, expected key=value")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def load_checkpoint_cfg(ckpt_path: Path) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("cfg", {})
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported cfg type in checkpoint: {type(cfg)}")
    return cfg


def safe_exp_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "eval"


def build_train_command(
    python_bin: str,
    ckpt_path: str,
    exp_name: str,
    cfg: Dict[str, Any],
    extra_set: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    cmd = [
        python_bin,
        "train_mvp.py",
        "--set",
        f"exp_name={exp_name}",
        "--set",
        "eval_only=True",
        "--set",
        f"init_checkpoint={ckpt_path}",
        "--set",
        "strict_load_checkpoint=False",
        "--set",
        "use_odometry_eval=True",
        "--set",
        "save_odom_trajectory_debug=True",
    ]
    for k in FIELD_RESTORE_LIST:
        if k in cfg:
            cmd.extend(["--set", f"{k}={_format_value(cfg[k])}"])
    for k, v in extra_set.items():
        cmd.extend(["--set", f"{k}={v}"])
    return cmd, [f"{k}={cfg[k]!r}" for k in FIELD_RESTORE_LIST if k in cfg]


def format_cmd_for_print(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def check_keywords(log_path: Path) -> Dict[str, bool]:
    keys = [
        "InitCkpt",
        "skipped shape-mismatched",
        "missing",
        "unexpected",
        "OdomCfg",
        "OdomEval",
        "Done",
    ]
    found = {k: False for k in keys}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for k in keys:
        found[k] = k in text
    return found


TMAG_HEAD_WARNING_PATTERNS = [
    "coarse.mag_head.scale_cls_head",
    "coarse.mag_head.log_centers",
    "coarse.mag_head.residual_head",
    "fine.mag_head.scale_cls_head",
    "fine.mag_head.log_centers",
    "fine.mag_head.residual_head",
]


def summarize_load_from_log(log_path: Path) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    init_lines = [ln.strip() for ln in lines if "[InitCkpt]" in ln]
    out: Dict[str, Any] = {
        "init_lines": init_lines,
        "warning_hits": [],
    }
    m = re.search(r"missing=(\d+)\s+\|\s+unexpected=(\d+)", text)
    if m:
        out["missing_count"] = int(m.group(1))
        out["unexpected_count"] = int(m.group(2))
    unexpected_preview = None
    for ln in init_lines:
        if "unexpected preview=" in ln:
            unexpected_preview = ln.split("unexpected preview=", 1)[1].strip()
            break
    out["unexpected_preview"] = unexpected_preview
    for pat in TMAG_HEAD_WARNING_PATTERNS:
        if pat in text:
            out["warning_hits"].append(pat)
    return out


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    extra_set = parse_extra_set(args.extra_set)
    if args.dry_run:
        try:
            cfg = load_checkpoint_cfg(ckpt_path)
        except ModuleNotFoundError:
            cfg = {}
            print("[WARN] torch not available in this runtime; skipping cfg recovery for dry-run.")
        except Exception as exc:
            print(f"[WARN] failed to load checkpoint on dry-run: {exc}")
            cfg = {}
    else:
        cfg = load_checkpoint_cfg(ckpt_path)

    cmd, recovered = build_train_command(
        args.python_bin,
        args.ckpt,
        args.exp_name,
        cfg,
        extra_set,
    )

    print("Recovered cfg fields:")
    for item in recovered:
        print(f"  {item}")
    print("\nRecovered tmag cfg summary:")
    for k in [
        "tmag_head_mode",
        "tmag_multiscale_num_bins",
        "tmag_multiscale_log_centers",
        "tmag_multiscale_residual_scale",
        "tmag_multiscale_cls_w",
    ]:
        if k in cfg:
            print(f"  {k}={cfg[k]!r}")
        else:
            print(f"  {k}=<not in checkpoint cfg>")

    printable = format_cmd_for_print(cmd)
    print("\nFinal command:")
    print(printable)

    log_path = Path(f"/tmp/eval_traj_repro_{safe_exp_name(args.exp_name)}.log")

    if args.dry_run:
        print(f"\n[DRY-RUN] command not executed; log would be: {log_path}")
        return 0

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"\nLog saved to: {log_path}")
    matches = check_keywords(log_path)
    print("\nLog keyword checks:")
    for k, v in matches.items():
        print(f"  {k}: {'found' if v else 'missing'}")

    load_summary = summarize_load_from_log(log_path)
    print("\nLoad summary from train log:")
    if "missing_count" in load_summary and "unexpected_count" in load_summary:
        print(
            f"  load missing/unexpected: "
            f"{load_summary['missing_count']}/{load_summary['unexpected_count']}"
        )
    if load_summary.get("unexpected_preview"):
        print(f"  unexpected preview: {load_summary['unexpected_preview']}")
    for ln in load_summary.get("init_lines", [])[:4]:
        print(f"  {ln}")
    if load_summary.get("warning_hits"):
        print(
            "\n[WARNING] Tmag head parameters were not loaded; "
            "eval result may be a wrapper-default cfg mismatch."
        )
        for pat in load_summary["warning_hits"]:
            print(f"  hit: {pat}")

    if proc.returncode != 0:
        print(f"\nExecution failed with code {proc.returncode}")
        return proc.returncode

    print("\nExecution finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
