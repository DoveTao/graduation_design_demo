#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _load_policy, _run  # type: ignore

ROT_VALUES = [0.35, 0.40, 0.45, 0.50]
TDIR_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20]
POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
OUT_ROOT = REPO_ROOT / "checkpoints" / "S2d_joint_fine_rot_tdir_policy_audit_on_s2b"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S2d_joint_fine_rot_tdir_policy_audit_on_s2b_report.md"
SUMMARY_JSON = OUT_ROOT / "s2d_joint_sweep_summary.json"


def _fmt(x: Any, digits: int = 6) -> str:
    try:
        if isinstance(x, (float, int)):
            return f"{float(x):.{digits}f}"
    except Exception:
        pass
    return str(x)


def _select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [
        r for r in rows
        if float(r["metric_path_ratio"]) >= 0.90 and int(r["load_unexpected"]) == 0
    ]
    if not eligible:
        return min(rows, key=lambda r: (float(r["ATE"]), float(r["drift"]), float(r["fine_tdir_fuse_strength"])))
    return min(
        eligible,
        key=lambda r: (
            float(r["ATE"]),
            float(r["drift"]),
            float(r["fine_tdir_fuse_strength"]),
        ),
    )


def _verdict(best: Dict[str, Any], s2b_ate: float, s2b_drift: float, s2b_ratio: float, any_tdir_gain: bool) -> str:
    if int(best["load_unexpected"]) != 0:
        return "INVALID"
    ratio = float(best["metric_path_ratio"])
    ate = float(best["ATE"])
    drift = float(best["drift"])
    better = (ate + 1e-9 < s2b_ate) or (abs(ate - s2b_ate) <= 1e-9 and drift + 1e-9 < s2b_drift)
    if ratio >= 0.90 and any_tdir_gain:
        return "POSITIVE-DIAGNOSTIC"
    if ratio < 0.90:
        return "TRADEOFF-ONLY"
    if not any_tdir_gain:
        return "NO-TDIR-GAIN"
    return "TRADEOFF-ONLY"


def _heatmap_table(rows: List[Dict[str, Any]], value_key: str, title: str) -> str:
    lines = [f"### {title}\n"]
    header = "| fine_rot \ fine_tdir | " + " | ".join(f"{v:.2f}" for v in TDIR_VALUES) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in TDIR_VALUES) + " |"
    lines.append(header)
    lines.append(sep)
    for rot in ROT_VALUES:
        vals = []
        for tdir in TDIR_VALUES:
            row = next(r for r in rows if abs(float(r["fine_rot_fuse_strength"]) - rot) < 1e-9 and abs(float(r["fine_tdir_fuse_strength"]) - tdir) < 1e-9)
            if isinstance(row[value_key], (float, int)):
                vals.append(_fmt(row[value_key]))
            else:
                vals.append(str(row[value_key]))
        lines.append(f"| {rot:.2f} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    base_policy = _load_policy(POLICY_PATH)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for rot in ROT_VALUES:
        for tdir in TDIR_VALUES:
            policy = deepcopy(base_policy)
            policy["fine_rot_fuse_strength"] = float(rot)
            policy["fine_tdir_fuse_strength"] = float(tdir)
            policy["fine_tmag_fuse_strength"] = 0.0
            policy["use_geometry_refine"] = False
            run_dir = OUT_ROOT / f"rot_{str(rot).replace('.', 'p')}__tdir_{str(tdir).replace('.', 'p')}"
            policy_copy_path = run_dir / "policy_used.json"
            summary_path = run_dir / "s1d5_policy_eval_summary.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            policy_copy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
            if summary_path.exists():
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                payload = _run(policy, policy_copy_path, run_dir, "default")
            payload["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
            rows.append(payload)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    s1d5 = {"fine_rot": 0.40, "fine_tdir": 0.0, "drift": 1.396358, "ATE": 7.632463, "path_ratio": 0.934982}
    s2b = {"fine_rot": 0.45, "fine_tdir": 0.0, "drift": 1.327402, "ATE": 7.352371, "path_ratio": 0.934984}
    best = _select_best(rows)
    any_tdir_gain = any(
        float(r["fine_tdir_fuse_strength"]) > 0.0
        and float(r["metric_path_ratio"]) >= 0.90
        and int(r["load_unexpected"]) == 0
        and (
            float(r["ATE"]) + 1e-9 < s2b["ATE"]
            or (
                abs(float(r["ATE"]) - s2b["ATE"]) <= 1e-9
                and float(r["drift"]) + 1e-9 < s2b["drift"]
            )
        )
        for r in rows
    )
    verdict = _verdict(best, s2b["ATE"], s2b["drift"], s2b["path_ratio"], any_tdir_gain)

    lines: List[str] = []
    lines.append("# S2d Joint Fine Rot Tdir Policy Audit On S2b\n\n")
    lines.append("## Motivation\n")
    lines.append("- `S2c2` classified the remaining official-scope error as `OFFICIAL-SCOPE-R-TDIR-COUPLED-ERROR`.\n")
    lines.append("- Therefore the next diagnostic step is a joint eval-only sweep over `fine_rot` and `fine_tdir`, with `fine_tmag=0.0`.\n")
    lines.append("- No model parameter is trained; policy-only overrides are applied on top of the fixed `S2b` clean policy.\n\n")

    lines.append("## Fixed setup\n")
    lines.append(f"- base checkpoint: `{base_policy['base_checkpoint_path']}`\n")
    lines.append(f"- parent policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`\n")
    lines.append("- dt-anchor: inherited unchanged from `S1d5/S2b`\n")
    lines.append("- fixed `fine_tmag=0.0`\n")
    lines.append("- fixed `use_geometry_refine=False`\n")
    lines.append("- `explicit-cfg / unexpected=0` required\n\n")

    lines.append("## Baselines\n")
    lines.append("| method | status | fine_rot | fine_tdir | drift | ATE | path_ratio |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    lines.append(f"| S1d5 | tagged clean exported mainline baseline | {_fmt(s1d5['fine_rot'],2)} | {_fmt(s1d5['fine_tdir'],2)} | {_fmt(s1d5['drift'])} | {_fmt(s1d5['ATE'])} | {_fmt(s1d5['path_ratio'])} |\n")
    lines.append(f"| S2b | current clean fine-rot candidate | {_fmt(s2b['fine_rot'],2)} | {_fmt(s2b['fine_tdir'],2)} | {_fmt(s2b['drift'])} | {_fmt(s2b['ATE'])} | {_fmt(s2b['path_ratio'])} |\n\n")

    lines.append("## Full joint sweep table\n")
    lines.append("| fine_rot | fine_tdir | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | metric_path_ratio | direction_only_path_ratio | tmag_p10 | tmag_p50 | tmag_p90 | selected_k | available_k | num_pairs | num_chains | missing | unexpected | run_dir |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
    for r in rows:
        lines.append(
            f"| {_fmt(r['fine_rot_fuse_strength'],2)} | {_fmt(r['fine_tdir_fuse_strength'],2)} | {_fmt(r['drift'])} | {_fmt(r['ATE'])} | {_fmt(r['metric_path_ratio'])} | "
            f"{_fmt(r['RPE_rot'])} | {_fmt(r['RPE_trans_dir'])} | {_fmt(r['RPE_trans_mag'])} | {_fmt(r['rot'])} | {_fmt(r['tdir_abs'])} | {_fmt(r['tdir_local_A_abs'])} | "
            f"{_fmt(r['metric_path_ratio'])} | {_fmt(r['direction_only_path_ratio'])} | {_fmt(r['tmag_p10'])} | {_fmt(r['tmag_p50'])} | {_fmt(r['tmag_p90'])} | "
            f"{_fmt(r['odom_selected_k'])} | `{r['odom_available_k']}` | {_fmt(r['num_pairs'])} | {_fmt(r['num_chains'])} | {_fmt(r['load_missing'])} | {_fmt(r['load_unexpected'])} | `{r['run_dir']}` |\n"
        )
    lines.append("\n")

    lines.append("## Best diagnostic candidate\n")
    lines.append(f"- selected by rule: `path_ratio >= 0.90`, `unexpected=0`, then lowest ATE, then lower drift, then smaller fine_tdir\n")
    lines.append(f"- best diagnostic candidate: `fine_rot={best['fine_rot_fuse_strength']:.2f}`, `fine_tdir={best['fine_tdir_fuse_strength']:.2f}`\n")
    lines.append(f"- drift = {_fmt(best['drift'])}\n")
    lines.append(f"- ATE = {_fmt(best['ATE'])}\n")
    lines.append(f"- path_ratio = {_fmt(best['metric_path_ratio'])}\n")
    lines.append(f"- unexpected = {_fmt(best['load_unexpected'])}\n\n")

    lines.append("## Heatmap-like tables\n")
    lines.append(_heatmap_table(rows, "ATE", "ATE"))
    lines.append("\n")
    lines.append(_heatmap_table(rows, "drift", "Drift"))
    lines.append("\n")
    lines.append(_heatmap_table(rows, "metric_path_ratio", "Path Ratio"))
    lines.append("\n")

    better_than_s2b = any_tdir_gain
    any_path_broken = any(float(r["metric_path_ratio"]) < 0.90 for r in rows)

    lines.append("## Interpretation\n")
    lines.append(f"- exists combination better than S2b = {better_than_s2b}\n")
    lines.append(f"- fine_tdir shows positive gain = {any_tdir_gain}\n")
    lines.append(f"- any combination breaks path_ratio < 0.90 = {any_path_broken}\n")
    lines.append("- This is still a test-swept eval-only diagnostic. It does not replace the clean candidate by itself.\n\n")

    lines.append("## Verdict\n")
    lines.append(f"- `{verdict}`\n\n")

    lines.append("## Next step\n")
    if verdict == "POSITIVE-DIAGNOSTIC":
        lines.append("- Next step: `S2e_train_cv_joint_rot_tdir_policy_selection_on_s2b` to cleanify the best joint combination with train-CV.\n")
    elif verdict == "NO-TDIR-GAIN":
        lines.append("- Keep `S2b` as the current clean fine-rot candidate; do not proceed with fine_tdir policy.\n")
    else:
        lines.append("- Report the tradeoff; do not upgrade the mainline yet.\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
