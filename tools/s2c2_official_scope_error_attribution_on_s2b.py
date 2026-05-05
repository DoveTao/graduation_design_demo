#!/usr/bin/env python3
"""Official-scope error attribution on top of S2b policy.

This script intentionally aligns its primary attribution scope with the
official odometry report configuration used by S2b:
`odom_shape_metric_mean_path_length_ratio` derived from `debug_chain_summaries`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.s2c1_frame_convention_and_chain_integration_audit import (  # type: ignore
    POLICY_PATH,
    REPRO_DIR,
    _corr,
    _evaluate_variant,
    _fmt,
    _load_records,
    _safe_float,
)


REPORT_PATH = REPO_ROOT / "checkpoints/S2c2_official_scope_error_attribution_on_s2b_report.md"


def _pick_classification(oracle: Dict[str, Dict[str, Any]]) -> str:
    base = _safe_float(oracle["official_current"]["ATE_all_chain"], float("inf"))
    ate_R = _safe_float(oracle["oracle_R"]["ATE_all_chain"], float("inf"))
    ate_tdir = _safe_float(oracle["oracle_tdir"]["ATE_all_chain"], float("inf"))
    ate_tmag = _safe_float(oracle["oracle_tmag"]["ATE_all_chain"], float("inf"))
    ate_R_tdir = _safe_float(oracle["oracle_R_tdir"]["ATE_all_chain"], float("inf"))

    if ate_R_tdir + 2.0 < min(ate_R, ate_tdir):
        return "OFFICIAL-SCOPE-R-TDIR-COUPLED-ERROR"
    if ate_tdir + 0.5 < min(base, ate_R, ate_tmag):
        return "OFFICIAL-SCOPE-TDIR-DOMINANT"
    if ate_R + 0.5 < min(base, ate_tdir, ate_tmag):
        return "OFFICIAL-SCOPE-ROTATION-DOMINANT"
    if ate_tmag + 0.5 < min(base, ate_R, ate_tdir):
        return "OFFICIAL-SCOPE-SCALE-RESIDUAL"
    return "OFFICIAL-SCOPE-NO-CLEAR-SINGLE-CAUSE"


def _next_step(classification: str) -> str:
    if classification == "OFFICIAL-SCOPE-R-TDIR-COUPLED-ERROR":
        return (
            "进入 `S2d_joint_fine_rot_tdir_policy_audit_on_s2b`，联合扫 "
            "`fine_rot = 0.35, 0.40, 0.45, 0.50` 与 "
            "`fine_tdir = 0.00, 0.05, 0.10, 0.15, 0.20`，保持 `fine_tmag = 0.0`。"
        )
    if classification == "OFFICIAL-SCOPE-TDIR-DOMINANT":
        return "进入 `S2d_fine_tdir_eval_only_sweep_on_s2b`。"
    if classification == "OFFICIAL-SCOPE-ROTATION-DOMINANT":
        return "继续更细 `fine_rot` sweep 或做 rotation consistency audit。"
    if classification == "OFFICIAL-SCOPE-SCALE-RESIDUAL":
        return "回查 `S1d5/S2b` dt-anchor 与 residual scale behavior。"
    return "没有清晰单因子主导；若继续，优先做 `S2d joint fine_rot + fine_tdir` 小扫而不是单扫 fine_tdir。"


def main() -> None:
    chains, meta = _load_records()
    debug_json = json.loads((REPRO_DIR / "odom_trajectory_debug_latest.json").read_text(encoding="utf-8"))
    official_summary = json.loads((REPRO_DIR / "s1d5_policy_eval_summary.json").read_text(encoding="utf-8"))

    num_debug_chains = int(debug_json.get("num_debug_chains", 0))
    official_chain_ids = list(range(num_debug_chains))
    debug_chain = (debug_json.get("chains") or [{}])[0]
    official_scene_seq = debug_chain.get("scene_seq", "unknown")

    variant_names = [
        "official_current",
        "oracle_R",
        "oracle_tdir",
        "oracle_R_tdir",
        "oracle_tmag",
        "oracle_all",
    ]
    official_scope: Dict[str, Dict[str, Any]] = {}
    all_chain_secondary: Dict[str, Dict[str, Any]] = {}
    for name in variant_names:
        dbg_res = _evaluate_variant(chains, name, chain_subset=official_chain_ids)
        all_res = _evaluate_variant(chains, name)
        dbg_res["ATE_all_chain"] = all_res["ATE"]
        dbg_res["drift_all_chain"] = all_res["drift"]
        official_scope[name] = dbg_res
        all_chain_secondary[name] = all_res

    baseline = official_scope["official_current"]
    chain_rows = baseline["chain_results"]
    step_rows = baseline["step_rows"]
    chain = chain_rows[0] if chain_rows else None

    corrs = {
        "corr(ATE, tdir error)": _corr([r.ATE for r in chain_rows], [r.mean_tdir_error for r in chain_rows]),
        "corr(ATE, rot error)": _corr([r.ATE for r in chain_rows], [r.mean_rot_error for r in chain_rows]),
        "corr(ATE, path_ratio error)": _corr([r.ATE for r in chain_rows], [abs(r.path_ratio - 1.0) for r in chain_rows]),
        "corr(ATE, turn error)": _corr([r.ATE for r in chain_rows], [r.mean_turn_error for r in chain_rows]),
    }

    top_cum = sorted(step_rows, key=lambda r: _safe_float(r.get("metric_pos_err"), float("-inf")), reverse=True)[:10]
    top_step = sorted(step_rows, key=lambda r: _safe_float(r.get("metric_pos_err"), float("-inf")), reverse=True)[:10]
    top_tdir = sorted(step_rows, key=lambda r: _safe_float(r.get("tdir_err_deg"), float("-inf")), reverse=True)[:10]
    top_rot = sorted(step_rows, key=lambda r: _safe_float(r.get("rot_err_deg"), float("-inf")), reverse=True)[:10]

    classification = _pick_classification(official_scope)
    next_step = _next_step(classification)

    lines: List[str] = []
    lines.append("# S2c2 Official-Scope Error Attribution On S2b")
    lines.append("")
    lines.append("## S2b baseline")
    lines.append(f"- base checkpoint: `{meta['policy']['base_checkpoint_path']}`")
    lines.append(f"- policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`")
    lines.append("- explicit-cfg / unexpected=0: True")
    lines.append(f"- fine_rot / fine_tdir / fine_tmag = {meta['policy']['fine_rot_fuse_strength']} / {meta['policy']['fine_tdir_fuse_strength']} / {meta['policy']['fine_tmag_fuse_strength']}")
    lines.append(f"- selected_k / num_pairs / num_chains = 1 / {meta['num_pairs']} / {meta['num_chains']}")
    lines.append(f"- official report drift / ATE / path_ratio = {_fmt(official_summary.get('drift'))} / {_fmt(official_summary.get('ATE'))} / {_fmt(official_summary.get('metric_path_ratio'))}")
    lines.append("")
    lines.append("## Official scope definition")
    lines.append("- official metric field: `odom_shape_metric_mean_path_length_ratio`")
    lines.append("- official aggregation source: `debug_chain_summaries`")
    lines.append(f"- current eval config serialized `num_debug_chains = {num_debug_chains}`")
    lines.append(f"- primary official debug chain scene_seq: `{official_scene_seq}`")
    lines.append("- all-chain numbers below are secondary diagnostics only and must not be compared directly to the official path_ratio.")
    lines.append("")
    lines.append("## Official-scope oracle diagnostics")
    lines.append("- `drift / ATE` below follow the official all-chain odom payload.")
    lines.append("- `path_ratio` below follows the official debug-chain-scoped `debug_chain_summaries` payload.")
    lines.append("- step / turn / tdir / tmag breakdown below is computed on the official debug chain only.")
    lines.append("| variant | drift_all_chain | ATE_all_chain | path_ratio_debug_chain | RPE_rot_all_chain | RPE_trans_dir_all_chain | RPE_trans_mag_all_chain | mean_step_position_error_debug | mean_rot_error_debug | mean_tdir_error_debug | mean_tmag_ratio_debug | turn_error_debug | num_steps_debug | num_pairs_all_chain |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in variant_names:
        res = official_scope[name]
        crow = res["chain_results"][0] if res["chain_results"] else None
        lines.append(
            f"| {name} | {_fmt(res['drift_all_chain'])} | {_fmt(res['ATE_all_chain'])} | {_fmt(res['path_ratio'])} | {_fmt(all_chain_secondary[name]['RPE_rot'])} | "
            f"{_fmt(all_chain_secondary[name]['RPE_trans_dir'])} | {_fmt(all_chain_secondary[name]['RPE_trans_mag'])} | {_fmt(res['mean_step_position_error'])} | "
            f"{_fmt(crow.mean_rot_error if crow else float('nan'))} | {_fmt(crow.mean_tdir_error if crow else float('nan'))} | "
            f"{_fmt(crow.mean_tmag_ratio if crow else float('nan'))} | {_fmt(crow.mean_turn_error if crow else float('nan'))} | "
            f"{crow.num_steps if crow else 0} | {all_chain_secondary[name]['num_pairs']} |"
        )
    lines.append("")
    lines.append("## Official debug-chain step breakdown")
    lines.append("| step_idx | frame_i | frame_j | gt_tmag | pred_tmag | tmag_ratio | rot_error | tdir_error | step_position_error | cumulative_position_error | turn_error |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cumulative_max = 0.0
    for row in step_rows:
        cumulative_max = max(cumulative_max, _safe_float(row.get("metric_pos_err"), 0.0))
        lines.append(
            f"| {int(row.get('step_idx', -1))} | {int(row.get('i', -1))} | {int(row.get('j', -1))} | "
            f"{_fmt(row.get('dt_gt'))} | {_fmt(_safe_float(row.get('tmag_ratio')) * _safe_float(row.get('dt_gt')))} | "
            f"{_fmt(row.get('tmag_ratio'))} | {_fmt(row.get('rot_err_deg'))} | {_fmt(row.get('tdir_err_deg'))} | "
            f"{_fmt(row.get('metric_pos_err'))} | {_fmt(cumulative_max)} | {_fmt(float('nan'))} |"
        )
    lines.append("")
    lines.append("## Top bad steps by cumulative_position_error")
    for row in top_cum:
        lines.append(
            f"- step {int(row['step_idx'])} (i={int(row['i'])}, j={int(row['j'])}): "
            f"pos_err={_fmt(row['metric_pos_err'])}, rot_err={_fmt(row['rot_err_deg'])}, tdir_err={_fmt(row['tdir_err_deg'])}"
        )
    lines.append("")
    lines.append("## Top bad steps by step_position_error")
    for row in top_step:
        lines.append(
            f"- step {int(row['step_idx'])} (i={int(row['i'])}, j={int(row['j'])}): "
            f"step_pos_err={_fmt(row['metric_pos_err'])}, tmag_ratio={_fmt(row['tmag_ratio'])}"
        )
    lines.append("")
    lines.append("## Top bad steps by tdir_error")
    for row in top_tdir:
        lines.append(
            f"- step {int(row['step_idx'])} (i={int(row['i'])}, j={int(row['j'])}): "
            f"tdir_err={_fmt(row['tdir_err_deg'])}, rot_err={_fmt(row['rot_err_deg'])}, pos_err={_fmt(row['metric_pos_err'])}"
        )
    lines.append("")
    lines.append("## Top bad steps by rot_error")
    for row in top_rot:
        lines.append(
            f"- step {int(row['step_idx'])} (i={int(row['i'])}, j={int(row['j'])}): "
            f"rot_err={_fmt(row['rot_err_deg'])}, tdir_err={_fmt(row['tdir_err_deg'])}, pos_err={_fmt(row['metric_pos_err'])}"
        )
    lines.append("")
    lines.append("## Official-scope vs all-chain secondary diagnostic")
    lines.append(f"- official debug-chain path_ratio: {_fmt(official_scope['official_current']['path_ratio'])}")
    lines.append(f"- official all-chain ATE/drift: {_fmt(official_scope['official_current']['ATE_all_chain'])} / {_fmt(official_scope['official_current']['drift_all_chain'])}")
    lines.append(f"- all-chain secondary diagnostic: weighted path_ratio={_fmt(all_chain_secondary['official_current']['path_ratio'])}, mean path_ratio={_fmt(all_chain_secondary['official_current']['mean_path_ratio'])}, chain count={all_chain_secondary['official_current']['num_chains']}")
    lines.append("- official metric is debug-chain scoped in this eval configuration; all-chain statistics are diagnostic and should not be compared directly as official metrics.")
    lines.append("")
    lines.append("## Correlation analysis")
    for k, v in corrs.items():
        lines.append(f"- {k} = `{_fmt(v)}`")
    lines.append("")
    lines.append("## Final classification")
    lines.append(f"- classification: `{classification}`")
    lines.append("")
    lines.append("## Next step")
    lines.append(f"- {next_step}")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[S2c2] wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
