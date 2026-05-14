#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from eval_clean_policy import _load_policy, _run  # type: ignore

ROT_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
POLICY_PATH = REPO_ROOT / 'checkpoints' / 'S1d5_clean_dt_anchor_policy.json'
OUT_ROOT = REPO_ROOT / 'checkpoints' / 'S2a_fine_rot_policy_audit_on_s1d5'
REPORT_PATH = REPO_ROOT / 'checkpoints' / 'S2a_fine_rot_policy_audit_on_s1d5_report.md'
SUMMARY_JSON = OUT_ROOT / 's2a_rot_sweep_summary.json'

BASELINES = [
    {
        'label': 'true T57b multiscale 0/0',
        'status': 'valid clean baseline',
        'drift': 1.494,
        'ATE': 9.644,
        'path_ratio': 0.496,
        'fine_rot': 0.0,
        'notes': 'explicit-cfg / unexpected=0',
    },
    {
        'label': 'true T57b multiscale + rot-only',
        'status': 'valid eval-only diagnostic',
        'drift': 1.310,
        'ATE': 7.611,
        'path_ratio': 0.496,
        'fine_rot': 0.25,
        'notes': 'explicit-cfg / unexpected=0',
    },
    {
        'label': 'S1d3 diagnostic best',
        'status': 'test-swept diagnostic target',
        'drift': 1.433,
        'ATE': 7.971,
        'path_ratio': 0.970,
        'fine_rot': 0.35,
        'notes': 'alpha=1.10; not clean fitted mainline',
    },
    {
        'label': 'S1d4 train-selected policy',
        'status': 'clean train-CV selected mainline precursor',
        'drift': 1.396,
        'ATE': 7.632,
        'path_ratio': 0.935,
        'fine_rot': 0.40,
        'notes': 'alpha=1.05; train-only selected',
    },
    {
        'label': 'S1d5 current policy',
        'status': 'current clean exported mainline',
        'drift': 1.396358,
        'ATE': 7.632463,
        'path_ratio': 0.934982,
        'fine_rot': 0.40,
        'notes': 'exported policy',
    },
]


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _candidate(row: Dict[str, Any], baseline_ate: float, baseline_drift: float) -> bool:
    eps = 1e-4
    return (
        float(row['metric_path_ratio']) >= 0.90
        and float(row['ATE']) < (baseline_ate - eps)
        and float(row['drift']) < (baseline_drift - eps)
    )


def main() -> None:
    base_policy = _load_policy(POLICY_PATH)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for rot in ROT_VALUES:
        policy = deepcopy(base_policy)
        policy['fine_rot_fuse_strength'] = float(rot)
        policy['fine_tdir_fuse_strength'] = 0.0
        policy['fine_tmag_fuse_strength'] = 0.0
        run_dir = OUT_ROOT / f"rot_{str(rot).replace('.', 'p')}"
        policy_copy_path = run_dir / 'policy_used.json'
        summary_path = run_dir / 's1d5_policy_eval_summary.json'
        run_dir.mkdir(parents=True, exist_ok=True)
        policy_copy_path.write_text(json.dumps(policy, indent=2), encoding='utf-8')
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding='utf-8'))
        else:
            payload = _run(policy, policy_copy_path, run_dir, 'default')
        payload['run_dir'] = str(run_dir.relative_to(REPO_ROOT))
        rows.append(payload)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2), encoding='utf-8')

    s1d5 = next(b for b in BASELINES if b['label'] == 'S1d5 current policy')
    candidates = [r for r in rows if _candidate(r, s1d5['ATE'], s1d5['drift'])]
    best_by_ate = min(rows, key=lambda r: (float(r['ATE']), float(r['drift'])))
    best_by_drift = min(rows, key=lambda r: (float(r['drift']), float(r['ATE'])))

    lines: List[str] = []
    lines.append('# S2a fine rot policy audit on S1d5\n\n')
    lines.append('## Setup\n')
    lines.append(f'- branch: `optimize/s2-fine-refinement-on-s1d5`\n')
    lines.append(f'- base checkpoint: `{base_policy["base_checkpoint_path"]}`\n')
    lines.append(f'- policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`\n')
    lines.append('- fixed dt-anchor effective factors: unchanged from S1d5\n')
    lines.append('- fixed settings: `fine_tdir=0.0`, `fine_tmag=0.0`, `use_geometry_refine=False`\n')
    lines.append('- protocol: `explicit-cfg / unexpected=0` required\n')
    lines.append('- no test labels used for fitting or clean selection\n\n')

    lines.append('## Baselines\n')
    lines.append('| method | status | fine_rot | drift | ATE | path_ratio | notes |\n')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | --- |\n')
    for b in BASELINES:
        lines.append(f"| {b['label']} | {b['status']} | {_fmt(b['fine_rot'])} | {_fmt(b['drift'])} | {_fmt(b['ATE'])} | {_fmt(b['path_ratio'])} | {b['notes']} |\n")
    lines.append('\n')

    lines.append('## Sweep Results\n')
    lines.append('| fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | metric_path_ratio | direction_only_path_ratio | tmag_p10 | tmag_p50 | tmag_p90 | selected_k | num_pairs | num_chains | missing | unexpected | run_dir |\n')
    lines.append('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n')
    for r in rows:
        lines.append(
            f"| {_fmt(r['fine_rot_fuse_strength'])} | {_fmt(r['drift'])} | {_fmt(r['ATE'])} | {_fmt(r['metric_path_ratio'])} | {_fmt(r['RPE_rot'])} | {_fmt(r['RPE_trans_dir'])} | {_fmt(r['RPE_trans_mag'])} | {_fmt(r['rot'])} | {_fmt(r['tdir_abs'])} | {_fmt(r['tdir_local_A_abs'])} | {_fmt(r['metric_path_ratio'])} | {_fmt(r['direction_only_path_ratio'])} | {_fmt(r['tmag_p10'])} | {_fmt(r['tmag_p50'])} | {_fmt(r['tmag_p90'])} | {_fmt(r['odom_selected_k'])} | {_fmt(r['num_pairs'])} | {_fmt(r['num_chains'])} | {_fmt(r['load_missing'])} | {_fmt(r['load_unexpected'])} | `{r['run_dir']}` |\n"
        )
    lines.append('\n')

    lines.append('## Diagnostic Candidates\n')
    if candidates:
        lines.append('- Candidates that improve over S1d5 current policy on both ATE and drift while keeping `path_ratio >= 0.90`:\n')
        for r in candidates:
            lines.append(
                f"  - `fine_rot={r['fine_rot_fuse_strength']:.2f}`: drift={r['drift']:.6f}, ATE={r['ATE']:.6f}, path_ratio={r['metric_path_ratio']:.6f}\n"
            )
    else:
        lines.append('- No fine_rot value strictly improved both ATE and drift over S1d5 while keeping `path_ratio >= 0.90`.\n')
    lines.append('\n')

    lines.append('## Best Observed Settings\n')
    lines.append(f"- best by ATE: `fine_rot={best_by_ate['fine_rot_fuse_strength']:.2f}` -> drift={best_by_ate['drift']:.6f}, ATE={best_by_ate['ATE']:.6f}, path_ratio={best_by_ate['metric_path_ratio']:.6f}\n")
    lines.append(f"- best by drift: `fine_rot={best_by_drift['fine_rot_fuse_strength']:.2f}` -> drift={best_by_drift['drift']:.6f}, ATE={best_by_drift['ATE']:.6f}, path_ratio={best_by_drift['metric_path_ratio']:.6f}\n\n")

    lines.append('## Interpretation\n')
    lines.append('- This is an eval-only test sweep on top of the fixed S1d5 exported policy.\n')
    lines.append('- Any better fine_rot found here is a `diagnostic candidate`, not a clean replacement for S1d5.\n')
    lines.append('- If a better candidate exists, the next step should be `S2b` train-only / train-CV selection.\n\n')

    lines.append('## Mainline Status\n')
    lines.append('- `reports/current_valid_baselines.md` was not updated unless a clearly better clean policy existed.\n')
    lines.append('- `S1d5` remains the current clean exported mainline after this diagnostic audit.\n')
    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    main()
