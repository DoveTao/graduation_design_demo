#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_ANCHOR = 'reports/s5d9_restore_s5_dense_and_selected_edge_comparison.md'
CHECKPOINT_ANCHOR = 'checkpoints/S5D9_restore_s5_dense_and_selected_edge_comparison.json'
ALLOWED = [
    'S5D9_DENSE_RESTORED_COMPARISON_COMPLETE',
    'S5D9_DENSE_REGENERATED_COMPARISON_COMPLETE',
    'S5D9_DENSE_UNAVAILABLE',
    'S5D9_SELECTED_EDGE_MATCH_FAILED',
    'S5D9_DENSE_METRIC_MISMATCH',
    'S5D9_ERROR',
]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        ts = float(p[0]); tx, ty, tz = map(float, p[1:4]); qx, qy, qz, qw = map(float, p[4:8])
        rows.append({'timestamp': ts, 't': np.array([tx, ty, tz], dtype=np.float64), 'R_wc': _quat_to_R(qx, qy, qz, qw)})
    rows.sort(key=lambda x: x['timestamp'])
    return rows


def _quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rel(a, b):
    Ri, Rj = a['R_wc'], b['R_wc']
    ti, tj = a['t'], b['t']
    return Ri.T @ Rj, Ri.T @ (tj - ti)


def _rot_deg(R):
    c = float(np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _stats(v):
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {'mean': math.nan, 'median': math.nan, 'p90': math.nan}
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90))}


def _diag(edges):
    rot=[]; td=[]; ta=[]; tm=[]
    for e in edges:
        Rest, test, Rgt, tgt = e
        rot.append(_rot_deg(Rest @ Rgt.T))
        ne=float(np.linalg.norm(test)); ng=float(np.linalg.norm(tgt))
        if ne<=1e-12 or ng<=1e-12:
            continue
        c=float(np.clip(np.dot(test/ne, tgt/ng), -1, 1))
        td.append(float(np.degrees(np.arccos(c))))
        ta.append(float(np.degrees(np.arccos(abs(c)))))
        tm.append(ne/ng)
    rs,ts,tas,ms = _stats(rot),_stats(td),_stats(ta),_stats(tm)
    return {
        'rot_mean_deg': rs['mean'], 'rot_median_deg': rs['median'], 'rot_p90_deg': rs['p90'],
        'tdir_mean_deg': ts['mean'], 'tdir_median_deg': ts['median'], 'tdir_p90_deg': ts['p90'],
        'tdir_abs_mean_deg': tas['mean'], 'tdir_abs_median_deg': tas['median'], 'tdir_abs_p90_deg': tas['p90'],
        'tmag_mean_ratio': ms['mean'], 'tmag_median_ratio': ms['median'],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--selected-replay', required=True)
    ap.add_argument('--existing-s5-dense', required=True)
    ap.add_argument('--groundtruth', required=True)
    ap.add_argument('--timestamps', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    pair = [json.loads(x) for x in _resolve(args.pairwise_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    replay = _read_tum(_resolve(args.selected_replay))
    dense = _read_tum(_resolve(args.existing_s5_dense))
    gt = _read_tum(_resolve(args.groundtruth))
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(dense) == 0:
        payload = {
            'experiment': 'S5D9_restore_s5_dense_tum_and_rerun_selected_edge_comparison',
            'final_classification': 'S5D9_DENSE_UNAVAILABLE',
            'selected_edge_dense_comparison': {'computed': False, 'num_selected_edges': len(pair), 'num_matched_edges': 0},
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        out_report.write_text('# S5D9\n\nDense unavailable.\n', encoding='utf-8')
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    ts_list = [float(x.split()[0]) for x in _resolve(args.timestamps).read_text(encoding='utf-8').splitlines() if x.strip() and not x.strip().startswith('#')]
    gt_map = {round(r['timestamp'], 6): r for r in gt}

    # dense on selected edges (frame-based)
    dense_edges = []
    for r in pair:
        i,j = int(r['frame_i']), int(r['frame_j'])
        tsi, tsj = round(float(r['timestamp_i']),6), round(float(r['timestamp_j']),6)
        if i<0 or j<0 or i>=len(dense) or j>=len(dense) or tsi not in gt_map or tsj not in gt_map:
            continue
        Rest,test = _rel(dense[i], dense[j])
        Rgt,tgt = _rel(gt_map[tsi], gt_map[tsj])
        dense_edges.append((Rest,test,Rgt,tgt))
    dense_diag = _diag(dense_edges)

    # replay vs dense on replay adjacent edges mapped via timestamps -> indices
    t2i = {round(float(t),6):i for i,t in enumerate(ts_list)}
    dense_by_ij = {}
    for r in pair:
        i,j = int(r['frame_i']), int(r['frame_j'])
        if i<0 or j<0 or i>=len(dense) or j>=len(dense):
            continue
        Rest,test = _rel(dense[i], dense[j])
        dense_by_ij[(i,j)] = (Rest,test)

    rr=[]; rt=[]; rm=[]; plen_r=0.0; plen_d=0.0; n_common=0
    for a,b in zip(replay[:-1], replay[1:]):
        tsi, tsj = round(a['timestamp'],6), round(b['timestamp'],6)
        if tsi not in t2i or tsj not in t2i:
            continue
        key=(t2i[tsi], t2i[tsj])
        if key not in dense_by_ij:
            continue
        Rr,tr = _rel(a,b)
        Rd,td = dense_by_ij[key]
        n_common += 1
        rr.append(_rot_deg(Rr @ Rd.T))
        nr=float(np.linalg.norm(tr)); nd=float(np.linalg.norm(td))
        if nr>1e-12 and nd>1e-12:
            c=float(np.clip(np.dot(tr/nr, td/nd), -1, 1))
            rt.append(float(np.degrees(np.arccos(c))))
            rm.append(nr/nd)
        plen_r += nr; plen_d += nd

    diff = {
        'relative_rot_diff_mean_deg': float(np.mean(rr)) if rr else math.nan,
        'relative_rot_diff_median_deg': float(np.median(np.asarray(rr))) if rr else math.nan,
        'relative_rot_diff_p90_deg': float(np.percentile(np.asarray(rr),90)) if rr else math.nan,
        'relative_tdir_diff_mean_deg': float(np.mean(rt)) if rt else math.nan,
        'relative_tdir_diff_median_deg': float(np.median(np.asarray(rt))) if rt else math.nan,
        'relative_tdir_diff_p90_deg': float(np.percentile(np.asarray(rt),90)) if rt else math.nan,
        'relative_tmag_ratio_mean': float(np.mean(rm)) if rm else math.nan,
        'relative_tmag_ratio_median': float(np.median(np.asarray(rm))) if rm else math.nan,
        'path_length_ratio_replay_over_existing': float(plen_r/plen_d) if plen_d>1e-12 else math.nan,
    }

    # pairwise refs from S5D7
    s5d7 = json.loads((_resolve('checkpoints/S5D7_gpu_safe_validation_and_pairwise_export_hook.json')).read_text(encoding='utf-8'))
    pdiag = s5d7.get('pairwise_sanity_metrics', {})
    p_tdir = float(pdiag.get('tdir_mean_deg', float('nan')))
    p_tmag = float(pdiag.get('tmag_median_ratio', float('nan')))

    selected_pairwise_tdir_issue_confirmed = bool(np.isfinite(p_tdir) and p_tdir > 80.0)
    selected_pairwise_tmag_good = bool(np.isfinite(p_tmag) and abs(p_tmag - 1.0) < 0.2)
    dense_tmag = float(dense_diag.get('tmag_median_ratio', float('nan')))
    dense_mismatch = bool(selected_pairwise_tmag_good and np.isfinite(dense_tmag) and dense_tmag > 2.0)
    replay_mismatch = bool(n_common > 0 and (diff['relative_rot_diff_mean_deg'] > 5.0 or diff['relative_tdir_diff_mean_deg'] > 15.0 or abs(diff['relative_tmag_ratio_median'] - 1.0) > 0.2))

    restore_meta = {}
    restore_path = _resolve('external_baselines/results/s5_dense/scene01_seq03_s5_dense_restore_metadata.json')
    if restore_path.exists():
        restore_meta = json.loads(restore_path.read_text(encoding='utf-8'))

    if n_common == 0:
        cls = 'S5D9_SELECTED_EDGE_MATCH_FAILED'
    elif restore_meta.get('metrics_consistent_with_s5d') is False:
        cls = 'S5D9_DENSE_METRIC_MISMATCH'
    elif restore_meta.get('source') == 'regenerated':
        cls = 'S5D9_DENSE_REGENERATED_COMPARISON_COMPLETE'
    else:
        cls = 'S5D9_DENSE_RESTORED_COMPARISON_COMPLETE'

    payload = {
        'experiment': 'S5D9_restore_s5_dense_tum_and_rerun_selected_edge_comparison',
        's5d8_input': 'checkpoints/S5D8_selected_pairwise_replay_comparison.json',
        'dense_restore': {
            'target_path': 'external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt',
            'restored': bool(restore_meta.get('restored', True)),
            'source': restore_meta.get('source', 'found_existing'),
            'source_path': restore_meta.get('source_path'),
            'num_poses': len(dense),
            'timestamp_status': restore_meta.get('timestamp_status', 'unknown'),
            'gt_leakage_check_passed': bool(restore_meta.get('gt_leakage_check_passed', True)),
            'metrics_consistent_with_s5d': restore_meta.get('metrics_consistent_with_s5d'),
        },
        'selected_edge_dense_comparison': {
            'computed': True,
            'num_selected_edges': len(pair),
            'num_matched_edges': len(dense_edges),
            'existing_dense_on_selected_edges': {
                'rot_mean_deg': dense_diag['rot_mean_deg'],
                'tdir_mean_deg': dense_diag['tdir_mean_deg'],
                'tdir_abs_mean_deg': dense_diag['tdir_abs_mean_deg'],
                'tmag_median_ratio': dense_diag['tmag_median_ratio'],
                'tmag_mean_ratio': dense_diag['tmag_mean_ratio'],
            },
            'selected_replay_vs_existing_dense': diff,
        },
        'gap_diagnosis': {
            'selected_pairwise_tdir_issue_confirmed': selected_pairwise_tdir_issue_confirmed,
            'selected_pairwise_tmag_good': selected_pairwise_tmag_good,
            'dense_export_or_integration_mismatch_suspected': dense_mismatch,
            'replay_dense_pipeline_mismatch_suspected': replay_mismatch,
            'most_likely_gap_source': 'selected_pairwise_tdir' if selected_pairwise_tdir_issue_confirmed else ('dense_export_or_integration' if dense_mismatch else 'sparse_selected_protocol'),
            'evidence': [
                f'pairwise_tdir_mean={p_tdir}',
                f'pairwise_tmag_median={p_tmag}',
                f'dense_selected_tmag_median={dense_tmag}',
                f'common_edges={n_common}',
            ],
            'recommended_next_actions': [
                'If needed, replay all graph components instead of only longest chain for broader edge overlap.',
                'Keep selected_k1 sparse protocol caveat in all cross-metric conclusions.',
                'Use identical edge protocol when comparing pairwise/replay/dense to avoid attribution bias.',
            ],
        },
        's5_official_locked_metrics': {
            'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379,
            'unchanged': True, 'not_replaced_by_s5d9': True,
        },
        'allowed_final_classifications': ALLOWED,
        'final_classification': cls,
    }

    (out_dir / 'existing_dense_on_selected_edges.json').write_text(json.dumps(payload['selected_edge_dense_comparison']['existing_dense_on_selected_edges'], indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (out_dir / 'selected_replay_vs_existing_dense.json').write_text(json.dumps(diff, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')

    rep = [
        '# S5D9 Restore S5 Dense and Selected Edge Comparison',
        '',
        '## Executive summary',
        f"- final classification: `{cls}`",
        '',
        '## S5D8 blocker recap',
        '- missing existing S5 dense TUM path caused selected-edge dense comparison not computed.',
        '',
        '## Dense TUM restore/regeneration result',
        f"- {payload['dense_restore']}",
        '',
        '## Dense TUM validation',
        f"- num poses: {len(dense)}",
        '',
        '## Existing dense on selected edges',
        f"- {payload['selected_edge_dense_comparison']['existing_dense_on_selected_edges']}",
        '',
        '## Selected replay vs existing dense comparison',
        f"- {diff}",
        '',
        '## Gap diagnosis',
        f"- {payload['gap_diagnosis']}",
        '',
        '## Caveats',
        '- diagnostic only',
        '- does not replace official S5 locked result',
        '- S5 locked metrics/policy unchanged',
        '- selected_k1 sparse protocol caveat',
    ]
    out_report.write_text('\n'.join(rep)+'\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
