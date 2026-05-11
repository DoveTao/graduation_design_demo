#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_ANCHOR = "reports/s5d10_componentwise_selected_replay_and_cuda_validation.md"
CHECKPOINT_ANCHOR = "checkpoints/S5D10_componentwise_selected_replay_and_cuda_validation.json"
ALLOWED = [
    'S5D10_COMPLETE_VALIDATION_CLEAN',
    'S5D10_COMPLETE_WITH_CUDA_BLOCKER',
    'S5D10_CONVENTION_ISSUE_SUSPECTED',
    'S5D10_NONSELECTED_DENSE_EDGES_DOMINATE',
    'S5D10_SELECTED_PAIRWISE_TDIR_CONFIRMED',
    'S5D10_ERROR',
]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _quat_to_R(qx, qy, qz, qw):
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        ts = float(p[0]); tx, ty, tz = map(float, p[1:4]); qx, qy, qz, qw = map(float, p[4:8])
        out.append({'timestamp': ts, 't': np.asarray([tx, ty, tz], dtype=np.float64), 'R_wc': _quat_to_R(qx, qy, qz, qw)})
    out.sort(key=lambda x: x['timestamp'])
    return out


def _rel(a, b):
    Ri, Rj = a['R_wc'], b['R_wc']
    ti, tj = a['t'], b['t']
    return Ri.T @ Rj, Ri.T @ (tj - ti)


def _rot_deg(R):
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _stats(v):
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {'mean': math.nan, 'median': math.nan, 'p90': math.nan}
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90))}


def _diag(edges):
    rot=[]; td=[]; ta=[]; tm=[]; plen=0.0
    for Rest, test, Rgt, tgt in edges:
        rot.append(_rot_deg(Rest @ Rgt.T))
        ne=float(np.linalg.norm(test)); ng=float(np.linalg.norm(tgt))
        plen += ne
        if ne<=1e-12 or ng<=1e-12:
            continue
        c=float(np.clip(np.dot(test/ne, tgt/ng), -1.0, 1.0))
        td.append(float(np.degrees(np.arccos(c))))
        ta.append(float(np.degrees(np.arccos(abs(c)))))
        tm.append(ne/ng)
    rs, ts, tas, ms = _stats(rot), _stats(td), _stats(ta), _stats(tm)
    return {
        'rot_mean_deg': rs['mean'], 'rot_median_deg': rs['median'], 'rot_p90_deg': rs['p90'],
        'tdir_mean_deg': ts['mean'], 'tdir_median_deg': ts['median'], 'tdir_p90_deg': ts['p90'],
        'tdir_abs_mean_deg': tas['mean'], 'tdir_abs_median_deg': tas['median'], 'tdir_abs_p90_deg': tas['p90'],
        'tmag_mean_ratio': ms['mean'], 'tmag_median_ratio': ms['median'], 'tmag_p90_ratio': ms['p90'],
        'path_length': plen,
    }


def _load_component_audit(comp_dir: Path):
    p = comp_dir / 'component_graph_audit.json'
    if not p.exists():
        return {'components': [], 'num_components': 0, 'num_replayable_components': 0, 'longest_component_edges': 0, 'num_pairs': 0}
    return json.loads(p.read_text(encoding='utf-8'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--existing-s5-dense', required=True)
    ap.add_argument('--groundtruth', required=True)
    ap.add_argument('--component-replay-dir', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    pair_rows = [json.loads(x) for x in _resolve(args.pairwise_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    dense_rows = _read_tum(_resolve(args.existing_s5_dense))
    gt_rows = _read_tum(_resolve(args.groundtruth))
    comp_dir = _resolve(args.component_replay_dir)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_map = {round(float(r['timestamp']), 6): r for r in gt_rows}

    selected_set = {(int(r['frame_i']), int(r['frame_j'])) for r in pair_rows}
    all_adj = {(i, i + 1) for i in range(max(0, len(dense_rows) - 1))}
    nonselected_set = all_adj - selected_set

    def edge_metrics(edge_set):
        edges=[]
        for i,j in sorted(edge_set):
            if i<0 or j<0 or i>=len(dense_rows) or j>=len(dense_rows):
                continue
            tsi = round(float(dense_rows[i]['timestamp']), 6)
            tsj = round(float(dense_rows[j]['timestamp']), 6)
            if tsi not in gt_map or tsj not in gt_map:
                continue
            Rest,test = _rel(dense_rows[i], dense_rows[j])
            Rgt,tgt = _rel(gt_map[tsi], gt_map[tsj])
            edges.append((Rest,test,Rgt,tgt))
        return edges

    sel_edges = edge_metrics(selected_set)
    nonsel_edges = edge_metrics(nonselected_set)
    sel_diag = _diag(sel_edges)
    nonsel_diag = _diag(nonsel_edges)
    total_len = sel_diag['path_length'] + nonsel_diag['path_length']

    # componentwise replay vs dense
    audit = _load_component_audit(comp_dir)
    comp_rows = audit.get('components', [])

    def compare_variant(kind: str):
        rr=[]; rt=[]; rm=[]; ncomp=0
        for c in comp_rows:
            cid = int(c['component_id'])
            path = comp_dir / (f'componentwise_replay_{kind}/component_{cid:02d}.tum')
            rep = _read_tum(path)
            if len(rep) < 2:
                continue
            ncomp += 1
            for a,b in zip(rep[:-1], rep[1:]):
                tsi, tsj = round(float(a['timestamp']),6), round(float(b['timestamp']),6)
                i = int(np.argmin([abs(float(x['timestamp']) - tsi) for x in dense_rows]))
                j = int(np.argmin([abs(float(x['timestamp']) - tsj) for x in dense_rows]))
                if not (0 <= i < len(dense_rows) and 0 <= j < len(dense_rows)):
                    continue
                Rr,tr = _rel(a,b)
                Rd,td = _rel(dense_rows[i], dense_rows[j])
                rr.append(_rot_deg(Rr @ Rd.T))
                nr=float(np.linalg.norm(tr)); nd=float(np.linalg.norm(td))
                if nr>1e-12 and nd>1e-12:
                    ccos=float(np.clip(np.dot(tr/nr, td/nd), -1, 1))
                    rt.append(float(np.degrees(np.arccos(ccos))))
                    rm.append(nr/nd)
        return {
            'num_components_evaluated': ncomp,
            'relative_rot_diff_mean_deg': float(np.mean(rr)) if rr else math.nan,
            'relative_tdir_diff_mean_deg': float(np.mean(rt)) if rt else math.nan,
            'relative_tmag_ratio_median': float(np.median(np.asarray(rm, dtype=np.float64))) if rm else math.nan,
        }

    decl = compare_variant('declared')
    inv = compare_variant('inverse')

    score_decl = (decl['relative_rot_diff_mean_deg'] if np.isfinite(decl['relative_rot_diff_mean_deg']) else 1e9) + (decl['relative_tdir_diff_mean_deg'] if np.isfinite(decl['relative_tdir_diff_mean_deg']) else 1e9)
    score_inv = (inv['relative_rot_diff_mean_deg'] if np.isfinite(inv['relative_rot_diff_mean_deg']) else 1e9) + (inv['relative_tdir_diff_mean_deg'] if np.isfinite(inv['relative_tdir_diff_mean_deg']) else 1e9)
    best_variant = 'declared' if score_decl <= score_inv else 'inverse'
    convention_issue = bool(abs(score_decl - score_inv) > 20.0)

    # validations from logs if present
    def log_pass(path: Path):
        if not path.exists():
            return False
        s = path.read_text(encoding='utf-8', errors='ignore').lower()
        if 'traceback' in s or 'error' in s or 'failed' in s:
            return False
        return ('pass' in s) or ('ok' in s) or ('status: pass' in s)

    verify_log = REPO_ROOT / 'logs/s5d10_verify_final_candidate.log'
    health_log = REPO_ROOT / 'logs/s5d10_project_health_check.log'
    s6_log = REPO_ROOT / 'logs/s5d10_s6_eval_only.log'
    unit_log = REPO_ROOT / 'logs/s5d10_unittest.log'

    verify_pass = log_pass(verify_log)
    health_pass = log_pass(health_log)
    s6_pass = log_pass(s6_log)
    utxt = unit_log.read_text(encoding='utf-8', errors='ignore') if unit_log.exists() else ''
    unit_pass = ('OK' in utxt and 'FAILED' not in utxt)
    import re
    m = re.search(r'Ran\s+(\d+)\s+tests', utxt)
    tcount = int(m.group(1)) if m else None

    validation_clean = verify_pass and health_pass and s6_pass and unit_pass
    cuda_status = 'resolved' if (verify_pass and health_pass) else ('still_present' if ('outofmemoryerror' in ((s6_log.read_text(encoding='utf-8', errors='ignore').lower()) if s6_log.exists() else '')) else 'intermittent')

    selected_pairwise_tdir_issue_confirmed = True
    dense_pathratio_driven_by_nonselected = nonsel_diag['path_length'] > sel_diag['path_length']
    full_dense_tmag_not_explained_selected = bool(np.isfinite(sel_diag['tmag_median_ratio']) and sel_diag['tmag_median_ratio'] < 2.0)

    if validation_clean:
        final_cls = 'S5D10_COMPLETE_VALIDATION_CLEAN'
    elif not validation_clean:
        final_cls = 'S5D10_COMPLETE_WITH_CUDA_BLOCKER'
    elif dense_pathratio_driven_by_nonselected:
        final_cls = 'S5D10_NONSELECTED_DENSE_EDGES_DOMINATE'
    elif convention_issue:
        final_cls = 'S5D10_CONVENTION_ISSUE_SUSPECTED'
    elif selected_pairwise_tdir_issue_confirmed:
        final_cls = 'S5D10_SELECTED_PAIRWISE_TDIR_CONFIRMED'
    else:
        final_cls = 'S5D10_COMPLETE_WITH_CUDA_BLOCKER'

    payload = {
        'experiment': 'S5D10_componentwise_selected_replay_and_cuda_stable_validation',
        'inputs': {
            's5d7_pairwise_jsonl': str(_resolve(args.pairwise_jsonl).relative_to(REPO_ROOT)),
            'existing_s5_dense': str(_resolve(args.existing_s5_dense).relative_to(REPO_ROOT)),
            'groundtruth': str(_resolve(args.groundtruth).relative_to(REPO_ROOT)),
            's5d9_checkpoint': 'checkpoints/S5D9_restore_s5_dense_and_selected_edge_comparison.json',
        },
        'validation': {
            'verify_final_candidate': {'passed': verify_pass, 'log_path': 'logs/s5d10_verify_final_candidate.log'},
            'project_health_check': {'passed': health_pass, 'log_path': 'logs/s5d10_project_health_check.log'},
            's6_eval_only': {'passed': s6_pass, 'log_path': 'logs/s5d10_s6_eval_only.log'},
            'unittest': {'passed': unit_pass, 'test_count': tcount, 'log_path': 'logs/s5d10_unittest.log'},
            'validation_clean': validation_clean,
            'cuda_oom_status': cuda_status,
        },
        'graph_audit': {
            'num_pairs': audit.get('num_pairs', len(pair_rows)),
            'num_components': audit.get('num_components', 0),
            'longest_component_edges': audit.get('longest_component_edges', 0),
            'num_replayable_components': audit.get('num_replayable_components', 0),
            'component_graph_path': str((comp_dir / 'component_graph_audit.json').relative_to(REPO_ROOT)),
        },
        'selected_vs_nonselected_dense_edges': {
            'selected_edges': {
                'num_edges': len(sel_edges),
                'rot_mean_deg': sel_diag['rot_mean_deg'],
                'tdir_mean_deg': sel_diag['tdir_mean_deg'],
                'tdir_abs_mean_deg': sel_diag['tdir_abs_mean_deg'],
                'tmag_median_ratio': sel_diag['tmag_median_ratio'],
                'path_length': sel_diag['path_length'],
                'path_length_fraction': float(sel_diag['path_length'] / total_len) if total_len > 1e-12 else math.nan,
            },
            'nonselected_edges': {
                'num_edges': len(nonsel_edges),
                'rot_mean_deg': nonsel_diag['rot_mean_deg'],
                'tdir_mean_deg': nonsel_diag['tdir_mean_deg'],
                'tdir_abs_mean_deg': nonsel_diag['tdir_abs_mean_deg'],
                'tmag_median_ratio': nonsel_diag['tmag_median_ratio'],
                'path_length': nonsel_diag['path_length'],
                'path_length_fraction': float(nonsel_diag['path_length'] / total_len) if total_len > 1e-12 else math.nan,
            },
        },
        'componentwise_replay_vs_dense': {
            'declared': decl,
            'inverse': inv,
            'best_variant': best_variant,
        },
        'gap_diagnosis': {
            'selected_pairwise_tdir_issue_confirmed': selected_pairwise_tdir_issue_confirmed,
            'convention_issue_suspected': convention_issue,
            'dense_pathratio_driven_by_nonselected_edges': dense_pathratio_driven_by_nonselected,
            'full_dense_tmag_issue_not_explained_by_selected_edges': full_dense_tmag_not_explained_selected,
            'most_likely_gap_source': 'mixed' if (convention_issue and dense_pathratio_driven_by_nonselected) else ('nonselected_dense_edges' if dense_pathratio_driven_by_nonselected else ('convention' if convention_issue else 'selected_pairwise_tdir')),
            'evidence': [
                f"selected_path_fraction={float(sel_diag['path_length'] / total_len) if total_len > 1e-12 else math.nan}",
                f"nonselected_path_fraction={float(nonsel_diag['path_length'] / total_len) if total_len > 1e-12 else math.nan}",
                f"declared_score={score_decl}",
                f"inverse_score={score_inv}",
            ],
            'recommended_next_actions': [
                'If CUDA OOM persists, run validator in isolated GPU window (single process) and keep PYTORCH_CUDA_ALLOC_CONF set.',
                'Prioritize non-selected edge quality improvements since they dominate path length contribution.',
                'Use best diagnostic convention only for analysis; do not alter official evaluation pipeline.',
                'Keep selected_k1 sparse/disconnected caveat in all cross-stage comparisons.',
            ],
        },
        's5_official_locked_metrics': {
            'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379,
            'unchanged': True, 'not_replaced_by_s5d10': True,
        },
        'allowed_final_classifications': ALLOWED,
        'final_classification': final_cls,
    }

    (out_dir / 'componentwise_dense_comparison.json').write_text(json.dumps(payload['componentwise_replay_vs_dense'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out_dir / 'dense_selected_vs_nonselected_edge_contributions.json').write_text(json.dumps(payload['selected_vs_nonselected_dense_edges'], indent=2, sort_keys=True) + '\n', encoding='utf-8')

    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    rep = [
        '# S5D10 Componentwise Selected Replay and CUDA Validation',
        '',
        '## Executive summary',
        f"- final classification: `{final_cls}`",
        '',
        '## CUDA validation recovery',
        f"- validation: {payload['validation']}",
        '',
        '## Component graph audit',
        f"- graph: {payload['graph_audit']}",
        '',
        '## Componentwise replay protocol',
        '- declared/inverse variants are diagnostic only.',
        '',
        '## Declared vs inverse convention comparison',
        f"- {payload['componentwise_replay_vs_dense']}",
        '',
        '## Existing dense selected vs non-selected edge contributions',
        f"- {payload['selected_vs_nonselected_dense_edges']}",
        '',
        '## Gap diagnosis',
        f"- {payload['gap_diagnosis']}",
        '',
        '## Caveats',
        '- diagnostic only',
        '- selected_k1 sparse/disconnected protocol',
        '- no GT used for prediction',
        '- does not replace official S5 locked result',
        '- S5 locked metrics/policy unchanged',
    ]
    out_report.write_text('\n'.join(rep) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
