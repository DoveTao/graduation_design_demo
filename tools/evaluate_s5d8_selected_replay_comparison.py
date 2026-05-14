#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED = [
    'S5D8_SELECTED_REPLAY_COMPLETE',
    'S5D8_REPLAY_GRAPH_DISCONNECTED',
    'S5D8_DENSE_EXPORT_MISMATCH_SUSPECTED',
    'S5D8_SELECTED_PAIRWISE_TDIR_ISSUE_CONFIRMED',
    'S5D8_SELECTED_SPARSE_ONLY',
    'S5D8_ERROR',
]

DEFAULT_REPORT = 'reports/s5d8_selected_pairwise_replay_comparison.md'
DEFAULT_CKPT = 'checkpoints/S5D8_selected_pairwise_replay_comparison.json'


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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


def _read_timestamps(path: Path) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        vals.append(float(s.split()[0]))
    return vals


def _map(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    return {round(float(r['timestamp']), 6): r for r in rows}


def _rel(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    Ri, Rj = a['R_wc'], b['R_wc']
    ti, tj = a['t'], b['t']
    return Ri.T @ Rj, Ri.T @ (tj - ti)


def _rot_deg(R: np.ndarray) -> float:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _stats(v: List[float]) -> Dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {'mean': math.nan, 'median': math.nan, 'p90': math.nan}
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90))}


def _component_from_edges(edges: List[Tuple]) -> Dict[str, Any]:
    rot, tdir, tdir_abs, tmag = [], [], [], []
    for e in edges:
        Rest, test, Rgt, tgt = e[-4], e[-3], e[-2], e[-1]
        rot.append(_rot_deg(Rest @ Rgt.T))
        ne = float(np.linalg.norm(test)); ng = float(np.linalg.norm(tgt))
        if ne <= 1e-12 or ng <= 1e-12:
            continue
        c = float(np.clip(np.dot(test / ne, tgt / ng), -1.0, 1.0))
        tdir.append(float(np.degrees(np.arccos(c))))
        tdir_abs.append(float(np.degrees(np.arccos(abs(c)))))
        tmag.append(ne / ng)
    rs, ts, tas, ms = _stats(rot), _stats(tdir), _stats(tdir_abs), _stats(tmag)
    return {
        'rot_mean_deg': rs['mean'], 'rot_median_deg': rs['median'], 'rot_p90_deg': rs['p90'],
        'tdir_mean_deg': ts['mean'], 'tdir_median_deg': ts['median'], 'tdir_p90_deg': ts['p90'],
        'tdir_abs_mean_deg': tas['mean'], 'tdir_abs_median_deg': tas['median'], 'tdir_abs_p90_deg': tas['p90'],
        'tmag_mean_ratio': ms['mean'], 'tmag_median_ratio': ms['median'],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--replayed-tum', required=True)
    ap.add_argument('--existing-s5-dense', required=True)
    ap.add_argument('--groundtruth', required=True)
    ap.add_argument('--timestamps', required=True)
    ap.add_argument('--out-json', default=DEFAULT_CKPT)
    ap.add_argument('--out-report', default=DEFAULT_REPORT)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    pair_path = _resolve(args.pairwise_jsonl)
    replay_path = _resolve(args.replayed_tum)
    dense_path = _resolve(args.existing_s5_dense)
    gt_path = _resolve(args.groundtruth)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    pair_rows = [json.loads(x) for x in pair_path.read_text(encoding='utf-8', errors='ignore').splitlines() if x.strip()] if pair_path.exists() else []
    gt_rows = _read_tum(gt_path)
    gt_map = _map(gt_rows)

    replay_meta_path = replay_path.parent / 'selected_k1_replay_metadata.json'
    replay_meta = json.loads(replay_meta_path.read_text(encoding='utf-8')) if replay_meta_path.exists() else {}

    # artifact graph audit
    nodes = set()
    und: Dict[int, set] = {}
    for r in pair_rows:
        i, j = int(r['frame_i']), int(r['frame_j'])
        nodes |= {i, j}
        und.setdefault(i, set()).add(j)
        und.setdefault(j, set()).add(i)
    seen = set(); comp_sizes = []
    for n in sorted(nodes):
        if n in seen:
            continue
        st = [n]; seen.add(n); c = 0
        while st:
            u = st.pop(); c += 1
            for v in und.get(u, set()):
                if v not in seen:
                    seen.add(v); st.append(v)
        comp_sizes.append(c)
    comp_sizes.sort(reverse=True)
    graph_connected = len(comp_sizes) <= 1

    # pairwise component diagnostics
    edges_pair = []
    for r in pair_rows:
        tsi, tsj = round(float(r['timestamp_i']), 6), round(float(r['timestamp_j']), 6)
        if tsi not in gt_map or tsj not in gt_map:
            continue
        ga, gb = gt_map[tsi], gt_map[tsj]
        Rgt, tgt = _rel(ga, gb)
        Rest = np.asarray(r['rotation']['value'], dtype=np.float64)
        test = np.asarray(r['translation']['value'], dtype=np.float64)
        edges_pair.append((tsi, tsj, Rest, test, Rgt, tgt))
    pair_diag = _component_from_edges(edges_pair)

    # replay trajectory diagnostics (adjacent in replayed tum order)
    rep_rows = _read_tum(replay_path)
    ts_list = _read_timestamps(_resolve(args.timestamps))
    ts_to_idx = {round(float(t), 6): i for i, t in enumerate(ts_list)}
    rep_edges = []
    for a, b in zip(rep_rows[:-1], rep_rows[1:]):
        tsi, tsj = round(float(a['timestamp']), 6), round(float(b['timestamp']), 6)
        if tsi not in gt_map or tsj not in gt_map:
            continue
        Rest, test = _rel(a, b)
        Rgt, tgt = _rel(gt_map[tsi], gt_map[tsj])
        rep_edges.append((tsi, tsj, ts_to_idx.get(tsi, -1), ts_to_idx.get(tsj, -1), Rest, test, Rgt, tgt))
    replay_diag = _component_from_edges(rep_edges)

    # existing dense on same selected edges
    dense_rows = _read_tum(dense_path)
    dense_available = len(dense_rows) > 0
    dense_edges = []
    for r in pair_rows:
        if not dense_available:
            break
        fi, fj = int(r['frame_i']), int(r['frame_j'])
        tsi, tsj = round(float(r['timestamp_i']), 6), round(float(r['timestamp_j']), 6)
        if fi < 0 or fj < 0 or fi >= len(dense_rows) or fj >= len(dense_rows) or tsi not in gt_map or tsj not in gt_map:
            continue
        Rest, test = _rel(dense_rows[fi], dense_rows[fj])
        Rgt, tgt = _rel(gt_map[tsi], gt_map[tsj])
        dense_edges.append((float(fi), float(fj), Rest, test, Rgt, tgt))
    dense_sel_diag = _component_from_edges(dense_edges)

    # replay vs existing on common edges of replay path
    replay_vs_dense = {'compared': False, 'num_common_edges': 0, 'relative_rot_diff_mean_deg': math.nan, 'relative_tdir_diff_mean_deg': math.nan, 'relative_tmag_ratio_median': math.nan, 'path_length_ratio_replay_over_existing': math.nan}
    if rep_edges:
        replay_edge_map = {(int(e['i']), int(e['j'])): e for e in replay_meta.get('replayable_chain_edges', [])}
        d = {(int(e[0]), int(e[1])): e for e in dense_edges}
        rr, rt, rm = [], [], []
        p_rep = 0.0
        p_den = 0.0
        n_common = 0
        for e in rep_edges:
            fi, fj = int(e[2]), int(e[3])
            if fi < 0 or fj < 0:
                continue
            key = (fi, fj)
            if key not in d:
                continue
            de = d[key]
            n_common += 1
            rr.append(_rot_deg(e[4] @ de[2].T))
            ne = float(np.linalg.norm(e[5])); nd = float(np.linalg.norm(de[3]))
            if ne > 1e-12 and nd > 1e-12:
                c = float(np.clip(np.dot(e[5]/ne, de[3]/nd), -1.0, 1.0))
                rt.append(float(np.degrees(np.arccos(c))))
                rm.append(ne / nd)
            p_rep += ne
            p_den += nd
        if n_common > 0:
            replay_vs_dense = {
                'compared': True,
                'num_common_edges': n_common,
                'relative_rot_diff_mean_deg': float(np.mean(rr)) if rr else math.nan,
                'relative_tdir_diff_mean_deg': float(np.mean(rt)) if rt else math.nan,
                'relative_tmag_ratio_median': float(np.median(np.asarray(rm, dtype=np.float64))) if rm else math.nan,
                'path_length_ratio_replay_over_existing': float(p_rep / p_den) if p_den > 1e-12 else math.nan,
            }

    # external eval
    external = {'attempted': False, 'none': {}, 'se3': {}, 'sim3': {}}
    if len(rep_rows) >= 2:
        external['attempted'] = True
        for a in ('none', 'se3', 'sim3'):
            outp = out_dir / f'eval_selected_replay_alignment_{a}.json'
            proc = subprocess.run([
                '/home/dovetao/miniconda3/envs/pytorch/bin/python',
                'tools/evaluate_external_baseline_trajectory.py',
                '--trajectory', str(replay_path),
                '--groundtruth', str(gt_path),
                '--alignment', a,
                '--output-json', str(outp),
            ], cwd=REPO_ROOT, capture_output=True, text=True)
            if outp.exists():
                external[a] = json.loads(outp.read_text(encoding='utf-8'))
            else:
                external[a] = {'status': 'failed', 'stderr_tail': proc.stderr[-400:]}

    # judgments
    dense_export_consistent = None
    mismatch_suspected = None
    if replay_vs_dense['compared']:
        dense_export_consistent = bool(replay_vs_dense['relative_rot_diff_mean_deg'] < 1.0 and replay_vs_dense['relative_tdir_diff_mean_deg'] < 5.0 and abs(replay_vs_dense['relative_tmag_ratio_median'] - 1.0) < 0.05)
        mismatch_suspected = not dense_export_consistent

    tmag_good = pair_diag['tmag_median_ratio'] == pair_diag['tmag_median_ratio'] and abs(pair_diag['tmag_median_ratio'] - 1.0) < 0.2
    dense_tmag_bad = dense_sel_diag['tmag_median_ratio'] == dense_sel_diag['tmag_median_ratio'] and dense_sel_diag['tmag_median_ratio'] > 2.0
    selected_tdir_bad = pair_diag['tdir_mean_deg'] == pair_diag['tdir_mean_deg'] and pair_diag['tdir_mean_deg'] > 80.0

    if (not graph_connected):
        final_cls = 'S5D8_REPLAY_GRAPH_DISCONNECTED'
    elif selected_tdir_bad:
        final_cls = 'S5D8_SELECTED_PAIRWISE_TDIR_ISSUE_CONFIRMED'
    elif tmag_good and dense_tmag_bad:
        final_cls = 'S5D8_DENSE_EXPORT_MISMATCH_SUSPECTED'
    elif len(rep_rows) < 2:
        final_cls = 'S5D8_SELECTED_SPARSE_ONLY'
    else:
        final_cls = 'S5D8_SELECTED_REPLAY_COMPLETE'

    payload = {
        'experiment': 'S5D8_replay_s5d7_pairwise_artifact_and_compare_dense',
        'inputs': {
            's5d7_checkpoint': 'checkpoints/S5D7_gpu_safe_validation_and_pairwise_export_hook.json',
            'pairwise_jsonl': str(pair_path.relative_to(REPO_ROOT)),
            'existing_s5_dense': str(dense_path.relative_to(REPO_ROOT)),
            'groundtruth': str(gt_path.relative_to(REPO_ROOT)),
            'timestamps': args.timestamps,
        },
        'artifact_audit': {
            'num_pairs': len(pair_rows),
            'pair_selection': 'selected_k1',
            'coverage_vs_adjacent_453': float(len(pair_rows) / 453.0),
            'graph_connected': graph_connected,
            'num_components': len(comp_sizes),
            'num_replayable_edges': len(replay_meta.get('replayable_chain_edges', [])),
            'metadata_sufficient_for_replay': bool(pair_rows),
        },
        'pairwise_component_diagnostics': pair_diag,
        'replay': {
            'attempted': True,
            'variants': {
                'model_convention_as_declared': {
                    'trajectory_path': str(replay_path.relative_to(REPO_ROOT)),
                    'num_poses': int(replay_meta.get('variants', {}).get('model_convention_as_declared', {}).get('num_poses', len(rep_rows))),
                    'coverage_vs_454': float(replay_meta.get('variants', {}).get('model_convention_as_declared', {}).get('coverage_vs_454', len(rep_rows)/454.0 if rep_rows else 0.0)),
                    'component_diagnostics_path': str((out_dir / 'selected_k1_component_diagnostics.json').relative_to(REPO_ROOT)),
                },
                'inverse_relative_variant': {
                    'trajectory_path': str((replay_path.parent / (replay_path.stem + '_inverse.txt')).relative_to(REPO_ROOT)),
                    'num_poses': int(replay_meta.get('variants', {}).get('inverse_relative_variant', {}).get('num_poses', 0)),
                    'coverage_vs_454': float(replay_meta.get('variants', {}).get('inverse_relative_variant', {}).get('coverage_vs_454', 0.0)),
                    'component_diagnostics_path': str((out_dir / 'selected_k1_component_diagnostics_inverse.json').relative_to(REPO_ROOT)),
                },
            },
            'selected_variant': replay_meta.get('selected_variant', 'none'),
        },
        'selected_replay_component_diagnostics': replay_diag,
        'existing_dense_on_selected_edges': {'computed': dense_available, 'num_edges': len(dense_edges), **dense_sel_diag},
        'replay_vs_existing_dense': {
            **replay_vs_dense,
            'dense_export_consistent_with_selected_pairwise': dense_export_consistent,
            'dense_export_or_integration_mismatch_suspected': mismatch_suspected,
        },
        'external_eval': external,
        'gap_diagnosis': {
            'selected_pairwise_tdir_issue_confirmed': selected_tdir_bad,
            'selected_pairwise_tmag_good': tmag_good,
            'dense_tmag_mismatch_on_selected_edges': dense_tmag_bad,
            'integration_gap_confirmed': bool(mismatch_suspected) if mismatch_suspected is not None else None,
            'most_likely_gap_source': 'selected_pairwise_tdir' if selected_tdir_bad else ('dense_export_or_integration' if dense_tmag_bad else ('graph_disconnected' if not graph_connected else 'sparse_selected_protocol')),
            'evidence': [
                f"pairwise tdir mean={pair_diag['tdir_mean_deg']}",
                f"pairwise tmag median ratio={pair_diag['tmag_median_ratio']}",
                f"dense(selected) tmag median ratio={dense_sel_diag['tmag_median_ratio']}",
                f"components={len(comp_sizes)}",
            ],
            'recommended_next_actions': [
                'Run S5D8 inverse-variant replay diagnostics side-by-side in downstream analysis.',
                'Add explicit selected_k1 edge list export to official diagnostic path for reproducible replay.',
                'For dense conclusions, compare on identical edge protocol before path-level claims.',
                'Keep selected_k1 sparse protocol caveat in all tdir/tmag reporting.',
            ],
        },
        's5_official_locked_metrics': {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'unchanged': True, 'not_replaced_by_s5d8': True},
        'final_classification': final_cls,
        'allowed_final_classifications': ALLOWED,
    }

    # side output diagnostics
    (out_dir / 'selected_k1_component_diagnostics.json').write_text(json.dumps(replay_diag, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out_dir / 'selected_k1_vs_existing_dense_comparison.json').write_text(json.dumps(payload['replay_vs_existing_dense'], indent=2, sort_keys=True) + '\n', encoding='utf-8')

    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    lines = [
        '# S5D8 Selected Pairwise Replay Comparison',
        '',
        '## Executive summary',
        f"- final classification: `{final_cls}`",
        '',
        '## S5D7 artifact recap',
        f"- pairs: {len(pair_rows)} (`selected_k1` sparse protocol)",
        '',
        '## Artifact graph audit',
        f"- connected: `{graph_connected}`",
        f"- components: `{len(comp_sizes)}`",
        f"- replayable edges: `{len(replay_meta.get('replayable_chain_edges', []))}`",
        '',
        '## Pairwise component diagnostics',
        f"- {pair_diag}",
        '',
        '## Replay protocol and coverage',
        f"- selected variant: `{replay_meta.get('selected_variant','none')}`",
        f"- num poses: `{len(rep_rows)}` / 454",
        '',
        '## Existing dense restricted to selected edges',
        f"- {payload['existing_dense_on_selected_edges']}",
        '',
        '## Replay vs existing dense comparison',
        f"- {payload['replay_vs_existing_dense']}",
        '',
        '## Gap diagnosis',
        f"- {payload['gap_diagnosis']}",
        '',
        '## Caveats',
        '- diagnostic only',
        '- selected_k1 sparse protocol',
        '- does not replace official S5 locked result',
        '- S5 locked metrics/policy unchanged',
    ]
    out_report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
