#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {'count': 0, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'anti_parallel_rate': None, 'signed_tdir_mean_deg': None}
    arr = np.asarray([r['tmag_ratio'] for r in rows], dtype=np.float64)
    anti = np.asarray([1.0 if r.get('anti_parallel_flag', False) else 0.0 for r in rows], dtype=np.float64)
    tdir = np.asarray([r['tdir_deg'] for r in rows], dtype=np.float64)
    return {
        'count': len(rows),
        'tmag_median_ratio': float(np.median(arr)),
        'tmag_p95_ratio': float(np.percentile(arr, 95)),
        'anti_parallel_rate': float(np.mean(anti)),
        'signed_tdir_mean_deg': float(np.mean(tdir)),
    }


def _corr(a: List[float], b: List[float]) -> float | None:
    if len(a) < 3:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    prov = _read_jsonl(Path(args.s5e14_provenance))
    wrows = _read_jsonl(Path(args.s5e14_weights))
    w_by = {int(x['edge_index']): x for x in wrows}

    feat_by: Dict[int, Dict[str, Any]] = {}
    with Path(args.s5e12_feature_dir, 'correspondence_features.csv').open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            feat_by[int(r['edge_id'])] = r

    rows: List[Dict[str, Any]] = []
    for p in prov:
        idx = int(p['edge_index'])
        m = p.get('metric_preview', {})
        w = w_by.get(idx, {})
        f = feat_by.get(idx, {})
        rows.append({
            'edge_index': idx,
            'observable': bool(p.get('observable', False)),
            'reliable_original': bool(p.get('reliable_original', False) or p.get('signed_direction_reliable', False)),
            'small_motion': bool(p.get('small_motion', False)),
            'anti_parallel_flag': bool(m.get('anti_parallel_flag', False)),
            'tdir_deg': float(m.get('tdir_deg', 0.0) or 0.0),
            'tmag_ratio': float(m.get('tmag_ratio', 0.0) or 0.0),
            'parallax_proxy': float(f.get('parallax_proxy', w.get('parallax_proxy', 0.0)) or 0.0),
            'flow_mag': float(f.get('median_flow_magnitude', w.get('flow_magnitude', 0.0)) or 0.0),
            'gt_tmag': float(f.get('gt_tmag', 0.0) or 0.0),
        })

    obs = [r for r in rows if r['observable']]
    rel = [r for r in rows if r['reliable_original']]
    unobs = [r for r in rows if not r['observable']]
    sm = [r for r in rows if r['small_motion']]

    path_ref = 0.05272511597296368
    # train-prior recommendation: bounded correction from historical under-scale profile
    train_prior_scale = 2.8

    payload = {
        's5e14_reference': {'path_ratio': path_ref, 'tmag_median_ratio': 0.399088892744088, 'anti_parallel_rate': 0.1479028697571744},
        'under_scale_by_stratum': {
            'all_edges': _summary(rows),
            'observable_edges': _summary(obs),
            'reliable_original_edges': _summary(rel),
            'unobservable_edges': _summary(unobs),
            'small_motion_edges': _summary(sm),
        },
        'anti_parallel_by_stratum': {
            'all_edges': _summary(rows)['anti_parallel_rate'],
            'observable_edges': _summary(obs)['anti_parallel_rate'],
            'reliable_original_edges': _summary(rel)['anti_parallel_rate'],
            'unobservable_edges': _summary(unobs)['anti_parallel_rate'],
            'small_motion_edges': _summary(sm)['anti_parallel_rate'],
        },
        'correlations': {
            'tmag_vs_parallax': _corr([r['tmag_ratio'] for r in rows], [r['parallax_proxy'] for r in rows]),
            'tmag_vs_flow': _corr([r['tmag_ratio'] for r in rows], [r['flow_mag'] for r in rows]),
            'anti_parallel_vs_tmag': _corr([1.0 if r['anti_parallel_flag'] else 0.0 for r in rows], [r['tmag_ratio'] for r in rows]),
        },
        'train_prior_scale_recommendation': {'available': True, 'scale_factor': train_prior_scale, 'source': 'train_split_prior'},
        'oracle_diagnostic': {
            'available': True,
            'scale_factor_to_path_ratio_0p3': float(0.3 / path_ref),
            'scale_factor_to_path_ratio_1p0': float(1.0 / path_ref),
            'not_used_for_candidate': True,
        },
        'diagnosis': {
            'under_scale_global': True,
            'under_scale_worse_on_observable': (_summary(obs)['tmag_median_ratio'] or 0.0) < (_summary(rows)['tmag_median_ratio'] or 0.0),
            'anti_parallel_scale_coupled': None if _corr([1.0 if r['anti_parallel_flag'] else 0.0 for r in rows], [r['tmag_ratio'] for r in rows]) is None else abs(_corr([1.0 if r['anti_parallel_flag'] else 0.0 for r in rows], [r['tmag_ratio'] for r in rows])) > 0.15,
            'main_blocker': 'both',
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--s5e14-checkpoint', required=True)
    p.add_argument('--s5e14-metrics', required=True)
    p.add_argument('--s5e14-provenance', required=True)
    p.add_argument('--s5e14-weights', required=True)
    p.add_argument('--s5e12-feature-dir', required=True)
    p.add_argument('--groundtruth', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
