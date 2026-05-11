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


def _bin(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {'count': 0, 'anti_parallel_rate': None, 'tdir_mean_deg': None, 'tmag_median_ratio': None}
    return {
        'count': len(rows),
        'anti_parallel_rate': float(np.mean([1.0 if r['anti_parallel'] else 0.0 for r in rows])),
        'tdir_mean_deg': float(np.mean([r['tdir'] for r in rows])),
        'tmag_median_ratio': float(np.median([r['tmag'] for r in rows])),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    prov = _read_jsonl(Path(args.s5e15_provenance))
    ws = _read_jsonl(Path(args.s5e15_weights))
    wby = {int(w['edge_index']): w for w in ws}

    fby: Dict[int, Dict[str, Any]] = {}
    with Path(args.s5e12_feature_dir, 'correspondence_features.csv').open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            fby[int(r['edge_id'])] = r

    rows = []
    for p in prov:
        idx = int(p['edge_index'])
        m = p.get('metric_preview', {})
        w = wby.get(idx, {})
        f = fby.get(idx, {})
        corr_conf = float(min(1.0, max(0.0, 0.65 * float(f.get('inlier_ratio', 0.0) or 0.0) + 0.35 * min(1.0, float(f.get('parallax_proxy', 0.0) or 0.0)))))
        dir_conf = float(min(1.0, max(0.0, 1.0 - float(m.get('tdir_abs_deg', 180.0) or 180.0) / 180.0)))
        anti_risk = float(min(1.0, max(0.0, (float(f.get('flow_angle_dispersion', 0.0) or 0.0) / 120.0 + (1.0 - corr_conf)) * 0.5)))
        scale_conf = float(min(1.0, max(0.0, 1.0 - abs(float(m.get('tmag_ratio', 0.0) or 0.0) - 1.0) / 3.0)))
        rows.append({
            'edge_index': idx,
            'corr_conf': corr_conf,
            'dir_conf': dir_conf,
            'anti_risk': anti_risk,
            'scale_conf': scale_conf,
            'anti_parallel': bool(m.get('anti_parallel_flag', False)),
            'tdir': float(m.get('tdir_deg', 0.0) or 0.0),
            'tmag': float(m.get('tmag_ratio', 0.0) or 0.0),
            'observable': bool(p.get('observable', False)),
            'low_parallax': bool(p.get('low_parallax', False)),
            'near_static': bool(p.get('near_static', False)),
            'small_motion': bool(p.get('small_motion', False)),
        })

    hi = [r for r in rows if r['corr_conf'] >= 0.75]
    md = [r for r in rows if 0.45 <= r['corr_conf'] < 0.75]
    lo = [r for r in rows if r['corr_conf'] < 0.45]

    out = {
        's5e15_reference': {
            'path_ratio': 0.1572,
            'tmag_median_ratio': 1.2685,
            'anti_parallel_rate': 0.1479,
            'tdir_mean_deg': 50.3530,
            'sim3_ate': 3.9682,
        },
        'confidence_bins': {
            'high_confidence': _bin(hi),
            'medium_confidence': _bin(md),
            'low_confidence': _bin(lo),
        },
        'risk_bins': {
            'anti_parallel_high_risk': _bin([r for r in rows if r['anti_risk'] >= 0.60]),
            'scale_high_risk': _bin([r for r in rows if r['scale_conf'] < 0.45]),
            'near_static': _bin([r for r in rows if r['near_static']]),
            'low_parallax': _bin([r for r in rows if r['low_parallax']]),
        },
        'router_design_recommendation': {
            'use_correspondence_direction_when': 'corr_conf >= 0.75 and anti_risk < 0.60',
            'fallback_direction_when': 'corr_conf < 0.45 or near_static or low_parallax or anti_risk >= 0.60',
            'use_train_prior_scale_when': 'scale_conf >= 0.45',
            'scale_guard_when': 'scale_conf < 0.45 or small_motion',
            'anti_parallel_guard_when': 'anti_risk >= 0.60',
        },
        'diagnosis': {
            'router_feasible': True,
            'main_router_signal': 'correspondence_confidence',
            'main_remaining_risk': 'anti_parallel',
        },
    }

    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--s5e15-checkpoint', required=True)
    p.add_argument('--s5e15-metrics', required=True)
    p.add_argument('--s5e15-provenance', required=True)
    p.add_argument('--s5e15-weights', required=True)
    p.add_argument('--s5e14-checkpoint', required=True)
    p.add_argument('--s5e13-checkpoint', required=True)
    p.add_argument('--s5e12-feature-dir', required=True)
    p.add_argument('--groundtruth', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
