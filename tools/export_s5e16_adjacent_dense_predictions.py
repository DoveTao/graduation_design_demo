#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, vector_angle_deg, write_json


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj['R'].T @ gi['R'], gj['R'].T @ (gi['t'] - gj['t'])


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    tdir = vector_angle_deg(t, tg, absolute=False)
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        'rot_deg': angle_deg_from_rot(R @ Rg.T),
        'tdir_deg': tdir,
        'tdir_abs_deg': tdir_abs,
        'tdir_cosine': cos,
        'anti_parallel_flag': bool(cos is not None and cos < 0.0),
        'severe_wrong_sign_flag': bool(tdir is not None and tdir > 120.0),
        'direction_abs_good_but_signed_bad_flag': bool(tdir is not None and tdir_abs is not None and tdir > 120.0 and tdir_abs < 45.0),
        'tmag_ratio': float(np.linalg.norm(t) / max(np.linalg.norm(tg), 1.0e-12)),
        'pred_step_length': float(np.linalg.norm(t)),
        'gt_step_length': float(np.linalg.norm(tg)),
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(k: str):
        return np.asarray([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
    def _mean(k: str):
        v = _vals(k)
        return None if v.size == 0 else float(np.mean(v))
    def _pct(k: str, q: float):
        v = _vals(k)
        return None if v.size == 0 else float(np.percentile(v, q))
    pred = sum(float(r.get('pred_step_length') or 0.0) for r in rows)
    gt = sum(float(r.get('gt_step_length') or 0.0) for r in rows)
    return {
        'rot_mean_deg': _mean('rot_deg'), 'rot_median_deg': _pct('rot_deg', 50), 'rot_p90_deg': _pct('rot_deg', 90),
        'tdir_mean_deg': _mean('tdir_deg'), 'tdir_median_deg': _pct('tdir_deg', 50), 'tdir_p90_deg': _pct('tdir_deg', 90),
        'tdir_abs_mean_deg': _mean('tdir_abs_deg'), 'tdir_abs_median_deg': _pct('tdir_abs_deg', 50), 'tdir_abs_p90_deg': _pct('tdir_abs_deg', 90),
        'tdir_mean_cosine': _mean('tdir_cosine'), 'anti_parallel_rate': _mean('anti_parallel_flag'),
        'severe_wrong_sign_rate': _mean('severe_wrong_sign_flag'), 'direction_abs_good_but_signed_bad_rate': _mean('direction_abs_good_but_signed_bad_flag'),
        'tmag_median_ratio': _pct('tmag_ratio', 50), 'tmag_mean_ratio': _mean('tmag_ratio'), 'tmag_p90_ratio': _pct('tmag_ratio', 90), 'tmag_p95_ratio': _pct('tmag_ratio', 95), 'tmag_p99_ratio': _pct('tmag_ratio', 99), 'tmag_max_ratio': None if _vals('tmag_ratio').size == 0 else float(np.max(_vals('tmag_ratio'))),
        'path_ratio': pred / max(gt, 1e-12),
    }


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    Rw = np.eye(3)
    tw = np.zeros(3)
    lines = []
    qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
    lines.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for i, (R_BA, t_BA) in enumerate(rels):
        Rb = Rw @ R_BA.T
        tb = tw - Rb @ t_BA
        Rw, tw = Rb, tb
        qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
        lines.append(f"{timestamps[i+1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s15 = _read_jsonl(Path('external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl'))
    s14 = _read_jsonl(Path('external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl'))
    s13 = _read_jsonl(Path('external_baselines/results/s5e13_traceable_dense/edge_provenance.jsonl'))
    s14_by = {int(x['edge_index']): x for x in s14}
    s13_by = {int(x['edge_index']): x for x in s13}

    policy = json.loads((Path(args.candidate) / 'router_policy.json').read_text(encoding='utf-8'))
    th = policy['thresholds']
    sc = policy['scale']

    feat_by: Dict[int, Dict[str, Any]] = {}
    with Path(args.s5e12_feature_dir, 'correspondence_features.csv').open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            feat_by[int(r['edge_id'])] = r

    gt = read_tum(Path(args.groundtruth))
    ts = read_timestamps(Path(args.timestamps))

    rels = []
    prov_out = []
    decision_out = []
    mrows = []

    for p in s15:
        idx = int(p['edge_index'])
        f = feat_by.get(idx, {})
        p14 = s14_by[idx]
        p13 = s13_by[idx]
        dir15 = np.asarray(p['translation_direction'], dtype=np.float64)
        dir14 = np.asarray(p14['translation_direction'], dtype=np.float64)
        dir13 = np.asarray(p13['translation_direction'], dtype=np.float64)
        for d in (dir15, dir14, dir13):
            n = np.linalg.norm(d)
            if n > 1e-12:
                d /= n

        inlier = float(f.get('inlier_ratio', 0.0) or 0.0)
        parallax = float(f.get('parallax_proxy', 0.0) or 0.0)
        flow_disp = float(f.get('flow_angle_dispersion', 0.0) or 0.0)
        corr_conf = float(min(1.0, max(0.0, 0.65 * inlier + 0.35 * min(1.0, parallax))))
        anti_risk = float(min(1.0, max(0.0, (flow_disp / 120.0 + (1.0 - corr_conf)) * 0.5)))
        scale_conf = float(min(1.0, max(0.0, 1.0 - abs(float(p.get('translation_magnitude', 0.0)) - float(p14.get('translation_magnitude', 0.0))) / max(float(p.get('translation_magnitude', 1.0)), 1e-6))))
        router_conf = float(min(1.0, max(0.0, 0.5 * corr_conf + 0.3 * (1.0 - anti_risk) + 0.2 * scale_conf)))

        low_parallax = bool(p.get('low_parallax', False))
        near_static = bool(p.get('near_static', False))
        small_motion = bool(p.get('small_motion', False))

        if corr_conf >= th['high_confidence_min'] and anti_risk < th['anti_parallel_risk_min'] and not near_static:
            direction_route = 'correspondence_direction'
            dir_use = dir15
        elif corr_conf < th['low_confidence_max'] or low_parallax or near_static:
            direction_route = 'fallback_prior'
            dir_use = dir13
        else:
            direction_route = 's5e14_direction'
            dir_use = dir14

        sign_guard = 'apply' if anti_risk >= th['anti_parallel_risk_min'] else 'no_apply'
        if sign_guard == 'apply':
            # conservative sign guard: choose direction closer to prior consensus
            cands = [dir_use, dir14, dir13]
            scores = [float(np.dot(c, dir14) + np.dot(c, dir13)) for c in cands]
            dir_use = cands[int(np.argmax(scores))]

        if scale_conf >= 0.45 and not small_motion:
            scale_route = 's5e15_train_prior_scale'
            sf = min(sc['scale_factor_max'], sc['train_prior_scale_base'])
        else:
            scale_route = 'conservative_scale'
            sf = min(sc['scale_factor_max'], sc['conservative_scale_base'])

        mag = float(p['translation_magnitude']) * sf / max(sc['train_prior_scale_base'], 1e-6)

        R = np.asarray(p['rotation']['value'], dtype=np.float64)
        t = dir_use * mag
        rels.append((R, t))

        mm = _metric(R, t, _gt_rel(gt, float(p['timestamp_i']), float(p['timestamp_j'])))
        mrows.append(mm)

        decision = {
            'edge_index': idx,
            'direction_route': direction_route,
            'scale_route': scale_route,
            'sign_guard_route': sign_guard,
            'router_confidence': router_conf,
            'reason': f'corr_conf={corr_conf:.3f},anti_risk={anti_risk:.3f},scale_conf={scale_conf:.3f}',
            'uses_eval_gt': False,
        }
        decision_out.append(decision)

        prov_out.append({
            'edge_index': idx,
            'timestamp_i': float(p['timestamp_i']),
            'timestamp_j': float(p['timestamp_j']),
            'source_type': 'direct_adjacent_prediction',
            'source_model': 'S5E16_confidence_calibrated_sign_scale_router',
            'uses_gt_for_prediction': False,
            'uses_eval_gt_for_router': False,
            'uses_eval_gt_for_scale': False,
            'uses_eval_gt_for_sign': False,
            'router_type': policy['router_type'],
            'direction_route': direction_route,
            'scale_route': scale_route,
            'sign_guard_route': sign_guard,
            'router_confidence': router_conf,
            'correspondence_confidence': corr_conf,
            'anti_parallel_risk_score': anti_risk,
            'scale_confidence': scale_conf,
            'scale_factor': sf,
            'scale_factor_source': 'train_split_prior',
            'observable': bool(p.get('observable', False)),
            'reliable_original': bool(p.get('reliable_original', False)),
            'low_parallax': low_parallax,
            'near_static': near_static,
            'small_motion': small_motion,
            'rotation': {'representation': 'matrix', 'value': R.tolist()},
            'translation': {'frame': 'B/local', 'value': t.tolist()},
            'translation_direction': dir_use.tolist(),
            'translation_magnitude': mag,
            'metric_preview': mm,
            'notes': ['router_mixed_candidate', 'no eval GT calibration'],
        })

    _write_tum(Path(args.out_tum), ts, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in prov_out) + '\n', encoding='utf-8')
    Path(args.out_router_decisions).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in decision_out) + '\n', encoding='utf-8')

    payload = {
        'experiment': 'S5E16_confidence_calibrated_sign_scale_router',
        'coverage': {'num_poses': len(ts), 'num_edges': len(prov_out), 'direct_adjacent_prediction_edges': len(prov_out), 'all_edges_traceable': True},
        'component_metrics': _summary(mrows),
        'metrics_rows': mrows,
        'trajectory_path': str(args.out_tum),
    }
    write_json(Path(args.out_metrics), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--seq', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--s5e15-checkpoint', required=True)
    p.add_argument('--s5e14-checkpoint', required=True)
    p.add_argument('--s5e13-checkpoint', required=True)
    p.add_argument('--s5e12-feature-dir', required=True)
    p.add_argument('--timestamps', required=True)
    p.add_argument('--groundtruth', required=True)
    p.add_argument('--out-tum', required=True)
    p.add_argument('--out-provenance', required=True)
    p.add_argument('--out-metrics', required=True)
    p.add_argument('--out-router-decisions', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
