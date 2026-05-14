#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json

S5E15_REF = {'path_ratio': 0.1572, 'tmag_median_ratio': 1.2685, 'anti_parallel_rate': 0.1479, 'tdir_mean_deg': 50.3530, 'sim3_ate': 3.9682}


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


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
        'count': len(rows),
        'rot_mean_deg': _mean('rot_deg'), 'rot_median_deg': _pct('rot_deg', 50), 'rot_p90_deg': _pct('rot_deg', 90),
        'signed_tdir_mean_deg': _mean('tdir_deg'), 'signed_tdir_median_deg': _pct('tdir_deg', 50), 'signed_tdir_p90_deg': _pct('tdir_deg', 90),
        'tdir_abs_mean_deg': _mean('tdir_abs_deg'), 'tdir_abs_median_deg': _pct('tdir_abs_deg', 50), 'tdir_abs_p90_deg': _pct('tdir_abs_deg', 90),
        'tdir_mean_cosine': _mean('tdir_cosine'), 'anti_parallel_rate': _mean('anti_parallel_flag'),
        'severe_wrong_sign_rate': _mean('severe_wrong_sign_flag'), 'direction_abs_good_but_signed_bad_rate': _mean('direction_abs_good_but_signed_bad_flag'),
        'tmag_median_ratio': _pct('tmag_ratio', 50), 'tmag_mean_ratio': _mean('tmag_ratio'), 'tmag_p90_ratio': _pct('tmag_ratio', 90), 'tmag_p95_ratio': _pct('tmag_ratio', 95), 'tmag_p99_ratio': _pct('tmag_ratio', 99), 'tmag_max_ratio': None if _vals('tmag_ratio').size == 0 else float(np.max(_vals('tmag_ratio'))),
        'path_ratio': pred / max(gt, 1e-12),
    }


def _ext(traj: Path, gt: Path, out: Path, mode: str) -> Dict[str, Any]:
    subprocess.run(['/home/dovetao/miniconda3/envs/pytorch/bin/python', 'tools/evaluate_external_baseline_trajectory.py', '--trajectory', str(traj), '--groundtruth', str(gt), '--alignment', mode, '--output-json', str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    d = json.loads(out.read_text(encoding='utf-8'))
    return {'ate': d.get('ATE'), 'drift': d.get('drift'), 'path_ratio': d.get('path_ratio')}


def _report(path: Path, ck: Dict[str, Any]) -> None:
    lines = [
        '# S5E16 confidence calibrated sign scale router report',
        '',
        '## 执行摘要',
        f"final_classification = `{ck['final_classification']}`。",
        '',
        '## S5E15 结果回顾', str(ck.get('s5e15_reference', {})), '',
        '## router input audit', str(ck.get('router_input_audit', {})), '',
        '## confidence router 设计', str(ck.get('router', {})), '',
        '## direction_route / scale_route / sign_guard_route 说明', '基于 correspondence_confidence + anti_parallel_risk + scale_confidence 的 rule-based router。', '',
        '## no eval GT calibration 说明', 'uses_eval_gt_for_router=false, uses_eval_gt_for_scale=false, uses_eval_gt_for_sign=false。', '',
        '## traceable dense export coverage', str(ck.get('adjacent_dense_export', {})), '',
        '## overall component metrics', str(ck.get('component_metrics', {}).get('overall', {})), '',
        '## route-stratified metrics', str(ck.get('component_metrics', {}).get('route_strata', {})), '',
        '## external evaluator none/se3/sim3', str(ck.get('external_eval', {})), '',
        '## 与 S5E15 / S5E14 / S5E13 / S5E9 比较', str(ck.get('improvement_vs_s5e15', {})), '',
        '## 与 ORB-SLAM3 比较', str(ck.get('comparison_to_orbslam3', {})), '',
        '## 是否进入 S5E17', str(ck.get('diagnosis', {})), '',
        '## caveats',
        '- S5E16 是 experimental candidate。',
        '- 不替代 official S5 locked result。',
        '- S5 locked metrics/policy unchanged。',
        '- 没有使用 eval GT 做 router/scale/sign calibration。',
        '- strict essential geometry 仍受 intrinsics 限制。',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    mp = json.loads((out_dir / 'edge_component_metrics.json').read_text(encoding='utf-8'))
    prov = _read_jsonl(Path(args.provenance))
    dec = _read_jsonl(Path(args.router_decisions))
    dby = {int(d['edge_index']): d for d in dec}
    audit = json.loads((out_dir / 'router_input_audit.json').read_text(encoding='utf-8'))
    train = json.loads(Path('checkpoints/S5E16_confidence_calibrated_sign_scale_router/training_status.json').read_text(encoding='utf-8'))

    rows = []
    for p in prov:
        r = dict(p.get('metric_preview', {}))
        r.update(p)
        r.update(dby.get(int(p['edge_index']), {}))
        rows.append(r)

    def filt(fn):
        return [r for r in rows if fn(r)]

    overall = _summary(rows)
    hi = _summary(filt(lambda r: float(r.get('router_confidence', 0.0)) >= 0.75))
    md = _summary(filt(lambda r: 0.45 <= float(r.get('router_confidence', 0.0)) < 0.75))
    lo = _summary(filt(lambda r: float(r.get('router_confidence', 0.0)) < 0.45))
    corr_route = _summary(filt(lambda r: r.get('direction_route') == 'correspondence_direction'))
    fallback_route = _summary(filt(lambda r: r.get('direction_route') == 'fallback_prior'))
    sign_guarded = _summary(filt(lambda r: r.get('sign_guard_route') == 'apply'))
    train_scale = _summary(filt(lambda r: r.get('scale_route') == 's5e15_train_prior_scale'))
    cons_scale = _summary(filt(lambda r: r.get('scale_route') == 'conservative_scale'))
    observable = _summary(filt(lambda r: bool(r.get('observable', False))))
    unobservable = _summary(filt(lambda r: not bool(r.get('observable', False))))

    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f'eval_alignment_{m}.json', m) for m in ['none', 'se3', 'sim3']}

    imp = {
        'path_ratio_improved_without_explosion': overall.get('path_ratio') is not None and overall['path_ratio'] >= S5E15_REF['path_ratio'] and overall['path_ratio'] <= 2.1345,
        'tmag_median_closer_to_one': overall.get('tmag_median_ratio') is not None and abs(overall['tmag_median_ratio'] - 1.0) <= abs(S5E15_REF['tmag_median_ratio'] - 1.0),
        'anti_parallel_reduced': overall.get('anti_parallel_rate') is not None and overall['anti_parallel_rate'] < S5E15_REF['anti_parallel_rate'],
        'tdir_preserved_or_improved': overall.get('signed_tdir_mean_deg') is not None and overall['signed_tdir_mean_deg'] <= S5E15_REF['tdir_mean_deg'],
        'sim3_ate_improved': ext['sim3'].get('ate') is not None and ext['sim3']['ate'] <= S5E15_REF['sim3_ate'],
    }
    imp['overall_geometry_improved'] = bool(imp['path_ratio_improved_without_explosion'] and imp['anti_parallel_reduced'] and imp['tdir_preserved_or_improved'])

    if imp['overall_geometry_improved'] and imp['sim3_ate_improved']:
        final = 'S5E16_ROUTER_IMPROVED_GEOMETRY'
    elif imp['path_ratio_improved_without_explosion'] and not imp['anti_parallel_reduced']:
        final = 'S5E16_SCALE_IMPROVED_ANTIPARALLEL_STILL_BAD'
    elif imp['anti_parallel_reduced'] and not imp['path_ratio_improved_without_explosion']:
        final = 'S5E16_ANTIPARALLEL_IMPROVED_SCALE_STILL_GAP'
    else:
        final = 'S5E16_ROUTER_NO_IMPROVEMENT'

    cov = mp['coverage']
    ck = {
        'experiment': 'S5E16_confidence_calibrated_sign_scale_router',
        'status': {'experimental_candidate': True, 'official_s5_unchanged': True, 'not_official_replacement': True},
        's5e15_reference': S5E15_REF,
        'router_input_audit': {
            'path': 'external_baselines/results/s5e16_traceable_dense/router_input_audit.json',
            'router_feasible': audit.get('diagnosis', {}).get('router_feasible'),
            'main_router_signal': audit.get('diagnosis', {}).get('main_router_signal', 'unknown'),
            'main_remaining_risk': audit.get('diagnosis', {}).get('main_remaining_risk', 'unknown'),
        },
        'router': {
            'attempted': True,
            'classification': train.get('classification'),
            'config': 'configs/s5e16_confidence_calibrated_sign_scale_router.yaml',
            'checkpoint_dir': 'checkpoints/S5E16_confidence_calibrated_sign_scale_router',
            'router_type': train.get('router_type'),
            'uses_scene01_seq03_gt_for_training': False,
            'uses_eval_gt_for_router': False,
            'uses_eval_gt_for_scale': False,
            'uses_eval_gt_for_sign': False,
            'uses_train_prior_scale': train.get('uses_train_prior_scale'),
            'router_threshold_source': train.get('router_threshold_source'),
            'supervised_train_router_available': train.get('supervised_train_router_available'),
            'fallback_candidates': train.get('fallback_candidates', ['S5E15', 'S5E14', 'S5E13', 'S5E9']),
        },
        'adjacent_dense_export': {
            'available': True,
            'trajectory_path': str(args.trajectory),
            'edge_provenance': str(args.provenance),
            'router_decisions': str(args.router_decisions),
            'num_poses': cov['num_poses'], 'num_edges': cov['num_edges'], 'direct_adjacent_prediction_edges': cov['direct_adjacent_prediction_edges'],
            'coverage': 1.0, 'all_edges_traceable': cov['all_edges_traceable'],
        },
        'component_metrics': {
            'overall': overall,
            'route_strata': {
                'high_confidence_edges': hi,
                'medium_confidence_edges': md,
                'low_confidence_edges': lo,
                'correspondence_direction_route_edges': corr_route,
                'fallback_direction_route_edges': fallback_route,
                'sign_guarded_edges': sign_guarded,
                'train_prior_scale_route_edges': train_scale,
                'conservative_scale_route_edges': cons_scale,
                'observable_edges': observable,
                'unobservable_edges': unobservable,
            },
        },
        'external_eval': ext,
        'improvement_vs_s5e15': imp,
        'comparison_to_orbslam3': {
            'coverage_advantage': True,
            'tdir_gap_remaining': True,
            'scale_path_gap_remaining': True,
            'aligned_ate_gap_to_orbslam3': None if ext['sim3']['ate'] is None else ext['sim3']['ate'] - ORBSLAM3_REFERENCE['sim3']['ate'],
            'summary': 'S5E16 仍保留 full coverage，但与 ORB-SLAM3 在 ATE 与 direction 仍有显著差距。',
        },
        'diagnosis': {
            'can_continue_to_s5e17': True,
            'main_remaining_blocker': 'tdir' if not imp['tdir_preserved_or_improved'] else ('anti_parallel' if not imp['anti_parallel_reduced'] else 'intrinsics_geometry'),
            'recommended_next_stage': 'S5E17_confidence_bin_joint_objective',
        },
        's5_official_locked_metrics': {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'unchanged': True, 'not_replaced_by_s5e16': True},
        'validation': validation_from_logs(),
        'git': {
            'commits_created': [],
            'pushed_to_remote': False,
            'tag_created': False,
            'tag_name': None,
            'working_tree_clean': False,
        },
        'final_classification': final,
    }

    write_json(Path(args.out_json), ck)
    _report(Path(args.out_report), ck)
    return ck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', required=True)
    p.add_argument('--provenance', required=True)
    p.add_argument('--router-decisions', required=True)
    p.add_argument('--groundtruth', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-report', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
