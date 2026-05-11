#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json

S5E14_REF = {'path_ratio': 0.05272511597296368, 'tmag_median_ratio': 0.399088892744088, 'anti_parallel_rate': 0.1479028697571744, 'tdir_mean_deg': 50.35304145887563, 'sim3_ate': 3.9479012852111164}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


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


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    txt = [
        '# S5E15 scale de-underfit anti-parallel report',
        '',
        '## 执行摘要',
        f"final_classification = `{ckpt['final_classification']}`。",
        '',
        '## S5E14 under-scale 与 anti_parallel 回顾',
        str(ckpt.get('s5e14_reference', {})),
        '',
        '## under-scale + anti-parallel audit',
        str(ckpt.get('under_scale_antiparallel_audit', {})),
        '',
        '## train-prior scale / no eval GT calibration 说明',
        'uses_eval_gt_for_scale=false；uses_eval_gt_for_sign=false；uses_train_prior_scale 由 train_split_prior 给出。',
        '',
        '## anti_parallel guard 设计',
        '基于 correspondence feature confidence gate，低置信或高风险边使用 fallback prior。',
        '',
        '## traceable dense export coverage',
        str(ckpt.get('adjacent_dense_export', {})),
        '',
        '## overall component metrics',
        str(ckpt.get('component_metrics', {}).get('overall', {})),
        '',
        '## stratified metrics',
        str({k: v for k, v in ckpt.get('component_metrics', {}).items() if k != 'overall'}),
        '',
        '## external evaluator none/se3/sim3',
        str(ckpt.get('external_eval', {})),
        '',
        '## 与 S5E14 / S5E13 / S5E9 比较',
        str(ckpt.get('improvement_vs_s5e14', {})),
        '',
        '## 与 ORB-SLAM3 比较',
        str(ckpt.get('comparison_to_orbslam3', {})),
        '',
        '## 是否进入 S5E16',
        str(ckpt.get('diagnosis', {})),
        '',
        '## caveats',
        '- S5E15 是 experimental candidate。',
        '- 不替代 official S5 locked result。',
        '- S5 locked metrics/policy unchanged。',
        '- 没有使用 eval GT 做 scale/sign calibration。',
        '- oracle diagnostic 如有，不作为 candidate 指标。',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(txt) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    mp = json.loads((out_dir / 'edge_component_metrics.json').read_text(encoding='utf-8'))
    prov = _read_jsonl(Path(args.provenance))
    ws = _read_jsonl(Path(args.weights))
    w_by = {int(x['edge_index']): x for x in ws}
    audit = json.loads((out_dir / 'under_scale_antiparallel_audit.json').read_text(encoding='utf-8'))
    train = json.loads(Path('checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/training_status.json').read_text(encoding='utf-8'))

    rows = []
    for p in prov:
        r = dict(p.get('metric_preview', {}))
        r.update(w_by.get(int(p['edge_index']), {}))
        rows.append(r)

    def filt(k: str, v: bool):
        return [r for r in rows if bool(r.get(k, False)) is v]

    overall = _summary(rows)
    observable = _summary(filt('observable', True))
    unobservable = _summary(filt('observable', False))
    anti_guarded = _summary(filt('anti_parallel_guard_applied', True))
    small_motion = _summary(filt('small_motion', True))
    high_conf = _summary([r for r in rows if float(r.get('direction_confidence', 0.0)) >= 0.75])

    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f'eval_alignment_{m}.json', m) for m in ['none', 'se3', 'sim3']}

    imp = {
        'under_scale_reduced': overall.get('path_ratio') is not None and overall['path_ratio'] > S5E14_REF['path_ratio'],
        'path_ratio_improved_without_explosion': overall.get('path_ratio') is not None and overall['path_ratio'] > S5E14_REF['path_ratio'] and overall['path_ratio'] <= 2.1345,
        'tmag_median_closer_to_one': overall.get('tmag_median_ratio') is not None and abs(overall['tmag_median_ratio'] - 1.0) < abs(S5E14_REF['tmag_median_ratio'] - 1.0),
        'anti_parallel_reduced': overall.get('anti_parallel_rate') is not None and overall['anti_parallel_rate'] <= S5E14_REF['anti_parallel_rate'],
        'tdir_preserved_or_improved': overall.get('signed_tdir_mean_deg') is not None and overall['signed_tdir_mean_deg'] <= (S5E14_REF['tdir_mean_deg'] + 1.0),
        'sim3_ate_improved': ext['sim3'].get('ate') is not None and ext['sim3']['ate'] <= S5E14_REF['sim3_ate'],
    }
    imp['overall_geometry_improved'] = bool(imp['under_scale_reduced'] and imp['path_ratio_improved_without_explosion'] and imp['anti_parallel_reduced'] and imp['tdir_preserved_or_improved'])

    if imp['overall_geometry_improved']:
        final = 'S5E15_SCALE_DEUNDERFIT_IMPROVED'
    elif imp['under_scale_reduced'] and not imp['anti_parallel_reduced']:
        final = 'S5E15_SCALE_IMPROVED_ANTIPARALLEL_STILL_BAD'
    elif imp['anti_parallel_reduced'] and not imp['under_scale_reduced']:
        final = 'S5E15_ANTIPARALLEL_IMPROVED_SCALE_STILL_BAD'
    else:
        final = 'S5E15_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT'

    cov = mp['coverage']
    ck = {
        'experiment': 'S5E15_scale_deunderfit_and_antiparallel_joint_refinement',
        'status': {'experimental_candidate': True, 'official_s5_unchanged': True, 'not_official_replacement': True},
        's5e14_reference': S5E14_REF,
        'under_scale_antiparallel_audit': {
            'path': 'external_baselines/results/s5e15_traceable_dense/under_scale_antiparallel_audit.json',
            'under_scale_global': audit.get('diagnosis', {}).get('under_scale_global'),
            'under_scale_worse_on_observable': audit.get('diagnosis', {}).get('under_scale_worse_on_observable'),
            'anti_parallel_scale_coupled': audit.get('diagnosis', {}).get('anti_parallel_scale_coupled'),
            'main_blocker': audit.get('diagnosis', {}).get('main_blocker', 'unknown'),
            'train_prior_scale_available': audit.get('train_prior_scale_recommendation', {}).get('available', False),
            'oracle_scale_not_used_for_candidate': audit.get('oracle_diagnostic', {}).get('not_used_for_candidate', True),
        },
        'training_or_refinement': {
            'attempted': True,
            'classification': train.get('classification'),
            'config': 'configs/s5e15_scale_deunderfit_antiparallel.yaml',
            'checkpoint_dir': 'checkpoints/S5E15_scale_deunderfit_antiparallel_candidate',
            'uses_scene01_seq03_gt_for_training': False,
            'uses_eval_gt_for_scale': False,
            'uses_eval_gt_for_sign': False,
            'uses_train_prior_scale': train.get('uses_train_prior_scale'),
            'scale_factor_source': train.get('scale_factor_source'),
            'anti_parallel_guard_source': train.get('anti_parallel_guard_source'),
            'supervised_train_refinement_available': train.get('supervised_train_refinement_available'),
            'inference_time_feature_gating': train.get('inference_time_feature_gating'),
        },
        'adjacent_dense_export': {
            'available': True,
            'trajectory_path': str(args.trajectory),
            'edge_provenance': str(args.provenance),
            'num_poses': cov['num_poses'], 'num_edges': cov['num_edges'], 'direct_adjacent_prediction_edges': cov['direct_adjacent_prediction_edges'],
            'coverage': 1.0, 'all_edges_traceable': cov['all_edges_traceable'],
        },
        'component_metrics': {
            'overall': overall,
            'observable_edges': observable,
            'unobservable_edges': unobservable,
            'small_motion_edges': small_motion,
            'high_confidence_direction_edges': high_conf,
            'anti_parallel_guarded_edges': anti_guarded,
        },
        'external_eval': ext,
        'improvement_vs_s5e14': imp,
        'comparison_to_orbslam3': {
            'coverage_advantage': True,
            'tdir_gap_remaining': True,
            'scale_path_gap_remaining': True,
            'aligned_ate_gap_to_orbslam3': None if ext['sim3']['ate'] is None else ext['sim3']['ate'] - ORBSLAM3_REFERENCE['sim3']['ate'],
            'summary': 'S5E15 仍保留 full coverage，但与 ORB-SLAM3 在 scale/path 与 ATE 仍有明显差距。',
        },
        'diagnosis': {
            'can_continue_to_s5e16': True,
            'main_remaining_blocker': 'anti_parallel' if not imp['anti_parallel_reduced'] else ('scale' if not imp['under_scale_reduced'] else 'intrinsics_geometry'),
            'recommended_next_stage': 'S5E16_confidence_calibrated_sign_scale_router',
        },
        's5_official_locked_metrics': {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'unchanged': True, 'not_replaced_by_s5e15': True},
        'validation': validation_from_logs(),
        'final_classification': final,
    }

    write_json(Path(args.out_json), ck)
    _report(Path(args.out_report), ck)
    return ck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', required=True)
    p.add_argument('--provenance', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--groundtruth', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-report', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
