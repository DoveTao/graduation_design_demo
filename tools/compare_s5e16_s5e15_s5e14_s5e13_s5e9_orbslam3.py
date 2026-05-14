#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, write_json


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _pick(ck: Dict[str, Any], name: str) -> Dict[str, Any]:
    comp = ck.get('component_metrics', {})
    ov = comp.get('overall', comp)
    ext = ck.get('external_eval', {})
    return {
        'name': name,
        'coverage': ck.get('adjacent_dense_export', {}).get('coverage'),
        'rot_mean_deg': ov.get('rot_mean_deg'),
        'signed_tdir_mean_deg': ov.get('signed_tdir_mean_deg', ov.get('tdir_mean_deg')),
        'anti_parallel_rate': ov.get('anti_parallel_rate'),
        'tmag_median_ratio': ov.get('tmag_median_ratio'),
        'tmag_p95_ratio': ov.get('tmag_p95_ratio'),
        'none_ate': ext.get('none', {}).get('ate'),
        'se3_ate': ext.get('se3', {}).get('ate'),
        'sim3_ate': ext.get('sim3', {}).get('ate'),
        'path_ratio': ov.get('path_ratio'),
        'caveat': ck.get('final_classification'),
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = ['# S5E16 comparison', '', '## 执行摘要', payload['summary'], '', '## 比较表']
    for r in payload['rows']:
        lines.append(str(r))
    lines += ['', '## 解释', '- S5E16 关注 confidence router 是否降低 anti_parallel 并维持 scale/path 改善。', '- 不能将 S5E16 视作 official S5 替代。']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s16 = _load(Path(args.s5e16_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    s14 = _load(Path(args.s5e14_checkpoint))
    s13 = _load(Path(args.s5e13_checkpoint))
    s9 = _load(Path('checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json'))

    rows: List[Dict[str, Any]] = [
        {'name': 'S5 official locked result', 'coverage': None, 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'none_ate': None, 'se3_ate': 7.352288, 'sim3_ate': 7.352288, 'path_ratio': 0.932379, 'caveat': 'official_locked'},
        _pick(s9, 'S5E9 stable scale/path baseline'),
        _pick(s13, 'S5E13'),
        _pick(s14, 'S5E14'),
        _pick(s15, 'S5E15'),
        _pick(s16, 'S5E16'),
        {'name': 'ORB-SLAM3 external baseline', 'coverage': ORBSLAM3_REFERENCE['tracking_success_rate'], 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'none_ate': ORBSLAM3_REFERENCE['none']['ate'], 'se3_ate': ORBSLAM3_REFERENCE['se3']['ate'], 'sim3_ate': ORBSLAM3_REFERENCE['sim3']['ate'], 'path_ratio': ORBSLAM3_REFERENCE['se3']['path_ratio'], 'caveat': 'external_baseline'},
    ]
    payload = {'experiment': 'S5E16_confidence_calibrated_sign_scale_router', 'rows': rows, 'summary': 'S5E16 通过 confidence-calibrated route mixing 尝试在不爆炸的前提下改善 scale/path 与 anti_parallel。'}
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--s5e16-checkpoint', required=True)
    p.add_argument('--s5e15-checkpoint', required=True)
    p.add_argument('--s5e14-checkpoint', required=True)
    p.add_argument('--s5e13-checkpoint', required=True)
    p.add_argument('--orbslam3-eval-dir', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-report', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
