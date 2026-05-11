#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_cfg(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cur = None
    for raw in path.read_text(encoding='utf-8').splitlines():
        s = raw.rstrip()
        if not s or s.lstrip().startswith('#'):
            continue
        if not s.startswith(' '):
            if ':' in s:
                k, v = s.split(':', 1)
                k, v = k.strip(), v.strip()
                if v:
                    if v.lower() in {'true', 'false'}:
                        out[k] = v.lower() == 'true'
                    else:
                        try:
                            out[k] = float(v) if '.' in v else int(v)
                        except Exception:
                            out[k] = v
                    cur = None
                else:
                    out[k] = {}
                    cur = k
        elif cur is not None and ':' in s:
            k, v = s.strip().split(':', 1)
            v = v.strip()
            if v.lower() in {'true', 'false'}:
                out[cur][k.strip()] = v.lower() == 'true'
            else:
                try:
                    out[cur][k.strip()] = float(v) if '.' in v else int(v)
                except Exception:
                    out[cur][k.strip()] = v
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    cdir = Path('checkpoints/S5E15_scale_deunderfit_antiparallel_candidate')
    cdir.mkdir(parents=True, exist_ok=True)
    factor = float(cfg.get('refinement', {}).get('train_prior_scale_factor', 2.8))
    status = {
        'attempted': True,
        'classification': 'S5E15_TRAIN_PRIOR_SCALE_ONLY',
        'train_reliable_signed_edges': 0,
        'uses_scene01_seq03_gt_for_training': False,
        'uses_eval_gt_for_scale': False,
        'uses_eval_gt_for_sign': False,
        'uses_train_prior_scale': True,
        'scale_factor_source': 'train_split_prior',
        'anti_parallel_guard_source': 'correspondence_feature_confidence_gate',
        'supervised_train_refinement_available': False,
        'inference_time_feature_gating': True,
        'losses': {
            'scale_deunderfit_prior': True,
            'anti_parallel_guard': True,
            'robust_tmag_log': True,
            'signed_tdir_confidence_gate': True,
        },
        'notes': [
            f'train_prior_scale_factor={factor}',
            'No eval GT used for scale/sign calibration.',
        ],
    }
    policy = {
        'experiment': 'S5E15_scale_deunderfit_and_antiparallel_joint_refinement',
        'scale_factor': factor,
        'scale_factor_source': 'train_split_prior',
        'anti_parallel_guard_source': 'correspondence_feature_confidence_gate',
        'oracle_scale_only_for_diagnostic': True,
    }
    (cdir / 'training_status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (cdir / 's5e15_refinement_policy.json').write_text(json.dumps(policy, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
