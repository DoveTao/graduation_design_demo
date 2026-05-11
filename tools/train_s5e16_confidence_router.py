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
    cdir = Path('checkpoints/S5E16_confidence_calibrated_sign_scale_router')
    cdir.mkdir(parents=True, exist_ok=True)

    router_cfg = cfg.get('router', {})
    status = {
        'attempted': True,
        'classification': 'S5E16_RULE_BASED_ROUTER_READY',
        'router_type': str(router_cfg.get('router_type', 'rule_based')),
        'uses_scene01_seq03_gt_for_training': False,
        'uses_eval_gt_for_router': False,
        'uses_eval_gt_for_scale': False,
        'uses_eval_gt_for_sign': False,
        'uses_train_prior_scale': True,
        'router_threshold_source': str(router_cfg.get('threshold_source', 'config')),
        'supervised_train_router_available': False,
        'fallback_candidates': ['S5E15', 'S5E14', 'S5E13', 'S5E9'],
        'notes': [
            'Rule-based router selected due to lack of reliable train sign supervision.',
            'No eval GT used for router/scale/sign thresholds.',
        ],
    }
    policy = {
        'router_type': status['router_type'],
        'thresholds': {
            'high_confidence_min': float(router_cfg.get('high_confidence_min', 0.75)),
            'low_confidence_max': float(router_cfg.get('low_confidence_max', 0.45)),
            'anti_parallel_risk_min': float(router_cfg.get('anti_parallel_risk_min', 0.60)),
        },
        'scale': {
            'train_prior_scale_base': float(router_cfg.get('train_prior_scale_base', 2.8)),
            'conservative_scale_base': float(router_cfg.get('conservative_scale_base', 2.2)),
            'scale_factor_max': float(router_cfg.get('scale_factor_max', 5.0)),
        },
        'uses_eval_gt': False,
    }

    (cdir / 'training_status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (cdir / 'router_policy.json').write_text(json.dumps(policy, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
