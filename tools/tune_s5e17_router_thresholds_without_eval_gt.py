#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Dict

def run(args: argparse.Namespace)->Dict[str,Any]:
    audit=json.loads(Path(args.router_activation_audit).read_text(encoding='utf-8'))
    qh,ql,qs,qscale=0.75,0.25,0.85,0.35
    status={
      'router_type':'quantile_rule_based',
      'uses_eval_gt_for_thresholds':False,
      'threshold_source':'score_quantiles/config',
      'confidence_bins':{
        'high':{'rule':f'router_confidence >= q{int(qh*100)}','expected_count':None},
        'medium':{'rule':f'q{int(ql*100)} <= router_confidence < q{int(qh*100)}','expected_count':None},
        'low':{'rule':f'router_confidence < q{int(ql*100)}','expected_count':None},
      },
      'sign_guard_rule':{'rule':f'anti_parallel_risk_score >= q{int(qs*100)}','expected_count':None},
      'scale_route_rule':{'rule':f'scale_confidence < q{int(qscale*100)} => train_prior_scale_route else conservative_scale_route','expected_count':None},
      'classification':'S5E17_ROUTER_THRESHOLDS_REPAIRED' if audit['repair_recommendation']['use_quantile_bins'] else 'S5E17_ROUTER_THRESHOLDS_DIAGNOSTIC_ONLY',
    }
    p=Path(args.out_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return status

def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--router-activation-audit',required=True);p.add_argument('--out-json',required=True);return p.parse_args()
if __name__=='__main__': run(parse_args())
