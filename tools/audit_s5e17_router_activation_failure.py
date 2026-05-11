#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def _dist(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {k: None for k in ['min','max','mean','median','p10','p25','p50','p75','p90']}
    a=np.asarray(vals,dtype=np.float64)
    return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'p50':float(np.percentile(a,50)),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90))}

def run(args: argparse.Namespace)->Dict[str,Any]:
    dec=_read_jsonl(Path(args.s5e16_router_decisions))
    prov=_read_jsonl(Path('external_baselines/results/s5e16_traceable_dense/edge_provenance.jsonl'))
    high=sum(1 for r in dec if float(r.get('router_confidence',0))>=0.75)
    low=sum(1 for r in dec if float(r.get('router_confidence',0))<0.45)
    sign=sum(1 for r in dec if r.get('sign_guard_route')=='apply')
    fallback=sum(1 for r in dec if r.get('direction_route')=='fallback_prior')
    train_scale=sum(1 for r in dec if r.get('scale_route')=='s5e15_train_prior_scale')

    rc=[float(r.get('router_confidence',0.0)) for r in dec]
    cc=[float(r.get('correspondence_confidence',0.0)) for r in prov]
    ar=[float(r.get('anti_parallel_risk_score',0.0)) for r in prov]
    sc=[float(r.get('scale_confidence',0.0)) for r in prov]

    thresholds_outside=(np.percentile(rc,75)<0.75) or (np.percentile(rc,25)>=0.45)
    short_circuit=(fallback>300 and sign==0)
    field_mismatch=(all(('correspondence_confidence' not in r) for r in dec))
    fallback_override=(fallback>300 and high<=3)
    if field_mismatch: src='field_mismatch'
    elif short_circuit: src='code_path'
    elif thresholds_outside: src='threshold'
    elif fallback_override: src='fallback_override'
    else: src='unknown'

    out={
      's5e16_activation':{'high_confidence_edges':high,'low_confidence_edges':low,'sign_guarded_edges':sign,'fallback_direction_edges':fallback,'train_prior_scale_edges':train_scale},
      'score_distributions':{'router_confidence':_dist(rc),'correspondence_confidence':_dist(cc),'anti_parallel_risk_score':_dist(ar),'scale_confidence':_dist(sc)},
      'threshold_diagnosis':{'thresholds_outside_distribution':bool(thresholds_outside),'if_else_short_circuit_suspected':bool(short_circuit),'field_name_mismatch_suspected':bool(field_mismatch),'default_fallback_overwrites_route':bool(fallback_override),'main_activation_failure_source':src},
      'repair_recommendation':{'use_quantile_bins':True,'recommended_high_conf_quantile':0.75,'recommended_low_conf_quantile':0.25,'recommended_sign_guard_quantile':0.85,'recommended_scale_route_rule':'scale_confidence < q35 => train_prior_scale_route'},
    }
    p=Path(args.out_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return out

def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser();
    p.add_argument('--s5e16-checkpoint',required=True);p.add_argument('--s5e16-metrics',required=True);p.add_argument('--s5e16-router-decisions',required=True);p.add_argument('--s5e16-router-input-audit',required=True);p.add_argument('--s5e12-feature-dir',required=True);p.add_argument('--out-json',required=True)
    return p.parse_args()
if __name__=='__main__': run(parse_args())
