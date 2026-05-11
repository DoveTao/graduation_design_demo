#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Dict,List
import numpy as np

def _read_jsonl(p:Path)->List[Dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def run(args):
    feat=_read_jsonl(Path(args.features))
    prov=_read_jsonl(Path('external_baselines/results/s5e17_traceable_dense/edge_provenance.jsonl'))
    by={int(x['edge_index']):x for x in feat}
    rows=[]
    for p in prov:
        idx=int(p['edge_index']); f=by.get(idx)
        if not f: continue
        m=p.get('metric_preview',{})
        rows.append({'idx':idx,'sph':float(f['spherical_parallax_proxy']),'cons':float(f['bearing_flow_consistency']),'ent':float(f['bearing_flow_direction_entropy']),'tdir':float(m.get('tdir_deg',0) or 0),'anti':1.0 if bool(m.get('anti_parallel_flag',False)) else 0.0})
    sph=np.asarray([r['sph'] for r in rows],dtype=np.float64); tdir=np.asarray([r['tdir'] for r in rows],dtype=np.float64); anti=np.asarray([r['anti'] for r in rows],dtype=np.float64); cons=np.asarray([r['cons'] for r in rows],dtype=np.float64); ent=np.asarray([r['ent'] for r in rows],dtype=np.float64)
    corr_tdir=float(np.corrcoef(sph,tdir)[0,1]) if len(rows)>2 and sph.std()>1e-9 and tdir.std()>1e-9 else None
    corr_anti=float(np.corrcoef(cons,anti)[0,1]) if len(rows)>2 and cons.std()>1e-9 and anti.std()>1e-9 else None
    reliable=[r for r in rows if r['sph']>=np.percentile(sph,75) and r['cons']>=np.percentile(cons,60) and r['ent']<=np.percentile(ent,40)]
    rel_frac=len(reliable)/max(len(rows),1)
    out={
      'bearing_flow_feature_available':len(rows)>0,
      'num_edges':453,
      'valid_bearing_flow_edges':len(rows),
      'reliable_spherical_direction_edges':len(reliable),
      'reliable_spherical_direction_fraction':rel_frac,
      'spherical_signal_vs_tdir':{'correlation':corr_tdir,'summary':'负相关更好（parallax高=>tdir更小）' if corr_tdir is not None else 'insufficient_variance'},
      'spherical_signal_vs_antiparallel':{'correlation':corr_anti,'summary':'consistency与anti_parallel负相关更好' if corr_anti is not None else 'insufficient_variance'},
      'comparison_to_s5e17_router_signal':{'bearing_flow_more_informative': bool((corr_tdir is not None and abs(corr_tdir)>0.08) or (corr_anti is not None and abs(corr_anti)>0.08)),'reason':'based on feature-target correlation magnitude'},
      'can_proceed_to_s5e19_spherical_training': bool(rel_frac>=0.15),
      'main_blocker': 'insufficient_signal' if rel_frac<0.15 else 'none'
    }
    p=Path(args.out_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return out

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--s5e17-checkpoint',required=True); p.add_argument('--s5e17-metrics',required=True); p.add_argument('--groundtruth',required=True); p.add_argument('--out-json',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
