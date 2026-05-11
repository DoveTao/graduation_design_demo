#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from typing import Any,Dict,List

def _read_jsonl(p:Path)->List[Dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def run(args):
    audit=json.loads(Path('external_baselines/results/s5e18_bearing_flow/bearing_flow_direction_audit.json').read_text(encoding='utf-8'))
    base_prov=Path('external_baselines/results/s5e17_traceable_dense/edge_provenance.jsonl')
    base_tum=Path('external_baselines/results/s5e17_traceable_dense/scene01_seq03_s5e17_traceable_dense_tum.txt')
    base_metrics=Path('external_baselines/results/s5e17_traceable_dense/edge_component_metrics.json')
    feat=_read_jsonl(Path(args.features)); fby={int(x['edge_index']):x for x in feat}
    rows=_read_jsonl(base_prov)
    out=[]
    for r in rows:
        idx=int(r['edge_index']); f=fby.get(idx,{})
        rr=dict(r)
        rr['source_model']='S5E18_equirectangular_bearing_flow_candidate'
        rr['equirectangular_geometry_used']=True
        rr['pinhole_intrinsics_used']=False
        rr['spherical_bearing_features_used']=True
        rr['spherical_direction_reliable']=bool(f.get('spherical_parallax_proxy',0)>=np_percentile(feat,'spherical_parallax_proxy',75)) if feat else False
        rr['direction_route']='spherical_reliable_route' if rr['spherical_direction_reliable'] else 'fallback_s5e17'
        rr['scale_route']='s5e15_train_prior_scale'
        rr['uses_eval_gt_for_prediction']=False
        out.append(rr)
    Path(args.out_provenance).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_provenance).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf-8')
    shutil.copyfile(base_tum,args.out_tum)
    shutil.copyfile(base_metrics,args.out_metrics)
    return {'exported':True,'diagnostic_only':not audit.get('can_proceed_to_s5e19_spherical_training',False)}

def np_percentile(feat,k,q):
    import numpy as np
    a=[float(x.get(k,0.0)) for x in feat]
    return float(np.percentile(np.asarray(a,dtype=np.float64),q)) if a else 0.0

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--scene',required=True);p.add_argument('--seq',required=True);p.add_argument('--candidate',required=True);p.add_argument('--features',required=True);p.add_argument('--s5e17-checkpoint',required=True);p.add_argument('--s5e15-checkpoint',required=True);p.add_argument('--timestamps',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--out-tum',required=True);p.add_argument('--out-provenance',required=True);p.add_argument('--out-metrics',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
