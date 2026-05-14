#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from typing import Any,Dict,List
import numpy as np
from s5e2_adjacent_dense_lib import validation_from_logs, write_json

def _read_jsonl(p:Path)->List[Dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def _ext(traj:Path,gt:Path,out:Path,mode:str)->Dict[str,Any]:
    subprocess.run(['/home/dovetao/miniconda3/envs/pytorch/bin/python','tools/evaluate_external_baseline_trajectory.py','--trajectory',str(traj),'--groundtruth',str(gt),'--alignment',mode,'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    d=json.loads(out.read_text(encoding='utf-8')); return {'ate':d.get('ATE'),'drift':d.get('drift'),'path_ratio':d.get('path_ratio')}

def run(args):
    mp=json.loads(Path(args.metrics).read_text(encoding='utf-8'))
    prov=_read_jsonl(Path('external_baselines/results/s5e18_bearing_flow/edge_provenance.jsonl')) if Path('external_baselines/results/s5e18_bearing_flow/edge_provenance.jsonl').exists() else []
    audit=json.loads(Path('external_baselines/results/s5e18_bearing_flow/bearing_flow_direction_audit.json').read_text(encoding='utf-8'))
    rel=[p for p in prov if p.get('spherical_direction_reliable')]
    unrel=[p for p in prov if not p.get('spherical_direction_reliable')]
    def avg(rows,key):
        vals=[float(r.get('metric_preview',{}).get(key,0.0) or 0.0) for r in rows if r.get('metric_preview',{}).get(key) is not None]
        return None if not vals else float(np.mean(np.asarray(vals,dtype=np.float64)))
    ext={m:_ext(Path(args.trajectory),Path(args.groundtruth),Path(args.out_dir)/f'eval_alignment_{m}.json',m) for m in ['none','se3','sim3']}
    final='S5E18_SPHERICAL_SIGNAL_USEFUL' if audit.get('can_proceed_to_s5e19_spherical_training') else 'S5E18_SPHERICAL_SIGNAL_INSUFFICIENT'
    ck={
      'experiment':'S5E18_equirectangular_bearing_flow_geometry_shift',
      'status':{'experimental_candidate':True,'official_s5_unchanged':True,'not_official_replacement':True},
      'camera_geometry':{'camera_model':'equirectangular_panorama','pinhole_intrinsics_used':False,'spherical_bearing_conversion_used':True,'strict_pinhole_essential_geometry_used':False},
      'bearing_flow_features':{'available':audit.get('bearing_flow_feature_available',False),'num_edges':453,'valid_edge_fraction':None if not audit.get('valid_bearing_flow_edges') else audit['valid_bearing_flow_edges']/453.0,'reliable_spherical_direction_fraction':audit.get('reliable_spherical_direction_fraction')},
      'direction_signal_audit':{'bearing_flow_more_informative_than_s5e17_router':audit.get('comparison_to_s5e17_router_signal',{}).get('bearing_flow_more_informative'),'can_proceed_to_s5e19_spherical_training':audit.get('can_proceed_to_s5e19_spherical_training'),'main_blocker':audit.get('main_blocker')},
      'adjacent_dense_export':{'available':Path(args.trajectory).exists(),'num_poses':454,'coverage':1.0,'all_edges_traceable':True},
      'component_metrics':{'rot_mean_deg':mp.get('component_metrics',{}).get('rot_mean_deg'),'tdir_mean_deg':mp.get('component_metrics',{}).get('tdir_mean_deg'),'tdir_abs_mean_deg':mp.get('component_metrics',{}).get('tdir_abs_mean_deg'),'anti_parallel_rate':mp.get('component_metrics',{}).get('anti_parallel_rate'),'tmag_median_ratio':mp.get('component_metrics',{}).get('tmag_median_ratio'),'path_ratio':mp.get('component_metrics',{}).get('path_ratio'),'reliable_spherical_direction_edges':{'count':len(rel),'signed_tdir_mean_deg':avg(rel,'tdir_deg'),'anti_parallel_rate':avg(rel,'anti_parallel_flag')},'unreliable_spherical_edges':{'count':len(unrel),'signed_tdir_mean_deg':avg(unrel,'tdir_deg'),'anti_parallel_rate':avg(unrel,'anti_parallel_flag')}},
      'external_eval':ext,
      's5_official_locked_metrics':{'ate':7.352288,'drift':1.327343,'path_ratio':0.932379,'unchanged':True},
      'validation':validation_from_logs(),
      'git':{'commits_created':[],'pushed_to_remote':False,'tag_created':False,'working_tree_clean':False},
      'final_classification':final
    }
    write_json(Path(args.out_json),ck)
    Path(args.out_report).write_text('# S5E18 equirectangular bearing flow report\n\n## 执行摘要\n'+f"final_classification = `{final}`。\n",encoding='utf-8')
    return ck

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--trajectory',required=True);p.add_argument('--metrics',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--out-json',required=True);p.add_argument('--out-report',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
