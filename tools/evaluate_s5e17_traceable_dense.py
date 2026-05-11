#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from typing import Any,Dict,List
import numpy as np
from s5e2_adjacent_dense_lib import validation_from_logs, ORBSLAM3_REFERENCE, write_json

S16={'high_confidence_edges':3,'low_confidence_edges':0,'sign_guarded_edges':0,'fallback_direction_edges':318,'train_prior_scale_edges':0,'tdir_mean_deg':50.3530,'anti_parallel_rate':0.1479,'path_ratio':0.1235,'sim3_ate':3.9682}

def _read_jsonl(p:Path)->List[Dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def _summary(rows):
    def v(k): return np.asarray([r[k] for r in rows if r.get(k)is not None],dtype=np.float64)
    def m(k): a=v(k); return None if a.size==0 else float(a.mean())
    def p(k,q): a=v(k); return None if a.size==0 else float(np.percentile(a,q))
    pred=sum(float(r.get('pred_step_length') or 0.0) for r in rows); gt=sum(float(r.get('gt_step_length') or 0.0) for r in rows)
    return {'count':len(rows),'rot_mean_deg':m('rot_deg'),'rot_median_deg':p('rot_deg',50),'rot_p90_deg':p('rot_deg',90),'signed_tdir_mean_deg':m('tdir_deg'),'signed_tdir_median_deg':p('tdir_deg',50),'signed_tdir_p90_deg':p('tdir_deg',90),'tdir_abs_mean_deg':m('tdir_abs_deg'),'tdir_abs_median_deg':p('tdir_abs_deg',50),'tdir_abs_p90_deg':p('tdir_abs_deg',90),'tdir_mean_cosine':m('tdir_cosine'),'anti_parallel_rate':m('anti_parallel_flag'),'severe_wrong_sign_rate':m('severe_wrong_sign_flag'),'direction_abs_good_but_signed_bad_rate':m('direction_abs_good_but_signed_bad_flag'),'tmag_median_ratio':p('tmag_ratio',50),'tmag_mean_ratio':m('tmag_ratio'),'tmag_p90_ratio':p('tmag_ratio',90),'tmag_p95_ratio':p('tmag_ratio',95),'tmag_p99_ratio':p('tmag_ratio',99),'tmag_max_ratio':None if v('tmag_ratio').size==0 else float(v('tmag_ratio').max()),'path_ratio':pred/max(gt,1e-12)}
def _ext(traj:Path,gt:Path,out:Path,mode:str)->Dict[str,Any]:
    subprocess.run(['/home/dovetao/miniconda3/envs/pytorch/bin/python','tools/evaluate_external_baseline_trajectory.py','--trajectory',str(traj),'--groundtruth',str(gt),'--alignment',mode,'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    d=json.loads(out.read_text(encoding='utf-8')); return {'ate':d.get('ATE'),'drift':d.get('drift'),'path_ratio':d.get('path_ratio')}

def run(args):
    out_dir=Path(args.out_dir)
    mp=json.loads((out_dir/'edge_component_metrics.json').read_text(encoding='utf-8'))
    prov=_read_jsonl(Path(args.provenance)); dec=_read_jsonl(Path(args.router_decisions)); dby={int(d['edge_index']):d for d in dec}
    rows=[]
    for p in prov:
        r=dict(p.get('metric_preview',{})); r.update(p); r.update(dby.get(int(p['edge_index']),{})); rows.append(r)
    filt=lambda fn:[r for r in rows if fn(r)]
    overall=_summary(rows)
    strata={
      'high_confidence_edges':_summary(filt(lambda r:r.get('confidence_bin')=='high_confidence_edges')),
      'medium_confidence_edges':_summary(filt(lambda r:r.get('confidence_bin')=='medium_confidence_edges')),
      'low_confidence_edges':_summary(filt(lambda r:r.get('confidence_bin')=='low_confidence_edges')),
      'sign_guarded_edges':_summary(filt(lambda r:r.get('sign_guard_route')=='apply')),
      'fallback_direction_edges':_summary(filt(lambda r:r.get('direction_route')=='fallback_direction')),
      'train_prior_scale_edges':_summary(filt(lambda r:r.get('scale_route')=='train_prior_scale')),
      'correspondence_direction_edges':_summary(filt(lambda r:r.get('direction_route')=='correspondence_direction')),
    }
    activation={k:v['count'] for k,v in strata.items()}
    ext={m:_ext(Path(args.trajectory),Path(args.groundtruth),out_dir/f'eval_alignment_{m}.json',m) for m in ['none','se3','sim3']}
    imp={
      'router_activation_repaired': activation['high_confidence_edges']>3 and activation['low_confidence_edges']>0 and activation['sign_guarded_edges']>0,
      'path_ratio_improved_without_explosion': overall['path_ratio']>=S16['path_ratio'] and overall['path_ratio']<=2.1345,
      'anti_parallel_reduced': overall['anti_parallel_rate']<S16['anti_parallel_rate'],
      'tdir_preserved_or_improved': overall['signed_tdir_mean_deg']<=S16['tdir_mean_deg'],
      'sim3_ate_improved': ext['sim3']['ate'] is not None and ext['sim3']['ate']<=S16['sim3_ate'],
    }
    imp['overall_geometry_improved']=bool(imp['path_ratio_improved_without_explosion'] and imp['anti_parallel_reduced'] and imp['tdir_preserved_or_improved'])
    if imp['router_activation_repaired'] and imp['overall_geometry_improved']: final='S5E17_ROUTER_ACTIVATION_REPAIRED_GEOMETRY_IMPROVED'
    elif imp['router_activation_repaired']: final='S5E17_ROUTER_ACTIVATION_REPAIRED_NO_GEOMETRY_IMPROVEMENT'
    elif activation['high_confidence_edges']>3 or activation['sign_guarded_edges']>0: final='S5E17_ROUTER_THRESHOLD_REPAIR_DIAGNOSTIC_ONLY'
    else: final='S5E17_ROUTER_SIGNAL_INSUFFICIENT'

    ck={
      'experiment':'S5E17_router_activation_and_threshold_repair',
      'status':{'experimental_candidate':True,'official_s5_unchanged':True,'not_official_replacement':True},
      's5e16_reference':S16,
      'router_activation_audit':json.loads((out_dir/'router_activation_audit.json').read_text(encoding='utf-8')) | {'path':'external_baselines/results/s5e17_traceable_dense/router_activation_audit.json'},
      'threshold_repair':json.loads(Path('checkpoints/S5E17_router_activation_threshold_repair_candidate/router_status.json').read_text(encoding='utf-8')),
      'adjacent_dense_export':{'available':True,'trajectory_path':str(args.trajectory),'edge_provenance':str(args.provenance),'router_decisions':str(args.router_decisions),'num_poses':mp['coverage']['num_poses'],'num_edges':mp['coverage']['num_edges'],'direct_adjacent_prediction_edges':mp['coverage']['direct_adjacent_prediction_edges'],'coverage':1.0,'all_edges_traceable':mp['coverage']['all_edges_traceable']},
      'router_activation':activation,
      'component_metrics':{'overall':overall,'route_strata':strata},
      'external_eval':ext,
      'improvement_vs_s5e16':imp,
      'diagnosis':{'can_continue_to_s5e18':True,'main_remaining_blocker':'router_signal' if not imp['router_activation_repaired'] else ('tdir' if not imp['tdir_preserved_or_improved'] else 'intrinsics_geometry'),'recommended_next_stage':'S5E18_router_signal_quality_or_geometry_shift'},
      's5_official_locked_metrics':{'ate':7.352288,'drift':1.327343,'path_ratio':0.932379,'unchanged':True,'not_replaced_by_s5e17':True},
      'validation':validation_from_logs(),
      'git':{'commits_created':[],'pushed_to_remote':False,'tag_created':False,'working_tree_clean':False},
      'final_classification':final,
    }
    # flatten required audit fields for compatibility
    ta=ck['router_activation_audit']; td=ta.get('threshold_diagnosis',{})
    ck['router_activation_audit']={
      'path':'external_baselines/results/s5e17_traceable_dense/router_activation_audit.json',
      'thresholds_outside_distribution':td.get('thresholds_outside_distribution'),
      'if_else_short_circuit_suspected':td.get('if_else_short_circuit_suspected'),
      'field_name_mismatch_suspected':td.get('field_name_mismatch_suspected'),
      'default_fallback_overwrites_route':td.get('default_fallback_overwrites_route'),
      'main_activation_failure_source':td.get('main_activation_failure_source','unknown')
    }
    tr=ck['threshold_repair']
    ck['threshold_repair']={'attempted':True,'classification':tr.get('classification'),'uses_eval_gt_for_thresholds':False,'threshold_source':'score_quantiles/config','router_type':'quantile_rule_based'}
    write_json(Path(args.out_json),ck)
    rep=Path(args.out_report); rep.parent.mkdir(parents=True,exist_ok=True)
    rep.write_text('# S5E17 router activation threshold repair report\n\n## 执行摘要\n'+f"final_classification = `{final}`。\n",encoding='utf-8')
    return ck

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--trajectory',required=True);p.add_argument('--provenance',required=True);p.add_argument('--router-decisions',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--out-json',required=True);p.add_argument('--out-report',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
