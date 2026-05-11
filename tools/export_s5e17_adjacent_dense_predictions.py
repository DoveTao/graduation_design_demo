#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, vector_angle_deg, write_json

def _read_jsonl(p:Path)->List[Dict[str,Any]]: return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def _gt_rel(gt:Dict[float,Dict[str,np.ndarray]],a:float,b:float)->Optional[Tuple[np.ndarray,np.ndarray]]:
    if a not in gt or b not in gt: return None
    gi,gj=gt[a],gt[b]; return gj['R'].T@gi['R'], gj['R'].T@(gi['t']-gj['t'])
def _metric(R:np.ndarray,t:np.ndarray,gr):
    if gr is None: return {}
    Rg,tg=gr; tdir=vector_angle_deg(t,tg,absolute=False); tdir_abs=vector_angle_deg(t,tg,absolute=True); cos=None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {'rot_deg':angle_deg_from_rot(R@Rg.T),'tdir_deg':tdir,'tdir_abs_deg':tdir_abs,'tdir_cosine':cos,'anti_parallel_flag':bool(cos is not None and cos<0),'severe_wrong_sign_flag':bool(tdir is not None and tdir>120),'direction_abs_good_but_signed_bad_flag':bool(tdir is not None and tdir_abs is not None and tdir>120 and tdir_abs<45),'tmag_ratio':float(np.linalg.norm(t)/max(np.linalg.norm(tg),1e-12)),'pred_step_length':float(np.linalg.norm(t)),'gt_step_length':float(np.linalg.norm(tg))}
def _summary(rows):
    def v(k): return np.asarray([r[k] for r in rows if r.get(k)is not None],dtype=np.float64)
    def m(k): a=v(k); return None if a.size==0 else float(a.mean())
    def p(k,q): a=v(k); return None if a.size==0 else float(np.percentile(a,q))
    pred=sum(float(r.get('pred_step_length') or 0.0) for r in rows); gt=sum(float(r.get('gt_step_length') or 0.0) for r in rows)
    return {'rot_mean_deg':m('rot_deg'),'rot_median_deg':p('rot_deg',50),'rot_p90_deg':p('rot_deg',90),'tdir_mean_deg':m('tdir_deg'),'tdir_median_deg':p('tdir_deg',50),'tdir_p90_deg':p('tdir_deg',90),'tdir_abs_mean_deg':m('tdir_abs_deg'),'tdir_abs_median_deg':p('tdir_abs_deg',50),'tdir_abs_p90_deg':p('tdir_abs_deg',90),'tdir_mean_cosine':m('tdir_cosine'),'anti_parallel_rate':m('anti_parallel_flag'),'severe_wrong_sign_rate':m('severe_wrong_sign_flag'),'direction_abs_good_but_signed_bad_rate':m('direction_abs_good_but_signed_bad_flag'),'tmag_median_ratio':p('tmag_ratio',50),'tmag_mean_ratio':m('tmag_ratio'),'tmag_p90_ratio':p('tmag_ratio',90),'tmag_p95_ratio':p('tmag_ratio',95),'tmag_p99_ratio':p('tmag_ratio',99),'tmag_max_ratio':None if v('tmag_ratio').size==0 else float(v('tmag_ratio').max()),'path_ratio':pred/max(gt,1e-12)}
def _write_tum(path:Path,timestamps,rels):
    Rw=np.eye(3); tw=np.zeros(3); lines=[]; qx,qy,qz,qw=rot_to_quat_xyzw(Rw); lines.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for i,(R_BA,t_BA) in enumerate(rels):
        Rb=Rw@R_BA.T; tb=tw-Rb@t_BA; Rw,tw=Rb,tb; qx,qy,qz,qw=rot_to_quat_xyzw(Rw); lines.append(f"{timestamps[i+1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(args):
    s16=_read_jsonl(Path('external_baselines/results/s5e16_traceable_dense/edge_provenance.jsonl'))
    s15=_read_jsonl(Path('external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl')); s15_by={int(x['edge_index']):x for x in s15}
    s14=_read_jsonl(Path('external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl')); s14_by={int(x['edge_index']):x for x in s14}
    s13=_read_jsonl(Path('external_baselines/results/s5e13_traceable_dense/edge_provenance.jsonl')); s13_by={int(x['edge_index']):x for x in s13}
    status=json.loads(Path(args.candidate,'router_status.json').read_text(encoding='utf-8'))

    feat_by={}
    with Path(args.s5e12_feature_dir,'correspondence_features.csv').open('r',encoding='utf-8') as f:
        for r in csv.DictReader(f): feat_by[int(r['edge_id'])]=r

    # collect scores for quantiles
    scores=[]; risks=[]; scales=[]
    for p in s16:
        idx=int(p['edge_index']); f=feat_by.get(idx,{})
        inlier=float(f.get('inlier_ratio',0) or 0); parallax=float(f.get('parallax_proxy',0) or 0); flow_disp=float(f.get('flow_angle_dispersion',0) or 0)
        corr=float(min(1,max(0,0.65*inlier+0.35*min(1,parallax))))
        risk=float(min(1,max(0,(flow_disp/120.0+(1-corr))*0.5)))
        scl=float(min(1,max(0,1-abs(float(p.get('translation_magnitude',0))-float(s15_by[idx].get('translation_magnitude',0)))/max(float(p.get('translation_magnitude',1)),1e-6))))
        score=float(min(1,max(0,0.5*corr+0.3*(1-risk)+0.2*scl)))
        scores.append(score); risks.append(risk); scales.append(scl)
    q25,q75=np.percentile(scores,25),np.percentile(scores,75); qrisk=np.percentile(risks,85); qscale=np.percentile(scales,35)

    gt=read_tum(Path(args.groundtruth)); ts=read_timestamps(Path(args.timestamps))
    rels=[]; prov=[]; dec=[]; mrows=[]
    for p in s16:
        idx=int(p['edge_index']); f=feat_by.get(idx,{})
        inlier=float(f.get('inlier_ratio',0) or 0); parallax=float(f.get('parallax_proxy',0) or 0); flow_disp=float(f.get('flow_angle_dispersion',0) or 0)
        corr=float(min(1,max(0,0.65*inlier+0.35*min(1,parallax))))
        risk=float(min(1,max(0,(flow_disp/120.0+(1-corr))*0.5)))
        scl=float(min(1,max(0,1-abs(float(p.get('translation_magnitude',0))-float(s15_by[idx].get('translation_magnitude',0)))/max(float(p.get('translation_magnitude',1)),1e-6))))
        score=float(min(1,max(0,0.5*corr+0.3*(1-risk)+0.2*scl)))

        if score>=q75: cbin='high_confidence_edges'
        elif score<q25: cbin='low_confidence_edges'
        else: cbin='medium_confidence_edges'

        d16=np.asarray(p['translation_direction'],dtype=np.float64); d15=np.asarray(s15_by[idx]['translation_direction'],dtype=np.float64); d14=np.asarray(s14_by[idx]['translation_direction'],dtype=np.float64); d13=np.asarray(s13_by[idx]['translation_direction'],dtype=np.float64)
        for d in (d16,d15,d14,d13):
            n=np.linalg.norm(d)
            if n>1e-12: d/=n

        if cbin=='high_confidence_edges': direction_route='correspondence_direction'; d=d16
        elif cbin=='medium_confidence_edges': direction_route='s5e15_direction'; d=d15
        else: direction_route='fallback_direction'; d=d13

        sign_guard='apply' if risk>=qrisk else 'no_apply'
        if sign_guard=='apply':
            cand=[d,d14,d13]; scores=[float(np.dot(c,d14)+np.dot(c,d13)) for c in cand]; d=cand[int(np.argmax(scores))]

        if scl<qscale: scale_route='train_prior_scale'; sf=2.8
        else: scale_route='conservative_scale'; sf=2.2
        mag=float(s15_by[idx]['translation_magnitude'])*sf/2.8

        R=np.asarray(p['rotation']['value'],dtype=np.float64); t=d*mag; rels.append((R,t))
        mm=_metric(R,t,_gt_rel(gt,float(p['timestamp_i']),float(p['timestamp_j']))); mrows.append(mm)

        route_reason=f'bin={cbin},score={score:.3f},risk={risk:.3f},scl={scl:.3f}'
        decrow={'edge_index':idx,'direction_route':direction_route,'scale_route':scale_route,'sign_guard_route':sign_guard,'router_confidence':score,'reason':route_reason,'uses_eval_gt':False}
        dec.append(decrow)
        prov.append({'edge_index':idx,'timestamp_i':float(p['timestamp_i']),'timestamp_j':float(p['timestamp_j']),'source_type':'direct_adjacent_prediction','source_model':'S5E17_router_activation_threshold_repair_candidate','uses_gt_for_prediction':False,'uses_eval_gt_for_thresholds':False,'uses_eval_gt_for_router':False,'uses_eval_gt_for_scale':False,'uses_eval_gt_for_sign':False,'router_type':'quantile_rule_based','confidence_bin':cbin,'direction_route':direction_route,'scale_route':scale_route,'sign_guard_route':sign_guard,'router_confidence':score,'correspondence_confidence':corr,'anti_parallel_risk_score':risk,'scale_confidence':scl,'scale_factor':sf,'scale_factor_source':'train_split_prior','route_reason':route_reason,'rotation':{'representation':'matrix','value':R.tolist()},'translation':{'frame':'B/local','value':t.tolist()},'translation_direction':d.tolist(),'translation_magnitude':mag,'metric_preview':mm,'notes':['threshold_repair_no_eval_gt']})

    _write_tum(Path(args.out_tum),ts,rels)
    Path(args.out_provenance).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_provenance).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in prov)+'\n',encoding='utf-8')
    Path(args.out_router_decisions).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in dec)+'\n',encoding='utf-8')
    payload={'experiment':'S5E17_router_activation_and_threshold_repair','coverage':{'num_poses':len(ts),'num_edges':len(prov),'direct_adjacent_prediction_edges':len(prov),'all_edges_traceable':True},'component_metrics':_summary(mrows),'metrics_rows':mrows,'trajectory_path':str(args.out_tum)}
    write_json(Path(args.out_metrics),payload); return payload

def parse_args():
    p=argparse.ArgumentParser();
    p.add_argument('--scene',required=True);p.add_argument('--seq',required=True);p.add_argument('--candidate',required=True);p.add_argument('--s5e16-checkpoint',required=True);p.add_argument('--s5e15-checkpoint',required=True);p.add_argument('--s5e14-checkpoint',required=True);p.add_argument('--s5e13-checkpoint',required=True);p.add_argument('--s5e12-feature-dir',required=True);p.add_argument('--timestamps',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--out-tum',required=True);p.add_argument('--out-provenance',required=True);p.add_argument('--out-metrics',required=True);p.add_argument('--out-router-decisions',required=True)
    return p.parse_args()
if __name__=='__main__': run(parse_args())
