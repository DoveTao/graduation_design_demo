#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Dict,List
from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, write_json

def _load(p:Path)->Dict[str,Any]: return json.loads(p.read_text(encoding='utf-8'))
def _pick(ck:Dict[str,Any],name:str)->Dict[str,Any]:
    ov=ck.get('component_metrics',{}).get('overall',ck.get('component_metrics',{})); ext=ck.get('external_eval',{})
    return {'name':name,'coverage':ck.get('adjacent_dense_export',{}).get('coverage'),'rot_mean_deg':ov.get('rot_mean_deg'),'signed_tdir_mean_deg':ov.get('signed_tdir_mean_deg',ov.get('tdir_mean_deg')),'anti_parallel_rate':ov.get('anti_parallel_rate'),'tmag_median_ratio':ov.get('tmag_median_ratio'),'tmag_p95_ratio':ov.get('tmag_p95_ratio'),'none_ate':ext.get('none',{}).get('ate'),'se3_ate':ext.get('se3',{}).get('ate'),'sim3_ate':ext.get('sim3',{}).get('ate'),'path_ratio':ov.get('path_ratio'),'caveat':ck.get('final_classification')}

def run(args):
    s17=_load(Path(args.s5e17_checkpoint)); s16=_load(Path(args.s5e16_checkpoint)); s15=_load(Path(args.s5e15_checkpoint))
    rows=[{'name':'S5 official locked result','coverage':None,'rot_mean_deg':None,'signed_tdir_mean_deg':None,'anti_parallel_rate':None,'tmag_median_ratio':None,'tmag_p95_ratio':None,'none_ate':None,'se3_ate':7.352288,'sim3_ate':7.352288,'path_ratio':0.932379,'caveat':'official_locked'},_pick(s15,'S5E15'),_pick(s16,'S5E16'),_pick(s17,'S5E17'),{'name':'ORB-SLAM3 external baseline','coverage':ORBSLAM3_REFERENCE['tracking_success_rate'],'rot_mean_deg':None,'signed_tdir_mean_deg':None,'anti_parallel_rate':None,'tmag_median_ratio':None,'tmag_p95_ratio':None,'none_ate':ORBSLAM3_REFERENCE['none']['ate'],'se3_ate':ORBSLAM3_REFERENCE['se3']['ate'],'sim3_ate':ORBSLAM3_REFERENCE['sim3']['ate'],'path_ratio':ORBSLAM3_REFERENCE['se3']['path_ratio'],'caveat':'external_baseline'}]
    payload={'experiment':'S5E17_router_activation_and_threshold_repair','rows':rows,'summary':'S5E17 重点验证 router activation 修复是否转化为几何指标改善。'}
    write_json(Path(args.out_json),payload)
    Path(args.out_report).write_text('# S5E17 comparison\n\n## 执行摘要\n'+payload['summary']+'\n',encoding='utf-8')
    return payload

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--s5e17-checkpoint',required=True);p.add_argument('--s5e16-checkpoint',required=True);p.add_argument('--s5e15-checkpoint',required=True);p.add_argument('--orbslam3-eval-dir',required=True);p.add_argument('--out-json',required=True);p.add_argument('--out-report',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
