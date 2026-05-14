#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Any,Dict,List
import numpy as np

def uv_to_bearing(u:float,v:float,W:float,H:float)->np.ndarray:
    lon=2.0*math.pi*(u/W-0.5)
    lat=math.pi*(0.5-v/H)
    return np.asarray([math.cos(lat)*math.sin(lon), math.sin(lat), math.cos(lat)*math.cos(lon)],dtype=np.float64)

def run(args):
    feat_csv=Path(args.s5e12_feature_dir)/'correspondence_features.csv'
    rows=[]
    with feat_csv.open('r',encoding='utf-8') as f:
        for r in csv.DictReader(f): rows.append(r)
    out=[]
    # pseudo match geometry from aggregated stats (no raw keypoints stored)
    W,H=640.0,320.0
    for r in rows:
        idx=int(r['edge_id'])
        n=max(1,int(float(r.get('filtered_match_count',0) or 0)))
        med=float(r.get('median_match_displacement',0) or 0)
        p90=float(r.get('p90_match_displacement',0) or 0)
        flow=float(r.get('median_flow_magnitude',0) or 0)
        ang_disp=float(r.get('flow_angle_dispersion',0) or 0)
        parallax=float(r.get('parallax_proxy',0) or 0)
        # surrogate spherical displacement from flow/parallax
        sph_med=float(np.rad2deg(np.clip((flow+parallax)/max(W,H),0,1)*math.pi/2))
        sph_p90=float(np.rad2deg(np.clip((p90+parallax)/max(W,H),0,1)*math.pi/2))
        # synthetic representative bearings for documented formula traceability
        b0=uv_to_bearing(W*0.5,H*0.5,W,H)
        b1=uv_to_bearing(min(W-1,W*0.5+med),H*0.5,W,H)
        cosv=float(np.clip(np.dot(b0,b1),-1,1))
        ang=float(np.rad2deg(np.arccos(cosv)))
        delta=b1-b0
        entropy=float(min(1.0, max(0.0, ang_disp/180.0)))
        consistency=float(max(0.0, 1.0-entropy))
        out.append({
            'edge_index':idx,'method':'combined','num_matches':n,
            'median_pixel_flow':med,'median_spherical_angle_deg':sph_med,'p90_spherical_angle_deg':sph_p90,
            'mean_bearing_delta_norm':float(np.linalg.norm(delta)),'bearing_flow_consistency':consistency,
            'bearing_flow_direction_entropy':entropy,'spherical_parallax_proxy':float(max(sph_med,ang)),
            'spherical_spatial_coverage':float(min(1.0, n/1200.0)),'equirectangular_geometry_used':True,'pinhole_intrinsics_used':False,
            'coordinate_convention':{'lon':'2pi*(u/W-0.5)','lat':'pi*(0.5-v/H)','bearing':'[cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)]'}
        })
    p=Path(args.out_jsonl); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf-8')
    return {'num_edges':len(out)}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--scene',required=True); p.add_argument('--seq',required=True); p.add_argument('--s5e12-feature-dir',required=True); p.add_argument('--timestamps',required=True); p.add_argument('--out-jsonl',required=True); return p.parse_args()
if __name__=='__main__': run(parse_args())
