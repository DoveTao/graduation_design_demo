#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_ANCHOR = "reports/s5d10_componentwise_selected_replay_and_cuda_validation.md"
CHECKPOINT_ANCHOR = "checkpoints/S5D10_componentwise_selected_replay_and_cuda_validation.json"
DECLARED_VARIANT_ANCHOR = "model_convention_as_declared"
INVERSE_VARIANT_ANCHOR = "inverse_relative_variant"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _read_timestamps(path: Path) -> List[float]:
    vals = []
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        vals.append(float(s.split()[0]))
    return vals


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


def _dump_tum(path: Path, poses: Dict[int, np.ndarray], ts_all: List[float]) -> int:
    lines = []
    for idx in sorted(poses.keys()):
        if idx < 0 or idx >= len(ts_all):
            continue
        T = poses[idx]
        q = _quat_xyzw(T[:3, :3]); t = T[:3, 3]
        lines.append(f"{ts_all[idx]:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--timestamps', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    pair_path = _resolve(args.pairwise_jsonl)
    ts_all = _read_timestamps(_resolve(args.timestamps))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_decl = out_dir / 'componentwise_replay_declared'
    out_inv = out_dir / 'componentwise_replay_inverse'
    out_decl.mkdir(parents=True, exist_ok=True)
    out_inv.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in pair_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    edges = []
    und = defaultdict(set)
    outm = defaultdict(list)
    indeg = defaultdict(int)
    nodes = set()
    for r in rows:
        i, j = int(r['frame_i']), int(r['frame_j'])
        R = np.asarray(r['rotation']['value'], dtype=np.float64)
        t = np.asarray(r['translation']['value'], dtype=np.float64)
        e = {'i': i, 'j': j, 'R': R, 't': t, 'tsi': float(r['timestamp_i']), 'tsj': float(r['timestamp_j'])}
        edges.append(e)
        und[i].add(j); und[j].add(i)
        outm[i].append(e)
        indeg[j] += 1
        nodes |= {i, j}

    comps = []
    seen = set()
    for n in sorted(nodes):
        if n in seen:
            continue
        st = [n]; seen.add(n); comp=[]
        while st:
            u = st.pop(); comp.append(u)
            for v in und[u]:
                if v not in seen:
                    seen.add(v); st.append(v)
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)

    comp_infos = []
    for cid, comp in enumerate(comps):
        cset = set(comp)
        ce = [e for e in edges if e['i'] in cset and e['j'] in cset]
        ce.sort(key=lambda x: (x['i'], x['j']))
        outdeg = defaultdict(int)
        indeg_c = defaultdict(int)
        for e in ce:
            outdeg[e['i']] += 1
            indeg_c[e['j']] += 1
        has_branch = any(outdeg[n] > 1 for n in cset)
        has_cycle = not any(indeg_c[n] == 0 for n in cset)
        is_chain = (not has_branch) and (sum(1 for n in cset if indeg_c[n] == 0) <= 1)
        replayable = len(ce) > 0

        # build deterministic walk from source/min node
        start = min([n for n in cset if indeg_c[n] == 0], default=min(cset))
        used = set(); walk=[]; cur = start
        while True:
            cand = [e for e in outm.get(cur, []) if e['j'] in cset and (e['i'], e['j']) not in used]
            if not cand:
                break
            cand.sort(key=lambda x: x['j'])
            e = cand[0]
            walk.append(e)
            used.add((e['i'], e['j']))
            cur = e['j']

        def replay(inverse=False):
            poses = {walk[0]['i']: np.eye(4, dtype=np.float64)} if walk else {}
            for e in walk:
                if e['i'] not in poses:
                    continue
                Tij = _T(e['R'], e['t'])
                if inverse:
                    Tij = np.linalg.inv(Tij)
                poses[e['j']] = poses[e['i']] @ Tij
            return poses

        p_decl = replay(False)
        p_inv = replay(True)
        decl_path = out_decl / f'component_{cid:02d}.tum'
        inv_path = out_inv / f'component_{cid:02d}.tum'
        n_decl = _dump_tum(decl_path, p_decl, ts_all)
        n_inv = _dump_tum(inv_path, p_inv, ts_all)

        comp_infos.append({
            'component_id': cid,
            'num_edges': len(ce),
            'num_nodes': len(cset),
            'start_timestamp': min((e['tsi'] for e in ce), default=None),
            'end_timestamp': max((e['tsj'] for e in ce), default=None),
            'is_chain': is_chain,
            'has_branch': has_branch,
            'has_cycle': has_cycle,
            'replayable': replayable,
            'walk_edges': len(walk),
            'declared_tum': str(decl_path.relative_to(REPO_ROOT)),
            'inverse_tum': str(inv_path.relative_to(REPO_ROOT)),
            'declared_num_poses': n_decl,
            'inverse_num_poses': n_inv,
        })

    audit = {
        'num_pairs': len(rows),
        'num_components': len(comps),
        'longest_component_edges': max((c['num_edges'] for c in comp_infos), default=0),
        'num_replayable_components': sum(1 for c in comp_infos if c['replayable']),
        'components': comp_infos,
        'selected_k1_sparse_protocol_caveat': True,
    }
    (out_dir / 'component_graph_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
