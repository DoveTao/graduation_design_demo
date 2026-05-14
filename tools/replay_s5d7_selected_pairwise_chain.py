#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent



def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    return q


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def _read_timestamps(path: Path) -> List[float]:
    vals: List[float] = []
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        vals.append(float(s.split()[0]))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairwise-jsonl', required=True)
    ap.add_argument('--timestamps', required=True)
    ap.add_argument('--out-tum', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    pair_path = _resolve(args.pairwise_jsonl)
    ts_path = _resolve(args.timestamps)
    out_tum = _resolve(args.out_tum)
    out_json = _resolve(args.out_json)
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in pair_path.read_text(encoding='utf-8', errors='ignore').splitlines() if x.strip()] if pair_path.exists() else []
    ts_all = _read_timestamps(ts_path) if ts_path.exists() else []

    meta: Dict[str, Any] = {
        'selected_k1_caveat': 'selected_k1 sparse protocol; not full dense 454 replay',
        'attempted': True,
        'num_pairs': len(rows),
        'graph_connected': False,
        'num_components': 0,
        'num_replayable_edges': 0,
        'variants': {
            'model_convention_as_declared': {'trajectory_path': str(out_tum.relative_to(REPO_ROOT)), 'num_poses': 0, 'coverage_vs_454': 0.0},
            'inverse_relative_variant': {'trajectory_path': str((out_tum.parent / (out_tum.stem + '_inverse.txt')).relative_to(REPO_ROOT)), 'num_poses': 0, 'coverage_vs_454': 0.0},
        },
        'selected_variant': 'none',
    }

    if not rows:
        out_tum.write_text('', encoding='utf-8')
        out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    edges = []
    und = defaultdict(set)
    out_map = defaultdict(list)
    in_deg = defaultdict(int)
    nodes = set()
    for r in rows:
        i = int(r['frame_i']); j = int(r['frame_j'])
        R = np.asarray(r['rotation']['value'], dtype=np.float64)
        t = np.asarray(r['translation']['value'], dtype=np.float64)
        edges.append((i, j, R, t, float(r['timestamp_i']), float(r['timestamp_j'])))
        out_map[i].append((j, R, t, float(r['timestamp_i']), float(r['timestamp_j'])))
        und[i].add(j); und[j].add(i)
        in_deg[j] += 1
        nodes.add(i); nodes.add(j)

    # components
    seen = set(); components = []
    for n in sorted(nodes):
        if n in seen:
            continue
        st = [n]; seen.add(n); comp = []
        while st:
            u = st.pop(); comp.append(u)
            for v in und[u]:
                if v not in seen:
                    seen.add(v); st.append(v)
        components.append(sorted(comp))
    components.sort(key=len, reverse=True)
    meta['num_components'] = len(components)
    meta['graph_connected'] = len(components) == 1

    # pick largest component and derive replayable chain by frame order continuity
    comp_nodes = set(components[0])
    comp_edges = [e for e in edges if e[0] in comp_nodes and e[1] in comp_nodes]
    comp_edges.sort(key=lambda x: (x[0], x[1]))

    # build contiguous path: prefer indegree==0 start else min frame
    start_candidates = [n for n in comp_nodes if in_deg.get(n, 0) == 0]
    cur = min(start_candidates) if start_candidates else min(comp_nodes)
    chain = []
    used = set()
    while True:
        cands = [c for c in out_map.get(cur, []) if (cur, c[0]) not in used and c[0] in comp_nodes]
        if not cands:
            break
        cands.sort(key=lambda x: x[0])
        j, R, t, tsi, tsj = cands[0]
        chain.append((cur, j, R, t, tsi, tsj))
        used.add((cur, j))
        cur = j

    meta['num_replayable_edges'] = len(chain)

    def replay(chain_edges: List[Tuple[int,int,np.ndarray,np.ndarray,float,float]], inverse: bool=False):
        poses = {}
        if not chain_edges:
            return poses
        first = chain_edges[0][0]
        poses[first] = np.eye(4, dtype=np.float64)
        for i, j, R, t, _, _ in chain_edges:
            if i not in poses:
                continue
            Tij = _T(R, t)
            if inverse:
                Tij = np.linalg.inv(Tij)
            poses[j] = poses[i] @ Tij
        return poses

    poses_m = replay(chain, inverse=False)
    poses_i = replay(chain, inverse=True)

    def dump_tum(path: Path, poses: Dict[int, np.ndarray]):
        lines = []
        for idx in sorted(poses.keys()):
            if idx < 0 or idx >= len(ts_all):
                continue
            T = poses[idx]
            q = _rot_to_quat_xyzw(T[:3,:3]); t = T[:3,3]
            lines.append(f"{ts_all[idx]:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
        path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        return len(lines)

    n_m = dump_tum(out_tum, poses_m)
    inv_path = out_tum.parent / (out_tum.stem + '_inverse.txt')
    n_i = dump_tum(inv_path, poses_i)

    meta['variants']['model_convention_as_declared']['num_poses'] = n_m
    meta['variants']['model_convention_as_declared']['coverage_vs_454'] = float(n_m / 454.0)
    meta['variants']['inverse_relative_variant']['num_poses'] = n_i
    meta['variants']['inverse_relative_variant']['coverage_vs_454'] = float(n_i / 454.0)
    meta['selected_variant'] = 'model_convention_as_declared' if n_m >= 2 else ('inverse_relative_variant' if n_i >= 2 else 'none')
    meta['replayable_chain_edges'] = [{'i':e[0], 'j':e[1], 'timestamp_i':e[4], 'timestamp_j':e[5]} for e in chain]

    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
