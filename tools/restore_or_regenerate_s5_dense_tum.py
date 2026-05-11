#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DEFAULT = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
REPORT_ANCHOR = "reports/s5d9_restore_s5_dense_and_selected_edge_comparison.md"
CHECKPOINT_ANCHOR = "checkpoints/S5D9_restore_s5_dense_and_selected_edge_comparison.json"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _read_tum(path: Path) -> List[List[float]]:
    rows = []
    if not path.exists():
        return rows
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        rows.append([float(x) for x in p[:8]])
    return rows


def _read_ts(path: Path) -> List[float]:
    vals = []
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        vals.append(float(s.split()[0]))
    return vals


def _validate_tum(rows: List[List[float]], ts: List[float], gt_rows: List[List[float]]) -> Dict[str, Any]:
    num = len(rows)
    aligned = False
    if num == len(ts):
        est_ts = np.asarray([r[0] for r in rows], dtype=np.float64)
        gt_ts = np.asarray(ts, dtype=np.float64)
        aligned = bool(np.allclose(est_ts, gt_ts, atol=1e-6, rtol=0.0))
    quat_ok = True
    for r in rows:
        q = np.asarray(r[4:8], dtype=np.float64)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n < 1e-6 or abs(n - 1.0) > 1e-2:
            quat_ok = False
            break
    # rough leakage check: not identical to GT trajectory columns 1:8
    leak = False
    if len(gt_rows) == len(rows) and len(rows) > 0:
        a = np.asarray([r[1:8] for r in rows], dtype=np.float64)
        b = np.asarray([r[1:8] for r in gt_rows], dtype=np.float64)
        leak = bool(np.allclose(a, b, atol=1e-9, rtol=0.0))
    return {
        'num_poses': num,
        'timestamp_status': 'aligned_exact' if aligned else 'mismatch',
        'quat_valid': quat_ok,
        'gt_leakage_check_passed': (not leak),
    }


def _run_eval(traj: Path, gt: Path, out_dir: Path) -> Dict[str, Any]:
    out = {}
    refs = {
        'none': 21.681522044975264,
        'se3': 8.231468716451547,
        'sim3': 4.07912293550008,
    }
    ok = True
    for a in ('none', 'se3', 'sim3'):
        j = out_dir / f'eval_alignment_{a}.json'
        subprocess.run([
            '/home/dovetao/miniconda3/envs/pytorch/bin/python',
            'tools/evaluate_external_baseline_trajectory.py',
            '--trajectory', str(traj),
            '--groundtruth', str(gt),
            '--alignment', a,
            '--output-json', str(j),
        ], cwd=REPO_ROOT, check=False)
        if j.exists():
            obj = json.loads(j.read_text(encoding='utf-8'))
            out[a] = obj
            ate = float(obj.get('ATE', float('nan')))
            if not np.isfinite(ate) or abs(ate - refs[a]) > 1e-3:
                ok = False
        else:
            ok = False
            out[a] = {'status': 'missing'}
    out['metrics_consistent_with_s5d'] = ok
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=TARGET_DEFAULT)
    ap.add_argument('--timestamps', required=True)
    ap.add_argument('--groundtruth', required=True)
    ap.add_argument('--out-metadata', required=True)
    args = ap.parse_args()

    target = _resolve(args.target)
    ts_path = _resolve(args.timestamps)
    gt_path = _resolve(args.groundtruth)
    out_meta = _resolve(args.out_metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    ts = _read_ts(ts_path)
    gt_rows = _read_tum(gt_path)

    restored = False
    source = 'unavailable'
    source_path = None

    if target.exists() and target.stat().st_size > 0:
        restored = True
        source = 'found_existing'
        source_path = str(target.relative_to(REPO_ROOT))
    else:
        # restore from known branch artifact
        branch_path = 'external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt'
        try:
            content = subprocess.check_output(['git', 'show', f'experiment/orbslam3-fisheye-strong-baseline:{branch_path}'], cwd=REPO_ROOT)
            target.write_bytes(content)
            restored = True
            source = 'found_existing'
            source_path = f'git:experiment/orbslam3-fisheye-strong-baseline:{branch_path}'
        except Exception:
            restored = False
            source = 'unavailable'

    rows = _read_tum(target)
    val = _validate_tum(rows, ts, gt_rows) if restored else {'num_poses': 0, 'timestamp_status': 'missing', 'quat_valid': False, 'gt_leakage_check_passed': False}

    eval_info = {'metrics_consistent_with_s5d': None}
    if restored and val['num_poses'] > 1:
        eval_info = _run_eval(target, gt_path, target.parent)

    meta = {
        'target_path': str(target.relative_to(REPO_ROOT)),
        'restored': restored,
        'source': source,
        'source_path': source_path,
        'num_poses': val['num_poses'],
        'timestamp_status': val['timestamp_status'],
        'gt_leakage_check_passed': val['gt_leakage_check_passed'],
        'quat_valid': val['quat_valid'],
        'metrics_consistent_with_s5d': eval_info.get('metrics_consistent_with_s5d'),
        'eval': eval_info,
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
