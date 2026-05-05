#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from eval_clean_policy import _build_eval_dataset, _load_policy, _load_fine_model  # type: ignore
from train_mvp import (
    _build_odometry_chains,
    _camera_center_from_T_c0_np,
    _compose_rel_pose_np,
    _rot_geodesic_deg_np,
    _trajectory_shape_summary,
    _vec_angle_deg_np,
)

CKPT_PATH = REPO_ROOT / 'checkpoints' / 'T57b_no_dt_multiscale_tmag_head_400' / 'final.pt'
POLICY_PATH = REPO_ROOT / 'checkpoints' / 'S2b_clean_fine_rot_policy.json'
REPORT_PATH = REPO_ROOT / 'checkpoints' / 'S2c_error_attribution_on_s2b_report.md'


@dataclass
class PairRecord:
    chain_id: int
    scene_seq: str
    step_idx: int
    ds_idx: int
    i: int
    j: int
    dt_world: float
    R_gt: np.ndarray
    t_gt: np.ndarray
    t_gt_mag: float
    t_gt_dir: np.ndarray
    R_pred: np.ndarray
    t_dir_pred: np.ndarray
    t_mag_pred: float


@dataclass
class ChainSummary:
    chain_id: int
    scene_seq: str
    num_steps: int
    gt_path_length: float
    pred_path_length: float
    path_ratio: float
    ate: float
    drift: float
    mean_rot_err: float
    mean_tdir_err: float
    mean_tmag_ratio: float
    mean_turn_err: float
    max_step_error: float
    worst_step_index: int


def _safe_float(v: Any, default: float = float('nan')) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _q(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'p10': float('nan'), 'p50': float('nan'), 'p90': float('nan')}
    return {
        'p10': float(np.percentile(arr, 10)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
    }


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float('nan')
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _bucket(dt: float) -> str | None:
    if 0.1 <= dt < 0.3:
        return '[0.1,0.3)'
    if 0.3 <= dt < 0.5:
        return '[0.3,0.5)'
    if 0.5 <= dt < 1.0:
        return '[0.5,1)'
    return None


def _load_pairs() -> Tuple[List[List[PairRecord]], Dict[str, Any]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = _load_policy(POLICY_PATH)
    model, cfg, load_summary = _load_fine_model(
        CKPT_PATH,
        device,
        fine_rot=float(policy['fine_rot_fuse_strength']),
        fine_tdir=float(policy['fine_tdir_fuse_strength']),
        fine_tmag=float(policy['fine_tmag_fuse_strength']),
        max_eval_batches=None,
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get('use_geometry_refine', False))
    factors = {k: float(v) for k, v in policy['effective_bucket_factors'].items()}
    ds = _build_eval_dataset(cfg, split='test')
    manifest = ds.manifest()
    chains_meta = _build_odometry_chains(manifest, selected_k=1, max_pairs=0)
    chains: List[List[PairRecord]] = []
    with torch.no_grad():
        for chain_id, chain in enumerate(chains_meta):
            scene_seq = f"{chain['scene_seq'][0]}/{chain['scene_seq'][1]}"
            out_chain: List[PairRecord] = []
            for step_idx, item in enumerate(chain['pairs']):
                ds_idx = int(item['_ds_idx'])
                sample = ds[ds_idx]
                IA = sample['IA'].unsqueeze(0).to(device, non_blocking=True)
                IB = sample['IB'].unsqueeze(0).to(device, non_blocking=True)
                dt_world = float(item.get('dt_world', sample.get('t_gt_mag', 0.0)))
                dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
                R_pred_t, t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                R_pred = R_pred_t.detach().float().cpu().numpy()[0]
                t_dir_pred_t = aux.get('t_dir_out', t_pred_t) if isinstance(aux, dict) else t_pred_t
                t_dir_pred = torch.nn.functional.normalize(t_dir_pred_t.detach().float(), dim=-1, eps=1e-6).cpu().numpy()[0]
                t_mag_pred = float(aux['t_mag'].detach().float().view(-1).cpu().numpy()[0])
                factor = float(factors.get(_bucket(dt_world), 1.0))
                t_mag_pred *= factor
                R_gt = sample['R_gt'].float().numpy()
                if sample.get('t_gt_vec', None) is not None:
                    t_gt = sample['t_gt_vec'].float().numpy()
                else:
                    t_gt_dir = sample['t_gt_dir'].float().numpy()
                    t_gt_mag = float(sample['t_gt_mag'])
                    t_gt = t_gt_dir * t_gt_mag
                t_gt_mag = float(np.linalg.norm(t_gt))
                t_gt_dir = t_gt / max(t_gt_mag, 1e-12)
                out_chain.append(PairRecord(
                    chain_id=chain_id,
                    scene_seq=scene_seq,
                    step_idx=step_idx,
                    ds_idx=ds_idx,
                    i=int(item['i']),
                    j=int(item['j']),
                    dt_world=dt_world,
                    R_gt=R_gt.astype(np.float64),
                    t_gt=t_gt.astype(np.float64),
                    t_gt_mag=float(t_gt_mag),
                    t_gt_dir=t_gt_dir.astype(np.float64),
                    R_pred=R_pred.astype(np.float64),
                    t_dir_pred=t_dir_pred.astype(np.float64),
                    t_mag_pred=float(t_mag_pred),
                ))
            if out_chain:
                chains.append(out_chain)
    meta = {
        'load_missing': len(load_summary['missing']),
        'load_unexpected': len(load_summary['unexpected']),
        'selected_k': 1,
        'available_k': [1, 2, 3, 5, 10, 20],
        'num_pairs': sum(len(c) for c in chains),
        'num_chains': len(chains),
    }
    return chains, meta


def _variant_components(rec: PairRecord, variant: str) -> Tuple[np.ndarray, np.ndarray, float]:
    use_gt_R = variant in {'oracle_R', 'oracle_R_tdir', 'oracle_all'}
    use_gt_tdir = variant in {'oracle_tdir', 'oracle_R_tdir', 'oracle_all'}
    use_gt_tmag = variant in {'oracle_tmag', 'oracle_all'}
    R = rec.R_gt if use_gt_R else rec.R_pred
    tdir = rec.t_gt_dir if use_gt_tdir else rec.t_dir_pred
    tmag = rec.t_gt_mag if use_gt_tmag else rec.t_mag_pred
    return R, tdir, float(tmag)


def _evaluate_variant(chains: List[List[PairRecord]], variant: str) -> Dict[str, Any]:
    step_pos_err_sq: List[float] = []
    step_pos_err: List[float] = []
    rot_errs: List[float] = []
    tdir_errs: List[float] = []
    tmag_abs_errs: List[float] = []
    chain_summaries: List[ChainSummary] = []
    worst_tdir_steps: List[Dict[str, Any]] = []
    worst_rot_steps: List[Dict[str, Any]] = []

    for chain in chains:
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pred_c0 = np.eye(3, dtype=np.float64)
        t_pred_c0 = np.zeros(3, dtype=np.float64)
        gt_positions = [np.zeros(3, dtype=np.float64)]
        pred_positions = [np.zeros(3, dtype=np.float64)]
        chain_rows = []
        for rec in chain:
            R_use, tdir_use, tmag_use = _variant_components(rec, variant)
            tvec_use = tdir_use * tmag_use
            R_gt_c0, t_gt_c0 = _compose_rel_pose_np(rec.R_gt, rec.t_gt, R_gt_c0, t_gt_c0)
            R_pred_c0, t_pred_c0 = _compose_rel_pose_np(R_use, tvec_use, R_pred_c0, t_pred_c0)
            p_gt = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            p_pred = _camera_center_from_T_c0_np(R_pred_c0, t_pred_c0)
            pos_err = float(np.linalg.norm(p_pred - p_gt))
            gt_step = rec.t_gt
            pred_step = tvec_use
            rot_err = 0.0 if variant in {'oracle_R', 'oracle_R_tdir', 'oracle_all'} else _rot_geodesic_deg_np(rec.R_pred, rec.R_gt)
            tdir_err = 0.0 if variant in {'oracle_tdir', 'oracle_R_tdir', 'oracle_all'} else _vec_angle_deg_np(pred_step, gt_step)
            tmag_abs_err = abs(tmag_use - rec.t_gt_mag)
            tmag_ratio = float(tmag_use / max(rec.t_gt_mag, 1e-12))
            chain_rows.append({
                'step_idx': rec.step_idx,
                'i': rec.i,
                'j': rec.j,
                'dt_gt': rec.t_gt_mag,
                'gt_x': float(p_gt[0]), 'gt_y': float(p_gt[1]), 'gt_z': float(p_gt[2]),
                'metric_x': float(p_pred[0]), 'metric_y': float(p_pred[1]), 'metric_z': float(p_pred[2]),
                'metric_pos_err': pos_err,
                'rot_err_deg': rot_err,
                'tdir_err_deg': tdir_err,
                'tmag_ratio': tmag_ratio,
                'tmag_rel_err': float(tmag_abs_err / max(rec.t_gt_mag, 1e-12)),
            })
            step_pos_err.append(pos_err)
            step_pos_err_sq.append(pos_err ** 2)
            rot_errs.append(rot_err)
            tdir_errs.append(tdir_err)
            tmag_abs_errs.append(tmag_abs_err)
            gt_positions.append(p_gt)
            pred_positions.append(p_pred)
            worst_tdir_steps.append({
                'scene_seq': rec.scene_seq,
                'chain_id': rec.chain_id,
                'step_idx': rec.step_idx,
                'ds_idx': rec.ds_idx,
                'i': rec.i,
                'j': rec.j,
                'tdir_err_deg': tdir_err,
                'rot_err_deg': rot_err,
                'pos_err': pos_err,
            })
            worst_rot_steps.append({
                'scene_seq': rec.scene_seq,
                'chain_id': rec.chain_id,
                'step_idx': rec.step_idx,
                'ds_idx': rec.ds_idx,
                'i': rec.i,
                'j': rec.j,
                'rot_err_deg': rot_err,
                'tdir_err_deg': tdir_err,
                'pos_err': pos_err,
            })
        shape = _trajectory_shape_summary(chain_rows, 'metric')
        ate = float(math.sqrt(np.mean([r['metric_pos_err'] ** 2 for r in chain_rows]))) if chain_rows else float('nan')
        drift = float(chain_rows[-1]['metric_pos_err']) if chain_rows else float('nan')
        max_step = max(chain_rows, key=lambda r: r['metric_pos_err']) if chain_rows else None
        chain_summaries.append(ChainSummary(
            chain_id=chain[0].chain_id,
            scene_seq=chain[0].scene_seq,
            num_steps=len(chain_rows),
            gt_path_length=float(shape.get('gt_path_length', float('nan'))),
            pred_path_length=float(shape.get('pred_path_length', float('nan'))),
            path_ratio=float(shape.get('path_length_ratio', float('nan'))),
            ate=ate,
            drift=drift,
            mean_rot_err=float(np.mean([r['rot_err_deg'] for r in chain_rows])) if chain_rows else float('nan'),
            mean_tdir_err=float(np.mean([r['tdir_err_deg'] for r in chain_rows])) if chain_rows else float('nan'),
            mean_tmag_ratio=float(np.mean([r['tmag_ratio'] for r in chain_rows])) if chain_rows else float('nan'),
            mean_turn_err=float(shape.get('mean_turn_abs_err_deg', float('nan'))),
            max_step_error=float(max_step['metric_pos_err']) if max_step else float('nan'),
            worst_step_index=int(max_step['step_idx']) if max_step else -1,
        ))

    by_chain_ate = [c.ate for c in chain_summaries]
    by_chain_drift = [c.drift for c in chain_summaries]
    by_chain_path = [c.path_ratio for c in chain_summaries]
    gt_total_path = float(sum(c.gt_path_length for c in chain_summaries))
    pred_total_path = float(sum(c.pred_path_length for c in chain_summaries))
    weighted_path_ratio = float(pred_total_path / max(gt_total_path, 1e-12)) if chain_summaries else float('nan')
    return {
        'variant': variant,
        'drift': float(np.mean(by_chain_drift)) if by_chain_drift else float('nan'),
        'ATE': float(math.sqrt(np.mean(step_pos_err_sq))) if step_pos_err_sq else float('nan'),
        'path_ratio': weighted_path_ratio,
        'RPE_rot': float(np.mean(rot_errs)) if rot_errs else float('nan'),
        'RPE_trans_dir': float(np.mean(tdir_errs)) if tdir_errs else float('nan'),
        'RPE_trans_mag': float(np.mean(tmag_abs_errs)) if tmag_abs_errs else float('nan'),
        'mean_step_position_error': float(np.mean(step_pos_err)) if step_pos_err else float('nan'),
        'chain_level_ATE': float(np.mean(by_chain_ate)) if by_chain_ate else float('nan'),
        'chain_level_drift': float(np.mean(by_chain_drift)) if by_chain_drift else float('nan'),
        'chain_level_path_ratio': float(np.mean(by_chain_path)) if by_chain_path else float('nan'),
        'gt_total_path_length': gt_total_path,
        'pred_total_path_length': pred_total_path,
        'chain_summaries': chain_summaries,
        'worst_tdir_steps': sorted(worst_tdir_steps, key=lambda r: r['tdir_err_deg'], reverse=True)[:10],
        'worst_rot_steps': sorted(worst_rot_steps, key=lambda r: r['rot_err_deg'], reverse=True)[:10],
    }


def _classify(results: Dict[str, Dict[str, Any]], corrs: Dict[str, float]) -> Tuple[str, str]:
    cur = results['S2b current']
    imp_R = cur['ATE'] - results['oracle_R']['ATE']
    imp_tdir = cur['ATE'] - results['oracle_tdir']['ATE']
    imp_tmag = cur['ATE'] - results['oracle_tmag']['ATE']
    imp_all = cur['ATE'] - results['oracle_all']['ATE']

    if imp_tmag > max(imp_R, imp_tdir) * 1.2 and imp_tmag > 0.2:
        return 'SCALE-RESIDUAL', '回查 dt-anchor'
    if imp_tdir > imp_R * 1.2 and imp_tdir > 0.2:
        return 'TDIR-DOMINANT', 'S2d fine_tdir eval-only sweep'
    if imp_R > imp_tdir * 1.2 and imp_R > 0.2:
        return 'ROTATION-DOMINANT', '继续 fine_rot / rotation consistency'
    if imp_all > 0.2 and (imp_R > 0.1 or imp_tdir > 0.1):
        return 'MIXED-ROT-TDIR', 'fine_rot + fine_tdir joint sweep'
    if abs(corrs.get('corr_ATE_path_ratio_error', float('nan'))) > 0.7 and imp_tmag > 0.1:
        return 'SCALE-RESIDUAL', '回查 dt-anchor'
    return 'CHAIN/FRAME-CONVENTION-ISSUE', '查 frame convention / integration'


def main() -> None:
    chains, meta = _load_pairs()
    official_summary = json.loads((REPO_ROOT / 'checkpoints' / 'S2b_final_repro' / 's1d5_policy_eval_summary.json').read_text(encoding='utf-8'))
    variants = [
        'S2b current',
        'oracle_R',
        'oracle_tdir',
        'oracle_R_tdir',
        'oracle_tmag',
        'oracle_all',
    ]
    result_map = {v: _evaluate_variant(chains, v) for v in variants}
    baseline = result_map['S2b current']
    chain_rows = baseline['chain_summaries']
    corrs = {
        'corr_ATE_tdir_error': _corr([c.ate for c in chain_rows], [c.mean_tdir_err for c in chain_rows]),
        'corr_ATE_rot_error': _corr([c.ate for c in chain_rows], [c.mean_rot_err for c in chain_rows]),
        'corr_ATE_path_ratio_error': _corr([c.ate for c in chain_rows], [abs(c.path_ratio - 1.0) for c in chain_rows]),
        'corr_ATE_turn_error': _corr([c.ate for c in chain_rows], [c.mean_turn_err for c in chain_rows]),
        'corr_drift_tdir_error': _corr([c.drift for c in chain_rows], [c.mean_tdir_err for c in chain_rows]),
        'corr_drift_rot_error': _corr([c.drift for c in chain_rows], [c.mean_rot_err for c in chain_rows]),
    }
    label, next_step = _classify(result_map, corrs)

    lines: List[str] = []
    lines.append('# S2c Error Attribution On S2b\n\n')
    lines.append('## S2b baseline\n')
    lines.append(f"- base checkpoint: `{CKPT_PATH.relative_to(REPO_ROOT)}`\n")
    lines.append(f"- policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`\n")
    lines.append(f"- explicit-cfg / unexpected=0: {meta['load_unexpected'] == 0}\n")
    lines.append(f"- selected_k / num_pairs / num_chains = {meta['selected_k']} / {meta['num_pairs']} / {meta['num_chains']}\n")
    lines.append(f"- drift = {_fmt(baseline['drift'])}\n")
    lines.append(f"- ATE = {_fmt(baseline['ATE'])}\n")
    lines.append(f"- official reported path_ratio = {_fmt(_safe_float(official_summary.get('metric_path_ratio')))}\n")
    lines.append(f"- all-chain weighted path_ratio = {_fmt(baseline['path_ratio'])}\n")
    lines.append(f"- all-chain mean path_ratio = {_fmt(baseline['chain_level_path_ratio'])}\n")
    lines.append(f"- RPE_rot = {_fmt(baseline['RPE_rot'])}\n")
    lines.append(f"- RPE_trans_dir = {_fmt(baseline['RPE_trans_dir'])}\n")
    lines.append(f"- RPE_trans_mag = {_fmt(baseline['RPE_trans_mag'])}\n\n")

    lines.append('## Oracle variant comparison\n')
    lines.append('| variant | drift | ATE | weighted_path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | mean_step_pos_err | chain_ATE | chain_drift | chain_mean_path_ratio |\n')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for v in variants:
        r = result_map[v]
        lines.append(f"| {v} | {_fmt(r['drift'])} | {_fmt(r['ATE'])} | {_fmt(r['path_ratio'])} | {_fmt(r['RPE_rot'])} | {_fmt(r['RPE_trans_dir'])} | {_fmt(r['RPE_trans_mag'])} | {_fmt(r['mean_step_position_error'])} | {_fmt(r['chain_level_ATE'])} | {_fmt(r['chain_level_drift'])} | {_fmt(r['chain_level_path_ratio'])} |\n")
    lines.append('\n')

    top_ate = sorted(chain_rows, key=lambda c: c.ate, reverse=True)[:10]
    top_drift = sorted(chain_rows, key=lambda c: c.drift, reverse=True)[:10]
    lines.append('## Chain-level breakdown\n')
    lines.append('| chain_id | scene_seq | num_steps | gt_path_length | pred_path_length | path_ratio | ATE | drift | mean_rot_error | mean_tdir_error | mean_tmag_ratio | mean_turn_error | max_step_error | worst_step_index |\n')
    lines.append('| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for c in chain_rows:
        lines.append(f"| {c.chain_id} | {c.scene_seq} | {c.num_steps} | {_fmt(c.gt_path_length)} | {_fmt(c.pred_path_length)} | {_fmt(c.path_ratio)} | {_fmt(c.ate)} | {_fmt(c.drift)} | {_fmt(c.mean_rot_err)} | {_fmt(c.mean_tdir_err)} | {_fmt(c.mean_tmag_ratio)} | {_fmt(c.mean_turn_err)} | {_fmt(c.max_step_error)} | {c.worst_step_index} |\n")
    lines.append('\n## Worst cases summary\n')
    lines.append('### Top 10 highest ATE chains\n')
    for c in top_ate:
        lines.append(f"- chain {c.chain_id} `{c.scene_seq}`: ATE={_fmt(c.ate)}, drift={_fmt(c.drift)}, path_ratio={_fmt(c.path_ratio)}, mean_tdir_err={_fmt(c.mean_tdir_err)}, mean_rot_err={_fmt(c.mean_rot_err)}\n")
    lines.append('\n### Top 10 highest drift chains\n')
    for c in top_drift:
        lines.append(f"- chain {c.chain_id} `{c.scene_seq}`: drift={_fmt(c.drift)}, ATE={_fmt(c.ate)}, path_ratio={_fmt(c.path_ratio)}, mean_tdir_err={_fmt(c.mean_tdir_err)}, mean_rot_err={_fmt(c.mean_rot_err)}\n")
    lines.append('\n### Top 10 highest tdir error steps\n')
    for s in baseline['worst_tdir_steps']:
        lines.append(f"- chain {s['chain_id']} step {s['step_idx']} ({s['scene_seq']} i={s['i']} j={s['j']}): tdir_err={_fmt(s['tdir_err_deg'])}, rot_err={_fmt(s['rot_err_deg'])}, pos_err={_fmt(s['pos_err'])}\n")
    lines.append('\n### Top 10 highest rot error steps\n')
    for s in baseline['worst_rot_steps']:
        lines.append(f"- chain {s['chain_id']} step {s['step_idx']} ({s['scene_seq']} i={s['i']} j={s['j']}): rot_err={_fmt(s['rot_err_deg'])}, tdir_err={_fmt(s['tdir_err_deg'])}, pos_err={_fmt(s['pos_err'])}\n")
    lines.append('\n## Correlation analysis\n')
    for k, v in corrs.items():
        lines.append(f"- {k} = {_fmt(v)}\n")
    lines.append('\n## Conclusion\n')
    lines.append(f'- classification: `{label}`\n')
    lines.append(f'- next step: `{next_step}`\n')
    REPORT_PATH.write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'classification': label, 'next_step': next_step, 'baseline_ATE': baseline['ATE']}, indent=2))


if __name__ == '__main__':
    main()
