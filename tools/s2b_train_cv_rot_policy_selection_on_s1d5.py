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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from eval_clean_policy import (  # type: ignore
    _build_eval_dataset,
    _cfg_from_dict,
    _extract_pairs,
    _load_ckpt_cfg,
    _q,
    _read_debug_summary,
    _safe_float,
    DtBucketScaledMagnitudeModel,
)
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence

CKPT_PATH = REPO_ROOT / 'checkpoints' / 'T57b_no_dt_multiscale_tmag_head_400' / 'final.pt'
POLICY_PATH = REPO_ROOT / 'checkpoints' / 'S1d5_clean_dt_anchor_policy.json'
OUT_ROOT = REPO_ROOT / 'checkpoints' / 'S2b_train_cv_rot_policy_selection_on_s1d5'
REPORT_PATH = REPO_ROOT / 'checkpoints' / 'S2b_train_cv_rot_policy_selection_on_s1d5_report.md'
ROT_VALUES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
STABILITY_VARIANTS = ('default', 'max_eval_batches_off', 'explicit_selected_k')


class ManifestSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices: Sequence[int]) -> None:
        self.base_ds = base_ds
        self.indices = [int(i) for i in indices]
        self._manifest = [base_ds.manifest()[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_ds[self.indices[idx]]

    def manifest(self):
        return self._manifest


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _load_policy() -> Dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding='utf-8'))


def _build_loader(cfg, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, 'pin_memory', False)),
        drop_last=False,
    )


def _load_fine_model(
    device: torch.device,
    *,
    fine_rot: float,
    fine_tdir: float = 0.0,
    fine_tmag: float = 0.0,
    max_eval_batches: int | None = None,
    explicit_selected_k: bool = False,
) -> Tuple[PanoramaRelPoseModel, Any, Dict[str, Any]]:
    cfg_dict = _load_ckpt_cfg(CKPT_PATH)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(fine_rot)
    cfg.fine_tdir_fuse_strength = float(fine_tdir)
    cfg.fine_tmag_fuse_strength = float(fine_tmag)
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    if max_eval_batches is not None:
        cfg.max_eval_batches = int(max_eval_batches)
    if explicit_selected_k:
        cfg.odom_eval_prefer_k = 1
        cfg.odom_eval_fallback_to_min_k = False
        cfg.eval_k_list = (1, 2, 3, 5, 10, 20)
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(CKPT_PATH), map_location=device)
    state = payload.get('model', payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, {'missing': list(msg.missing_keys), 'unexpected': list(msg.unexpected_keys)}


def _path_weighted_ratio(rows: Sequence[Dict[str, Any]]) -> float:
    gt = sum(float(r['tmag_gt']) for r in rows)
    pred = sum(float(r['tmag_pred']) for r in rows)
    return float(pred / max(gt, 1.0e-8))


@dataclass
class EvalRow:
    split: str
    fold: str
    fine_rot: float
    drift: float
    ate: float
    path_ratio: float
    rpe_rot: float
    rpe_trans_dir: float
    rpe_trans_mag: float
    rot: float
    tdir_abs: float
    tdir_local_A_abs: float
    tmag_p10: float
    tmag_p50: float
    tmag_p90: float
    metric_path_ratio: float
    direction_only_path_ratio: float
    path_weighted_tmag_ratio: float
    odom_selected_k: int
    odom_available_k: List[int]
    num_pairs: int
    num_chains: int
    load_missing: int
    load_unexpected: int
    output_dir: str
    eval_variant: str = 'default'


@dataclass
class CandidateAgg:
    fine_rot: float
    fold_rows: List[EvalRow]
    mean_drift: float
    mean_ate: float
    mean_path_ratio: float
    mean_rpe_trans_mag: float
    satisfies: bool


def _run_eval(device: torch.device, ds, *, split: str, fold: str, fine_rot: float, out_dir: Path, eval_variant: str='default') -> EvalRow:
    max_eval_batches = None
    explicit_selected_k = False
    if eval_variant == 'max_eval_batches_off':
        max_eval_batches = 0
    elif eval_variant == 'explicit_selected_k':
        explicit_selected_k = True
    model, cfg, load_summary = _load_fine_model(
        device,
        fine_rot=fine_rot,
        fine_tdir=0.0,
        fine_tmag=0.0,
        max_eval_batches=max_eval_batches,
        explicit_selected_k=explicit_selected_k,
    )
    if load_summary['unexpected']:
        raise RuntimeError(f"{split}/{fold}: unexpected checkpoint keys: {load_summary['unexpected'][:8]}")
    policy = _load_policy()
    factors = {k: float(v) for k, v in policy['effective_bucket_factors'].items()}
    wrapped = DtBucketScaledMagnitudeModel(model, factors)
    loader = _build_loader(cfg, ds)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        wrapped, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(wrapped, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    debug_summary = _read_debug_summary(out_dir)
    base_rows = _extract_pairs(model, ds, device)
    rows = []
    for r in base_rows:
        rec = dict(r)
        dt = float(rec['dt_world'])
        if 0.1 <= dt < 0.3:
            bucket = '[0.1,0.3)'
        elif 0.3 <= dt < 0.5:
            bucket = '[0.3,0.5)'
        elif 0.5 <= dt < 1.0:
            bucket = '[0.5,1)'
        else:
            bucket = None
        factor = float(factors.get(bucket, 1.0))
        rec['tmag_pred'] = float(rec['tmag_pred']) * factor
        rows.append(rec)
    q = _q([float(r['tmag_pred']) for r in rows])
    return EvalRow(
        split=split,
        fold=fold,
        fine_rot=float(fine_rot),
        drift=_safe_float(odom.get('odom_metric_drift')),
        ate=_safe_float(odom.get('odom_metric_ATE')),
        path_ratio=_safe_float(odom.get('odom_shape_metric_mean_path_length_ratio')),
        rpe_rot=_safe_float(odom.get('odom_metric_RPE_rot')),
        rpe_trans_dir=_safe_float(odom.get('odom_metric_RPE_trans_dir')),
        rpe_trans_mag=_safe_float(odom.get('odom_metric_RPE_trans_mag')),
        rot=float(rot),
        tdir_abs=float(tdir_abs),
        tdir_local_A_abs=float(tdir_local_A_abs),
        tmag_p10=float(q['p10']),
        tmag_p50=float(q['p50']),
        tmag_p90=float(q['p90']),
        metric_path_ratio=_safe_float(odom.get('odom_shape_metric_mean_path_length_ratio')),
        direction_only_path_ratio=_safe_float(debug_summary.get('direction_only_mean_path_length_ratio'), 1.0),
        path_weighted_tmag_ratio=_path_weighted_ratio(rows),
        odom_selected_k=int(odom.get('odom_selected_k', -1)),
        odom_available_k=list(odom.get('odom_available_k', [])),
        num_pairs=int(odom.get('odom_num_pairs', 0)),
        num_chains=int(odom.get('odom_num_chains', 0)),
        load_missing=len(load_summary['missing']),
        load_unexpected=len(load_summary['unexpected']),
        output_dir=str(out_dir),
        eval_variant=eval_variant,
    )


def _collect_train_groups(device: torch.device):
    model, cfg, load = _load_fine_model(device, fine_rot=0.0, fine_tdir=0.0, fine_tmag=0.0)
    if load['unexpected']:
        raise RuntimeError(f"base load unexpected keys: {load['unexpected'][:8]}")
    train_ds = _build_eval_dataset(cfg, split='train')
    manifest = train_ds.manifest()
    groups = sorted({(str(m.get('scene')), str(m.get('seq'))) for m in manifest})
    seq_to_indices: Dict[Tuple[str, str], List[int]] = {g: [] for g in groups}
    for idx, meta in enumerate(manifest):
        key = (str(meta.get('scene')), str(meta.get('seq')))
        seq_to_indices[key].append(idx)
    return cfg, train_ds, groups, seq_to_indices


def _candidate_satisfies(path_ratio: float, ate: float, drift: float, baseline_ate: float, baseline_drift: float) -> bool:
    return (
        path_ratio >= 0.90
        and ate <= 1.10 * baseline_ate
        and drift <= 1.10 * baseline_drift
    )


def _select_candidate(cands: Sequence[CandidateAgg]) -> CandidateAgg:
    satisfying = [c for c in cands if c.satisfies]
    if satisfying:
        return sorted(satisfying, key=lambda c: (c.mean_ate, c.mean_drift, abs(c.fine_rot - 0.40)))[0]
    return sorted(cands, key=lambda c: (abs(0.95 - c.mean_path_ratio), c.mean_ate, c.mean_drift, abs(c.fine_rot - 0.40)))[0]


def _write_report(scene_seqs, baseline_cv, cands, selected, final_row, stability_rows, report_path: Path) -> str:
    lines = []
    lines.append('# S2b train-CV rot policy selection on S1d5\n\n')
    verdict = 'FAIL'
    stable = all(
        abs(r.path_ratio - final_row.path_ratio) <= 0.02
        and abs(r.ate - final_row.ate) <= 0.2
        and abs(r.drift - final_row.drift) <= 0.05
        and r.load_unexpected == 0
        for r in stability_rows
    )
    if stable and final_row.path_ratio >= 0.90 and final_row.ate < 7.632463 and final_row.drift <= 1.45 and final_row.load_unexpected == 0:
        verdict = 'SUCCESS'
    elif stable and final_row.path_ratio >= 0.90 and final_row.ate <= 10.5 and final_row.drift <= 1.8 and final_row.load_unexpected == 0:
        verdict = 'PARTIAL'
    lines.append('## Summary verdict\n\n')
    lines.append(f'{verdict}\n\n')
    lines.append('## Setup\n')
    lines.append(f'- branch: `optimize/s2-fine-refinement-on-s1d5`\n')
    lines.append(f'- base checkpoint: `{CKPT_PATH.relative_to(REPO_ROOT)}`\n')
    lines.append(f'- fixed dt-anchor policy: `{POLICY_PATH.relative_to(REPO_ROOT)}`\n')
    lines.append('- fixed alpha / bucket factors = S1d5\n')
    lines.append('- no model parameter training\n')
    lines.append(f'- train groups: {list(scene_seqs)}\n\n')
    lines.append('## Train-CV baseline\n')
    lines.append(f"- S1d5 current clean mainline: fine_rot=0.40, drift=1.396358, ATE=7.632463, path_ratio=0.934982\n")
    lines.append(f"- train rot-only baseline (0.40) mean drift={_fmt(baseline_cv['drift'])}, mean ATE={_fmt(baseline_cv['ate'])}, mean path_ratio={_fmt(baseline_cv['path_ratio'])}\n\n")
    lines.append('## Train-CV grid mean\n')
    lines.append('| fine_rot | mean_drift | mean_ATE | mean_path_ratio | mean_RPE_trans_mag | satisfies |\n')
    lines.append('| ---: | ---: | ---: | ---: | ---: | --- |\n')
    for c in sorted(cands, key=lambda x: x.fine_rot):
        lines.append(f"| {_fmt(c.fine_rot,2)} | {_fmt(c.mean_drift)} | {_fmt(c.mean_ate)} | {_fmt(c.mean_path_ratio)} | {_fmt(c.mean_rpe_trans_mag)} | {c.satisfies} |\n")
    lines.append('\n## Fold details\n')
    lines.append('| fold | fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | selected_k | num_pairs | num_chains | missing | unexpected |\n')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for c in sorted(cands, key=lambda x: x.fine_rot):
        for r in c.fold_rows:
            lines.append(f"| {r.fold} | {_fmt(r.fine_rot,2)} | {_fmt(r.drift)} | {_fmt(r.ate)} | {_fmt(r.path_ratio)} | {_fmt(r.rpe_rot)} | {_fmt(r.rpe_trans_dir)} | {_fmt(r.rpe_trans_mag)} | {r.odom_selected_k} | {r.num_pairs} | {r.num_chains} | {r.load_missing} | {r.load_unexpected} |\n")
    lines.append('\n## Selection\n')
    lines.append(f"- selected fine_rot = {_fmt(selected.fine_rot,2)}\n")
    lines.append(f"- selected equals S2a diagnostic best 0.45 = {abs(selected.fine_rot - 0.45) < 1e-9}\n")
    lines.append('- selection used train-CV only = True\n\n')
    lines.append('## Final test eval\n')
    lines.append(f"- drift = {_fmt(final_row.drift)}\n")
    lines.append(f"- ATE = {_fmt(final_row.ate)}\n")
    lines.append(f"- path_ratio = {_fmt(final_row.path_ratio)}\n")
    lines.append(f"- RPE_rot = {_fmt(final_row.rpe_rot)}\n")
    lines.append(f"- RPE_trans_dir = {_fmt(final_row.rpe_trans_dir)}\n")
    lines.append(f"- RPE_trans_mag = {_fmt(final_row.rpe_trans_mag)}\n")
    lines.append(f"- rot = {_fmt(final_row.rot)}\n")
    lines.append(f"- tdir_abs = {_fmt(final_row.tdir_abs)}\n")
    lines.append(f"- tdir_local_A_abs = {_fmt(final_row.tdir_local_A_abs)}\n")
    lines.append(f"- tmag P10/P50/P90 = {_fmt(final_row.tmag_p10)}/{_fmt(final_row.tmag_p50)}/{_fmt(final_row.tmag_p90)}\n")
    lines.append(f"- metric path_ratio = {_fmt(final_row.metric_path_ratio)}\n")
    lines.append(f"- direction_only path_ratio = {_fmt(final_row.direction_only_path_ratio)}\n")
    lines.append(f"- selected_k = {final_row.odom_selected_k}\n")
    lines.append(f"- available_k = {final_row.odom_available_k}\n")
    lines.append(f"- num_pairs / num_chains = {final_row.num_pairs} / {final_row.num_chains}\n")
    lines.append(f"- load missing/unexpected = {final_row.load_missing}/{final_row.load_unexpected}\n\n")
    lines.append('## Comparison\n')
    lines.append('| row | fine_rot | drift | ATE | path_ratio | status |\n')
    lines.append('| --- | ---: | ---: | ---: | ---: | --- |\n')
    lines.append('| S1d5 current clean mainline | 0.40 | 1.396358 | 7.632463 | 0.934982 | current clean exported mainline |\n')
    lines.append('| S2a diagnostic best | 0.45 | 1.327402 | 7.352371 | 0.934984 | test-swept diagnostic, not clean selection |\n')
    lines.append(f"| S2b train-CV selected candidate | {_fmt(final_row.fine_rot,2)} | {_fmt(final_row.drift)} | {_fmt(final_row.ate)} | {_fmt(final_row.path_ratio)} | train-CV selected |\n")
    lines.append('\n## Stability audit\n')
    lines.append('| variant | drift | ATE | path_ratio | selected_k | num_pairs | num_chains | unexpected |\n')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
    for r in stability_rows:
        lines.append(f"| {r.eval_variant} | {_fmt(r.drift)} | {_fmt(r.ate)} | {_fmt(r.path_ratio)} | {r.odom_selected_k} | {r.num_pairs} | {r.num_chains} | {r.load_unexpected} |\n")
    lines.append('\n## Mainline decision\n')
    lines.append(f'- explicit-cfg / unexpected=0 = {final_row.load_unexpected == 0}\n')
    lines.append(f'- clean replacement for S1d5 = {verdict == "SUCCESS"}\n')
    lines.append('- F1d remains paused = True\n')
    report_path.write_text(''.join(lines), encoding='utf-8')
    return verdict


def _update_valid_baselines(path: Path, final_row: EvalRow, verdict: str) -> None:
    lines = path.read_text(encoding='utf-8').splitlines()
    insert = f"| S2b train-CV rot policy candidate | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean / train-CV selected / no test labels used for selection | {_fmt(final_row.drift)} | {_fmt(final_row.ate)} | {_fmt(final_row.path_ratio)} | {verdict.lower()} clean fine-rot policy candidate |"
    if any('S2b train-CV rot policy candidate' in line for line in lines):
        lines = [insert if 'S2b train-CV rot policy candidate' in line else line for line in lines]
    else:
        lines.insert(3, insert)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, train_ds_full, scene_seqs, seq_to_indices = _collect_train_groups(device)
    test_ds = _build_eval_dataset(cfg, split='test')
    print(f'[S2b] branch assumes optimize/s2-fine-refinement-on-s1d5')
    print(f'[S2b] checkpoint={CKPT_PATH}')
    print(f'[S2b] policy={POLICY_PATH}')
    print(f'[S2b] train groups={scene_seqs}')
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[EvalRow] = []
    cand_map: Dict[float, List[EvalRow]] = {}
    for fold_idx, seq in enumerate(scene_seqs):
        fold_name = f"fold_{fold_idx}_{seq[0]}_{seq[1]}"
        val_ds = ManifestSubsetDataset(train_ds_full, seq_to_indices[seq])
        baseline = _run_eval(device, val_ds, split='train_cv', fold=fold_name, fine_rot=0.40, out_dir=OUT_ROOT / fold_name / 'baseline_rot_0p40')
        baseline_rows.append(baseline)
        for rot in ROT_VALUES:
            row = _run_eval(device, val_ds, split='train_cv', fold=fold_name, fine_rot=float(rot), out_dir=OUT_ROOT / fold_name / f'rot_{str(rot).replace(".","p")}')
            cand_map.setdefault(float(rot), []).append(row)

    baseline_cv = {
        'ate': float(np.mean([r.ate for r in baseline_rows])),
        'drift': float(np.mean([r.drift for r in baseline_rows])),
        'path_ratio': float(np.mean([r.path_ratio for r in baseline_rows])),
    }
    candidates: List[CandidateAgg] = []
    for rot, rows in cand_map.items():
        mean_ate = float(np.mean([r.ate for r in rows]))
        mean_drift = float(np.mean([r.drift for r in rows]))
        mean_path = float(np.mean([r.path_ratio for r in rows]))
        mean_rpe_mag = float(np.mean([r.rpe_trans_mag for r in rows]))
        satisfies = _candidate_satisfies(mean_path, mean_ate, mean_drift, baseline_cv['ate'], baseline_cv['drift'])
        candidates.append(CandidateAgg(fine_rot=rot, fold_rows=rows, mean_drift=mean_drift, mean_ate=mean_ate, mean_path_ratio=mean_path, mean_rpe_trans_mag=mean_rpe_mag, satisfies=satisfies))

    selected = _select_candidate(candidates)
    final_row = _run_eval(device, test_ds, split='test', fold='final_test', fine_rot=selected.fine_rot, out_dir=OUT_ROOT / 'final_test')
    stability_rows = [
        _run_eval(device, test_ds, split='test', fold='final_test', fine_rot=selected.fine_rot, out_dir=OUT_ROOT / f'stability_{variant}', eval_variant=variant)
        for variant in STABILITY_VARIANTS
    ]
    verdict = _write_report(scene_seqs, baseline_cv, candidates, selected, final_row, stability_rows, REPORT_PATH)
    if verdict == 'SUCCESS':
        _update_valid_baselines(REPO_ROOT / 'reports' / 'current_valid_baselines.md', final_row, verdict)
    print(json.dumps({
        'selected_fine_rot': selected.fine_rot,
        'selected_equals_s2a_best_0p45': abs(selected.fine_rot - 0.45) < 1e-9,
        'final_test': {
            'drift': final_row.drift,
            'ATE': final_row.ate,
            'path_ratio': final_row.path_ratio,
        },
        'verdict': verdict,
    }, indent=2))


if __name__ == '__main__':
    main()
