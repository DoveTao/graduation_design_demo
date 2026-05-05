#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import (  # type: ignore
    DtBucketScaledMagnitudeModel,
    _build_eval_dataset,
    _cfg_from_dict,
    _extract_pairs,
    _load_ckpt_cfg,
    _q,
    _read_debug_summary,
    _safe_float,
)
from model import PanoramaRelPoseModel
from train_mvp import eval_model, eval_odometry_sequence


CKPT_PATH = REPO_ROOT / "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
POLICY_PATH = REPO_ROOT / "checkpoints/S2b_clean_fine_rot_policy.json"
OUT_ROOT = REPO_ROOT / "checkpoints/S2g_train_cv_dt_aware_fine_rot_policy_selection_on_s2b"
REPORT_PATH = REPO_ROOT / "checkpoints/S2g_train_cv_dt_aware_fine_rot_policy_selection_on_s2b_report.md"
POLICY_EXPORT_PATH = REPO_ROOT / "checkpoints/S2g_clean_rot_policy.json"
VALID_BASELINES_PATH = REPO_ROOT / "reports/current_valid_baselines.md"


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


def _dt_bucket(dt: float) -> str | None:
    if 0.1 <= dt < 0.3:
        return "[0.1,0.3)"
    if 0.3 <= dt < 0.5:
        return "[0.3,0.5)"
    if 0.5 <= dt < 1.0:
        return "[0.5,1)"
    return None


@dataclass
class Candidate:
    name: str
    policy_type: str
    mapping: Dict[str, float]
    simplicity_rank: int


@dataclass
class EvalRow:
    candidate_name: str
    policy_type: str
    split: str
    fold: str
    mapping: Dict[str, float]
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
    odom_selected_k: int
    odom_available_k: List[int]
    num_pairs: int
    num_chains: int
    load_missing: int
    load_unexpected: int
    output_dir: str


def _candidates() -> List[Candidate]:
    return [
        Candidate("G0_global_0p45", "global", {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.45, "[0.5,1)": 0.45}, 0),
        Candidate("G1_global_0p475", "global", {"[0.1,0.3)": 0.475, "[0.3,0.5)": 0.475, "[0.5,1)": 0.475}, 0),
        Candidate("G2_global_0p50", "global", {"[0.1,0.3)": 0.50, "[0.3,0.5)": 0.50, "[0.5,1)": 0.50}, 0),
        Candidate("D0_conservative_small_dt", "dt-aware", {"[0.1,0.3)": 0.40, "[0.3,0.5)": 0.45, "[0.5,1)": 0.50}, 2),
        Candidate("D1_aggressive_large_dt", "dt-aware", {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.50, "[0.5,1)": 0.55}, 3),
        Candidate("D2_mild_large_dt", "dt-aware", {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.475, "[0.5,1)": 0.50}, 1),
        Candidate("D3_conservative_large_dt", "dt-aware", {"[0.1,0.3)": 0.45, "[0.3,0.5)": 0.45, "[0.5,1)": 0.40}, 2),
        Candidate("D4_smooth_ramp", "dt-aware", {"[0.1,0.3)": 0.425, "[0.3,0.5)": 0.475, "[0.5,1)": 0.525}, 1),
    ]


class RotPolicyWrapper(torch.nn.Module):
    def __init__(self, base: DtBucketScaledMagnitudeModel, rot_policy: Callable[[float], float]) -> None:
        super().__init__()
        self.base = base
        self.rot_policy = rot_policy
        self.cfg = base.cfg

    def forward(self, IA, IB, *, enable_depth_fusion=None, dt_world=None):
        old_rot = float(getattr(self.base.base.cfg, "fine_rot_fuse_strength", 0.45))
        dt_val = float(dt_world.detach().float().view(-1)[0].cpu().item()) if dt_world is not None else 0.0
        new_rot = float(self.rot_policy(dt_val))
        self.base.base.cfg.fine_rot_fuse_strength = new_rot
        try:
            return self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        finally:
            self.base.base.cfg.fine_rot_fuse_strength = old_rot


def _load_base_cfg():
    cfg_dict = _load_ckpt_cfg(CKPT_PATH)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    return cfg


def _load_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    cfg = _load_base_cfg()
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(CKPT_PATH), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_wrapped_model(device: torch.device, mapping: Dict[str, float]):
    base_model, load_summary = _load_model(device)
    cfg = base_model.cfg
    base_policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    factors = {str(k): float(v) for k, v in base_policy["effective_bucket_factors"].items()}
    scaled = DtBucketScaledMagnitudeModel(base_model, factors).to(device)

    def rot_policy(dt: float) -> float:
        return float(mapping.get(_dt_bucket(dt), 0.45))

    wrapped = RotPolicyWrapper(scaled, rot_policy).to(device)
    return wrapped, cfg, load_summary


def _build_loader(cfg, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        drop_last=False,
    )


def _path_weighted_ratio(rows: Sequence[Dict[str, Any]]) -> float:
    gt = sum(float(r["tmag_gt"]) for r in rows)
    pred = sum(float(r["tmag_pred"]) for r in rows)
    return float(pred / max(gt, 1.0e-8))


def _run_eval(device: torch.device, ds, candidate: Candidate, *, split: str, fold: str, out_dir: Path) -> EvalRow:
    wrapped, cfg, load_summary = _build_wrapped_model(device, candidate.mapping)
    if load_summary["unexpected"]:
        raise RuntimeError(f"{candidate.name}: unexpected checkpoint keys: {load_summary['unexpected'][:8]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = _build_loader(cfg, ds)
    rot, _tdir, tdir_abs, _local, tdir_local_A_abs, _diag, _msg, _msg_local, _vis, _bk, _bdt, _bkdt, _bmsg = eval_model(
        wrapped, loader, device, cfg, collect_vis=False
    )
    odom = eval_odometry_sequence(wrapped, ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    debug_summary = _read_debug_summary(out_dir)

    # tmag quantiles from pair predictions under the selected rot policy
    base_model, _ = _load_model(device)
    base_policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    factors = {str(k): float(v) for k, v in base_policy["effective_bucket_factors"].items()}
    scaled = DtBucketScaledMagnitudeModel(base_model, factors).to(device)
    rot_wrap = RotPolicyWrapper(scaled, lambda dt: float(candidate.mapping.get(_dt_bucket(dt), 0.45))).to(device)
    rows = _extract_pairs(rot_wrap, ds, device)
    q = _q([float(r["tmag_pred"]) for r in rows])

    return EvalRow(
        candidate_name=candidate.name,
        policy_type=candidate.policy_type,
        split=split,
        fold=fold,
        mapping=dict(candidate.mapping),
        drift=_safe_float(odom.get("odom_metric_drift")),
        ate=_safe_float(odom.get("odom_metric_ATE")),
        path_ratio=_safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        rpe_rot=_safe_float(odom.get("odom_metric_RPE_rot")),
        rpe_trans_dir=_safe_float(odom.get("odom_metric_RPE_trans_dir")),
        rpe_trans_mag=_safe_float(odom.get("odom_metric_RPE_trans_mag")),
        rot=float(rot),
        tdir_abs=float(tdir_abs),
        tdir_local_A_abs=float(tdir_local_A_abs),
        tmag_p10=float(q["p10"]),
        tmag_p50=float(q["p50"]),
        tmag_p90=float(q["p90"]),
        odom_selected_k=int(odom.get("odom_selected_k", -1)),
        odom_available_k=list(odom.get("odom_available_k", [])),
        num_pairs=int(odom.get("odom_num_pairs", 0)),
        num_chains=int(odom.get("odom_num_chains", 0)),
        load_missing=len(load_summary["missing"]),
        load_unexpected=len(load_summary["unexpected"]),
        output_dir=str(out_dir.relative_to(REPO_ROOT)),
    )


def _collect_train_groups(device: torch.device):
    cfg = _load_base_cfg()
    train_ds = _build_eval_dataset(cfg, split="train")
    manifest = train_ds.manifest()
    groups = sorted({(str(m.get("scene")), str(m.get("seq"))) for m in manifest})
    seq_to_indices: Dict[Tuple[str, str], List[int]] = {g: [] for g in groups}
    for idx, meta in enumerate(manifest):
        key = (str(meta.get("scene")), str(meta.get("seq")))
        seq_to_indices[key].append(idx)
    return cfg, train_ds, groups, seq_to_indices


def _mean(rows: Sequence[EvalRow], attr: str) -> float:
    vals = [float(getattr(r, attr)) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _select_candidate(cands: List[Tuple[Candidate, List[EvalRow]]]) -> Tuple[Candidate, List[EvalRow]]:
    eligible = []
    for cand, rows in cands:
        mean_ate = _mean(rows, "ate")
        mean_drift = _mean(rows, "drift")
        mean_path = _mean(rows, "path_ratio")
        if mean_path >= 0.90 and all(r.load_unexpected == 0 for r in rows):
            eligible.append((cand, rows, mean_ate, mean_drift))
    if not eligible:
        return min(cands, key=lambda cr: (_mean(cr[1], "ate"), _mean(cr[1], "drift")))

    # primary rank: mean CV ATE
    eligible.sort(key=lambda x: x[2])
    best_ate = eligible[0][2]
    close_ate = [x for x in eligible if abs(x[2] - best_ate) < 0.03]
    if len(close_ate) == 1:
        return close_ate[0][0], close_ate[0][1]
    close_ate.sort(key=lambda x: x[3])
    best_drift = close_ate[0][3]
    close_drift = [x for x in close_ate if abs(x[3] - best_drift) < 1e-9 or abs(x[3] - best_drift) < 0.01]
    if len(close_drift) == 1:
        return close_drift[0][0], close_drift[0][1]
    close_drift.sort(key=lambda x: (x[0].simplicity_rank, abs(x[0].mapping.get("[0.3,0.5)", 0.45) - 0.475), x[2], x[3]))
    return close_drift[0][0], close_drift[0][1]


def _verdict(final_row: EvalRow) -> str:
    if final_row.load_unexpected != 0:
        return "INVALID"
    if final_row.path_ratio >= 0.90 and final_row.ate < 7.352371 and final_row.drift <= 1.35:
        return "SUCCESS"
    if final_row.path_ratio >= 0.90 and final_row.ate <= 7.40 and final_row.drift <= 1.36:
        return "PARTIAL"
    return "FAIL"


def _write_report(
    train_groups,
    candidates: List[Candidate],
    fold_rows: Dict[str, List[EvalRow]],
    mean_rank_rows: List[Tuple[Candidate, float, float, float]],
    selected: Candidate,
    final_row: EvalRow,
    verdict: str,
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# S2g Train-CV Dt-Aware Fine Rot Policy Selection On S2b\n\n")
    lines.append("## Candidate policy definitions\n")
    for c in candidates:
        lines.append(f"- `{c.name}` ({c.policy_type}): `{c.mapping}`\n")
    lines.append("\n## Train-CV setup\n")
    lines.append(f"- train_only_leave_one_train_seq_out_cv groups: `{list(train_groups)}`\n")
    lines.append("- no model parameter training\n")
    lines.append("- no test labels used for selection\n\n")

    lines.append("## Train-CV fold table\n")
    lines.append("| candidate | fold | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | selected_k | num_pairs | num_chains | missing | unexpected |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for c in candidates:
        for r in fold_rows[c.name]:
            lines.append(
                f"| {c.name} | {r.fold} | {_fmt(r.drift)} | {_fmt(r.ate)} | {_fmt(r.path_ratio)} | {_fmt(r.rpe_rot)} | {_fmt(r.rpe_trans_dir)} | {_fmt(r.rpe_trans_mag)} | {_fmt(r.rot)} | {_fmt(r.tdir_abs)} | {_fmt(r.tdir_local_A_abs)} | {r.odom_selected_k} | {r.num_pairs} | {r.num_chains} | {r.load_missing} | {r.load_unexpected} |\n"
            )
    lines.append("\n## Mean CV ranking\n")
    lines.append("| candidate | type | mapping | mean_drift | mean_ATE | mean_path_ratio | simpler_rank |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
    for c, md, ma, mp in mean_rank_rows:
        lines.append(f"| {c.name} | {c.policy_type} | `{c.mapping}` | {_fmt(md)} | {_fmt(ma)} | {_fmt(mp)} | {c.simplicity_rank} |\n")
    lines.append("\n## Selected policy\n")
    lines.append(f"- selected policy name: `{selected.name}`\n")
    lines.append(f"- policy type: `{selected.policy_type}`\n")
    lines.append(f"- selected mapping: `{selected.mapping}`\n")
    lines.append(f"- selected by train-CV only: `True`\n")
    lines.append(f"- matches S2f diagnostic global 0.475: `{selected.name == 'G1_global_0p475'}`\n")
    lines.append(f"- matches S2f diagnostic dt-aware aggressive_large_dt: `{selected.name == 'D1_aggressive_large_dt'}`\n")
    lines.append("\n## Final test result\n")
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
    lines.append(f"- selected_k = {final_row.odom_selected_k}\n")
    lines.append(f"- available_k = {final_row.odom_available_k}\n")
    lines.append(f"- num_pairs / num_chains = {final_row.num_pairs} / {final_row.num_chains}\n")
    lines.append(f"- missing / unexpected = {final_row.load_missing} / {final_row.load_unexpected}\n")
    lines.append("\n## Comparison\n")
    lines.append("| method | drift | ATE | path_ratio | status |\n")
    lines.append("| --- | ---: | ---: | ---: | --- |\n")
    lines.append("| S1d5 | 1.396358 | 7.632463 | 0.934982 | previous clean exported baseline |\n")
    lines.append("| S2b | 1.327402 | 7.352371 | 0.934984 | current clean fine-rot candidate |\n")
    lines.append("| S2f diagnostic global 0.475 | 1.331278 | 7.309820 | 0.934984 | test-swept diagnostic |\n")
    lines.append("| S2f diagnostic dt-aware aggressive_large_dt | 1.332847 | 7.345742 | 0.934983 | test-swept diagnostic |\n")
    lines.append(f"| S2g selected | {_fmt(final_row.drift)} | {_fmt(final_row.ate)} | {_fmt(final_row.path_ratio)} | train-CV selected |\n")
    lines.append("\n## Verdict\n")
    lines.append(f"- `{verdict}`\n")
    lines.append(f"- selected policy improves over S2b = `{final_row.ate < 7.352371 and final_row.path_ratio >= 0.90 and final_row.drift <= 1.35}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def _export_policy(selected: Candidate) -> None:
    base = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload = {
        "name": "S2g_clean_rot_policy",
        "base_checkpoint_path": base["base_checkpoint_path"],
        "inherits_policy": "checkpoints/S2b_clean_fine_rot_policy.json",
        "parent": "S2b clean fine-rot policy candidate",
        "cfg_restore_requirement": "explicit-cfg / unexpected=0",
        "bucket_edges": base["bucket_edges"],
        "bucket_labels": base["bucket_labels"],
        "effective_bucket_factors": base["effective_bucket_factors"],
        "fine_rot_mapping": selected.mapping,
        "fine_tdir_fuse_strength": 0.0,
        "fine_tmag_fuse_strength": 0.0,
        "use_geometry_refine": False,
        "selection_source": "train_only_leave_one_train_seq_out_cv",
        "test_labels_used_for_selection": False,
        "status": "clean rot-policy candidate",
        "notes": [
            f"Selected by S2g train-CV from candidate `{selected.name}`.",
            "No model parameter training.",
            "Do not use test labels to modify the fine_rot mapping.",
        ],
    }
    POLICY_EXPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_valid_baselines(selected: Candidate, final_row: EvalRow) -> None:
    lines = VALID_BASELINES_PATH.read_text(encoding="utf-8").splitlines()
    insert = (
        f"| S2g train-CV selected dt-aware/global fine-rot policy | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | "
        f"explicit-cfg / unexpected=0 / clean / train-CV selected / no test labels used for selection | "
        f"{_fmt(final_row.drift)} | {_fmt(final_row.ate)} | {_fmt(final_row.path_ratio)} | current clean rot-policy candidate |"
    )
    lines = [line for line in lines if "S2g train-CV selected dt-aware/global fine-rot policy" not in line]
    lines.insert(3, insert)
    for idx, line in enumerate(lines):
        if line.startswith("| S2b train-CV selected S1d5 dt-anchor + fine_rot=0.45 policy |"):
            lines[idx] = line.replace("current clean fine-rot policy candidate", "previous clean fine-rot candidate")
    VALID_BASELINES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, train_ds_full, scene_seqs, seq_to_indices = _collect_train_groups(device)
    test_ds = _build_eval_dataset(cfg, split="test")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = _candidates()

    fold_rows: Dict[str, List[EvalRow]] = {c.name: [] for c in candidates}
    for fold_idx, seq in enumerate(scene_seqs):
        fold_name = f"fold_{fold_idx}_{seq[0]}_{seq[1]}"
        val_ds = ManifestSubsetDataset(train_ds_full, seq_to_indices[seq])
        for cand in candidates:
            row = _run_eval(
                device,
                val_ds,
                cand,
                split="train_cv",
                fold=fold_name,
                out_dir=OUT_ROOT / fold_name / cand.name,
            )
            fold_rows[cand.name].append(row)

    mean_rank_rows: List[Tuple[Candidate, float, float, float]] = []
    for cand in candidates:
        rows = fold_rows[cand.name]
        mean_rank_rows.append((cand, _mean(rows, "drift"), _mean(rows, "ate"), _mean(rows, "path_ratio")))
    selected, _selected_rows = _select_candidate([(c, fold_rows[c.name]) for c in candidates])

    final_row = _run_eval(device, test_ds, selected, split="test", fold="final_test", out_dir=OUT_ROOT / "final_test" / selected.name)
    verdict = _verdict(final_row)
    _write_report(scene_seqs, candidates, fold_rows, mean_rank_rows, selected, final_row, verdict, REPORT_PATH)

    if verdict == "SUCCESS":
        _export_policy(selected)
        _update_valid_baselines(selected, final_row)

    print(json.dumps({
        "selected_policy": selected.name,
        "policy_type": selected.policy_type,
        "mapping": selected.mapping,
        "final_test": {
            "drift": final_row.drift,
            "ATE": final_row.ate,
            "path_ratio": final_row.path_ratio,
        },
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
