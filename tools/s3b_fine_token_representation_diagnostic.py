#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from model import (
    PanoramaRelPoseModel,
    _apply_dt_bucket_scale_anchor,
    _blend_direction,
    _blend_magnitude,
    _blend_rotation,
    _local_t_to_output_frame,
    _resolve_fine_fuse_strength,
    _set_transform_outputs,
)
from train_mvp import (
    _apply_dt_bucket_scale_anchor_policy,
    _epipolar_matching_diagnostics,
    _restore_cfg_from_policy_base_checkpoint,
    eval_odometry_sequence,
)


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
BASE_CKPT = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S3b_fine_token_representation_diagnostic_report.md"
FIG_DIR = REPO_ROOT / "checkpoints" / "S3b_fine_token_representation_diagnostic_figures"
MAX_TRAIN_PAIR_SAMPLES = 512
MAX_TEST_PAIR_SAMPLES = 512
PROGRESS_EVERY = 32


@dataclass
class ProbeSummary:
    feature_set: str
    train_rot_r2: float
    test_rot_r2: float
    train_rot_mae: float
    test_rot_mae: float
    train_tdir_r2: float
    test_tdir_r2: float
    train_tdir_mae: float
    test_tdir_mae: float
    train_rot_bucket_acc: float
    test_rot_bucket_acc: float
    train_tdir_bucket_acc: float
    test_tdir_bucket_acc: float
    train_high_err_acc: float
    test_high_err_acc: float


class ManifestSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: RflyPanoPanoramaPairsEvalFixedKList, indices: Sequence[int]) -> None:
        self.base_ds = base_ds
        self.indices = [int(i) for i in indices]
        self._manifest = [base_ds.manifest()[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample = self.base_ds[self.indices[idx]]
        sample = dict(sample)
        meta = dict(sample.get("meta", {}))
        meta["orig_ds_idx"] = int(self.indices[idx])
        sample["meta"] = meta
        return sample

    def manifest(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for orig_idx, meta in zip(self.indices, self._manifest):
            m = dict(meta)
            m["orig_ds_idx"] = int(orig_idx)
            out.append(m)
        return out


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _build_cfg() -> Tuple[Config, Dict[str, Any], Dict[str, Any]]:
    cfg = Config()
    cfg.dt_bucket_scale_anchor_policy_json = str(POLICY_PATH)
    cfg, restore_summary = _restore_cfg_from_policy_base_checkpoint(cfg, [])
    policy_summary = _apply_dt_bucket_scale_anchor_policy(cfg)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = 0.45
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.batch_size = 1
    cfg.pin_memory = False
    cfg.num_workers = 0
    return cfg, restore_summary, policy_summary


def _load_model(cfg: Config, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(BASE_CKPT), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_dataset(cfg: Config, split: str, *, k_list_override: Sequence[int] | None = None) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=split,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in (k_list_override if k_list_override is not None else cfg.eval_k_list)),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
    )


def _rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(R_pred.float().transpose(-1, -2), R_gt.float())
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _pool_token_features(feat: torch.Tensor, weight: torch.Tensor | None = None) -> np.ndarray:
    feat = feat.detach().float()
    if feat.dim() != 3 or feat.shape[0] != 1:
        raise ValueError(f"Unexpected token feature shape: {tuple(feat.shape)}")
    feat = feat[0]
    if weight is None:
        mean = feat.mean(dim=0)
        diff = feat - mean.unsqueeze(0)
        std = torch.sqrt(diff.pow(2).mean(dim=0).clamp_min(1.0e-8))
        maxv = feat.max(dim=0).values
    else:
        w = weight.detach().float().view(-1)
        w = w / w.sum().clamp_min(1.0e-8)
        mean = (feat * w.unsqueeze(-1)).sum(dim=0)
        diff = feat - mean.unsqueeze(0)
        std = torch.sqrt((diff.pow(2) * w.unsqueeze(-1)).sum(dim=0).clamp_min(1.0e-8))
        maxv = feat.max(dim=0).values
    return torch.cat([mean, maxv, std], dim=0).cpu().numpy()


def _scalarize(diag: Dict[str, torch.Tensor], key: str) -> float:
    if key not in diag or diag[key] is None:
        return float("nan")
    v = diag[key]
    if torch.is_tensor(v):
        return float(v.detach().float().view(-1)[0].cpu())
    return _safe_float(v)


def _manual_forward(
    model: PanoramaRelPoseModel,
    sample: Dict[str, Any],
    cfg: Config,
    device: torch.device,
) -> Dict[str, Any]:
    IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
    IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
    R_gt = sample["R_gt"].unsqueeze(0).to(device, non_blocking=True)
    t_gt_vec = sample["t_gt_vec"].unsqueeze(0).to(device, non_blocking=True)
    dt_world_val = float(sample["meta"].get("dt_world", float(sample["t_gt_mag"])))
    dt_world = torch.tensor([dt_world_val], device=device, dtype=torch.float32)

    with torch.no_grad():
        tokens = model.module2(IA, IB)
        TokA_c = tokens["TokA_c"]
        TokB_c = tokens["TokB_c"]
        TokA_f = tokens["TokA_f"]
        TokB_f = tokens["TokB_f"]
        out_c = model.coarse(
            TokA_c,
            TokB_c,
            tokens_a_t=tokens.get("TokA_c_t"),
            tokens_b_t=tokens.get("TokB_c_t"),
            dt_world=dt_world,
            tmag_dt_clamp_min=float(getattr(cfg, "tmag_dt_clamp_min", 0.01)),
        )
        aux: Dict[str, Any] = dict(out_c)
        aux["tc_dir_local"] = out_c["tc_dir"]
        aux["tc_dir_out"] = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        aux["tc_dir"] = out_c["tc_dir"]
        aux["tc_mag"] = out_c["tc_mag"]
        aux["log_tc_mag"] = out_c["log_tc_mag"]
        aux["tc_vec_out"] = aux["tc_dir_out"] * aux["tc_mag"].unsqueeze(-1)
        aux["tc_vec_local"] = aux["tc_dir_local"] * aux["tc_mag"].unsqueeze(-1)
        t_mag_coarse, log_t_mag_coarse = model._bias_magnitude(out_c["tc_mag"], out_c["log_tc_mag"])
        aux["t_mag_unbiased"] = out_c["tc_mag"]
        aux["log_t_mag_unbiased"] = out_c["log_tc_mag"]
        _set_transform_outputs(aux, out_c["Rc"], out_c["tc_dir"], t_mag_coarse, log_t_mag_coarse)

        out_f = model.fine(
            TokA_f=TokA_f,
            TokB_f=TokB_f,
            Wc_ab=out_c["Wc_ab"].detach(),
            Rc=out_c["Rc"].detach(),
            tc_dir=aux["tc_dir_out"].detach(),
            depth_tok_a=None,
            topk_coarse=cfg.topk_coarse,
            use_epipolar_bias=cfg.use_epipolar_bias,
            epi_angle_thresh_deg=cfg.epi_angle_thresh_deg,
            epi_bias_strength=cfg.epi_bias_strength,
            epi_mode=cfg.epi_mode,
        )
        aux.update(out_f)
        fine_rot_strength = _resolve_fine_fuse_strength(cfg, "fine_rot_fuse_strength")
        fine_tdir_strength = _resolve_fine_fuse_strength(cfg, "fine_tdir_fuse_strength")
        fine_tmag_strength = _resolve_fine_fuse_strength(cfg, "fine_tmag_fuse_strength")
        R_final = _blend_rotation(out_c["Rc"], out_f["R"], fine_rot_strength)
        t_local_final = _blend_direction(aux["tc_dir_local"], out_f["t_dir"], fine_tdir_strength)
        t_mag_final = _blend_magnitude(aux["tc_mag"], out_f["t_mag"], fine_tmag_strength)
        log_t_mag_final = torch.log(t_mag_final.clamp_min(float(getattr(cfg, "tmag_min", 1.0e-3))))
        t_mag_final_unbiased = t_mag_final
        t_mag_final, log_t_mag_final = model._bias_magnitude(t_mag_final, log_t_mag_final)
        aux["Rf_raw"] = out_f["R"]
        aux["R_final"] = R_final
        aux["t_dir_fine_raw"] = out_f["t_dir"]
        aux["t_mag_fine_raw"] = out_f["t_mag"]
        aux["log_t_mag_fine_raw"] = out_f["log_t_mag"]
        aux["t_vec_fine_raw"] = _local_t_to_output_frame(out_f["R"], out_f["t_dir"]) * out_f["t_mag"].float().view(-1, 1)
        aux["t_mag_unbiased"] = t_mag_final_unbiased
        aux["log_t_mag_unbiased"] = torch.log(t_mag_final_unbiased.clamp_min(float(getattr(cfg, "tmag_min", 1.0e-3))))
        _set_transform_outputs(aux, R_final, t_local_final, t_mag_final, log_t_mag_final)
        _apply_dt_bucket_scale_anchor(cfg, aux, dt_world=dt_world)

        tg = F.normalize(t_gt_vec.float(), dim=-1, eps=1.0e-6)
        tg_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1.0e-6)
        tdir_out = aux["t_dir_out"].float()
        tdir_local = aux["t_dir_local"].float()
        tdir_abs = torch.rad2deg(torch.acos(torch.sum(tdir_out * tg, dim=-1).clamp(-1.0, 1.0)))
        tdir_abs = torch.minimum(tdir_abs, 180.0 - tdir_abs)
        tdir_local_a_abs = torch.rad2deg(torch.acos(torch.sum(tdir_local * tg_Rt, dim=-1).clamp(-1.0, 1.0)))
        tdir_local_a_abs = torch.minimum(tdir_local_a_abs, 180.0 - tdir_local_a_abs)
        rot_err = _rotation_error_deg(R_final, R_gt)

        coarse_diag = _epipolar_matching_diagnostics(
            out_c["Wc_ab"],
            out_c["Wc_ba"],
            TokA_c.bearing,
            TokB_c.bearing,
            R_gt,
            t_gt_vec,
            angle_thresh_deg=cfg.epi_angle_thresh_deg,
            allowed_mask=None,
            routing_mask=None,
            topk=5,
        )
        fine_diag = _epipolar_matching_diagnostics(
            out_f["Wf_ab"],
            out_f["Wf_ba"],
            TokA_f.bearing,
            TokB_f.bearing,
            R_gt,
            t_gt_vec,
            angle_thresh_deg=cfg.epi_angle_thresh_deg,
            allowed_mask=out_f.get("allowed_mask"),
            routing_mask=out_f.get("routing_mask"),
            topk=5,
        )

        coarse_weight = out_c["Wc_ab"].max(dim=-1).values
        fine_weight = out_f["token_weight_f"]
        coarse_pool = _pool_token_features(out_c["Fc"], coarse_weight)
        fine_pool = _pool_token_features(out_f["Ff_t"], fine_weight)

        t_vec_err = torch.linalg.norm(aux["t_vec_out"].float() - t_gt_vec.float(), dim=-1)
        t_mag_gt = float(sample["t_gt_mag"])
        t_mag_pred = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        t_mag_rel_err = abs(t_mag_pred - t_mag_gt) / max(t_mag_gt, 1.0e-8)
        coarse_rot_err = float(_rotation_error_deg(out_c["Rc"], R_gt).detach().cpu().view(-1)[0])
        coarse_tdir_local = out_c["tc_dir"].float()
        coarse_tdir_abs = torch.rad2deg(torch.acos(torch.sum(coarse_tdir_local * tg_Rt, dim=-1).clamp(-1.0, 1.0)))
        coarse_tdir_abs = torch.minimum(coarse_tdir_abs, 180.0 - coarse_tdir_abs)

    return {
        "IA": IA,
        "IB": IB,
        "R_gt": R_gt,
        "t_gt_vec": t_gt_vec,
        "dt_world": dt_world,
        "aux": aux,
        "coarse_diag": coarse_diag,
        "fine_diag": fine_diag,
        "coarse_pool": coarse_pool,
        "fine_pool": fine_pool,
        "rot_err": float(rot_err.detach().cpu().view(-1)[0]),
        "tdir_abs": float(tdir_abs.detach().cpu().view(-1)[0]),
        "tdir_local_a_abs": float(tdir_local_a_abs.detach().cpu().view(-1)[0]),
        "coarse_rot_err": coarse_rot_err,
        "coarse_tdir_local_a_abs": float(coarse_tdir_abs.detach().cpu().view(-1)[0]),
        "trans_vec_l2": float(t_vec_err.detach().cpu().view(-1)[0]),
        "t_mag_gt": t_mag_gt,
        "t_mag_pred": t_mag_pred,
        "t_mag_rel_err": float(t_mag_rel_err),
    }


def _extract_split_rows(
    model: PanoramaRelPoseModel,
    cfg: Config,
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    split_name: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest = ds.manifest()
    for ds_idx, meta in enumerate(manifest):
        if (ds_idx + 1) % PROGRESS_EVERY == 0 or ds_idx == 0 or (ds_idx + 1) == len(manifest):
            print(f"[S3b] {split_name} feature extraction {ds_idx + 1}/{len(manifest)}", flush=True)
        sample = ds[ds_idx]
        pack = _manual_forward(model, sample, cfg, device)
        aux = pack["aux"]
        coarse_diag = pack["coarse_diag"]
        fine_diag = pack["fine_diag"]
        row: Dict[str, Any] = {
            "split": split_name,
            "ds_idx": int(meta.get("orig_ds_idx", ds_idx)),
            "scene": str(meta.get("scene")),
            "seq": str(meta.get("seq")),
            "scene_seq": str((meta.get("scene"), meta.get("seq"))),
            "i": int(meta.get("i", -1)),
            "j": int(meta.get("j", -1)),
            "k": int(meta.get("k", -1)),
            "dt_world": float(meta.get("dt_world", pack["t_mag_gt"])),
            "tmag_gt": float(pack["t_mag_gt"]),
            "tmag_pred": float(pack["t_mag_pred"]),
            "tmag_rel_err": float(pack["t_mag_rel_err"]),
            "rot_error": float(pack["rot_err"]),
            "tdir_abs": float(pack["tdir_abs"]),
            "tdir_local_A_abs": float(pack["tdir_local_a_abs"]),
            "RPE_rot": float(pack["rot_err"]),
            "RPE_trans_dir": float(pack["tdir_abs"]),
            "pair_trans_vec_l2": float(pack["trans_vec_l2"]),
            "delta_rot_to_gt_deg": float(pack["rot_err"]),
            "delta_tdir_to_gt_deg": float(pack["tdir_local_a_abs"]),
            "coarse_rot_error": float(pack["coarse_rot_err"]),
            "coarse_tdir_local_A_abs": float(pack["coarse_tdir_local_a_abs"]),
            "coarse_epi_mass": _scalarize(coarse_diag, "epi_mass_in_gt_band"),
            "coarse_top1": _scalarize(coarse_diag, "top1_in_gt_band"),
            "coarse_top5": _scalarize(coarse_diag, "top5_in_gt_band"),
            "coarse_entropy": _scalarize(coarse_diag, "matching_entropy"),
            "coarse_confidence": _scalarize(coarse_diag, "max_matching_prob"),
            "coarse_cycle_error": _scalarize(coarse_diag, "cycle_error"),
            "fine_epi_mass": _scalarize(fine_diag, "epi_mass_in_gt_band"),
            "fine_top1": _scalarize(fine_diag, "top1_in_gt_band"),
            "fine_top5": _scalarize(fine_diag, "top5_in_gt_band"),
            "fine_entropy": _scalarize(fine_diag, "matching_entropy"),
            "fine_confidence": _scalarize(fine_diag, "max_matching_prob"),
            "fine_cycle_error": _scalarize(fine_diag, "cycle_error"),
            "routing_recall": _scalarize(fine_diag, "routing_recall_in_gt_band"),
            "allowed_mask_density": _scalarize(fine_diag, "allowed_mask_density"),
            "no_candidate_row_ratio": _scalarize(fine_diag, "no_candidate_row_ratio"),
            "dt_bucket_scale_anchor_factor": _safe_float(aux["dt_bucket_scale_anchor_factor"].detach().cpu().item()),
            "t_dir_pred_x": float(aux["t_dir_out"].detach().float()[0, 0].cpu()),
            "t_dir_pred_y": float(aux["t_dir_out"].detach().float()[0, 1].cpu()),
            "t_dir_pred_z": float(aux["t_dir_out"].detach().float()[0, 2].cpu()),
            "t_dir_local_pred_x": float(aux["t_dir_local"].detach().float()[0, 0].cpu()),
            "t_dir_local_pred_y": float(aux["t_dir_local"].detach().float()[0, 1].cpu()),
            "t_dir_local_pred_z": float(aux["t_dir_local"].detach().float()[0, 2].cpu()),
            "coarse_t_dir_local_x": float(aux["tc_dir_local"].detach().float()[0, 0].cpu()),
            "coarse_t_dir_local_y": float(aux["tc_dir_local"].detach().float()[0, 1].cpu()),
            "coarse_t_dir_local_z": float(aux["tc_dir_local"].detach().float()[0, 2].cpu()),
        }
        for r_idx in range(3):
            for c_idx in range(3):
                row[f"R_pred_{r_idx}{c_idx}"] = float(aux["R_final"].detach().float()[0, r_idx, c_idx].cpu())
                row[f"R_coarse_{r_idx}{c_idx}"] = float(aux["Rc"].detach().float()[0, r_idx, c_idx].cpu())
        row["coarse_pool"] = pack["coarse_pool"]
        row["fine_pool"] = pack["fine_pool"]
        rows.append(row)
    return rows


def _select_stratified_indices(
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    limit: int,
    *,
    seed: int,
) -> List[int]:
    manifest = ds.manifest()
    if limit <= 0 or len(manifest) <= limit:
        return list(range(len(manifest)))
    groups: Dict[Tuple[str, str, int], List[int]] = {}
    for idx, meta in enumerate(manifest):
        key = (str(meta.get("scene")), str(meta.get("seq")), int(meta.get("k", -1)))
        groups.setdefault(key, []).append(idx)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    group_items = sorted(groups.items(), key=lambda kv: kv[0])
    total = len(manifest)
    for _key, idxs in group_items:
        share = max(1, int(round(limit * len(idxs) / total)))
        take = min(len(idxs), share)
        chosen = rng.choice(np.asarray(idxs, dtype=np.int64), size=take, replace=False)
        selected.extend(int(x) for x in chosen.tolist())
    selected = sorted(set(selected))
    if len(selected) > limit:
        selected = sorted(rng.choice(np.asarray(selected, dtype=np.int64), size=limit, replace=False).tolist())
    elif len(selected) < limit:
        remain = [idx for idx in range(len(manifest)) if idx not in set(selected)]
        extra = rng.choice(np.asarray(remain, dtype=np.int64), size=min(limit - len(selected), len(remain)), replace=False)
        selected.extend(int(x) for x in extra.tolist())
        selected = sorted(set(selected))
    return selected[:limit]


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(train_x, axis=0)
    sigma = np.nanstd(train_x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1.0e-8), sigma, 1.0)
    train = (np.where(np.isfinite(train_x), train_x, mu) - mu) / sigma
    test = (np.where(np.isfinite(test_x), test_x, mu) - mu) / sigma
    return train, test, {"mean": mu, "std": sigma}


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(xb.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return xb @ w


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1.0e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _quantile_bins(y: np.ndarray, num_bins: int = 3) -> np.ndarray:
    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(y, q)
    edges[0] -= 1.0e-6
    edges[-1] += 1.0e-6
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1.0e-6
    return edges


def _digitize(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(y, edges[1:-1], right=False).astype(np.int64)


def _linear_classifier_fit(x: np.ndarray, labels: np.ndarray, num_classes: int, lam: float = 1.0) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return _ridge_fit(x, y, lam=lam)


def _linear_classifier_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    logits = _ridge_predict(x, w)
    return np.argmax(logits, axis=1).astype(np.int64)


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _rows_to_matrix(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[k]) for k in keys] for row in rows], dtype=np.float64)


def _compose_feature_sets(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    scalar_keys = [
        "coarse_epi_mass",
        "coarse_top1",
        "coarse_top5",
        "coarse_entropy",
        "coarse_confidence",
        "coarse_cycle_error",
        "fine_epi_mass",
        "fine_top1",
        "fine_top5",
        "fine_entropy",
        "fine_confidence",
        "fine_cycle_error",
        "routing_recall",
        "allowed_mask_density",
        "no_candidate_row_ratio",
        "dt_world",
        "k",
        "tmag_gt",
        "tmag_pred",
        "tmag_rel_err",
        "dt_bucket_scale_anchor_factor",
        "coarse_rot_error",
        "coarse_tdir_local_A_abs",
        "t_dir_pred_x",
        "t_dir_pred_y",
        "t_dir_pred_z",
        "t_dir_local_pred_x",
        "t_dir_local_pred_y",
        "t_dir_local_pred_z",
        "coarse_t_dir_local_x",
        "coarse_t_dir_local_y",
        "coarse_t_dir_local_z",
    ]
    coarse_pose_keys = [f"R_coarse_{r}{c}" for r in range(3) for c in range(3)]
    fine_pose_keys = [f"R_pred_{r}{c}" for r in range(3) for c in range(3)]
    coarse_diag_keys = scalar_keys[:6] + ["coarse_rot_error", "coarse_tdir_local_A_abs"] + coarse_pose_keys + [
        "coarse_t_dir_local_x",
        "coarse_t_dir_local_y",
        "coarse_t_dir_local_z",
    ]
    fine_diag_keys = scalar_keys[6:15] + fine_pose_keys + [
        "t_dir_pred_x",
        "t_dir_pred_y",
        "t_dir_pred_z",
        "t_dir_local_pred_x",
        "t_dir_local_pred_y",
        "t_dir_local_pred_z",
    ]
    conf_keys = scalar_keys[:15]
    dtk_keys = ["dt_world", "k", "tmag_gt", "tmag_pred", "tmag_rel_err", "dt_bucket_scale_anchor_factor"]

    feature_names = {
        "coarse_only": ["coarse_pool"] + coarse_diag_keys,
        "fine_only": ["fine_pool"] + fine_diag_keys,
        "coarse_fine": ["coarse_pool", "fine_pool"] + coarse_diag_keys + fine_diag_keys,
        "confidence_only": conf_keys,
        "dt_k_tmag_only": dtk_keys,
        "all_features": ["coarse_pool", "fine_pool"] + scalar_keys + coarse_pose_keys + fine_pose_keys,
    }
    feature_arrays: Dict[str, np.ndarray] = {}
    for name, defs in feature_names.items():
        mats: List[np.ndarray] = []
        for key in defs:
            if key == "coarse_pool":
                mats.append(np.stack([np.asarray(row["coarse_pool"], dtype=np.float64) for row in rows], axis=0))
            elif key == "fine_pool":
                mats.append(np.stack([np.asarray(row["fine_pool"], dtype=np.float64) for row in rows], axis=0))
            else:
                mats.append(_rows_to_matrix(rows, [key]))
        feature_arrays[name] = np.concatenate(mats, axis=1)
    return feature_arrays, feature_names


def _pair_probe(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[ProbeSummary], Dict[str, Any]]:
    train_sets, feature_names = _compose_feature_sets(train_rows)
    test_sets, _ = _compose_feature_sets(test_rows)
    y_rot_train = np.asarray([float(r["rot_error"]) for r in train_rows], dtype=np.float64)
    y_rot_test = np.asarray([float(r["rot_error"]) for r in test_rows], dtype=np.float64)
    y_tdir_train = np.asarray([float(r["tdir_local_A_abs"]) for r in train_rows], dtype=np.float64)
    y_tdir_test = np.asarray([float(r["tdir_local_A_abs"]) for r in test_rows], dtype=np.float64)
    y_l2_train = np.asarray([float(r["pair_trans_vec_l2"]) for r in train_rows], dtype=np.float64)

    comp_train = (
        (y_rot_train - y_rot_train.mean()) / max(y_rot_train.std(), 1.0e-6)
        + (y_tdir_train - y_tdir_train.mean()) / max(y_tdir_train.std(), 1.0e-6)
        + (y_l2_train - y_l2_train.mean()) / max(y_l2_train.std(), 1.0e-6)
    )
    high_err_thr = float(np.quantile(comp_train, 0.75))
    high_rot_thr = float(np.quantile(y_rot_train, 0.75))
    high_tdir_thr = float(np.quantile(y_tdir_train, 0.75))

    comp_test = (
        (y_rot_test - y_rot_train.mean()) / max(y_rot_train.std(), 1.0e-6)
        + (y_tdir_test - y_tdir_train.mean()) / max(y_tdir_train.std(), 1.0e-6)
        + (
            np.asarray([float(r["pair_trans_vec_l2"]) for r in test_rows], dtype=np.float64) - y_l2_train.mean()
        ) / max(y_l2_train.std(), 1.0e-6)
    )
    high_err_train = (comp_train >= high_err_thr).astype(np.int64)
    high_err_test = (comp_test >= high_err_thr).astype(np.int64)

    rot_edges = _quantile_bins(y_rot_train, num_bins=3)
    tdir_edges = _quantile_bins(y_tdir_train, num_bins=3)
    rot_bucket_train = _digitize(y_rot_train, rot_edges)
    rot_bucket_test = _digitize(y_rot_test, rot_edges)
    tdir_bucket_train = _digitize(y_tdir_train, tdir_edges)
    tdir_bucket_test = _digitize(y_tdir_test, tdir_edges)

    results: List[ProbeSummary] = []
    preds_store: Dict[str, Any] = {
        "high_error_threshold": high_err_thr,
        "high_rot_threshold": high_rot_thr,
        "high_tdir_threshold": high_tdir_thr,
    }
    for name in ["coarse_only", "fine_only", "coarse_fine", "confidence_only", "dt_k_tmag_only", "all_features"]:
        x_train_raw = train_sets[name]
        x_test_raw = test_sets[name]
        x_train, x_test, stats = _standardize(x_train_raw, x_test_raw)

        rot_w = _ridge_fit(x_train, y_rot_train, lam=1.0)
        tdir_w = _ridge_fit(x_train, y_tdir_train, lam=1.0)
        rot_hat_train = _ridge_predict(x_train, rot_w).reshape(-1)
        rot_hat_test = _ridge_predict(x_test, rot_w).reshape(-1)
        tdir_hat_train = _ridge_predict(x_train, tdir_w).reshape(-1)
        tdir_hat_test = _ridge_predict(x_test, tdir_w).reshape(-1)

        rot_cls_w = _linear_classifier_fit(x_train, rot_bucket_train, num_classes=3, lam=1.0)
        tdir_cls_w = _linear_classifier_fit(x_train, tdir_bucket_train, num_classes=3, lam=1.0)
        high_err_w = _linear_classifier_fit(x_train, high_err_train, num_classes=2, lam=1.0)

        rot_cls_train = _linear_classifier_predict(x_train, rot_cls_w)
        rot_cls_test = _linear_classifier_predict(x_test, rot_cls_w)
        tdir_cls_train = _linear_classifier_predict(x_train, tdir_cls_w)
        tdir_cls_test = _linear_classifier_predict(x_test, tdir_cls_w)
        high_err_pred_train = _linear_classifier_predict(x_train, high_err_w)
        high_err_pred_test = _linear_classifier_predict(x_test, high_err_w)

        results.append(
            ProbeSummary(
                feature_set=name,
                train_rot_r2=_r2(y_rot_train, rot_hat_train),
                test_rot_r2=_r2(y_rot_test, rot_hat_test),
                train_rot_mae=_mae(y_rot_train, rot_hat_train),
                test_rot_mae=_mae(y_rot_test, rot_hat_test),
                train_tdir_r2=_r2(y_tdir_train, tdir_hat_train),
                test_tdir_r2=_r2(y_tdir_test, tdir_hat_test),
                train_tdir_mae=_mae(y_tdir_train, tdir_hat_train),
                test_tdir_mae=_mae(y_tdir_test, tdir_hat_test),
                train_rot_bucket_acc=_acc(rot_bucket_train, rot_cls_train),
                test_rot_bucket_acc=_acc(rot_bucket_test, rot_cls_test),
                train_tdir_bucket_acc=_acc(tdir_bucket_train, tdir_cls_train),
                test_tdir_bucket_acc=_acc(tdir_bucket_test, tdir_cls_test),
                train_high_err_acc=_acc(high_err_train, high_err_pred_train),
                test_high_err_acc=_acc(high_err_test, high_err_pred_test),
            )
        )
        preds_store[name] = {
            "stats": stats,
            "rot_pred_test": rot_hat_test,
            "tdir_pred_test": tdir_hat_test,
            "high_err_pred_test": high_err_pred_test,
            "x_train": x_train,
            "x_test": x_test,
            "feature_names": feature_names[name],
        }
    preds_store["high_err_test"] = high_err_test
    preds_store["high_rot_test"] = (y_rot_test >= high_rot_thr).astype(np.int64)
    preds_store["high_tdir_test"] = (y_tdir_test >= high_tdir_thr).astype(np.int64)
    preds_store["rot_test"] = y_rot_test
    preds_store["tdir_test"] = y_tdir_test
    return results, preds_store


def _pca_fit(x: np.ndarray, n_components: int = 2) -> Dict[str, np.ndarray]:
    mu = x.mean(axis=0)
    xc = x - mu
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    comp = vh[:n_components]
    return {"mean": mu, "components": comp}


def _pca_transform(x: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - pca["mean"]) @ pca["components"].T


def _cosine_topk_fraction(x: np.ndarray, mask: np.ndarray, k: int = 5) -> float:
    if int(mask.sum()) <= 1:
        return float("nan")
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1.0e-8)
    sims = x @ x.T
    out: List[float] = []
    for idx in np.where(mask == 1)[0]:
        order = np.argsort(-sims[idx])
        order = order[order != idx][:k]
        if order.size == 0:
            continue
        out.append(float(mask[order].mean()))
    return float(np.mean(out)) if out else float("nan")


def _kmeans(x: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(0)
    if x.shape[0] <= k:
        return np.arange(x.shape[0], dtype=np.int64)
    centers = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    labels = np.zeros((x.shape[0],), dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            sel = labels == ci
            if np.any(sel):
                centers[ci] = x[sel].mean(axis=0)
    return labels


def _best_cluster_purity(labels: np.ndarray, mask: np.ndarray) -> float:
    best = 0.0
    for cid in np.unique(labels):
        sel = labels == cid
        if sel.sum() < 5:
            continue
        purity = float(mask[sel].mean())
        if purity > best:
            best = purity
    return best


def _run_odometry_debug(
    model: PanoramaRelPoseModel,
    ds: RflyPanoPanoramaPairsEvalFixedKList,
    cfg: Config,
    split_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"s3b_{split_name}_", dir=str(REPO_ROOT / "checkpoints")))
    old_flag = bool(getattr(cfg, "save_odom_trajectory_debug", False))
    cfg.save_odom_trajectory_debug = True
    try:
        payload = eval_odometry_sequence(model, ds, device, cfg, output_dir=str(tmp_dir), step=0, upd=0)
        debug_json = _read_json(tmp_dir / "odom_trajectory_debug_latest.json")
        step_rows: List[Dict[str, Any]] = []
        csv_path = tmp_dir / "odom_trajectory_steps_latest.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_rows.append({k: row[k] for k in reader.fieldnames or []})
    finally:
        cfg.save_odom_trajectory_debug = old_flag
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"payload": payload, "debug_json": debug_json, "step_rows": step_rows}


def _merge_chain_info(rows: List[Dict[str, Any]], odom_debug: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    steps = odom_debug["step_rows"]
    by_key = {
        (str(step.get("scene_seq")), int(step.get("i", -1)), int(step.get("j", -1)), int(step.get("k", -1))): step
        for step in steps
    }
    for row in rows:
        step = by_key.get((str(row["scene_seq"]), int(row["i"]), int(row["j"]), int(row["k"])))
        if step is None:
            row["chain_id"] = -1
            row["metric_pos_err"] = float("nan")
            row["metric_step_dir_err_deg"] = float("nan")
            row["chain_high_ate"] = 0
            continue
        row["chain_id"] = int(step["chain_id"])
        row["metric_pos_err"] = _safe_float(step.get("metric_pos_err"))
        row["metric_step_dir_err_deg"] = _safe_float(step.get("metric_step_dir_err_deg"))

    chains: List[Dict[str, Any]] = []
    for chain_id, chain in enumerate(odom_debug["debug_json"].get("chains", [])):
        rec = {
            "chain_id": int(chain_id),
            "scene_seq": str(chain.get("scene_seq")),
            "num_steps": int(chain.get("num_steps", 0)),
            "chain_ate": _safe_float(chain.get("mean_metric_pos_err")),
            "chain_drift": _safe_float(chain.get("final_metric_pos_err")),
            "chain_path_ratio": _safe_float(chain.get("shape_metric", {}).get("path_length_ratio")),
            "chain_mean_tdir_err_deg": _safe_float(chain.get("mean_tdir_err_deg")),
            "chain_mean_tmag_rel_err": _safe_float(chain.get("mean_tmag_rel_err")),
            "chain_mean_step_dir_err_deg": _safe_float(chain.get("shape_metric", {}).get("mean_step_dir_err_deg")),
            "chain_turn_abs_err_deg": _safe_float(chain.get("shape_metric", {}).get("mean_turn_abs_err_deg")),
            "chain_straightness_abs_err": _safe_float(chain.get("shape_metric", {}).get("straightness_abs_err")),
        }
        chains.append(rec)
    return rows, chains


def _aggregate_chain_features(
    rows: Sequence[Dict[str, Any]],
    chains: Sequence[Dict[str, Any]],
    pca: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    row_by_chain: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        row_by_chain.setdefault(int(row["chain_id"]), []).append(row)
    for chain in chains:
        cid = int(chain["chain_id"])
        items = row_by_chain.get(cid, [])
        if not items:
            continue
        fine_mat = np.stack([np.asarray(r["fine_pool"], dtype=np.float64) for r in items], axis=0)
        fine_pca = _pca_transform(fine_mat, pca)
        conf = np.asarray([float(r["fine_confidence"]) for r in items], dtype=np.float64)
        conf = np.where(np.isfinite(conf), conf, 0.0)
        conf = conf / max(conf.sum(), 1.0e-8)
        entropy = np.asarray([float(r["fine_entropy"]) for r in items], dtype=np.float64)
        rot_err = np.asarray([float(r["rot_error"]) for r in items], dtype=np.float64)
        tdir_err = np.asarray([float(r["tdir_local_A_abs"]) for r in items], dtype=np.float64)
        pos_err = np.asarray([_safe_float(r["metric_pos_err"]) for r in items], dtype=np.float64)
        rec = dict(chain)
        rec["fine_pca_mean_0"] = float(fine_pca[:, 0].mean())
        rec["fine_pca_mean_1"] = float(fine_pca[:, 1].mean()) if fine_pca.shape[1] > 1 else 0.0
        rec["fine_pca_max_0"] = float(fine_pca[:, 0].max())
        rec["fine_pca_max_1"] = float(fine_pca[:, 1].max()) if fine_pca.shape[1] > 1 else 0.0
        rec["fine_pca_wmean_0"] = float((fine_pca[:, 0] * conf).sum())
        rec["fine_pca_wmean_1"] = float((fine_pca[:, 1] * conf).sum()) if fine_pca.shape[1] > 1 else 0.0
        rec["fine_conf_mean"] = float(np.mean([float(r["fine_confidence"]) for r in items]))
        rec["fine_conf_max"] = float(np.max([float(r["fine_confidence"]) for r in items]))
        rec["fine_entropy_mean"] = float(np.nanmean(entropy))
        rec["fine_entropy_max"] = float(np.nanmax(entropy))
        rec["pair_rot_err_mean"] = float(np.mean(rot_err))
        rec["pair_tdir_err_mean"] = float(np.mean(tdir_err))
        rec["pair_metric_pos_err_mean"] = float(np.nanmean(pos_err))
        out.append(rec)
    return out


def _chain_probe(
    train_chains: Sequence[Dict[str, Any]],
    test_chains: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    feat_keys = [
        "fine_pca_mean_0",
        "fine_pca_mean_1",
        "fine_pca_max_0",
        "fine_pca_max_1",
        "fine_pca_wmean_0",
        "fine_pca_wmean_1",
        "fine_conf_mean",
        "fine_conf_max",
        "fine_entropy_mean",
        "fine_entropy_max",
        "pair_rot_err_mean",
        "pair_tdir_err_mean",
        "pair_metric_pos_err_mean",
        "chain_mean_tdir_err_deg",
        "chain_mean_tmag_rel_err",
        "chain_mean_step_dir_err_deg",
        "chain_turn_abs_err_deg",
        "chain_straightness_abs_err",
    ]
    x_train_raw = _rows_to_matrix(train_chains, feat_keys)
    x_test_raw = _rows_to_matrix(test_chains, feat_keys)
    x_train, x_test, _ = _standardize(x_train_raw, x_test_raw)
    y_ate_train = np.asarray([float(r["chain_ate"]) for r in train_chains], dtype=np.float64)
    y_ate_test = np.asarray([float(r["chain_ate"]) for r in test_chains], dtype=np.float64)
    y_drift_train = np.asarray([float(r["chain_drift"]) for r in train_chains], dtype=np.float64)
    y_drift_test = np.asarray([float(r["chain_drift"]) for r in test_chains], dtype=np.float64)
    y_path_train = np.asarray([float(r["chain_path_ratio"]) for r in train_chains], dtype=np.float64)
    y_path_test = np.asarray([float(r["chain_path_ratio"]) for r in test_chains], dtype=np.float64)
    high_ate_thr = float(np.quantile(y_ate_train, 0.75))
    high_ate_train = (y_ate_train >= high_ate_thr).astype(np.int64)
    high_ate_test = (y_ate_test >= high_ate_thr).astype(np.int64)

    ate_w = _ridge_fit(x_train, y_ate_train, lam=1.0)
    drift_w = _ridge_fit(x_train, y_drift_train, lam=1.0)
    path_w = _ridge_fit(x_train, y_path_train, lam=1.0)
    cls_w = _linear_classifier_fit(x_train, high_ate_train, num_classes=2, lam=1.0)

    ate_pred_test = _ridge_predict(x_test, ate_w).reshape(-1)
    drift_pred_test = _ridge_predict(x_test, drift_w).reshape(-1)
    path_pred_test = _ridge_predict(x_test, path_w).reshape(-1)
    cls_pred_test = _linear_classifier_predict(x_test, cls_w)
    return {
        "high_ate_threshold": high_ate_thr,
        "test_ate_r2": _r2(y_ate_test, ate_pred_test),
        "test_ate_mae": _mae(y_ate_test, ate_pred_test),
        "test_drift_r2": _r2(y_drift_test, drift_pred_test),
        "test_drift_mae": _mae(y_drift_test, drift_pred_test),
        "test_path_r2": _r2(y_path_test, path_pred_test),
        "test_path_mae": _mae(y_path_test, path_pred_test),
        "test_high_ate_acc": _acc(high_ate_test, cls_pred_test),
        "ate_true_test": y_ate_test,
        "ate_pred_test": ate_pred_test,
        "high_ate_test": high_ate_test,
    }


def _classification_from_results(pair_results: Sequence[ProbeSummary], chain_probe: Dict[str, Any]) -> Tuple[str, str]:
    by_name = {r.feature_set: r for r in pair_results}
    coarse = by_name["coarse_only"]
    fine = by_name["fine_only"]
    conf = by_name["confidence_only"]
    dtk = by_name["dt_k_tmag_only"]
    allf = by_name["all_features"]

    fine_mean_r2 = np.nanmean([fine.test_rot_r2, fine.test_tdir_r2])
    coarse_mean_r2 = np.nanmean([coarse.test_rot_r2, coarse.test_tdir_r2])
    all_mean_r2 = np.nanmean([allf.test_rot_r2, allf.test_tdir_r2])
    conf_mean_r2 = np.nanmean([conf.test_rot_r2, conf.test_tdir_r2])
    dtk_mean_r2 = np.nanmean([dtk.test_rot_r2, dtk.test_tdir_r2])
    chain_signal = np.nanmean([chain_probe["test_ate_r2"], chain_probe["test_drift_r2"], chain_probe["test_path_r2"]])

    if fine_mean_r2 >= 0.15 and fine_mean_r2 > coarse_mean_r2 + 0.03 and fine.test_high_err_acc >= conf.test_high_err_acc:
        return "FINE-FEATURE-HAS-STABLE-RESIDUAL-SIGNAL", "fine feature shows the strongest stable test-time residual readability"
    if (
        fine_mean_r2 < 0.05
        and dtk.test_high_err_acc >= 0.65
        and dtk.test_high_err_acc >= allf.test_high_err_acc + 0.15
        and dtk.test_rot_mae < fine.test_rot_mae
    ):
        return "DT/K-DOMINATED-ERROR", "frozen fine features do not generalize well, while dt / k / magnitude features explain most of the readable pair-level error structure"
    if conf_mean_r2 >= all_mean_r2 - 0.02 and conf.test_high_err_acc >= max(fine.test_high_err_acc, coarse.test_high_err_acc):
        return "CONFIDENCE-ONLY-EXPLAINS-ERROR", "confidence / entropy features explain most readable error variation without a richer token signal"
    if dtk_mean_r2 >= all_mean_r2 - 0.02 and dtk.test_high_err_acc >= max(fine.test_high_err_acc, coarse.test_high_err_acc) - 0.01:
        return "DT/K-DOMINATED-ERROR", "dt / k / magnitude features explain nearly all readable variation"
    if all_mean_r2 < 0.05 and chain_signal >= 0.15:
        return "CHAIN-LEVEL-SIGNAL-ONLY", "pair-level readability is weak while chain-level aggregation carries clearer signal"
    if all_mean_r2 >= 0.05:
        return "FINE-FEATURE-WEAK-SIGNAL", "there is some readable signal, but it is not strong or stable enough to justify residual-head training yet"
    return "INCONCLUSIVE", "probe signal is too weak or unstable to support a stronger conclusion"


def _plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.scatter(y_true, y_pred, s=10, alpha=0.7)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pca(points: np.ndarray, high_err: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.scatter(points[high_err == 0, 0], points[high_err == 0, 1], s=10, alpha=0.45, label="normal")
    ax.scatter(points[high_err == 1, 0], points[high_err == 1, 1], s=16, alpha=0.75, label="high-error")
    ax.set_title("Fine Feature PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_chain_score(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.scatter(y_true, y_pred, s=18, alpha=0.8)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Chain ATE Proxy vs Predicted Score")
    ax.set_xlabel("Chain ATE Proxy")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(
    cfg: Config,
    restore_summary: Dict[str, Any],
    policy_summary: Dict[str, Any],
    load_summary: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    pair_results: Sequence[ProbeSummary],
    clustering: Dict[str, Any],
    chain_probe: Dict[str, Any],
    classification: str,
    classification_reason: str,
) -> None:
    by_name = {r.feature_set: r for r in pair_results}
    fine_better = (
        by_name["fine_only"].test_rot_mae < by_name["coarse_only"].test_rot_mae
        and by_name["fine_only"].test_tdir_mae < by_name["coarse_only"].test_tdir_mae
        and by_name["fine_only"].test_high_err_acc >= by_name["coarse_only"].test_high_err_acc
    )
    train_gap_all = np.nanmean(
        [
            by_name["all_features"].train_rot_r2 - by_name["all_features"].test_rot_r2,
            by_name["all_features"].train_tdir_r2 - by_name["all_features"].test_tdir_r2,
        ]
    )
    lines: List[str] = []
    lines.append("# S3b Fine Token Representation Diagnostic Report\n\n")
    lines.append("## 1. Executive summary\n\n")
    lines.append(f"- final classification: `{classification}`\n")
    lines.append(f"- summary: {classification_reason}\n")
    lines.append(f"- fine token stable residual signal: `{classification == 'FINE-FEATURE-HAS-STABLE-RESIDUAL-SIGNAL'}`\n")
    lines.append(
        f"- worth continuing head training now: `{classification == 'FINE-FEATURE-HAS-STABLE-RESIDUAL-SIGNAL'}`\n\n"
    )

    lines.append("## 2. Feature extraction summary\n\n")
    lines.append(f"- policy path: `{POLICY_PATH}`\n")
    lines.append(f"- base checkpoint: `{BASE_CKPT}`\n")
    lines.append(f"- explicit cfg / policy-path requirement satisfied: `{bool(policy_summary.get('enabled', False)) and len(load_summary['unexpected']) == 0}`\n")
    lines.append(f"- missing / unexpected: `{len(load_summary['missing'])}` / `{len(load_summary['unexpected'])}`\n")
    lines.append(f"- fine_rot / fine_tdir / fine_tmag: `{cfg.fine_rot_fuse_strength}` / `{cfg.fine_tdir_fuse_strength}` / `{cfg.fine_tmag_fuse_strength}`\n")
    lines.append(f"- use_geometry_refine: `{cfg.use_geometry_refine}`\n")
    lines.append(f"- train pair samples: `{len(train_rows)}`\n")
    lines.append(f"- test pair samples: `{len(test_rows)}`\n")
    lines.append(f"- pair sampling cap train/test: `{MAX_TRAIN_PAIR_SAMPLES}` / `{MAX_TEST_PAIR_SAMPLES}`\n")
    lines.append("- extracted features include:\n")
    lines.append("  fine pooled feature, coarse pooled feature, entropy/top1/epi-mass/confidence, predicted pose representation, dt/k/tmag, allowed-mask density, routing recall\n\n")

    lines.append("## 3. Probe results table\n\n")
    lines.append("| feature set | test rot R2 | test rot MAE | test tdir R2 | test tdir MAE | test rot-bucket acc | test tdir-bucket acc | test high-error acc |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for name in ["coarse_only", "fine_only", "coarse_fine", "confidence_only", "dt_k_tmag_only", "all_features"]:
        r = by_name[name]
        lines.append(
            f"| {name} | {_fmt(r.test_rot_r2)} | {_fmt(r.test_rot_mae)} | {_fmt(r.test_tdir_r2)} | {_fmt(r.test_tdir_mae)} | {_fmt(r.test_rot_bucket_acc)} | {_fmt(r.test_tdir_bucket_acc)} | {_fmt(r.test_high_err_acc)} |\n"
        )
    lines.append("\n")

    lines.append("## 4. Generalization analysis\n\n")
    lines.append(f"- all-features mean train-test R2 gap: `{_fmt(train_gap_all)}`\n")
    lines.append(f"- fine feature better than coarse feature on test: `{fine_better}`\n")
    lines.append(
        f"- confidence-only nearly explains all-features: `{by_name['confidence_only'].test_high_err_acc >= by_name['all_features'].test_high_err_acc - 0.02}`\n"
    )
    lines.append(
        f"- dt/k/tmag-only nearly explains all-features: `{by_name['dt_k_tmag_only'].test_high_err_acc >= by_name['all_features'].test_high_err_acc - 0.02}`\n"
    )
    lines.append(f"- overfitting concern visible: `{train_gap_all > 0.15}`\n\n")

    lines.append("## 5. High-error clustering / nearest-neighbor summary\n\n")
    lines.append(f"- high-error base rate on test: `{_fmt(clustering['base_rate'])}`\n")
    lines.append(f"- fine-feature high-error NN purity@5: `{_fmt(clustering['high_err_nn_purity'])}`\n")
    lines.append(f"- high-rot-error NN purity@5: `{_fmt(clustering['high_rot_nn_purity'])}`\n")
    lines.append(f"- high-tdir-error NN purity@5: `{_fmt(clustering['high_tdir_nn_purity'])}`\n")
    lines.append(f"- best k-means high-error cluster purity: `{_fmt(clustering['cluster_purity'])}`\n")
    lines.append(
        f"- bad chains share similar feature signatures: `{_fmt(clustering['chain_high_ate_nn_purity'])}` neighbor purity\n\n"
    )

    lines.append("## 6. Chain-level diagnostic\n\n")
    lines.append(f"- test chain ATE-proxy R2: `{_fmt(chain_probe['test_ate_r2'])}`\n")
    lines.append(f"- test chain drift-proxy R2: `{_fmt(chain_probe['test_drift_r2'])}`\n")
    lines.append(f"- test chain path-ratio R2: `{_fmt(chain_probe['test_path_r2'])}`\n")
    lines.append(f"- test high-ATE chain accuracy: `{_fmt(chain_probe['test_high_ate_acc'])}`\n")
    lines.append(f"- train/test chain counts: `{len(train_chains)}` / `{len(test_chains)}`\n")
    lines.append(
        "- interpretation: chain-level probe uses mean/final metric position error as chain ATE/drift proxies from the odometry debug chain summaries.\n"
    )
    lines.append("- limitation: the current train/test split yields only one debug chain per split at `k=1`, so chain-level generalization remains weakly identified.\n\n")

    lines.append("## 7. Final classification\n\n")
    lines.append(f"- `{classification}`\n\n")

    lines.append("## 8. Next-step recommendation\n\n")
    if classification == "FINE-FEATURE-HAS-STABLE-RESIDUAL-SIGNAL":
        lines.append("- recommend `S3c_lightweight_probe_initialized_residual_head`\n")
    elif classification == "FINE-FEATURE-WEAK-SIGNAL":
        lines.append("- do not continue residual head; move to `S3d fine token representation redesign`\n")
    elif classification == "CONFIDENCE-ONLY-EXPLAINS-ERROR":
        lines.append("- prefer confidence-gated policy work over a new residual head\n")
    elif classification == "DT/K-DOMINATED-ERROR":
        lines.append("- return to dt/k-aware evaluation or data-distribution diagnosis\n")
    elif classification == "CHAIN-LEVEL-SIGNAL-ONLY":
        lines.append("- explore a trajectory-level aggregator rather than pair-level residual head\n")
    else:
        lines.append("- keep S3b diagnostic as evidence, but do not start new residual-head training yet\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cfg, restore_summary, policy_summary = _build_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_summary = _load_model(cfg, device)
    if load_summary["unexpected"]:
        raise RuntimeError(f"unexpected checkpoint keys found: {load_summary['unexpected'][:8]}")

    train_ds_full = _build_dataset(cfg, split="train")
    test_ds_full = _build_dataset(cfg, split="test")
    train_idx = _select_stratified_indices(train_ds_full, MAX_TRAIN_PAIR_SAMPLES, seed=0)
    test_idx = _select_stratified_indices(test_ds_full, MAX_TEST_PAIR_SAMPLES, seed=1)
    train_ds = ManifestSubsetDataset(train_ds_full, train_idx)
    test_ds = ManifestSubsetDataset(test_ds_full, test_idx)

    print(f"[S3b] extracting train pair features: {len(train_ds)} sampled from {len(train_ds_full)}", flush=True)
    train_rows = _extract_split_rows(model, cfg, train_ds, "train", device)
    print(f"[S3b] extracting test pair features: {len(test_ds)} sampled from {len(test_ds_full)}", flush=True)
    test_rows = _extract_split_rows(model, cfg, test_ds, "test", device)

    print("[S3b] running pair probes", flush=True)
    pair_results, probe_store = _pair_probe(train_rows, test_rows)

    print("[S3b] running chain diagnostics", flush=True)
    train_ds_chain = _build_dataset(cfg, split="train", k_list_override=(1,))
    test_ds_chain = _build_dataset(cfg, split="test", k_list_override=(1,))
    odom_train = _run_odometry_debug(model, train_ds_chain, cfg, "train", device)
    odom_test = _run_odometry_debug(model, test_ds_chain, cfg, "test", device)
    train_chain_rows = _extract_split_rows(model, cfg, train_ds_chain, "train_chain", device)
    test_chain_rows = _extract_split_rows(model, cfg, test_ds_chain, "test_chain", device)
    train_rows, _ = _merge_chain_info(train_rows, odom_train)
    test_rows, _ = _merge_chain_info(test_rows, odom_test)
    train_chain_rows, train_chains = _merge_chain_info(train_chain_rows, odom_train)
    test_chain_rows, test_chains = _merge_chain_info(test_chain_rows, odom_test)

    fine_train = np.stack([np.asarray(r["fine_pool"], dtype=np.float64) for r in train_rows], axis=0)
    fine_test = np.stack([np.asarray(r["fine_pool"], dtype=np.float64) for r in test_rows], axis=0)
    fine_train_std, fine_test_std, _ = _standardize(fine_train, fine_test)
    pca = _pca_fit(fine_train_std, n_components=2)
    pca_test = _pca_transform(fine_test_std, pca)

    train_chain_features = _aggregate_chain_features(train_chain_rows, train_chains, pca)
    test_chain_features = _aggregate_chain_features(test_chain_rows, test_chains, pca)
    chain_probe = _chain_probe(train_chain_features, test_chain_features)

    high_err_test = np.asarray(probe_store["high_err_test"], dtype=np.int64)
    high_rot_test = np.asarray(probe_store["high_rot_test"], dtype=np.int64)
    high_tdir_test = np.asarray(probe_store["high_tdir_test"], dtype=np.int64)
    cluster_labels = _kmeans(pca_test, k=min(8, max(2, pca_test.shape[0] // 25)))
    chain_high_ate_test = (np.asarray([float(r["chain_ate"]) for r in test_chain_features], dtype=np.float64) >= chain_probe["high_ate_threshold"]).astype(np.int64)
    chain_x = _rows_to_matrix(
        test_chain_features,
        [
            "fine_pca_mean_0",
            "fine_pca_mean_1",
            "fine_conf_mean",
            "fine_entropy_mean",
            "pair_rot_err_mean",
            "pair_tdir_err_mean",
        ],
    )
    clustering = {
        "base_rate": float(high_err_test.mean()),
        "high_err_nn_purity": _cosine_topk_fraction(fine_test_std, high_err_test, k=5),
        "high_rot_nn_purity": _cosine_topk_fraction(fine_test_std, high_rot_test, k=5),
        "high_tdir_nn_purity": _cosine_topk_fraction(fine_test_std, high_tdir_test, k=5),
        "cluster_purity": _best_cluster_purity(cluster_labels, high_err_test),
        "chain_high_ate_nn_purity": _cosine_topk_fraction(chain_x, chain_high_ate_test, k=3) if len(test_chain_features) >= 3 else float("nan"),
    }

    classification, reason = _classification_from_results(pair_results, chain_probe)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_pred_vs_true(
        np.asarray(probe_store["rot_test"], dtype=np.float64),
        np.asarray(probe_store["all_features"]["rot_pred_test"], dtype=np.float64),
        FIG_DIR / "probe_pred_vs_true_rot_error.png",
        "All-Features Probe: Rotation Error",
        "True rotation error (deg)",
    )
    _plot_pred_vs_true(
        np.asarray(probe_store["tdir_test"], dtype=np.float64),
        np.asarray(probe_store["all_features"]["tdir_pred_test"], dtype=np.float64),
        FIG_DIR / "probe_pred_vs_true_tdir_error.png",
        "All-Features Probe: Tdir Error",
        "True tdir_local_A_abs (deg)",
    )
    _plot_pca(pca_test, high_err_test, FIG_DIR / "feature_pca_high_error.png")
    _plot_chain_score(
        np.asarray(chain_probe["ate_true_test"], dtype=np.float64),
        np.asarray(chain_probe["ate_pred_test"], dtype=np.float64),
        FIG_DIR / "chain_error_vs_feature_score.png",
    )

    _write_report(
        cfg,
        restore_summary,
        policy_summary,
        load_summary,
        train_rows,
        test_rows,
        pair_results,
        clustering,
        chain_probe,
        classification,
        reason,
    )
    _write_json(
        REPORT_PATH.with_suffix(".json"),
        {
            "classification": classification,
            "reason": reason,
            "pair_results": [r.__dict__ for r in pair_results],
            "clustering": clustering,
            "chain_probe": {k: v for k, v in chain_probe.items() if not isinstance(v, np.ndarray)},
            "load_summary": {"missing": len(load_summary["missing"]), "unexpected": len(load_summary["unexpected"])},
            "counts": {
                "train_pairs": len(train_rows),
                "test_pairs": len(test_rows),
                "train_pairs_full": len(train_ds_full),
                "test_pairs_full": len(test_ds_full),
                "train_chain_pairs_full_k1": len(train_ds_chain),
                "test_chain_pairs_full_k1": len(test_ds_chain),
                "train_chains": len(train_chains),
                "test_chains": len(test_chains),
            },
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "classification": classification,
                "reason": reason,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
