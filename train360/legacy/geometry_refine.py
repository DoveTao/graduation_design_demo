"""
Geometry refinement utilities for learned spherical correspondences.

This module is intentionally eval/inference-only for the current stage.  It
extracts weighted bearing correspondences from soft matching, estimates an
essential matrix with a weighted eight-point solve, and selects the decomposed
pose closest to the network prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(v.float(), dim=-1, eps=eps)


def _empty_corr(device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    return [], [], [], {
        "num_matches": [],
        "mean_prob": [],
        "mutual_ratio": [],
        "allowed_ratio": [],
    }


def extract_correspondences_from_softmax(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    *,
    topk_per_row: int = 1,
    min_prob: float = 0.01,
    mutual_check: bool = False,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    max_matches: int = 512,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    if W_ab is None or bearing_a is None or bearing_b is None:
        device = W_ab.device if torch.is_tensor(W_ab) else torch.device("cpu")
        return _empty_corr(device)

    W = W_ab.float()
    b_a = _normalize(bearing_a)
    b_b = _normalize(bearing_b)
    B, Na, Nb = W.shape
    k = max(1, min(int(topk_per_row), Nb))
    max_matches = max(1, int(max_matches))
    vals, idx = torch.topk(W, k=k, dim=-1)
    rows = torch.arange(Na, device=W.device).view(1, Na, 1).expand(B, Na, k)
    keep = vals >= float(min_prob)

    if allowed_mask is not None:
        allowed_vals = torch.gather(allowed_mask.to(dtype=torch.bool), -1, idx)
        keep = keep & allowed_vals

    if mutual_check and W_ba is not None:
        ba_best = W_ba.float().argmax(dim=-1)
        mutual_rows = torch.gather(ba_best, 1, idx.reshape(B, -1)).reshape(B, Na, k)
        keep = keep & (mutual_rows == rows)

    out_a: List[torch.Tensor] = []
    out_b: List[torch.Tensor] = []
    out_w: List[torch.Tensor] = []
    diag = {"num_matches": [], "mean_prob": [], "mutual_ratio": [], "allowed_ratio": []}
    for bi in range(B):
        flat_keep = keep[bi].reshape(-1)
        flat_val = vals[bi].reshape(-1)
        flat_row = rows[bi].reshape(-1)
        flat_col = idx[bi].reshape(-1)
        kept = torch.nonzero(flat_keep, as_tuple=False).view(-1)
        if kept.numel() > max_matches:
            best_local = torch.topk(flat_val[kept], k=max_matches, dim=0).indices
            kept = kept[best_local]
        ma = b_a[bi, flat_row[kept]] if kept.numel() > 0 else b_a.new_zeros((0, 3))
        mb = b_b[bi, flat_col[kept]] if kept.numel() > 0 else b_b.new_zeros((0, 3))
        mw = flat_val[kept] if kept.numel() > 0 else W.new_zeros((0,))
        out_a.append(ma)
        out_b.append(mb)
        out_w.append(mw)
        diag["num_matches"].append(int(kept.numel()))
        diag["mean_prob"].append(float(mw.mean().detach().cpu()) if mw.numel() > 0 else 0.0)
        diag["allowed_ratio"].append(float(flat_keep.float().mean().detach().cpu()))
        if mutual_check and W_ba is not None:
            diag["mutual_ratio"].append(float(flat_keep.float().mean().detach().cpu()))
        else:
            diag["mutual_ratio"].append(float("nan"))
    return out_a, out_b, out_w, diag


def estimate_essential_weighted(
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_matches: int = 8,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    n = int(bearing_a.shape[0])
    if n < int(min_matches):
        return None, {"success": False, "reason": "not_enough_matches", "num_matches": n}
    a = _normalize(bearing_a)
    b = _normalize(bearing_b)
    w = weights.float().view(-1).clamp_min(1e-6)
    w = w / w.mean().clamp_min(1e-6)
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    xp, yp, zp = b[:, 0], b[:, 1], b[:, 2]
    A = torch.stack([xp * x, xp * y, xp * z, yp * x, yp * y, yp * z, zp * x, zp * y, zp * z], dim=-1)
    Aw = A * torch.sqrt(w).unsqueeze(-1)
    try:
        _u, _s, vh = torch.linalg.svd(Aw, full_matrices=False)
        E = vh[-1].reshape(3, 3)
        u2, s2, vh2 = torch.linalg.svd(E, full_matrices=False)
        s_new = torch.stack([0.5 * (s2[0] + s2[1]), 0.5 * (s2[0] + s2[1]), s2.new_zeros(())])
        E = u2 @ torch.diag(s_new) @ vh2
    except Exception as exc:
        return None, {"success": False, "reason": f"svd_failed:{type(exc).__name__}", "num_matches": n}
    if not torch.isfinite(E).all():
        return None, {"success": False, "reason": "nonfinite_E", "num_matches": n}
    return E, {"success": True, "reason": "ok", "num_matches": n}


def _project_rotation(R: torch.Tensor) -> torch.Tensor:
    u, _s, vh = torch.linalg.svd(R.float(), full_matrices=False)
    Rproj = u @ vh
    if torch.det(Rproj) < 0:
        u = u.clone()
        u[:, -1] *= -1
        Rproj = u @ vh
    return Rproj


def _rot_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    rel = R1.float() @ R2.float().transpose(-1, -2)
    cos = ((torch.trace(rel) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos)


def decompose_essential_with_network_prior(
    E: torch.Tensor,
    R_prior: torch.Tensor,
    t_prior: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    try:
        U, _S, Vh = torch.linalg.svd(E.float(), full_matrices=False)
    except Exception as exc:
        return None, None, {"success": False, "reason": f"svd_failed:{type(exc).__name__}"}
    if torch.det(U @ Vh) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1
    W = E.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R_candidates = [_project_rotation(U @ W @ Vh), _project_rotation(U @ W.transpose(0, 1) @ Vh)]
    t_base = _normalize(U[:, 2])
    t_candidates = [t_base, -t_base]
    R_prior = R_prior.float()
    t_prior = _normalize(t_prior.float().view(3))

    best_score: Optional[torch.Tensor] = None
    best_R: Optional[torch.Tensor] = None
    best_t: Optional[torch.Tensor] = None
    for R in R_candidates:
        r_score = _rot_angle(R, R_prior)
        for t in t_candidates:
            t_score = 1.0 - torch.sum(_normalize(t) * t_prior).clamp(-1.0, 1.0)
            score = r_score + t_score
            if best_score is None or bool(score < best_score):
                best_score = score
                best_R = R
                best_t = _normalize(t)
    if best_R is None or best_t is None or (not torch.isfinite(best_R).all()) or (not torch.isfinite(best_t).all()):
        return None, None, {"success": False, "reason": "invalid_decomposition"}
    return best_R, best_t, {"success": True, "reason": "ok", "selection_score": float(best_score.detach().cpu())}


def refine_pose_from_matches(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_prior: torch.Tensor,
    t_prior: torch.Tensor,
    *,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    min_prob: float = 0.01,
    max_matches: int = 512,
    mutual_check: bool = False,
    min_matches: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    corr_a, corr_b, corr_w, corr_diag = extract_correspondences_from_softmax(
        W_ab,
        bearing_a,
        bearing_b,
        topk_per_row=1,
        min_prob=min_prob,
        mutual_check=mutual_check,
        W_ba=W_ba,
        allowed_mask=allowed_mask,
        max_matches=max_matches,
    )
    R_out = R_prior.float().clone()
    t_out = _normalize(t_prior.float())
    success = []
    reasons = []
    for bi in range(R_prior.shape[0]):
        E, e_diag = estimate_essential_weighted(corr_a[bi], corr_b[bi], corr_w[bi], min_matches=min_matches)
        if E is None:
            success.append(False)
            reasons.append(e_diag.get("reason", "estimate_failed"))
            continue
        R_ref, t_ref, d_diag = decompose_essential_with_network_prior(E, R_prior[bi], t_out[bi])
        if R_ref is None or t_ref is None:
            success.append(False)
            reasons.append(d_diag.get("reason", "decompose_failed"))
            continue
        R_out[bi] = R_ref.to(R_out.device)
        t_out[bi] = t_ref.to(t_out.device)
        success.append(True)
        reasons.append("ok")
    diag = dict(corr_diag)
    diag["success"] = success
    diag["success_rate"] = float(sum(1 for x in success if x) / max(len(success), 1))
    diag["failure_reasons"] = reasons
    return R_out, t_out, diag
