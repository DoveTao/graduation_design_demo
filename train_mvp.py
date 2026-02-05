# train_mvp.py
from __future__ import annotations

import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import Config
from model import PanoramaRelPoseModel
from dataset_pano_only import RflyPanoPanoramaPairs, RflyPanoPanoramaPairsMixedK
from losses import (
    pose_loss,
    xlevel_loss,
    cycle_loss,
    reliability_weighted,
    reliability_reg_loss,
    epipolar_strict_loss,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_model(model, test_loader, device, max_batches: int | None = 50):
    model.eval()
    rot_errs = []
    t_errs = []
    seen = 0

    for batch in test_loader:
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, _ = model(IA, IB)

        # rotation geodesic (deg) via trace
        Rp = R_pred
        Rg = R_gt
        tr = torch.einsum("bij,bji->b", Rp, Rg.transpose(-1, -2)).clamp(-1.0, 3.0)
        cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
        rot = torch.acos(cos_theta) * (180.0 / np.pi)

        # translation direction angle (deg)
        tp = t_pred / (t_pred.norm(dim=-1, keepdim=True) + 1e-9)
        tg = t_gt / (t_gt.norm(dim=-1, keepdim=True) + 1e-9)
        cos_t = (tp * tg).sum(dim=-1).clamp(-1.0, 1.0)
        t_ang = torch.acos(cos_t) * (180.0 / np.pi)

        rot_errs.append(rot.detach().cpu())
        t_errs.append(t_ang.detach().cpu())

        seen += IA.size(0)
        if max_batches is not None and seen >= max_batches:
            break

    rot_err = torch.cat(rot_errs).mean().item() if rot_errs else float("nan")
    t_err = torch.cat(t_errs).mean().item() if t_errs else float("nan")
    model.train()
    return rot_err, t_err


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (cfg.amp and device.type == "cuda")

    # -------------------------
    # Dataset / loader
    # -------------------------
    if cfg.use_mixed_k:
        train_ds = RflyPanoPanoramaPairsMixedK(
            data_root=cfg.data_root,
            scenes=cfg.train_scenes,
            seqs=cfg.train_seqs,
            hw=(cfg.H, cfg.W),
            k_choices=cfg.k_choices,
            k_probs=cfg.k_probs,
            pair_step=cfg.pair_step,
            min_dt=cfg.min_dt,
            max_tries=cfg.max_tries,
            seed=cfg.seed,
        )
    else:
        train_ds = RflyPanoPanoramaPairs(
            data_root=cfg.data_root,
            scenes=cfg.train_scenes,
            seqs=cfg.train_seqs,
            hw=(cfg.H, cfg.W),
            k_stride=cfg.test_k_stride,
            pair_step=cfg.pair_step,
        )

    test_ds = RflyPanoPanoramaPairs(
        data_root=cfg.data_root,
        scenes=cfg.test_scenes,
        seqs=cfg.test_seqs,
        hw=(cfg.H, cfg.W),
        k_stride=cfg.test_k_stride,
        pair_step=cfg.pair_step,
    )

    # DataLoader 性能设置
    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    if cfg.num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    print(f"[Data] len(train_ds)={len(train_ds)} | len(train_loader)={len(train_loader)}")
    print(f"[Data] len(test_ds)={len(test_ds)}")

    # -------------------------
    # Model / optim
    # -------------------------
    model = PanoramaRelPoseModel(cfg, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()

    # gradient accumulation
    accum_steps = max(1, int(cfg.grad_accum))
    update_step = 0

    # -------------------------
    # IMPORTANT: run to cfg.max_steps even if dataloader exhausts
    # -------------------------
    train_it = iter(train_loader)
    epoch_like = 0  # just for debug printing

    for step in range(int(cfg.max_steps)):
        try:
            batch = next(train_it)
        except StopIteration:
            epoch_like += 1
            train_it = iter(train_loader)
            batch = next(train_it)

        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            R_pred, t_pred, aux = model(IA, IB)

            # --- Pose loss (base) ---
            L_pose = pose_loss(R_pred, t_pred, R_gt, t_gt)

            # --- Cross-level consistency ---
            L_x = xlevel_loss(aux["Wc_tilde"], aux["Wc_ab"])

            # --- Cycle consistency (fine) ---
            L_cycle = cycle_loss(aux["Wf_ab"], aux["Wf_ba"])

            # --- Reliability regularizer ---
            L_rel = reliability_reg_loss(aux["cA_f"], aux["cB_f"])

            # reliability-weighted pose/x/cycle
            L_pose_w = reliability_weighted(L_pose, aux["cA_f"], aux["cB_f"])
            L_x_w = reliability_weighted(L_x, aux["cA_f"], aux["cB_f"])
            L_cycle_w = reliability_weighted(L_cycle, aux["cA_f"], aux["cB_f"])

            # --- Strict epipolar loss ---
            bearing_a = aux.get("bearingA_f", None)
            bearing_b = aux.get("bearingB_f", None)
            if bearing_a is None or bearing_b is None:
                fine_b = model.module2.hier.fine_bearing.to(device)  # [Nf,3]
                Bsz = IA.size(0)
                bearing_a = fine_b.view(1, cfg.Nf, 3).expand(Bsz, -1, -1)
                bearing_b = fine_b.view(1, cfg.Nf, 3).expand(Bsz, -1, -1)

            L_epi = epipolar_strict_loss(
                R=aux["Rc"],
                t_dir=aux["tc_dir"],
                W_ab=aux["Wf_ab"],
                bearing_a=bearing_a,
                bearing_b=bearing_b,
            )

            # Total (unscaled)
            L_total = (
                L_pose_w
                + cfg.lam_x * L_x_w
                + cfg.lam_c * L_cycle_w
                + cfg.lam_r * L_rel
                + cfg.lam_e * L_epi
            )

            # Scale for accumulation
            L = L_total / accum_steps

        scaler.scale(L).backward()

        did_update = False
        if (step + 1) % accum_steps == 0:
            # AMP: unscale before clipping
            if amp_enabled:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            update_step += 1
            did_update = True

        # ---- Logging ----
        if step % cfg.log_every == 0:
            print(
                f"[Train] step {step:05d} upd {update_step:05d} ep{epoch_like:03d} | "
                f"L={L_total.item():.3f} | pose={L_pose.item():.3f} | "
                f"x={L_x.item():.4f} | cyc={L_cycle.item():.4f} | "
                f"rel={L_rel.item():.3f} | epi={L_epi.item():.3f}"
            )
            if step == 0:
                print("[Shapes] IA", tuple(IA.shape), "| R", tuple(R_pred.shape), "| t", tuple(t_pred.shape))

        # ---- Eval (by update_step; only after a real update) ----
        if did_update and (update_step % cfg.eval_every) == 0:
            rot_err, t_err = eval_model(model, test_loader, device, max_batches=cfg.max_eval_batches)
            print(f"[Eval ] upd {update_step:05d} (step {step:05d}) | rot={rot_err:.4f}° | tdir={t_err:.4f}°")


if __name__ == "__main__":
    main()
