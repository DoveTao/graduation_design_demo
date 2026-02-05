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
    epipolar_simplified_loss,
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
    n = 0

    for batch in test_loader:
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt_dir = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, _ = model(IA, IB)

        # rotation error (deg): acos((trace(R_gt^T R_pred)-1)/2)
        R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)  # [B,3,3]
        tr = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # [B]
        cos = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        rot_deg = torch.rad2deg(torch.acos(cos))              # [B]

        # translation direction error (deg)
        t_pred_dir = t_pred / (t_pred.norm(dim=-1, keepdim=True) + 1e-9)
        cos_t = torch.clamp((t_pred_dir * t_gt_dir).sum(-1), -1.0, 1.0)
        t_deg = torch.rad2deg(torch.acos(cos_t))              # [B]

        rot_errs.append(rot_deg.detach().cpu())
        t_errs.append(t_deg.detach().cpu())

        n += 1
        if (max_batches is not None) and (n >= max_batches):
            break

    rot_mean = torch.cat(rot_errs).mean().item() if rot_errs else float("nan")
    t_mean = torch.cat(t_errs).mean().item() if t_errs else float("nan")

    model.train()
    return rot_mean, t_mean


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (cfg.amp and device.type == "cuda")
    accum_steps = max(1, int(getattr(cfg, "grad_accum", 1)))

    print(f"[Info] device={device}, amp={cfg.amp}, grad_accum={accum_steps}")
    print(f"[Info] data_root={cfg.data_root}")
    print(f"[Info] pano size HxW={cfg.H}x{cfg.W}, batch={cfg.batch_size}")
    print(f"[Info] train split: scenes={cfg.train_scenes}, seqs={cfg.train_seqs}")
    print(f"[Info] test  split: scenes={cfg.test_scenes},  seqs={cfg.test_seqs}")

    # ---- Datasets / Loaders ----
    if cfg.use_mixed_k:
        train_set = RflyPanoPanoramaPairsMixedK(
            data_root=cfg.data_root,
            hw=(cfg.H, cfg.W),
            k_choices=cfg.k_choices,
            k_probs=cfg.k_probs,
            pair_step=cfg.pair_step,
            scenes=cfg.train_scenes,
            seqs=cfg.train_seqs,
            min_dt=cfg.min_dt,
            max_tries=cfg.max_tries,
            seed=cfg.seed,
        )
        print(f"[Info] train sampling: mixed_k choices={cfg.k_choices}, probs={cfg.k_probs}, min_dt={cfg.min_dt}")
    else:
        train_set = RflyPanoPanoramaPairs(
            data_root=cfg.data_root,
            hw=(cfg.H, cfg.W),
            k_stride=cfg.train_k_stride,
            pair_step=cfg.pair_step,
            scenes=cfg.train_scenes,
            seqs=cfg.train_seqs,
        )
        print(f"[Info] train sampling: fixed_k k_stride={cfg.train_k_stride}")

    test_set = RflyPanoPanoramaPairs(
        data_root=cfg.data_root,
        hw=(cfg.H, cfg.W),
        k_stride=cfg.test_k_stride,
        pair_step=cfg.pair_step,
        scenes=cfg.test_scenes,
        seqs=cfg.test_seqs,
    )
    print(f"[Info] test sampling: fixed_k k_stride={cfg.test_k_stride}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    print(f"[Info] train pairs = {len(train_set)}, test pairs = {len(test_set)}")
    print(f"[Info] max_steps={cfg.max_steps}, log_every={cfg.log_every}, eval_every(updates)={cfg.eval_every}")

    # ---- Model / Optim ----
    model = PanoramaRelPoseModel(cfg, device=device).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Baseline Eval (before training) ----
    rot0, t0 = eval_model(model, test_loader, device, max_batches=cfg.max_eval_batches)
    print(f"[Eval ] baseline | rot={rot0:.4f}° | tdir={t0:.4f}°")

    # ---- Training loop with gradient accumulation ----
    it = iter(train_loader)
    opt.zero_grad(set_to_none=True)

    update_step = 0  # true optimizer updates

    for step in range(cfg.max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        IA = batch["IA"].to(device, non_blocking=True)             # [B,3,H,W]
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)         # [B,3,3]
        t_gt_dir = batch["t_gt_dir"].to(device, non_blocking=True) # [B,3]
        B = IA.shape[0]

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            R_pred, t_pred, aux = model(IA, IB)

            # Core losses
            L_pose = pose_loss(R_pred, t_pred, R_gt, t_gt_dir)
            L_x = xlevel_loss(aux["Wc_tilde"], aux["Wc_ab"])

            L_cycle_c = cycle_loss(aux["Wc_ab"], aux["Wc_ba"])
            L_cycle_f = cycle_loss(aux["Wf_ab"], aux["Wf_ba"])
            L_cycle = 0.5 * (L_cycle_c + L_cycle_f)

            # Reliability
            L_pose_w = reliability_weighted(L_pose, aux["cA_f"], aux["cB_f"])
            L_x_w = reliability_weighted(L_x, aux["cA_f"], aux["cB_f"])
            L_cycle_w = reliability_weighted(L_cycle, aux["cA_f"], aux["cB_f"])
            L_rel = reliability_reg_loss(aux["cA_f"], aux["cB_f"])

            # Epipolar simplified
            if ("bearingA_f" in aux) and ("bearingB_f" in aux):
                bearing_a = aux["bearingA_f"]
                bearing_b = aux["bearingB_f"]
            else:
                fine_b = model.module2.hier.fine_bearing.to(device)  # [Nf,3]
                bearing_a = fine_b.view(1, cfg.Nf, 3).expand(B, -1, -1)
                bearing_b = fine_b.view(1, cfg.Nf, 3).expand(B, -1, -1)

            L_epi = epipolar_simplified_loss(
                R=aux["Rc"],
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
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            update_step += 1
            did_update = True

        # ---- Logging ----
        if step % cfg.log_every == 0:
            # show both "step" and "update_step" to avoid confusion under accumulation
            print(
                f"[Train] step {step:05d} upd {update_step:05d} | "
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

    # ---- Flush remaining grads if max_steps not divisible by accum_steps ----
    if (cfg.max_steps % accum_steps) != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        update_step += 1
        rot_err, t_err = eval_model(model, test_loader, device, max_batches=cfg.max_eval_batches)
        print(f"[Eval ] upd {update_step:05d} (final flush) | rot={rot_err:.4f}° | tdir={t_err:.4f}°")

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()
