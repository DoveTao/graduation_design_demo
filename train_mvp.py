# train_mvp.py
from __future__ import annotations

import time
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


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _print_startup_info(cfg: Config, device: torch.device, amp_enabled: bool):
    print("=" * 80)
    print("[Env ] torch", torch.__version__)
    print("[Env ] cuda available:", torch.cuda.is_available(), "| device:", device)
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        prop = torch.cuda.get_device_properties(idx)
        print(f"[Env ] GPU  : {name}")
        print(f"[Env ] VRAM : total={_bytes_to_gb(prop.total_memory):.2f} GB")
        print(f"[Env ] CC   : {prop.major}.{prop.minor}")
        print(f"[Env ] AMP  : {amp_enabled}")
        print(f"[Env ] TF32 : matmul={torch.backends.cuda.matmul.allow_tf32} | cudnn={torch.backends.cudnn.allow_tf32}")
        try:
            print(f"[Env ] matmul_precision: {torch.get_float32_matmul_precision()}")
        except Exception:
            pass
    else:
        print(f"[Env ] AMP  : {amp_enabled} (disabled on CPU)")

    det = torch.backends.cudnn.deterministic
    bench = torch.backends.cudnn.benchmark
    print(f"[Env ] cudnn: deterministic={det} | benchmark={bench}")
    if det:
        print("[Warn] deterministic=True 会明显变慢（尤其 conv / grid_sample）。若不需要严格复现，建议设 deterministic=False。")

    eff_bs = cfg.batch_size * max(1, int(cfg.grad_accum))
    print("-" * 80)
    print("[Cfg ] data_root:", cfg.data_root)
    print(f"[Cfg ] HxW={cfg.H}x{cfg.W} | D={cfg.D} | Nc={cfg.Nc} | Nf={cfg.Nf}")
    print(f"[Cfg ] topk_coarse={cfg.topk_coarse} | epi_angle={cfg.epi_angle_thresh_deg} | epi_bias={cfg.epi_bias_strength}")
    print(f"[Cfg ] bs={cfg.batch_size} | grad_accum={cfg.grad_accum} | eff_bs={eff_bs}")
    print(f"[Cfg ] lr={cfg.lr} | wd={cfg.weight_decay} | amp={cfg.amp}")
    print(f"[Cfg ] max_steps={cfg.max_steps} | log_every={cfg.log_every} | eval_every(upd)={cfg.eval_every} | max_eval_batches={cfg.max_eval_batches}")
    print(f"[Cfg ] num_workers={cfg.num_workers} | pin_memory={cfg.pin_memory}")
    print("=" * 80)


class EMA:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.v = None

    def update(self, x: float) -> float:
        if self.v is None:
            self.v = x
        else:
            self.v = self.v * (1.0 - self.alpha) + x * self.alpha
        return self.v


@torch.no_grad()
def eval_model(model, test_loader, device, max_batches: int | None = 50):
    model.eval()
    rot_errs = []
    t_errs = []
    t_abs_errs = []
    seen = 0

    for batch in test_loader:
        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, _ = model(IA, IB)

        # rot geodesic (deg) using trace
        Rp = R_pred
        Rg = R_gt
        tr = torch.einsum("bij,bji->b", Rp, Rg.transpose(-1, -2)).clamp(-1.0, 3.0)
        cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
        rot = torch.acos(cos_theta) * (180.0 / np.pi)

        # tdir angle (deg)
        tp = t_pred / (t_pred.norm(dim=-1, keepdim=True) + 1e-9)
        tg = t_gt / (t_gt.norm(dim=-1, keepdim=True) + 1e-9)
        cos_t = (tp * tg).sum(dim=-1).clamp(-1.0, 1.0)
        t_ang = torch.acos(cos_t) * (180.0 / np.pi)

        # sign-invariant tdir angle: acos(|cos|)
        t_ang_abs = torch.acos(cos_t.abs()) * (180.0 / np.pi)

        rot_errs.append(rot.detach().cpu())
        t_errs.append(t_ang.detach().cpu())
        t_abs_errs.append(t_ang_abs.detach().cpu())

        seen += IA.size(0)
        if max_batches is not None and seen >= max_batches:
            break

    rot_err = torch.cat(rot_errs).mean().item() if rot_errs else float("nan")
    t_err = torch.cat(t_errs).mean().item() if t_errs else float("nan")
    t_abs_err = torch.cat(t_abs_errs).mean().item() if t_abs_errs else float("nan")

    model.train()
    return rot_err, t_err, t_abs_err


def main():
    cfg = Config()

    # speed knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    deterministic = getattr(cfg, "deterministic", True)
    set_seed(cfg.seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (bool(cfg.amp) and device.type == "cuda")

    _print_startup_info(cfg, device, amp_enabled)

    # Dataset / loader
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
        max_dt = getattr(cfg, "max_dt", None)
        train_ds.max_dt = float(max_dt) if max_dt is not None else None
        print(
            f"[Data] MixedK strict dt: min_dt={cfg.min_dt} max_dt={max_dt} "
            f"k_choices={cfg.k_choices} k_probs={cfg.k_probs}"
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

    loader_kwargs = dict(
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=True,
    )
    if int(cfg.num_workers) > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=False,
    )

    print(f"[Data] len(train_ds)={len(train_ds)} | len(train_loader)={len(train_loader)} (batches per epoch-like)")
    print(f"[Data] len(test_ds)={len(test_ds)}")
    print("-" * 80)

    # Model / optim
    model = PanoramaRelPoseModel(cfg, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp_enabled,
    init_scale=1024.0,
    growth_interval=2000,
    )
    model.train()

    # probe param (to verify weights actually change)
    probe_param = None
    for p in model.parameters():
        probe_param = p
        break

    accum_steps = max(1, int(cfg.grad_accum))
    accum_count = 0  # only increment after successful backward

    update_step = 0           # REAL successful optimizer updates
    skip_updates = 0          # AMP skipped updates due to inf/nan grads

    train_it = iter(train_loader)
    epoch_like = 0

    ema_data = EMA(0.05)
    ema_fwd = EMA(0.05)
    ema_bwd = EMA(0.05)
    ema_opt = EMA(0.05)
    ema_iter = EMA(0.05)

    last_iter_t0 = time.perf_counter()

    last_grad_norm = float("nan")
    last_scale = scaler.get_scale() if amp_enabled else 1.0

    for step in range(int(cfg.max_steps)):
        # data time
        t0 = time.perf_counter()
        try:
            batch = next(train_it)
        except StopIteration:
            epoch_like += 1
            train_it = iter(train_loader)
            batch = next(train_it)
        t1 = time.perf_counter()
        data_t = t1 - t0

        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        # forward + loss
        t2 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            R_pred, t_pred, aux = model(IA, IB)

            L_pose = pose_loss(R_pred, t_pred, R_gt, t_gt)
            L_x = xlevel_loss(aux["Wc_tilde"], aux["Wc_ab"])
            L_cycle = cycle_loss(aux["Wf_ab"], aux["Wf_ba"])
            L_rel = reliability_reg_loss(aux["cA_f"], aux["cB_f"])

            L_pose_w = reliability_weighted(L_pose, aux["cA_f"], aux["cB_f"])
            L_x_w = reliability_weighted(L_x, aux["cA_f"], aux["cB_f"])
            L_cycle_w = reliability_weighted(L_cycle, aux["cA_f"], aux["cB_f"])

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

            L_total = (
                L_pose_w
                + float(cfg.lam_x) * L_x_w
                + float(cfg.lam_c) * L_cycle_w
                + float(cfg.lam_r) * L_rel
                + float(cfg.lam_e) * L_epi
            )
            L = L_total / accum_steps

        t3 = time.perf_counter()
        fwd_t = t3 - t2

        # NaN/Inf guard (loss-level)
        if not torch.isfinite(L_total.detach()).item():
            meta = batch.get("meta", {})
            print("[NaN] non-finite loss detected. meta=", meta)
            print("      L_pose=", float(L_pose.detach().float().cpu()) if torch.isfinite(L_pose) else L_pose)
            print("      L_x   =", float(L_x.detach().float().cpu()) if torch.isfinite(L_x) else L_x)
            print("      L_cyc =", float(L_cycle.detach().float().cpu()) if torch.isfinite(L_cycle) else L_cycle)
            print("      L_rel =", float(L_rel.detach().float().cpu()) if torch.isfinite(L_rel) else L_rel)
            print("      L_epi =", float(L_epi.detach().float().cpu()) if torch.isfinite(L_epi) else L_epi)
            opt.zero_grad(set_to_none=True)
            accum_count = 0
            continue

        # backward
        t4 = time.perf_counter()
        scaler.scale(L).backward()
        t5 = time.perf_counter()
        bwd_t = t5 - t4

        accum_count += 1

        did_update = False
        opt_t = 0.0

        if accum_count >= accum_steps:
            t6 = time.perf_counter()

            scale_before = scaler.get_scale() if amp_enabled else 1.0

            if amp_enabled:
                scaler.unscale_(opt)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_grad_norm = float(grad_norm.detach().float().cpu()) if torch.isfinite(grad_norm) else float("nan")

            # if grad_norm is non-finite, force skip (avoid fake update_step)
            if not torch.isfinite(grad_norm):
                skip_updates += 1
                opt.zero_grad(set_to_none=True)
                scaler.update()  # shrink scale
                accum_count = 0
                last_scale = scaler.get_scale() if amp_enabled else 1.0
            else:
                p_before = probe_param.detach().float().clone()

                scaler.step(opt)
                scaler.update()

                p_after = probe_param.detach().float()
                did_real_step = (p_after - p_before).abs().max().item() > 0.0
                if did_real_step:
                    update_step += 1
                else:
                    skip_updates += 1

                opt.zero_grad(set_to_none=True)

                scale_after = scaler.get_scale() if amp_enabled else 1.0
                last_scale = float(scale_after)

                # Heuristic: if scale decreased, scaler detected inf and skipped the step
                stepped = (scale_after >= scale_before) if amp_enabled else True

                if stepped:
                    update_step += 1
                    did_update = True
                else:
                    skip_updates += 1
                    did_update = False

                accum_count = 0

            t7 = time.perf_counter()
            opt_t = t7 - t6

        # iter timing
        now = time.perf_counter()
        iter_t = now - last_iter_t0
        last_iter_t0 = now

        ema_data.update(data_t)
        ema_fwd.update(fwd_t)
        ema_bwd.update(bwd_t)
        ema_opt.update(opt_t)
        ema_iter.update(iter_t)

        # logging
        if step % int(cfg.log_every) == 0:
            it = ema_iter.v if ema_iter.v else 1e-6
            sps = 1.0 / it

            p0_mean = float(probe_param.detach().float().mean().cpu()) if probe_param is not None else float("nan")

            print(
                f"[Train] step {step:05d} upd {update_step:05d} ep{epoch_like:03d} | "
                f"L={L_total.item():.3f} | pose={L_pose.item():.3f} | "
                f"x={L_x.item():.4f} | cyc={L_cycle.item():.4f} | "
                f"rel={L_rel.item():.3f} | epi={L_epi.item():.3f}"
            )
            print(
                f"[Time ] avg/iter={it*1000:.1f}ms ({sps:.2f} it/s) | "
                f"data={ema_data.v*1000:.1f}ms fwd={ema_fwd.v*1000:.1f}ms "
                f"bwd={ema_bwd.v*1000:.1f}ms opt={ema_opt.v*1000:.1f}ms"
            )
            print(
                f"[Stat ] grad_norm={last_grad_norm:.4f} | scaler_scale={last_scale:.1f} | "
                f"skip_updates={skip_updates} | p0_mean={p0_mean:.6e}"
            )
            if step == 0:
                print("[Shapes] IA", tuple(IA.shape), "| R", tuple(R_pred.shape), "| t", tuple(t_pred.shape))
                print("-" * 80)

        # eval only after REAL updates
        if did_update and int(cfg.eval_every) > 0 and (update_step % int(cfg.eval_every) == 0):
            t_eval0 = time.perf_counter()
            rot_err, t_err, t_abs_err = eval_model(model, test_loader, device, max_batches=int(cfg.max_eval_batches))
            t_eval1 = time.perf_counter()
            print(
                f"[Eval ] upd {update_step:05d} (step {step:05d}) | "
                f"rot={rot_err:.4f}° | tdir={t_err:.4f}° | tdir_abs={t_abs_err:.4f}° | "
                f"time={(t_eval1-t_eval0):.2f}s"
            )
            # avoid eval time contaminating next iter_t
            last_iter_t0 = time.perf_counter()

    print("[Done] training finished.")


if __name__ == "__main__":
    main()
