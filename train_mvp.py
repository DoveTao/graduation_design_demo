\
import os
import time
import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsMixedK
from model import PanoramaRelPoseModel
from losses import pose_loss, epipolar_simplified_loss
from pose_head import matrix_geodesic_distance


def _fmt_deg(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


@torch.no_grad()
def eval_model(model, loader, device, cfg: Config):
    model.eval()

    rot_sum = 0.0
    tdir_sum = 0.0
    tdir_abs_sum = 0.0
    n = 0

    # TDIR-CHK accumulators (diagnose frame/sign convention)
    acc = {
        "raw": 0.0,
        "flip": 0.0,
        "R@t": 0.0,
        "R@(-t)": 0.0,
        "Rt@t": 0.0,
        "Rt@(-t)": 0.0,
        "cnt": 0,
    }

    # AMP in eval: usually safe to disable (keep fp32)
    for bi, batch in enumerate(loader):
        if cfg.max_eval_batches and bi >= cfg.max_eval_batches:
            break

        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, _aux = model(IA, IB)

        # rotation error (deg)
        rot_rad = matrix_geodesic_distance(R_pred.float(), R_gt.float())  # [B]
        rot_deg = rot_rad * (180.0 / math.pi)

        # translation-direction error (deg)
        tp = F.normalize(t_pred.float(), dim=-1, eps=1e-6)
        tg = F.normalize(t_gt.float(), dim=-1, eps=1e-6)
        cos = torch.sum(tp * tg, dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos) * (180.0 / math.pi)  # [B]
        ang_abs = torch.minimum(ang, 180.0 - ang)

        bsz = IA.shape[0]
        rot_sum += float(rot_deg.sum().cpu())
        tdir_sum += float(ang.sum().cpu())
        tdir_abs_sum += float(ang_abs.sum().cpu())
        n += bsz

        # ---- TDIR-CHK: what if the convention is different? ----
        # Dataset convention:
        #   - R_gt = R_{B<-A} (rotation A -> B)
        #   - t_gt = t_{BA} in frame B (baseline B -> A; origin(A) in B)
        #
        # The checks below compare t_pred to several transformed variants of t_gt:
        #
        #   raw      : tp vs  tg               => tp is B-frame, baseline B->A (matches dataset)
        #   flip     : tp vs -tg               => tp is B-frame, baseline A->B
        #   R@t      : tp vs  R_gt @ tg        => tp is B-frame, but tg was actually A-frame (B->A), rotate A->B
        #   R@(-t)   : tp vs -R_gt @ tg        => tp is B-frame, but tg was A-frame (A->B), rotate A->B
        #   Rt@t     : tp vs  R_gt^T @ tg      => tp is A-frame, baseline B->A (rotate B->A into A)
        #   Rt@(-t)  : tp vs -R_gt^T @ tg      => tp is A-frame, baseline A->B
        #
        # If one of these is consistently much smaller than the others, that is the
        # convention your network is implicitly using.
        tg_R = F.normalize(torch.matmul(R_gt.float(), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)
        tg_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)

        def ang_deg(a, b):
            c = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
            return torch.acos(c) * (180.0 / math.pi)

        acc["raw"] += float(ang_deg(tp, tg).sum().cpu())
        acc["flip"] += float(ang_deg(tp, -tg).sum().cpu())
        acc["R@t"] += float(ang_deg(tp, tg_R).sum().cpu())
        acc["R@(-t)"] += float(ang_deg(tp, -tg_R).sum().cpu())
        acc["Rt@t"] += float(ang_deg(tp, tg_Rt).sum().cpu())
        acc["Rt@(-t)"] += float(ang_deg(tp, -tg_Rt).sum().cpu())
        acc["cnt"] += bsz

    rot = rot_sum / max(n, 1)
    tdir = tdir_sum / max(n, 1)
    tdir_abs = tdir_abs_sum / max(n, 1)

    if acc["cnt"] > 0:
        denom = float(acc["cnt"])
        msg = (
            f"[TDIR-CHK] "
            f"raw(B: B->A)={acc['raw']/denom:.2f}  "
            f"flip(B: A->B)={acc['flip']/denom:.2f}  "
            f"R@t(B, gt-in-A,B->A)={acc['R@t']/denom:.2f}  "
            f"R@(-t)(B, gt-in-A,A->B)={acc['R@(-t)']/denom:.2f}  "
            f"Rt@t(A, gt-in-B,B->A)={acc['Rt@t']/denom:.2f}  "
            f"Rt@(-t)(A, gt-in-B,A->B)={acc['Rt@(-t)']/denom:.2f}  "
            f"(n={int(denom)})"
        )
    else:
        msg = "[TDIR-CHK] n=0"

    model.train()
    return rot, tdir, tdir_abs, msg


def main():
    cfg = Config()

    # ---------------- Env / perf knobs ----------------
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print("=" * 80)
    print(f"[Env ] torch {torch.__version__}")
    print(f"[Env ] cuda available: {torch.cuda.is_available()} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"[Env ] GPU  : {props.name}")
        print(f"[Env ] VRAM : total={props.total_memory/1024**3:.2f} GB")
        print(f"[Env ] CC   : {props.major}.{props.minor}")
        try:
            print(f"[Env ] BF16 : {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    else:
        dev = torch.device("cpu")

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.benchmark)

    print(f"[Env ] AMP  : {cfg.amp} (dtype={cfg.amp_dtype})")
    print(f"[Env ] TF32 : matmul={torch.backends.cuda.matmul.allow_tf32} | cudnn={torch.backends.cudnn.allow_tf32}")
    print(f"[Env ] matmul_precision: {cfg.matmul_precision}")
    print(f"[Env ] cudnn: deterministic={torch.backends.cudnn.deterministic} | benchmark={torch.backends.cudnn.benchmark}")
    print("-" * 80)

    # ---------------- Config print ----------------
    eff_bs = cfg.batch_size * cfg.grad_accum
    print(f"[Cfg ] data_root: {cfg.data_root}")
    print(f"[Cfg ] HxW={cfg.H}x{cfg.W} | D={cfg.D} | Nc={cfg.Nc} | Nf={cfg.Nf}")
    print(f"[Cfg ] topk_coarse={cfg.topk_coarse} | epi_angle={cfg.epi_angle_thresh_deg} | epi_bias={cfg.epi_bias_strength}")
    print(f"[Cfg ] bs={cfg.batch_size} | grad_accum={cfg.grad_accum} | eff_bs={eff_bs}")
    print(f"[Cfg ] lr={cfg.lr} | wd={cfg.wd} | amp={cfg.amp}")
    print(f"[Cfg ] max_steps={cfg.max_steps} | log_every={cfg.log_every} | eval_every(upd)={cfg.eval_every} | max_eval_batches={cfg.max_eval_batches}")
    print(f"[Cfg ] num_workers={cfg.num_workers} | pin_memory={cfg.pin_memory}")
    print(f"[Cfg ] split_by={cfg.split_by} | train_ratio={cfg.train_ratio} | split_seed={cfg.split_seed} | data_seed={cfg.data_seed}")
    print("=" * 80)

    # ---------------- Data ----------------
    train_ds = RflyPanoPanoramaPairsMixedK(
        data_root=cfg.data_root,
        split="train",
        split_by=cfg.split_by,
        train_ratio=cfg.train_ratio,
        split_seed=cfg.split_seed,
        seed=cfg.data_seed,
        H=cfg.H,
        W=cfg.W,
        min_dt=cfg.min_dt,
        k_choices=cfg.k_choices,
        k_probs=cfg.k_probs,
        strict_dt=True,
    )
    train_ds.max_dt = cfg.max_dt

    test_ds = RflyPanoPanoramaPairsMixedK(
        data_root=cfg.data_root,
        split="test",
        split_by=cfg.split_by,
        train_ratio=cfg.train_ratio,
        split_seed=cfg.split_seed,
        seed=cfg.data_seed,
        H=cfg.H,
        W=cfg.W,
        min_dt=cfg.min_dt,
        k_choices=cfg.k_choices,
        k_probs=cfg.k_probs,
        strict_dt=True,
    )
    test_ds.max_dt = cfg.max_dt

    train_keys = set(getattr(train_ds, "sequence_keys", []))
    test_keys = set(getattr(test_ds, "sequence_keys", []))
    overlap = train_keys & test_keys
    if overlap:
        overlap_preview = sorted(list(overlap))[:8]
        raise RuntimeError(f"train/test split leakage detected: {len(overlap)} overlapping (scene, seq), e.g. {overlap_preview}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    train_summary = getattr(train_ds, "split_summary", None)
    test_summary = getattr(test_ds, "split_summary", None)
    if train_summary is not None:
        train_preview = train_summary.get("selected_seq_keys", [])[:5]
        print(
            f"[Split] train by {train_summary['split_by']} | "
            f"groups={train_summary['num_selected_groups']}/{train_summary['num_total_groups']} | "
            f"seqs={train_summary['num_selected_seq']}/{train_summary['num_total_seq']} | "
            f"pairs={train_summary.get('num_pairs', 'NA')} | preview={train_preview}"
        )
    if test_summary is not None:
        test_preview = test_summary.get("selected_seq_keys", [])[:5]
        print(
            f"[Split] test  by {test_summary['split_by']} | "
            f"groups={test_summary['num_selected_groups']}/{test_summary['num_total_groups']} | "
            f"seqs={test_summary['num_selected_seq']}/{test_summary['num_total_seq']} | "
            f"pairs={test_summary.get('num_pairs', 'NA')} | preview={test_preview}"
        )

    print(f"[Data] MixedK strict dt: min_dt={cfg.min_dt} max_dt={cfg.max_dt} k_choices={list(cfg.k_choices)} k_probs={list(cfg.k_probs)}")
    print(f"[Data] len(train_ds)={len(train_ds)} | len(train_loader)={len(train_loader)} (batches per epoch-like)")
    print(f"[Data] len(test_ds)={len(test_ds)}")
    print("-" * 80)

    # ---------------- Model ----------------
    model = PanoramaRelPoseModel(cfg, device=dev).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # AMP mode selection
    use_amp = bool(cfg.amp) and dev.type == "cuda"
    if cfg.amp_dtype == "auto":
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    elif cfg.amp_dtype.lower() in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---------------- Training loop ----------------
    step = 0
    upd = 0
    skip_updates = 0
    bad_forward = 0
    last_bad_grad_name = "-"
    t_last = time.perf_counter()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t0 = time.perf_counter()

        IA = batch["IA"].to(dev, non_blocking=True)
        IB = batch["IB"].to(dev, non_blocking=True)
        R_gt = batch["R_gt"].to(dev, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(dev, non_blocking=True)

        t_data = time.perf_counter()

        # Forward
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            R_pred, t_pred, aux = model(IA, IB)

            # core pose loss (float32 inside)
            L_pose = pose_loss(R_pred, t_pred, R_gt, t_gt, pose_t_alpha=cfg.pose_t_alpha)

            # optional regularizers (guarded if aux is missing keys)
            Wf_ab = aux.get("Wf_ab", None)
            Wf_ba = aux.get("Wf_ba", None)
            cA_f = aux.get("cA_f", None)
            cB_f = aux.get("cB_f", None)

            # entropy: encourage peaky assignment (very small in practice)
            if Wf_ab is not None:
                P = Wf_ab.float().clamp(min=1e-9)
                L_x = (-P * P.log()).sum(dim=-1).mean()
            else:
                L_x = torch.zeros((), device=dev)

            # cycle: W_ab @ W_ba ≈ I
            if (Wf_ab is not None) and (Wf_ba is not None):
                C = torch.matmul(Wf_ab.float(), Wf_ba.float())  # [B,NfA,NfA]
                I = torch.eye(C.shape[-1], device=dev).unsqueeze(0)
                L_cyc = F.mse_loss(C, I)
            else:
                L_cyc = torch.zeros((), device=dev)

            # reliability: logits should be high (target=1)
            if (cA_f is not None) and (cB_f is not None):
                L_rel = 0.5 * (F.softplus(-cA_f.float()).mean() + F.softplus(-cB_f.float()).mean())
            else:
                L_rel = torch.zeros((), device=dev)

            # epipolar (rotation-only, uses fine bearings if present)
            if ("bearingA_f" in aux) and ("bearingB_f" in aux):
                L_epi = epipolar_simplified_loss(
                    aux["bearingA_f"],
                    aux["bearingB_f"],
                    R_gt,
                    angle_thresh_deg=cfg.epi_angle_thresh_deg,
                )
            else:
                L_epi = torch.zeros((), device=dev)

            L = (
                cfg.w_pose * L_pose
                + cfg.w_x * L_x
                + cfg.w_cyc * L_cyc
                + cfg.w_rel * L_rel
                + cfg.w_epi * L_epi
            )

            # scale for grad accumulation
            L_scaled = L / float(cfg.grad_accum)

        t_fwd = time.perf_counter()

        forward_ok = (
            torch.isfinite(R_pred).all() and
            torch.isfinite(t_pred).all() and
            torch.isfinite(L_pose) and
            torch.isfinite(L_x) and
            torch.isfinite(L_cyc) and
            torch.isfinite(L_rel) and
            torch.isfinite(L_epi) and
            torch.isfinite(L)
        )
        if not bool(forward_ok):
            bad_forward += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                print(f"[Warn ] non-finite forward detected, batch skipped | bad_forward={bad_forward}")
            continue

        # Backward
        if use_scaler:
            scaler.scale(L_scaled).backward()
        else:
            L_scaled.backward()

        t_bwd = time.perf_counter()

        # Step on accumulation boundary
        if (step + 1) % cfg.grad_accum == 0:
            # unscale before clipping when using scaler
            if use_scaler:
                scaler.unscale_(optimizer)

            bad_name = None
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_name = name
                    break

            if bad_name is not None:
                skip_updates += 1
                last_bad_grad_name = bad_name
                optimizer.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
                grad_norm = torch.tensor(float("nan"), device=dev)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skip_updates += 1
                    last_bad_grad_name = "clip_grad_norm"
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                else:
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    upd += 1

                    # periodic eval
                    if cfg.eval_every and (upd % cfg.eval_every == 0):
                        t_eval0 = time.perf_counter()
                        rot, tdir, tdir_abs, tdir_msg = eval_model(model, test_loader, dev, cfg)
                        t_eval = time.perf_counter() - t_eval0
                        print(f"[Eval ] upd {upd:05d} (step {step:05d}) | rot={rot:.4f}° | tdir={tdir:.4f}° | tdir_abs={tdir_abs:.4f}° | time={t_eval:.2f}s")
                        print(tdir_msg)

        t_opt = time.perf_counter()

        # logging
        if step % cfg.log_every == 0:
            # p0_mean as quick param sanity
            with torch.no_grad():
                p0 = next(model.parameters())
                p0_mean = float(p0.mean().detach().cpu())

            # AMP scale
            scaler_scale = float(scaler.get_scale()) if use_scaler else 1.0

            dt = time.perf_counter() - t0
            data_ms = (t_data - t0) * 1000.0
            fwd_ms = (t_fwd - t_data) * 1000.0
            bwd_ms = (t_bwd - t_fwd) * 1000.0
            opt_ms = (t_opt - t_bwd) * 1000.0

            # show current "ep" approximately
            ep = step // max(len(train_loader), 1)

            print(
                f"[Train] step {step:05d} upd {upd:05d} ep{ep:03d} | "
                f"L={float(L.detach().cpu()):.3f} | "
                f"pose={float(L_pose.detach().cpu()):.3f} | "
                f"x={float(L_x.detach().cpu()):.4f} | "
                f"cyc={float(L_cyc.detach().cpu()):.4f} | "
                f"rel={float(L_rel.detach().cpu()):.3f} | "
                f"epi={float(L_epi.detach().cpu()):.3f}"
            )
            it_s = 1.0 / max(dt, 1e-9)
            print(
                f"[Time ] avg/iter={dt*1000.0:.1f}ms ({it_s:.2f} it/s) | "
                f"data={data_ms:.1f}ms fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms opt={opt_ms:.1f}ms"
            )

            # grad_norm is only valid on update boundary; otherwise print NaN-like
            if (step + 1) % cfg.grad_accum == 0:
                g = float(grad_norm.detach().cpu()) if torch.isfinite(grad_norm) else float("nan")
            else:
                g = float("nan")

            print(f"[Stat ] grad_norm={g} | scaler_scale={scaler_scale} | skip_updates={skip_updates} | bad_forward={bad_forward} | bad_grad={last_bad_grad_name} | p0_mean={p0_mean:.6e}")

            if step == 0:
                # debug shapes
                if isinstance(aux, dict):
                    for k in ["Wc_ab", "Wf_ab", "bearingA_f", "bearingB_f"]:
                        if k in aux:
                            print(f"[DBG] {k}: {tuple(aux[k].shape)}")
                print(f"[Shapes] IA {tuple(IA.shape)} | R {tuple(R_gt.shape)} | t {tuple(t_gt.shape)}")
                print("-" * 80)

        step += 1


if __name__ == "__main__":
    main()
