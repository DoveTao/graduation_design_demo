import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsMixedK
from losses import epipolar_simplified_loss, pose_loss
from model import PanoramaRelPoseModel
from pose_head import matrix_geodesic_distance


def _cfg_to_dict(cfg):
    try:
        return asdict(cfg)
    except Exception:
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _scheduler_lambda(
    total_updates: int,
    warmup_updates: int,
    min_lr_scale: float,
    hold_updates: int = 200,
    drop1_updates: int = 300,
    drop1_scale: float = 0.25,
    drop2_scale: float = 0.10,
):
    total_updates = max(int(total_updates), 1)
    warmup_updates = max(int(warmup_updates), 0)
    hold_updates = max(int(hold_updates), warmup_updates)
    drop1_updates = max(int(drop1_updates), hold_updates)
    drop1_scale = float(drop1_scale)
    drop2_scale = float(drop2_scale)

    def fn(step_idx: int):
        s = min(int(step_idx), total_updates)

        # Linear warmup to peak LR.
        if warmup_updates > 0 and s < warmup_updates:
            return max((s + 1) / max(warmup_updates, 1), 1e-6)

        # Short plateau around the region where earlier runs achieved the best tdir.
        if s < hold_updates:
            return 1.0

        # Hard decay after the likely best region to preserve a good checkpoint.
        if s < drop1_updates:
            return drop1_scale

        return drop2_scale

    return fn


def _save_ckpt(path, model, optimizer, scaler, scheduler, cfg, step, upd, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": _cfg_to_dict(cfg),
            "step": int(step),
            "upd": int(upd),
            "metrics": dict(metrics),
        },
        path,
    )


@torch.no_grad()
def eval_model(model, loader, device, cfg: Config):
    model.eval()

    rot_sum = 0.0
    tdir_sum = 0.0
    tdir_abs_sum = 0.0
    tdir_local_A_sum = 0.0
    tdir_local_A_abs_sum = 0.0
    n = 0
    n_local = 0

    acc = {
        "raw": 0.0,
        "flip": 0.0,
        "R@t": 0.0,
        "R@(-t)": 0.0,
        "Rt@t": 0.0,
        "Rt@(-t)": 0.0,
        "cnt": 0,
    }
    variant_desc = {
        "raw": "output frame B, baseline B->A",
        "flip": "output frame B, baseline A->B",
        "R@t": "compare to R*t_gt (B->A mapped to A-local)",
        "R@(-t)": "compare to -R*t_gt",
        "Rt@t": "compare to R^T*t_gt",
        "Rt@(-t)": "compare to -R^T*t_gt",
    }

    for bi, batch in enumerate(loader):
        if cfg.max_eval_batches and bi >= cfg.max_eval_batches:
            break

        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, aux = model(IA, IB)

        rot_rad = matrix_geodesic_distance(R_pred.float(), R_gt.float())
        rot_deg = rot_rad * (180.0 / math.pi)

        tp = F.normalize(t_pred.float(), dim=-1, eps=1e-6)
        tg = F.normalize(t_gt.float(), dim=-1, eps=1e-6)
        cos = torch.sum(tp * tg, dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos) * (180.0 / math.pi)
        ang_abs = torch.minimum(ang, 180.0 - ang)

        bsz = IA.shape[0]
        rot_sum += float(rot_deg.sum().cpu())
        tdir_sum += float(ang.sum().cpu())
        tdir_abs_sum += float(ang_abs.sum().cpu())
        n += bsz

        # Dual translation metrics:
        #   raw_B      : compare final output t_pred against gt in output B-frame
        #   local_A    : compare local translation head output against gt mapped into A-local
        tg_R = F.normalize(torch.matmul(R_gt.float(), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)
        tg_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)

        t_local_pred = aux.get("t_dir_local", None) if isinstance(aux, dict) else None
        t_local_frame = aux.get("t_local_frame", None) if isinstance(aux, dict) else None
        if t_local_pred is not None and t_local_frame == "A":
            tp_local = F.normalize(t_local_pred.float(), dim=-1, eps=1e-6)
            ang_local = torch.acos(torch.sum(tp_local * tg_R, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
            ang_local_abs = torch.minimum(ang_local, 180.0 - ang_local)
            tdir_local_A_sum += float(ang_local.sum().cpu())
            tdir_local_A_abs_sum += float(ang_local_abs.sum().cpu())
            n_local += bsz

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
    tdir_local_A = tdir_local_A_sum / max(n_local, 1) if n_local > 0 else float("nan")
    tdir_local_A_abs = tdir_local_A_abs_sum / max(n_local, 1) if n_local > 0 else float("nan")

    if acc["cnt"] > 0:
        denom = float(acc["cnt"])
        mean_map = {k: acc[k] / denom for k in ["raw", "flip", "R@t", "R@(-t)", "Rt@t", "Rt@(-t)"]}
        ranked = sorted(mean_map.items(), key=lambda kv: kv[1])
        best_key, best_val = ranked[0]
        gap2 = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else float("nan")
        msg = (
            f"[TDIR-CHK] best={best_key} ({variant_desc[best_key]})={best_val:.2f}° | gap2={gap2:.2f}° | "
            f"raw={mean_map['raw']:.2f} flip={mean_map['flip']:.2f} "
            f"R@t={mean_map['R@t']:.2f} R@(-t)={mean_map['R@(-t)']:.2f} "
            f"Rt@t={mean_map['Rt@t']:.2f} Rt@(-t)={mean_map['Rt@(-t)']:.2f} (n={int(denom)})"
        )
    else:
        mean_map = {}
        msg = "[TDIR-CHK] n=0"

    if n_local > 0:
        msg_local = (
            f"[TDIR-LOCAL] local_A={tdir_local_A:.2f}° | local_A_abs={tdir_local_A_abs:.2f}° | "
            f"gt_local_A=R*t_gt | n={n_local}"
        )
    else:
        msg_local = "[TDIR-LOCAL] unavailable"

    model.train()
    return rot, tdir, tdir_abs, tdir_local_A, tdir_local_A_abs, mean_map, msg, msg_local


def main():
    cfg = Config()
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

    print(f"[Cfg ] model_variant: coarse_interaction={cfg.use_coarse_interaction} | fine_stage={cfg.use_fine_stage} | epi_bias={cfg.use_epipolar_bias} | epi_loss={cfg.use_epipolar_loss}")
    print(f"[Cfg ] HxW={cfg.H}x{cfg.W} | D={cfg.D} | Nc={cfg.Nc} | Nf={cfg.Nf} | p={cfg.p}")
    print(f"[Cfg ] temp(coarse/fine)={cfg.coarse_temperature}/{cfg.fine_temperature} | logits_clip={cfg.logits_clip} | topk_coarse={cfg.topk_coarse}")
    print(f"[Cfg ] lr={cfg.lr} | wd={cfg.wd} | warmup_updates={cfg.warmup_updates} | min_lr_scale={cfg.min_lr_scale}")
    print(f"[Cfg ] piecewise_lr: hold<{getattr(cfg, 'lr_hold_updates', 200)} | drop1<{getattr(cfg, 'lr_drop1_updates', 300)}@{getattr(cfg, 'lr_drop1_scale', 0.25)} | drop2@{getattr(cfg, 'lr_drop2_scale', 0.10)}")
    print(f"[Cfg ] split_by={cfg.split_by} | train_ratio={cfg.train_ratio} | split_seed={cfg.split_seed} | data_seed={cfg.data_seed}")
    print("=" * 80)

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
        raise RuntimeError(f"train/test split leakage detected: {len(overlap)} overlapping groups, e.g. {sorted(list(overlap))[:8]}")

    for tag, ds in [("train", train_ds), ("test", test_ds)]:
        summary = getattr(ds, "split_summary", None)
        if summary is not None:
            preview = summary.get("selected_seq_keys", [])[:6]
            print(f"[Split] {tag} by {summary.get('split_by')} | groups={summary.get('num_selected_groups')} | seqs={summary.get('num_selected_seq')} | pairs={summary.get('num_pairs')} | preview={preview}")

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

    model = PanoramaRelPoseModel(cfg, device=dev).to(dev)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    total_updates = math.ceil(cfg.max_steps / max(cfg.grad_accum, 1))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_scheduler_lambda(
            total_updates=total_updates,
            warmup_updates=cfg.warmup_updates,
            min_lr_scale=cfg.min_lr_scale,
            hold_updates=getattr(cfg, "lr_hold_updates", 200),
            drop1_updates=getattr(cfg, "lr_drop1_updates", 300),
            drop1_scale=getattr(cfg, "lr_drop1_scale", 0.25),
            drop2_scale=getattr(cfg, "lr_drop2_scale", 0.10),
        ),
    )

    use_amp = bool(cfg.amp) and dev.type == "cuda"
    if cfg.amp_dtype.lower() == "auto":
        amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    elif cfg.amp_dtype.lower() in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ckpt_root = os.path.join(cfg.ckpt_dir, cfg.exp_name)
    os.makedirs(ckpt_root, exist_ok=True)

    step = 0
    upd = 0
    skip_updates = 0
    bad_forward = 0
    last_bad_grad_name = "-"
    last_grad_norm = float("nan")
    best_rot = float("inf")
    best_tdir_abs = float("inf")
    best_tdir_raw = float("inf")
    best_tdir_local_A = float("inf")
    best_tdir_local_A_abs = float("inf")

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

        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            R_pred, t_pred, aux = model(IA, IB)

            t_pose_pred = aux.get("t_dir_local", t_pred)
            t_pose_frame = aux.get("t_local_frame", "B") if aux.get("t_dir_local", None) is not None else "B"
            L_pose = pose_loss(
                R_pred,
                t_pose_pred,
                R_gt,
                t_gt,
                pose_t_alpha=cfg.pose_t_alpha,
                pred_t_frame=t_pose_frame,
            )

            W_ab = _first_not_none(aux.get("Wf_ab", None), aux.get("Wc_ab", None))
            W_ba = _first_not_none(aux.get("Wf_ba", None), aux.get("Wc_ba", None))
            cA = _first_not_none(aux.get("cA_f", None), aux.get("cA_c", None))
            cB = _first_not_none(aux.get("cB_f", None), aux.get("cB_c", None))
            bearingA = _first_not_none(aux.get("bearingA_f", None), aux.get("bearingA_c", None))
            bearingB = _first_not_none(aux.get("bearingB_f", None), aux.get("bearingB_c", None))

            if W_ab is not None and cfg.w_x > 0:
                P = W_ab.float().clamp_min(1e-9)
                L_x = (-P * P.log()).sum(dim=-1).mean()
            else:
                L_x = torch.zeros((), device=dev)

            if W_ab is not None and W_ba is not None and cfg.w_cyc > 0:
                cyc = torch.matmul(W_ab.float(), W_ba.float())
                I = torch.eye(cyc.shape[-1], device=dev).unsqueeze(0)
                L_cyc = F.mse_loss(cyc, I)
            else:
                L_cyc = torch.zeros((), device=dev)

            if cA is not None and cB is not None and cfg.w_rel > 0:
                L_rel = 0.5 * (F.softplus(-cA.float()).mean() + F.softplus(-cB.float()).mean())
            else:
                L_rel = torch.zeros((), device=dev)

            if cfg.use_epipolar_loss and W_ab is not None and bearingA is not None and bearingB is not None and cfg.w_epi > 0:
                L_epi = epipolar_simplified_loss(
                    W_ab=W_ab,
                    W_ba=W_ba,
                    bearing_a=bearingA,
                    bearing_b=bearingB,
                    R_gt=R_gt,
                    t_gt=t_gt,
                    allowed_mask=aux.get("allowed_mask", None),
                    angle_thresh_deg=cfg.epi_angle_thresh_deg,
                    use_bidir=cfg.epi_loss_use_bidir,
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
            L_scaled = L / float(cfg.grad_accum)

        t_fwd = time.perf_counter()

        forward_ok = all(
            bool(torch.isfinite(x).all())
            for x in [R_pred, t_pred, L_pose, L_x, L_cyc, L_rel, L_epi, L]
        )
        if not forward_ok:
            bad_forward += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                print(f"[Warn ] non-finite forward detected, batch skipped | bad_forward={bad_forward}")
            continue

        if use_scaler:
            scaler.scale(L_scaled).backward()
        else:
            L_scaled.backward()
        t_bwd = time.perf_counter()

        if (step + 1) % cfg.grad_accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)

            bad_name = None
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_name = name
                    break

            if bad_name is not None:
                skip_updates += 1
                last_bad_grad_name = bad_name
                last_grad_norm = float("nan")
                optimizer.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skip_updates += 1
                    last_bad_grad_name = "clip_grad_norm"
                    last_grad_norm = float("nan")
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                else:
                    last_grad_norm = float(grad_norm.detach().cpu())
                    last_bad_grad_name = "-"
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    upd += 1

                    if cfg.eval_every and upd % cfg.eval_every == 0:
                        t_eval0 = time.perf_counter()
                        rot, tdir, tdir_abs, tdir_local_A, tdir_local_A_abs, tdiag, tdir_msg, tdir_local_msg = eval_model(model, test_loader, dev, cfg)
                        t_eval = time.perf_counter() - t_eval0
                        metrics = {
                            "rot": rot,
                            "tdir": tdir,
                            "tdir_abs": tdir_abs,
                            "tdir_local_A": tdir_local_A,
                            "tdir_local_A_abs": tdir_local_A_abs,
                            **{f"tdir_diag_{k}": v for k, v in tdiag.items()},
                        }
                        print(
                            f"[Eval ] upd {upd:05d} (step {step:05d}) | rot={rot:.4f}° | "
                            f"tdir={tdir:.4f}° | tdir_abs={tdir_abs:.4f}° | "
                            f"tdir_local_A={tdir_local_A:.4f}° | tdir_local_A_abs={tdir_local_A_abs:.4f}° | "
                            f"time={t_eval:.2f}s"
                        )
                        print(tdir_msg)
                        print(tdir_local_msg)

                        _save_ckpt(os.path.join(ckpt_root, "last_eval.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if rot < best_rot:
                            best_rot = rot
                            _save_ckpt(os.path.join(ckpt_root, "best_rot.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir < best_tdir_raw:
                            best_tdir_raw = tdir
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_raw.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir_abs < best_tdir_abs:
                            best_tdir_abs = tdir_abs
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_abs.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if math.isfinite(tdir_local_A) and tdir_local_A < best_tdir_local_A:
                            best_tdir_local_A = tdir_local_A
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_local_A.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if math.isfinite(tdir_local_A_abs) and tdir_local_A_abs < best_tdir_local_A_abs:
                            best_tdir_local_A_abs = tdir_local_A_abs
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_local_A_abs.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)

        t_opt = time.perf_counter()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                p0_mean = float(next(model.parameters()).mean().detach().cpu())
            scaler_scale = float(scaler.get_scale()) if use_scaler else 1.0
            lr_now = optimizer.param_groups[0]["lr"]
            dt = time.perf_counter() - t0
            ep = step // max(len(train_loader), 1)
            print(
                f"[Train] step {step:05d} upd {upd:05d} ep{ep:03d} | "
                f"L={float(L.detach().cpu()):.3f} | pose={float(L_pose.detach().cpu()):.3f} | "
                f"x={float(L_x.detach().cpu()):.4f} | cyc={float(L_cyc.detach().cpu()):.4f} | "
                f"rel={float(L_rel.detach().cpu()):.4f} | epi={float(L_epi.detach().cpu()):.4f} | "
                f"lr={lr_now:.6e}"
            )
            print(
                f"[Time ] avg/iter={dt*1000.0:.1f}ms | data={(t_data-t0)*1000.0:.1f}ms "
                f"fwd={(t_fwd-t_data)*1000.0:.1f}ms bwd={(t_bwd-t_fwd)*1000.0:.1f}ms opt={(t_opt-t_bwd)*1000.0:.1f}ms"
            )
            print(
                f"[Stat ] grad_norm={last_grad_norm:.6f} | scaler_scale={scaler_scale} | "
                f"skip_updates={skip_updates} | bad_forward={bad_forward} | bad_grad={last_bad_grad_name} | p0_mean={p0_mean:.6e}"
            )

            if step == 0:
                for k in ["Wc_ab", "Wf_ab", "bearingA_c", "bearingA_f", "routing_mask", "allowed_mask"]:
                    if isinstance(aux, dict) and k in aux and aux[k] is not None:
                        print(f"[DBG] {k}: {tuple(aux[k].shape)}")
                print(f"[Shapes] IA {tuple(IA.shape)} | R {tuple(R_gt.shape)} | t {tuple(t_gt.shape)} | stage={aux.get('stage', '-')} | t_local={aux.get('t_local_frame', '-')} -> t_out={aux.get('t_output_frame', '-')}")
                print("-" * 80)

        step += 1

    _save_ckpt(
        os.path.join(ckpt_root, "last_train_state.pt"),
        model,
        optimizer,
        scaler,
        scheduler,
        cfg,
        step,
        upd,
        {
            "best_rot": best_rot,
            "best_tdir_raw": best_tdir_raw,
            "best_tdir_abs": best_tdir_abs,
            "best_tdir_local_A": best_tdir_local_A,
            "best_tdir_local_A_abs": best_tdir_local_A_abs,
        },
    )
    print(
        f"[Done ] training finished | best_rot={best_rot:.4f}° | "
        f"best_tdir_raw={best_tdir_raw:.4f}° | best_tdir_abs={best_tdir_abs:.4f}° | "
        f"best_tdir_local_A={best_tdir_local_A:.4f}° | best_tdir_local_A_abs={best_tdir_local_A_abs:.4f}° | "
        f"ckpt_dir={ckpt_root}"
    )


if __name__ == "__main__":
    main()
