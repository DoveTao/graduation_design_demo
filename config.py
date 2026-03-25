from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ---------------- Data ----------------
    data_root: str = "data"
    H: int = 1024
    W: int = 2048

    # Group-aware dataset split
    split_by: str = "scene_seq"   # "scene_seq" | "scene"
    train_ratio: float = 0.8
    split_seed: int = 3407
    data_seed: int = 1234

    # strict dt sampling for MixedK
    min_dt: float = 0.06
    max_dt: float = 5.0
    k_choices: Tuple[int, ...] = (10, 20)
    k_probs: Tuple[float, ...] = (0.84, 0.16)

    # ---------------- Model: spherical tokenization / encoder ----------------
    D: int = 256
    Nc: int = 192
    Nf: int = 768
    p: int = 16
    in_ch: int = 3
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.05

    # Translation convention
    translation_local_frame: str = "A"
    translation_output_frame: str = "B"

    # ---------------- Model: interaction / routing / geometry ----------------
    use_coarse_interaction: bool = True
    use_fine_stage: bool = True
    use_epipolar_bias: bool = True
    use_epipolar_loss: bool = True
    epi_mode: str = "bias"   # "bias" | "mask"

    coarse_temperature: float = 0.10
    fine_temperature: float = 0.07
    logits_clip: float = 20.0
    topk_coarse: int = 16

    epi_angle_thresh_deg: float = 30.0
    epi_bias_strength: float = 8.0
    epi_loss_use_bidir: bool = True

    # ---------------- Optimization ----------------
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 5.0e-5
    wd: float = 0.01
    max_grad_norm: float = 1.0

    # Piecewise LR schedule:
    # warmup -> short hold -> sharp drop -> low tail
    warmup_updates: int = 40
    min_lr_scale: float = 0.03  # kept for compatibility; piecewise schedule is used

    lr_hold_updates: int = 75
    lr_drop1_updates: int = 150
    lr_drop1_scale: float = 0.20
    lr_drop2_scale: float = 0.05

    # ---------------- Loss weights ----------------
    w_pose: float = 1.0
    w_x: float = 0.10
    w_cyc: float = 0.05
    w_rel: float = 0.02
    w_epi: float = 0.02
    pose_t_alpha: float = 1.0

    # ---------------- Schedule / logging ----------------
    max_steps: int = 1000
    log_every: int = 50
    eval_every: int = 25       # in update steps
    max_eval_batches: int = 100
    ckpt_dir: str = "checkpoints"
    exp_name: str = "week4_week5_impl_piecewise_joint"

    # Joint checkpoint selection:
    # only checkpoints passing these thresholds participate in best_joint.
    joint_rot_thresh_deg: float = 6.0
    joint_tdir_thresh_deg: float = 40.0
    joint_rot_weight: float = 2.0

    # ---------------- Dataloader ----------------
    num_workers: int = 4
    pin_memory: bool = True

    # ---------------- AMP / perf ----------------
    amp: bool = False
    amp_dtype: str = "auto"   # "auto" | "bf16" | "fp16"
    tf32: bool = True
    matmul_precision: str = "high"
    deterministic: bool = False
    benchmark: bool = True
