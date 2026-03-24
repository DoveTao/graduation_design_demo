from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ---------------- Data ----------------
    data_root: str = "data"
    H: int = 1024
    W: int = 2048

    # Group-aware dataset split.
    # scene_seq: split by complete (scene, seq) folders, no pair leakage across train/test.
    # scene    : split by complete scenes (stricter, but train/test may become imbalanced if scenes are few).
    split_by: str = "scene_seq"
    train_ratio: float = 0.8
    split_seed: int = 3407
    data_seed: int = 1234

    # strict dt sampling for MixedK
    min_dt: float = 0.06
    max_dt: float = 5.0
    k_choices: Tuple[int, ...] = (10, 20)
    k_probs: Tuple[float, ...] = (0.84, 0.16)

    # ---------------- Model ----------------
    D: int = 256
    Nc: int = 192
    Nf: int = 768
    topk_coarse: int = 16

    # Patch/token encoder hyper-params used by model.py
    p: int = 16
    in_ch: int = 3
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Epipolar band (used inside interaction.py as a bias/mask)
    epi_angle_thresh_deg: float = 30.0
    epi_bias_strength: float = 10.0

    # ---------------- Optimization ----------------
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-4
    wd: float = 0.01
    max_grad_norm: float = 1.0

    # Loss weights
    w_pose: float = 0.75
    w_x: float = 1.0
    w_cyc: float = 1.0
    w_rel: float = 0.1
    w_epi: float = 0.1

    # pose loss hyperparam
    pose_t_alpha: float = 1.0

    # ---------------- Schedule / logging ----------------
    max_steps: int = 5000
    log_every: int = 50
    eval_every: int = 200  # in update steps (after grad_accum)
    max_eval_batches: int = 100

    # ---------------- Dataloader ----------------
    num_workers: int = 4
    pin_memory: bool = True

    # ---------------- AMP / perf ----------------
    amp: bool = False
    # "auto" => bf16 if supported else fp16; "bf16" => bf16; "fp16" => fp16
    amp_dtype: str = "auto"

    # Torch perf knobs
    tf32: bool = True
    matmul_precision: str = "high"  # "highest"|"high"|"medium"
    deterministic: bool = False
    benchmark: bool = True
