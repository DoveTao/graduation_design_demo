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
    min_dt: float = 0.1
    max_dt: float = 5.0
    k_choices: Tuple[int, ...] = (5, 10, 20, 40)
    k_probs: Tuple[float, ...] = (0.25, 0.30, 0.30, 0.15)
    train_color_aug: bool = True
    train_color_aug_strength: float = 0.75

    # Deterministic evaluation protocol
    eval_use_fixed_pairs: bool = True
    eval_k_list: Tuple[int, ...] = (5, 10, 20, 40)
    eval_pair_step: int = 1
    eval_min_dt: float = 0.1
    eval_max_dt: float = 5.0
    eval_dt_bucket_edges: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0)
    stable_eval_min_dt: float = 0.5
    save_eval_bucket_history: bool = False

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
    use_epipolar_bias: bool = False
    use_epipolar_loss: bool = True
    epi_mode: str = "bias"   # "bias" | "mask"

    coarse_temperature: float = 0.10
    fine_temperature: float = 0.07
    logits_clip: float = 20.0
    topk_coarse: int = 64

    epi_angle_thresh_deg: float = 10.0
    epi_bias_strength: float = 2.0
    epi_loss_use_bidir: bool = True
    # GT-pose epipolar matching supervision: directly supervises W_ab/W_ba.
    epi_loss_type: str = "gt_band_nll"
    epi_gt_target_temp: float = 0.25
    epi_ramp_updates: int = 100
    use_geometric_t_fusion: bool = False
    geometric_t_fuse_strength: float = 0.0

    # ---------------- Auxiliary depth branch ----------------
    use_depth_branch: bool = False
    depth_feat_dim: int = 32
    depth_pred_scales: Tuple[int, ...] = (16, 32)
    depth_fuse_scale: int = 16
    depth_loss_scale: int = 16
    depth_inv_min: float = 0.0125      # 1 / 80 m
    depth_inv_max: float = 10.0        # 1 / 0.1 m
    depth_min: float = 0.10
    depth_max: float = 80.0
    depth_fuse_to_translation_only: bool = False
    depth_use_detached_pose: bool = True
    depth_warmup_updates: int = 100
    depth_photo_ramp_updates: int = 100
    depth_fuse_start_updates: int = 250
    depth_use_automask: bool = True
    depth_fuse_strength: float = 0.25
    depth_fuse_detach_feature: bool = True

    # ---------------- Optimization ----------------
    batch_size: int = 2
    grad_accum: int = 2
    lr: float = 1.0e-4
    wd: float = 0.01
    max_grad_norm: float = 1.0

    # Piecewise LR schedule:
    # warmup -> short hold -> sharp drop -> low tail
    warmup_updates: int = 100
    min_lr_scale: float = 0.03  # kept for compatibility; piecewise schedule is used

    lr_hold_updates: int = 1000
    lr_drop1_updates: int = 3000
    lr_drop1_scale: float = 0.30
    lr_drop2_scale: float = 0.10

    # ---------------- Loss weights ----------------
    w_pose: float = 1.0
    w_x: float = 0.0
    w_cyc: float = 0.0
    w_rel: float = 0.0
    w_epi: float = 0.05
    w_coarse_pose_aux: float = 0.30
    w_coarse_epi_aux: float = 0.02
    pose_t_alpha: float = 1.0
    pose_t_oriented_weight: float = 1.0
    pose_t_axis_weight: float = 0.0
    small_dt_thresh: float = 0.3
    small_dt_t_weight: float = 0.20
    w_photo: float = 0.0
    w_smooth: float = 0.0

    # ---------------- Schedule / logging ----------------
    max_steps: int = 5000
    log_every: int = 50
    eval_every: int = 300       # in update steps
    max_eval_batches: int = 128
    max_train_eval_batches: int = 64
    ckpt_dir: str = "checkpoints"
    exp_name: str = "C10_color_aug_stablemetric"

    # Export helpers for PPT figures / tables
    save_eval_history: bool = True
    save_eval_buckets_latest: bool = True
    save_final_summary: bool = True
    save_vis_examples: bool = False
    save_vis_payload_npz: bool = False
    save_vis_diag_json: bool = True
    save_last_eval_checkpoint: bool = False
    save_last_train_state: bool = False
    save_metric_checkpoints: bool = False
    save_best_joint_checkpoint: bool = True
    vis_eval_index: int = 0
    vis_dump_every_eval: bool = False
    plot_curves_after_train: bool = False
    vis_plot_max_side: int = 64

    # Joint checkpoint selection:
    # only checkpoints passing these thresholds participate in best_joint.
    joint_rot_thresh_deg: float = 10.0
    joint_tdir_thresh_deg: float = 55.0
    joint_rot_weight: float = 2.0

    # ---------------- Dataloader ----------------
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # ---------------- AMP / perf ----------------
    amp: bool = True
    amp_dtype: str = "bf16"   # "auto" | "bf16" | "fp16"
    tf32: bool = True
    matmul_precision: str = "high"
    deterministic: bool = False
    benchmark: bool = True
