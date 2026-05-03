"""
File: config.py
Description:
    Central configuration dataclass for the graduation design MVP experiments.
    It defines data paths, model size, matching behavior, losses, optimization,
    logging, evaluation, and checkpoint settings.

Main Components:
    - Dataset split, sampling, and evaluation protocol settings
    - Model architecture and coarse/fine matching controls
    - Epipolar, pose, depth, and auxiliary loss weights
    - Optimizer, learning-rate schedule, dataloader, and AMP settings

Usage / Role:
    Provides the default experiment configuration consumed by training,
    evaluation, datasets, and model construction.

Notes:
    Defaults are set for the current coarse-only MVP with a lightweight
    translation feature branch, optional odometry-scale prediction, frequent
    evaluation, and limited-GPU training.
"""

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
    train_color_aug: bool = False
    train_color_aug_strength: float = 0.25

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
    patch_embed_use_coords: bool = False
    patch_embed_pool_mode: str = "avg"  # "avg" | "avgmax" | "gated_avgmax"
    patch_embed_pool_gate_init: float = -2.0
    patch_embed_avgmax_pool: bool = False
    use_bearing_fuse: bool = False
    use_cross_context: bool = False
    cross_context_layers: int = 1
    cross_context_strength: float = 0.25
    use_translation_feature_branch: bool = True
    translation_patch_pool_mode: str = "gated_avgmax"
    translation_patch_pool_gate_init: float = -2.0
    translation_pos_enc_scale: float = 1.0
    translation_branch_encoder_layers: int = 0
    translation_branch_detach_match: bool = True

    # Translation convention
    translation_local_frame: str = "A"
    translation_output_frame: str = "B"

    # ---------------- Model: interaction / routing / geometry ----------------
    use_coarse_interaction: bool = True
    use_fine_stage: bool = False
    use_epipolar_bias: bool = False
    use_epipolar_loss: bool = True
    epi_mode: str = "bias"   # "bias" | "mask"
    fine_pose_fuse_strength: float = 0.0

    coarse_temperature: float = 0.10
    fine_temperature: float = 0.07
    logits_clip: float = 20.0
    topk_coarse: int = 64
    pose_use_stats_pool: bool = True

    epi_angle_thresh_deg: float = 10.0
    epi_bias_strength: float = 2.0
    epi_loss_use_bidir: bool = True
    # GT-pose epipolar matching supervision: directly supervises W_ab/W_ba.
    epi_loss_type: str = "gt_match_ce"
    epi_gt_target_temp: float = 0.25
    epi_ramp_updates: int = 100
    use_geometric_t_fusion: bool = False
    geometric_t_fuse_strength: float = 0.0
    use_geometry_refine: bool = False
    geom_refine_min_prob: float = 0.01
    geom_refine_max_matches: int = 512
    geom_refine_mutual_check: bool = False
    geom_refine_use_fine_if_available: bool = True
    geom_refine_fallback_to_network: bool = True

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

    lr_hold_updates: int = 100
    lr_drop1_updates: int = 300
    lr_drop1_scale: float = 0.30
    lr_drop2_scale: float = 0.10

    # ---------------- Loss weights ----------------
    w_pose: float = 1.0
    w_x: float = 0.0
    w_cyc: float = 0.0
    w_rel: float = 0.0
    w_epi: float = 0.01
    w_coarse_pose_aux: float = 0.30
    w_coarse_epi_aux: float = 0.0
    w_pose_output_t: float = 0.0
    use_translation_magnitude_head: bool = True
    w_tmag: float = 0.10
    w_t_mag: float = 0.0  # compatibility alias; training prefers w_tmag
    tmag_loss_type: str = "log_smooth_l1"
    tmag_start_updates: int = 0
    tmag_ramp_updates: int = 0
    tmag_min: float = 1.0e-3
    tmag_pred_source: str = "translation_branch"  # "translation_branch" | "pose_feat"
    tmag_detach_features: bool = False
    use_tmag_global_bias: bool = False
    tmag_global_bias_init: float = 0.0
    use_tmag_affine_calib: bool = False
    tmag_affine_init_scale: float = 1.0
    tmag_affine_init_bias: float = 0.0
    log_tmag_clamp_min: float = -6.0
    log_tmag_clamp_max: float = 6.0
    use_tmag_scale_reg: bool = False
    w_tmag_scale: float = 0.0
    tmag_scale_reg_min_gt: float = 1e-4
    tmag_scale_reg_log: bool = True
    tmag_scale_reg_min_dt: float = 0.0
    use_tmag_under_reg: bool = False
    w_tmag_under: float = 0.0
    tmag_under_target_ratio: float = 0.35
    tmag_under_min_gt: float = 0.02
    tmag_under_min_dt: float = 0.02
    # T51: dt-conditioned magnitude head
    tmag_condition_on_dt: bool = False
    tmag_dt_clamp_min: float = 0.01
    use_tdir_anchor_loss: bool = False
    tdir_anchor_checkpoint: str = ""
    w_tdir_anchor: float = 0.0
    tdir_anchor_min_dt: float = 0.2
    tdir_anchor_min_k: int = 0
    tdir_anchor_start_updates: int = 0
    tdir_anchor_ramp_updates: int = 0
    pose_t_alpha: float = 1.0
    pose_t_oriented_weight: float = 1.0
    pose_t_axis_weight: float = 0.0
    large_k_rot_thresh: int = 40
    large_k_rot_weight: float = 1.0
    small_dt_thresh: float = 0.3
    small_dt_t_weight: float = 0.20
    tdir_loss_ignore_dt_below: float = 0.0
    tdir_loss_ignore_weight: float = 0.0
    tdir_loss_dt_ramp_enable: bool = False
    tdir_loss_dt_ramp_start: float = 0.02
    tdir_loss_dt_ramp_end: float = 0.10
    tdir_loss_dt_ramp_start_weight: float = 0.05
    tdir_loss_dt_ramp_end_weight: float = -1.0  # <0 means use small_dt_t_weight
    use_seq_turn_loss: bool = False
    seq_turn_loss_w: float = 0.006
    seq_turn_only_k: int = 1
    seq_turn_min_dt: float = 0.05
    seq_turn_max_dt: float = 0.20
    seq_turn_start_updates: int = 100
    seq_turn_ramp_updates: int = 200
    seq_turn_acos_eps: float = 1.0e-6
    seq_turn_loss_clamp_deg: float = 0.0
    use_seq_turn_chain_loss: bool = False
    seq_turn_chain_loss_w: float = 0.003
    seq_turn_chain_min_pairs: int = 1
    seq_turn_chain_start_updates: int = 150
    seq_turn_chain_ramp_updates: int = 250
    use_odom_chain_len_loss: bool = False
    odom_chain_len_loss_w: float = 0.0
    odom_chain_len_target_ratio: float = 0.55
    odom_chain_len_min_gt: float = 0.02
    odom_chain_len_only_k: int = 1
    odom_chain_len_start_updates: int = 50
    odom_chain_len_ramp_updates: int = 150
    use_odom_chain_vec_loss: bool = False
    odom_chain_vec_loss_w: float = 0.0
    odom_chain_vec_only_k: int = 1
    odom_chain_vec_min_gt: float = 0.02
    odom_chain_vec_start_updates: int = 50
    odom_chain_vec_ramp_updates: int = 150
    w_photo: float = 0.0
    w_smooth: float = 0.0

    # ---------------- Schedule / logging ----------------
    max_steps: int = 1200
    log_every: int = 50
    eval_every: int = 100       # in update steps
    max_eval_batches: int = 128
    max_train_eval_batches: int = 64
    ckpt_dir: str = "checkpoints"
    exp_name: str = "C11_tbranch_gated_L0"
    eval_only: bool = False
    init_checkpoint: str = ""
    strict_load_checkpoint: bool = False

    # Export helpers for PPT figures / tables
    save_eval_history: bool = True
    save_eval_buckets_latest: bool = True
    save_final_summary: bool = True
    use_odometry_eval: bool = True
    odom_eval_prefer_k: int = 1
    odom_eval_fallback_to_min_k: bool = True
    odom_eval_max_pairs: int = 0
    odom_eval_smooth_tmag_window: int = 0
    odom_eval_scale_fit: bool = False
    odom_eval_dtcalib: bool = False
    odom_eval_dtcalib_min_count: int = 1
    save_odom_trajectory_debug: bool = False
    odom_trajectory_debug_max_chains: int = 1
    odom_trajectory_debug_segment_count: int = 4
    odom_trajectory_debug_topk_steps: int = 10
    save_odom_metrics_latest: bool = True
    save_vis_examples: bool = False
    save_vis_payload_npz: bool = False
    save_vis_diag_json: bool = True
    save_last_eval_checkpoint: bool = False
    save_last_train_state: bool = False
    save_metric_checkpoints: bool = False
    save_best_joint_checkpoint: bool = True
    save_best_local_joint_checkpoint: bool = True
    save_best_odom_checkpoint: bool = True
    odom_select_metric: str = "odom_metric_drift"
    odom_select_max_tdir_abs: float = 25.0
    odom_select_max_tmag_rel: float = 0.90
    odom_select_require_status_ok: bool = True
    odom_select_window: int = 1
    odom_select_min_points: int = 1
    save_best_smallk_odom_checkpoint: bool = True
    smallk_select_metric: str = "odom_metric_drift"
    smallk_select_k_list: Tuple[int, ...] = (1, 2, 3)
    smallk_select_max_tdir_abs: float = 28.0
    smallk_select_max_tmag_rel: float = 0.95
    smallk_select_require_status_ok: bool = True
    smallk_select_window: int = 1
    smallk_select_min_points: int = 1
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
    amp: bool = False
    amp_dtype: str = "bf16"   # "auto" | "bf16" | "fp16"
    tf32: bool = True
    matmul_precision: str = "high"
    deterministic: bool = False
    benchmark: bool = True
    cuda_flash_sdp: bool = True
    cuda_mem_efficient_sdp: bool = True
    cuda_math_sdp: bool = True
