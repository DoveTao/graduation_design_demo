# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent  # config.py 在项目根目录时成立


@dataclass
class Config:
    # =========================
    # Paths
    # =========================
    data_root: str = str(PROJECT_ROOT / "data")

    # =========================
    # Dataset split (only in scene01)
    # =========================
    train_scenes: Optional[List[str]] = field(default_factory=lambda: ["scene01"])
    train_seqs: Optional[List[str]] = field(default_factory=lambda: ["seq01", "seq02"])

    test_scenes: Optional[List[str]] = field(default_factory=lambda: ["scene01"])
    test_seqs: Optional[List[str]] = field(default_factory=lambda: ["seq03"])

    # ERP panorama resolution
    H: int = 1024
    W: int = 2048
    in_ch: int = 3

    # Pair building for fixed-k dataset
    pair_step: int = 1

    # =========================
    # k sampling strategy (align train with test, still help tdir)
    # =========================
    use_mixed_k: bool = True

    # 训练主要对齐 test_k_stride=10（避免分布偏移导致 tdir@k=10 变差）
    # 同时少量引入更长基线，帮助网络学习更明确的平移方向
    k_choices: List[int] = field(default_factory=lambda: [10, 20])
    k_probs: List[float] = field(default_factory=lambda: [0.84, 0.16])

    # 过滤太短基线，但不要高到大量拒绝 k=10（否则等价于训练/测试不匹配）
    min_dt: float = 0.06
    max_tries: int = 10
    # Optional: filter too-large baseline pairs (meters). None disables.
    max_dt: float | None = 5.0

    # Test: fixed k for comparable metrics
    test_k_stride: int = 10

    # =========================
    # Token / Transformer
    # =========================
    D: int = 256
    p: int = 8
    Nc: int = 192
    Nf: int = 768
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # =========================
    # Matching / routing
    # =========================
    topk_coarse: int = 16
    topk_fine: int = 32

    # =========================
    # Epipolar-guided band (keep looser early for stability)
    # =========================
    epi_angle_thresh_deg: float = 30.0
    epi_bias_strength: float = 10.0

    # =========================
    # Loss weights
    # =========================
    lam_x: float = 1.0
    lam_c: float = 0.5

    # reliability 正则不要太大（否则会压过 pose 监督）
    lam_r: float = 0.05

    # epipolar 项适度（不要太大导致早期粗位姿误导 fine）
    lam_e: float = 0.2

    # =========================
    # Train / Optim
    # =========================
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 1234

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    # Loop
    max_steps: int = 20000
    log_every: int = 50

    # Eval
    eval_every: int = 200
    max_eval_batches: int = 100

    # Speed / reproducibility
    deterministic: bool = False
