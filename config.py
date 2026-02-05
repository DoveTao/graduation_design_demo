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
    # Use relative path: <project_root>/data
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
    # k sampling strategy
    # =========================
    # Train: mixed-k sampling (recommended)
    use_mixed_k: bool = True
    k_choices: List[int] = field(default_factory=lambda: [5, 10])
    k_probs: List[float] = field(default_factory=lambda: [0.2, 0.8])

    # Optional: filter too-small baseline pairs (meters). 0 disables.
    min_dt: float = 0.05
    max_tries: int = 10

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
    # Epipolar-guided band (placeholder)
    # =========================
    epi_angle_thresh_deg: float = 30.0
    epi_bias_strength: float = 10.0

    # =========================
    # Loss weights
    # =========================
    lam_x: float = 1.0
    lam_c: float = 0.5
    lam_r: float = 0.2
    lam_e: float = 0.2

    # =========================
    # Train / Optim
    # =========================
    batch_size: int = 1                 # RTX 3060 6GB + 1024x2048 建议 1
    grad_accum: int = 4                # 等效大 batch：例如 4
    lr: float = 5e-5   # or 1e-5 if still unstable
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 1234

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    # Loop
    max_steps: int = 6000
    log_every: int = 50

    # Eval
    eval_every: int = 400
    max_eval_batches: int = 100

    deterministic: bool = False
