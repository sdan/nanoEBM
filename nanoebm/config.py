from __future__ import annotations

import os
import chz


@chz.chz
class ModelConfig:
    """Core model architecture configuration"""
    vocab_size: int = 50304  # GPT-2 vocab size (rounded to nearest multiple of 64)
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    # Energy-based thinking parameters
    energy_hidden: int = 4
    use_shared_embeddings: bool = True
    mcmc_num_steps: int = 0
    mcmc_step_size: float = 1.0
    mcmc_step_size_learnable: bool = False
    langevin_noise: float = 0.0


@chz.chz
class DataConfig:
    """Dataset configuration"""
    dataset: str = "shakespeare"  # shakespeare, wikitext
    data_path: str = "shakespeare.txt"
    block_size: int = 256
    batch_size: int = 64
    num_workers: int = 0


@chz.chz
class TrainConfig:
    """Training hyperparameters"""
    # Optimization
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5

    # Training loop
    max_steps: int = 5000
    grad_accum_steps: int = 1

    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 50

    # Logging
    log_interval: int = 10

    # System
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False


@chz.chz
class Config:
    """Main configuration for training nanoEBM models"""
    # Sub-configs
    model: ModelConfig = chz.field(default_factory=ModelConfig)
    data: DataConfig = chz.field(default_factory=DataConfig)
    train: TrainConfig = chz.field(default_factory=TrainConfig)

    # Output and logging
    out_dir: str = chz.field(
        default="out_ebt",
        munger=lambda _, s: os.path.expanduser(s)
    )
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing
    save_interval: int = 1000
    load_checkpoint: str | None = None
