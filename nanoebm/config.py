from __future__ import annotations

import os
import chz


@chz.chz
class ModelConfig:
    """Core model architecture configuration"""
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    # EBM thinking parameters (single refinement path by default)
    # Closed-form, stop-grad mixture refinement (no autograd.grad in inner loop)

    # Core scalars
    softmax_temperature: float = 1.0  # τ: p = softmax(v / τ)
    entropy_weight: float = 0.0       # λ: J = E_p[E] - λ H(p)

    # Iterative refinement (gradient descent on logits)
    think_steps: int = 0              # number of refinement steps (0 = no refinement)
    think_lr: float = 1.0             # step size for refinement (learnable scalar in model)
    think_lr_learnable: bool = False  # whether the step size parameter is learnable
    think_lr_lr_multiplier: float = 3.0  # optimizer LR multiplier for the step size parameter

    # Stability/behavior
    detach_refine: bool = True        # detach per-step by default (stop-grad through steps)
    truncate_refine: bool = False     # backprop only through the final step
    refine_last_position_only: bool = True  # train-time simplification
    think_max_move: float = 0.25      # per-step delta clamp in logit space (0 disables)
    absolute_clamp: float = 6.0       # keep logits bounded (0 disables)
    soften_target_prob_dist: float = 0.0  # label smoothing over steps
    langevin_noise: float = 0.0       # optional noise (usually 0)
    think_init: str = "base_energy"   # base_energy|zeros|random
    think_init_noise_std: float = 0.0 # optional tiny init noise on v0
    mixture_stopgrad: bool = True     # stop-grad mixture by default; set False for coupled

    # Hidden toggle for future coupled refinement (gradients flow through mixture)
    coupled_refine: bool = False

    # Loss combine: (1-λ_aux) * CE(v_T) + λ_aux * CE(-E_base)
    aux_ce_weight: float = 0.1

    # Optional logging of expected energy trace across steps (mean over active positions)
    log_expected_energy_trace: bool = False

    # Back-compat aliases (deprecated): keep for loading older checkpoints/configs
    # They are not referenced in new code paths but may appear in saved configs.
    refine_steps: int | None = None
    refine_step_size: float | None = None
    refine_step_size_learnable: bool | None = None
    refine_step_size_lr_multiplier: float | None = None
    clamp_update_max_change: float | None = None
    entropy_reg_tau: float | None = None
    denoising_initial_condition: str | None = None


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
