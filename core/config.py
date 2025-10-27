from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict


@dataclass
class EBTConfig:
    # Model / backbone
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # Energy head / interface
    energy_hidden: int = 4  # width multiplier for small MLP if used
    use_shared_embeddings: bool = True  # share Wte with candidate embedder

    # Thinking / refinement hyperparams (training-time defaults)
    mcmc_num_steps: int = 2
    mcmc_step_size: float = 1.0
    mcmc_step_size_learnable: bool = True
    langevin_noise: float = 0.0


@dataclass
class TrainerConfig:
    out_dir: str = "out_ebt"
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"  # 'float32'|'bfloat16'|'float16'
    compile: bool = True
    # Optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # LR schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5
    # Eval
    eval_interval: int = 1000
    eval_iters: int = 50
    log_interval: int = 10


@dataclass
class DataConfig:
    dataset: str = "tiny_shakespeare_char"  # suggest small vocab for exact normalization
    data_dir: str = "data"


@dataclass
class ThinkingConfig:
    steps: int = 8
    step_size: float = 1.0
    temperature: float = 0.8
    tau_entropy: float = 0.2
    init: str = "from_energy"  # 'uniform'|'from_energy'


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_config_dicts(
    model: EBTConfig,
    trainer: TrainerConfig,
    data: DataConfig,
    thinking: ThinkingConfig,
) -> Dict[str, Any]:
    return {
        "model": asdict(model),
        "trainer": asdict(trainer),
        "data": asdict(data),
        "thinking": asdict(thinking),
    }

