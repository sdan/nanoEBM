"""Utility functions for logging, checkpointing, and metrics"""

from __future__ import annotations

import os
import json
import time
import glob
from typing import Any, Dict, Optional
from contextlib import contextmanager

import chz
import torch
import torch.nn as nn


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Handles both file logging and wandb logging"""

    def __init__(self, log_dir: str, wandb_project: str | None = None, config: Any = None, wandb_name: str | None = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "metrics.jsonl")
        self.wandb_run = None

        # Initialize wandb if project specified
        if wandb_project:
            try:
                import wandb
                # Ensure we pass a plain dict to wandb for configs
                cfg_for_wandb = None
                if config is not None:
                    try:
                        cfg_for_wandb = chz.asdict(config)
                    except Exception:
                        cfg_for_wandb = config.to_dict() if hasattr(config, 'to_dict') else config
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=cfg_for_wandb,
                    dir=log_dir,
                )
                self.info(f"Initialized wandb: {wandb_project}/{wandb_name}")
            except ImportError:
                self.warning("wandb not installed, skipping wandb logging")
            except Exception as e:
                self.warning(f"Failed to initialize wandb: {e}")

    def info(self, message: str):
        """Print info message with checkmark"""
        print(f"✓ {message}")

    def warning(self, message: str):
        """Print warning message"""
        print(f"⚠ {message}")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to both file and wandb (sanitizing JSON types)."""

        def _to_jsonable(obj: Any) -> Any:
            try:
                import numpy as _np  # type: ignore
            except Exception:  # pragma: no cover
                _np = None

            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            # Torch tensors -> Python scalars or lists
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.detach().cpu().tolist()
            # Numpy scalars/arrays
            if _np is not None:
                if isinstance(obj, _np.ndarray):
                    if obj.size == 1:
                        return obj.item()
                    return obj.tolist()
                if isinstance(obj, (_np.floating, _np.integer)):
                    return obj.item()
            # Torch dtype numbers
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            # Fallback to string
            return str(obj)

        # Add step and sanitize
        metrics_with_step = {"step": int(step), **metrics}
        metrics_json = _to_jsonable(metrics_with_step)

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics_json) + "\n")

        # Log to wandb (let wandb handle conversions, but keep it simple floats)
        if self.wandb_run:
            self.wandb_run.log({k: _to_jsonable(v) for k, v in metrics.items()}, step=step)

    def close(self):
        """Cleanup resources"""
        if self.wandb_run:
            self.wandb_run.finish()


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Any,
    save_dir: str,
    prefix: str = "ckpt",
    keep_last_n: int = 3,
) -> str:
    """Save checkpoint and optionally remove old ones"""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": chz.asdict(config),
    }

    ckpt_path = os.path.join(save_dir, f"{prefix}_step_{step}.pt")
    torch.save(checkpoint, ckpt_path)

    # Clean up old checkpoints
    if keep_last_n > 0:
        all_ckpts = sorted(glob.glob(os.path.join(save_dir, f"{prefix}_step_*.pt")))
        if len(all_ckpts) > keep_last_n:
            for old_ckpt in all_ckpts[:-keep_last_n]:
                os.remove(old_ckpt)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load checkpoint and return metadata"""
    # Prefer safe loading; fall back if torch version doesn't support it
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
    }


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "ckpt") -> Optional[str]:
    """
    Get the latest checkpoint path, searching recursively under `checkpoint_dir`.
    Prefers files matching `{prefix}_step_*.pt`. Falls back to any `final.pt` if present.
    """
    # Search recursively for step checkpoints
    step_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "**", f"{prefix}_step_*.pt"), recursive=True))
    if step_ckpts:
        return step_ckpts[-1]

    # Fallback: any final.pt files
    finals = sorted(glob.glob(os.path.join(checkpoint_dir, "**", "final.pt"), recursive=True))
    return finals[-1] if finals else None


# ============================================================================
# Timing utilities
# ============================================================================

@contextmanager
def timed(name: str, metrics: Dict[str, Any]):
    """Context manager to time a block of code and add to metrics dict"""
    start = time.time()
    yield
    elapsed = time.time() - start
    metrics[f"time/{name}"] = elapsed


# ============================================================================
# Learning rate scheduling
# ============================================================================

def get_lr(step: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup"""
    # Linear warmup
    if step < warmup_iters:
        return learning_rate * step / warmup_iters

    # Cosine decay after warmup
    if step > lr_decay_iters:
        return min_lr

    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    # Use math.cos to ensure a pure float return (avoid Tensor in logs)
    import math as _m
    coeff = 0.5 * (1.0 + _m.cos(decay_ratio * _m.pi))
    return float(min_lr + coeff * (learning_rate - min_lr))


# ============================================================================
# FLOPs estimation (analytical, ballpark)
# ============================================================================

def _linear_flops(tokens: int, d_in: int, d_out: int) -> int:
    """FLOPs for a dense matmul + bias. We use 2 * tokens * d_in * d_out
    (counting multiply + add as two FLOPs), bias cost ignored.
    """
    return 2 * tokens * d_in * d_out


def estimate_transformer_forward_flops(batch: int, seq_len: int, n_layer: int, n_head: int, d_model: int, mlp_ratio: float = 4.0) -> int:
    """Approximate forward FLOPs for a GPT-style transformer stack.
    Includes QKV projections, attention matmuls, output projection, and MLP.
    LayerNorm/softmax costs are omitted (smaller relative to matmuls).
    """
    tokens = batch * seq_len
    d_head = d_model // max(1, n_head)
    flops_per_layer = 0
    # Q,K,V projections
    flops_per_layer += 3 * _linear_flops(tokens, d_model, d_model)
    # Attention score: Q @ K^T and A @ V
    flops_per_layer += 2 * batch * n_head * seq_len * seq_len * d_head * 2  # two matmuls, *2 for mul+add
    # Output projection
    flops_per_layer += _linear_flops(tokens, d_model, d_model)
    # MLP (d -> 4d -> d)
    flops_per_layer += _linear_flops(tokens, d_model, int(mlp_ratio * d_model))
    flops_per_layer += _linear_flops(tokens, int(mlp_ratio * d_model), d_model)
    return flops_per_layer * n_layer


def estimate_energy_head_forward_flops(batch: int, time: int, vocab: int, d_model: int, d_energy: int | None = None, training: bool = True) -> int:
    """Forward FLOPs for the energy head: context projection, token projection (if training),
    and the (B,T,V,dE) einsum to compute energies.
    time is the number of active time positions (typically T-1).
    """
    dE = d_model if d_energy is None else int(d_energy)
    tokens = batch * time
    flops = 0
    # context_proj: (B,T,d) -> (B,T,dE)
    flops += _linear_flops(tokens, d_model, dE)
    # token_proj: (V,d) -> (V,dE) (training path computes it fresh)
    if training:
        flops += _linear_flops(vocab, d_model, dE)
    # energies: (B,T,dE) x (V,dE) => (B,T,V)
    flops += 2 * batch * time * vocab * dE
    return flops


def estimate_step_flops(batch: int, seq_len: int, vocab: int, n_layer: int, n_head: int, d_model: int, d_energy: int | None, refine_steps: int, refine_last_only: bool, training: bool = True) -> dict:
    """Ballpark FLOPs per training step.
    Returns forward_gflops and train_gflops (heuristic: ~3x forward),
    including an approximate inner-loop refinement cost ~ B*T_work*V*refine_steps.
    """
    # Backbone forward across T-1 positions
    f_gpt = estimate_transformer_forward_flops(batch, seq_len - 1, n_layer, n_head, d_model)
    # Energy head forward
    f_energy = estimate_energy_head_forward_flops(batch, seq_len - 1, vocab, d_model, d_energy, training=training)
    forward = f_gpt + f_energy

    # Inner loop cost (very approximate): softmax + expectation + simple grads w.r.t logits
    T_work = 1 if (training and refine_last_only) else (seq_len - 1)
    inner = 6 * batch * T_work * vocab * max(0, refine_steps)  # scaled constant captures softmax+grad cost
    forward += inner

    # Heuristic: training backward is ~2x forward for matmul-dominant nets
    train_total = forward * 3
    g = 1e-9
    return {
        "forward_gflops": forward * g,
        "train_gflops": train_total * g,
        "inner_loop_gflops": inner * g,
    }
