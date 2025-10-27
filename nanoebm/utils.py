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
                print(f"✓ Initialized wandb: {wandb_project}/{wandb_name}")
            except ImportError:
                print("⚠ wandb not installed, skipping wandb logging")
            except Exception as e:
                print(f"⚠ Failed to initialize wandb: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to both file and wandb"""
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics_with_step) + "\n")

        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
    }


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "ckpt") -> Optional[str]:
    """Get the latest checkpoint path"""
    ckpt_pattern = os.path.join(checkpoint_dir, f"{prefix}_step_*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    return checkpoints[-1] if checkpoints else None


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
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * torch.pi)))
    return min_lr + coeff * (learning_rate - min_lr)
