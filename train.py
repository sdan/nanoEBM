"""
train.py - Simple training script for nanoEBM

Usage:
    python train.py
    python train.py --train.learning_rate=1e-3 --train.max_steps=5000
    python train.py --model.n_layer=12 --data.batch_size=128
    python train.py --wandb_project=nanoebm
"""

import os
import json
import chz
import torch
from nanoebm.config import Config
from nanoebm.model import EBTLanguageModel
from nanoebm.data import get_loader
from nanoebm.utils import Logger, save_checkpoint, get_latest_checkpoint, load_checkpoint, timed, get_lr


def main(cfg: Config):
    # Setup
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)

    # Logging
    logger = Logger(
        log_dir=cfg.out_dir,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    # Save config using chz.asdict()
    config_path = os.path.join(cfg.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    print(f"✓ Saved config to {config_path}")

    # Data
    train_loader, train_ds = get_loader(
        cfg.data.data_path,
        cfg.data.block_size,
        cfg.data.batch_size,
        "train"
    )
    vocab_size = len(train_ds.stoi)
    print(f"✓ Loaded dataset: {cfg.data.dataset} | vocab_size={vocab_size}")

    # Update model config with actual vocab size using chz.replace()
    model_cfg = chz.replace(cfg.model, vocab_size=vocab_size)

    # Model
    model = EBTLanguageModel(model_cfg, model_cfg).to(device)
    if cfg.train.compile:
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )

    # Resume from checkpoint if exists
    start_step = 0
    if cfg.load_checkpoint:
        ckpt_path = cfg.load_checkpoint
        metadata = load_checkpoint(ckpt_path, model, optimizer)
        start_step = metadata["step"]
        print(f"✓ Resumed from {ckpt_path} at step {start_step}")
    else:
        latest_ckpt = get_latest_checkpoint(cfg.out_dir)
        if latest_ckpt:
            metadata = load_checkpoint(latest_ckpt, model, optimizer)
            start_step = metadata["step"]
            print(f"✓ Resumed from {latest_ckpt} at step {start_step}")

    # Training loop
    model.train()
    print(f"✓ Training from step {start_step} to {cfg.train.max_steps}")

    for step, (x, y) in enumerate(train_loader, start=start_step):
        if step >= cfg.train.max_steps:
            break

        metrics = {}

        # Learning rate schedule
        lr = get_lr(
            step,
            cfg.train.warmup_iters,
            cfg.train.lr_decay_iters,
            cfg.train.learning_rate,
            cfg.train.min_lr,
        ) if cfg.train.decay_lr else cfg.train.learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        metrics["lr"] = lr

        # Forward pass
        x, y = x.to(device), y.to(device)
        with timed("forward", metrics):
            loss, logits = model(x, targets=y)
            loss = loss / cfg.train.grad_accum_steps

        # Backward pass
        with timed("backward", metrics):
            loss.backward()

        # Optimizer step (every grad_accum_steps)
        grad_step = (step + 1) % cfg.train.grad_accum_steps == 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip * grad_step)
        optimizer.step() if grad_step else None
        optimizer.zero_grad(set_to_none=True) if grad_step else None

        # Logging
        metrics["loss"] = loss.item() * cfg.train.grad_accum_steps

        if step % cfg.train.log_interval == 0:
            logger.log_metrics(metrics, step=step)
            print(f"step {step:5d} | loss {metrics['loss']:.3f} | lr {metrics['lr']:.2e}")

        # Checkpointing
        if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, cfg, cfg.out_dir)
            print(f"✓ Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(cfg.out_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": cfg.train.max_steps,
        "config": chz.asdict(cfg),
    }, final_path)
    print(f"✓ Saved final model: {final_path}")

    logger.close()
    print("✓ Training complete")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
