"""
train a nanoEBM model

Usage:
    python train.py
    python train.py train.learning_rate=1e-3 train.max_steps=5000
    python train.py model.n_layer=12 data.batch_size=128
    python train.py wandb_project=nanoebm
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
    # Logging
    logger = Logger(
        log_dir=cfg.out_dir,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    
    # Setup based in the config
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    logger.info(f"Setup: device={device}, seed={cfg.train.seed}")

    # Save config to json for future reproducibility
    config_path = os.path.join(cfg.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Load the dataset
    train_loader, train_ds = get_loader(
        cfg.data.data_path,
        cfg.data.block_size,
        cfg.data.batch_size,
        "train"
    )
    
    # Because we're not using BPE for character-level model we need to set the vocab size to the actual size of the dataset
    vocab_size = len(train_ds.stoi)
    # update the model config with the actual vocab size using chz.replace()
    model_cfg = chz.replace(cfg.model, vocab_size=vocab_size)
    logger.info(f"Loaded dataset: {cfg.data.dataset} | vocab_size={vocab_size}")

    # Initialize the model
    model = EBTLanguageModel(model_cfg, model_cfg).to(device)
    if cfg.train.compile:
        model = torch.compile(model)

    # Initialize the optimizer
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
        logger.info(f"Resumed from {ckpt_path} at step {start_step}")
    else:
        latest_ckpt = get_latest_checkpoint(cfg.out_dir)
        if latest_ckpt:
            metadata = load_checkpoint(latest_ckpt, model, optimizer)
            start_step = metadata["step"]
            logger.info(f"Resumed from {latest_ckpt} at step {start_step}")

    # Training loop
    model.train()
    logger.info(f"Training from step {start_step} to {cfg.train.max_steps}")

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
            logger.info(f"Step {step:5d} | Loss {metrics['loss']:.3f} | Learning Rate {metrics['lr']:.2e}")

        # Checkpointing
        if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
            # Save checkpoints with the effective model config (incl. actual vocab_size)
            ckpt_path = save_checkpoint(
                model, optimizer, step, chz.replace(cfg, model=model_cfg), cfg.out_dir
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(cfg.out_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": cfg.train.max_steps,
        # Persist the effective model config (with actual vocab_size)
        "config": chz.asdict(chz.replace(cfg, model=model_cfg)),
    }, final_path)
    logger.info(f"Saved final model: {final_path}")

    logger.close()
    logger.info("Training complete")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)
