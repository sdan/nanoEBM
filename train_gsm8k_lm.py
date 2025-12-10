"""
Train a plain GPT-style LM on GSM8K (no energy head, no refinement).

This is the LM-head baseline to compare against the EBM + refinement model
trained in train_gsm8k.py. The backbone architecture is identical (same
ModelConfig) but we optimize the tied LM head only.

Usage (example):
    uv run python train_gsm8k_lm.py \
        data.block_size=512 \
        data.batch_size=32 \
        model.n_layer=12 model.n_head=12 model.n_embd=768 \
        train.max_steps=20000 \
        wandb_project=nanoebm-gsm8k-lm
"""

import os
import datetime
import json

import chz
import torch

from nanoebm.config import Config
from nanoebm.transformer import Transformer
from nanoebm.data import get_gsm8k_loader
from nanoebm.utils import (
    Logger,
    save_checkpoint,
    load_checkpoint,
    timed,
    get_lr,
)


def main(cfg: Config):
    # Force dataset tag for bookkeeping
    cfg = chz.replace(cfg, data=chz.replace(cfg.data, dataset="gsm8k"))

    # Create a unique run directory
    base_out = cfg.out_dir
    os.makedirs(base_out, exist_ok=True)
    run_name = datetime.datetime.now().strftime("gsm8k_lm_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    logger = Logger(
        log_dir=run_dir,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name or run_name,
    )

    # Setup device / seed
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    logger.info(f"GSM8K LM setup: device={device}, seed={cfg.train.seed}")

    # Save config for reproducibility
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Load GSM8K dataset (train split)
    train_loader, train_ds = get_gsm8k_loader(
        split="train",
        block_size=cfg.data.block_size,
        batch_size=cfg.data.batch_size,
        encoding_name=cfg.data.bpe_encoding,
        cache_dir=cfg.data.hf_cache_dir,
    )

    vocab_size = train_ds.vocab_size
    model_cfg = chz.replace(cfg.model, vocab_size=vocab_size, block_size=cfg.data.block_size)
    logger.info(
        f"Loaded GSM8K LM data: split=train | vocab_size={vocab_size} "
        f"| block_size={cfg.data.block_size} | bpe={cfg.data.bpe_encoding}"
    )

    # Initialize Transformer LM (no energy head here)
    model = Transformer(model_cfg).to(device)
    if cfg.train.compile:
        model = torch.compile(model)

    # Optimizer (single param group)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )

    # Optional resume
    start_step = 0
    if cfg.load_checkpoint:
        ckpt_path = cfg.load_checkpoint
        try:
            metadata = load_checkpoint(ckpt_path, model, optimizer)
            start_step = metadata["step"]
            logger.info(f"Resumed LM from {ckpt_path} at step {start_step}")
            run_dir = os.path.dirname(ckpt_path)
            logger.info(f"Continuing LM run in {run_dir}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to resume LM from {ckpt_path}: {e}. Starting fresh.")

    # Training loop
    model.train()
    logger.info(f"Training GSM8K LM from step {start_step} to {cfg.train.max_steps}")

    print("\n" + "=" * 60)
    print("GSM8K LM Training Configuration")
    print("=" * 60)
    print(f"Steps:         {start_step:>6d} → {cfg.train.max_steps:>6d}")
    print(f"LR warmup:     {cfg.train.warmup_iters:>6d} steps")
    print(f"Learning rate: {cfg.train.learning_rate:>10.2e}")
    print(f"Batch size:    {cfg.data.batch_size:>6d}")
    print(f"Block size:    {cfg.data.block_size:>6d}")
    print(f"Vocab size:    {vocab_size:>6d}")
    print(f"BPE encoding:  {cfg.data.bpe_encoding}")
    print("=" * 60 + "\n")

    header_printed = False
    table_cols = [
        ("step", "step", "d", 6),
        ("loss", "loss", ".3f", 8),
        ("ppl", "perplexity", ".3f", 7),
        ("lr", "lr", ".2e", 11),
        ("t/fwd", "time/forward", ".3f", 8),
        ("t/bwd", "time/backward", ".3f", 8),
    ]

    def _fmt_cell(val, fmt, width):
        if val is None:
            s = ""
        else:
            try:
                if fmt == "d":
                    s = f"{int(val):d}"
                else:
                    s = f"{float(val):{fmt}}"
            except Exception:
                s = str(val)
        return s.rjust(width)

    def _print_row(row_metrics: dict):
        nonlocal header_printed
        if not header_printed:
            header = " ".join(lbl.rjust(w) for lbl, _, _, w in table_cols)
            sep = " ".join("-" * w for _, _, _, w in table_cols)
            print(header)
            print(sep)
            header_printed = True
        cells = []
        for _, key, fmt, w in table_cols:
            cells.append(_fmt_cell(row_metrics.get(key), fmt, w))
        print(" ".join(cells))

    for step, (x, y) in enumerate(train_loader, start=start_step):
        if step >= cfg.train.max_steps:
            break

        metrics: dict = {}

        # LR schedule
        lr = (
            get_lr(
                step,
                cfg.train.warmup_iters,
                cfg.train.lr_decay_iters,
                cfg.train.learning_rate,
                cfg.train.min_lr,
            )
            if cfg.train.decay_lr
            else cfg.train.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        metrics["lr"] = lr

        # Forward
        x, y = x.to(device), y.to(device)
        with timed("forward", metrics):
            logits, loss = model(x, targets=y)
            loss = loss / cfg.train.grad_accum_steps

        # Backward
        with timed("backward", metrics):
            loss.backward()

        # Optimizer step
        grad_step = (step + 1) % cfg.train.grad_accum_steps == 0
        if grad_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        metrics["loss"] = loss.item() * cfg.train.grad_accum_steps
        metrics["perplexity"] = float(torch.exp(loss * cfg.train.grad_accum_steps).item())

        if step % cfg.train.log_interval == 0:
            logger.log_metrics(metrics, step=step)
            row = {**metrics, "step": step}
            _print_row(row)

        # Checkpointing
        if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
            ckpt_path = save_checkpoint(
                model,
                optimizer,
                step,
                chz.replace(cfg, model=model_cfg),
                run_dir,
                prefix="ckpt_lm",
            )
            logger.info(f"Saved LM checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(run_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": cfg.train.max_steps,
            "config": chz.asdict(chz.replace(cfg, model=model_cfg)),
        },
        final_path,
    )
    logger.info(f"Saved final GSM8K LM model: {final_path}")

    logger.close()
    logger.info("GSM8K LM training complete")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)

