"""
Train nanoEBM on GSM8K math word problems using BPE.

This mirrors train.py but swaps in:
- GSM8K HF dataset
- tiktoken BPE (gpt2 by default)

Usage (example):
    uv run python train_gsm8k.py \
        data.block_size=512 \
        data.batch_size=32 \
        model.n_layer=12 model.n_head=12 model.n_embd=768 \
        train.max_steps=20000 \
        wandb_project=nanoebm-gsm8k
"""

import os
import datetime
import json

import chz
import torch

from nanoebm.config import Config
from nanoebm.model import EBM
from nanoebm.data import get_gsm8k_loader
from nanoebm.contrastive import create_contrastive_loss
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
    run_name = datetime.datetime.now().strftime("gsm8k_%Y%m%d_%H%M%S")
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
    logger.info(f"GSM8K setup: device={device}, seed={cfg.train.seed}")

    # Save config for reproducibility
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Load GSM8K dataset (train split only here; test is used in eval_gsm8k.py)
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
        f"Loaded GSM8K: split=train | vocab_size={vocab_size} "
        f"| block_size={cfg.data.block_size} | bpe={cfg.data.bpe_encoding}"
    )

    # Initialize model
    model = EBM(model_cfg).to(device)
    if cfg.train.compile:
        model = torch.compile(model)

    # Contrastive loss (optional)
    contrastive_loss_fn = create_contrastive_loss(model, model_cfg)
    if contrastive_loss_fn is not None:
        logger.info(
            f"Contrastive enabled: type={model_cfg.contrastive_type}, "
            f"k={model_cfg.contrastive_k}, weight={model_cfg.contrastive_weight}"
        )

    # Optimizer with separate LR for alpha
    base_lr = cfg.train.learning_rate
    alpha_lr_multiplier = getattr(model_cfg, "alpha_lr_multiplier", 3.0)

    alpha_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name == "alpha":
            alpha_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": base_lr,
                "betas": (cfg.train.beta1, cfg.train.beta2),
                "weight_decay": cfg.train.weight_decay,
            }
        )
    if alpha_params:
        param_groups.append(
            {
                "params": alpha_params,
                "lr": base_lr * alpha_lr_multiplier,
                "betas": (cfg.train.beta1, cfg.train.beta2),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(param_groups)

    # Optional resume
    start_step = 0
    if cfg.load_checkpoint:
        ckpt_path = cfg.load_checkpoint
        try:
            metadata = load_checkpoint(ckpt_path, model, optimizer)
            start_step = metadata["step"]
            logger.info(f"Resumed from {ckpt_path} at step {start_step}")
            run_dir = os.path.dirname(ckpt_path)
            logger.info(f"Continuing run in {run_dir}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to resume from {ckpt_path}: {e}. Starting fresh.")

    # Training loop
    model.train()
    logger.info(f"Training GSM8K from step {start_step} to {cfg.train.max_steps}")

    # Print key config to console
    print("\n" + "=" * 60)
    print("GSM8K Training Configuration")
    print("=" * 60)
    print(f"Steps:           {start_step:>6d} → {cfg.train.max_steps:>6d}")
    print(f"LR warmup:       {cfg.train.warmup_iters:>6d} steps")
    print(f"System 2 warmup: {cfg.model.warmup_steps_no_refine:>6d} steps (System 1 only)")
    print(f"Learning rate:   {cfg.train.learning_rate:>10.2e} (base)")
    print(f"Alpha (step):    {cfg.model.alpha_value:>10.3f} (fixed)")
    print(f"Refine steps:    {cfg.model.refine_steps:>6d}")
    print(f"Batch size:      {cfg.data.batch_size:>6d}")
    print(f"Block size:      {cfg.data.block_size:>6d}")
    print(f"Vocab size:      {vocab_size:>6d}")
    print(f"BPE encoding:    {cfg.data.bpe_encoding}")
    if model_cfg.use_contrastive:
        print(
            f"Contrastive:     {model_cfg.contrastive_type:>6s} "
            f"(k={model_cfg.contrastive_k}, weight={model_cfg.contrastive_weight:.2f})"
        )
    print("=" * 60 + "\n")

    # Pretty console logging (same layout as train.py)
    header_printed = False
    table_cols = [
        ("step", "step", "d", 6),
        ("loss", "loss", ".3f", 8),
        ("ppl", "perplexity", ".3f", 7),
        ("lr", "lr", ".2e", 11),
        ("alpha", "alpha", ".3f", 8),
        ("Egap", "energy_gap", ".4f", 10),
        ("E0", "initial_energy", ".4f", 10),
        ("EK", "final_energy", ".4f", 10),
    ]
    if model_cfg.use_contrastive:
        table_cols.append(("CD", "cd_loss", ".4f", 8))
    table_cols.extend(
        [
            ("t/fwd", "time/forward", ".3f", 8),
            ("t/bwd", "time/backward", ".3f", 8),
        ]
    )

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

        # Learning rate schedule
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
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                param_group["lr"] = lr
            else:
                param_group["lr"] = lr * alpha_lr_multiplier
        metrics["lr"] = lr

        # Forward
        x, y = x.to(device), y.to(device)
        with timed("forward", metrics):
            use_refine = step >= cfg.model.warmup_steps_no_refine
            if contrastive_loss_fn is not None:
                loss, logits, extras = model.forward_with_contrastive(
                    x,
                    targets=y,
                    use_refine=use_refine,
                    refine_steps=model_cfg.refine_steps,
                    contrastive_loss_fn=contrastive_loss_fn,
                )
            else:
                loss, logits, extras = model(
                    x,
                    targets=y,
                    use_refine=use_refine,
                    refine_steps=model_cfg.refine_steps,
                )
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
        for k in (
            "perplexity",
            "energy_gap",
            "initial_energy",
            "final_energy",
            "cd_loss",
            "nll_loss",
            "total_loss",
        ):
            if k in extras:
                metrics[k] = extras[k]
        metrics["alpha"] = float(model.alpha.item())

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
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

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
    logger.info(f"Saved final GSM8K model: {final_path}")

    logger.close()
    logger.info("GSM8K training complete")


if __name__ == "__main__":
    config = chz.entrypoint(Config)
    main(config)

