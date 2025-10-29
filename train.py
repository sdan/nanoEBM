"""
train nanoEBM

Usage:
    python train.py
    python train.py train.learning_rate=1e-3 train.max_steps=5000
    python train.py model.n_layer=12 data.batch_size=128
    python train.py wandb_project=nanoebm
    # train with step size learning (refinement)
    python train.py model.refine_steps=2 model.truncate_refine=True model.detach_refine=True model.refine_step_size_learnable=True
"""

import os
import datetime
import json
import chz
import torch
from nanoebm.config import Config
from nanoebm.model import EBTLanguageModel
from nanoebm.data import get_loader
from nanoebm.utils import (
    Logger,
    save_checkpoint,
    get_latest_checkpoint,
    load_checkpoint,
    timed,
    get_lr,
)


def main(cfg: Config):
    # Create a unique run directory to avoid overwrites
    base_out = cfg.out_dir
    os.makedirs(base_out, exist_ok=True)
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    logger = Logger(
        log_dir=run_dir,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name or run_name,
    )
    
    # Setup based in the config
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    logger.info(f"Setup: device={device}, seed={cfg.train.seed}")

    # Save config to json for future reproducibility
    config_path = os.path.join(run_dir, "config.json")
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

    # Initialize the optimizer with separate learning rates for alpha (refine step size)
    base_lr = cfg.train.learning_rate
    alpha_lr_multiplier = getattr(model_cfg, 'refine_step_size_lr_multiplier', 3.0)
    
    # Separate parameters into alpha and non-alpha groups
    alpha_params = []
    other_params = []
    for name, param in model.named_parameters():
        if name == 'alpha':
            alpha_params.append(param)
        else:
            other_params.append(param)
    
    # Build parameter groups with different learning rates
    param_groups = []
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'betas': (cfg.train.beta1, cfg.train.beta2),
            'weight_decay': cfg.train.weight_decay,
        })
    
    if alpha_params:
        param_groups.append({
            'params': alpha_params,
            'lr': base_lr * alpha_lr_multiplier,
            'betas': (cfg.train.beta1, cfg.train.beta2),
            'weight_decay': 0.0,  # No weight decay for alpha
        })
    optimizer = torch.optim.AdamW(param_groups)

    # Resume from checkpoint if exists
    start_step = 0
    # if cfg.load_checkpoint:
    #     ckpt_path = cfg.load_checkpoint
    #     metadata = load_checkpoint(ckpt_path, model, optimizer)
    #     start_step = metadata["step"]
    #     logger.info(f"Resumed from {ckpt_path} at step {start_step}")
    # else:
    # Try to find the latest checkpoint recursively under base_out
    from nanoebm.utils import get_latest_checkpoint
    # Prefer latest in any previous run dir; if none, start fresh in this run_dir
    latest_ckpt = get_latest_checkpoint(base_out)  # may return from nested run dirs
    if latest_ckpt:
        metadata = load_checkpoint(latest_ckpt, model, optimizer)
        start_step = metadata["step"]
        logger.info(f"Resumed from {latest_ckpt} at step {start_step}")
        # Continue writing into the same directory as the checkpoint
        run_dir = os.path.dirname(latest_ckpt)
        logger.info(f"Continuing run in {run_dir}")

    # Training loop
    model.train()
    logger.info(f"Training from step {start_step} to {cfg.train.max_steps}")

    # Pretty, columnar console logging
    header_printed = False
    # (label, metrics_key, fmt, width)
    table_cols = [
        ("step", "step", "d", 6),
        ("loss", "loss", ".3f", 8),
        ("ppl", "perplexity", ".3f", 7),
        ("lr", "lr", ".2e", 11),
        ("alpha", "alpha", ".3f", 8),
        ("Egap", "energy_gap", ".4f", 10),
        ("E0", "initial_energy", ".4f", 10),
        ("EK", "final_energy", ".4f", 10),
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
        # header once
        if not header_printed:
            header = " ".join(lbl.rjust(w) for lbl, _, _, w in table_cols)
            sep = " ".join("-" * w for _, _, _, w in table_cols)
            print(header)
            print(sep)
            header_printed = True
        # row
        cells = []
        for _, key, fmt, w in table_cols:
            cells.append(_fmt_cell(row_metrics.get(key), fmt, w))
        print(" ".join(cells))

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

        # Respect LR multiplier for alpha group
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:  # main params
                param_group["lr"] = lr
            else:  # alpha group (if present)
                param_group["lr"] = lr * alpha_lr_multiplier
        metrics["lr"] = lr

        # Forward pass
        x, y = x.to(device), y.to(device)
        with timed("forward", metrics):
            out = model(x, targets=y)
            if isinstance(out, tuple) and len(out) == 3:
                loss, logits, extras = out
            else:
                loss, logits = out
                extras = {}
            loss = loss / cfg.train.grad_accum_steps

        # Backward pass
        with timed("backward", metrics):
            loss.backward()

        # Optimizer step (every grad_accum_steps)
        grad_step = (step + 1) % cfg.train.grad_accum_steps == 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip * grad_step)
        optimizer.step() if grad_step else None
        optimizer.zero_grad(set_to_none=True) if grad_step else None

        # Logging (metrics dict persisted to JSONL/W&B)
        metrics["loss"] = loss.item() * cfg.train.grad_accum_steps
        # Optional metrics from forward extras
        for k in ("perplexity", "energy_gap", "initial_energy", "final_energy"):
            if k in extras:
                metrics[k] = extras[k]
        # Track current alpha (step size)
        try:
            metrics["alpha"] = float(model.alpha.detach().clamp(min=1e-6).item())
        except Exception:
            pass

        if step % cfg.train.log_interval == 0:
            # Persist metrics
            logger.log_metrics(metrics, step=step)
            # Pretty table row on console
            row = {**metrics, "step": step}
            _print_row(row)

        # Checkpointing
        if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
            # Save checkpoints with the effective model config (incl. actual vocab_size)
            ckpt_path = save_checkpoint(
                model, optimizer, step, chz.replace(cfg, model=model_cfg), run_dir
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(run_dir, "final.pt")
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
