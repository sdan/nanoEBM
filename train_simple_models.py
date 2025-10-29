"""
Train simple GPT vs EBM models for clear comparison.

CLI usage examples:
  - Train GPT for 5000 steps:
      python train_simple_models.py --model gpt --max_steps 5000
  - Train EBM for 5000 steps:
      python train_simple_models.py --model ebm --max_steps 5000
  - Train both (defaults):
      python train_simple_models.py --model both --max_steps 2000
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
from nanoebm.model import EBM
from nanoebm.data import get_loader
from nanoebm.config import Config, ModelConfig, DataConfig, TrainConfig
import chz
import argparse

def create_small_config():
    """Create a smaller model config for faster training."""
    model_cfg = ModelConfig(
        vocab_size=67,  # Shakespeare vocab size
        block_size=128,  # Smaller context
        n_layer=4,      # Fewer layers
        n_head=4,       # Fewer heads
        n_embd=256,     # Smaller embedding
        dropout=0.1,
        bias=True,
        # EBM parameters
        refine_steps=4,
        alpha_value=0.1,  # Larger step size for more visible refinement
        langevin_noise=0.01,
        energy_convergence_threshold=1e-4,
        warmup_steps_no_refine=100
    )

    data_cfg = DataConfig(
        dataset="shakespeare",
        data_path="shakespeare.txt",
        block_size=128,
        batch_size=32,
        num_workers=0
    )

    train_cfg = TrainConfig(
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip=1.0,
        decay_lr=True,
        warmup_iters=100,
        lr_decay_iters=2000,
        min_lr=3e-5,
        max_steps=1000,  # Quick training
        grad_accum_steps=1,
        eval_interval=100,
        eval_iters=20,
        log_interval=10,
        seed=1337,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        dtype="float32",
        compile=False
    )

    return Config(model=model_cfg, data=data_cfg, train=train_cfg)

def train_model(model_type="gpt", cfg=None):
    """Train a model (either GPT or EBM)."""

    if cfg is None:
        cfg = create_small_config()

    device = cfg.train.device
    print(f"\n🚀 Training {model_type.upper()} on {device}")
    print("="*60)

    # Load data
    train_loader, train_ds = get_loader(
        cfg.data.data_path,
        cfg.data.block_size,
        cfg.data.batch_size,
        "train"
    )

    val_loader, val_ds = get_loader(
        cfg.data.data_path,
        cfg.data.block_size,
        cfg.data.batch_size,
        "val"
    )

    # Ensure vocab_size matches dataset (avoid decoding mismatches later)
    ds_vocab = len(train_ds.stoi)
    if ds_vocab != cfg.model.vocab_size:
        print(f"[warn] Overriding cfg.model.vocab_size {cfg.model.vocab_size} -> {ds_vocab} to match dataset")
        cfg.model.vocab_size = ds_vocab

    # Initialize model
    model = EBM(cfg.model).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params/1e6:.2f}M parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay
    )

    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for step in range(cfg.train.max_steps):
        model.train()

        # Get batch
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        # Forward pass
        if model_type == "gpt":
            # Pure Transformer - no refinement
            loss, _, _ = model(x, targets=y, use_refine=False)
        else:
            # EBM - use refinement after warmup
            use_refine = step >= cfg.model.warmup_steps_no_refine
            loss, _, metrics = model(x, targets=y, use_refine=use_refine, refine_steps=cfg.model.refine_steps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        # Logging
        if step % cfg.train.log_interval == 0:
            if model_type == "ebm" and step >= cfg.model.warmup_steps_no_refine:
                energy_gap = metrics.get('energy_gap', 0) if 'metrics' in locals() else 0
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | PPL: {np.exp(loss.item()):.2f} | Energy Gap: {energy_gap:.4f}")
            else:
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | PPL: {np.exp(loss.item()):.2f}")

        # Evaluation
        if step % cfg.train.eval_interval == 0:
            model.eval()
            val_loss = 0
            val_loss_refined = 0
            n_val = 0

            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_loader):
                    if i >= cfg.train.eval_iters:
                        break

                    x_val, y_val = x_val.to(device), y_val.to(device)

                    # Always evaluate both modes
                    loss_base, _, _ = model(x_val, targets=y_val, use_refine=False)
                    val_loss += loss_base.item()

                    if model_type == "ebm":
                        with torch.enable_grad():
                            loss_refined, _, _ = model(x_val, targets=y_val, use_refine=True, refine_steps=cfg.model.refine_steps)
                            loss_refined = loss_refined.detach()
                        val_loss_refined += loss_refined.item()

                    n_val += 1

            val_loss /= n_val
            val_losses.append(val_loss)

            print(f"\n📊 Validation at step {step}:")
            print(f"  Base loss: {val_loss:.4f} (PPL: {np.exp(val_loss):.2f})")

            if model_type == "ebm":
                val_loss_refined /= n_val
                improvement = (val_loss - val_loss_refined) / val_loss * 100
                print(f"  Refined loss: {val_loss_refined:.4f} (PPL: {np.exp(val_loss_refined):.2f})")
                print(f"  Improvement: {improvement:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': {
                        'model': cfg.model.__dict__,
                        'data': cfg.data.__dict__,
                        'train': cfg.train.__dict__
                    },
                    # Save minimal vocab mapping info for consistent decoding
                    'meta': {
                        'vocab_size': len(train_ds.stoi),
                        'data_path': cfg.data.data_path
                    },
                    'val_loss': val_loss,
                    'model_type': model_type
                }
                torch.save(checkpoint, f"simple_{model_type}_model.pt")
                print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")

            print()

    print(f"\n✅ Training complete for {model_type.upper()}")
    print(f"Best validation loss: {best_val_loss:.4f} (PPL: {np.exp(best_val_loss):.2f})")

    return model, cfg

def compare_trained_models():
    """Compare the trained GPT and EBM models."""

    print("\n" + "="*60)
    print("COMPARING TRAINED MODELS")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    # Load both models
    gpt_checkpoint = torch.load("simple_gpt_model.pt", map_location=device, weights_only=False)
    ebm_checkpoint = torch.load("simple_ebm_model.pt", map_location=device, weights_only=False)

    # Create configs
    gpt_cfg = create_small_config()
    ebm_cfg = create_small_config()

    # Load models
    gpt_model = EBM(gpt_cfg.model).to(device)
    gpt_model.load_state_dict(gpt_checkpoint['model'])
    gpt_model.eval()

    ebm_model = EBM(ebm_cfg.model).to(device)
    ebm_model.load_state_dict(ebm_checkpoint['model'])
    ebm_model.eval()

    # Load validation data
    val_loader, _ = get_loader(
        "shakespeare.txt",
        gpt_cfg.data.block_size,
        32,
        "val"
    )

    # Test both models
    results = {
        'gpt': {'losses': [], 'correct': 0, 'total': 0},
        'ebm_base': {'losses': [], 'correct': 0, 'total': 0},
        'ebm_refined': {'losses': [], 'correct': 0, 'total': 0}
    }

    print("\nEvaluating models...")

    for i, (x, y) in enumerate(val_loader):
        if i >= 20:  # Test on 20 batches
            break

        x, y = x.to(device), y.to(device)

        # GPT (trained without refinement)
        with torch.no_grad():
            loss_gpt, logits_gpt, _ = gpt_model(x, targets=y, use_refine=False)
            preds_gpt = torch.argmax(logits_gpt, dim=-1)
            correct_gpt = (preds_gpt == y).sum().item()

        results['gpt']['losses'].append(loss_gpt.item())
        results['gpt']['correct'] += correct_gpt
        results['gpt']['total'] += y.numel()

        # EBM without refinement (base)
        with torch.no_grad():
            loss_ebm_base, logits_ebm_base, _ = ebm_model(x, targets=y, use_refine=False)
            preds_ebm_base = torch.argmax(logits_ebm_base, dim=-1)
            correct_ebm_base = (preds_ebm_base == y).sum().item()

        results['ebm_base']['losses'].append(loss_ebm_base.item())
        results['ebm_base']['correct'] += correct_ebm_base
        results['ebm_base']['total'] += y.numel()

        # EBM with refinement
        with torch.enable_grad():
            loss_ebm_refined, logits_ebm_refined, _ = ebm_model(x, targets=y, use_refine=True, refine_steps=4)
            loss_ebm_refined = loss_ebm_refined.detach()
            logits_ebm_refined = logits_ebm_refined.detach()
            preds_ebm_refined = torch.argmax(logits_ebm_refined, dim=-1)
            correct_ebm_refined = (preds_ebm_refined == y).sum().item()

        results['ebm_refined']['losses'].append(loss_ebm_refined.item())
        results['ebm_refined']['correct'] += correct_ebm_refined
        results['ebm_refined']['total'] += y.numel()

    # Print results
    print("\n📊 RESULTS:")
    print("-" * 40)
    print("Model               Loss    PPL     Acc")
    print("-" * 40)

    for name, label in [('gpt', 'GPT (baseline)'),
                        ('ebm_base', 'EBM (no refine)'),
                        ('ebm_refined', 'EBM (refined)')]:
        loss = np.mean(results[name]['losses'])
        ppl = np.exp(loss)
        acc = results[name]['correct'] / results[name]['total']
        print(f"{label:18} {loss:.3f}  {ppl:.2f}  {acc:.3f}")

    # Calculate improvements
    gpt_ppl = np.exp(np.mean(results['gpt']['losses']))
    ebm_refined_ppl = np.exp(np.mean(results['ebm_refined']['losses']))
    improvement = (gpt_ppl - ebm_refined_ppl) / gpt_ppl * 100

    print("\n" + "="*60)
    print("VERDICT:")
    if improvement > 0:
        print(f"✅ EBM with refinement is {improvement:.1f}% better than GPT!")
    else:
        print(f"❌ EBM refinement didn't help (worse by {-improvement:.1f}%)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Train simple GPT vs EBM models")
    parser.add_argument("--model", choices=["gpt", "ebm", "both"], default="both",
                        help="Which model to train")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Number of training steps")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda|mps|cpu)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--block_size", type=int, default=None,
                        help="Override context length")
    args = parser.parse_args()

    print("\n🎓 TRAINING SIMPLE MODELS FOR COMPARISON")
    print("This will train small GPT and/or EBM models on Shakespeare\n")

    def cfg_with_overrides():
        cfg = create_small_config()
        t = cfg.train
        d = cfg.data
        m = cfg.model
        # chz configs are frozen; use chz.replace to override
        if args.max_steps is not None:
            t = chz.replace(t, max_steps=int(args.max_steps))
        if args.device:
            t = chz.replace(t, device=args.device)
        if args.batch_size:
            d = chz.replace(d, batch_size=int(args.batch_size))
        if args.block_size:
            bs = int(args.block_size)
            d = chz.replace(d, block_size=bs)
            m = chz.replace(m, block_size=bs)
        cfg = chz.replace(cfg, train=t, data=d, model=m)
        return cfg

    if args.model in ("gpt", "both"):
        print("1. Training GPT baseline (no refinement)...")
        _gpt_model, _gpt_cfg = train_model("gpt", cfg_with_overrides())

    if args.model in ("ebm", "both"):
        print("\n2. Training EBM (with refinement)...")
        _ebm_model, _ebm_cfg = train_model("ebm", cfg_with_overrides())

    if args.model == "both":
        compare_trained_models()

if __name__ == "__main__":
    main()
