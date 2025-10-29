"""
Debug refinement: let's see if refinement actually changes anything
"""

import torch
import torch.nn.functional as F
import numpy as np
from nanoebm.model import EBM
from nanoebm.data import get_loader
from nanoebm.config import Config, ModelConfig, DataConfig, TrainConfig
import chz

def load_model(checkpoint_path):
    """Load the trained model."""
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']

    # Create sub-configs from checkpoint data
    model_cfg = ModelConfig(**config_dict.get('model', {}))
    data_cfg = DataConfig(**config_dict.get('data', {}))
    train_cfg = TrainConfig(**config_dict.get('train', {}))

    # Create full config with the loaded sub-configs
    cfg = Config(model=model_cfg, data=data_cfg, train=train_cfg)

    model = EBM(cfg.model).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, cfg, device

def debug_refinement():
    """Debug what happens during refinement."""

    # Load model
    checkpoint_path = "/Users/sdan/Developer/nanoebm/out_ebt/refine4.pt"
    model, cfg, device = load_model(checkpoint_path)
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"✓ Using device: {device}")

    # Load validation data
    val_loader, _ = get_loader(
        "shakespeare.txt",
        cfg.model.block_size,
        batch_size=2,  # Very small batch for detailed analysis
        split="val"
    )
    print(f"✓ Loaded Shakespeare validation set\n")

    # Get a batch
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)

    print("="*60)
    print("DEBUGGING REFINEMENT")
    print("="*60)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # 1. System 1 (no refinement)
    print("\n1. SYSTEM 1 (Transformer - no refinement):")
    print("-" * 40)
    with torch.no_grad():
        loss_s1, logits_s1, metrics_s1 = model(x, targets=y, use_refine=False)

    print(f"  Loss: {loss_s1.item():.4f}")
    print(f"  Logits shape: {logits_s1.shape}")
    print(f"  Logits mean: {logits_s1.mean().item():.4f}")
    print(f"  Logits std: {logits_s1.std().item():.4f}")
    if metrics_s1:
        print(f"  Metrics: {metrics_s1}")

    # 2. System 2 (with refinement)
    print("\n2. SYSTEM 2 (EBM with refinement):")
    print("-" * 40)

    # Let's trace through refinement step by step
    print("\nDetailed refinement analysis:")

    # Get initial logits
    with torch.no_grad():
        _, initial_logits, _ = model(x, use_refine=False)

    print(f"  Initial logits mean: {initial_logits.mean().item():.4f}")
    print(f"  Initial logits std: {initial_logits.std().item():.4f}")

    # Now apply refinement manually
    print("\n  Applying refinement steps:")

    logits = initial_logits.clone().detach().requires_grad_(True)

    for step in range(4):
        # Compute energy (negative log likelihood)
        energies = -F.log_softmax(logits, dim=-1)

        # Current probability distribution
        probs = F.softmax(logits, dim=-1)

        # Expected energy under current distribution
        expected_energy = (probs * energies).sum(dim=-1).mean()

        # Compute gradient
        grad = torch.autograd.grad(
            expected_energy,
            logits,
            create_graph=False
        )[0]

        print(f"    Step {step+1}:")
        print(f"      Energy: {expected_energy.item():.4f}")
        print(f"      Grad norm: {grad.norm().item():.4f}")
        print(f"      Grad mean: {grad.mean().item():.6f}")
        print(f"      Grad std: {grad.std().item():.6f}")

        # Apply gradient descent
        step_size = cfg.model.alpha_value if hasattr(cfg.model, 'alpha_value') else 0.02
        logits = logits - step_size * grad.clamp(-5, 5)
        logits = logits.detach().requires_grad_(True)

        # Check if logits changed
        change = (logits - initial_logits).abs().mean().item()
        print(f"      Logits change from initial: {change:.6f}")

    print(f"\n  Final logits mean: {logits.mean().item():.4f}")
    print(f"  Final logits std: {logits.std().item():.4f}")

    # Compare with model's refinement
    print("\n3. MODEL'S BUILT-IN REFINEMENT:")
    print("-" * 40)
    with torch.enable_grad():
        loss_s2, logits_s2, metrics_s2 = model(x, targets=y, use_refine=True, refine_steps=4)
        loss_s2 = loss_s2.detach()
        logits_s2 = logits_s2.detach()

    print(f"  Loss: {loss_s2.item():.4f}")
    print(f"  Logits mean: {logits_s2.mean().item():.4f}")
    print(f"  Logits std: {logits_s2.std().item():.4f}")
    if metrics_s2:
        print(f"  Metrics: {metrics_s2}")

    # Check differences
    print("\n4. COMPARISON:")
    print("-" * 40)

    logits_diff = (logits_s2 - logits_s1).abs()
    print(f"  Mean absolute difference in logits: {logits_diff.mean().item():.6f}")
    print(f"  Max absolute difference in logits: {logits_diff.max().item():.6f}")

    # Check if predictions changed
    preds_s1 = torch.argmax(logits_s1, dim=-1)
    preds_s2 = torch.argmax(logits_s2, dim=-1)
    changed_predictions = (preds_s1 != preds_s2).sum().item()
    total_predictions = preds_s1.numel()

    print(f"  Predictions changed: {changed_predictions}/{total_predictions} ({100*changed_predictions/total_predictions:.1f}%)")

    # Sample some actual predictions
    print("\n5. SAMPLE PREDICTIONS:")
    print("-" * 40)

    # Decode function
    itos = val_loader.dataset.itos
    decode = lambda ids: ''.join([itos[i] for i in ids])

    for i in range(min(5, x.shape[1])):
        context = decode(x[0, max(0, i-5):i].cpu().tolist())
        target = decode([y[0, i].cpu().item()])
        pred_s1 = decode([preds_s1[0, i].cpu().item()])
        pred_s2 = decode([preds_s2[0, i].cpu().item()])

        prob_s1 = F.softmax(logits_s1[0, i], dim=-1).max().item()
        prob_s2 = F.softmax(logits_s2[0, i], dim=-1).max().item()

        print(f"  Pos {i}: '{context}' -> Target: '{target}'")
        print(f"    S1: '{pred_s1}' (conf: {prob_s1:.2f})")
        print(f"    S2: '{pred_s2}' (conf: {prob_s2:.2f})")
        if pred_s1 != pred_s2:
            print(f"    ✨ CHANGED!")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_refinement()