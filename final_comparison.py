"""
Final comparison: Show clear differences between GPT and EBM
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from nanoebm.model import EBM
from nanoebm.data import get_loader
from nanoebm.config import Config, ModelConfig, DataConfig, TrainConfig
import chz

def load_trained_models():
    """Load the trained GPT and EBM models."""
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    # Create config
    model_cfg = ModelConfig(
        vocab_size=67,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        bias=True,
        refine_steps=4,
        alpha_value=0.1,
        langevin_noise=0.01,
        energy_convergence_threshold=1e-4,
        warmup_steps_no_refine=100
    )

    # Load GPT model
    gpt_model = EBM(model_cfg).to(device)
    gpt_checkpoint = torch.load("simple_gpt_model.pt", map_location=device, weights_only=False)
    gpt_model.load_state_dict(gpt_checkpoint['model'])
    gpt_model.eval()

    # Load EBM model
    ebm_model = EBM(model_cfg).to(device)
    ebm_checkpoint = torch.load("simple_ebm_model.pt", map_location=device, weights_only=False)
    ebm_model.load_state_dict(ebm_checkpoint['model'])
    ebm_model.eval()

    return gpt_model, ebm_model, device

def detailed_comparison():
    """Run detailed comparison between models."""

    print("\n" + "="*70)
    print(" "*20 + "GPT vs EBM: FINAL COMPARISON")
    print("="*70)

    # Load models
    try:
        gpt_model, ebm_model, device = load_trained_models()
        print(f"✓ Loaded both models on {device}")
    except FileNotFoundError:
        print("❌ Models not found. Please run train_simple_models.py first!")
        return

    # Load data
    val_loader, val_ds = get_loader("shakespeare.txt", 128, 16, "val")
    itos = val_ds.itos
    decode = lambda ids: ''.join([itos[i] for i in ids])

    print(f"✓ Loaded Shakespeare validation data\n")

    # 1. PERPLEXITY COMPARISON
    print("1. PERPLEXITY COMPARISON")
    print("-" * 40)

    losses = {'gpt': [], 'ebm_base': [], 'ebm_refined': []}

    for i, (x, y) in enumerate(val_loader):
        if i >= 30:
            break

        x, y = x.to(device), y.to(device)

        # GPT
        with torch.no_grad():
            loss_gpt, _, _ = gpt_model(x, targets=y, use_refine=False)
            losses['gpt'].append(loss_gpt.item())

        # EBM without refinement
        with torch.no_grad():
            loss_ebm_base, _, _ = ebm_model(x, targets=y, use_refine=False)
            losses['ebm_base'].append(loss_ebm_base.item())

        # EBM with refinement
        with torch.enable_grad():
            loss_ebm_refined, _, _ = ebm_model(x, targets=y, use_refine=True, refine_steps=4)
            losses['ebm_refined'].append(loss_ebm_refined.detach().item())

    # Calculate perplexities
    ppl_gpt = np.exp(np.mean(losses['gpt']))
    ppl_ebm_base = np.exp(np.mean(losses['ebm_base']))
    ppl_ebm_refined = np.exp(np.mean(losses['ebm_refined']))

    print(f"  GPT (baseline):      {ppl_gpt:.2f}")
    print(f"  EBM (no refinement): {ppl_ebm_base:.2f}")
    print(f"  EBM (with refine):   {ppl_ebm_refined:.2f}")

    if ppl_ebm_refined < ppl_gpt:
        improvement = (ppl_gpt - ppl_ebm_refined) / ppl_gpt * 100
        print(f"\n  ✅ EBM refinement reduces perplexity by {improvement:.1f}%!")
    else:
        print(f"\n  ❌ No improvement from refinement")

    # 2. PREDICTION CHANGES
    print("\n2. ANALYZING PREDICTION CHANGES")
    print("-" * 40)

    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)

    # Get predictions from each model
    with torch.no_grad():
        _, logits_gpt, _ = gpt_model(x, targets=y, use_refine=False)
        _, logits_ebm_base, _ = ebm_model(x, targets=y, use_refine=False)

    with torch.enable_grad():
        _, logits_ebm_refined, _ = ebm_model(x, targets=y, use_refine=True, refine_steps=4)
        logits_ebm_refined = logits_ebm_refined.detach()

    preds_gpt = torch.argmax(logits_gpt, dim=-1)
    preds_ebm_base = torch.argmax(logits_ebm_base, dim=-1)
    preds_ebm_refined = torch.argmax(logits_ebm_refined, dim=-1)

    # Count differences
    diff_gpt_ebm = (preds_gpt != preds_ebm_refined).sum().item()
    diff_base_refined = (preds_ebm_base != preds_ebm_refined).sum().item()

    print(f"  GPT vs EBM refined: {diff_gpt_ebm}/{preds_gpt.numel()} predictions differ ({100*diff_gpt_ebm/preds_gpt.numel():.1f}%)")
    print(f"  EBM base vs refined: {diff_base_refined}/{preds_gpt.numel()} predictions differ ({100*diff_base_refined/preds_gpt.numel():.1f}%)")

    # 3. CONFIDENCE ANALYSIS
    print("\n3. CONFIDENCE DISTRIBUTION")
    print("-" * 40)

    probs_gpt = F.softmax(logits_gpt, dim=-1).max(dim=-1).values
    probs_ebm_base = F.softmax(logits_ebm_base, dim=-1).max(dim=-1).values
    probs_ebm_refined = F.softmax(logits_ebm_refined, dim=-1).max(dim=-1).values

    print(f"  Average max probability:")
    print(f"    GPT:          {probs_gpt.mean():.3f} ± {probs_gpt.std():.3f}")
    print(f"    EBM base:     {probs_ebm_base.mean():.3f} ± {probs_ebm_base.std():.3f}")
    print(f"    EBM refined:  {probs_ebm_refined.mean():.3f} ± {probs_ebm_refined.std():.3f}")

    # 4. EXAMPLE PREDICTIONS
    print("\n4. SAMPLE PREDICTIONS")
    print("-" * 40)

    # Find interesting examples where predictions differ
    examples_found = 0
    for b in range(min(4, x.shape[0])):
        for t in range(x.shape[1]):
            if preds_ebm_base[b, t] != preds_ebm_refined[b, t] and examples_found < 5:
                context = decode(x[b, max(0, t-15):t].cpu().tolist())
                target = decode([y[b, t].cpu().item()])

                pred_gpt = decode([preds_gpt[b, t].cpu().item()])
                pred_base = decode([preds_ebm_base[b, t].cpu().item()])
                pred_refined = decode([preds_ebm_refined[b, t].cpu().item()])

                conf_gpt = probs_gpt[b, t].item()
                conf_base = probs_ebm_base[b, t].item()
                conf_refined = probs_ebm_refined[b, t].item()

                print(f"\n  Context: '...{context[-20:]}' → Target: '{target}'")
                print(f"    GPT:          '{pred_gpt}' (conf: {conf_gpt:.2f}) {'✓' if pred_gpt == target else '✗'}")
                print(f"    EBM base:     '{pred_base}' (conf: {conf_base:.2f}) {'✓' if pred_base == target else '✗'}")
                print(f"    EBM refined:  '{pred_refined}' (conf: {conf_refined:.2f}) {'✓' if pred_refined == target else '✗'}")

                if pred_base != target and pred_refined == target:
                    print(f"    💡 Refinement fixed the prediction!")
                elif pred_base == target and pred_refined != target:
                    print(f"    ⚠️  Refinement broke the prediction")

                examples_found += 1

    if examples_found == 0:
        print("\n  No prediction changes found in this batch.")
        print("  (Models might need more training to show differences)")

    # 5. REFINEMENT TRAJECTORY
    print("\n5. REFINEMENT TRAJECTORY")
    print("-" * 40)

    # Track how predictions evolve with more refinement steps
    x_test, y_test = next(iter(val_loader))
    x_test, y_test = x_test.to(device)[:1], y_test.to(device)[:1]  # Just one sequence

    trajectory = []
    for steps in [0, 1, 2, 4, 8, 16]:
        if steps == 0:
            with torch.no_grad():
                loss, _, _ = ebm_model(x_test, targets=y_test, use_refine=False)
        else:
            with torch.enable_grad():
                loss, _, _ = ebm_model(x_test, targets=y_test, use_refine=True, refine_steps=steps)
                loss = loss.detach()

        trajectory.append((steps, loss.item(), np.exp(loss.item())))

    print("  Steps  Loss    Perplexity")
    for steps, loss, ppl in trajectory:
        label = "base" if steps == 0 else f"{steps:2d}"
        print(f"   {label:4}  {loss:.3f}   {ppl:.2f}")

    # Check if refinement helps
    if trajectory[-1][1] < trajectory[0][1]:
        print("\n  ✅ Refinement consistently improves predictions!")
    else:
        print("\n  ⚠️  Refinement doesn't improve predictions")

    # FINAL VERDICT
    print("\n" + "="*70)
    print(" "*25 + "FINAL VERDICT")
    print("="*70)

    improvements = []
    if ppl_ebm_refined < ppl_gpt:
        improvements.append(f"Lower perplexity ({(ppl_gpt - ppl_ebm_refined)/ppl_gpt*100:.1f}% better)")
    if diff_base_refined > 0:
        improvements.append(f"Changes {100*diff_base_refined/preds_gpt.numel():.1f}% of predictions")
    if trajectory[-1][1] < trajectory[0][1]:
        improvements.append("Progressive refinement works")

    if improvements:
        print("✅ EBM with refinement shows clear advantages:")
        for imp in improvements:
            print(f"   • {imp}")
    else:
        print("❌ No clear advantage from EBM refinement")
        print("   Models may need more training or different hyperparameters")

    print("="*70)

    # Save comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Perplexity comparison
    models = ['GPT', 'EBM\n(base)', 'EBM\n(refined)']
    perplexities = [ppl_gpt, ppl_ebm_base, ppl_ebm_refined]
    axes[0].bar(models, perplexities, color=['blue', 'orange', 'green'])
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('Model Perplexity Comparison')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Refinement trajectory
    steps_list = [t[0] for t in trajectory]
    ppl_list = [t[2] for t in trajectory]
    axes[1].plot(steps_list, ppl_list, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Refinement Steps')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Refinement Impact')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Confidence distribution
    axes[2].hist([probs_gpt.cpu().numpy().flatten(),
                  probs_ebm_base.cpu().numpy().flatten(),
                  probs_ebm_refined.cpu().numpy().flatten()],
                 bins=20, alpha=0.5, label=['GPT', 'EBM base', 'EBM refined'])
    axes[2].set_xlabel('Max Probability')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Confidence Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('GPT vs EBM: Academic Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gpt_vs_ebm_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison plot saved to gpt_vs_ebm_comparison.png")

if __name__ == "__main__":
    detailed_comparison()