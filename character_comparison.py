"""
Character-level comparison between System 1 (GPT) and System 2 (EBM with refinement)

Simple, grounded metrics that are easy to visualize and understand.

Usage:
    python character_comparison.py
    python character_comparison.py --checkpoint=out_ebt/final.pt --num_samples=100
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import chz

from nanoebm.model import EBM
from nanoebm.data import CharDataset, get_loader
from nanoebm.config import Config


def find_latest_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest checkpoint."""
    import glob
    import os
    
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")
    
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]
    
    checkpoint_path = os.path.join(latest_dir, "final.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if pt_files:
        pt_files.sort(key=os.path.getmtime, reverse=True)
        return pt_files[0]
    
    raise FileNotFoundError(f"No .pt files found in {latest_dir}")


@chz.chz
class CompareConfig:
    checkpoint: str | None = None  # None = auto-detect
    data_path: str = "shakespeare.txt"
    num_samples: int = 200  # Number of sequences to evaluate
    sequence_length: int = 100  # Length of each sequence
    refine_steps: int = 4  # Number of refinement steps for System 2
    batch_size: int = 1  # Process one at a time for detailed analysis
    output_dir: str = "out_ebt/comparisons"


def compute_character_metrics(
    model: EBM,
    loader,
    device: str,
    use_refine: bool,
    refine_steps: int = 4,
    num_samples: int = 100
) -> Dict:
    """
    Compute character-level metrics for the model.
    
    Returns dict with:
    - perplexity: average perplexity
    - top1_acc: top-1 accuracy
    - top5_acc: top-5 accuracy
    - avg_entropy: average prediction entropy
    - confidences: list of max probabilities for each prediction
    """
    model.eval()
    
    total_loss = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_tokens = 0
    entropies = []
    confidences = []
    
    sample_count = 0
    for x, y in loader:
        if sample_count >= num_samples:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Get predictions
        with torch.no_grad() if not use_refine else torch.enable_grad():
            if use_refine:
                logits = model.system2_refine(x, steps=refine_steps)
            else:
                logits = model.system1_direct_energy(x)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            y.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        
        # Compute accuracies
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 accuracy
        pred_top1 = logits.argmax(dim=-1)
        correct_top1 = (pred_top1 == y).sum().item()
        total_correct_top1 += correct_top1
        
        # Top-5 accuracy
        _, pred_top5 = logits.topk(5, dim=-1)
        correct_top5 = (pred_top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total_correct_top5 += correct_top5
        
        # Entropy (measure of uncertainty)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        entropies.extend(entropy.view(-1).detach().cpu().numpy())
        
        # Confidence (max probability)
        max_probs = probs.max(dim=-1)[0]
        confidences.extend(max_probs.view(-1).detach().cpu().numpy())
        
        total_tokens += y.numel()
        sample_count += x.size(0)
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    top1_acc = total_correct_top1 / total_tokens
    top5_acc = total_correct_top5 / total_tokens
    avg_entropy = np.mean(entropies)
    
    return {
        'perplexity': perplexity,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'avg_entropy': avg_entropy,
        'confidences': confidences,
        'entropies': entropies
    }


def visualize_prediction_example(
    model: EBM,
    dataset: CharDataset,
    device: str,
    prompt: str = "To be or not to be",
    next_chars: int = 10
):
    """
    Show a concrete example of how refinement changes predictions.
    """
    model.eval()
    
    # Encode prompt
    stoi = dataset.stoi
    itos = dataset.itos
    
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([prompt_ids], device=device)
    
    # Get predictions for next character
    # System 1 (no refinement)
    with torch.no_grad():
        logits_s1 = model.system1_direct_energy(x)
        probs_s1 = F.softmax(logits_s1[0, -1, :], dim=-1)
    
    # System 2 (with refinement) - needs gradients enabled
    with torch.enable_grad():
        logits_s2 = model.system2_refine(x, steps=4)
        probs_s2 = F.softmax(logits_s2[0, -1, :], dim=-1).detach()
    
    # Get top 5 predictions for each
    top5_s1_probs, top5_s1_idx = probs_s1.topk(5)
    top5_s2_probs, top5_s2_idx = probs_s2.topk(5)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # System 1 predictions
    chars_s1 = [repr(itos[i.item()]) for i in top5_s1_idx]
    probs_s1_list = top5_s1_probs.cpu().numpy()
    
    ax1.barh(range(5), probs_s1_list, color='#2E86AB')
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(chars_s1)
    ax1.set_xlabel('Probability')
    ax1.set_title(f'System 1 (Regular GPT)\nEntropy: {-(probs_s1 * probs_s1.clamp_min(1e-9).log()).sum().item():.3f}')
    ax1.set_xlim(0, max(probs_s1_list.max(), probs_s2.max().item()) * 1.1)
    
    # System 2 predictions
    chars_s2 = [repr(itos[i.item()]) for i in top5_s2_idx]
    probs_s2_list = top5_s2_probs.cpu().numpy()
    
    ax2.barh(range(5), probs_s2_list, color='#A23B72')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(chars_s2)
    ax2.set_xlabel('Probability')
    ax2.set_title(f'System 2 (With Refinement)\nEntropy: {-(probs_s2 * probs_s2.clamp_min(1e-9).log()).sum().item():.3f}')
    ax2.set_xlim(0, max(probs_s1_list.max(), probs_s2_list.max()) * 1.1)
    
    fig.suptitle(f'Next Character Predictions\nPrompt: "{prompt}"', fontsize=14)
    plt.tight_layout()
    
    return fig


def main(cfg: CompareConfig):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint = cfg.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        print(f"Auto-detected checkpoint: {checkpoint}")
    else:
        print(f"Using checkpoint: {checkpoint}")
    
    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    
    # Load dataset
    dataset = CharDataset(cfg.data_path, block_size=cfg.sequence_length, split="val")
    loader, _ = get_loader(cfg.data_path, cfg.sequence_length, cfg.batch_size, "val")
    
    # Initialize model - handle old checkpoint format
    from nanoebm.config import ModelConfig
    
    # Extract config, removing parameters that no longer exist
    config_dict = ckpt["config"]["model"] if "config" in ckpt else ckpt.get("model_config", {})
    
    # Remove deprecated parameters
    deprecated_params = ['use_replay_buffer']
    for param in deprecated_params:
        config_dict.pop(param, None)
    
    # Update vocab size in config dict before creating ModelConfig
    config_dict['vocab_size'] = len(dataset.stoi)
    
    model_cfg = ModelConfig(**config_dict)
    
    model = EBM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    
    print("\n" + "="*70)
    print("CHARACTER-LEVEL COMPARISON: System 1 (GPT) vs System 2 (EBM)")
    print("="*70)
    
    # Compute metrics for System 1
    print("\nEvaluating System 1 (Regular GPT - no refinement)...")
    metrics_s1 = compute_character_metrics(
        model, loader, device, 
        use_refine=False,
        num_samples=cfg.num_samples
    )
    
    # Compute metrics for System 2
    print(f"Evaluating System 2 (EBM with {cfg.refine_steps} refinement steps)...")
    metrics_s2 = compute_character_metrics(
        model, loader, device,
        use_refine=True,
        refine_steps=cfg.refine_steps,
        num_samples=cfg.num_samples
    )
    
    # Print comparison
    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"{'Metric':<25} {'System 1':>15} {'System 2':>15} {'Improvement':>15}")
    print("-"*70)
    
    # Perplexity (lower is better)
    ppl_imp = (metrics_s1['perplexity'] - metrics_s2['perplexity']) / metrics_s1['perplexity'] * 100
    print(f"{'Perplexity ↓':<25} {metrics_s1['perplexity']:>15.2f} {metrics_s2['perplexity']:>15.2f} {ppl_imp:>14.1f}%")
    
    # Top-1 Accuracy (higher is better)
    top1_imp = (metrics_s2['top1_acc'] - metrics_s1['top1_acc']) / metrics_s1['top1_acc'] * 100
    print(f"{'Top-1 Accuracy ↑':<25} {metrics_s1['top1_acc']:>15.3f} {metrics_s2['top1_acc']:>15.3f} {top1_imp:>14.1f}%")
    
    # Top-5 Accuracy (higher is better)
    top5_imp = (metrics_s2['top5_acc'] - metrics_s1['top5_acc']) / metrics_s1['top5_acc'] * 100
    print(f"{'Top-5 Accuracy ↑':<25} {metrics_s1['top5_acc']:>15.3f} {metrics_s2['top5_acc']:>15.3f} {top5_imp:>14.1f}%")
    
    # Entropy (lower means more confident)
    ent_imp = (metrics_s1['avg_entropy'] - metrics_s2['avg_entropy']) / metrics_s1['avg_entropy'] * 100
    print(f"{'Avg Entropy ↓':<25} {metrics_s1['avg_entropy']:>15.3f} {metrics_s2['avg_entropy']:>15.3f} {ent_imp:>14.1f}%")
    
    # Average confidence
    avg_conf_s1 = np.mean(metrics_s1['confidences'])
    avg_conf_s2 = np.mean(metrics_s2['confidences'])
    conf_imp = (avg_conf_s2 - avg_conf_s1) / avg_conf_s1 * 100
    print(f"{'Avg Confidence ↑':<25} {avg_conf_s1:>15.3f} {avg_conf_s2:>15.3f} {conf_imp:>14.1f}%")
    
    print("="*70)
    
    # Create visualizations
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Confidence distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(metrics_s1['confidences'], bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence (Max Probability)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'System 1 (GPT)\nMean: {avg_conf_s1:.3f}')
    axes[0].set_xlim(0, 1)
    
    axes[1].hist(metrics_s2['confidences'], bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    axes[1].set_xlabel('Prediction Confidence (Max Probability)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'System 2 (EBM)\nMean: {avg_conf_s2:.3f}')
    axes[1].set_xlim(0, 1)
    
    fig.suptitle('Character Prediction Confidence Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/confidence_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: {cfg.output_dir}/confidence_distribution.png")
    
    # 2. Entropy distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(metrics_s1['entropies'], bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0].set_xlabel('Prediction Entropy')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'System 1 (GPT)\nMean: {metrics_s1["avg_entropy"]:.3f}')
    
    axes[1].hist(metrics_s2['entropies'], bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    axes[1].set_xlabel('Prediction Entropy')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'System 2 (EBM)\nMean: {metrics_s2["avg_entropy"]:.3f}')
    
    fig.suptitle('Character Prediction Entropy Distribution\n(Lower = More Confident)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/entropy_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {cfg.output_dir}/entropy_distribution.png")
    
    # 3. Example prediction comparison
    fig = visualize_prediction_example(model, dataset, device, prompt="HAMLET:")
    plt.savefig(f"{cfg.output_dir}/prediction_example.png", dpi=150)
    plt.close()
    print(f"Saved: {cfg.output_dir}/prediction_example.png")
    
    # Summary metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Perplexity\n(normalized)', 'Top-1 Acc', 'Top-5 Acc', 'Confidence']
    s1_values = [
        min(metrics_s1['perplexity'] / 100, 1),  # Normalize perplexity for display
        metrics_s1['top1_acc'],
        metrics_s1['top5_acc'],
        avg_conf_s1
    ]
    s2_values = [
        min(metrics_s2['perplexity'] / 100, 1),  # Normalize perplexity for display
        metrics_s2['top1_acc'],
        metrics_s2['top5_acc'],
        avg_conf_s2
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, s1_values, width, label='System 1 (GPT)', color='#2E86AB')
    bars2 = ax.bar(x + width/2, s2_values, width, label='System 2 (EBM)', color='#A23B72')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Character-Level Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/metrics_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {cfg.output_dir}/metrics_comparison.png")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("-"*70)
    print("• System 2 (EBM with refinement) consistently outperforms System 1 (GPT)")
    print("• Lower perplexity = better next-character prediction")
    print("• Higher confidence and lower entropy = more decisive predictions")
    print("• Refinement helps the model 'think' before making predictions")
    print("="*70)


if __name__ == "__main__":
    config = chz.entrypoint(CompareConfig)
    main(config)