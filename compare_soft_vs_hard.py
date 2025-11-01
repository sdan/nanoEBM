"""
Compare hard tokens (frozen context) vs soft tokens (shifting context) refinement.

Usage:
    # Compare two trained checkpoints
    python compare_soft_vs_hard.py \
        --hard_ckpt out_ebt/hard_tokens/final.pt \
        --soft_ckpt out_ebt/soft_tokens/final.pt

    # Or train from scratch and compare
    python compare_soft_vs_hard.py --train_steps 1000
"""

import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path

from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import get_loader


def evaluate_model(model, data_loader, use_soft_tokens=False, num_batches=50, refine_steps=4):
    """
    Evaluate model and track detailed metrics.

    Returns:
        metrics: dict with perplexity, energies, convergence info
    """
    model.eval()

    all_metrics = {
        'nll': [],
        'perplexity': [],
        'initial_energy': [],
        'final_energy': [],
        'energy_gap': [],
        'convergence_steps': [],
        'time_per_step': [],
    }

    # Track refinement trajectory for visualization
    trajectories = []

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data_loader, desc="Evaluating", total=num_batches)):
            if i >= num_batches:
                break

            device = next(model.parameters()).device
            x, y = x.to(device), y.to(device)

            # Get S1 baseline
            h = model.get_hidden_states(x)
            energies = model.energy_head(h)
            logits_s1 = -energies

            # Compute S1 metrics
            nll_s1 = F.cross_entropy(logits_s1.view(-1, model.config.vocab_size), y.view(-1))
            probs_s1 = F.softmax(logits_s1, dim=-1)
            energy_s1 = (probs_s1 * energies).sum(dim=-1).mean()

            # Get S2 with refinement
            import time
            start_time = time.time()
            logits_s2, trajectory = model.system2_refine(
                x,
                steps=refine_steps,
                use_soft_tokens=use_soft_tokens,
                return_trajectory=True
            )
            elapsed = time.time() - start_time

            # Compute S2 metrics
            nll_s2 = F.cross_entropy(logits_s2.view(-1, model.config.vocab_size), y.view(-1))
            probs_s2 = F.softmax(logits_s2, dim=-1)

            # Recompute energy for S2 (may be different if soft tokens)
            if use_soft_tokens:
                soft_emb = torch.einsum('btv,ve->bte', probs_s2, model.transformer.transformer.wte.weight)
                h_s2 = model.get_hidden_states(soft_emb)
                energies_s2 = model.energy_head(h_s2)
            else:
                energies_s2 = energies

            energy_s2 = (probs_s2 * energies_s2).sum(dim=-1).mean()

            # Track metrics
            all_metrics['nll'].append(nll_s2.item())
            all_metrics['perplexity'].append(torch.exp(nll_s2).item())
            all_metrics['initial_energy'].append(energy_s1.item())
            all_metrics['final_energy'].append(energy_s2.item())
            all_metrics['energy_gap'].append((energy_s1 - energy_s2).item())
            all_metrics['time_per_step'].append(elapsed / refine_steps)

            # Store first batch trajectory for visualization
            if i == 0:
                trajectories.append({
                    'logits': [t.cpu() for t in trajectory],
                    'energies_list': []
                })

    # Aggregate metrics
    summary = {
        'mean_nll': np.mean(all_metrics['nll']),
        'mean_ppl': np.mean(all_metrics['perplexity']),
        'mean_initial_energy': np.mean(all_metrics['initial_energy']),
        'mean_final_energy': np.mean(all_metrics['final_energy']),
        'mean_energy_gap': np.mean(all_metrics['energy_gap']),
        'mean_time_per_step': np.mean(all_metrics['time_per_step']),
        'std_ppl': np.std(all_metrics['perplexity']),
    }

    return summary, all_metrics, trajectories


def visualize_comparison(hard_metrics, soft_metrics, hard_traj, soft_traj, save_path):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hard Tokens vs Soft Tokens Refinement Comparison', fontsize=16, fontweight='bold')

    # 1. Perplexity distribution
    ax = axes[0, 0]
    ax.hist(hard_metrics['perplexity'], bins=30, alpha=0.5, label='Hard', color='blue')
    ax.hist(soft_metrics['perplexity'], bins=30, alpha=0.5, label='Soft', color='red')
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Count')
    ax.set_title('Perplexity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energy gap comparison
    ax = axes[0, 1]
    ax.hist(hard_metrics['energy_gap'], bins=30, alpha=0.5, label='Hard', color='blue')
    ax.hist(soft_metrics['energy_gap'], bins=30, alpha=0.5, label='Soft', color='red')
    ax.set_xlabel('Energy Gap (E0 - EK)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Improvement from Refinement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final energy comparison
    ax = axes[0, 2]
    ax.scatter(hard_metrics['initial_energy'], hard_metrics['final_energy'],
               alpha=0.5, s=20, label='Hard', color='blue')
    ax.scatter(soft_metrics['initial_energy'], soft_metrics['final_energy'],
               alpha=0.5, s=20, label='Soft', color='red')
    ax.plot([min(hard_metrics['initial_energy']), max(hard_metrics['initial_energy'])],
            [min(hard_metrics['initial_energy']), max(hard_metrics['initial_energy'])],
            'k--', alpha=0.3, label='No improvement')
    ax.set_xlabel('Initial Energy (E0)')
    ax.set_ylabel('Final Energy (EK)')
    ax.set_title('Energy Before vs After Refinement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Box plot comparison
    ax = axes[1, 0]
    data = [hard_metrics['perplexity'], soft_metrics['perplexity']]
    bp = ax.boxplot(data, labels=['Hard', 'Soft'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity Comparison')
    ax.grid(True, alpha=0.3)

    # 5. Computational cost
    ax = axes[1, 1]
    methods = ['Hard', 'Soft']
    times = [
        np.mean(hard_metrics['time_per_step']) * 1000,  # Convert to ms
        np.mean(soft_metrics['time_per_step']) * 1000
    ]
    bars = ax.bar(methods, times, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Time per refinement step (ms)')
    ax.set_title('Computational Cost')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')

    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = [
        ['Metric', 'Hard', 'Soft', 'Diff'],
        ['Mean PPL', f"{np.mean(hard_metrics['perplexity']):.3f}",
         f"{np.mean(soft_metrics['perplexity']):.3f}",
         f"{np.mean(soft_metrics['perplexity']) - np.mean(hard_metrics['perplexity']):.3f}"],
        ['Std PPL', f"{np.std(hard_metrics['perplexity']):.3f}",
         f"{np.std(soft_metrics['perplexity']):.3f}", ''],
        ['Mean E-gap', f"{np.mean(hard_metrics['energy_gap']):.4f}",
         f"{np.mean(soft_metrics['energy_gap']):.4f}",
         f"{np.mean(soft_metrics['energy_gap']) - np.mean(hard_metrics['energy_gap']):.4f}"],
        ['Time/step (ms)', f"{np.mean(hard_metrics['time_per_step'])*1000:.2f}",
         f"{np.mean(soft_metrics['time_per_step'])*1000:.2f}",
         f"{(np.mean(soft_metrics['time_per_step']) - np.mean(hard_metrics['time_per_step']))*1000:.2f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header row styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare hard vs soft token refinement')
    parser.add_argument('--hard_ckpt', type=str, help='Path to hard tokens checkpoint')
    parser.add_argument('--soft_ckpt', type=str, help='Path to soft tokens checkpoint')
    parser.add_argument('--train_steps', type=int, default=0, help='Train from scratch for N steps')
    parser.add_argument('--data_path', type=str, default='shakespeare.txt', help='Data file')
    parser.add_argument('--refine_steps', type=int, default=4, help='Number of refinement steps')
    parser.add_argument('--eval_batches', type=int, default=50, help='Number of batches to evaluate')
    parser.add_argument('--out_dir', type=str, default='comparison_results', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load or create models
    if args.hard_ckpt and args.soft_ckpt:
        print(f"\nLoading checkpoints...")
        print(f"  Hard: {args.hard_ckpt}")
        print(f"  Soft: {args.soft_ckpt}")

        # Load hard tokens model
        ckpt_hard = torch.load(args.hard_ckpt, map_location=device)
        config_hard = ModelConfig(**ckpt_hard['config']['model'])
        config_hard = ModelConfig(
            **{k: v for k, v in ckpt_hard['config']['model'].items()
               if k in ModelConfig.__dataclass_fields__}
        )
        model_hard = EBM(config_hard).to(device)
        model_hard.load_state_dict(ckpt_hard['model'])

        # Load soft tokens model
        ckpt_soft = torch.load(args.soft_ckpt, map_location=device)
        config_soft = ModelConfig(
            **{k: v for k, v in ckpt_soft['config']['model'].items()
               if k in ModelConfig.__dataclass_fields__}
        )
        model_soft = EBM(config_soft).to(device)
        model_soft.load_state_dict(ckpt_soft['model'])

    else:
        print("\nCreating models from scratch...")
        config = ModelConfig(
            vocab_size=256,
            block_size=256,
            n_layer=4,
            n_head=4,
            n_embd=256,
            refine_steps=args.refine_steps,
        )

        model_hard = EBM(config).to(device)
        model_soft = EBM(config).to(device)

        if args.train_steps > 0:
            print(f"\n⚠ Training from scratch not implemented in this script.")
            print(f"  Use train.py with model.use_soft_tokens=true/false instead.")
            return

    # Load evaluation data
    print(f"\nLoading data: {args.data_path}")
    val_loader, _ = get_loader(args.data_path, 256, 32, "val")

    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating Hard Tokens (Frozen Context)")
    print("="*60)
    hard_summary, hard_metrics, hard_traj = evaluate_model(
        model_hard, val_loader,
        use_soft_tokens=False,
        num_batches=args.eval_batches,
        refine_steps=args.refine_steps
    )

    print("\n" + "="*60)
    print("Evaluating Soft Tokens (Shifting Context)")
    print("="*60)
    soft_summary, soft_metrics, soft_traj = evaluate_model(
        model_soft, val_loader,
        use_soft_tokens=True,
        num_batches=args.eval_batches,
        refine_steps=args.refine_steps
    )

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\n{'Metric':<25} {'Hard':<15} {'Soft':<15} {'Δ (Soft-Hard)':<15}")
    print("-" * 70)
    print(f"{'Perplexity':<25} {hard_summary['mean_ppl']:<15.3f} {soft_summary['mean_ppl']:<15.3f} {soft_summary['mean_ppl'] - hard_summary['mean_ppl']:<15.3f}")
    print(f"{'NLL':<25} {hard_summary['mean_nll']:<15.3f} {soft_summary['mean_nll']:<15.3f} {soft_summary['mean_nll'] - hard_summary['mean_nll']:<15.3f}")
    print(f"{'Energy Gap':<25} {hard_summary['mean_energy_gap']:<15.4f} {soft_summary['mean_energy_gap']:<15.4f} {soft_summary['mean_energy_gap'] - hard_summary['mean_energy_gap']:<15.4f}")
    print(f"{'Time/step (ms)':<25} {hard_summary['mean_time_per_step']*1000:<15.2f} {soft_summary['mean_time_per_step']*1000:<15.2f} {(soft_summary['mean_time_per_step'] - hard_summary['mean_time_per_step'])*1000:<15.2f}")

    print("\n" + "="*60)

    # Interpret results
    print("\nInterpretation:")
    if soft_summary['mean_ppl'] < hard_summary['mean_ppl']:
        print(f"  ✓ Soft tokens achieve BETTER perplexity ({soft_summary['mean_ppl']:.3f} vs {hard_summary['mean_ppl']:.3f})")
    else:
        print(f"  ✗ Hard tokens achieve better perplexity ({hard_summary['mean_ppl']:.3f} vs {soft_summary['mean_ppl']:.3f})")

    if soft_summary['mean_energy_gap'] > hard_summary['mean_energy_gap']:
        print(f"  ✓ Soft tokens show LARGER energy improvement from refinement")
    else:
        print(f"  ✗ Hard tokens show larger energy improvement from refinement")

    speedup = hard_summary['mean_time_per_step'] / soft_summary['mean_time_per_step']
    if speedup > 1:
        print(f"  ! Soft tokens are {1/speedup:.2f}x SLOWER (expected due to recomputation)")
    else:
        print(f"  ! Soft tokens are {speedup:.2f}x faster (unexpected)")

    # Save results
    results = {
        'hard': hard_summary,
        'soft': soft_summary,
        'args': vars(args),
    }

    results_path = out_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")

    # Create visualization
    viz_path = out_dir / 'comparison_viz.png'
    visualize_comparison(hard_metrics, soft_metrics, hard_traj, soft_traj, viz_path)

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == '__main__':
    main()
