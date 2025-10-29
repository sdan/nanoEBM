"""
Train two models for comparison: Pure GPT vs EBM
Quick training script to demonstrate the difference between the two approaches.

Usage:
    python train_comparison_models.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from nanoebm.model import EBM
from nanoebm.config import ModelConfig
from nanoebm.data import get_loader, CharDataset
import chz


@chz.chz
class QuickTrainConfig:
    # Training params
    max_steps: int = 2000  # Increased from 500 for better training
    learning_rate: float = 3e-4
    batch_size: int = 64
    block_size: int = 128
    
    # Model params (small for quick training)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    
    # Data
    data_path: str = "shakespeare.txt"
    
    # Output
    output_dir: str = "out_ebt/comparison_models"


def train_model(name: str, use_refinement: bool, cfg: QuickTrainConfig):
    """Train a single model with or without refinement."""
    
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Refinement: {'ENABLED' if use_refinement else 'DISABLED'}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load dataset
    train_loader, train_ds = get_loader(
        cfg.data_path,
        cfg.block_size,
        cfg.batch_size,
        "train"
    )
    
    # Model config
    model_cfg = ModelConfig(
        vocab_size=len(train_ds.stoi),
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        refine_steps=4 if use_refinement else 0,
        alpha_value=0.5,  # Fixed step size for refinement
    )
    
    # Initialize model
    model = EBM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    # Training loop
    model.train()
    losses = []
    perplexities = []
    
    for step, (x, y) in enumerate(train_loader):
        if step >= cfg.max_steps:
            break
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        loss, logits, metrics = model(
            x, 
            targets=y, 
            use_refine=use_refinement,
            refine_steps=4 if use_refinement else 0
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        perplexities.append(np.exp(loss.item()))
        
        # Log progress
        if step % 50 == 0:
            avg_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
            avg_ppl = np.exp(avg_loss)
            print(f"Step {step:4d} | Loss: {avg_loss:.4f} | Perplexity: {avg_ppl:.2f}")
            
            if use_refinement and 'energy_gap' in metrics:
                print(f"           | Energy Gap: {metrics['energy_gap']:.4f}")
    
    # Save model
    save_path = Path(cfg.output_dir) / f"{name.lower().replace(' ', '_')}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model': model.state_dict(),
        'config': {
            'model': model_cfg.__dict__,
            'training': {
                'use_refinement': use_refinement,
                'steps': cfg.max_steps,
                'final_loss': losses[-1],
                'final_perplexity': perplexities[-1]
            }
        }
    }, save_path)
    
    print(f"\nSaved model to: {save_path}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Perplexity: {perplexities[-1]:.2f}")
    
    return model, losses, perplexities, save_path


def compare_generation(model_gpt, model_ebm, dataset, prompt="ROMEO:", max_tokens=100):
    """Generate text from both models and compare."""
    
    device = next(model_gpt.parameters()).device
    stoi = dataset.stoi
    itos = dataset.itos
    
    # Encode prompt
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([prompt_ids], device=device)
    
    # Store per-character metrics
    gpt_probs = []
    ebm_s1_probs = []
    ebm_s2_probs = []
    generated_chars = []
    
    # Generate and track metrics
    for _ in range(max_tokens):
        # GPT model (no refinement)
        with torch.no_grad():
            logits_gpt = model_gpt.system1_direct_energy(x)
            probs_gpt = F.softmax(logits_gpt[0, -1, :], dim=-1)
            next_token_gpt = torch.multinomial(probs_gpt, 1)
            
        # EBM model - System 1
        with torch.no_grad():
            logits_ebm_s1 = model_ebm.system1_direct_energy(x)
            probs_ebm_s1 = F.softmax(logits_ebm_s1[0, -1, :], dim=-1)
            
        # EBM model - System 2 (with refinement)
        with torch.enable_grad():
            logits_ebm_s2 = model_ebm.system2_refine(x, steps=4)
            probs_ebm_s2 = F.softmax(logits_ebm_s2[0, -1, :], dim=-1).detach()
            next_token_ebm = torch.multinomial(probs_ebm_s2, 1)
        
        # Use EBM's choice for continuation
        next_token = next_token_ebm.unsqueeze(0) if next_token_ebm.dim() == 1 else next_token_ebm
        x = torch.cat([x, next_token], dim=1)
        
        # Store metrics
        char = itos[next_token.item()]
        generated_chars.append(char)
        
        # Get probability of the chosen character from each model
        token_idx = next_token.item()
        gpt_probs.append(probs_gpt[token_idx].item())
        ebm_s1_probs.append(probs_ebm_s1[token_idx].item())
        ebm_s2_probs.append(probs_ebm_s2[token_idx].item())
    
    generated_text = ''.join(generated_chars)
    
    # Calculate perplexities
    gpt_ppl = np.exp(-np.mean(np.log(np.array(gpt_probs) + 1e-10)))
    ebm_s1_ppl = np.exp(-np.mean(np.log(np.array(ebm_s1_probs) + 1e-10)))
    ebm_s2_ppl = np.exp(-np.mean(np.log(np.array(ebm_s2_probs) + 1e-10)))
    
    return {
        'prompt': prompt,
        'generated': generated_text,
        'gpt_probs': gpt_probs,
        'ebm_s1_probs': ebm_s1_probs,
        'ebm_s2_probs': ebm_s2_probs,
        'gpt_ppl': gpt_ppl,
        'ebm_s1_ppl': ebm_s1_ppl,
        'ebm_s2_ppl': ebm_s2_ppl,
        'chars': generated_chars
    }


def visualize_sequence_comparison(results, output_dir):
    """Create visualization showing sequence with underlying metrics."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Prepare data
    chars = results['chars'][:50]  # First 50 chars for visibility
    positions = np.arange(len(chars))
    
    gpt_probs = results['gpt_probs'][:50]
    ebm_s1_probs = results['ebm_s1_probs'][:50]
    ebm_s2_probs = results['ebm_s2_probs'][:50]
    
    # Convert probabilities to negative log likelihood (lower is better)
    gpt_nll = -np.log(np.array(gpt_probs) + 1e-10)
    ebm_s1_nll = -np.log(np.array(ebm_s1_probs) + 1e-10)
    ebm_s2_nll = -np.log(np.array(ebm_s2_probs) + 1e-10)
    
    # Plot 1: Character probabilities
    ax1 = axes[0]
    width = 0.25
    ax1.bar(positions - width, gpt_probs, width, label='Pure GPT', color='#2E86AB', alpha=0.7)
    ax1.bar(positions, ebm_s1_probs, width, label='EBM System 1', color='#A23B72', alpha=0.7)
    ax1.bar(positions + width, ebm_s2_probs, width, label='EBM System 2', color='#F18F01', alpha=0.7)
    ax1.set_ylabel('Probability')
    ax1.set_title('Character-by-Character Probability Comparison')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add character labels
    ax1.set_xticks(positions)
    ax1.set_xticklabels([repr(c) for c in chars], rotation=45, ha='right', fontsize=8)
    
    # Plot 2: Negative log likelihood (perplexity component)
    ax2 = axes[1]
    ax2.plot(positions, gpt_nll, 'o-', label=f'Pure GPT (avg: {np.mean(gpt_nll):.2f})', color='#2E86AB', alpha=0.7)
    ax2.plot(positions, ebm_s1_nll, 's-', label=f'EBM S1 (avg: {np.mean(ebm_s1_nll):.2f})', color='#A23B72', alpha=0.7)
    ax2.plot(positions, ebm_s2_nll, '^-', label=f'EBM S2 (avg: {np.mean(ebm_s2_nll):.2f})', color='#F18F01', alpha=0.7)
    ax2.set_ylabel('Negative Log Likelihood')
    ax2.set_title('Character-Level NLL (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative perplexity
    ax3 = axes[2]
    cumulative_positions = np.arange(1, len(chars) + 1)
    cum_gpt_ppl = [np.exp(-np.mean(np.log(gpt_probs[:i] + np.array([1e-10])))) for i in cumulative_positions]
    cum_ebm_s1_ppl = [np.exp(-np.mean(np.log(ebm_s1_probs[:i] + np.array([1e-10])))) for i in cumulative_positions]
    cum_ebm_s2_ppl = [np.exp(-np.mean(np.log(ebm_s2_probs[:i] + np.array([1e-10])))) for i in cumulative_positions]
    
    ax3.plot(cumulative_positions, cum_gpt_ppl, label=f'Pure GPT (final: {cum_gpt_ppl[-1]:.2f})', color='#2E86AB', linewidth=2)
    ax3.plot(cumulative_positions, cum_ebm_s1_ppl, label=f'EBM S1 (final: {cum_ebm_s1_ppl[-1]:.2f})', color='#A23B72', linewidth=2)
    ax3.plot(cumulative_positions, cum_ebm_s2_ppl, label=f'EBM S2 (final: {cum_ebm_s2_ppl[-1]:.2f})', color='#F18F01', linewidth=2)
    ax3.set_xlabel('Position in Sequence')
    ax3.set_ylabel('Cumulative Perplexity')
    ax3.set_title('Running Perplexity (Lower is Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add generated text as title
    fig.suptitle(f'Prompt: "{results["prompt"]}"\nGenerated: "{results["generated"][:50]}..."', fontsize=12)
    
    plt.tight_layout()
    save_path = Path(output_dir) / "sequence_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization to: {save_path}")


def main():
    cfg = QuickTrainConfig()
    
    # Train both models
    print("\n" + "="*60)
    print("TRAINING TWO MODELS FOR COMPARISON")
    print("="*60)
    
    # Model A: Pure GPT (no refinement)
    model_gpt, losses_gpt, ppl_gpt, path_gpt = train_model(
        "Pure GPT (No Refinement)",
        use_refinement=False,
        cfg=cfg
    )
    
    # Model B: EBM (with refinement)
    model_ebm, losses_ebm, ppl_ebm, path_ebm = train_model(
        "EBM (With Refinement)",
        use_refinement=True,
        cfg=cfg
    )
    
    # Load dataset for generation
    dataset = CharDataset(cfg.data_path, block_size=cfg.block_size, split="val")
    
    # Compare on validation data
    val_loader, _ = get_loader(cfg.data_path, cfg.block_size, batch_size=1, split="val")
    
    print("\n" + "="*60)
    print("COMPARING MODELS ON VALIDATION DATA")
    print("="*60)
    
    device = next(model_gpt.parameters()).device
    
    # Quick validation comparison
    model_gpt.eval()
    model_ebm.eval()
    
    val_losses_gpt = []
    val_losses_ebm_s1 = []
    val_losses_ebm_s2 = []
    
    for i, (x, y) in enumerate(val_loader):
        if i >= 50:  # Quick validation
            break
        
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            # Pure GPT
            loss_gpt, _, _ = model_gpt(x, y, use_refine=False)
            val_losses_gpt.append(loss_gpt.item())
            
            # EBM System 1
            loss_ebm_s1, _, _ = model_ebm(x, y, use_refine=False)
            val_losses_ebm_s1.append(loss_ebm_s1.item())
        
        # EBM System 2 (needs gradients)
        with torch.enable_grad():
            loss_ebm_s2, _, _ = model_ebm(x, y, use_refine=True, refine_steps=4)
            val_losses_ebm_s2.append(loss_ebm_s2.item())
    
    print(f"\nValidation Perplexities:")
    print(f"  Pure GPT:    {np.exp(np.mean(val_losses_gpt)):.2f}")
    print(f"  EBM System 1: {np.exp(np.mean(val_losses_ebm_s1)):.2f}")
    print(f"  EBM System 2: {np.exp(np.mean(val_losses_ebm_s2)):.2f}")
    
    # Generate and compare sequences
    print("\n" + "="*60)
    print("GENERATING AND COMPARING SEQUENCES")
    print("="*60)
    
    results = compare_generation(model_gpt, model_ebm, dataset, prompt="HAMLET:", max_tokens=100)
    
    print(f"\nPrompt: {results['prompt']}")
    print(f"Generated: {results['generated'][:100]}...")
    print(f"\nPerplexities on generated sequence:")
    print(f"  Pure GPT:     {results['gpt_ppl']:.2f}")
    print(f"  EBM System 1: {results['ebm_s1_ppl']:.2f}")
    print(f"  EBM System 2: {results['ebm_s2_ppl']:.2f}")
    
    # Create visualization
    visualize_sequence_comparison(results, cfg.output_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Models trained for {cfg.max_steps} steps")
    print(f"Pure GPT final training perplexity: {ppl_gpt[-1]:.2f}")
    print(f"EBM final training perplexity: {ppl_ebm[-1]:.2f}")
    print(f"\nKey Insight: EBM with refinement should show lower perplexity")
    print(f"especially on harder/ambiguous sequences where 'thinking' helps.")
    print("="*60)


if __name__ == "__main__":
    main()