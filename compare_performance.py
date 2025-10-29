"""
Compare performance between System 1 (regular GPT) and System 2 (EBM with refinement)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
from nanoebm.model import EBM
from nanoebm.data import get_loader
from nanoebm.config import ModelConfig
import chz

def evaluate_model(model, loader, device, num_batches=50, use_refine=True, refine_steps=4):
    """Evaluate model performance on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    metrics = {
        'nll': 0,
        'perplexity': 0,
        'energy_gap': 0,
        'initial_energy': 0,
        'final_energy': 0,
    }
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Get model outputs
            loss, logits, batch_metrics = model(x, targets=y, use_refine=use_refine, refine_steps=refine_steps)
            
            # Accumulate metrics
            batch_size = x.size(0) * x.size(1)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size
            
            # Track additional metrics
            for key in metrics:
                if key in batch_metrics:
                    metrics[key] += batch_metrics[key] * batch_size
    
    # Average metrics
    avg_loss = total_loss / total_tokens
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    for key in metrics:
        metrics[key] /= num_batches
    
    return avg_loss, avg_perplexity, metrics

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_path = "/Users/sdan/Developer/nanoebm/out_ebt/refine4.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    
    # Create config from checkpoint
    cfg = ModelConfig()
    cfg = chz.replace(cfg, **config_dict)
    
    # Load dataset
    val_loader, val_ds = get_loader(
        "shakespeare.txt",
        cfg.model.block_size,
        cfg.train.batch_size,
        "val"
    )
    
    # Initialize model and load weights
    model = EBM(cfg.model).to(device)
    model.load_state_dict(checkpoint['model'])
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: System 1 vs System 2")
    print("="*60)
    
    # Evaluate System 1 (no refinement - regular GPT)
    print("\n🧠 System 1 (Regular GPT - No Refinement):")
    print("-" * 40)
    loss_s1, ppl_s1, metrics_s1 = evaluate_model(
        model, val_loader, device, 
        use_refine=False
    )
    print(f"  Loss:       {loss_s1:.4f}")
    print(f"  Perplexity: {ppl_s1:.2f}")
    print(f"  Energy:     {metrics_s1['initial_energy']:.4f}")
    
    # Evaluate System 2 with different refinement steps
    for steps in [2, 4, 8]:
        print(f"\n🔄 System 2 (EBM with {steps} refinement steps):")
        print("-" * 40)
        loss_s2, ppl_s2, metrics_s2 = evaluate_model(
            model, val_loader, device,
            use_refine=True,
            refine_steps=steps
        )
        
        # Calculate improvements
        ppl_improvement = ((ppl_s1 - ppl_s2) / ppl_s1) * 100
        loss_improvement = ((loss_s1 - loss_s2) / loss_s1) * 100
        
        print(f"  Loss:       {loss_s2:.4f} (↓ {loss_improvement:.1f}%)")
        print(f"  Perplexity: {ppl_s2:.2f} (↓ {ppl_improvement:.1f}%)")
        print(f"  Energy:     {metrics_s2['final_energy']:.4f}")
        print(f"  Energy Gap: {metrics_s2['energy_gap']:.4f} (S1 - S2)")
        
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("-" * 40)
    print("• System 1 is equivalent to a regular GPT model")
    print("• System 2 uses gradient descent to refine predictions")
    print("• Lower perplexity = better prediction quality")
    print("• Energy gap shows how much 'thinking' helps")
    print("• More refinement steps generally improve performance")
    print("  (with diminishing returns)")
    print("="*60)

if __name__ == "__main__":
    main()