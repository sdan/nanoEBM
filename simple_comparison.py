"""
Simple comparison: Does EBM refinement actually help on Shakespeare?
Let's find out with direct inference comparison!
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
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

def compare_predictions(model, loader, device, num_samples=10):
    """Compare actual predictions between Transformer and EBM mode."""

    print("\n" + "="*60)
    print("DIRECT PREDICTION COMPARISON")
    print("="*60)

    # Get decoder from dataset
    itos = loader.dataset.itos
    decode = lambda ids: ''.join([itos[i] for i in ids])

    results = {
        'transformer': {'losses': [], 'correct': 0, 'total': 0},
        'ebm': {'losses': [], 'correct': 0, 'total': 0},
        'examples': []
    }

    for i, (x, y) in enumerate(loader):
        if i >= num_samples:
            break

        x, y = x.to(device), y.to(device)

        # Run Transformer (no refinement) - no gradients needed
        with torch.no_grad():
            loss_t, logits_t, _ = model(x, targets=y, use_refine=False)

        # Run EBM (with refinement) - needs gradients for refinement
        with torch.enable_grad():
            loss_e, logits_e, _ = model(x, targets=y, use_refine=True, refine_steps=4)
            # Detach for further processing
            loss_e = loss_e.detach()
            logits_e = logits_e.detach()

        # Get predictions
        preds_t = torch.argmax(logits_t, dim=-1)
        preds_e = torch.argmax(logits_e, dim=-1)

        # Calculate accuracy
        correct_t = (preds_t == y).sum().item()
        correct_e = (preds_e == y).sum().item()
        total = y.numel()

        results['transformer']['losses'].append(loss_t.item())
        results['transformer']['correct'] += correct_t
        results['transformer']['total'] += total

        results['ebm']['losses'].append(loss_e.item())
        results['ebm']['correct'] += correct_e
        results['ebm']['total'] += total

        # Store interesting examples (first sequence of batch)
        if i < 3:  # Store first 3 examples
            # Take middle part of sequence for readability
            mid = x.shape[1] // 2
            context_ids = x[0, mid-10:mid].cpu().tolist()
            target_id = y[0, mid].cpu().item()
            pred_t_id = preds_t[0, mid].cpu().item()
            pred_e_id = preds_e[0, mid].cpu().item()

            # Get probabilities for top predictions
            probs_t = F.softmax(logits_t[0, mid], dim=-1)
            probs_e = F.softmax(logits_e[0, mid], dim=-1)

            example = {
                'context': decode(context_ids),
                'target': decode([target_id]),
                'transformer_pred': decode([pred_t_id]),
                'ebm_pred': decode([pred_e_id]),
                'transformer_conf': probs_t[pred_t_id].item(),
                'ebm_conf': probs_e[pred_e_id].item(),
                'transformer_correct': pred_t_id == target_id,
                'ebm_correct': pred_e_id == target_id
            }
            results['examples'].append(example)

    return results

def print_results(results):
    """Print comparison results in a clear format."""

    # Calculate metrics
    t_loss = np.mean(results['transformer']['losses'])
    e_loss = np.mean(results['ebm']['losses'])
    t_ppl = np.exp(t_loss)
    e_ppl = np.exp(e_loss)
    t_acc = results['transformer']['correct'] / results['transformer']['total']
    e_acc = results['ebm']['correct'] / results['ebm']['total']

    print("\n📊 AGGREGATE METRICS")
    print("-" * 40)
    print(f"                  Transformer    EBM-4")
    print(f"Loss:             {t_loss:.4f}      {e_loss:.4f}")
    print(f"Perplexity:       {t_ppl:.2f}       {e_ppl:.2f}")
    print(f"Accuracy:         {t_acc:.3f}      {e_acc:.3f}")
    print()

    # Calculate improvements
    ppl_improve = ((t_ppl - e_ppl) / t_ppl) * 100
    acc_improve = ((e_acc - t_acc) / t_acc) * 100

    if ppl_improve > 0:
        print(f"✅ Perplexity improved by {ppl_improve:.1f}%")
    else:
        print(f"❌ Perplexity worse by {-ppl_improve:.1f}%")

    if acc_improve > 0:
        print(f"✅ Accuracy improved by {acc_improve:.1f}%")
    else:
        print(f"❌ Accuracy worse by {-acc_improve:.1f}%")

    print("\n📝 EXAMPLE PREDICTIONS")
    print("-" * 40)

    for i, ex in enumerate(results['examples'], 1):
        print(f"\nExample {i}:")
        print(f"Context: ...{ex['context'][-30:]}")
        print(f"Target: '{ex['target']}'")
        print()

        # Transformer prediction
        symbol = "✓" if ex['transformer_correct'] else "✗"
        print(f"  Transformer: '{ex['transformer_pred']}' (conf: {ex['transformer_conf']:.2f}) {symbol}")

        # EBM prediction
        symbol = "✓" if ex['ebm_correct'] else "✗"
        print(f"  EBM-4:       '{ex['ebm_pred']}' (conf: {ex['ebm_conf']:.2f}) {symbol}")

        # Show improvement if any
        if not ex['transformer_correct'] and ex['ebm_correct']:
            print("  💡 EBM fixed this prediction!")
        elif ex['transformer_correct'] and not ex['ebm_correct']:
            print("  ⚠️  EBM broke this prediction")

def analyze_confidence_distribution(model, loader, device):
    """Check if refinement changes confidence distributions."""

    print("\n🎯 CONFIDENCE ANALYSIS")
    print("-" * 40)

    t_confidences = []
    e_confidences = []

    for i, (x, y) in enumerate(loader):
        if i >= 20:  # Sample 20 batches
            break

        x, y = x.to(device), y.to(device)

        # Get logits
        with torch.no_grad():
            _, logits_t, _ = model(x, targets=y, use_refine=False)

        with torch.enable_grad():
            _, logits_e, _ = model(x, targets=y, use_refine=True, refine_steps=4)
            logits_e = logits_e.detach()

        # Get probabilities
        probs_t = F.softmax(logits_t, dim=-1)
        probs_e = F.softmax(logits_e, dim=-1)

        # Get max confidence for each prediction
        max_conf_t = probs_t.max(dim=-1).values
        max_conf_e = probs_e.max(dim=-1).values

        t_confidences.extend(max_conf_t.cpu().numpy().flatten())
        e_confidences.extend(max_conf_e.cpu().numpy().flatten())

    # Statistics
    print(f"Average confidence:")
    print(f"  Transformer: {np.mean(t_confidences):.3f} ± {np.std(t_confidences):.3f}")
    print(f"  EBM-4:       {np.mean(e_confidences):.3f} ± {np.std(e_confidences):.3f}")

    # Check calibration
    print(f"\nHigh confidence (>0.9) predictions:")
    print(f"  Transformer: {100 * np.sum(np.array(t_confidences) > 0.9) / len(t_confidences):.1f}%")
    print(f"  EBM-4:       {100 * np.sum(np.array(e_confidences) > 0.9) / len(e_confidences):.1f}%")

    print(f"\nLow confidence (<0.3) predictions:")
    print(f"  Transformer: {100 * np.sum(np.array(t_confidences) < 0.3) / len(t_confidences):.1f}%")
    print(f"  EBM-4:       {100 * np.sum(np.array(e_confidences) < 0.3) / len(e_confidences):.1f}%")

def check_refinement_steps(model, loader, device):
    """See how performance changes with refinement steps."""

    print("\n🔄 REFINEMENT STEPS ANALYSIS")
    print("-" * 40)

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    print("Loss at each refinement step:")
    for steps in [0, 1, 2, 4, 8]:
        if steps == 0:
            with torch.no_grad():
                loss, _, _ = model(x, targets=y, use_refine=False)
        else:
            with torch.enable_grad():
                loss, _, _ = model(x, targets=y, use_refine=True, refine_steps=steps)
                loss = loss.detach()

        label = "Transformer" if steps == 0 else f"EBM-{steps}"
        print(f"  {label:12} Loss: {loss.item():.4f}  Perplexity: {np.exp(loss.item()):.2f}")

def main():
    print("\n🎭 SHAKESPEARE INFERENCE COMPARISON")
    print("Does EBM refinement actually help? Let's find out!\n")

    # Load model
    checkpoint_path = "/Users/sdan/Developer/nanoebm/out_ebt/refine4.pt"
    model, cfg, device = load_model(checkpoint_path)
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"✓ Using device: {device}")

    # Load validation data
    val_loader, _ = get_loader(
        "shakespeare.txt",
        cfg.model.block_size,
        batch_size=8,  # Small batch for detailed analysis
        split="val"
    )
    print(f"✓ Loaded Shakespeare validation set")

    # Run comparisons
    results = compare_predictions(model, val_loader, device, num_samples=50)
    print_results(results)

    # Additional analyses
    analyze_confidence_distribution(model, val_loader, device)
    check_refinement_steps(model, val_loader, device)

    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    t_ppl = np.exp(np.mean(results['transformer']['losses']))
    e_ppl = np.exp(np.mean(results['ebm']['losses']))

    if e_ppl < t_ppl:
        improve = ((t_ppl - e_ppl) / t_ppl) * 100
        print(f"✅ EBM with refinement IS better!")
        print(f"   Reduces perplexity by {improve:.1f}% on Shakespeare")
    else:
        worse = ((e_ppl - t_ppl) / t_ppl) * 100
        print(f"❌ EBM with refinement is NOT better")
        print(f"   Perplexity is {worse:.1f}% worse than plain Transformer")

    print("\nThis makes sense because refinement adds computational")
    print("cost, so it better improve performance to be worth it!")
    print("="*60)

if __name__ == "__main__":
    main()