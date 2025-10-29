"""
Try to find ANY case where refinement actually changes predictions
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

    model_cfg = ModelConfig(**config_dict.get('model', {}))
    data_cfg = DataConfig(**config_dict.get('data', {}))
    train_cfg = TrainConfig(**config_dict.get('train', {}))
    cfg = Config(model=model_cfg, data=data_cfg, train=train_cfg)

    model = EBM(cfg.model).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, cfg, device

def find_impact():
    """Search for cases where refinement changes predictions."""

    # Load model
    checkpoint_path = "/Users/sdan/Developer/nanoebm/out_ebt/refine4.pt"
    model, cfg, device = load_model(checkpoint_path)

    # Load data
    val_loader, _ = get_loader(
        "shakespeare.txt",
        cfg.model.block_size,
        batch_size=16,
        split="val"
    )

    print("🔍 SEARCHING FOR REFINEMENT IMPACT")
    print("="*60)

    # Decode function
    itos = val_loader.dataset.itos
    decode = lambda ids: ''.join([itos[i] for i in ids])

    total_predictions = 0
    changed_predictions = 0
    examples = []

    # Test different refinement strengths
    print("\n1. Testing different refinement steps:")
    print("-" * 40)

    for num_batches in range(50):  # Check many batches
        x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _, logits_base, _ = model(x, targets=y, use_refine=False)

        for steps in [1, 2, 4, 8, 16, 32]:
            with torch.enable_grad():
                _, logits_refined, _ = model(x, targets=y, use_refine=True, refine_steps=steps)
                logits_refined = logits_refined.detach()

            preds_base = torch.argmax(logits_base, dim=-1)
            preds_refined = torch.argmax(logits_refined, dim=-1)

            changed = (preds_base != preds_refined).sum().item()
            total = preds_base.numel()

            if changed > 0:
                print(f"  ✨ Batch {num_batches}, Steps {steps}: {changed}/{total} predictions changed!")

                # Find specific examples
                for b in range(x.shape[0]):
                    for t in range(x.shape[1]):
                        if preds_base[b, t] != preds_refined[b, t]:
                            context = decode(x[b, max(0, t-10):t].cpu().tolist())
                            target = decode([y[b, t].cpu().item()])
                            pred_base = decode([preds_base[b, t].cpu().item()])
                            pred_refined = decode([preds_refined[b, t].cpu().item()])

                            prob_base = F.softmax(logits_base[b, t], dim=-1).max().item()
                            prob_refined = F.softmax(logits_refined[b, t], dim=-1).max().item()

                            examples.append({
                                'context': context,
                                'target': target,
                                'base': pred_base,
                                'refined': pred_refined,
                                'base_conf': prob_base,
                                'refined_conf': prob_refined,
                                'correct_before': pred_base == target,
                                'correct_after': pred_refined == target,
                                'steps': steps
                            })

                            if len(examples) <= 5:
                                print(f"\n    Example: '{context}' -> '{target}'")
                                print(f"      Base:    '{pred_base}' (conf: {prob_base:.2f}) {'✓' if pred_base == target else '✗'}")
                                print(f"      Refined: '{pred_refined}' (conf: {prob_refined:.2f}) {'✓' if pred_refined == target else '✗'}")

            total_predictions += total
            changed_predictions += changed

    print(f"\n2. OVERALL STATISTICS:")
    print("-" * 40)
    print(f"  Total predictions tested: {total_predictions}")
    print(f"  Predictions changed: {changed_predictions} ({100*changed_predictions/max(1,total_predictions):.3f}%)")

    if examples:
        # Analyze the changes
        improvements = sum(1 for ex in examples if not ex['correct_before'] and ex['correct_after'])
        degradations = sum(1 for ex in examples if ex['correct_before'] and not ex['correct_after'])
        neutral = len(examples) - improvements - degradations

        print(f"\n3. CHANGE ANALYSIS:")
        print("-" * 40)
        print(f"  Improvements (wrong→right): {improvements}")
        print(f"  Degradations (right→wrong): {degradations}")
        print(f"  Neutral changes: {neutral}")

        # Confidence analysis
        base_confs = [ex['base_conf'] for ex in examples]
        refined_confs = [ex['refined_conf'] for ex in examples]

        print(f"\n  Average confidence when predictions change:")
        print(f"    Base:    {np.mean(base_confs):.3f}")
        print(f"    Refined: {np.mean(refined_confs):.3f}")

    # Test with corrupted inputs
    print(f"\n4. TESTING WITH CORRUPTED INPUTS:")
    print("-" * 40)

    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)

    # Add noise
    noise_level = 0.3
    mask = torch.rand_like(x.float()) < noise_level
    x_noisy = x.clone()
    x_noisy[mask] = torch.randint_like(x[mask], 0, cfg.model.vocab_size)

    with torch.no_grad():
        _, logits_base_noisy, _ = model(x_noisy, targets=y, use_refine=False)

    with torch.enable_grad():
        _, logits_refined_noisy, _ = model(x_noisy, targets=y, use_refine=True, refine_steps=8)
        logits_refined_noisy = logits_refined_noisy.detach()

    preds_base_noisy = torch.argmax(logits_base_noisy, dim=-1)
    preds_refined_noisy = torch.argmax(logits_refined_noisy, dim=-1)

    changed_noisy = (preds_base_noisy != preds_refined_noisy).sum().item()
    total_noisy = preds_base_noisy.numel()

    # Check if refinement helps recover from noise
    acc_base_noisy = (preds_base_noisy == y).float().mean().item()
    acc_refined_noisy = (preds_refined_noisy == y).float().mean().item()

    print(f"  With {noise_level:.0%} corruption:")
    print(f"    Predictions changed: {changed_noisy}/{total_noisy} ({100*changed_noisy/total_noisy:.1f}%)")
    print(f"    Base accuracy: {acc_base_noisy:.3f}")
    print(f"    Refined accuracy: {acc_refined_noisy:.3f}")

    if acc_refined_noisy > acc_base_noisy:
        print(f"    ✅ Refinement improves noisy predictions by {100*(acc_refined_noisy-acc_base_noisy):.1f}%!")
    else:
        print(f"    ❌ No improvement from refinement")

    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

    if changed_predictions == 0:
        print("\n⚠️  REFINEMENT HAS NO IMPACT ON PREDICTIONS")
        print("The model learned to keep predictions unchanged during refinement.")
        print("This suggests the training didn't successfully teach the model")
        print("to use iterative refinement for better predictions.")
    else:
        print(f"\n✅ Found {changed_predictions} cases where refinement changes predictions")

if __name__ == "__main__":
    find_impact()