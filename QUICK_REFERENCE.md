# this repo/reference is for the repo in  /Users/sdan/Developer/EBT/QUICK_REFERENCE.md
# Energy-Based Transformers (EBT) - Quick Reference


## The Core Idea (One Sentence)
Treat next token/embedding prediction as an optimization problem: iteratively refine noisy predictions by descending gradients of an energy function.

## Key Equation
```
x_{t+1} = x_t - α ∇_x E(x) + noise
```
- **x**: current prediction (logits for NLP, embeddings for vision)
- **E(x)**: energy function (low energy = good prediction)
- **α**: learnable step size
- **∇_x E(x)**: gradient of energy w.r.t. prediction
- Repeat K times (typically K=2-4)

## Architecture
```
[Real Embeddings | Predicted Embeddings] 
        ↓
[Transformer blocks with special attention]
        ↓
[RMSNorm + Linear(1)]
        ↓
Single Scalar Energy Value
```

## Three Loss Types
1. **NLP**: Cross-entropy on predicted tokens
2. **Vision/Video**: Regression (MSE/L1) on embeddings
3. **Optional**: Contrastive loss (discriminate real vs fake)

## Key Hyperparameters
| Param | Value | Notes |
|-------|-------|-------|
| `mcmc_step_size` | ~500 | Main tuning parameter |
| `mcmc_num_steps` | 2-4 | More steps = more thinking |
| `mcmc_step_size_learnable` | True | Let α adapt |
| `peak_learning_rate` | varies | Depends on model size |
| `langevin_dynamics_noise` | 0-1 | Optional exploration |

## Training Flow (Pseudocode)
```python
for batch in dataloader:
    # Forward pass: corrupt → iterative refinement
    predictions = []
    energies = []
    
    pred = corrupt(batch)  # random noise
    
    for step in range(mcmc_num_steps):
        pred = pred.detach().requires_grad_()
        energy = transformer(cat([real, pred]))
        grad = ∇energy
        pred = pred - alpha * grad
        
        predictions.append(pred)
        energies.append(energy)
    
    # Compute loss
    loss = 0
    for i, (pred_dist, target) in enumerate(zip(predictions, targets)):
        loss += cross_entropy(pred_dist, target)
    loss /= mcmc_num_steps
    
    loss.backward()
    optimizer.step()
```

## Inference Flow (System 2 Advanced)
```python
# Generate multiple candidates
samples = []
for candidate_idx in range(num_samples):
    pred = corrupt(input)
    for step in range(infer_steps):
        energy = transformer(pred)
        pred = pred - alpha * ∇energy
    samples.append((pred, energy))

# Select best by energy
best = samples[argmin(energies)]
return best
```

## File Map (Most Important)
```
train_model.py          ← Entry point (args, setup, launcher)
base_model_trainer.py   ← Training loop (PyTorch Lightning)
model/nlp/ebt.py        ← EBT_NLP class (forward, loss)
model/ar_ebt_default.py ← Transformer architecture
optimization.py         ← LR scheduler, optimizer setup
```

## Model Sizes
```
Small:   12 layers,  12 heads,  768 dim    → LR=0.0006
Medium:  24 layers,  16 heads, 1024 dim    → LR=0.0003
Large:   24 layers,  16 heads, 1536 dim    → LR=0.00025
XL:      24 layers,  32 heads, 2048 dim    → LR=0.0002
```

## Differences from Standard Transformers

| Aspect | Standard | EBT |
|--------|----------|-----|
| **How** | Single forward pass | K iterative refinement steps |
| **Why** | Direct probability | Energy optimization |
| **Speed** | 1x | ~(K+1)x slower |
| **Memory** | Moderate | Higher (compute gradients) |
| **Quality** | Good on seen data | Better on OOD/reasoning |

## Debugging Checklist
- [ ] NaN gradients? → clamp gradient magnitudes
- [ ] Divergence? → reduce mcmc_step_size
- [ ] Slow convergence? → increase mcmc_step_size or mcmc_num_steps
- [ ] OOM? → reduce batch size or mcmc_num_steps
- [ ] Not learning α? → check mcmc_step_size_learnable=True + alpha in optimizer

## When to Use Each Variant
- **Default** (`ar_ebt_default.py`): Most cases, simple and stable
- **AdaLN** (`ar_ebt_adaln.py`): When attention/norm vary by step
- **TimeEmbed** (`ar_ebt_time_embed.py`): Paper's best variant, slightly more params

## Related Concepts
- **EBM**: Energy-Based Model (implicit distribution via E)
- **MCMC**: Markov Chain Monte Carlo (iterative sampling)
- **Langevin Dynamics**: Gradient descent + noise
- **Contrastive Learning**: Pull correct, push wrong (optional)
- **System 2**: Slow, deliberate thinking (vs System 1: fast, reflex)

## Paper Reference
```
Energy-Based Transformers are Scalable Learners and Thinkers
Alexi Gladstone et al.
arXiv:2507.02092
https://energy-based-transformers.github.io/
```

## Quick Start
```bash
# Install
conda create -n ebt python=3.12
pip install -r requirements.txt
wandb login

# Run minimal example (won't reproduce paper results)
python example_code/minimal_nlp_training_loop.py

# Run full training
bash job_scripts/nlp/pretrain/ebt_s1.sh
```

---

## Energy-Based Model Theory (30-Second Version)

Standard transformers maximize P(x) directly. EBTs work with an implicit distribution:
```
P(x) ∝ exp(-E(x))
```

Lower energy = higher probability. Training pushes E to be low for good completions, high for bad ones. Inference descends E to find good completions. This enables iterative "thinking."

**Why it works**: 
- Gradient ∇E gives direction to improve predictions
- Multiple steps allow complex refinements
- Learnable α adapts to task complexity
