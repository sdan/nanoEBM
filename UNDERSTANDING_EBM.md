# Understanding nanoEBM: From minGPT to Energy-Based Language Models

## The Foundation: minGPT

This implementation starts with minGPT - a minimal GPT implementation. The core is a standard transformer that:
- Takes text tokens as input
- Passes them through attention layers
- Outputs logits (scores) for next token prediction

Think of this as the "base brain" that understands patterns in text.

## What Are Energy-Based Models (EBMs)?

Instead of directly predicting "what comes next", EBMs ask a different question: **"How good is this text?"**

The key idea:
- Define an energy function E(x, y) where low energy = good text
- To generate text, find the y that minimizes energy given context x
- Use gradient descent to iteratively improve predictions

## The nanoEBM Architecture

This implementation adds an energy layer on top of minGPT:

```
Text → Transformer (minGPT) → Hidden States → Energy Head → Energy Values
```

The Energy Head is a learned function that assigns energy scores to each possible next token.

## System 1 vs System 2 Thinking

The clever innovation here is implementing two modes of thinking:

### System 1: Fast, Intuitive
- Direct readout: `logits = -energy`
- One forward pass, like regular GPT
- Quick but sometimes wrong

### System 2: Slow, Deliberative  
- Iterative refinement using gradient descent
- Start with System 1's guess
- Repeatedly adjust logits to minimize expected energy:
  ```
  x_{t+1} = x_t - α ∇_x E(x) + noise
  ```
- Takes 2-4 steps typically
- More compute but better quality

## How Refinement Works

The refinement process is like "thinking harder" about the answer:

1. **Initialize**: Start with System 1's quick guess
2. **Compute Energy**: Use the energy head to score all possibilities
3. **Calculate Expected Energy**: `E_expected = sum(softmax(logits) * energies)`
4. **Gradient Step**: Move logits in direction that reduces expected energy
5. **Repeat**: Do this 2-4 times

The key insight: the energy function is **fixed** during refinement - we're only adjusting our prediction (logits) to find lower energy states.

## Training Process

During training:
- Both System 1 and System 2 predictions are computed
- Loss is based on System 2's refined predictions
- The model learns an energy landscape where correct text has low energy
- Gradients flow through the entire refinement process

## Why This Matters

1. **Better Quality**: System 2 can correct System 1's mistakes
2. **Interpretable**: Energy gap shows how much "thinking" helped
3. **Flexible**: Can trade compute for quality at inference time
4. **Theoretically Grounded**: Based on energy-based model theory

## Key Implementation Details

- **Energy Head**: Separate from language model head (no weight tying)
- **Step Size (α)**: Small (0.02) for stable gradient descent  
- **Langevin Noise**: Small noise (0.005) for exploration during training
- **Warmup**: Train System 1 first, then add refinement

## The Core Equation

The heart of the system is minimizing expected energy under the current distribution:

```
minimize E_p[E(x,y)] where p = softmax(logits)
```

This is solved via gradient descent on the logits while keeping the energy function fixed.

## Practical Usage

```bash
# Training
python train.py model.refine_steps=4

# Generation with thinking
python sample.py use_thinking=true think_steps=4

# Fast generation (System 1 only)  
python sample.py use_thinking=false
```

The model can dynamically choose how hard to think based on the task difficulty.