# nanoEBM

key idea:
```
x_{t+1} = x_t - α ∇_x E(x) + noise
```
- **x**: current prediction (logits for NLP, embeddings for vision)
- **E(x)**: energy function (low energy = good prediction)
- **α**: learnable step size
- **∇_x E(x)**: gradient of energy w.r.t. prediction
- Repeat K times (typically K=2-4)

an energy-based model is a model trained to minimize a function. if we train a transformer to minize a function it learns to reason about the input implicitly as we iterate

