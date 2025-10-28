# nanoEBM

key idea:
```
x_{t+1} = x_t - α ∇_x E(x) + noise
```
- **x**: current prediction (logits for text, embeddings for vision)
- **E(x)**: energy function (low energy = good prediction)
- **α**: learnable step size
- **∇_x E(x)**: gradient of energy w.r.t. prediction
- Repeat K times (typically K=2-4)

an energy-based model is a model trained to minimize a function. if we train a transformer to minize a function it learns to reason about the input implicitly as we iterate

this current implementation is character-level on shakespeare.txt with a block size of 256.

commands to run:
```bash
uv sync
uv run python train.py model.mcmc_num_steps=2 model.truncate_mcmc=True model.no_mcmc_detach=False model.mcmc_step_size_learnable=True


uv run python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64 sample=true sample_temp=1.2 sample_top_p=0.9
```