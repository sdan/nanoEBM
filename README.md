# nanoEBM

key idea:
```
x_{t+1} = x_t - α ∇_x E(x) + noise
```
- **x**: current prediction (logits for text, embeddings for vision)
- **E(x)**: energy function (low energy = good prediction)
- **α**: learnable step size
- **∇_x E(x)**: gradient of energy w.r.t. prediction
- Repeat a few refinement steps (typically 2–4)

an energy-based model is a model trained to minimize a function. if we train a transformer to minize a function it learns to reason about the input implicitly as we iterate

this current implementation is character-level on shakespeare.txt with a block size of 256.

Quick commands

Training
```bash
uv sync

# default training
uv run python train.py

# train with refinement and learnable step size
uv run python train.py \
  model.think_steps=2 \
  model.truncate_refine=true \
  model.detach_refine=true \
  model.think_lr_learnable=true

# enable W&B logging (optional)
uv run python train.py wandb_project=nanoebm
```

Sampling
```bash
# generate with thinking refinement (default)
uv run python sample.py checkpoint=out_ebt/final.pt max_new_tokens=500 prompt="HAMLET:"

# refine 4 steps and restrict to top-k
uv run python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64

# refine + sample with temperature and nucleus sampling
uv run python sample.py \
  checkpoint=out_ebt/final.pt \
  use_thinking=true think_steps=4 topk=64 \
  sample=true sample_temp=1.2 sample_top_p=0.9

# disable refinement (greedy via energy argmin)
uv run python sample.py checkpoint=out_ebt/final.pt use_thinking=false
```

Common overrides
```bash
# data
data.data_path=shakespeare.txt
data.batch_size=64

# model/refinement
model.softmax_temperature=1.0
model.entropy_weight=0.1
model.aux_ce_weight=0.1
model.think_steps=2
model.think_lr=0.5
model.think_max_move=0.25
model.mixture_stopgrad=true

# train
train.max_steps=5000
train.learning_rate=3e-4
```
