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

```bash
# Train with contrastive divergence (CD-1) - shapes energy landscape with negative samples
uv run python train.py model.use_contrastive=true

# Train with persistent contrastive divergence (PCD) - maintains sample chains across batches
uv run python train.py model.use_contrastive=true model.contrastive_type=pcd

# Train with Fast PCD - best for exploration with momentum-based sampling
uv run python train.py model.use_contrastive=true model.contrastive_type=fast_pcd

# Full example with custom contrastive settings
uv run python train.py \
  model.use_contrastive=true \
  model.contrastive_type=cd \
  model.contrastive_k=5 \
  model.contrastive_weight=0.2 \
  model.langevin_step_size=0.01 \
  model.langevin_noise_scale=0.005

# Basic contrastive divergence test
uv run python train.py \
  model.use_contrastive=true \
  model.contrastive_weight=0.1 \
  train.max_steps=100

# PCD for better mixing
uv run python train.py \
  model.use_contrastive=true \
  model.contrastive_type=pcd \
  model.contrastive_weight=0.2 \
  train.max_steps=100

# Conservative CD settings
uv run python train.py \
  model.use_contrastive=true \
  model.contrastive_weight=0.01 \
  model.contrastive_k=1 \
  train.max_steps=100
```
![final_landscape](https://github.com/user-attachments/assets/4216efb5-655b-460b-91b1-d2a6ab29f1ec)
![arvsebm](https://github.com/user-attachments/assets/66d8432e-bb45-4d82-91e8-d83540ef37d6)
<img width="2234" height="740" alt="final_unified" src="https://github.com/user-attachments/assets/bb90420d-3887-468f-9e6b-a6853501c703" />



<img width="1140" height="623" alt="Screenshot 2025-10-29 at 7 05 52 PM" src="https://github.com/user-attachments/assets/745d456c-ce9a-41be-b363-8ea75f52d540" />
<img width="871" height="661" alt="Screenshot 2025-10-29 at 10 24 16 AM" src="https://github.com/user-attachments/assets/12f2d62d-3fcd-4f26-ba1e-e0c2d5d05dd0" />
