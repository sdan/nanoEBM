# nanoEBM

<img width="800" alt="arvsebm" src="https://github.com/user-attachments/assets/66d8432e-bb45-4d82-91e8-d83540ef37d6" />

key idea:

$$x_{t+1} = x_t - \alpha \nabla_x E(x) + \text{noise}$$
- **x**: current prediction (logits for text, embeddings for vision)
- **E(x)**: energy function (low energy = good prediction)
- **α**: learnable step size
- **∇_x E(x)**: gradient of energy w.r.t. prediction
- Repeat a few refinement steps (typically 2–4)

An energy-based model is a model trained to minimize a function. if we train a transformer to minize a function it learns to reason about the input implicitly as we iterate

this current implementation is character-level on shakespeare.txt with a block size of 256.

Training
```bash
uv sync

# default training
uv run python train.py model.use_contrastive=true

# enable W&B logging (optional)
uv run python train.py wandb_project=nanoebm
```
<img width="800" alt="Screenshot 2025-10-29 at 7 05 52 PM" src="https://github.com/user-attachments/assets/745d456c-ce9a-41be-b363-8ea75f52d540" />

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

# greedy
uv run python sample.py checkpoint=out_ebt/final.pt use_thinking=false
```
Acknowledgements to [minGPT](https://github.com/karpathy/minGPT)

Some good reading:

[Yann LeCun Energy-Based Models](https://atcold.github.io/NYU-DLSP20/en/week07/07-1/)

[EBT](https://alexiglad.github.io/blog/2025/ebt/)

[EBT paper](https://energy-based-transformers.github.io/static/pdfs/paper.pdf)

[EBT Tutotial](https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial8/Deep_Energy_Models.ipynb)

---------

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

# train super long
!uv run python train.py \
    model.n_layer=8 \
    model.n_head=8 \
    model.n_embd=512 \
    model.refine_steps=6 \
    model.dropout=0.2 \
    model.use_contrastive=true \
    model.contrastive_type=pcd \
    model.contrastive_k=10 \
    model.contrastive_weight=0.15 \
    model.n_persistent=500 \
    data.batch_size=128 \
    data.block_size=256 \
    train.max_steps=10000 \
    train.learning_rate=6e-4 \
    train.compile=true \
    wandb_project=nanoebm \
    wandb_name=h100_8L_512d_pcd10_shakespeare
```



<img width="800" alt="final_landscape" src="https://github.com/user-attachments/assets/4216efb5-655b-460b-91b1-d2a6ab29f1ec" />

<img width="800" alt="final_unified" src="https://github.com/user-attachments/assets/bb90420d-3887-468f-9e6b-a6853501c703" />


<img width="800" alt="Screenshot 2025-10-29 at 10 24 16 AM" src="https://github.com/user-attachments/assets/12f2d62d-3fcd-4f26-ba1e-e0c2d5d05dd0" />
