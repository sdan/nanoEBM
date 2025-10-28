# nanoEBM

Energy-based Transformer with optional “thinking” (per-step refinement).

This README covers install, data download, training (char/BPE), and sampling.

## Install

- With uv (recommended)
  - `uv sync`
  - Optional: `uv pip install wandb` (for logging)

- With pip
  - `pip install -e .`

Project deps include `torch`, `numpy`, `matplotlib`, `tiktoken`, `pyarrow` in `pyproject.toml`.

## Data

You can train on the included Shakespeare text (char-level), or a small subset of OpenFineWeb (BPE).

- Option A: Shakespeare (already in repo)
  - File: `shakespeare.txt`

- Option B: OpenFineWeb subset (Parquet + BPE)
  - ```bash
    mkdir -p ~/data/fineweb_small && cd ~/data/fineweb_small
    BASE=https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main
    for i in $(seq -f %05g 0 3); do curl -L -o shard_${i}.parquet "$BASE/shard_${i}.parquet"; done
    ```

## Train

Both char- and BPE-tokenized training are supported. Checkpoints and metrics write to `out_ebt/run_*`.

- Char-level (baseline, no thinking)
  - ```bash
    uv run python train.py 
    ```

- Char-level (with thinking)
  - ```bash
    uv run python train.py \
      model.mcmc_num_steps=2 \
      model.truncate_mcmc=True \
      model.no_mcmc_detach=False
    ```

- BPE (OpenFineWeb subset, baseline)
  - ```bash
    uv run python train.py \
      data.data_path=~/data/fineweb_small \
      data.tokenizer=gpt2 \
      data.max_shards=4 \
      model.mcmc_num_steps=0 \
      train.eval_interval=1000 \
      train.eval_iters=50 \
      train.max_steps=20000
    ```

- BPE (with thinking)
  - ```bash
    uv run python train.py \
      data.data_path=~/data/fineweb_small \
      data.tokenizer=gpt2 \
      data.max_shards=4 \
      model.mcmc_num_steps=2 \
      model.truncate_mcmc=True \
      model.no_mcmc_detach=False
    ```

Notes
- The trainer logs: loss, perplexity, and bits metrics:
  - `bpb` (bits-per-char) for char tokenizer
  - `bpt` (bits-per-token) for BPE
- Validation reports `val_perplexity` and `val_bpb`/`val_bpt`.

## Sample

`sample.py` auto-detects char vs BPE from the checkpoint config. Use `tokenizer`/`bpe_encoding` to override if needed.

- Char-level
  - ```bash
    uv run python sample.py checkpoint=out_ebt/final.pt max_new_tokens=300 prompt="HAMLET:"
    ```

- BPE (auto-detected; requires `tiktoken`)
  - ```bash
    uv run python sample.py checkpoint=out_ebt/final.pt max_new_tokens=128 prompt="The future of AI is"
    ```

- Thinking mode (iterative refinement)
  - Greedy refinement only:
    ```bash
    uv run python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64
    ```
  - With sampling controls (temperature / top-p):
    ```bash
    uv run python sample.py \
      checkpoint=out_ebt/final.pt \
      use_thinking=true think_steps=4 topk=64 \
      sample=true sample_temp=1.2 sample_top_p=0.9
    ```

Tips
- `sample.py` will auto-pick the latest run under `out_ebt` if `checkpoint` is omitted.
- For BPE, ensure `tiktoken` is installed (it is listed in project deps).

## Weights & Biases (optional)

- Install and login
  - `uv pip install wandb`
  - `uv run wandb login`  (or set `WANDB_API_KEY=...`)
- Enable in training
  - ```bash
    uv run python train.py \
      data.data_path=~/data/fineweb_small data.tokenizer=gpt2 \
      wandb_project=nanoebm wandb_name=baseline_k0
    ```

## Outputs & Checkpoints

- Each run writes to `out_ebt/run_YYYYmmdd_HHMMSS/`:
  - `metrics.jsonl`, periodic checkpoints (`ckpt_step_*.pt`), and `final.pt`.
- `sample.py` can load any checkpoint or auto-detect the latest `final.pt`.

## Visualization (optional, char-focused)

Some visualizations in `viz.py` assume a char-level dataset for decoding.

- Example (top-k energies for the next token):
  - ```bash
    uv run python viz.py --checkpoint=out_ebt/final.pt --prompt="ROMEO:" --topk=20
    ```

---

Have ideas or want a BPE-friendly viz path? Open an issue or ping in chat.
