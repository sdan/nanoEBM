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

**Recommended**: Use the full OpenFineWeb dataset for production training. Shakespeare is included for quick testing.

### Option A: OpenFineWeb (Recommended - Full Dataset)

The FineWeb-Edu-100B dataset contains 1823 parquet shards (~100B tokens). Download with the included utility:

```bash
# Quick start: 10 shards (~550M tokens, ~6GB)
python download_data.py -n 10 -w 4

# Medium scale: 100 shards (~5.5B tokens, ~60GB)
python download_data.py -n 100 -w 8

# Full dataset: 1823 shards (~100B tokens, ~1TB)
python download_data.py -n -1 -w 8
```

Data downloads to `~/data/openfineweb` by default (configurable with `-d`).

### Option B: Shakespeare (Quick Testing)

Included in the repo for quick char-level testing:
- File: `shakespeare.txt`
- Size: ~1MB
- Tokens: ~1M characters

## Train

Both char- and BPE-tokenized training are supported. Checkpoints and metrics write to `out_ebt/run_*`.

### Recommended: OpenFineWeb with BPE Tokenization

- Baseline (no thinking)
  - ```bash
    uv run python train.py \
      data.data_path=~/data/openfineweb \
      data.tokenizer=gpt2 \
      data.max_shards=10 \
      model.mcmc_num_steps=0 \
      train.eval_interval=1000 \
      train.eval_iters=50 \
      train.max_steps=20000
    ```

- With thinking (iterative refinement)
  - ```bash
    uv run python train.py \
      data.data_path=~/data/openfineweb \
      data.tokenizer=gpt2 \
      data.max_shards=10 \
      model.mcmc_num_steps=2 \
      model.truncate_mcmc=True \
      model.no_mcmc_detach=False \
      train.max_steps=20000
    ```

### Quick Testing: Shakespeare (Char-level)

- Baseline (no thinking)
  - ```bash
    uv run python train.py
    ```

- With thinking
  - ```bash
    uv run python train.py \
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

### BPE (OpenFineWeb models)

- Basic generation
  - ```bash
    uv run python sample.py \
      checkpoint=out_ebt/final.pt \
      max_new_tokens=256 \
      prompt="The future of AI is"
    ```

- With thinking (iterative refinement)
  - ```bash
    uv run python sample.py \
      checkpoint=out_ebt/final.pt \
      use_thinking=true \
      think_steps=4 \
      topk=64 \
      sample=true \
      sample_temp=1.0 \
      sample_top_p=0.95 \
      prompt="In the year 2050,"
    ```

### Char-level (Shakespeare models)

- Basic generation
  - ```bash
    uv run python sample.py \
      checkpoint=out_ebt/final.pt \
      max_new_tokens=300 \
      prompt="HAMLET:"
    ```

- With thinking
  - ```bash
    uv run python sample.py \
      checkpoint=out_ebt/final.pt \
      use_thinking=true \
      think_steps=4 \
      topk=64 \
      prompt="ROMEO:"
    ```

Tips:
- `sample.py` will auto-pick the latest run under `out_ebt` if `checkpoint` is omitted
- BPE models require `tiktoken` (included in project deps)

## Weights & Biases (optional)

- Install and login
  - `uv pip install wandb`
  - `uv run wandb login`  (or set `WANDB_API_KEY=...`)

- Enable in training
  - ```bash
    uv run python train.py \
      data.data_path=~/data/openfineweb \
      data.tokenizer=gpt2 \
      data.max_shards=10 \
      wandb_project=nanoebm \
      wandb_name=ofw_baseline_k0
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
