NanoEBM — minimal Energy-Based Transformer scaffolding for text reasoning.

What’s here
- Baseline GPT model and scripts (existing files: model.py, train.py, bench.py, sample.py)
- New nanoEBM scaffolding for a text-only Energy-Based Transformer:
  - Package: nanoebm/ with config, model skeleton, energy head, thinking API, and data helpers
  - Configs: configs/ebt_char_small.json (model/trainer/data/thinking settings)
  - Entry points: scripts/train_ebt.py, scripts/sample_ebt.py
  - configurator.py (for legacy scripts like train.py, bench.py)

Quick start (scaffold)
- Inspect config: `configs/ebt_char_small.json`
- Run scaffold train entry: `python scripts/train_ebt.py --config configs/ebt_char_small.json`
  - Prints parameter count and exits. Fill in the training loop next.

Prepare WikiText with GPT‑2 BPE (Hugging Face)
- Install: `pip install datasets transformers`
- Build memmaps: `python scripts/prepare_wikitext_gpt2.py --name wikitext-2-raw-v1 --out data/wikitext-gpt2`
- Use config: `configs/ebt_gpt2_small.json`

Design (high level)
- EBT model = context encoder (reuses GPT blocks) + verifier energy head
- Thinking loop stub in nanoebm/thinking.py for iterative refinement
- Small-character/byte vocab recommended first (exact normalization is easy)

Next steps to implement
- In nanoebm/model.py forward(): compute energy over all tokens and CE loss
- In scripts/train_ebt.py: optimizer, schedule, eval, checkpointing
- In scripts/sample_ebt.py: next-token distribution with optional thinking
# nanoEBM
