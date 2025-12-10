"""
Evaluate a plain GPT-style LM baseline on GSM8K.

This loads a checkpoint from train_gsm8k_lm.py and computes exact-match
accuracy on GSM8K, using the same prompting and answer extraction logic
as eval_gsm8k.py but without any energy head or refinement.

Usage (example):
    uv run python eval_gsm8k_lm.py \
        checkpoint=out_ebt/gsm8k_lm_*/final.pt \
        max_examples=512
"""

from __future__ import annotations

import glob
import os
import time

import chz
import torch
from datasets import load_dataset
import tiktoken

from nanoebm.config import ModelConfig
from nanoebm.transformer import Transformer
from eval_gsm8k import (
    EvalConfig,
    build_prompt,
    extract_final_answer,
    extract_pred_answer,
)


def find_latest_gsm8k_lm_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest GSM8K LM run checkpoint by run directory name."""
    run_dirs = glob.glob(os.path.join(base_dir, "gsm8k_lm_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No gsm8k_lm_* run directories found in {base_dir}")
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]
    final = os.path.join(latest_dir, "final.pt")
    if os.path.exists(final):
        return final
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {latest_dir}")
    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]


def main(cfg: EvalConfig):
    # Device
    if cfg.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = cfg.device
    print(f"Using device: {device}")

    # Checkpoint
    checkpoint = cfg.checkpoint or find_latest_gsm8k_lm_checkpoint()
    print(f"Loading LM checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    model_cfg = ModelConfig(**cfg_dict["model"])
    data_cfg = cfg_dict.get("data", {})

    encoding_name = data_cfg.get("bpe_encoding", "gpt2")
    enc = tiktoken.get_encoding(encoding_name)
    print(f"Using BPE encoding: {encoding_name} (vocab_size={enc.n_vocab})")

    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load GSM8K split
    ds = load_dataset("gsm8k", "main", split=cfg.split)
    n_total = len(ds)
    n_eval = n_total if cfg.max_examples is None else min(cfg.max_examples, n_total)
    print(f"Evaluating LM on GSM8K[{cfg.split}] with {n_eval}/{n_total} examples")

    correct = 0
    times = []

    for i, ex in enumerate(ds):
        if i >= n_eval:
            break

        question = ex["question"]
        gt_solution = ex["answer"]
        gt_ans = extract_final_answer(gt_solution)

        prompt = build_prompt(question)
        prompt_tokens = enc.encode(prompt)
        idx_prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        t0 = time.time()
        with torch.no_grad():
            idx = idx_prompt.clone()
            for _ in range(cfg.max_new_tokens):
                idx_cond = idx if idx.size(1) <= model_cfg.block_size else idx[:, -model_cfg.block_size :]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]

                if cfg.topk is not None:
                    v, _ = torch.topk(logits, min(cfg.topk, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                if cfg.sample:
                    logits = logits / max(cfg.temperature, 1e-8)
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                idx = torch.cat((idx, idx_next), dim=1)
        t1 = time.time()
        times.append(t1 - t0)

        decoded = enc.decode(idx[0].tolist())
        pred_ans = extract_pred_answer(decoded, prompt)
        if pred_ans == gt_ans:
            correct += 1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{n_eval}] LM_acc={correct/(i+1):.3f}")

    acc = correct / n_eval
    avg_t = sum(times) / len(times) if times else 0.0

    print("\n" + "=" * 80)
    print("GSM8K LM evaluation results")
    print("=" * 80)
    print(f"Examples evaluated:   {n_eval}")
    print(f"LM accuracy:          {acc:.4f}")
    print(f"Avg time / query:     {avg_t*1000:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    config = chz.entrypoint(EvalConfig)
    main(config)

