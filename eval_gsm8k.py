"""
Evaluate nanoEBM on GSM8K with System 1 vs System 2.

We:
- Load a trained GSM8K checkpoint (from train_gsm8k.py).
- For each problem in the test split, prompt:
      "Question: {problem}\\nAnswer:"
- Generate an answer with:
      * System 1 (no refinement)
      * System 2 (refinement with think_steps > 0)
- Extract the final numeric answer from "#### ..." and compute exact-match
  accuracy for each system, plus basic timing and energy stats.

Usage (example):
    uv run python eval_gsm8k.py \
        checkpoint=out_ebt/gsm8k_*/final.pt \
        max_examples=512 \
        think_steps=4
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import chz
import torch
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken

from nanoebm.config import ModelConfig
from nanoebm.model import EBM


def find_latest_gsm8k_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest GSM8K run checkpoint by run directory name."""
    run_dirs = glob.glob(os.path.join(base_dir, "gsm8k_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No gsm8k_* run directories found in {base_dir}")
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]
    final = os.path.join(latest_dir, "final.pt")
    if os.path.exists(final):
        return final
    # Fallback: any .pt file in the latest directory
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {latest_dir}")
    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]


@chz.chz
class EvalConfig:
    """Configuration for GSM8K evaluation."""

    checkpoint: str | None = None  # None => auto-detect latest gsm8k_* run
    split: str = "test"  # GSM8K has 'train' and 'test'
    max_examples: int | None = 256  # Limit for quick runs; None = full split
    max_new_tokens: int = 128  # Answer generation budget

    # Refinement settings
    think_steps: int = 4  # System 2 refinement steps
    refine_scope: str = "all"  # 'all' or 'answer' (answer-only refinement)

    # Decoding settings
    topk: int | None = 64
    temperature: float = 0.7  # Only used if sampling
    sample: bool = False  # Greedy by default for evaluation

    # Misc
    device: str | None = None  # None = auto


def build_prompt(question: str) -> str:
    """Standard text prompt used for both training and evaluation."""
    q = question.strip()
    return f"Question: {q}\nAnswer:"


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a GSM8K-style solution string.

    Prefer a line starting with '####', otherwise fall back to the last
    non-empty line. We keep the logic simple but consistent between
    ground truth and predictions.
    """
    import re

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Search for '#### answer' lines from the end
    for line in reversed(lines):
        if line.startswith("####"):
            candidate = line.lstrip("#").strip()
            return canonicalize_answer(candidate)
    if lines:
        return canonicalize_answer(lines[-1])
    return ""


def canonicalize_answer(s: str) -> str:
    """Normalize answer strings for rough exact-match comparison."""
    import re

    s = s.strip()
    # Drop trailing period
    if s.endswith("."):
        s = s[:-1].strip()
    # Remove commas for numbers like 1,000
    s = s.replace(",", "")
    # If there's an '=', keep the RHS
    if "=" in s:
        s = s.split("=", 1)[-1].strip()
    # If the string contains a number, keep the last one
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", s)
    if nums:
        return nums[-1]
    # Fallback: lowercase text
    return s.lower()


def extract_pred_answer(full_text: str, prompt: str) -> str:
    """Extract just the answer portion given the full decoded sequence."""
    if full_text.startswith(prompt):
        answer_part = full_text[len(prompt) :].strip()
    else:
        # Fallback: look for literal 'Answer:' substring
        idx = full_text.find("Answer:")
        if idx != -1:
            answer_part = full_text[idx + len("Answer:") :].strip()
        else:
            answer_part = full_text
    return extract_final_answer(answer_part)


def compute_expected_energy(model: EBM, idx: torch.Tensor) -> float:
    """Expected energy under System 1 logits for a given token sequence."""
    # Respect model block size (crop from the end if needed)
    if idx.size(1) > model.config.block_size:
        idx = idx[:, -model.config.block_size :]
    with torch.no_grad():
        h = model.get_hidden_states(idx)  # (B, T, n_embd)
        energies = model.energy_head(h)  # (B, T, V)
        logits_s1 = -energies
        probs_s1 = F.softmax(logits_s1, dim=-1)
        expected_energy = (probs_s1 * energies).sum(dim=-1).mean()
    return float(expected_energy.item())


def compute_energy_gap(model: EBM, idx: torch.Tensor, think_steps: int) -> Tuple[float, float]:
    """Compute System 1 and System 2 expected energy for a sequence."""
    if idx.size(1) > model.config.block_size:
        idx = idx[:, -model.config.block_size :]
    with torch.no_grad():
        h = model.get_hidden_states(idx)
        energies = model.energy_head(h)
        logits_s1 = -energies
        probs_s1 = F.softmax(logits_s1, dim=-1)
        ee_s1 = (probs_s1 * energies).sum(dim=-1).mean()

        logits_s2 = model.system2_refine(idx, steps=think_steps, use_soft_tokens=model.config.use_soft_tokens)
        probs_s2 = F.softmax(logits_s2, dim=-1)
        ee_s2 = (probs_s2 * energies).sum(dim=-1).mean()

    return float(ee_s1.item()), float(ee_s2.item())


def main(cfg: EvalConfig):
    # Device
    if cfg.device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = cfg.device
    print(f"Using device: {device}")

    # Checkpoint
    checkpoint = cfg.checkpoint or find_latest_gsm8k_checkpoint()
    print(f"Loading checkpoint: {checkpoint}")

    # Load checkpoint config and model
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_dict = ckpt["config"]
    model_cfg = ModelConfig(**cfg_dict["model"])
    data_cfg = cfg_dict.get("data", {})

    encoding_name = data_cfg.get("bpe_encoding", "gpt2")
    enc = tiktoken.get_encoding(encoding_name)
    print(f"Using BPE encoding: {encoding_name} (vocab_size={enc.n_vocab})")

    model = EBM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load GSM8K split
    ds = load_dataset("gsm8k", "main", split=cfg.split)
    n_total = len(ds)
    n_eval = n_total if cfg.max_examples is None else min(cfg.max_examples, n_total)
    print(f"Evaluating on GSM8K[{cfg.split}] with {n_eval}/{n_total} examples")

    correct_s1 = 0
    correct_s2 = 0
    energy_gaps = []
    times_s1 = []
    times_s2 = []

    for i, ex in enumerate(ds):
        if i >= n_eval:
            break

        question = ex["question"]
        gt_solution = ex["answer"]
        gt_ans = extract_final_answer(gt_solution)

        prompt = build_prompt(question)
        prompt_tokens = enc.encode(prompt)
        idx_prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        answer_start = len(prompt_tokens)

        # System 1 (no refinement)
        t0 = time.time()
        with torch.no_grad():
            out_s1 = model.generate(
                idx_prompt.clone(),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.topk,
                use_thinking=False,
                think_steps=0,
                sample=cfg.sample,
            )
        t1 = time.time()
        times_s1.append(t1 - t0)

        decoded_s1 = enc.decode(out_s1[0].tolist())
        pred_ans_s1 = extract_pred_answer(decoded_s1, prompt)
        if pred_ans_s1 == gt_ans:
            correct_s1 += 1

        # System 2 (with refinement)
        t2 = time.time()
        with torch.no_grad():
            out_s2 = model.generate(
                idx_prompt.clone(),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.topk,
                use_thinking=True,
                think_steps=cfg.think_steps,
                sample=cfg.sample,
                answer_start=answer_start if cfg.refine_scope == "answer" else None,
                refine_scope=cfg.refine_scope,
            )
        t3 = time.time()
        times_s2.append(t3 - t2)

        decoded_s2 = enc.decode(out_s2[0].tolist())
        pred_ans_s2 = extract_pred_answer(decoded_s2, prompt)
        if pred_ans_s2 == gt_ans:
            correct_s2 += 1

        # Energy gap on System 2 prediction sequence
        idx_s2 = torch.tensor(enc.encode(decoded_s2), dtype=torch.long, device=device).unsqueeze(0)
        ee_s1, ee_s2 = compute_energy_gap(model, idx_s2, cfg.think_steps)
        energy_gaps.append(ee_s1 - ee_s2)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{n_eval}] "
                f"S1_acc={correct_s1/(i+1):.3f} "
                f"S2_acc={correct_s2/(i+1):.3f} "
                f"Δacc={correct_s2/(i+1)-correct_s1/(i+1):+.3f}"
            )

    # Aggregate metrics
    acc_s1 = correct_s1 / n_eval
    acc_s2 = correct_s2 / n_eval
    delta_acc = acc_s2 - acc_s1
    avg_gap = sum(energy_gaps) / len(energy_gaps) if energy_gaps else 0.0
    avg_t_s1 = sum(times_s1) / len(times_s1) if times_s1 else 0.0
    avg_t_s2 = sum(times_s2) / len(times_s2) if times_s2 else 0.0

    print("\n" + "=" * 80)
    print("GSM8K evaluation results")
    print("=" * 80)
    print(f"Examples evaluated:         {n_eval}")
    print(f"System 1 accuracy:          {acc_s1:.4f}")
    print(f"System 2 accuracy:          {acc_s2:.4f}")
    print(f"Accuracy gain (S2 - S1):    {delta_acc:+.4f}")
    print(f"Mean energy gap (E1 - E2):  {avg_gap:.4f}")
    print(f"Avg time / query (S1):      {avg_t_s1*1000:.2f} ms")
    print(f"Avg time / query (S2):      {avg_t_s2*1000:.2f} ms "
          f"(×{(avg_t_s2/avg_t_s1) if avg_t_s1 > 0 else 0:.2f} slower)")
    print("=" * 80)


if __name__ == "__main__":
    config = chz.entrypoint(EvalConfig)
    main(config)
