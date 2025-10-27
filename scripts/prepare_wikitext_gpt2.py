"""
Prepare WikiText (wikitext-2-raw-v1 or wikitext-103-raw-v1) into nanoGPT-style memmaps
using Hugging Face GPT-2 byte-level BPE. Produces train.bin, val.bin, meta.pkl.

Usage:
  python scripts/prepare_wikitext_gpt2.py --name wikitext-2-raw-v1 --out data/wikitext-gpt2

Requires:
  pip install datasets transformers
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import List

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def encode_corpus(texts: List[str], tok: GPT2TokenizerFast) -> List[int]:
    ids: List[int] = []
    # Do not add special tokens automatically; keep raw GPT-2 encoding
    for s in texts:
        out = tok(s, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        ids.extend(out["input_ids"])
    return ids


def write_memmap(ids: List[int], path: str):
    arr = np.array(ids, dtype=np.uint16)  # GPT-2 vocab size (50257) fits in uint16
    arr.tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="wikitext-2-raw-v1",
                    choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"]) 
    ap.add_argument("--out", type=str, default="data/wikitext-gpt2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading dataset {args.name}...")
    ds = load_dataset("wikitext", args.name)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # Ensure EOS exists (it does by default for GPT-2)
    if tok.eos_token_id is None:
        tok.add_special_tokens({"eos_token": "<|endoftext|>"})

    for split_src, split_dst in [("train", "train"), ("validation", "val")]:
        print(f"Encoding split: {split_src}")
        texts = ds[split_src]["text"]
        ids = encode_corpus(texts, tok)
        out_path = os.path.join(args.out, f"{split_dst}.bin")
        print(f"Writing {len(ids):,} tokens to {out_path}")
        write_memmap(ids, out_path)

    meta = {
        "vocab_size": tok.vocab_size,
        "tokenizer": "gpt2",
    }
    with open(os.path.join(args.out, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Done.")


if __name__ == "__main__":
    main()
