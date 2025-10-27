"""
EBT sampling entrypoint (scaffold): loads model and prints a usage note.

To extend: implement energy-based next-token distribution with optional thinking steps.
"""
from __future__ import annotations

import argparse
import json

import torch

from nanoebm import EBTConfig, ThinkingConfig
from nanoebm.model import EBT


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ebt_char_small.json")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    m = EBTConfig(**cfg["model"])  # type: ignore[arg-type]
    th = ThinkingConfig(**cfg["thinking"])  # type: ignore[arg-type]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EBT(m).to(device).eval()
    print("[nanoEBM] Model ready. Implement generation + thinking next.")


if __name__ == "__main__":
    main()

