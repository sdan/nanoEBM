"""
EBT training entrypoint (scaffold): loads configs, instantiates model, prints summary.

To extend: wire in optimizer, training loop, evaluation and checkpointing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from nanoebm import EBTConfig, TrainerConfig, DataConfig, ThinkingConfig
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
    t = TrainerConfig(**cfg["trainer"])  # type: ignore[arg-type]
    d = DataConfig(**cfg["data"])  # type: ignore[arg-type]
    th = ThinkingConfig(**cfg["thinking"])  # type: ignore[arg-type]

    print("[nanoEBM] Config summary:")
    print(json.dumps(cfg, indent=2))

    device = t.device
    torch.manual_seed(t.seed)
    model = EBT(m).to(device)
    if t.compile:
        print("Compiling model (PyTorch 2.x)...")
        model = torch.compile(model)  # type: ignore[attr-defined]

    # Placeholder: show parameter counts and exit (scaffold only)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nanoEBM] parameters: {n_params/1e6:.2f}M")
    print("[nanoEBM] Scaffolding complete. Implement training loop next.")


if __name__ == "__main__":
    main()

