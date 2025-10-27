### nanoEBM model implementation using pytorch

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EBTConfig
from model import GPT, GPTConfig  # reuse existing Transformer blocks as context encoder

import math
import inspect


class EBT(nn.Module):
    """
    Energy-Based Transformer (text-only) skeleton.

    - Uses the existing GPT backbone as the context encoder.
    - Adds a candidate embedding interface and a small verifier head producing energies.

    """

    def __init__(self, cfg: EBTConfig):
        super().__init__()
        self.cfg = cfg
        # Backbone GPT as context encoder
        self.backbone = GPT(
            GPTConfig(
                block_size=cfg.block_size,
                vocab_size=cfg.vocab_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_embd=cfg.n_embd,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        )
        # Candidate embedder
        self.wte = self.backbone.transformer.wte if cfg.use_shared_embeddings else nn.Embedding(
            cfg.vocab_size, cfg.n_embd
        )
        # Verifier head
        self.energy_head = EnergyHead(cfg.n_embd, cfg.energy_hidden)
        # Step size for refinement (optional learnable)
        step = torch.tensor(float(cfg.mcmc_step_size))
        self.mcmc_step_size = nn.Parameter(step) if cfg.mcmc_step_size_learnable else step

    def context_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Return last-position hidden states for each position.

        Args:
            idx: token ids [B, T]
        Returns:
            h: hidden states [B, T, d]
        """
        device = idx.device
        b, t = idx.size()
        # replicate GPT forward but keep internal states before lm_head
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.backbone.transformer.wte(idx) + self.backbone.transformer.wpe(pos)
        x = self.backbone.transformer.drop(x)
        for block in self.backbone.transformer.h:
            x = block(x)
        x = self.backbone.transformer.ln_f(x)
        return x

    def energy(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Compute scalar energy for pairs of (context state, candidate embedding).

        Shapes are broadcast-compatible; last dimension must be d.
        Returns energy with trailing singleton dimension.
        """
        return self.energy_head(h, e)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_energies: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """

        - Compute context states h [B, T, d]
        - If targets provided, intended to compute energy for all vocab at each position
          and return a cross-entropy loss over energies (to be implemented).

        Returns:
            energies_all (optional): placeholder None for now
            loss (optional): placeholder None for now
        """
        h = self.context_hidden(idx)
        # Placeholder outputs for scaffold; implementation to follow
        energies_all = None
        loss = None
        if return_energies:
            return energies_all, loss
        return None, loss


class EnergyHead(nn.Module):
    """
    Skeleton verifier head mapping (context_state, candidate_emb) -> scalar energy.
    """

    def __init__(self, hidden_dim: int, energy_hidden_mult: int = 4):
        super().__init__()
        inner = energy_hidden_mult * hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(4 * hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: context state [..., d]
            e: candidate embedding [..., d]
        Returns:
            energy: [..., 1]
        """
        z = torch.cat([h, e, h * e, torch.abs(h - e)], dim=-1)
        s = self.proj(z)
        # Energy is negative score by convention (low energy = high compatibility)
        return -s

#--------------------------------

