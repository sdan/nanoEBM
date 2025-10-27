from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .transformer import Transformer


class ContextTransformer(nn.Module):
    """Expose final hidden states (pre-lm_head) of the Transformer backbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.core = Transformer(cfg)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.cfg.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.core.transformer.wte(idx)
        pos_emb = self.core.transformer.wpe(pos)
        x = self.core.transformer.drop(tok_emb + pos_emb)
        for block in self.core.transformer.h:
            x = block(x)
        x = self.core.transformer.ln_f(x)
        return x

    @property
    def wte(self) -> nn.Embedding:
        return self.core.transformer.wte


class EnergyHead(nn.Module):
    """
    Vectorized energy: E(x, y) = b_y - (Wc h) · (We e_y)
    where e_y are token embeddings. Cheap & stable, but more expressive than plain dot-product.
    """

    def __init__(self, gpt_cfg: ModelConfig, ebt_cfg: ModelConfig):
        super().__init__()
        d_model = gpt_cfg.n_embd
        d_energy = getattr(ebt_cfg, "d_energy", d_model)
        self.context_proj = nn.Linear(d_model, d_energy, bias=True)
        self.token_proj = nn.Linear(d_model, d_energy, bias=False)
        self.use_token_bias = getattr(ebt_cfg, "use_token_bias", True)
        if self.use_token_bias:
            self.token_bias = nn.Parameter(torch.zeros(gpt_cfg.vocab_size))
        else:
            self.register_parameter("token_bias", None)
        self.cached_token_features = None  # (V, dE)

    def token_features(self, wte_weight: torch.Tensor) -> torch.Tensor:
        # cache to avoid recomputing every step
        if (
            self.cached_token_features is None
            or self.cached_token_features.shape[0] != wte_weight.shape[0]
            or self.cached_token_features.device != wte_weight.device
        ):
            self.cached_token_features = None
        if self.cached_token_features is None:
            with torch.no_grad():
                self.cached_token_features = self.token_proj(wte_weight)
        return self.cached_token_features

    def energies_all_tokens(self, h: torch.Tensor, wte_weight: torch.Tensor) -> torch.Tensor:
        # h: (B,T,d) -> E: (B,T,V)
        C = self.context_proj(h)  # (B,T,dE)
        R = self.token_features(wte_weight)  # (V,dE)
        E = -torch.einsum("btd,vd->btv", C, R)
        if self.use_token_bias:
            E = E + self.token_bias.view(1, 1, -1)
        return E


class EBTLanguageModel(nn.Module):
    """
    Normalized EBM: train with CE on logits = -E. Optional thinking loop at inference.
    """

    def __init__(self, gpt_cfg: ModelConfig, ebt_cfg: ModelConfig):
        super().__init__()
        self.backbone = ContextTransformer(gpt_cfg)
        self.energy = EnergyHead(gpt_cfg, ebt_cfg)
        self.gpt_cfg = gpt_cfg
        self.ebt_cfg = ebt_cfg

    @property
    def wte_weight(self) -> torch.Tensor:
        return self.backbone.wte.weight  # (V, d_model)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Teacher forcing over all positions.
        idx: (B, T) int tokens
        targets: (B, T) int tokens or None
        Returns: (loss, extras) where extras has 'logits' and 'energies'
        """
        x = idx[:, :-1]
        y = idx[:, 1:]
        h = self.backbone(x)  # (B, T-1, d_model)
        E = self.energy.energies_all_tokens(h, self.wte_weight)  # (B, T-1, V)
        logits = -E
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1
            )
        return loss, logits

    @torch.no_grad()
    def generate_greedy(self, idx: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        self.energy.cached_token_features = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.gpt_cfg.block_size :]
            h = self.backbone(idx_cond)[:, -1]  # (B, d)
            E = self.energy.energies_all_tokens(h.unsqueeze(1), self.wte_weight)[:, 0, :]
            nxt = torch.argmin(E, dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def generate_think(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        tau: Optional[float] = None,
        noise: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        steps = steps if steps is not None else 0
        lr = lr if lr is not None else 1.0
        tau = tau if tau is not None else 1.0
        noise = noise if noise is not None else 0.0

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.gpt_cfg.block_size :]
            h_last = self.backbone(idx_cond)[:, -1]  # (B, d)
            E = self.energy.energies_all_tokens(h_last.unsqueeze(1), self.wte_weight)[:, 0, :]  # (B,V)
            if topk is not None:
                topE, topI = torch.topk(-E, k=topk, dim=-1)  # (B,K)
                v = topE.clone()
                cand_idx = topI  # (B,K)
            else:
                v = (-E).clone()
                cand_idx = None

            v = v.detach().requires_grad_(True)
            for _k in range(steps):
                p = F.softmax(v / tau, dim=-1)  # (B,K) or (B,V)
                R = self.energy.token_features(self.wte_weight)  # (V,dE)
                if cand_idx is not None:
                    R_sel = R[cand_idx]  # (B,K,dE)
                else:
                    R_sel = R.unsqueeze(0).expand(p.size(0), -1, -1)  # (B,V,dE)
                c = self.energy.context_proj(h_last)  # (B,dE)
                e = torch.einsum("bk,bkd->bd", p, R_sel)  # (B,dE)
                if self.energy.use_token_bias:
                    if cand_idx is not None:
                        b_sel = self.energy.token_bias[cand_idx]  # (B,K)
                        b = torch.einsum("bk,bk->b", p, b_sel)
                    else:
                        b = torch.einsum("bk,k->b", p, self.energy.token_bias)
                else:
                    b = torch.zeros(p.size(0), device=v.device)

                E_relaxed = b - (c * e).sum(-1)  # (B,)
                H = -(p.clamp_min(1e-9) * (p.clamp_min(1e-9)).log()).sum(-1)  # (B,)
                J = E_relaxed - tau * H
                (g,) = torch.autograd.grad(J.sum(), v, retain_graph=False, create_graph=False)
                v = (v - lr * g).detach()
                if noise and noise > 0:
                    v = v + noise * torch.randn_like(v)
                v.requires_grad_(True)

            nxt_local = torch.argmax(v, dim=-1, keepdim=True)  # (B,1)
            if cand_idx is not None:
                nxt = cand_idx.gather(-1, nxt_local)
            else:
                nxt = nxt_local
            idx = torch.cat([idx, nxt], dim=1)
        return idx

