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
    E(x, y) = b_y - (Wc h) · (We e_y)
    where e_y are token embeddings. 
    """

    def __init__(self, gpt_cfg: ModelConfig, ebt_cfg: ModelConfig):
        super().__init__()
        d_model = gpt_cfg.n_embd
        d_energy = getattr(ebt_cfg, "d_energy", d_model)
        self.context_proj = nn.Linear(d_model, d_energy, bias=True)
        self.token_proj = nn.Linear(d_model, d_energy, bias=False)
        # Mixture-aware pathway: combines context h with mixture embedding y_mix
        self.mixture_mlp = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_energy),
        )
        self.use_token_bias = getattr(ebt_cfg, "use_token_bias", True)
        if self.use_token_bias:
            self.token_bias = nn.Parameter(torch.zeros(gpt_cfg.vocab_size))
        else:
            self.register_parameter("token_bias", None)
        self.cached_token_features = None  # (V, dE)

    def token_features(self, wte_weight: torch.Tensor) -> torch.Tensor:
        # Train-aware policy: compute fresh during training to allow gradients into token_proj;
        # cache only for eval to keep inference fast.
        if self.training:
            return self.token_proj(wte_weight)

        # Eval path: cache to avoid recompute across steps/tokens
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

    def energies_with_mixture(
        self,
        h: torch.Tensor,
        wte_weight: torch.Tensor,
        p_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture-aware energies conditioned on current distribution p.

        h: (B,T,d_model)
        wte_weight: (V,d_model)
        p_dist: (B,T,V)
        Returns: (B,T,V) energies
        """
        # Mixture embedding of tokens under current distribution
        y_mix = torch.einsum("btv,vd->btd", p_dist, wte_weight)  # (B,T,d_model)
        context_mix = torch.cat([h, y_mix], dim=-1)  # (B,T,2*d_model)
        r = self.mixture_mlp(context_mix)  # (B,T,d_energy)

        # Base energy term
        C = self.context_proj(h)  # (B,T,dE)
        R = self.token_features(wte_weight)  # (V,dE)
        E_base = -torch.einsum("btd,vd->btv", C, R)
        # Mixture-aware correction
        E_corr = -torch.einsum("btd,vd->btv", r, R)
        E = E_base + E_corr
        if self.use_token_bias:
            E = E + self.token_bias.view(1, 1, -1)
        return E


class EBTLanguageModel(nn.Module):
    """
    Normalized energy-based model with optional thinking loop at inference.
    """

    def __init__(self, gpt_cfg: ModelConfig, ebt_cfg: ModelConfig):
        super().__init__()
        self.backbone = ContextTransformer(gpt_cfg)
        # Mode selection: lm_head-as-energy or factorized EnergyHead
        self.use_lm_energy = bool(getattr(ebt_cfg, "use_lm_head_as_energy", False))
        self.energy_temperature = float(getattr(ebt_cfg, "energy_temperature", 1.0) or 1.0)
        self.energy = None if self.use_lm_energy else EnergyHead(gpt_cfg, ebt_cfg)
        self.gpt_cfg = gpt_cfg
        self.ebt_cfg = ebt_cfg
        # Learnable step size alpha (per-model scalar)
        alpha_init = float(getattr(ebt_cfg, "refine_step_size", 1.0))
        learn_alpha = bool(getattr(ebt_cfg, "refine_step_size_learnable", False))
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

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
        if self.use_lm_energy:
            logits_raw = self.backbone.core.lm_head(h)  # (B,T-1,V)
            E = -(logits_raw / max(1e-6, self.energy_temperature))
        else:
            E = self.energy.energies_all_tokens(h, self.wte_weight)  # (B, T-1, V)

        # Baseline logits from energies and auxiliary CE to explicitly train the energy head
        logits_baseline = -E
        loss_aux = None
        if targets is not None:
            loss_aux = F.cross_entropy(
                logits_baseline.reshape(-1, logits_baseline.size(-1)),
                y.reshape(-1),
                ignore_index=-1,
            )

        # If no refinement steps requested, fall back to standard CE on -E
        steps = int(getattr(self.ebt_cfg, "refine_steps", 0))
        if steps <= 0:
            logits = logits_baseline
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1
                )
            extras = {}
            return loss, logits, extras

        # Optionally refine only the last position (training-time simplification)
        refine_last_only = bool(getattr(self.ebt_cfg, "refine_last_position_only", False)) and self.training

        # Iterative refinement over per-position logits v
        B, Tm1, V = E.shape
        device = E.device
        init_mode = getattr(self.ebt_cfg, "denoising_initial_condition", "random_noise")
        # Slice to last position if enabled
        if refine_last_only:
            E_work = E[:, -1:, :]
            h_work = h[:, -1:, :]
            y_work = y[:, -1:]
            T_work = 1
        else:
            E_work = E
            h_work = h
            y_work = y
            T_work = Tm1
        # Initialize v in model dtype, then cast to float32 for inner math
        if init_mode == "zeros":
            v = torch.zeros(B, T_work, V, device=device, dtype=E_work.dtype)
        else:
            v = torch.randn(B, T_work, V, device=device, dtype=E_work.dtype)

        # Metrics accumulators
        step_losses = []
        loss_main = None
        tau_H = float(getattr(self.ebt_cfg, "entropy_reg_tau", 0.0) or 0.0)
        label_smooth_base = float(getattr(self.ebt_cfg, "soften_target_prob_dist", 0.0) or 0.0)
        # Detach policy and truncation (prefer new names, fallback to legacy)
        do_detach = bool(getattr(self.ebt_cfg, "detach_refine", True))
        truncate = bool(getattr(self.ebt_cfg, "truncate_refine", False))
        langevin_std = float(getattr(self.ebt_cfg, "langevin_noise", 0.0) or 0.0)
        clamp_change = float(getattr(self.ebt_cfg, "clamp_update_max_change", 0.0) or 0.0)
        abs_clamp = float(getattr(self.ebt_cfg, "absolute_clamp", 0.0) or 0.0)

        # Cast refinement state and static energies to float32 for stability (esp. bf16 training)
        v32 = v.float()
        E32 = E_work.float()

        # Initial expected energy (for logging gap) over the active tokens only
        if not torch.isfinite(E32).all():
            raise RuntimeError("Energies contain NaN/Inf; check model stability.")
        with torch.no_grad():
            p0 = F.softmax(v32, dim=-1)
            init_energy_mean = ((p0 * E32).sum(dim=-1)).mean()
        trace_enabled = bool(getattr(self.ebt_cfg, "log_expected_energy_trace", False))
        energy_trace = []
        if trace_enabled:
            energy_trace.append(float(init_energy_mean.detach().cpu()))

        # Main loop
        for k in range(steps):
            # Detach per-step by default; keep the final step connected when truncating (one-step-through)
            if do_detach and (not truncate or k < steps - 1):
                v32 = v32.detach()
            v32.requires_grad_(True)

            # Langevin noise on logits
            if langevin_std != 0.0:
                v32 = v32 + (langevin_std * torch.randn_like(v32))

            # Compute relaxed objective J = E_p[E] - tau * H(p)
            p = F.softmax(v32, dim=-1)
            # Mixture-aware energy update if using EnergyHead; otherwise keep static E32
            if self.energy is not None:
                E32_used = self.energy.energies_with_mixture(h_work, self.wte_weight, p).float()
            else:
                E32_used = E32
            J = (p * E32_used).sum(dim=-1).sum()
            if tau_H != 0.0:
                p_safe = p.clamp_min(1e-9)
                H = -(p_safe * p_safe.log()).sum(dim=-1).sum()
                J = J - tau_H * H

            # Gradient wrt logits v. We do not need higher-order graphs here:
            # alpha gradients flow without create_graph=True (v' = v - alpha * g ⇒ dv'/dalpha = -g).
            # Keep create_graph=False for stability and to avoid graph reuse errors.
            (g,) = torch.autograd.grad(J, v32, retain_graph=False, create_graph=False)

            # Update logits with explicit delta clamp (alpha-invariant semantics)
            alpha_eff = self.alpha.clamp(min=1e-6).float()
            delta = -alpha_eff * g
            if clamp_change and clamp_change > 0.0:
                delta = delta.clamp(min=-clamp_change, max=clamp_change)
            v32 = v32 + delta

            # Optional absolute clamp on logits range
            if abs_clamp and abs_clamp > 0.0:
                v32 = v32.clamp(min=-abs_clamp, max=abs_clamp)

            # Quick stability checks
            if not torch.isfinite(v32).all():
                raise RuntimeError("Refinement state v has NaN/Inf; lower step size or enable clamps.")

            # Per-step loss vs ground truth
            if targets is not None:
                if label_smooth_base != 0.0 and steps > 1:
                    ls = ((steps - 1 - k) / (steps - 1)) * label_smooth_base
                else:
                    ls = 0.0
                ce = F.cross_entropy(
                    v32.reshape(-1, V), y_work.reshape(-1), ignore_index=-1, label_smoothing=ls if ls > 0 else 0.0
                )
                if truncate:
                    if k == steps - 1:
                        loss_main = ce
                else:
                    step_losses.append(ce)

            # Log expected energy trace per step if requested
            if trace_enabled:
                with torch.no_grad():
                    p_step = F.softmax(v32, dim=-1)
                    en_mean = ((p_step * E32).sum(dim=-1)).mean()
                    energy_trace.append(float(en_mean.detach().cpu()))

        if not truncate and targets is not None:
            loss_main = torch.stack(step_losses).mean()

        # Final expected energy (for logging gap)
        with torch.no_grad():
            pf = F.softmax(v32, dim=-1)
            final_energy_mean = ((pf * E32).sum(dim=-1)).mean()
            energy_gap = init_energy_mean - final_energy_mean

        # Build full logits tensor (B,T-1,V)
        if refine_last_only:
            # Fill others with -E (one-pass baseline), last with refined logits
            logits = -E
            logits[:, -1:, :] = v32.to(E.dtype)
        else:
            logits = v32.to(E.dtype)
        extras = {
            "initial_energy": float(init_energy_mean.detach().cpu()),
            "final_energy": float(final_energy_mean.detach().cpu()),
            "energy_gap": float(energy_gap.detach().cpu()),
        }
        if trace_enabled:
            extras["expected_energy_trace"] = energy_trace
        if targets is not None:
            # ppl from main loss (refined logits), not including auxiliary
            with torch.no_grad():
                ppl_main = torch.exp(loss_main.detach()) if loss_main is not None else torch.tensor(0.0)
                extras["perplexity"] = float(ppl_main.cpu())
        
        # Final loss combine: (1 - lambda_aux) * loss_main + lambda_aux * loss_aux
        if targets is not None:
            lambda_aux = float(getattr(self.ebt_cfg, "aux_ce_weight", 0.5))
            if loss_main is None:
                # Safety: default to 0 if no main loss (shouldn't happen when K>0 & targets provided)
                loss_main = torch.tensor(0.0, device=E.device, dtype=E.dtype)
            if loss_aux is None:
                loss_aux = torch.tensor(0.0, device=E.device, dtype=E.dtype)
            loss = (1.0 - lambda_aux) * loss_main + lambda_aux * loss_aux
            extras["loss_main"] = float(loss_main.detach().cpu())
            extras["loss_aux"] = float(loss_aux.detach().cpu())
            extras["loss_total"] = float(loss.detach().cpu())
        else:
            loss = None

        return loss, logits, extras

    def energies(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute energies E for all positions given input tokens.

        idx: (B, T) int tokens
        Returns: (B, T, V) energies for next-token at each position
        """
        h = self.backbone(idx)  # (B, T, d_model)
        if self.use_lm_energy:
            logits = self.backbone.core.lm_head(h)
            return -(logits / max(1e-6, self.energy_temperature))
        else:
            return self.energy.energies_all_tokens(h, self.wte_weight)

    @torch.no_grad()
    def generate_greedy(self, idx: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        if self.energy is not None:
            self.energy.cached_token_features = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.gpt_cfg.block_size :]
            h = self.backbone(idx_cond)[:, -1]  # (B, d)
            if self.use_lm_energy:
                logits = self.backbone.core.lm_head(h)
                E = -(logits / max(1e-6, self.energy_temperature))
            else:
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
        temp: Optional[float] = None,
        entropy: Optional[float] = None,
        noise: Optional[float] = None,
        topk: Optional[int] = None,
        # Decoding options
        sample: bool = False,
        sample_temp: float = 1.0,
        sample_top_p: Optional[float] = None,
    ) -> torch.Tensor:
        # Default to config/model params if not provided
        if steps is None:
            steps = int(getattr(self.ebt_cfg, "refine_steps", 0) or 0)
        if lr is None:
            lr = float(self.alpha.detach().clamp(min=1e-6).item())
        # Temperature (softmax) and entropy weight defaults
        if temp is None:
            temp = 1.0
        if entropy is None:
            entropy = float(getattr(self.ebt_cfg, "entropy_reg_tau", 0.0) or 0.0)
        if noise is None:
            noise = float(getattr(self.ebt_cfg, "langevin_noise", 0.0) or 0.0)
        clamp_change = float(getattr(self.ebt_cfg, "clamp_update_max_change", 0.0) or 0.0)
        abs_clamp = float(getattr(self.ebt_cfg, "absolute_clamp", 0.0) or 0.0)

        def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
            """Filter a batch of logits using nucleus (top-p) filtering.
            Returns masked logits where tokens outside the nucleus are set to -inf.
            logits: (B, K_or_V)
            """
            if top_p is None or top_p <= 0.0 or top_p >= 1.0:
                return logits
            # sort by descending logit
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cdf = probs.cumsum(dim=-1)
            # keep smallest set with cumulative prob >= top_p
            # mask positions where cdf exceeds top_p, but always keep at least one
            cutoff = (cdf > top_p)
            # ensure first token kept
            cutoff[..., 0] = False
            # shift so we drop tokens after the last kept position
            cutoff = cutoff.cumsum(dim=-1) > 0
            # map mask back to original indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(dim=-1, index=sorted_indices, src=cutoff)
            masked = logits.masked_fill(mask, float("-inf"))
            return masked

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.gpt_cfg.block_size :]
            h_last = self.backbone(idx_cond)[:, -1]  # (B, d)
            if self.use_lm_energy:
                # Simple path: E = -(logits/T), refine logits directly by minimizing Ebar - entropy*H
                logits0 = self.backbone.core.lm_head(h_last)  # (B,V)
                E_full = -(logits0 / max(1e-6, self.energy_temperature))
                if topk is not None:
                    v, cand_idx = torch.topk(logits0, k=topk, dim=-1)  # start from top-k logits
                    E_sel = torch.gather(E_full, dim=-1, index=cand_idx)
                else:
                    v = logits0.clone()
                    cand_idx = None
                    E_sel = E_full

                v = v.detach().requires_grad_(True)
                for _k in range(steps):
                    p = F.softmax(v / max(1e-6, float(temp)), dim=-1)
                    Ebar = (p * E_sel).sum(dim=-1).mean()
                    if float(entropy) != 0.0:
                        p_safe = p.clamp_min(1e-9)
                        H = -(p_safe * p_safe.log()).sum(dim=-1).mean()
                        J = Ebar - float(entropy) * H
                    else:
                        J = Ebar
                    (g,) = torch.autograd.grad(J, v, retain_graph=False, create_graph=False)
                    delta = -lr * g
                    if clamp_change and clamp_change > 0.0:
                        delta = delta.clamp(min=-clamp_change, max=clamp_change)
                    v = (v + delta).detach()
                    if abs_clamp and abs_clamp > 0.0:
                        v = v.clamp(min=-abs_clamp, max=abs_clamp)
                    if noise and noise > 0:
                        v = v + noise * torch.randn_like(v)
                    v.requires_grad_(True)
            else:
                # Original factorized EnergyHead refinement
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
                    p = F.softmax(v / max(1e-6, float(temp)), dim=-1)  # (B,K) or (B,V)
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
                    J = E_relaxed - float(entropy) * H
                    (g,) = torch.autograd.grad(J.sum(), v, retain_graph=False, create_graph=False)
                    # Apply delta clamp for stability
                    delta = -lr * g
                    if clamp_change and clamp_change > 0.0:
                        delta = delta.clamp(min=-clamp_change, max=clamp_change)
                    v = (v + delta).detach()
                    # Optional absolute clamp
                    if abs_clamp and abs_clamp > 0.0:
                        v = v.clamp(min=-abs_clamp, max=abs_clamp)
                    if noise and noise > 0:
                        v = v + noise * torch.randn_like(v)
                    v.requires_grad_(True)

            if sample:
                # Sample from refined distribution with optional temperature and top-p
                logits_out = v / max(1e-6, float(sample_temp))
                if sample_top_p is not None:
                    logits_out = _top_p_filtering(logits_out, float(sample_top_p))
                probs = F.softmax(logits_out, dim=-1)
                # Handle potential invalid probs after filtering (fallback to unfiltered)
                if (not torch.isfinite(probs).all().item()) or (probs.sum(dim=-1) == 0).any().item():
                    probs = F.softmax(v, dim=-1)
                nxt_local = torch.multinomial(probs, num_samples=1)  # (B,1)
                if cand_idx is not None:
                    nxt = cand_idx.gather(-1, nxt_local)
                else:
                    nxt = nxt_local
            else:
                nxt_local = torch.argmax(v, dim=-1, keepdim=True)  # (B,1)
                if cand_idx is not None:
                    nxt = cand_idx.gather(-1, nxt_local)
                else:
                    nxt = nxt_local
            idx = torch.cat([idx, nxt], dim=1)
        return idx
