from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ThinkingState:
    logits: torch.Tensor  # [B, V]
    steps_done: int


def refine_logits_once(
    h_last: torch.Tensor,  # [B, d]
    wte: torch.Tensor,     # [V, d]
    energy_fn,             # callable(h) -> energies [B, V]
    logits: torch.Tensor,  # [B, V]
    step_size: float,
    softmax_temperature: float = 1.0,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """One closed-form gradient step on relaxed objective over logits.

    J = E_p[E] - λ H(p), with p = softmax(v / τ).
    Stop-grad mixture: compute E at p.detach() so gradients do not flow through E's
    dependence on p inside this step; only the softmax(v/τ) path is differentiated.
    Closed-form gradient (no autograd in the inner loop):
      dJ/dp = E_now + λ (log p + 1)
      ∂J/∂v = (1/τ) · p ⊙ (dJ/dp - ⟨dJ/dp⟩_p)
    """
    v = logits.detach()  # stop-grad per step
    tau = max(1e-6, float(softmax_temperature))
    with torch.no_grad():
        E = energy_fn(h_last)  # [B, V]
        p = torch.softmax(v / tau, dim=-1)
        if entropy_weight and entropy_weight != 0.0:
            p_safe = p.clamp_min(1e-9)
            dJdp = E + float(entropy_weight) * (p_safe.log() + 1.0)
        else:
            dJdp = E
        s = (dJdp * p).sum(dim=-1, keepdim=True)
        g = (p * (dJdp - s)) / tau
    v = v - step_size * g
    return v


def think(
    h_last: torch.Tensor,
    wte: torch.Tensor,
    energy_fn,
    steps: int,
    step_size: float,
    softmax_temperature: float = 1.0,
    entropy_weight: float = 0.0,
    init_noise_std: float = 0.0,
) -> ThinkingState:
    """Iterative refinement of next-token logits using closed-form updates.

    Initializes v0 = -E/τ (optionally + tiny noise) and applies steps of
    closed-form gradient updates under stop-grad mixture.
    """
    E0 = energy_fn(h_last)  # [B, V]
    logits = (-E0 / max(1e-6, float(softmax_temperature))).clone()
    if init_noise_std and init_noise_std != 0.0:
        logits = logits + float(init_noise_std) * torch.randn_like(logits)
    for _ in range(steps):
        logits = refine_logits_once(
            h_last,
            wte,
            energy_fn,
            logits,
            step_size,
            softmax_temperature=softmax_temperature,
            entropy_weight=entropy_weight,
        )
    return ThinkingState(logits=logits, steps_done=steps)
