from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ThinkingState:
    logits: torch.Tensor  # [B, V]
    steps_done: int


def refine_logits_once(
    h_last: torch.Tensor,  # [B, d]
    wte: torch.Tensor,     # [V, d] (unused in minimal path)
    energy_fn,             # callable(h) -> energies [B, V]
    logits: torch.Tensor,  # [B, V]
    step_size: float,
    tau_entropy: float = 0.0,
) -> torch.Tensor:
    """One Langevin/gradient step on relaxed objective over logits.

    Minimal interface assumes `energy_fn(h_last)` returns energies over all tokens: [B, V].
    We update logits using grad of J = E_p[E] - tau * H(p), where p = softmax(logits).
    """
    # Detach per-step by default in this utility
    v = logits.detach().requires_grad_(True)
    E = energy_fn(h_last)  # [B, V]
    p = torch.softmax(v, dim=-1)
    J = (p * E).sum(dim=-1).sum()
    if tau_entropy and tau_entropy != 0.0:
        p_safe = p.clamp_min(1e-9)
        H = -(p_safe * p_safe.log()).sum(dim=-1).sum()
        J = J - tau_entropy * H
    (g,) = torch.autograd.grad(J, v, retain_graph=False, create_graph=False)
    v = v - step_size * g
    return v.detach()


def think(
    h_last: torch.Tensor,
    wte: torch.Tensor,
    energy_fn,
    steps: int,
    step_size: float,
    init: str = "random_noise",
    tau_entropy: float = 0.0,
) -> ThinkingState:
    """Iterative refinement of next-token logits using energy expectations.

    energy_fn(h_last) -> [B,V]
    """
    B, _ = h_last.shape
    V = wte.shape[0]
    if init == "zeros":
        logits = torch.zeros(B, V, device=h_last.device)
    else:
        logits = torch.randn(B, V, device=h_last.device)
    for _ in range(steps):
        logits = refine_logits_once(h_last, wte, energy_fn, logits, step_size, tau_entropy)
    return ThinkingState(logits=logits, steps_done=steps)
