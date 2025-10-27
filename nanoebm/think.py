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
    energy_fn,             # callable(h, e) -> energy [B, 1]
    logits: torch.Tensor,  # [B, V]
    step_size: float,
    tau_entropy: float = 0.0,
) -> torch.Tensor:
    """Skeleton refinement: no-op for now."""
    return logits


def think(
    h_last: torch.Tensor,
    wte: torch.Tensor,
    energy_fn,
    steps: int,
    step_size: float,
    init: str = "from_energy",
    tau_entropy: float = 0.0,
) -> ThinkingState:
    """Skeleton iterative refinement of next-token distribution."""
    B, d = h_last.shape
    V = wte.shape[0]
    logits = torch.zeros(B, V, device=h_last.device)
    for _ in range(steps):
        logits = refine_logits_once(h_last, wte, energy_fn, logits, step_size, tau_entropy)
    return ThinkingState(logits=logits, steps_done=steps)
