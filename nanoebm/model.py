"""
EBM - Minimal Energy-Based Model
A simplified implementation demonstrating System 1/System 2 thinking through energy-based refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import deque
import random

from .transformer import Transformer
from .config import ModelConfig


class ReplayBuffer:
    """Simple replay buffer for storing and sampling previous predictions."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, logits: torch.Tensor):
        """Store logits for future use."""
        # Store as CPU tensor to save GPU memory
        self.buffer.append(logits.detach().cpu())
    
    def sample(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """Sample random previous predictions."""
        if len(self.buffer) < batch_size:
            return None
        
        samples = random.sample(self.buffer, batch_size)
        # Stack and move back to original device
        return torch.stack(samples)
    
    def has_samples(self) -> bool:
        """Check if buffer has any samples."""
        return len(self.buffer) > 0


class EBM(nn.Module):
    """
    Energy-Based Model for language.
    
    The model learns an energy function E(x, y) where low energy = good text.
    It has two modes:
    - System 1: Direct readout of energy (fast, like a regular LM)
    - System 2: Gradient descent on logits to minimize energy (slower but better)
    
    This is like having a quick intuition vs thinking harder about the answer.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.transformer = Transformer(config)
        
        # This linear layer defines our energy function
        # E(hidden_state, token) = -hidden_state @ W[token]
        self.energy_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Step size for gradient descent (learned during training)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Store previous predictions to warm-start gradient descent
        self.replay_buffer = ReplayBuffer(max_size=1000)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for energy head."""
        if isinstance(module, nn.Linear) and module == self.energy_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from transformer."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        
        # Get embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.transformer.wte(idx)
        pos_emb = self.transformer.transformer.wpe(pos)
        x = self.transformer.transformer.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.transformer.transformer.h:
            x = block(x)
        x = self.transformer.transformer.ln_f(x)
        
        return x  # (B, T, n_embd)
    
    def system1_direct_energy(self, idx: torch.Tensor) -> torch.Tensor:
        """
        System 1: Just read out the energy directly.
        This is what a normal language model does - one forward pass and done.
        
        Returns: logits where logit = -energy (high logit = low energy = good)
        """
        h = self.get_hidden_states(idx)  # (B, T, n_embd)
        energy = self.energy_head(h)  # (B, T, V)
        return -energy  # Flip sign: low energy should have high logit
    
    def system2_refine(
        self, 
        idx: torch.Tensor, 
        steps: int = 3,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        System 2: Gradient descent on the logits to minimize energy.
        
        We start with initial logits and ask: "what logits would minimize 
        the expected energy under the softmax distribution?"
        Then we take gradient steps to find better logits.
                """
        B, T = idx.shape
        V = self.config.vocab_size
        device = idx.device
        
        # Get hidden states (we'll reuse these, no need to recompute)
        h = self.get_hidden_states(idx).detach()  # (B, T, n_embd)
        
        # Start with System 1's guess
        logits = -self.energy_head(h)  # Initial logits
        
        # Maybe warm-start from a previous similar prediction
        if self.replay_buffer.has_samples() and self.training:
            sample = self.replay_buffer.sample(1)
            if sample is not None and sample.shape[-2:] == (T, V):
                logits = sample[0].to(device).expand(B, T, V).clone()
                logits = logits + 0.01 * torch.randn_like(logits)  # Add noise
            # If sample doesn't match, we already have logits from System 1 above
        
        trajectory = [logits.clone()] if return_trajectory else []
        
        # Gradient descent loop: improve logits to minimize expected energy
        for step in range(steps):
            logits = logits.detach().requires_grad_(True)
            
            # Current probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # Energy of each token choice
            energies = self.energy_head(h)
            
            # Expected energy: E_p[E(x,y)] = sum_y p(y) * E(x,y)
            expected_energy = (probs * energies).sum()
            
            # Gradient tells us how to change logits to reduce expected energy
            grad = torch.autograd.grad(expected_energy, logits)[0]
            
            # Take a gradient descent step
            step_size = self.alpha.clamp(min=1e-6, max=1.0)
            logits = logits - step_size * grad.clamp(-5, 5)
            
            # Center logits (doesn't change softmax, helps stability)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            
            if return_trajectory:
                trajectory.append(logits.clone())
        
        # Remember this refined prediction for future warm-starts
        if self.training and T == self.config.block_size:
            self.replay_buffer.add(logits[0])
        
        if return_trajectory:
            return logits, trajectory
        return logits
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        use_refine: bool = True,
        refine_steps: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Forward pass with optional refinement.
        
        Args:
            idx: Input tokens (B, T)
            targets: Target tokens for loss computation (B, T)
            use_refine: Whether to use System 2 refinement
            refine_steps: Number of refinement steps
        
        Returns: (loss, logits, metrics)
        """
        metrics = {}
        
        if use_refine:
            # System 2: Refined prediction
            with torch.no_grad():
                # Track initial energy for metrics
                initial_logits = self.system1_direct_energy(idx)
                metrics['initial_energy'] = initial_logits.mean().item()
            
            # Get refined logits
            logits = self.system2_refine(idx, steps=refine_steps)
            metrics['final_energy'] = logits.mean().item()
            metrics['energy_gap'] = metrics['initial_energy'] - metrics['final_energy']
        else:
            # System 1: Direct prediction
            logits = self.system1_direct_energy(idx)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
            # Track perplexity
            with torch.no_grad():
                metrics['perplexity'] = torch.exp(loss).item()
        
        return loss, logits, metrics
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_thinking: bool = True,
        think_steps: int = 4
    ) -> torch.Tensor:
        """
        Generate tokens with optional System 2 thinking.
        
        Args:
            idx: Initial context tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            use_thinking: Whether to use System 2 refinement
            think_steps: Number of refinement steps when thinking
        
        Returns: Generated token sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get logits for next token
            if use_thinking:
                logits = self.system2_refine(idx_cond, steps=think_steps)
            else:
                logits = self.system1_direct_energy(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Single training step.
        
        Args:
            batch: (input_ids, target_ids)
        
        Returns: (loss, metrics)
        """
        idx, targets = batch
        
        # Forward pass with refinement
        loss, logits, metrics = self.forward(
            idx, 
            targets=targets,
            use_refine=True,
            refine_steps=2
        )
        
        # Track alpha value
        metrics['alpha'] = self.alpha.item()
        
        return loss, metrics
    
    def inference(
        self,
        idx: torch.Tensor,
        mode: str = 'think'
    ) -> torch.Tensor:
        """
        Inference with different modes.
        
        Args:
            idx: Input tokens
            mode: 'fast' (System 1), 'think' (System 2), or 'adaptive'
        
        Returns: Logits for next token prediction
        """
        if mode == 'fast':
            return self.system1_direct_energy(idx)
        elif mode == 'think':
            return self.system2_refine(idx, steps=4)
        elif mode == 'adaptive':
            # Use System 1 for common patterns, System 2 for uncertain cases
            logits_s1 = self.system1_direct_energy(idx)
            entropy = -(F.softmax(logits_s1, dim=-1) * F.log_softmax(logits_s1, dim=-1)).sum(dim=-1)
            
            # High entropy = high uncertainty, use System 2
            if entropy.mean() > 2.0:  # Threshold can be tuned
                return self.system2_refine(idx, steps=3)
            else:
                return logits_s1
        else:
            raise ValueError(f"Unknown inference mode: {mode}")


# Example usage and testing
if __name__ == "__main__":
    # Create a simple config
    config = ModelConfig(
        vocab_size=100,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    
    # Initialize model
    model = EBM(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test System 1
    logits_s1 = model.system1_direct_energy(idx)
    print(f"System 1 logits shape: {logits_s1.shape}")
    
    # Test System 2
    logits_s2 = model.system2_refine(idx, steps=3)
    print(f"System 2 logits shape: {logits_s2.shape}")
    
    # Test full forward
    loss, logits, metrics = model.forward(idx, targets=targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, use_thinking=True)
    print(f"Generated shape: {generated.shape}")