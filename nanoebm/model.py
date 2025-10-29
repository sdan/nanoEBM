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
        
        # Fixed step size for gradient descent (not learned, following EBT)
        self.register_buffer('alpha', torch.tensor(config.alpha_value))
        
        # Store previous predictions to warm-start gradient descent
        self.replay_buffer = ReplayBuffer(max_size=1000)
        
        self.apply(self._init_weights)
        
        # Weight tying: energy head shares weights with token embeddings
        self.energy_head.weight = self.transformer.transformer.wte.weight
    
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
        steps: int = None,
        return_trajectory: bool = False,
        detach_hidden: bool = False
    ) -> torch.Tensor:
        """
        System 2: Gradient descent on the logits to minimize energy.
        
        Following EBT: trainable thinking with second-order gradients,
        randomized steps, Langevin noise, and early stopping.
        
        Args:
            idx: Input tokens
            steps: Number of refinement steps (None = use config/random)
            return_trajectory: Whether to return intermediate logits
            detach_hidden: Whether to detach hidden states (for stable early training)
        """
        B, T = idx.shape
        V = self.config.vocab_size
        device = idx.device
        
        # Get hidden states - optionally detach for more stable early training
        h = self.get_hidden_states(idx)  # (B, T, n_embd)
        if detach_hidden:
            h = h.detach()
        
        # Randomize step count during training (EBT uses 2-3 steps)
        if steps is None:
            if self.training:
                steps = random.randint(2, 3)
            else:
                steps = self.config.refine_steps
        
        # Initialize logits
        if self.training and random.random() < 0.5:
            # Random initialization during training (helps exploration)
            logits = 0.01 * torch.randn(B, T, V, device=device, requires_grad=True)
        else:
            # Start with System 1's guess
            logits = -self.energy_head(h)
            
            # Maybe warm-start from replay buffer
            if self.replay_buffer.has_samples() and self.training:
                sample = self.replay_buffer.sample(1)
                if sample is not None and sample.shape[-2:] == (T, V):
                    logits = sample[0].to(device).expand(B, T, V).clone()
                    logits = logits + 0.01 * torch.randn_like(logits)
        
        logits = logits.requires_grad_(True)
        trajectory = [logits.clone()] if return_trajectory else []
        
        # Track energy for early stopping
        prev_energy = None
        early_stop_patience = 0
        
        # Gradient descent loop with EBT improvements
        for step in range(steps):
            # Current probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # We want to MAXIMIZE expected logits (which minimizes expected energy)
            # Since logits = -energy, maximizing logits minimizes energy
            # Expected logits under current distribution
            expected_logits = (probs * logits).sum(dim=-1).mean()
            
            # To maximize, we need the negative of the loss
            # We want gradient ASCENT on expected_logits
            loss = -expected_logits
            
            # Gradient with create_graph=True for second-order gradients during training
            grad = torch.autograd.grad(
                loss,  # Minimize negative expected logits = maximize expected logits
                logits, 
                create_graph=self.training  # Key change for trainable thinking
            )[0]
            
            # Step size with jitter during training (EBT uses multiplicative noise)
            step_size = self.alpha
            if self.training:
                jitter = 1.0 + 0.5 * (torch.rand(1, device=device) - 0.5)  # [0.75, 1.25]
                step_size = step_size * jitter
            
            # Take gradient descent step (on the loss, which maximizes expected logits)
            logits = logits - step_size * grad.clamp(-5, 5)
            
            # Add Langevin noise during training (helps exploration)
            if self.training:
                noise_scale = self.config.langevin_noise * (1.0 - step / steps)  # Decay noise
                logits = logits + noise_scale * torch.randn_like(logits)
            
            # Center logits (doesn't change softmax, helps stability)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            
            # Make logits require grad for next iteration
            logits = logits.requires_grad_(True)
            
            if return_trajectory:
                trajectory.append(logits.clone())
            
            # Early stopping based on convergence (tracking expected logits)
            current_expected_logits = expected_logits.detach().item()
            if prev_energy is not None:
                change = abs(prev_energy - current_expected_logits)
                if change < self.config.energy_convergence_threshold:
                    early_stop_patience += 1
                    if early_stop_patience >= 2:  # Stop if converged for 2 steps
                        break
                else:
                    early_stop_patience = 0
            prev_energy = current_expected_logits
        
        # Remember this refined prediction for future warm-starts
        if self.training and T == self.config.block_size:
            self.replay_buffer.add(logits[0].detach())
        
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
            # Get initial System 1 logits for energy tracking
            with torch.no_grad():
                initial_logits = self.system1_direct_energy(idx)
                
                # Compute initial expected energy
                # Energy = -logits (since logits = -energy in our formulation)
                # So expected energy = E_p[-logits] = -E_p[logits]
                probs_initial = F.softmax(initial_logits, dim=-1)
                expected_logits_initial = (probs_initial * initial_logits).sum(dim=-1).mean()
                metrics['initial_energy'] = -expected_logits_initial.item()
            
            # Get refined logits through System 2
            logits = self.system2_refine(idx, steps=refine_steps)
            
            # Compute final expected energy after refinement
            with torch.no_grad():
                probs_final = F.softmax(logits, dim=-1)
                expected_logits_final = (probs_final * logits).sum(dim=-1).mean()
                metrics['final_energy'] = -expected_logits_final.item()
                
                # Energy gap should be positive (initial - final) if System 2 improves
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
