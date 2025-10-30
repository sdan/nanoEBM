"""
Contrastive Divergence training for Energy-Based Models
Implements CD, PCD, and Fast-PCD with Langevin sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LangevinSampler:
    """
    Langevin dynamics sampler for generating negative samples from the energy model.
    Uses gradient descent in the energy landscape with added noise.
    """
    
    def __init__(self, energy_fn, step_size: float = 0.01, noise_scale: float = 0.005):
        """
        Args:
            energy_fn: Function that computes energy given input
            step_size: Step size for gradient descent
            noise_scale: Scale of Langevin noise
        """
        self.energy_fn = energy_fn
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def sample(self, x_init: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """
        Run Langevin dynamics from initial samples.
        
        Args:
            x_init: Initial samples (B, T, V) for logits or (B, T) for tokens
            n_steps: Number of Langevin steps
            
        Returns:
            Updated samples after Langevin dynamics
        """
        x = x_init.clone().requires_grad_(True)
        
        for _ in range(n_steps):
            # Compute energy and gradient
            energy = self.energy_fn(x)
            grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
            
            # Langevin update: x = x - step_size * grad + noise
            with torch.no_grad():
                x = x - self.step_size * grad
                
                # Add Langevin noise
                if self.noise_scale > 0:
                    noise = torch.randn_like(x) * (2 * self.step_size * self.noise_scale) ** 0.5
                    x = x + noise
            
            x = x.detach().requires_grad_(True)
        
        return x.detach()


class ContrastiveDivergenceLoss:
    """
    Basic Contrastive Divergence (CD-k) loss.
    Pushes down energy of real data, pushes up energy of model samples.
    """
    
    def __init__(self, energy_network, sampler: LangevinSampler, k: int = 1, model=None):
        """
        Args:
            energy_network: Function that computes energy
            sampler: Langevin sampler for generating negative samples
            k: Number of MCMC steps for CD-k
            model: The EBM model (needed to convert tokens to logits)
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.model = model
        self.last_negative_samples = None  # For visualization
    
    def __call__(self, positive_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute CD loss.
        
        Args:
            positive_samples: Real data samples (B, T) tokens
            
        Returns:
            Scalar CD loss
        """
        # Positive phase: energy of real data (tokens)
        positive_energy = self.energy_network(positive_samples)
        # Normalize by sequence length
        T = positive_samples.shape[1]

        # Helper to compute expected energy under logits given fixed token energies
        def expected_energy_from_logits(logits: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
            # logits, energies: (B, T, V)
            probs = F.softmax(logits, dim=-1)
            # sum over vocab then time -> (B,)
            return (probs * energies).sum(dim=-1).sum(dim=-1)

        # Negative phase: generate samples via MCMC
        # For token inputs, we need to work in logit space against fixed energies from current batch
        if positive_samples.dtype in [torch.long, torch.int32, torch.int64]:
            # Compute hidden states and energies for current batch
            h = self.model.get_hidden_states(positive_samples)
            energies = self.model.energy_head(h)  # (B, T, V)

            # Initialize logits from system-1
            logits_init = -energies.detach()  # detach for stable sampling

            # Build a per-call sampler that uses the batch's fixed energies for gradients wrt logits
            def energy_fn_logits(x):
                # Use detached energies for the sampling dynamics
                return expected_energy_from_logits(x, energies.detach())

            local_sampler = LangevinSampler(
                energy_fn=energy_fn_logits,
                step_size=self.sampler.step_size,
                noise_scale=self.sampler.noise_scale,
            )

            # Sample in logit space
            negative_logits = local_sampler.sample(logits_init, n_steps=self.k)
            self.last_negative_samples = negative_logits

            # Compute negative energy with gradients flowing to model params
            negative_energy = expected_energy_from_logits(negative_logits, energies)
        else:
            # Already in continuous space
            negative_samples = self.sampler.sample(positive_samples.detach(), n_steps=self.k)
            self.last_negative_samples = negative_samples
            negative_energy = self.energy_network(negative_samples)
        
        # Normalize by sequence length
        positive_energy = positive_energy / T
        negative_energy = negative_energy / T

        # CD loss: minimize positive energy, maximize negative energy
        loss = positive_energy.mean() - negative_energy.mean()
        
        return loss


class PersistentContrastiveDivergenceLoss:
    """
    Persistent Contrastive Divergence (PCD) loss.
    Maintains persistent negative samples across training steps for better mixing.
    """
    
    def __init__(self, energy_network, sampler: LangevinSampler, k: int = 1, 
                 n_persistent: int = 100, buffer_init_std: float = 1.0, model=None):
        """
        Args:
            energy_network: Function that computes energy
            sampler: Langevin sampler
            k: MCMC steps per update
            n_persistent: Number of persistent particles
            buffer_init_std: Std for buffer initialization
            model: The EBM model (needed for token to logit conversion)
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.n_persistent = n_persistent
        self.buffer_init_std = buffer_init_std
        self.model = model
        self.persistent_particles = None
        self.last_negative_samples = None
    
    def _initialize_buffer(self, sample_shape: Tuple, device: torch.device):
        """Initialize persistent particle buffer."""
        buffer_shape = (self.n_persistent,) + sample_shape[1:]
        self.persistent_particles = torch.randn(buffer_shape, device=device) * self.buffer_init_std
    
    def __call__(self, positive_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute PCD loss.
        
        Args:
            positive_samples: Real data samples (B, T) tokens
            
        Returns:
            Scalar PCD loss
        """
        device = positive_samples.device
        batch_size = positive_samples.shape[0]
        
        # Positive phase
        positive_energy = self.energy_network(positive_samples)
        T = positive_samples.shape[1]

        # Helper to compute expected energy under logits given fixed token energies
        def expected_energy_from_logits(logits: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            return (probs * energies).sum(dim=-1).sum(dim=-1)

        # For token inputs, work in logit space
        if positive_samples.dtype in [torch.long, torch.int32, torch.int64]:
            # Get logits for initialization if buffer is empty
            if self.persistent_particles is None:
                with torch.no_grad():
                    h = self.model.get_hidden_states(positive_samples[:1])  # Just one sample for shape
                    logits_shape = (-self.model.energy_head(h)).shape  # (1, T, V)
                    buffer_shape = (self.n_persistent,) + logits_shape[1:]  # (n_persistent, T, V)
                    self.persistent_particles = torch.randn(buffer_shape, device=device) * self.buffer_init_std

            # Compute batch energies from current positives
            h_batch = self.model.get_hidden_states(positive_samples)
            energies_batch = self.model.energy_head(h_batch)  # (B, T, V)

            # Negative phase: sample from persistent particles mapped to batch
            indices = torch.randint(0, self.n_persistent, (batch_size,))
            selected_particles = self.persistent_particles[indices].clone()  # (B, T, V)

            # Build per-call sampler using batch energies for gradient wrt logits
            def energy_fn_logits(x):
                return expected_energy_from_logits(x, energies_batch.detach())

            local_sampler = LangevinSampler(
                energy_fn=energy_fn_logits,
                step_size=self.sampler.step_size,
                noise_scale=self.sampler.noise_scale,
            )

            negative_logits = local_sampler.sample(selected_particles, n_steps=self.k)
            self.last_negative_samples = negative_logits

            # Update persistent buffer
            self.persistent_particles[indices] = negative_logits.detach()

            # Energy of fantasy samples with gradients to model params
            negative_energy = expected_energy_from_logits(negative_logits, energies_batch)
        else:
            # Initialize buffer if needed
            if self.persistent_particles is None:
                self._initialize_buffer(positive_samples.shape, device)

            # Standard PCD for continuous inputs
            indices = torch.randint(0, self.n_persistent, (batch_size,))
            selected_particles = self.persistent_particles[indices].clone()
            negative_samples = self.sampler.sample(selected_particles, n_steps=self.k)
            self.last_negative_samples = negative_samples
            self.persistent_particles[indices] = negative_samples.detach()
            negative_energy = self.energy_network(negative_samples)

        # Normalize by sequence length
        positive_energy = positive_energy / T
        negative_energy = negative_energy / T
        
        # PCD loss
        loss = positive_energy.mean() - negative_energy.mean()
        
        return loss


class FastPersistentContrastiveDivergenceLoss:
    """
    Fast PCD with parallel chains and random restarts for better mixing.
    """
    
    def __init__(self, energy_network, sampler: LangevinSampler, k: int = 1,
                 n_chains: int = 20, restart_prob: float = 0.01, 
                 buffer_init_std: float = 1.0, model=None):
        """
        Args:
            energy_network: Function that computes energy
            sampler: Langevin sampler
            k: MCMC steps per update
            n_chains: Number of parallel chains
            restart_prob: Probability of restarting a chain
            buffer_init_std: Std for buffer initialization
            model: The EBM model (needed for token to logit conversion)
        """
        self.energy_network = energy_network
        self.sampler = sampler
        self.k = k
        self.n_chains = n_chains
        self.restart_prob = restart_prob
        self.buffer_init_std = buffer_init_std
        self.model = model
        self.parallel_chains = None
        self.last_negative_samples = None
    
    def _initialize_chains(self, sample_shape: Tuple, device: torch.device):
        """Initialize parallel chains."""
        chain_shape = (self.n_chains,) + sample_shape[1:]
        self.parallel_chains = torch.randn(chain_shape, device=device) * self.buffer_init_std
    
    def __call__(self, positive_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute Fast PCD loss.
        
        Args:
            positive_samples: Real data samples (B, T) tokens
            
        Returns:
            Scalar Fast PCD loss
        """
        device = positive_samples.device
        batch_size = positive_samples.shape[0]
        
        # Positive phase
        positive_energy = self.energy_network(positive_samples)
        T = positive_samples.shape[1]

        # Helper to compute expected energy under logits given fixed token energies
        def expected_energy_from_logits(logits: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            return (probs * energies).sum(dim=-1).sum(dim=-1)

        # For token inputs, work in logit space
        if positive_samples.dtype in [torch.long, torch.int32, torch.int64]:
            # Initialize chains if needed (in logit space)
            if self.parallel_chains is None:
                with torch.no_grad():
                    h = self.model.get_hidden_states(positive_samples[:1])
                    logits_shape = (-self.model.energy_head(h)).shape  # (1, T, V)
                    chain_shape = (self.n_chains,) + logits_shape[1:]  # (n_chains, T, V)
                    self.parallel_chains = torch.randn(chain_shape, device=device) * self.buffer_init_std

            # Compute batch energies and expand/assign to chains for sampling dynamics
            h_batch = self.model.get_hidden_states(positive_samples)
            energies_batch = self.model.energy_head(h_batch)  # (B, T, V)
            # Assign each chain a random sample's energies
            assign = torch.randint(0, batch_size, (self.n_chains,), device=device)
            energies_chains = energies_batch[assign]  # (n_chains, T, V)

            # Random restarts for some chains
            restart_mask = torch.rand(self.n_chains) < self.restart_prob
            if restart_mask.any():
                n_restart = restart_mask.sum().item()
                if n_restart <= batch_size:
                    restart_indices = torch.randperm(batch_size)[:n_restart]
                    with torch.no_grad():
                        h_restart = self.model.get_hidden_states(positive_samples[restart_indices])
                        restart_logits = -self.model.energy_head(h_restart)
                    self.parallel_chains[restart_mask] = restart_logits.detach()
                else:
                    restart_samples = torch.randn_like(self.parallel_chains[restart_mask]) * self.buffer_init_std
                    self.parallel_chains[restart_mask] = restart_samples

            # Build per-call sampler for all chains
            def energy_fn_logits_all(x):  # x: (n_chains, T, V)
                return expected_energy_from_logits(x, energies_chains.detach())

            local_sampler = LangevinSampler(
                energy_fn=energy_fn_logits_all,
                step_size=self.sampler.step_size,
                noise_scale=self.sampler.noise_scale,
            )

            # Sample from all chains in logit space
            updated_chains = local_sampler.sample(self.parallel_chains, n_steps=self.k)
            self.parallel_chains = updated_chains.detach()

            # Select samples for this batch
            indices = torch.randint(0, self.n_chains, (batch_size,))
            negative_logits = updated_chains[indices]
            self.last_negative_samples = negative_logits

            # Negative phase - compute energy from logits with batch energies
            negative_energy = expected_energy_from_logits(negative_logits, energies_batch)
        else:
            # Standard Fast PCD for continuous inputs
            if self.parallel_chains is None:
                self._initialize_chains(positive_samples.shape, device)

            restart_mask = torch.rand(self.n_chains) < self.restart_prob
            if restart_mask.any():
                n_restart = restart_mask.sum().item()
                if n_restart <= batch_size:
                    restart_indices = torch.randperm(batch_size)[:n_restart]
                    self.parallel_chains[restart_mask] = positive_samples[restart_indices].detach()
                else:
                    restart_samples = torch.randn_like(self.parallel_chains[restart_mask]) * self.buffer_init_std
                    self.parallel_chains[restart_mask] = restart_samples

            updated_chains = self.sampler.sample(self.parallel_chains, n_steps=self.k)
            self.parallel_chains = updated_chains.detach()
            indices = torch.randint(0, self.n_chains, (batch_size,))
            negative_samples = updated_chains[indices]
            self.last_negative_samples = negative_samples
            negative_energy = self.energy_network(negative_samples)

        # Normalize by sequence length
        positive_energy = positive_energy / T
        negative_energy = negative_energy / T

        # Fast PCD loss
        loss = positive_energy.mean() - negative_energy.mean()
        
        return loss


# Helper function to create contrastive loss based on config
def create_contrastive_loss(model, config):
    """
    Factory function to create appropriate contrastive loss.
    
    Args:
        model: The EBM model
        config: Configuration object with contrastive settings
        
    Returns:
        Contrastive loss instance or None if disabled
    """
    if not getattr(config, 'use_contrastive', False):
        return None
    
    # Create energy function that uses the model's compute_energy method
    def energy_fn(inputs):
        """Compute energy using the model's energy head."""
        if inputs.dim() == 2:  # (B, T) tokens
            # Use model's compute_energy method for tokens
            return model.compute_energy(inputs)
        else:  # (B, T, V) logits
            # Use model's compute_energy_from_logits for logits
            return model.compute_energy_from_logits(inputs)
    
    # Create Langevin sampler
    sampler = LangevinSampler(
        energy_fn=energy_fn,
        step_size=getattr(config, 'langevin_step_size', 0.01),
        noise_scale=getattr(config, 'langevin_noise_scale', 0.005)
    )
    
    # Create appropriate loss based on config
    cd_type = getattr(config, 'contrastive_type', 'cd')
    cd_k = getattr(config, 'contrastive_k', 1)
    
    if cd_type == 'cd':
        return ContrastiveDivergenceLoss(energy_fn, sampler, k=cd_k, model=model)
    elif cd_type == 'pcd':
        return PersistentContrastiveDivergenceLoss(
            energy_fn, sampler, k=cd_k,
            n_persistent=getattr(config, 'n_persistent', 100),
            buffer_init_std=getattr(config, 'buffer_init_std', 1.0),
            model=model
        )
    elif cd_type == 'fast_pcd':
        return FastPersistentContrastiveDivergenceLoss(
            energy_fn, sampler, k=cd_k,
            n_chains=getattr(config, 'n_chains', 20),
            restart_prob=getattr(config, 'restart_prob', 0.01),
            buffer_init_std=getattr(config, 'buffer_init_std', 1.0),
            model=model
        )
    else:
        raise ValueError(f"Unknown contrastive type: {cd_type}")
