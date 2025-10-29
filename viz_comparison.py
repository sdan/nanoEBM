"""
Simple comparison visualization: Autoregressive vs Energy-Based Model
Shows token generation side-by-side with energy bowl animation

Usage:
  python viz_comparison.py --checkpoint=out_ebt/refine4.pt --prompt="ROMEO:"
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import chz

from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.transformer import Transformer
from nanoebm.data import CharDataset


@chz.chz
class ComparisonVizConfig:
    checkpoint: str = "out_ebt/refine4.pt"
    prompt: str = "ROMEO:"
    max_tokens: int = 20
    refine_steps: int = 8
    
    # Output settings
    output_path: str = "out_ebt/comparison_animated.gif"
    fps: int = 2
    dpi: int = 120


def create_energy_colormap():
    """Simple blue (high energy) to red (low energy) colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2E86AB', '#4ECDC4', '#95E1D3', '#F38181', '#AA2E2E']
    return LinearSegmentedColormap.from_list('energy', colors, N=100)


@torch.no_grad()
def generate_with_transformer(model, prompt_tokens, max_tokens, dataset, device):
    """Generate tokens using standard autoregressive transformer"""
    tokens = prompt_tokens.copy()
    generated_chars = []
    
    for _ in range(max_tokens):
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        idx = idx[:, -model.config.block_size:]
        
        logits, _ = model(idx)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        tokens.append(next_token)
        generated_chars.append(dataset.itos[next_token])
    
    return generated_chars


@torch.no_grad()
def generate_with_ebm(model, prompt_tokens, max_tokens, refine_steps, dataset, device):
    """Generate tokens using EBM with energy tracking"""
    tokens = prompt_tokens.copy()
    generated_chars = []
    energy_traces = []  # Energy descent for each token
    final_energies = []  # Final energy of each token
    
    for _ in range(max_tokens):
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        idx = idx[:, -model.config.block_size:]
        
        # Get energy trace during refinement
        h = model.get_hidden_states(idx)
        energies = model.energy_head(h)[:, -1, :]  # (1, V)
        
        # Run refinement and track energy
        with torch.enable_grad():
            refined_logits = model.system2_refine(idx, steps=refine_steps, return_trajectory=True)
        
        if isinstance(refined_logits, tuple):
            refined_logits, trajectory = refined_logits
        else:
            trajectory = [refined_logits]
        
        # Compute energy trace from trajectory
        energy_trace = []
        for step_logits in trajectory:
            if step_logits.dim() == 3:
                step_logits = step_logits[0, -1, :]
            else:
                step_logits = step_logits[0] if step_logits.dim() == 2 else step_logits
            
            probs = F.softmax(step_logits, dim=-1)
            expected_energy = (probs * energies[0]).sum().item()
            energy_trace.append(expected_energy)
        
        energy_traces.append(energy_trace)
        final_energies.append(energy_trace[-1] if energy_trace else 0)
        
        # Sample from refined distribution
        if refined_logits.dim() == 3:
            final_logits = refined_logits[0, -1, :]
        else:
            final_logits = refined_logits[0] if refined_logits.dim() == 2 else refined_logits
        
        probs = F.softmax(final_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        tokens.append(next_token)
        generated_chars.append(dataset.itos[next_token])
    
    return generated_chars, energy_traces, final_energies


def plot_energy_bowl(ax, energy_trace, step_idx, global_energy_range, title="Energy Descent"):
    """Plot energy bowl with descending dot - using global energy scale"""
    ax.clear()
    
    energy_trace = np.array(energy_trace)
    e_min_global, e_max_global = global_energy_range
    
    # Normalize using global energy range across all tokens
    rng = max(1e-8, e_max_global - e_min_global)
    energy_norm = (energy_trace - e_min_global) / rng
    
    # Map energy to x position in bowl (high energy = left, low energy = center)
    x_positions = -0.7 * np.sqrt(np.clip(energy_norm, 0.0, 1.0))
    
    # Bowl curve
    x_curve = np.linspace(-0.9, 0.9, 200)
    bowl_curve = x_curve ** 2
    
    # Plot bowl
    ax.plot(x_curve, bowl_curve, color='#2E86AB', linewidth=2.5, alpha=0.7)
    ax.fill_between(x_curve, 0, bowl_curve, color='#2E86AB', alpha=0.1)
    
    # Plot current position
    if step_idx < len(x_positions):
        x_pos = x_positions[step_idx]
        y_pos = energy_norm[step_idx] * 0.49
        ax.scatter([x_pos], [y_pos], s=120, color='#A23B72', 
                  zorder=5, edgecolor='white', linewidth=2)
        
        # Add energy value
        ax.text(0, 0.7, f'{energy_trace[step_idx]:.2f}', 
               ha='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.02, 0.8)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
    ax.axis('off')


def create_comparison_animation(config):
    """Create animated comparison visualization"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load EBM model
    print(f"Loading checkpoint: {config.checkpoint}")
    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=True)
    
    config_dict = ckpt["config"]["model"]
    config_dict.pop('use_replay_buffer', None)
    model_cfg = ModelConfig(**config_dict)
    
    ebm_model = EBM(model_cfg).to(device)
    ebm_model.load_state_dict(ckpt["model"])
    ebm_model.eval()
    
    # Create standard transformer for comparison
    transformer_model = Transformer(model_cfg).to(device)
    # Copy weights from EBM's transformer
    transformer_model.load_state_dict(ebm_model.transformer.state_dict())
    transformer_model.eval()
    
    # Load dataset
    dataset = CharDataset("shakespeare.txt", block_size=model_cfg.block_size, split="train")
    
    # Encode prompt
    stoi = dataset.stoi
    prompt_tokens = [stoi[c] for c in config.prompt if c in stoi]
    print(f"Prompt: '{config.prompt}' ({len(prompt_tokens)} tokens)")
    
    # Generate with both models
    print("Generating with transformer...")
    ar_chars = generate_with_transformer(transformer_model, prompt_tokens, 
                                        config.max_tokens, dataset, device)
    
    print("Generating with EBM...")
    ebm_chars, energy_traces, final_energies = generate_with_ebm(
        ebm_model, prompt_tokens, config.max_tokens, 
        config.refine_steps, dataset, device
    )
    
    # Normalize energies for coloring (relative across sequence)
    if final_energies:
        energy_array = np.array(final_energies)
        e_min, e_max = energy_array.min(), energy_array.max()
        energy_range = max(1e-8, e_max - e_min)
        normalized_energies = (energy_array - e_min) / energy_range
    else:
        normalized_energies = np.zeros(len(ebm_chars))
    
    # Compute global energy range for consistent bowl scaling
    all_energies = []
    for trace in energy_traces:
        all_energies.extend(trace)
    global_energy_range = (min(all_energies), max(all_energies))
    
    # Create figure - emphasis on tokens
    fig = plt.figure(figsize=(16, 7))
    
    # Create grid: large token displays, small bowl and params
    gs = fig.add_gridspec(3, 2, width_ratios=[4.5, 1], height_ratios=[1.2, 1.2, 0.35],
                         hspace=0.25, wspace=0.25)
    
    ax_ar = fig.add_subplot(gs[0, 0])     # Autoregressive tokens (larger)
    ax_ebm = fig.add_subplot(gs[1, 0])    # EBM tokens (larger)
    ax_bowl = fig.add_subplot(gs[0:2, 1]) # Energy bowl (smaller)
    ax_params = fig.add_subplot(gs[2, :]) # Model parameters (smaller)
    
    # Colormap for energy
    cmap = create_energy_colormap()
    
    def init():
        ax_ar.clear()
        ax_ebm.clear()
        ax_bowl.clear()
        ax_params.clear()
        return []
    
    def update(frame):
        ax_ar.clear()
        ax_ebm.clear()
        
        # Set up token display axes
        for ax in [ax_ar, ax_ebm]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Display prompt
        prompt_text = config.prompt
        y_pos = 0.5
        
        # Autoregressive row - gray tokens (BIGGER)
        ax_ar.text(0.01, y_pos, prompt_text, fontsize=16, color='gray', 
                  fontweight='bold', family='monospace')
        
        x_offset = 0.01 + len(prompt_text) * 0.018
        for i in range(min(frame + 1, len(ar_chars))):
            char = ar_chars[i]
            ax_ar.text(x_offset, y_pos, char, fontsize=16, color='gray',
                      family='monospace', fontweight='bold')
            x_offset += 0.018
        
        ax_ar.text(0.5, 0.88, 'Autoregressive', 
                  ha='center', fontsize=12, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.7))
        
        # EBM row - colored by energy (BIGGER)
        ax_ebm.text(0.01, y_pos, prompt_text, fontsize=16, color='gray',
                   fontweight='bold', family='monospace')
        
        x_offset = 0.01 + len(prompt_text) * 0.018
        for i in range(min(frame + 1, len(ebm_chars))):
            char = ebm_chars[i]
            # Color by relative energy (inverted: low energy = red/warm)
            color = cmap(1.0 - normalized_energies[i])
            ax_ebm.text(x_offset, y_pos, char, fontsize=16, color=color,
                       family='monospace', fontweight='bold')
            x_offset += 0.018
        
        ax_ebm.text(0.5, 0.88, 'Energy-Based Model', 
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))
        
        # Energy legend (smaller)
        ax_ebm.text(0.5, 0.06, 'Blue (high) → Red (low)', 
                   ha='center', fontsize=8, style='italic', color='gray')
        
        # Energy bowl - show descent for current token
        if frame < len(energy_traces):
            plot_energy_bowl(ax_bowl, energy_traces[frame], 
                           len(energy_traces[frame]) - 1,
                           global_energy_range,
                           f"Token {frame + 1}")
        else:
            ax_bowl.clear()
            ax_bowl.axis('off')
        
        # Model parameters panel (minimal)
        ax_params.clear()
        ax_params.axis('off')
        
        # Simple config display
        param_text = f"Config: {model_cfg.n_layer}L {model_cfg.n_head}H {model_cfg.n_embd}D  |  Refine: {config.refine_steps} steps @ α={ebm_model.alpha.item():.3f}  |  Energy: [{np.min(final_energies):.2f}, {np.max(final_energies):.2f}]"
        
        ax_params.text(0.5, 0.5, param_text, ha='center', va='center',
                      fontsize=8, family='monospace', color='#666')
        
        # Main title
        fig.suptitle(f'Autoregressive vs Energy-Based Generation (Step {frame + 1}/{len(ebm_chars)})',
                    fontsize=13, fontweight='bold')
        
        return [ax_ar, ax_ebm, ax_bowl, ax_params]
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(ebm_chars), interval=1000//config.fps, 
                        blit=False, repeat=True)
    
    # Save as GIF
    os.makedirs(os.path.dirname(config.output_path) or '.', exist_ok=True)
    writer = PillowWriter(fps=config.fps)
    print(f"Saving animation to {config.output_path}...")
    anim.save(config.output_path, writer=writer, dpi=config.dpi)
    plt.close()
    
    print(f"✓ Saved animation to {config.output_path}")
    print(f"✓ Generated {len(ebm_chars)} tokens")
    print(f"✓ Autoregressive: '{config.prompt}{''.join(ar_chars)}'")
    print(f"✓ EBM: '{config.prompt}{''.join(ebm_chars)}'")


if __name__ == "__main__":
    config = chz.entrypoint(ComparisonVizConfig)
    create_comparison_animation(config)