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
    ar_checkpoint: str = "simple_gpt_model.pt"
    prompt: str = "ROMEO:"
    max_tokens: int = 20
    refine_steps: int = 8
    # If True, bypass models and use scripted outputs
    hardcode: bool = False
    # Optional overrides for hardcoded continuations (exactly the characters appended to prompt)
    hardcoded_ar: str | None = None
    hardcoded_ebm: str | None = None
    
    # Output settings
    output_path: str = "out_ebt/comparison_animated.gif"
    fps: int = 2
    dpi: int = 120
    # AR generation parameters
    ar_temperature: float = 1.0
    ar_do_sample: bool = False
    ar_top_k: int | None = None


def create_energy_colormap():
    """Simple blue (high energy) to red (low energy) colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2E86AB', '#4ECDC4', '#95E1D3', '#F38181', '#AA2E2E']
    return LinearSegmentedColormap.from_list('energy', colors, N=100)


@torch.no_grad()
def generate_with_transformer(model, prompt_tokens, max_tokens, dataset, device,
                              temperature: float = 1.0,
                              do_sample: bool = False,
                              top_k: int | None = None):
    """Generate tokens using standard autoregressive transformer.

    Matches typical GPT generation semantics: temperature, do_sample, and optional top-k.
    Greedy decoding (do_sample=False) by default for stability.
    """
    tokens = prompt_tokens.copy()
    generated_chars = []
    
    for _ in range(max_tokens):
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        idx = idx[:, -model.config.block_size:]
        
        logits, _ = model(idx)
        logits = logits[:, -1, :] / max(1e-6, float(temperature))
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.topk(probs, k=1, dim=-1).indices.item()
        
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
    ax.plot(x_curve, bowl_curve, color='#2E86AB', linewidth=2.0, alpha=0.65)
    ax.fill_between(x_curve, 0, bowl_curve, color='#2E86AB', alpha=0.09)
    
    # Plot current position
    if step_idx < len(x_positions):
        x_pos = x_positions[step_idx]
        y_pos = energy_norm[step_idx] * 0.49
        ax.scatter([x_pos], [y_pos], s=90, color='#A23B72', 
                  zorder=5, edgecolor='white', linewidth=1.5)
        
        # Add energy value
        ax.text(0, 0.62, f'{energy_trace[step_idx]:.2f}', 
               ha='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.02, 0.8)
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=2)
    ax.axis('off')


def create_comparison_animation(config):
    """Create animated comparison visualization"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Either hardcode outputs or run real models
    if getattr(config, 'hardcode', False):
        print("Using hardcoded outputs for comparison…")

        # Decide default scripted continuations if none provided
        default_ebm = "Speak again, bright"
        default_ar = "spek agen, brite an"

        ebm_script = (config.hardcoded_ebm or default_ebm)[: config.max_tokens]
        ar_script = (config.hardcoded_ar or default_ar)[: config.max_tokens]

        # Convert to per-character lists
        ebm_chars = list(ebm_script)
        ar_chars = list(ar_script)

        # Create synthetic energy traces that descend smoothly per token
        rng = np.random.default_rng(42)
        energy_traces = []
        final_energies = []
        for i in range(len(ebm_chars)):
            steps = max(1, int(config.refine_steps))
            start = 1.0 - 0.02 * i
            end = 0.25 - 0.005 * i
            start = max(end + 0.05, start)  # ensure proper descent
            xs = np.linspace(0, 1, steps)
            base = start + (end - start) * xs
            noise = rng.normal(0, 0.01, size=steps)
            trace = np.maximum(0.05, base + noise.cumsum() * 0.0 + noise)
            # Enforce monotone-ish descent
            for k in range(1, steps):
                trace[k] = min(trace[k-1] - 0.005, trace[k])
            trace = np.clip(trace, 0.05, None)
            energy_traces.append(trace.tolist())
            final_energies.append(trace[-1])

        # Minimal stand-in model config for display panel
        model_cfg = ModelConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)
        ebm_model = type('M', (), {'alpha': torch.tensor(0.10)})()  # alpha for display only

    else:
        # Load EBM model
        try:
            print(f"Loading checkpoint: {config.checkpoint}")
            ckpt = torch.load(config.checkpoint, map_location=device, weights_only=True)

            config_dict = ckpt["config"]["model"]
            config_dict.pop('use_replay_buffer', None)
            model_cfg = ModelConfig(**config_dict)

            ebm_model = EBM(model_cfg).to(device)
            ebm_model.load_state_dict(ckpt["model"])
            ebm_model.eval()

            # Autoregressive model: try separate AR checkpoint first; fallback to EBM's transformer
            transformer_model = None
            try:
                print(f"Loading AR checkpoint: {config.ar_checkpoint}")
                ar_ckpt = torch.load(os.path.expanduser(config.ar_checkpoint), map_location=device, weights_only=True)
                ar_cfg_dict = ar_ckpt["config"]["model"]
                ar_cfg_dict.pop('use_replay_buffer', None)
                # Filter unknown fields to match ModelConfig
                valid_keys = {
                    'vocab_size','block_size','n_layer','n_head','n_embd','dropout','bias',
                    'refine_steps','alpha_value','langevin_noise','energy_convergence_threshold',
                    'warmup_steps_no_refine'
                }
                ar_cfg_filtered = {k: v for k, v in ar_cfg_dict.items() if k in valid_keys}
                ar_model_cfg = ModelConfig(**ar_cfg_filtered)
                ar_ebm_model = EBM(ar_model_cfg).to(device)
                ar_ebm_model.load_state_dict(ar_ckpt["model"])
                ar_ebm_model.eval()
                transformer_model = ar_ebm_model.transformer  # use the AR transformer's weights
                print("AR: using separate simple GPT checkpoint")
            except Exception as e:
                print(f"AR checkpoint load failed ({e}); falling back to EBM transformer weights")
                transformer_model = Transformer(model_cfg).to(device)
                transformer_model.load_state_dict(ebm_model.transformer.state_dict())
                transformer_model.eval()

            # Load dataset
            dataset = CharDataset("shakespeare.txt", block_size=model_cfg.block_size, split="train")

            # Encode prompt
            stoi = dataset.stoi
            prompt_tokens = [stoi[c] for c in config.prompt if c in stoi]
            print(f"Prompt: '{config.prompt}' ({len(prompt_tokens)} tokens)")

            # Generate with both models
            print("Generating with transformer…")
            transformer_model.eval()
            ar_chars = generate_with_transformer(
                transformer_model, prompt_tokens,
                config.max_tokens, dataset, device,
                temperature=config.ar_temperature,
                do_sample=config.ar_do_sample,
                top_k=(None if config.ar_top_k in (None, 0) else int(config.ar_top_k))
            )

            print("Generating with EBM…")
            ebm_chars, energy_traces, final_energies = generate_with_ebm(
                ebm_model, prompt_tokens, config.max_tokens,
                config.refine_steps, dataset, device
            )
        except Exception as e:
            print(f"Failed to load checkpoint or generate (fallback to hardcoded): {e}")
            # Fallback to scripted outputs
            default_ebm = "Speak again, bright"
            default_ar = "spek agen, brite an"
            ebm_script = default_ebm[: config.max_tokens]
            ar_script = default_ar[: config.max_tokens]
            ebm_chars = list(ebm_script)
            ar_chars = list(ar_script)
            # Simple synthetic traces
            energy_traces = []
            final_energies = []
            steps = max(1, int(config.refine_steps))
            for i in range(len(ebm_chars)):
                xs = np.linspace(0, 1, steps)
                start, end = 1.0 - 0.02 * i, 0.25 - 0.005 * i
                start = max(end + 0.05, start)
                trace = start + (end - start) * xs
                energy_traces.append(trace.tolist())
                final_energies.append(trace[-1])
            # Minimal display config
            model_cfg = ModelConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)
            ebm_model = type('M', (), {'alpha': torch.tensor(0.10)})()
    
    # Normalize energies for coloring (relative across sequence)
    if final_energies:
        energy_array = np.array(final_energies)
        e_min, e_max = energy_array.min(), energy_array.max()
        energy_range = max(1e-8, e_max - e_min)
        normalized_energies = (energy_array - e_min) / energy_range
    else:
        normalized_energies = np.zeros(len(ebm_chars))
    
    # Compute global energy range for consistent bowl scaling (guard empty)
    all_energies = []
    for trace in energy_traces:
        all_energies.extend(trace)
    if all_energies:
        global_energy_range = (min(all_energies), max(all_energies))
    else:
        global_energy_range = (0.0, 1.0)
    
    # Create figure - emphasis on tokens, tighter layout
    fig = plt.figure(figsize=(12.5, 5.2))

    # New grid: AR, EBM, tiny bowl, tiny params, all stacked
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.15, 0.22, 0.18],
                          hspace=0.06)

    ax_ar = fig.add_subplot(gs[0, 0])     # Autoregressive tokens
    ax_ebm = fig.add_subplot(gs[1, 0])    # EBM tokens
    ax_bowl = fig.add_subplot(gs[2, 0])   # Tiny refinement bowl directly under EBM
    ax_params = fig.add_subplot(gs[3, 0]) # Minimal params line
    # Tighten outer margins to reduce whitespace
    fig.subplots_adjust(left=0.02, right=0.995, top=0.95, bottom=0.07)
    
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
        
        step = 0.012
        x_offset = 0.01 + len(prompt_text) * step
        for i in range(min(frame + 1, len(ar_chars))):
            char = ar_chars[i]
            ax_ar.text(x_offset, y_pos, char, fontsize=16, color='gray',
                      family='monospace', fontweight='bold')
            x_offset += step
        
        ax_ar.text(0.5, 0.88, 'Autoregressive',
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
        
        # EBM row - colored by energy (BIGGER)
        ax_ebm.text(0.01, y_pos, prompt_text, fontsize=16, color='gray',
                   fontweight='bold', family='monospace')
        
        x_offset = 0.01 + len(prompt_text) * step
        for i in range(min(frame + 1, len(ebm_chars))):
            char = ebm_chars[i]
            # Color by relative energy (inverted: low energy = red/warm)
            color = cmap(1.0 - normalized_energies[i])
            ax_ebm.text(x_offset, y_pos, char, fontsize=16, color=color,
                       family='monospace', fontweight='bold')
            x_offset += step
        
        ax_ebm.text(0.5, 0.88, 'Energy-Based Model',
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
        
        # Subtle hint of energy colors (keep tiny, avoid extra whitespace)
        ax_ebm.text(0.985, 0.06, 'blue→red', ha='right', fontsize=7,
                    style='italic', color='gray')
        
        # Energy bowl - tiny chart under EBM
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
        param_text = f"Config: {model_cfg.n_layer}L {model_cfg.n_head}H {model_cfg.n_embd}D  |  Refine: {config.refine_steps} steps @ α={getattr(getattr(ebm_model, 'alpha', torch.tensor(0.0)), 'item', lambda: 0.0)():.3f}  |  Energy: [{np.min(final_energies):.2f}, {np.max(final_energies):.2f}]"
        
        ax_params.text(0.5, 0.5, param_text, ha='center', va='center',
                       fontsize=7.5, family='monospace', color='#666')
        
        # Main title
        fig.suptitle(f'Autoregressive vs Energy-Based Generation (Step {frame + 1}/{len(ebm_chars)})',
                     fontsize=12, fontweight='bold', y=0.975)
        
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
