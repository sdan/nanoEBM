"""
Academic-style energy landscape visualization for EBM
Shows token generation with 3D/2D energy landscapes and gradient colors
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import chz

from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import CharDataset


@chz.chz
class AcademicVizConfig:
    checkpoint: str = "out_ebt/refine4.pt"
    prompt: str = "To be or not to be"
    max_tokens: int = 20
    refine_steps: int = 8

    # Visualization settings
    mode: str = "3d"  # 3d, 2d, or animated
    output_dir: str = "out_ebt/academic"
    dpi: int = 150
    figsize: tuple = (16, 10)

    # Color scheme
    colormap: str = "coolwarm"  # Blue (high energy) to Red (low energy)
    show_trajectory: bool = True
    fps: int = 2


def create_energy_colormap():
    """Create a custom colormap for energy visualization"""
    colors = ['#2E86AB', '#4ECDC4', '#95E1D3', '#F38181', '#AA2E2E']  # Blue to Red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('energy', colors, N=n_bins)
    return cmap


def compute_2d_energy_landscape(model, h, vocab_size, n_points=50):
    """
    Compute a 2D projection of the energy landscape using PCA on logits
    """
    device = h.device

    # Generate random logits around the current point
    center = -model.energy_head(h)[0, 0, :]  # Current logits

    # Create a grid of perturbations in 2D
    x = np.linspace(-10, 10, n_points)
    y = np.linspace(-10, 10, n_points)
    X, Y = np.meshgrid(x, y)

    # Initialize energy grid
    Z = np.zeros_like(X)

    # Use top 2 principal components for visualization
    with torch.no_grad():
        # Get top 2 varying dimensions
        logits_std = center.std()

        # Create two orthogonal directions in logit space
        dir1 = torch.randn(vocab_size, device=device)
        dir1 = dir1 / dir1.norm()

        dir2 = torch.randn(vocab_size, device=device)
        dir2 = dir2 - (dir2 @ dir1) * dir1  # Orthogonalize
        dir2 = dir2 / dir2.norm()

        # Compute energy for each point in the grid
        energies = model.energy_head(h)[0, 0, :]  # (vocab_size,)

        for i in range(n_points):
            for j in range(n_points):
                # Create perturbed logits
                perturbed = center + X[i, j] * dir1 * logits_std + Y[i, j] * dir2 * logits_std

                # Compute expected energy
                probs = F.softmax(perturbed, dim=-1)
                expected_energy = (probs * energies).sum().item()
                Z[i, j] = expected_energy

    return X, Y, Z, dir1, dir2


def plot_3d_energy_landscape(model, idx, step_idx, ax, config):
    """Plot 3D energy landscape with trajectory"""
    device = idx.device

    with torch.no_grad():
        h = model.get_hidden_states(idx)
        h_last = h[:, -1:, :]

        # Get 2D projection of energy landscape
        X, Y, Z, dir1, dir2 = compute_2d_energy_landscape(
            model, h_last, model.config.vocab_size, n_points=30
        )

        # Normalize Z for better visualization
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)

        # Create colormap
        cmap = create_energy_colormap()

        # Plot surface
        surf = ax.plot_surface(X, Y, Z_norm, cmap=cmap, alpha=0.8,
                               linewidth=0, antialiased=True, shade=True)

        # Run refinement and plot trajectory
        if config.show_trajectory:
            with torch.enable_grad():
                trajectory = model.system2_refine(
                    idx, steps=config.refine_steps, return_trajectory=True
                )

            if isinstance(trajectory, tuple):
                _, trajectory = trajectory

            if isinstance(trajectory, list) and len(trajectory) > 1:
                # Project trajectory points onto 2D space
                traj_x = []
                traj_y = []
                traj_z = []

                energies = model.energy_head(h_last)[0, 0, :]
                center = -energies
                logits_std = center.std() if center.dim() > 0 else torch.tensor(1.0)

                for logits in trajectory:
                    if logits.dim() == 3:
                        logits = logits[0, -1, :]
                    else:
                        logits = logits[0] if logits.dim() == 2 else logits

                    # Project onto 2D
                    diff = logits - center
                    x_coord = float((diff @ dir1).cpu().item()) / logits_std.cpu().item()
                    y_coord = float((diff @ dir2).cpu().item()) / logits_std.cpu().item()

                    # Compute energy at this point
                    probs = F.softmax(logits, dim=-1)
                    z_coord = float((probs * energies).sum().cpu().item())
                    z_coord = (z_coord - Z.min()) / (Z.max() - Z.min() + 1e-8)

                    traj_x.append(x_coord)
                    traj_y.append(y_coord)
                    traj_z.append(z_coord)

                # Plot trajectory as spheres descending
                for i, (x, y, z) in enumerate(zip(traj_x, traj_y, traj_z)):
                    color = cmap(1.0 - i / len(traj_x))  # Color gradient along trajectory
                    size = 100 * (1 + i / len(traj_x))  # Growing size
                    ax.scatter([x], [y], [z], color=color, s=size,
                             edgecolors='white', linewidth=2, alpha=0.9)

                # Connect with lines
                ax.plot(traj_x, traj_y, traj_z, 'w-', linewidth=2, alpha=0.5)

    # Set labels and title
    ax.set_xlabel('Logit Dimension 1', fontsize=10)
    ax.set_ylabel('Logit Dimension 2', fontsize=10)
    ax.set_zlabel('Energy', fontsize=10)
    ax.set_title(f'Energy Landscape at Token {step_idx}', fontsize=12, fontweight='bold')

    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)

    return surf


def generate_tokens_with_energy(model, prompt, dataset, max_tokens, refine_steps, device):
    """Generate tokens and track energy values"""
    stoi, itos = dataset.stoi, dataset.itos

    # Encode prompt
    tokens = [stoi[c] for c in prompt if c in stoi]
    generated_text = prompt
    energy_history = []
    token_history = [prompt]

    for _ in range(max_tokens):
        # Prepare input
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        idx = idx[:, -model.config.block_size:]  # Crop to block size

        # Get predictions with refinement
        with torch.enable_grad():
            _, logits, extras = model(idx, use_refine=True, refine_steps=refine_steps)

        # Sample next token
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token_idx = torch.multinomial(probs, 1).item()
        next_token = itos[next_token_idx]

        # Track energy
        energy = extras.get('final_energy', extras.get('initial_energy', 0))
        energy_gap = extras.get('energy_gap', 0)
        energy_history.append({
            'token': next_token,
            'energy': energy,
            'gap': energy_gap,
            'position': len(tokens)
        })

        # Update sequence
        tokens.append(next_token_idx)
        generated_text += next_token
        token_history.append(generated_text)

    return token_history, energy_history


def create_academic_visualization(config):
    """Create academic-style visualization with energy landscape"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=True)
    config_dict = checkpoint["config"]["model"]
    config_dict.pop('use_replay_buffer', None)  # Remove deprecated params
    model_cfg = ModelConfig(**config_dict)

    model = EBM(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load dataset
    dataset = CharDataset("shakespeare.txt", block_size=model_cfg.block_size, split="train")

    # Generate tokens with energy tracking
    print("Generating tokens with energy tracking...")
    token_history, energy_history = generate_tokens_with_energy(
        model, config.prompt, dataset, config.max_tokens,
        config.refine_steps, device
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    if config.mode == "3d":
        create_3d_visualization(
            model, dataset, token_history, energy_history, config, device
        )
    elif config.mode == "2d":
        create_2d_visualization(
            model, dataset, token_history, energy_history, config, device
        )
    elif config.mode == "animated":
        create_animated_visualization(
            model, dataset, token_history, energy_history, config, device
        )


def create_3d_visualization(model, dataset, token_history, energy_history, config, device):
    """Create static 3D visualization panels"""
    n_panels = min(6, len(energy_history))
    indices = np.linspace(0, len(energy_history)-1, n_panels, dtype=int)

    fig = plt.figure(figsize=config.figsize)
    gs = gridspec.GridSpec(2, n_panels//2 + 1, width_ratios=[1]*3 + [0.3])

    # Create colormap for energy
    cmap = create_energy_colormap()

    for i, idx in enumerate(indices):
        row = i // 3
        col = i % 3

        ax = fig.add_subplot(gs[row, col], projection='3d')

        # Prepare input up to this point
        text_so_far = token_history[idx]
        stoi = dataset.stoi
        tokens = [stoi[c] for c in text_so_far if c in stoi]
        input_idx = torch.tensor([tokens], dtype=torch.long, device=device)

        # Plot energy landscape
        surf = plot_3d_energy_landscape(model, input_idx, idx, ax, config)

        # Add token info
        current_token = energy_history[idx]['token'] if idx < len(energy_history) else ""
        energy_val = energy_history[idx]['energy'] if idx < len(energy_history) else 0
        gap = energy_history[idx]['gap'] if idx < len(energy_history) else 0

        # Text box with generation info
        textstr = f"Token: '{current_token}'\nE: {energy_val:.3f}\nGap: {gap:.3f}"
        ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Energy (Low → High)', fontsize=12)

    # Main title
    fig.suptitle(f'Energy-Based Model: Token Generation Dynamics\nPrompt: "{config.prompt}"',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_path = os.path.join(config.output_dir, "energy_landscape_3d.png")
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {output_path}")


def create_2d_visualization(model, dataset, token_history, energy_history, config, device):
    """Create 2D visualization with token sequence and energy profile"""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1])

    # Color scheme
    cmap = create_energy_colormap()

    # Panel 1: Token sequence with energy coloring
    ax1 = fig.add_subplot(gs[0, :2])

    # Display tokens with energy-based coloring
    energies = [e['energy'] for e in energy_history]
    if energies:
        norm_energies = (np.array(energies) - min(energies)) / (max(energies) - min(energies) + 1e-8)
    else:
        norm_energies = [0]

    y_pos = 0.5
    for i, hist in enumerate(energy_history[:15]):  # Show first 15 tokens
        token = hist['token']
        energy_norm = norm_energies[i]
        color = cmap(1 - energy_norm)  # Invert so low energy is red

        ax1.text(i * 0.06 + 0.05, y_pos, repr(token),
                fontsize=14, color=color, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white', edgecolor=color, linewidth=2))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Generated Token Sequence (colored by energy)', fontsize=12, fontweight='bold')

    # Panel 2: Energy trajectory
    ax2 = fig.add_subplot(gs[1, 0])

    positions = [e['position'] for e in energy_history]
    energies = [e['energy'] for e in energy_history]
    gaps = [e['gap'] for e in energy_history]

    ax2.plot(positions, energies, 'b-', linewidth=2, label='Energy', marker='o', markersize=6)
    ax2.fill_between(positions, min(energies), energies, alpha=0.3, color='blue')

    ax2.set_xlabel('Token Position', fontsize=11)
    ax2.set_ylabel('Energy', fontsize=11)
    ax2.set_title('Energy Evolution During Generation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Energy gap (refinement improvement)
    ax3 = fig.add_subplot(gs[1, 1])

    ax3.bar(positions, gaps, color=cmap(0.3), alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Token Position', fontsize=11)
    ax3.set_ylabel('Energy Gap (S1 - S2)', fontsize=11)
    ax3.set_title('Refinement Impact per Token', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Statistics box
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')

    stats_text = f"""Model Statistics
    ━━━━━━━━━━━━━━━

    Refinement Steps: {config.refine_steps}
    Alpha: {model.alpha.item():.3f}

    Generation Stats:
    • Tokens: {len(energy_history)}
    • Avg Energy: {np.mean(energies):.3f}
    • Avg Gap: {np.mean(gaps):.4f}
    • Max Gap: {max(gaps):.4f}

    System Performance:
    • S1 → S2 Improvement
    • {100 * np.mean([g > 0 for g in gaps]):.1f}% positive gaps
    """

    ax4.text(0.1, 0.9, stats_text, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

    # Main title
    fig.suptitle(f'Energy-Based Language Model: "{config.prompt}..."',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(config.output_dir, "energy_landscape_2d.png")
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D visualization to {output_path}")


def create_animated_visualization(model, dataset, token_history, energy_history, config, device):
    """Create animated visualization of token generation"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[2, 1])

    # 3D energy landscape
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')

    # Token display
    ax_tokens = fig.add_subplot(gs[0, 1])

    # Energy trajectory
    ax_energy = fig.add_subplot(gs[1, :])

    cmap = create_energy_colormap()

    def init():
        ax_3d.clear()
        ax_tokens.clear()
        ax_energy.clear()
        return []

    def update(frame):
        # Clear axes
        ax_3d.clear()
        ax_tokens.clear()
        ax_energy.clear()

        # Get current state
        current_text = token_history[min(frame, len(token_history)-1)]
        stoi = dataset.stoi
        tokens = [stoi[c] for c in current_text if c in stoi]
        input_idx = torch.tensor([tokens], dtype=torch.long, device=device)

        # Plot 3D landscape
        if frame < len(energy_history):
            plot_3d_energy_landscape(model, input_idx, frame, ax_3d, config)

        # Display tokens
        ax_tokens.axis('off')
        display_text = current_text[-30:]  # Show last 30 chars
        ax_tokens.text(0.5, 0.5, display_text, fontsize=12,
                      ha='center', va='center', wrap=True,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if frame < len(energy_history):
            current_energy = energy_history[frame]
            info_text = f"Token: '{current_energy['token']}'\n"
            info_text += f"Energy: {current_energy['energy']:.3f}\n"
            info_text += f"Gap: {current_energy['gap']:.4f}"
            ax_tokens.text(0.5, 0.2, info_text, fontsize=10,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # Plot energy trajectory up to current point
        if frame > 0:
            positions = [e['position'] for e in energy_history[:frame+1]]
            energies = [e['energy'] for e in energy_history[:frame+1]]
            ax_energy.plot(positions, energies, 'b-', linewidth=2, marker='o')
            ax_energy.fill_between(positions, min(energies), energies, alpha=0.3)
            ax_energy.set_xlabel('Token Position')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Trajectory')
            ax_energy.grid(True, alpha=0.3)

        fig.suptitle(f'Token Generation with Energy Dynamics - Step {frame+1}',
                    fontsize=14, fontweight='bold')

        return [ax_3d, ax_tokens, ax_energy]

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(energy_history),
                        interval=1000//config.fps, blit=False)

    # Save animation
    output_path = os.path.join(config.output_dir, "energy_landscape_animated.gif")
    writer = PillowWriter(fps=config.fps)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    config = chz.entrypoint(AcademicVizConfig)
    create_academic_visualization(config)