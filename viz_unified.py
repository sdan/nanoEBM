"""Unified Energy Landscape Visualization for EBM

Shows the energy landscape as a unified surface that System 2 traverses during refinement.

Usage:
  python viz_unified.py --checkpoint=out_ebt/final.pt --mode=landscape
  python viz_unified.py --checkpoint=out_ebt/final.pt --mode=trajectory
  python viz_unified.py --checkpoint=out_ebt/final.pt --mode=distribution
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import chz
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import CharDataset, get_loader


@chz.chz
class VizUnifiedConfig:
    checkpoint: str | None = None
    data_path: str = "shakespeare.txt"
    prompt: str = "ROMEO:"
    mode: str = "landscape"  # landscape|trajectory|distribution
    steps: int = 8
    batch_size: int = 1
    # Output paths
    out_dir: str = "out_ebt"
    out_landscape: str = "unified_landscape.png"
    out_trajectory: str = "unified_trajectory.png"
    out_distribution: str = "unified_distribution.png"


def find_latest_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest checkpoint."""
    import glob
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")
    
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]
    
    checkpoint_path = os.path.join(latest_dir, "final.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if pt_files:
        pt_files.sort(key=os.path.getmtime, reverse=True)
        return pt_files[0]
    
    raise FileNotFoundError(f"No .pt files found in {latest_dir}")


def get_refinement_trajectory(model: EBM, idx: torch.Tensor, steps: int):
    """Get the trajectory of logits and energies during refinement."""
    device = idx.device
    
    # Get hidden states and fixed energies
    with torch.no_grad():
        h = model.get_hidden_states(idx)
        energies = model.energy_head(h)[:, -1, :]  # Last position (B, V)
    
    # Initialize from System 1
    logits = -energies.clone().detach()
    trajectory_logits = [logits.cpu().numpy()]
    trajectory_energies = []
    
    # Track refinement steps - need to enable grad for this part
    for step in range(steps):
        # Enable gradients for this computation
        with torch.enable_grad():
            logits = logits.requires_grad_(True)
            probs = F.softmax(logits, dim=-1)
            expected_energy = (probs * energies.detach()).sum(dim=-1).mean()
            trajectory_energies.append(expected_energy.item())
            
            # Gradient step
            grad = torch.autograd.grad(expected_energy, logits)[0]
            logits = logits - model.alpha * grad.clamp(-5, 5)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            logits = logits.detach()
        
        trajectory_logits.append(logits.cpu().numpy())
    
    # Final energy
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        final_energy = (probs * energies).sum(dim=-1).mean()
        trajectory_energies.append(final_energy.item())
    
    return trajectory_logits, trajectory_energies, energies.cpu().numpy()


def plot_unified_landscape(model: EBM, ds: CharDataset, cfg: VizUnifiedConfig):
    """Visualize the energy landscape as a unified surface with gradient descent trajectory."""
    device = next(model.parameters()).device
    
    # Encode prompt
    stoi = ds.stoi
    ids = [stoi[c] for c in cfg.prompt if c in stoi]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    idx = idx[:, -model.config.block_size:]
    
    # Get trajectory
    traj_logits, traj_energies, fixed_energies = get_refinement_trajectory(model, idx, cfg.steps)
    
    # Project to 2D using PCA on the trajectory
    traj_array = np.vstack(traj_logits).squeeze()  # (steps+1, vocab_size)
    
    # Use PCA to find the 2D subspace that best captures the trajectory
    pca = PCA(n_components=2)
    traj_2d = pca.fit_transform(traj_array)
    
    # Create a grid in the 2D space for the landscape
    x_min, x_max = traj_2d[:, 0].min() - 2, traj_2d[:, 0].max() + 2
    y_min, y_max = traj_2d[:, 1].min() - 2, traj_2d[:, 1].max() + 2
    
    n_grid = 50
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid)
    )
    
    # Compute energy at each grid point
    # For each point in 2D, reconstruct logits and compute expected energy
    grid_energies = np.zeros((n_grid, n_grid))
    
    for i in range(n_grid):
        for j in range(n_grid):
            # Project back to logit space
            point_2d = np.array([[xx[i, j], yy[i, j]]])
            logits_reconstructed = pca.inverse_transform(point_2d)
            
            # Compute expected energy
            probs = F.softmax(torch.tensor(logits_reconstructed, dtype=torch.float32), dim=-1)
            expected_energy = (probs * torch.tensor(fixed_energies, dtype=torch.float32)).sum()
            grid_energies[i, j] = expected_energy.item()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Contour plot with trajectory
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(xx, yy, grid_energies.T, levels=20, cmap='viridis', alpha=0.8)
    ax1.contour(xx, yy, grid_energies.T, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    
    # Plot trajectory
    ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'r.-', linewidth=2, markersize=8, label='Refinement path')
    ax1.scatter(traj_2d[0, 0], traj_2d[0, 1], c='green', s=100, marker='o', label='System 1 (start)', zorder=5)
    ax1.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', s=100, marker='*', label='System 2 (end)', zorder=5)
    
    ax1.set_xlabel('PC1', fontsize=10)
    ax1.set_ylabel('PC2', fontsize=10)
    ax1.set_title('Energy Landscape (2D projection)', fontsize=12)
    ax1.legend(fontsize=9)
    plt.colorbar(contour, ax=ax1, label='Expected Energy')
    
    # 2. 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(xx, yy, grid_energies.T, cmap='viridis', alpha=0.7, edgecolor='none')
    
    # Plot 3D trajectory
    z_traj = np.array(traj_energies)
    ax2.plot(traj_2d[:, 0], traj_2d[:, 1], z_traj, 'r.-', linewidth=2, markersize=6)
    ax2.scatter(traj_2d[0, 0], traj_2d[0, 1], z_traj[0], c='green', s=100, marker='o')
    ax2.scatter(traj_2d[-1, 0], traj_2d[-1, 1], z_traj[-1], c='red', s=100, marker='*')
    
    ax2.set_xlabel('PC1', fontsize=10)
    ax2.set_ylabel('PC2', fontsize=10)
    ax2.set_zlabel('Energy', fontsize=10)
    ax2.set_title('3D Energy Surface', fontsize=12)
    
    # 3. Energy descent curve
    ax3 = fig.add_subplot(133)
    steps_array = np.arange(len(traj_energies))
    ax3.plot(steps_array, traj_energies, 'b.-', linewidth=2, markersize=8)
    ax3.fill_between(steps_array, traj_energies, traj_energies[-1], alpha=0.3)
    ax3.set_xlabel('Refinement Step', fontsize=10)
    ax3.set_ylabel('Expected Energy', fontsize=10)
    ax3.set_title('Energy Descent', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add energy gap annotation
    gap = traj_energies[0] - traj_energies[-1]
    ax3.annotate(f'Energy gap: {gap:.4f}',
                xy=(len(traj_energies)-1, traj_energies[-1]),
                xytext=(len(traj_energies)/2, (traj_energies[0] + traj_energies[-1])/2),
                arrowprops=dict(arrowstyle='->', alpha=0.5),
                fontsize=10)
    
    plt.suptitle(f'Unified Energy Landscape - Gradient Descent Refinement\nPrompt: "{cfg.prompt}"', fontsize=14)
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(cfg.out_dir, cfg.out_landscape)
    os.makedirs(cfg.out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved unified landscape to: {out_path}")


def plot_trajectory_details(model: EBM, ds: CharDataset, cfg: VizUnifiedConfig):
    """Show detailed trajectory with gradient information."""
    device = next(model.parameters()).device
    
    # Encode prompt
    stoi = ds.stoi
    ids = [stoi[c] for c in cfg.prompt if c in stoi]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    idx = idx[:, -model.config.block_size:]
    
    # Get trajectory with gradients
    traj_logits, traj_energies, fixed_energies = get_refinement_trajectory(model, idx, cfg.steps)
    
    # Compute gradient magnitudes
    grad_norms = []
    for i in range(1, len(traj_energies)):
        grad_norm = abs(traj_energies[i] - traj_energies[i-1]) / model.alpha.item()
        grad_norms.append(grad_norm)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Energy vs Step
    ax = axes[0, 0]
    ax.plot(traj_energies, 'b.-', linewidth=2, markersize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Expected Energy')
    ax.set_title('Energy Descent')
    ax.grid(True, alpha=0.3)
    
    # 2. Gradient norm vs Step
    ax = axes[0, 1]
    if grad_norms:
        ax.plot(grad_norms, 'r.-', linewidth=2, markersize=8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Magnitude During Refinement')
        ax.grid(True, alpha=0.3)
    
    # 3. Top token probabilities evolution
    ax = axes[1, 0]
    vocab_size = traj_logits[0].shape[-1]
    k = min(10, vocab_size)
    
    # Get top tokens from final distribution
    final_logits = traj_logits[-1].squeeze()
    top_indices = np.argsort(final_logits)[-k:][::-1]
    
    # Track their probabilities
    for idx_token in top_indices[:5]:  # Show top 5
        probs_over_time = []
        for logits in traj_logits:
            probs = F.softmax(torch.tensor(logits.squeeze()), dim=-1)
            probs_over_time.append(probs[idx_token].item())
        
        token_str = ds.itos.get(idx_token, f'[{idx_token}]')
        ax.plot(probs_over_time, '.-', label=repr(token_str), linewidth=1.5)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Probability')
    ax.set_title('Top Token Probability Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Energy landscape cross-section
    ax = axes[1, 1]
    
    # Create 1D cross-section along trajectory direction
    if len(traj_logits) > 1:
        start = traj_logits[0].squeeze()
        end = traj_logits[-1].squeeze()
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        # Sample points along this direction
        alphas = np.linspace(-0.5, 1.5, 100)
        energies_1d = []
        
        for alpha in alphas:
            point = start + alpha * np.linalg.norm(end - start) * direction
            probs = F.softmax(torch.tensor(point, dtype=torch.float32), dim=-1)
            energy = (probs * torch.tensor(fixed_energies.squeeze(), dtype=torch.float32)).sum()
            energies_1d.append(energy.item())
        
        ax.plot(alphas, energies_1d, 'b-', linewidth=2)
        
        # Mark trajectory points
        for i, logits in enumerate(traj_logits):
            # Project onto the line
            diff = logits.squeeze() - start
            alpha = np.dot(diff, direction) / np.linalg.norm(end - start)
            ax.scatter(alpha, traj_energies[i], c='red', s=50, zorder=5)
        
        ax.set_xlabel('Position along trajectory')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Cross-section Along Trajectory')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Refinement Trajectory Details - "{cfg.prompt}"', fontsize=14)
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(cfg.out_dir, cfg.out_trajectory)
    os.makedirs(cfg.out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory details to: {out_path}")


def plot_distribution_evolution(model: EBM, ds: CharDataset, cfg: VizUnifiedConfig):
    """Visualize how the probability distribution evolves during refinement."""
    device = next(model.parameters()).device
    
    # Encode prompt
    stoi = ds.stoi
    ids = [stoi[c] for c in cfg.prompt if c in stoi]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    idx = idx[:, -model.config.block_size:]
    
    # Get trajectory
    traj_logits, traj_energies, fixed_energies = get_refinement_trajectory(model, idx, cfg.steps)
    
    # Select key steps to visualize
    steps_to_show = [0, len(traj_logits)//3, 2*len(traj_logits)//3, len(traj_logits)-1]
    steps_to_show = list(set(steps_to_show))[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx_plot, step in enumerate(steps_to_show):
        ax = axes[idx_plot]
        
        logits = traj_logits[step].squeeze()
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        energies_np = fixed_energies.squeeze()
        
        # Get top tokens
        k = min(20, len(probs))
        top_indices = np.argsort(probs)[-k:][::-1]
        
        # Create bar plot
        x_pos = np.arange(k)
        colors = cm.coolwarm((energies_np[top_indices] - energies_np.min()) / 
                            (energies_np.max() - energies_np.min() + 1e-8))
        
        bars = ax.bar(x_pos, probs[top_indices], color=colors)
        
        # Add token labels
        labels = [repr(ds.itos.get(i, f'[{i}]'))[:5] for i in top_indices]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        ax.set_ylabel('Probability')
        ax.set_title(f'Step {step} - Energy: {traj_energies[min(step, len(traj_energies)-1)]:.4f}')
        ax.set_ylim(0, max(0.5, probs.max() * 1.1))
        
        # Add colorbar for energy
        sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, 
                                   norm=plt.Normalize(vmin=energies_np.min(), 
                                                     vmax=energies_np.max()))
        sm.set_array([])
        if idx_plot == 1:
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.15, aspect=20)
            cbar.set_label('Token Energy', fontsize=8)
    
    plt.suptitle(f'Probability Distribution Evolution During Refinement\n"{cfg.prompt}"', fontsize=14)
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(cfg.out_dir, cfg.out_distribution)
    os.makedirs(cfg.out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution evolution to: {out_path}")


def main(cfg: VizUnifiedConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = cfg.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        print(f"Auto-detected checkpoint: {checkpoint}")
    
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    
    # Load config
    if "config" in ckpt:
        config_dict = ckpt["config"]["model"]
        # Remove deprecated parameters
        for param in ['use_replay_buffer']:
            config_dict.pop(param, None)
        model_cfg = ModelConfig(**config_dict)
    else:
        model_cfg = ModelConfig()
    
    # Load model
    model = EBM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Load dataset
    ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")
    
    # Run visualization
    if cfg.mode == "landscape":
        plot_unified_landscape(model, ds, cfg)
    elif cfg.mode == "trajectory":
        plot_trajectory_details(model, ds, cfg)
    elif cfg.mode == "distribution":
        plot_distribution_evolution(model, ds, cfg)
    else:
        # Run all visualizations
        plot_unified_landscape(model, ds, cfg)
        plot_trajectory_details(model, ds, cfg)
        plot_distribution_evolution(model, ds, cfg)


if __name__ == "__main__":
    config = chz.entrypoint(VizUnifiedConfig)
    main(config)