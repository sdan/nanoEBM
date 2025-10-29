"""Visualization for EBM - Energy landscape and gradient descent

Usage examples:
  # Expected energy trace (gap across inner steps)
  python viz.py --checkpoint=out_ebt/final.pt --mode=gap --steps=8

  # Bowl panels (export-only): a dot descending as energy drops
  python viz.py --checkpoint=out_ebt/final.pt --mode=bowl_panels --steps=8

  # Bowl GIF animation (export-only)
  python viz.py --checkpoint=out_ebt/final.pt --mode=bowl_anim --steps=8
  
  # Token logit trajectories
  python viz.py --checkpoint=out_ebt/final.pt --mode=trajectories --steps=8
"""

from __future__ import annotations

import os
# Set matplotlib backend before importing pyplot to avoid backend issues
os.environ['MPLBACKEND'] = 'Agg'

import chz
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import CharDataset, get_loader


def find_latest_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest checkpoint by looking for the newest run_* directory."""
    import os
    import glob

    # Look for run_* directories
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")

    # Sort by modification time (newest first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]

    # Check for final.pt in the latest directory
    checkpoint_path = os.path.join(latest_dir, "final.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Fallback: look for any .pt file in the latest directory
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if pt_files:
        pt_files.sort(key=os.path.getmtime, reverse=True)
        return pt_files[0]

    raise FileNotFoundError(f"No .pt files found in {latest_dir}")


@chz.chz
class VizConfig:
    checkpoint: str | None = None  # None = auto-detect latest
    data_path: str = "shakespeare.txt"
    prompt: str = "ROMEO:"
    # visualization mode
    mode: str = "gap"  # gap|bowl_panels|bowl_anim|trajectories
    # refinement parameters
    steps: int = 8
    batch_size: int = 32
    # animation settings
    writer: str = "auto"  # auto|gif|mp4
    curve_from_prompt: bool = True  # use prompt for curve plots
    # output paths
    out_gap: str = "out_ebt/energy_gap.png"
    out_curve_panels: str = "out_ebt/energy_bowl_panels.png"
    out_curve_anim: str = "out_ebt/energy_bowl.gif"
    out_traj: str = "out_ebt/logit_trajectories.png"


def _ensure_dir(path: str):
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def _encode_prompt(ds: CharDataset, prompt: str, device: str):
    stoi, _ = ds.stoi, ds.itos
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        raise ValueError("Prompt produced empty token sequence (unknown chars?)")
    return torch.tensor([ids], dtype=torch.long, device=device)


def _think_trace_last(model: EBM, idx: torch.Tensor, steps: int):
    """
    Get energy and logit trajectories using EBM's system2_refine.
    Returns (energy_trace, logit_trajectories, grad_norms)
    """
    device = idx.device
    B = idx.shape[0]

    # Truncate to block size
    idx_cond = idx[:, -model.config.block_size:]

    # Run System 2 refinement with trajectory tracking
    refined_logits = model.system2_refine(
        idx_cond,
        steps=steps,
        return_trajectory=True
    )

    # Check if we got a trajectory or just final logits
    if isinstance(refined_logits, tuple):
        refined_logits, trajectory = refined_logits
    else:
        trajectory = refined_logits

    # Get energy values for computing expected energy
    with torch.no_grad():
        h = model.get_hidden_states(idx_cond)  # (B, T, n_embd)
        energies = model.energy_head(h)  # (B, T, V)
        last_energies = energies[:, -1, :]  # (B, V)

    # Extract energy trace from trajectory
    energy_trace = []
    logit_trajs = []
    grad_norms = []

    # Handle different trajectory formats
    if isinstance(trajectory, list):
        # List of logit tensors
        for step_logits in trajectory:
            # Get last position logits
            if step_logits.dim() == 3:  # (B, T, V)
                step_logits_last = step_logits[:, -1, :]
            else:  # (B, V)
                step_logits_last = step_logits

            # Compute expected energy
            probs = F.softmax(step_logits_last, dim=-1)
            expected_energy = (probs * last_energies).sum(dim=-1).mean()
            energy_trace.append(float(expected_energy.cpu()))

            # Store logits for B=1 case
            if B == 1:
                logit_trajs.append(step_logits_last[0].detach().cpu())
    elif trajectory.dim() == 3:  # Single tensor (B, T, V)
        # Use last position only
        last_logits = trajectory[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        expected_energy = (probs * last_energies).sum(dim=-1).mean()
        energy_trace = [float(expected_energy.cpu())]
        if B == 1:
            logit_trajs = [last_logits[0].detach().cpu()]

    # Compute gradient norms if we have multiple steps
    for i in range(1, len(energy_trace)):
        if hasattr(model, 'alpha'):
            grad_norm = abs(energy_trace[i] - energy_trace[i-1]) / float(model.alpha.item())
        else:
            grad_norm = abs(energy_trace[i] - energy_trace[i-1])
        grad_norms.append(grad_norm)

    return energy_trace, logit_trajs, grad_norms


def plot_expected_energy_gap(model: EBM, ds: CharDataset, cfg: VizConfig):
    """Plot how energy decreases during System 2 refinement"""
    device = next(model.parameters()).device
    # Use a small batch from train split
    loader, _ = get_loader(cfg.data_path, model.config.block_size, batch_size=cfg.batch_size, split="train")
    x, _ = next(iter(loader))
    x = x.to(device)
    
    # Get energy trace from refinement
    en_trace, _, _ = _think_trace_last(model, x, steps=cfg.steps)
    
    xs = np.arange(len(en_trace))
    plt.figure(figsize=(7, 5))
    plt.plot(xs, en_trace, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    plt.xlabel("Refinement Step", fontsize=12)
    plt.ylabel("Expected Energy E[E(x)]", fontsize=12)
    plt.title("Energy Descent via Gradient-Based Refinement (System 2)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation showing energy gap
    if len(en_trace) > 1:
        gap = en_trace[0] - en_trace[-1]
        plt.annotate(f"Energy gap: {gap:.3f}", 
                    xy=(len(en_trace)-1, en_trace[-1]),
                    xytext=(len(en_trace)/2, (en_trace[0] + en_trace[-1])/2),
                    arrowprops=dict(arrowstyle="->", alpha=0.5),
                    fontsize=10)
    
    plt.tight_layout()
    _ensure_dir(cfg.out_gap)
    plt.savefig(cfg.out_gap, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_gap}")


def plot_token_logit_trajectories(model: EBM, ds: CharDataset, cfg: VizConfig):
    """Visualize how logits evolve during refinement for top tokens"""
    device = next(model.parameters()).device
    idx = _encode_prompt(ds, cfg.prompt, device)
    
    # Get trajectories
    en_trace, logit_trajs, _ = _think_trace_last(model, idx, steps=cfg.steps)
    
    if not logit_trajs:
        print("No logit trajectories available")
        return
    
    # Stack trajectories
    traj = torch.stack(logit_trajs)  # (steps+1, vocab_size)
    
    # Get top-k tokens from initial logits
    k = min(10, traj.size(-1))  # Show top 10 tokens
    v0 = traj[0]
    topv, topi = torch.topk(v0, k=k)
    traj = traj[:, topi]
    toks = [ds.itos[i] for i in topi.cpu().tolist()]
    
    steps_axis = np.arange(traj.size(0))
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    for j in range(k):
        plt.plot(steps_axis, traj[:, j].numpy(), 
                label=repr(toks[j]), 
                linewidth=2, 
                marker='o',
                color=colors[j])
    
    plt.xlabel("Refinement Step", fontsize=12)
    plt.ylabel("Logit Value", fontsize=12)
    plt.title(f"Logit Evolution During Refinement\nPrompt: {cfg.prompt!r}", fontsize=14)
    plt.legend(fontsize=9, ncol=2, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    _ensure_dir(cfg.out_traj)
    plt.savefig(cfg.out_traj, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_traj}")


def _energy_trace_for_curve(model: EBM, ds: CharDataset, cfg: VizConfig):
    """Get energy trace for bowl visualization"""
    device = next(model.parameters()).device
    if cfg.curve_from_prompt:
        idx = _encode_prompt(ds, cfg.prompt, device)
    else:
        loader, _ = get_loader(cfg.data_path, model.config.block_size, batch_size=cfg.batch_size, split="train")
        x, _ = next(iter(loader))
        idx = x.to(device)
    
    en_trace, _, _ = _think_trace_last(model, idx, steps=cfg.steps)
    return en_trace


def plot_energy_curve_panels(model: EBM, ds: CharDataset, cfg: VizConfig):
    """Visualize energy descent as a ball rolling down a bowl"""
    en = np.asarray(_energy_trace_for_curve(model, ds, cfg), dtype=float)
    if en.size == 0 or not np.isfinite(en).all():
        raise RuntimeError("Energy trace is empty or non-finite.")
    
    # Normalize energies to [0,1] for display
    e_min, e_max = float(en.min()), float(en.max())
    rng = max(1e-8, e_max - e_min)
    en_norm = (en - e_min) / rng

    # Map energy to position on bowl curve
    # As energy decreases, ball moves toward center (x=0)
    x_pos = -np.sqrt(np.clip(en_norm, 0.0, 1.0))

    # Panel grid
    Kp1 = en_norm.shape[0]
    cols = min(4, Kp1)
    rows = int(np.ceil(Kp1 / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Bowl curve
    xs = np.linspace(-1.2, 1.2, 400)
    bowl = np.clip(xs**2, 0.0, 1.0)

    for i in range(Kp1):
        ax = axes[i]
        ax.plot(xs, bowl, color="#2E86AB", linewidth=3)
        ax.scatter([x_pos[i]], [en_norm[i]], s=120, color="#A23B72", zorder=5, edgecolor='white', linewidth=2)
        ax.set_title(f"Step {i}\nE={en[i]:.3f}", fontsize=10)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
    
    # Hide unused axes
    for j in range(Kp1, axes.size):
        axes[j].axis("off")

    fig.suptitle("Energy-Based Refinement: Gradient Descent Visualization", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _ensure_dir(cfg.out_curve_panels)
    plt.savefig(cfg.out_curve_panels, dpi=150)
    plt.close(fig)
    print(f"Saved: {cfg.out_curve_panels}")


def animate_energy_curve(model: EBM, ds: CharDataset, cfg: VizConfig):
    """Create animated GIF of energy descent"""
    from matplotlib.animation import FuncAnimation
    
    # Get energy trace
    en = np.asarray(_energy_trace_for_curve(model, ds, cfg), dtype=float)
    e_min, e_max = float(en.min()), float(en.max())
    rng = max(1e-8, e_max - e_min)
    en_norm = (en - e_min) / rng
    x_pos = -np.sqrt(np.clip(en_norm, 0.0, 1.0))
    
    # Bowl curve
    xs = np.linspace(-1.2, 1.2, 400)
    bowl = np.clip(xs**2, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, bowl, color="#2E86AB", linewidth=3)
    dot = ax.scatter([x_pos[0]], [en_norm[0]], s=150, color="#A23B72", zorder=5, edgecolor='white', linewidth=2)
    txt = ax.text(0.02, 0.92, f"Step 0\nE={en[0]:.4f}", transform=ax.transAxes, fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("Energy Descent (lower is better)")

    def update(f):
        dot.set_offsets([[x_pos[f], en_norm[f]]])
        txt.set_text(f"Step {f}\nE={en[f]:.4f}")
        return dot, txt

    anim = FuncAnimation(fig, update, frames=len(en), interval=600, blit=True)

    # Check for available animation writers
    writer_pref = (cfg.writer or "auto").lower()
    pillow_ok = ffmpeg_ok = False
    try:
        from matplotlib.animation import PillowWriter
        pillow_ok = True
    except Exception:
        pass
    try:
        from matplotlib.animation import FFMpegWriter
        ffmpeg_ok = True
    except Exception:
        pass

    # Decide writer and save
    saved = False
    if writer_pref in ("gif", "auto") and pillow_ok:
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=2)
            _ensure_dir(cfg.out_curve_anim)
            anim.save(cfg.out_curve_anim, writer=writer)
            print(f"Saved: {cfg.out_curve_anim}")
            saved = True
        except Exception as e:
            print(f"Could not save GIF ({e}).")
    if not saved and writer_pref in ("mp4", "auto") and ffmpeg_ok:
        try:
            from matplotlib.animation import FFMpegWriter
            _ensure_dir(cfg.out_curve_anim)
            mp4_path = cfg.out_curve_anim.rsplit('.', 1)[0] + '.mp4'
            writer = FFMpegWriter(fps=2)
            anim.save(mp4_path, writer=writer)
            print(f"Saved: {mp4_path}")
            saved = True
        except Exception as e:
            print(f"Could not save MP4 ({e}).")
    if not saved:
        print("Falling back to static panels.")
        plt.close(fig)
        plot_energy_curve_panels(model, ds, cfg)
        return

    plt.close(fig)


def main(cfg: VizConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Auto-detect latest checkpoint if not specified
    checkpoint = cfg.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        print(f"Auto-detected latest checkpoint: {checkpoint}")
    else:
        print(f"Loading checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    
    # Load config, filtering out removed parameters
    if "config" in ckpt:
        config_dict = ckpt["config"]["model"]
        # Remove parameters that no longer exist in ModelConfig
        removed_params = ['use_replay_buffer']
        for param in removed_params:
            config_dict.pop(param, None)
        model_cfg = ModelConfig(**config_dict)
    else:
        model_cfg = ModelConfig()
    
    model = EBM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")

    if cfg.mode == "gap":
        plot_expected_energy_gap(model, ds, cfg)
    elif cfg.mode in ("traj", "trajectories"):
        plot_token_logit_trajectories(model, ds, cfg)
    elif cfg.mode in ("bowl", "bowl_panels"):
        plot_energy_curve_panels(model, ds, cfg)
    elif cfg.mode in ("bowl_anim", "bowl_gif", "anim"):
        animate_energy_curve(model, ds, cfg)
    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode}. Supported modes: gap, trajectories (traj), bowl_panels (bowl), bowl_anim (bowl_gif, anim)"
        )


if __name__ == "__main__":
    config = chz.entrypoint(VizConfig)
    main(config)