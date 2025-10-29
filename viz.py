"""vibecoded visualization of nanoEBM

Usage examples:
  python viz.py --checkpoint=out_ebt/final.pt --prompt="ROMEO:" --topk=20
  python viz.py --checkpoint=out_ebt/final.pt --mode=correlation --batches=4
"""

from __future__ import annotations

import io
import chz
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

from nanoebm.config import ModelConfig
from nanoebm.model import EBTLanguageModel, ContextTransformer
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
    topk: int = 20
    # modes: legacy: "topk", "correlation"; new: "gap", "trajectories", "shift", "margin", "eval",
    # plus curve-style energy visuals: "curve" (panels) and "curve_anim" (GIF), or "all"
    mode: str = "topk"
    batches: int = 2     # for correlation mode
    # thinking loop params for analysis
    steps: int = 8
    tau: float = 0.5
    lr: float | None = None
    noise: float = 0.0
    topk_think: int = 32
    batch_size: int = 32
    # outputs
    out_topk: str = "out_ebt/energies_topk.png"
    out_corr: str = "out_ebt/energy_vs_ce.png"
    out_gap: str = "out_ebt/expected_energy_gap.png"
    out_traj: str = "out_ebt/token_logit_trajectories.png"
    out_shift: str = "out_ebt/expected_energy_hist_shift.png"
    out_margin: str = "out_ebt/energy_margin_report.txt"
    out_eval: str = "out_ebt/eval_report.txt"
    # curve-style outputs
    out_curve_panels: str = "out_ebt/energy_curve_panels.png"
    out_curve_anim: str = "out_ebt/energy_curve.gif"
    curve_from_prompt: bool = True  # when False, use a small train batch mean
    # surface-style landscape of J(v) over a 2D plane in logits space
    out_surface3d: str = "out_ebt/energy_surface3d.png"
    out_surface_contour: str = "out_ebt/energy_surface_contour.png"
    out_surface_anim: str = "out_ebt/energy_surface_traj.gif"
    surface_grid: int = 60          # grid resolution per axis
    surface_span: float = 6.0       # +/- span around v0 for the two dims (logit units)
    surface_dim_i: int | None = None  # choose 2 dims in candidate set; None => auto by |grad|
    surface_dim_j: int | None = None


@torch.no_grad()
def decode(idx, itos):
    return "".join(itos[i] for i in idx.tolist())


def _ensure_dir(path: str):
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def visualize_topk(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    stoi, itos = ds.stoi, ds.itos
    # Encode prompt and get last hidden state
    idx = torch.tensor([[stoi[c] for c in cfg.prompt if c in stoi]], dtype=torch.long, device=device)
    if idx.numel() == 0:
        raise ValueError("Prompt produced empty token sequence (unknown chars?)")
    ctx = idx[:, -model.gpt_cfg.block_size :]
    backbone = ContextTransformer(model.gpt_cfg).to(device)
    backbone.load_state_dict(model.backbone.state_dict())
    h = backbone(ctx)[:, -1]  # (B=1, d)
    # Energies for all tokens at next position
    E = model.energy.energies_all_tokens(h.unsqueeze(1), model.wte_weight)[:, 0, :]  # (1,V)
    E = E[0]
    # Take top-k lowest energies
    k = min(cfg.topk, E.numel())
    vals, inds = torch.topk(-E, k=k)  # highest -E -> lowest E
    vals = (-vals).cpu().tolist()
    toks = [itos[i] for i in inds.cpu().tolist()]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(k), vals)
    plt.xticks(range(k), toks, rotation=90)
    plt.ylabel("Energy (lower is better)")
    plt.title(f"Top-{k} lowest energies for next token | prompt={cfg.prompt!r}")
    plt.tight_layout()
    plt.savefig(cfg.out_topk, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_topk}")


def visualize_correlation(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    # Small loader from test split to avoid training leakage
    loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=8, split="test")
    xs, ys, energies_true, ce_losses = [], [], [], []
    batches_done = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        # Forward to get logits and also internal energies
        model.eval()
        loss, logits, _ = model(x, targets=y)
        # Energies of true next tokens per position
        with torch.no_grad():
            h = model.backbone(x)
            E = model.energy.energies_all_tokens(h, model.wte_weight)  # (B,T,V)
            # Align shapes: logits may have different seq length than y
            T_logits = logits.size(1)
            T_y = y.size(1)
            # Trim to match minimum length
            T_min = min(T_logits, T_y)
            logits_aligned = logits[:, :T_min, :]
            y_aligned = y[:, :T_min]
            E_aligned = E[:, :T_min, :]
            # gather energies at true token indices
            Et = E_aligned.gather(-1, y_aligned.unsqueeze(-1)).squeeze(-1)  # (B,T_min)
            energies_true.append(Et.flatten().cpu())
            # per-position CE from logits
            ce = torch.nn.functional.cross_entropy(
                logits_aligned.reshape(-1, logits_aligned.size(-1)), y_aligned.reshape(-1), reduction="none"
            ).reshape_as(Et)
            ce_losses.append(ce.flatten().cpu())

        batches_done += 1
        if batches_done >= cfg.batches:
            break

    import torch as _t
    E_all = _t.cat(energies_true)
    CE_all = _t.cat(ce_losses)
    # Filter any nans/infs
    mask = _t.isfinite(E_all) & _t.isfinite(CE_all)
    E_all = E_all[mask]
    CE_all = CE_all[mask]

    # Scatter and trendline
    plt.figure(figsize=(6, 5))
    plt.scatter(E_all.numpy(), CE_all.numpy(), s=4, alpha=0.3)
    plt.xlabel("Energy of true token")
    plt.ylabel("Cross-entropy loss")
    plt.title("Energy vs. CE across positions")
    # Simple linear fit
    if E_all.numel() > 100:
        import numpy as _np
        z = _np.polyfit(E_all.numpy(), CE_all.numpy(), 1)
        p = _np.poly1d(z)
        xs = _np.linspace(float(E_all.min()), float(E_all.max()), 100)
        plt.plot(xs, p(xs), "r--", alpha=0.8)
    plt.tight_layout()
    plt.savefig(cfg.out_corr, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_corr}")


@torch.no_grad()
def _encode_prompt(ds: CharDataset, prompt: str, device: str):
    stoi, _ = ds.stoi, ds.itos
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        raise ValueError("Prompt produced empty token sequence (unknown chars?)")
    return torch.tensor([ids], dtype=torch.long, device=device)


def _think_trace_last(model: EBTLanguageModel, idx: torch.Tensor, steps: int, tau: float, lr: float | None, noise: float, topk: int | None):
    device = idx.device
    # context and energies at last position
    idx_cond = idx[:, -model.gpt_cfg.block_size :]
    h_last = model.backbone(idx_cond)[:, -1]
    E = model.energy.energies_all_tokens(h_last.unsqueeze(1), model.wte_weight)[:, 0, :]  # (B,V)
    B, V = E.shape
    # init logits from -E or topk subset
    if topk is not None and topk > 0 and topk < V:
        topE, topI = torch.topk(-E, k=topk, dim=-1)  # (B,K)
        v = topE.clone()
        cand_idx = topI
    else:
        v = (-E).clone()
        cand_idx = None

    v = v.detach().requires_grad_(True)
    tau = float(tau)
    lr_val = float(lr) if lr is not None else float(model.alpha.detach().clamp(min=1e-6).item())
    clamp_change = float(getattr(model.ebt_cfg, "clamp_update_max_change", 0.0) or 0.0)
    abs_clamp = float(getattr(model.ebt_cfg, "absolute_clamp", 0.0) or 0.0)

    expected_en = []
    logit_trajs = []  # only for B=1 to keep plots readable

    def expected_energy_from(v_):
        p = torch.softmax(v_ / tau, dim=-1)
        if cand_idx is not None:
            R = model.energy.token_features(model.wte_weight)[cand_idx]  # (B,K,dE)
        else:
            R = model.energy.token_features(model.wte_weight).unsqueeze(0).expand(B, -1, -1)
        c = model.energy.context_proj(h_last)  # (B,dE)
        e = torch.einsum("bk,bkd->bd", p, R)
        b = 0.0
        if model.energy.use_token_bias:
            if cand_idx is not None:
                b_sel = model.energy.token_bias[cand_idx]
                b = torch.einsum("bk,bk->b", p, b_sel)
            else:
                b = torch.einsum("bk,k->b", p, model.energy.token_bias)
        return (b - (c * e).sum(-1))  # (B,)

    # step 0
    with torch.no_grad():
        expected_en.append(float(expected_energy_from(v).mean().detach().cpu()))
        if B == 1:
            logit_trajs.append(v.detach().cpu().clone())

    for _ in range(steps):
        v = v.detach().requires_grad_(True)
        p = torch.softmax(v / tau, dim=-1)
        # relaxed E and entropy
        E_relaxed = expected_energy_from(v)
        H = -(p.clamp_min(1e-9) * (p.clamp_min(1e-9)).log()).sum(-1)
        J = (E_relaxed - tau * H).sum()
        (g,) = torch.autograd.grad(J, v)
        delta = -lr_val * g
        if clamp_change and clamp_change > 0.0:
            delta = delta.clamp(min=-clamp_change, max=clamp_change)
        v = (v + delta).detach()
        if abs_clamp and abs_clamp > 0.0:
            v = v.clamp(min=-abs_clamp, max=abs_clamp)
        if noise and noise > 0:
            v = v + noise * torch.randn_like(v)
        with torch.no_grad():
            expected_en.append(float(expected_energy_from(v).mean().detach().cpu()))
            if B == 1:
                logit_trajs.append(v.detach().cpu().clone())

    return expected_en, logit_trajs, cand_idx


def _surface_objective_and_grad(model: EBTLanguageModel,
                                h_last: torch.Tensor,
                                v: torch.Tensor,
                                tau: float,
                                cand_idx: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scalar J(v) and grad dJ/dv for B=1 at the last position.
    v: (K,) or (V,) logits tensor requiring grad on the active candidate set.
    Returns (J_scalar, grad_same_shape).
    """
    assert v.dim() == 1, "v should be a 1D logits vector for B=1"
    # Ensure v lives on the same device as the model tensors (e.g., MPS)
    v_req = v.detach().to(h_last.device).clone().requires_grad_(True)
    tau = float(tau)
    # Build token features selection
    if cand_idx is not None:
        # cand_idx: (1, K)
        cand = cand_idx[0]
        R = model.energy.token_features(model.wte_weight)[cand]  # (K, dE)
    else:
        R = model.energy.token_features(model.wte_weight)        # (V, dE)
    c = model.energy.context_proj(h_last)  # (1, dE)
    p = torch.softmax(v_req / tau, dim=-1)  # (K) or (V)
    e = torch.einsum("k,kd->d", p, R)     # (dE)
    if model.energy.use_token_bias:
        if cand_idx is not None:
            b = torch.einsum("k,k->", p, model.energy.token_bias[cand])
        else:
            b = torch.einsum("k,k->", p, model.energy.token_bias)
    else:
        b = torch.tensor(0.0, device=v.device, dtype=v.dtype)
    E_relaxed = b - (c[0] * e).sum()  # scalar
    H = -(p.clamp_min(1e-9) * (p.clamp_min(1e-9)).log()).sum()
    J = E_relaxed - tau * H
    (g,) = torch.autograd.grad(J, v_req, retain_graph=False, create_graph=False)
    return J.detach(), g.detach()


@torch.no_grad()
def _surface_objective(model: EBTLanguageModel,
                       h_last: torch.Tensor,
                       v: torch.Tensor,
                       tau: float,
                       cand_idx: torch.Tensor | None) -> torch.Tensor:
    """Forward-only scalar J(v) for B=1. Does not build a graph."""
    tau = float(tau)
    # Move v to same device as h_last/model
    v = v.to(h_last.device)
    if cand_idx is not None:
        cand = cand_idx[0]
        R = model.energy.token_features(model.wte_weight)[cand]  # (K, dE)
    else:
        R = model.energy.token_features(model.wte_weight)        # (V, dE)
    c = model.energy.context_proj(h_last)  # (1, dE)
    p = torch.softmax(v / tau, dim=-1)
    e = torch.einsum("k,kd->d", p, R)
    if model.energy.use_token_bias:
        if cand_idx is not None:
            b = torch.einsum("k,k->", p, model.energy.token_bias[cand])
        else:
            b = torch.einsum("k,k->", p, model.energy.token_bias)
    else:
        b = torch.tensor(0.0, device=v.device, dtype=v.dtype)
    E_relaxed = b - (c[0] * e).sum()
    H = -(p.clamp_min(1e-9) * (p.clamp_min(1e-9)).log()).sum()
    J = E_relaxed - tau * H
    return J


def plot_expected_energy_gap(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    # use a small batch from train split
    loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=cfg.batch_size, split="train")
    x, _ = next(iter(loader))
    x = x.to(device)
    en_trace, _, _ = _think_trace_last(model, x, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think)
    import numpy as np
    xs = np.arange(len(en_trace))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, en_trace, marker="o")
    plt.xlabel("inner step k")
    plt.ylabel("E_p[E] (mean over batch)")
    plt.title("Expected energy gap across steps")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.out_gap, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_gap}")


def plot_token_logit_trajectories(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    idx = _encode_prompt(ds, cfg.prompt, device)
    en_trace, logit_trajs, cand_idx = _think_trace_last(model, idx, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think)
    if not logit_trajs:
        print("Token logit trajectories require B=1; using prompt mode for a single example.")
        return
    traj = torch.stack(logit_trajs)  # (K+1, 1, Kcand)
    # squeeze batch dimension (we only collect for B=1)
    if traj.dim() == 3 and traj.size(1) == 1:
        traj = traj.squeeze(1)  # (K+1, Kcand)
    if cand_idx is not None:
        toks = [ds.itos[i] for i in cand_idx[0].cpu().tolist()]
    else:
        # full vocab is too big; restrict to top-k of step 0
        k = min(cfg.topk, traj.size(-1))
        v0 = traj[0]
        topv, topi = torch.topk(v0, k=k)
        traj = traj[:, topi]
        toks = [ds.itos[i] for i in topi.cpu().tolist()]
    import numpy as np
    steps_axis = np.arange(traj.size(0))
    plt.figure(figsize=(10, 5))
    for j in range(min(len(toks), 10)):
        plt.plot(steps_axis, traj[:, j].numpy(), label=repr(toks[j]))
    plt.xlabel("inner step k")
    plt.ylabel("logits v_k(y)")
    plt.title("Token logit trajectories (top candidates)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(cfg.out_traj, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_traj}")


def plot_energy_surface(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig, make_anim: bool = False):
    """Plot a 3D surface and 2D contour of J(v) over a 2D plane in logits space
    for the last position of a single example (prompt). Optionally create a GIF
    animating the refinement trajectory on the contour.
    """
    import numpy as np
    from matplotlib.animation import FuncAnimation

    device = next(model.parameters()).device
    # Encode prompt -> last hidden and initial energies
    idx = _encode_prompt(ds, cfg.prompt, device)
    idx_cond = idx[:, -model.gpt_cfg.block_size :]
    h_last = model.backbone(idx_cond)[:, -1]  # (1, d)
    E = model.energy.energies_all_tokens(h_last.unsqueeze(1), model.wte_weight)[:, 0, :]  # (1, V)
    V = E.shape[-1]
    # Candidate set and initial logits v0
    if cfg.topk_think and cfg.topk_think > 0 and cfg.topk_think < V:
        topE, topI = torch.topk(-E, k=int(cfg.topk_think), dim=-1)  # (1, K)
        v0 = topE[0].detach().float()  # (K,)
        cand_idx = topI
    else:
        v0 = (-E[0]).detach().float()
        cand_idx = None
    K = v0.numel()

    # Pick two dimensions: user-provided or top-2 |grad J| at v0
    dim_i = cfg.surface_dim_i
    dim_j = cfg.surface_dim_j
    if dim_i is None or dim_j is None:
        J0, g0 = _surface_objective_and_grad(model, h_last, v0, cfg.tau, cand_idx)
        # choose top-2 unique indices by |grad|
        order = torch.argsort(g0.abs(), descending=True)
        dim_i = int(order[0].item())
        # find next distinct index
        dim_j = int(order[1].item()) if order.numel() > 1 else (0 if dim_i != 0 else 1)
    if dim_i == dim_j:
        dim_j = (dim_j + 1) % K

    # Axis labels from tokens if available
    if cand_idx is not None:
        toks = [ds.itos[i] for i in cand_idx[0].cpu().tolist()]
    else:
        toks = [ds.itos[i] for i in range(K)]
    label_i = f"logit[{dim_i}] ({repr(toks[dim_i])})"
    label_j = f"logit[{dim_j}] ({repr(toks[dim_j])})"

    # Build grid around v0 in the two selected dims
    span = float(cfg.surface_span)
    n = int(cfg.surface_grid)
    vi_center = float(v0[dim_i].item())
    vj_center = float(v0[dim_j].item())
    vi = np.linspace(vi_center - span, vi_center + span, n)
    vj = np.linspace(vj_center - span, vj_center + span, n)
    VI, VJ = np.meshgrid(vi, vj, indexing="xy")
    Z = np.zeros_like(VI, dtype=np.float64)

    # Evaluate J over the grid
    with torch.no_grad():
        for a in range(n):
            for b in range(n):
                v = v0.clone()
                v[dim_i] = float(VI[a, b])
                v[dim_j] = float(VJ[a, b])
                J_ab = _surface_objective(model, h_last, v, cfg.tau, cand_idx)
                Z[a, b] = float(J_ab.cpu().item())

    # Compute a coarse quiver field of projected gradients
    stride = max(1, n // 12)
    Qi, Qj, Qu, Qv = [], [], [], []
    for a in range(0, n, stride):
        for b in range(0, n, stride):
            v = v0.clone()
            v[dim_i] = float(VI[a, b])
            v[dim_j] = float(VJ[a, b])
            _, g = _surface_objective_and_grad(model, h_last, v, cfg.tau, cand_idx)
            # Negative gradient for descent direction
            Qi.append(VI[a, b])
            Qj.append(VJ[a, b])
            Qu.append(float(-g[dim_i].item()))
            Qv.append(float(-g[dim_j].item()))
    Qi = np.array(Qi); Qj = np.array(Qj); Qu = np.array(Qu); Qv = np.array(Qv)

    # Get refinement trajectory on these two dims
    en_trace, logit_trajs, _cand = _think_trace_last(
        model, idx, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think
    )
    traj = torch.stack(logit_trajs).squeeze(1) if logit_trajs else v0[None, :]
    path_i = traj[:, dim_i].cpu().numpy()
    path_j = traj[:, dim_j].cpu().numpy()
    # J along the path for 3D overlay
    J_path = []
    with torch.no_grad():
        for k in range(traj.size(0)):
            Jk = _surface_objective(model, h_last, traj[k], cfg.tau, cand_idx)
            J_path.append(float(Jk.cpu().item()))
    J_path = np.array(J_path)

    # 3D surface with path
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ensures 3D projection is registered)
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(VI, VJ, Z, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax3d.plot(path_i, path_j, J_path, color="k", linewidth=2, marker="o")
    ax3d.set_xlabel(label_i)
    ax3d.set_ylabel(label_j)
    ax3d.set_zlabel("J(v) = E_p[E] - tau H")
    ax3d.set_title("Energy Landscape and Refinement Path")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08)
    _ensure_dir(cfg.out_surface3d)
    plt.tight_layout()
    plt.savefig(cfg.out_surface3d, dpi=150)
    plt.close(fig)
    print(f"Saved: {cfg.out_surface3d}")

    # 2D contour + quiver + path
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    cs = ax2.contourf(VI, VJ, Z, levels=30, cmap="viridis")
    ax2.quiver(Qi, Qj, Qu, Qv, color="white", scale=50, width=0.002, alpha=0.9)
    ax2.plot(path_i, path_j, color="k", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel(label_i)
    ax2.set_ylabel(label_j)
    ax2.set_title("Projected Gradient Field and Path")
    fig2.colorbar(cs, ax=ax2, label="J(v)")
    _ensure_dir(cfg.out_surface_contour)
    plt.tight_layout()
    plt.savefig(cfg.out_surface_contour, dpi=150)
    plt.close(fig2)
    print(f"Saved: {cfg.out_surface_contour}")

    # Optional animation: dot moving along the path over the contour
    if make_anim and traj.size(0) > 1:
        try:
            from matplotlib.animation import PillowWriter
            figA, axA = plt.subplots(figsize=(7, 5))
            csA = axA.contourf(VI, VJ, Z, levels=30, cmap="viridis")
            axA.quiver(Qi, Qj, Qu, Qv, color="white", scale=50, width=0.002, alpha=0.9)
            axA.plot(path_i, path_j, color="k", linewidth=1.5, alpha=0.6)
            dot = axA.scatter([path_i[0]], [path_j[0]], s=120, color="#111111", zorder=5)
            txt = axA.text(0.02, 0.95, f"Step 0\nJ={J_path[0]:.4f}", transform=axA.transAxes, color="white")
            axA.set_xlabel(label_i)
            axA.set_ylabel(label_j)
            axA.set_title("Refinement Trajectory on Energy Contours")
            figA.colorbar(csA, ax=axA, label="J(v)")

            def update(f):
                dot.set_offsets([[path_i[f], path_j[f]]])
                txt.set_text(f"Step {f}\nJ={J_path[f]:.4f}")
                return dot, txt

            anim = FuncAnimation(figA, update, frames=len(J_path), interval=600, blit=True)
            _ensure_dir(cfg.out_surface_anim)
            anim.save(cfg.out_surface_anim, writer=PillowWriter(fps=2))
            plt.close(figA)
            print(f"Saved: {cfg.out_surface_anim}")
        except Exception as e:
            print(f"Could not save GIF ({e}). Skipping animation.")


def hist_expected_energy_shift(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=cfg.batch_size, split="train")
    x, _ = next(iter(loader))
    x = x.to(device)
    en_trace, _, _ = _think_trace_last(model, x, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think)
    import numpy as np
    # We only have batch-means in the helper; compute again per-example to histogram 0 vs K
    # Re-run for per-example
    idx_cond = x[:, -model.gpt_cfg.block_size :]
    h_last = model.backbone(idx_cond)[:, -1]
    E = model.energy.energies_all_tokens(h_last.unsqueeze(1), model.wte_weight)[:, 0, :]
    B, V = E.shape
    if cfg.topk_think and cfg.topk_think < V:
        topE, topI = torch.topk(-E, k=cfg.topk_think, dim=-1)
        v = topE.clone()
        cand_idx = topI
    else:
        v = (-E).clone()
        cand_idx = None
    v0 = v.clone()
    # K steps
    vK = v.detach().requires_grad_(True)
    tau = float(cfg.tau)
    lr_val = float(cfg.lr) if cfg.lr is not None else float(model.alpha.detach().clamp(min=1e-6).item())
    clamp_change = float(getattr(model.ebt_cfg, "clamp_update_max_change", 0.0) or 0.0)
    abs_clamp = float(getattr(model.ebt_cfg, "absolute_clamp", 0.0) or 0.0)
    def expected_energy_batch(v_):
        p = torch.softmax(v_ / tau, dim=-1)
        if cand_idx is not None:
            R = model.energy.token_features(model.wte_weight)[cand_idx]
        else:
            R = model.energy.token_features(model.wte_weight).unsqueeze(0).expand(B, -1, -1)
        c = model.energy.context_proj(h_last)
        e = torch.einsum("bk,bkd->bd", p, R)
        b = 0.0
        if model.energy.use_token_bias:
            if cand_idx is not None:
                b_sel = model.energy.token_bias[cand_idx]
                b = torch.einsum("bk,bk->b", p, b_sel)
            else:
                b = torch.einsum("bk,k->b", p, model.energy.token_bias)
        return (b - (c * e).sum(-1))
    with torch.no_grad():
        en0 = expected_energy_batch(v0).cpu()
    for _ in range(cfg.steps):
        vK = vK.detach().requires_grad_(True)
        p = torch.softmax(vK / tau, dim=-1)
        E_rel = expected_energy_batch(vK)
        H = -(p.clamp_min(1e-9) * (p.clamp_min(1e-9)).log()).sum(-1)
        J = (E_rel - tau * H).sum()
        (g,) = torch.autograd.grad(J, vK)
        delta = -lr_val * g
        if clamp_change and clamp_change > 0.0:
            delta = delta.clamp(min=-clamp_change, max=clamp_change)
        vK = (vK + delta).detach()
        if abs_clamp and abs_clamp > 0.0:
            vK = vK.clamp(min=-abs_clamp, max=abs_clamp)
        if cfg.noise and cfg.noise > 0:
            vK = vK + cfg.noise * torch.randn_like(vK)
    with torch.no_grad():
        enK = expected_energy_batch(vK).cpu()

    plt.figure(figsize=(7, 4))
    plt.hist(en0.numpy(), bins=30, alpha=0.6, label="step 0")
    plt.hist(enK.numpy(), bins=30, alpha=0.6, label=f"step {cfg.steps}")
    plt.xlabel("E_p[E]")
    plt.ylabel("count")
    plt.title("Expected energy shift (batch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_shift, dpi=150)
    plt.close()
    print(f"Saved: {cfg.out_shift}")


def _energy_trace_for_curve(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    """Return a 1D expected-energy trace across thinking steps for a single example (prompt)
    or a small batch mean, suitable for curve-style visualizations.
    """
    device = next(model.parameters()).device
    if cfg.curve_from_prompt:
        idx = _encode_prompt(ds, cfg.prompt, device)
    else:
        loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=cfg.batch_size, split="train")
        x, _ = next(iter(loader))
        idx = x.to(device)
    en_trace, _, _ = _think_trace_last(
        model, idx, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think
    )
    return en_trace


def plot_energy_curve_panels(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    """Draw a sequence of small panels with a bowl-shaped energy curve and a dot
    at the current expected energy level. This mirrors the conceptual 'ball rolling into
    a valley' figure while grounding the dot's vertical position in actual energies.
    """
    import numpy as np
    en = np.asarray(_energy_trace_for_curve(model, ds, cfg), dtype=float)
    if en.size == 0 or not np.isfinite(en).all():
        raise RuntimeError("Energy trace is empty or non-finite.")
    # Normalize energies to [0,1] for display; keep raw values for tooltips/labels
    e_min, e_max = float(en.min()), float(en.max())
    rng = max(1e-8, e_max - e_min)
    en_norm = (en - e_min) / rng

    # Map normalized energy y in [0,1] to an x on the left slope of y=x^2
    # so that as energy decreases, the point slides toward the bowl minimum (x→0).
    x_pos = -np.sqrt(np.clip(en_norm, 0.0, 1.0))

    # Panel grid
    Kp1 = en_norm.shape[0]
    cols = min(6, Kp1)
    rows = int(np.ceil(Kp1 / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Precompute bowl curve
    xs = np.linspace(-1.2, 1.2, 400)
    bowl = np.clip(xs**2, 0.0, 1.0)  # y in [0,1]

    for i in range(Kp1):
        ax = axes[i]
        ax.plot(xs, bowl, color="#6e8efb", linewidth=3)
        ax.scatter([x_pos[i]], [en_norm[i]], s=80, color="#111111", zorder=5)
        ax.set_title(f"Step {i}")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
    # Hide any unused axes
    for j in range(Kp1, axes.size):
        axes[j].axis("off")

    fig.suptitle("Energy Descent on a Bowl-Shaped Curve (dot uses actual E_p[E] per step)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _ensure_dir(cfg.out_curve_panels)
    plt.savefig(cfg.out_curve_panels, dpi=150)
    plt.close(fig)
    print(f"Saved: {cfg.out_curve_panels}")


def animate_energy_curve(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    """Create a simple GIF where a dot descends a bowl as the expected energy drops.
    Falls back gracefully if PillowWriter is unavailable.
    """
    import numpy as np
    from matplotlib.animation import FuncAnimation
    try:
        from matplotlib.animation import PillowWriter
        pillow_ok = True
    except Exception:
        pillow_ok = False

    en = np.asarray(_energy_trace_for_curve(model, ds, cfg), dtype=float)
    e_min, e_max = float(en.min()), float(en.max())
    rng = max(1e-8, e_max - e_min)
    en_norm = (en - e_min) / rng
    x_pos = -np.sqrt(np.clip(en_norm, 0.0, 1.0))
    xs = np.linspace(-1.2, 1.2, 400)
    bowl = np.clip(xs**2, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(xs, bowl, color="#6e8efb", linewidth=3)
    dot = ax.scatter([x_pos[0]], [en_norm[0]], s=120, color="#111111", zorder=5)
    txt = ax.text(0.02, 0.92, f"Step 0\nE={en[0]:.4f}", transform=ax.transAxes)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("Energy Descent (lower is better)")

    def update(f):
        dot.set_offsets([[x_pos[f], en_norm[f]]])
        txt.set_text(f"Step {f}\nE={en[f]:.4f}")
        return dot, txt

    anim = FuncAnimation(fig, update, frames=len(en), interval=600, blit=True)

    if pillow_ok:
        try:
            writer = PillowWriter(fps=2)
            _ensure_dir(cfg.out_curve_anim)
            anim.save(cfg.out_curve_anim, writer=writer)
            print(f"Saved: {cfg.out_curve_anim}")
        except Exception as e:
            print(f"Could not save GIF ({e}). Showing panels instead.")
            plt.close(fig)
            plot_energy_curve_panels(model, ds, cfg)
            return
    else:
        print("PillowWriter not available; saving panels instead of GIF.")
        plt.close(fig)
        plot_energy_curve_panels(model, ds, cfg)
        return

    plt.close(fig)

def energy_margin_report(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=cfg.batch_size, split="test")
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        h = model.backbone(x)
        E = model.energy.energies_all_tokens(h, model.wte_weight)  # (B,T,V)
        # Focus last position for thinking analysis
        E_last = E[:, -1, :]  # (B,V)
        true_last = y[:, -1]
        # baseline predicted token from v0 = -E
        v0 = -E_last
        pred0 = v0.argmax(-1)
    # refined prediction after K steps on last position (needs gradients)
    idx_cond = x[:, -model.gpt_cfg.block_size :]
    with torch.enable_grad():
        idx_gen = model.generate_think(idx_cond, max_new_tokens=1, steps=cfg.steps, tau=cfg.tau, lr=cfg.lr, noise=cfg.noise, topk=cfg.topk_think)
    predK = idx_gen[:, -1]
    # margins
    Et_true = E_last.gather(-1, true_last.unsqueeze(-1)).squeeze(-1)
    Et_pred0 = E_last.gather(-1, pred0.unsqueeze(-1)).squeeze(-1)
    Et_predK = E_last.gather(-1, predK.unsqueeze(-1)).squeeze(-1)
    frac_true_lt_pred0 = (Et_true < Et_pred0).float().mean().item()
    frac_true_lt_predK = (Et_true < Et_predK).float().mean().item()
    margin0 = (Et_pred0 - Et_true).mean().item()
    marginK = (Et_predK - Et_true).mean().item()
    text = (
        f"Energy margin report (last position)\n"
        f"steps={cfg.steps}, tau={cfg.tau}, topk={cfg.topk_think}\n"
        f"fraction E_true < E_pred@0: {frac_true_lt_pred0:.3f}\n"
        f"fraction E_true < E_pred@K: {frac_true_lt_predK:.3f}\n"
        f"mean margin (pred@0 - true): {margin0:.4f}\n"
        f"mean margin (pred@K - true): {marginK:.4f}\n"
    )
    with open(cfg.out_margin, "w") as f:
        f.write(text)
    print(text.strip())
    print(f"Saved: {cfg.out_margin}")


def eval_report(model: EBTLanguageModel, ds: CharDataset, cfg: VizConfig):
    device = next(model.parameters()).device
    # Build a small test loader
    loader, _ = get_loader(cfg.data_path, model.gpt_cfg.block_size, batch_size=cfg.batch_size, split="test")

    # Save original values (bypass frozen constraint with object.__setattr__)
    K_save = int(getattr(model.ebt_cfg, "mcmc_num_steps", 0) or 0)
    refine_last_save = bool(getattr(model.ebt_cfg, "refine_last_position_only", True))
    tau_save = float(getattr(model.ebt_cfg, "entropy_reg_tau", 1.0))

    # Greedy (no thinking): set K=0
    object.__setattr__(model.ebt_cfg, "mcmc_num_steps", 0)
    object.__setattr__(model.ebt_cfg, "refine_last_position_only", False)

    losses_g = []
    acc_energy_argmin = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 5:
                break
            x, y = x.to(device), y.to(device)
            loss, logits, _ = model(x, targets=y)
            losses_g.append(loss.item())
            # accuracy@1 of energy argmin (no loop)
            h = model.backbone(x)
            E = model.energy.energies_all_tokens(h, model.wte_weight)
            pred = (-E).argmax(-1)  # argmin E over vocab
            acc = (pred == y).float().mean().item()
            acc_energy_argmin.append(acc)
    ppl_g = float(torch.exp(torch.tensor(losses_g).mean()).item())
    acc_energy = float(torch.tensor(acc_energy_argmin).mean().item())

    # Thinking K steps: set K>0 (needs gradients)
    object.__setattr__(model.ebt_cfg, "mcmc_num_steps", max(1, int(cfg.steps)))
    object.__setattr__(model.ebt_cfg, "refine_last_position_only", True)
    object.__setattr__(model.ebt_cfg, "entropy_reg_tau", float(cfg.tau))

    losses_t = []
    with torch.enable_grad():
        for i, (x, y) in enumerate(loader):
            if i >= 5:
                break
            x, y = x.to(device), y.to(device)
            loss, logits, extras = model(x, targets=y)
            # extras["perplexity"] reflects refined main loss
            losses_t.append(loss.item())
    ppl_t = float(torch.exp(torch.tensor(losses_t).mean()).item())

    # Restore original values
    object.__setattr__(model.ebt_cfg, "mcmc_num_steps", K_save)
    object.__setattr__(model.ebt_cfg, "refine_last_position_only", refine_last_save)
    object.__setattr__(model.ebt_cfg, "entropy_reg_tau", tau_save)

    text = (
        f"Eval report (small test sample)\n"
        f"ppl_greedy: {ppl_g:.3f}\n"
        f"ppl_think(K={cfg.steps}): {ppl_t:.3f}\n"
        f"accuracy@1 (energy argmin, no loop): {acc_energy:.3f}\n"
    )
    with open(cfg.out_eval, "w") as f:
        f.write(text)
    print(text.strip())
    print(f"Saved: {cfg.out_eval}")


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
    model_cfg = ModelConfig(**ckpt["config"]["model"]) if "config" in ckpt else ModelConfig()
    model = EBTLanguageModel(model_cfg, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])  # load weights
    model.eval()

    ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")

    if cfg.mode == "topk":
        visualize_topk(model, ds, cfg)
    elif cfg.mode == "correlation":
        visualize_correlation(model, ds, cfg)
    elif cfg.mode == "gap":
        plot_expected_energy_gap(model, ds, cfg)
    elif cfg.mode == "trajectories":
        plot_token_logit_trajectories(model, ds, cfg)
    elif cfg.mode == "shift":
        hist_expected_energy_shift(model, ds, cfg)
    elif cfg.mode == "margin":
        energy_margin_report(model, ds, cfg)
    elif cfg.mode == "eval":
        eval_report(model, ds, cfg)
    elif cfg.mode == "curve":
        plot_energy_curve_panels(model, ds, cfg)
    elif cfg.mode == "curve_anim":
        animate_energy_curve(model, ds, cfg)
    elif cfg.mode == "surface":
        plot_energy_surface(model, ds, cfg, make_anim=False)
    elif cfg.mode == "surface_anim":
        plot_energy_surface(model, ds, cfg, make_anim=True)
    elif cfg.mode == "all":
        visualize_topk(model, ds, cfg)
        visualize_correlation(model, ds, cfg)
        plot_expected_energy_gap(model, ds, cfg)
        plot_token_logit_trajectories(model, ds, cfg)
        hist_expected_energy_shift(model, ds, cfg)
        energy_margin_report(model, ds, cfg)
        eval_report(model, ds, cfg)
        plot_energy_curve_panels(model, ds, cfg)
        animate_energy_curve(model, ds, cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    config = chz.entrypoint(VizConfig)
    main(config)
