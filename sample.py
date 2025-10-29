"""sample nanoEBM

Usage:
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt
    python sample.py checkpoint=out_ebt/final.pt max_new_tokens=500 prompt="HAMLET:"
    # Thinking mode (iterative refinement)
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt use_thinking=true think_steps=4 topk=64
    # Thinking + sampling (stabilizes and reduces repetition)
    python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64 sample=true sample_temp=1.2 sample_top_p=0.9
"""
import chz
import torch
import torch.nn.functional as F
from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import CharDataset


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
class SampleConfig:
    """Configuration for sampling from a trained EBM model"""
    checkpoint: str | None = None  # None = auto-detect latest checkpoint
    data_path: str = "shakespeare.txt"  # Path to training data (for vocab)
    prompt: str = "ROMEO:"  # Text prompt to start generation
    max_new_tokens: int = 200  # Number of tokens to generate
    
    # EBM parameters
    mode: str = "think"  # Sampling mode: 'fast' (System 1), 'think' (System 2), or 'adaptive'
    think_steps: int = 4       # Number of refinement steps when thinking
    topk: int | None = 50      # Restrict to top-k tokens (None = use all vocab)
    adaptive_threshold: float = 2.0  # Entropy threshold for adaptive mode
    
    # Sampling parameters
    sample: bool = False  # Sample from distribution vs greedy
    sample_temp: float = 1.0  # Temperature for sampling


@torch.no_grad()
def decode(idx, itos):
    return "".join(itos[i] for i in idx.tolist())


def main(cfg: SampleConfig):
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
    model_cfg = ModelConfig(**ckpt["config"]["model"])

    # Initialize model and load weights
    model = EBM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])  # shapes now match the checkpoint
    model.eval()

    print(f"Loaded model from step {ckpt['step']}")

    # Load dataset for vocabulary and decoding
    ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")
    stoi, itos = ds.stoi, ds.itos
    if len(stoi) != model_cfg.vocab_size:
        print(
            f"Warning: dataset vocab_size ({len(stoi)}) != model vocab_size ({model_cfg.vocab_size}). "
            "Ensure you are using the same data file used for training."
        )

    # Encode prompt (filter unknown chars for robustness)
    print(f"\nPrompt: {cfg.prompt!r}")
    known = [c for c in cfg.prompt if c in stoi]
    dropped = [c for c in cfg.prompt if c not in stoi]
    if dropped:
        print(f"[warn] Dropping {len(dropped)} unknown chars from prompt: {repr(''.join(dropped))}")
    idx = torch.tensor([[stoi[c] for c in known]], dtype=torch.long, device=device)

    # Generate based on mode
    if cfg.mode == "fast":
        print("Generating with System 1 (fast mode)...")
        out = model.generate(
            idx.clone(),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.sample_temp if cfg.sample else 1.0,
            top_k=cfg.topk,
            use_thinking=False
        )
    elif cfg.mode == "think":
        print(f"Generating with System 2 (thinking mode, steps={cfg.think_steps})...")
        out = model.generate(
            idx.clone(),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.sample_temp if cfg.sample else 1.0,
            top_k=cfg.topk,
            use_thinking=True,
            think_steps=cfg.think_steps
        )
    elif cfg.mode == "adaptive":
        print(f"Generating with adaptive mode (entropy threshold={cfg.adaptive_threshold})...")
        # Custom generation loop for adaptive mode
        generated = idx.clone()
        for _ in range(cfg.max_new_tokens):
            # Crop context if needed
            idx_cond = generated if generated.size(1) <= model.config.block_size else generated[:, -model.config.block_size:]
            
            # Get System 1 logits first
            with torch.no_grad():
                logits_s1 = model.system1_direct_energy(idx_cond)[:, -1, :]  # Last position
                
                # Calculate entropy to decide if we need System 2
                probs = F.softmax(logits_s1, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
                
                # Use System 2 if entropy is high (uncertain)
                if entropy.mean() > cfg.adaptive_threshold:
                    logits = model.system2_refine(idx_cond, steps=cfg.think_steps)[:, -1, :]
                else:
                    logits = logits_s1
                
                # Apply temperature and top-k
                logits = logits / (cfg.sample_temp if cfg.sample else 1.0)
                if cfg.topk is not None:
                    v, _ = torch.topk(logits, min(cfg.topk, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample or greedy
                probs = F.softmax(logits, dim=-1)
                if cfg.sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = probs.argmax(dim=-1, keepdim=True)
                
                # Append
                generated = torch.cat((generated, idx_next), dim=1)
        
        out = generated
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'fast', 'think', or 'adaptive'")

    # Decode and print
    txt = decode(out[0], itos)
    print("\n" + "="*80)
    print(txt)
    print("="*80)


if __name__ == "__main__":
    config = chz.entrypoint(SampleConfig)
    main(config)
