"""sample nanoEBM

Usage:
    # Character-level (legacy)
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt
    python sample.py checkpoint=out_ebt/final.pt max_new_tokens=500 prompt="HAMLET:"

    # BPE (auto-detected from checkpoint config; requires tiktoken)
    python sample.py checkpoint=out_ebt/final.pt prompt="The future of AI is" max_new_tokens=128

    # Thinking mode (iterative refinement)
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt use_thinking=true think_steps=4 topk=64
    # Thinking + sampling (stabilizes and reduces repetition)
    python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64 sample=true sample_temp=1.2 sample_top_p=0.9
"""
import chz
import torch
from nanoebm.config import ModelConfig
from nanoebm.model import EBTLanguageModel
from nanoebm.data import CharDataset
from typing import Optional

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


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
    """Configuration for sampling from a trained nanoEBM model"""
    checkpoint: str | None = None  # None = auto-detect latest checkpoint
    data_path: str = "shakespeare.txt"  # Path to training data (for vocab)
    prompt: str = "ROMEO:"  # Text prompt to start generation
    max_new_tokens: int = 200  # Number of tokens to generate
    # Optional overrides (by default, tokenizer is read from checkpoint config)
    tokenizer: str | None = None  # 'char' or 'gpt2' (None = use checkpoint)
    bpe_encoding: str = "gpt2"    # tiktoken encoding name if tokenizer != 'char'
    
    # Thinking/refinement parameters
    use_thinking: bool = False  # Whether to use iterative refinement
    think_steps: int = 4  # Number of refinement steps (if use_thinking=True)
    think_lr: float = 1.0  # Learning rate for refinement
    think_tau: float = 1.0  # Temperature for softmax
    think_noise: float = 0.0  # Noise level for Langevin dynamics
    topk: int | None = None  # Restrict to top-k tokens (None = use all vocab)
    # Decoding controls (when use_thinking=True)
    sample: bool = False  # Sample from refined logits instead of argmax
    sample_temp: float = 1.0  # Sampling temperature
    sample_top_p: float | None = None  # Nucleus sampling cutoff (e.g., 0.9)


@torch.no_grad()
def decode_tokens(idx: torch.Tensor, *, itos: Optional[dict[int, str]] = None, enc=None) -> str:
    """Decode token ids to string using either char vocab (itos) or a BPE encoder."""
    ids = idx.tolist()
    if enc is not None:
        return enc.decode(ids)
    if itos is not None:
        return "".join(itos[i] for i in ids)
    raise ValueError("Provide either a BPE encoder or itos mapping for decoding.")


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
    model = EBTLanguageModel(model_cfg, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])  # shapes now match the checkpoint
    model.eval()

    print(f"Loaded model from step {ckpt['step']}")

    # Determine tokenizer from checkpoint (or override via cfg)
    data_cfg = ckpt.get("config", {}).get("data", {}) if isinstance(ckpt.get("config"), dict) else {}
    tokenizer = (cfg.tokenizer or data_cfg.get("tokenizer") or "char").lower()
    bpe_encoding = cfg.bpe_encoding or data_cfg.get("bpe_encoding") or "gpt2"

    enc = None
    stoi = None
    itos = None
    if tokenizer != "char":
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for BPE decoding. Install with `pip install tiktoken` or run via uv."
            )
        try:
            enc = tiktoken.get_encoding(bpe_encoding)
        except Exception as e:
            raise RuntimeError(f"Failed to load BPE encoding '{bpe_encoding}': {e}")
        # Sanity check vocab size
        if enc.n_vocab != model_cfg.vocab_size:
            print(
                f"Warning: BPE encoder vocab_size ({enc.n_vocab}) != model vocab_size ({model_cfg.vocab_size}). "
                "Ensure you are sampling with the same encoding used for training."
            )
        # Encode prompt with BPE
        print(f"\nPrompt: {cfg.prompt!r}")
        prompt_ids = enc.encode(cfg.prompt)
        if not prompt_ids:
            raise ValueError("Prompt encodes to empty token sequence under BPE.")
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        # Character-level: load dataset for stoi/itos
        ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")
        stoi, itos = ds.stoi, ds.itos
        if len(stoi) != model_cfg.vocab_size:
            print(
                f"Warning: dataset vocab_size ({len(stoi)}) != model vocab_size ({model_cfg.vocab_size}). "
                "Ensure you are using the same data file used for training."
            )
        # Encode prompt as characters
        print(f"\nPrompt: {cfg.prompt!r}")
        idx = torch.tensor([[stoi[c] for c in cfg.prompt if c in stoi]], dtype=torch.long, device=device)
        if idx.numel() == 0:
            raise ValueError("Prompt produced empty token sequence (unknown chars?)")

    # Generate
    if cfg.use_thinking:
        print(f"Generating with thinking (steps={cfg.think_steps}, topk={cfg.topk})...")
        out = model.generate_think(
            idx.clone(),
            max_new_tokens=cfg.max_new_tokens,
            steps=cfg.think_steps,
            lr=cfg.think_lr,
            tau=cfg.think_tau,
            noise=cfg.think_noise,
            topk=cfg.topk,
            sample=cfg.sample,
            sample_temp=cfg.sample_temp,
            sample_top_p=cfg.sample_top_p,
        )
    else:
        print("Generating greedily...")
        out = model.generate_greedy(idx.clone(), max_new_tokens=cfg.max_new_tokens)

    # Decode and print
    txt = decode_tokens(out[0], itos=itos, enc=enc)
    print("\n" + "="*80)
    print(txt)
    print("="*80)


if __name__ == "__main__":
    config = chz.entrypoint(SampleConfig)
    main(config)
