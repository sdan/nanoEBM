"""
Quick demo: Train tiny models to show GPT vs EBM difference
Trains in ~1 minute on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class TinyGPT(nn.Module):
    """Minimal GPT for quick training."""

    def __init__(self, vocab_size=67, n_embd=64, n_head=2, n_layer=2, block_size=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

class TinyEBM(TinyGPT):
    """Tiny GPT with energy-based refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 0.1  # Step size for refinement

    def refine_logits(self, logits, steps=4, training=False):
        """Refine logits using gradient descent on energy."""
        # During training, keep gradients; during inference, detach
        if not training:
            logits = logits.detach()

        logits = logits.requires_grad_(True)

        for _ in range(steps):
            # Energy is negative log probability
            energies = -F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            # Expected energy
            expected_energy = (probs * energies).sum(dim=-1).mean()

            # Gradient step
            grad = torch.autograd.grad(expected_energy, logits, create_graph=training)[0]
            logits = logits - self.alpha * grad

            if not training:
                logits = logits.detach()
            logits = logits.requires_grad_(True)

        return logits

    def forward(self, idx, use_refine=False, refine_steps=4):
        logits = super().forward(idx)
        if use_refine:
            logits = self.refine_logits(logits, refine_steps, training=self.training)
        return logits

class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffwd(self.ln2(x))
        return x

def get_batch(data, block_size, batch_size):
    """Get a random batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train_tiny_model(model_type="gpt", steps=500):
    """Train a tiny model quickly."""

    # Load Shakespeare
    with open('shakespeare.txt', 'r') as f:
        text = f.read()[:100000]  # Use only first 100k chars for speed

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model settings
    vocab_size = len(chars)
    block_size = 32
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    if model_type == "gpt":
        model = TinyGPT(vocab_size=vocab_size, block_size=block_size).to(device)
    else:
        model = TinyEBM(vocab_size=vocab_size, block_size=block_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🚀 Training {model_type.upper()} (tiny version)")
    print("-" * 40)

    # Training loop
    for step in range(steps):
        model.train()

        # Get batch
        x, y = get_batch(train_data, block_size, batch_size)
        x, y = x.to(device), y.to(device)

        # Forward pass
        if model_type == "gpt":
            logits = model(x)
        else:
            # EBM: use refinement after warmup
            use_refine = step > 100
            if use_refine:
                # Refinement needs gradients
                logits = model(x, use_refine=True, refine_steps=2)
            else:
                logits = model(x, use_refine=False)

        # Loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, block_size, batch_size)
                x_val, y_val = x_val.to(device), y_val.to(device)

                if model_type == "gpt":
                    logits_val = model(x_val)
                else:
                    logits_val = model(x_val, use_refine=False)

                val_loss = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1))

            print(f"Step {step:3d}: train loss {loss.item():.3f}, val loss {val_loss.item():.3f}")

    # Save model
    torch.save({
        'model': model.state_dict(),
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi
    }, f'tiny_{model_type}.pt')

    return model, itos

def compare_tiny_models():
    """Compare the tiny models."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    gpt_ckpt = torch.load('tiny_gpt.pt', map_location=device, weights_only=False)
    ebm_ckpt = torch.load('tiny_ebm.pt', map_location=device, weights_only=False)

    vocab_size = gpt_ckpt['vocab_size']
    itos = gpt_ckpt['itos']

    gpt = TinyGPT(vocab_size=vocab_size).to(device)
    gpt.load_state_dict(gpt_ckpt['model'])
    gpt.eval()

    ebm = TinyEBM(vocab_size=vocab_size).to(device)
    ebm.load_state_dict(ebm_ckpt['model'])
    ebm.eval()

    # Load test data
    with open('shakespeare.txt', 'r') as f:
        text = f.read()[100000:110000]  # Different part for testing

    stoi = gpt_ckpt['stoi']
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Test both models
    print("\n" + "="*50)
    print("TINY MODEL COMPARISON")
    print("="*50)

    block_size = 32
    batch_size = 32
    n_batches = 20

    losses = {'gpt': [], 'ebm_base': [], 'ebm_refined': []}

    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size)
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            # GPT
            logits_gpt = gpt(x)
            loss_gpt = F.cross_entropy(logits_gpt.view(-1, logits_gpt.size(-1)), y.view(-1))
            losses['gpt'].append(loss_gpt.item())

            # EBM without refinement
            logits_ebm_base = ebm(x, use_refine=False)
            loss_ebm_base = F.cross_entropy(logits_ebm_base.view(-1, logits_ebm_base.size(-1)), y.view(-1))
            losses['ebm_base'].append(loss_ebm_base.item())

        # EBM with refinement (needs gradients)
        with torch.enable_grad():
            logits_ebm_refined = ebm(x, use_refine=True, refine_steps=4)
            loss_ebm_refined = F.cross_entropy(logits_ebm_refined.view(-1, logits_ebm_refined.size(-1)), y.view(-1))
            losses['ebm_refined'].append(loss_ebm_refined.detach().item())

    # Results
    print("\nAverage Loss / Perplexity:")
    print("-" * 30)

    for name, label in [('gpt', 'GPT'), ('ebm_base', 'EBM (no refine)'), ('ebm_refined', 'EBM (refined)')]:
        avg_loss = np.mean(losses[name])
        ppl = np.exp(avg_loss)
        print(f"{label:15} {avg_loss:.3f} / {ppl:.2f}")

    # Check if refinement helps
    gpt_ppl = np.exp(np.mean(losses['gpt']))
    ebm_refined_ppl = np.exp(np.mean(losses['ebm_refined']))

    print("\n" + "="*50)
    if ebm_refined_ppl < gpt_ppl:
        improvement = (gpt_ppl - ebm_refined_ppl) / gpt_ppl * 100
        print(f"✅ EBM refinement improves by {improvement:.1f}%!")
    else:
        print("❌ No improvement from refinement")
    print("="*50)

    # Generate samples to see the difference
    print("\n📝 SAMPLE GENERATION")
    print("-" * 30)

    # Start with a prompt
    prompt = "To be or not to be"
    prompt_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

    # Generate from both models
    def generate(model, start_ids, length=50, use_refine=False):
        ids = start_ids.clone()
        for _ in range(length):
            # Get next token prediction
            if hasattr(model, 'refine_logits'):
                logits = model(ids[:, -32:], use_refine=use_refine)  # Use last 32 tokens
            else:
                logits = model(ids[:, -32:])

            # Sample
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

        return ''.join([itos[i] for i in ids[0].cpu().tolist()])

    print(f"Prompt: '{prompt}'")
    print("\nGPT completion:")
    print(generate(gpt, prompt_ids))
    print("\nEBM (no refinement) completion:")
    print(generate(ebm, prompt_ids, use_refine=False))
    print("\nEBM (with refinement) completion:")
    print(generate(ebm, prompt_ids, use_refine=True))

def main():
    """Run the quick demo."""

    print("\n🎓 QUICK DEMO: GPT vs EBM")
    print("Training tiny models to show the difference...\n")

    # Train both models
    print("1. Training tiny GPT...")
    gpt_model, itos = train_tiny_model("gpt", steps=300)

    print("\n2. Training tiny EBM...")
    ebm_model, _ = train_tiny_model("ebm", steps=300)

    # Compare them
    compare_tiny_models()

if __name__ == "__main__":
    main()