"""Debug script to check what's happening during refinement"""

import torch
import torch.nn.functional as F
from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.data import CharDataset

# Load model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint = torch.load('out_ebt/refine4.pt', map_location=device, weights_only=True)

# Load config
config_dict = checkpoint["config"]["model"]
# Remove parameters that no longer exist
removed_params = ['use_replay_buffer']
for param in removed_params:
    config_dict.pop(param, None)
model_cfg = ModelConfig(**config_dict)

# Create and load model
model = EBM(model_cfg).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# Load dataset
ds = CharDataset("shakespeare.txt", block_size=model_cfg.block_size, split="train")

# Create a prompt
prompt = "ROMEO:"
stoi, itos = ds.stoi, ds.itos
idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long, device=device)

print(f"Model alpha: {model.alpha.item()}")
print(f"Model refine_steps: {model.config.refine_steps}")
print(f"Prompt: {prompt}")
print(f"Input shape: {idx.shape}")

# Get hidden states and energy
with torch.no_grad():
    h = model.get_hidden_states(idx)  # (B, T, n_embd)
    energies = model.energy_head(h)  # (B, T, V)
    print(f"Hidden states shape: {h.shape}")
    print(f"Energies shape: {energies.shape}")
    print(f"Energy range: [{energies.min().item():.3f}, {energies.max().item():.3f}]")

# Now let's manually run the refinement and see what happens
B, T, V = energies.shape
last_energies = energies[:, -1, :]  # (B, V) - energies for last position

# Initialize logits from System 1
logits = -last_energies.clone()  # S1 initialization
logits = logits.requires_grad_(True)

print(f"\nInitial logits shape: {logits.shape}")
print(f"Initial logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

# Manual gradient descent loop
alpha = model.alpha.item()
print(f"\nStarting gradient descent with alpha={alpha}")

for step in range(4):
    # Current probability distribution
    probs = F.softmax(logits, dim=-1)  # (B, V)

    # Expected energy under current distribution: E_p[E(x,y)]
    expected_energy = (probs * last_energies.detach()).sum(dim=-1).mean()

    # Compute gradient
    grad = torch.autograd.grad(expected_energy, logits, retain_graph=True)[0]

    print(f"\nStep {step}:")
    print(f"  Expected energy: {expected_energy.item():.6f}")
    print(f"  Grad norm: {grad.norm().item():.6f}")
    print(f"  Grad range: [{grad.min().item():.6f}, {grad.max().item():.6f}]")

    # Update logits with gradient descent
    with torch.no_grad():
        logits_before = logits.clone()
        logits -= alpha * grad
        change = (logits - logits_before).abs().max().item()
        print(f"  Max logit change: {change:.6f}")

    # Re-enable gradients for next iteration
    logits = logits.detach().requires_grad_(True)

# Now test model's own system2_refine
print("\n" + "="*50)
print("Testing model.system2_refine:")
refined_logits = model.system2_refine(idx, steps=4, return_trajectory=True)

if isinstance(refined_logits, tuple):
    refined_logits_final, trajectory = refined_logits
else:
    trajectory = refined_logits

print(f"Trajectory type: {type(trajectory)}")
if isinstance(trajectory, list):
    print(f"Trajectory length: {len(trajectory)}")
    if len(trajectory) > 0:
        print(f"First logits shape: {trajectory[0].shape}")
        print(f"Last logits shape: {trajectory[-1].shape}")

        # Check if logits are changing
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                if trajectory[i].dim() == 3:  # (B, T, V)
                    diff = (trajectory[i][:, -1, :] - trajectory[i-1][:, -1, :]).abs().max().item()
                else:  # (B, V)
                    diff = (trajectory[i] - trajectory[i-1]).abs().max().item()
                print(f"  Max change step {i-1} -> {i}: {diff:.6f}")