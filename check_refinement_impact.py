"""Check if refinement actually improves predictions"""

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

# Test prompts
prompts = ["ROMEO:", "To be or", "What is", "The king"]

print(f"Alpha: {model.alpha.item()}")
print(f"Config refine_steps: {model.config.refine_steps}")
print("\nComparing System 1 (no refinement) vs System 2 (with refinement):\n")

for prompt in prompts:
    # Encode prompt
    stoi = ds.stoi
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long, device=device)

    # System 1 (no refinement)
    with torch.no_grad():
        loss_s1, logits_s1, extras_s1 = model(idx, use_refine=False)
        probs_s1 = F.softmax(logits_s1[0, -1, :], dim=-1)
        top5_s1 = torch.topk(probs_s1, 5)

    # System 2 (with refinement) - needs gradients for refinement
    # Use torch.enable_grad() to allow gradient computation even in eval mode
    with torch.enable_grad():
        loss_s2, logits_s2, extras_s2 = model(idx, use_refine=True, refine_steps=8)

    with torch.no_grad():
        probs_s2 = F.softmax(logits_s2[0, -1, :], dim=-1)
        top5_s2 = torch.topk(probs_s2, 5)

    print(f"Prompt: '{prompt}'")
    print(f"  System 1 top 5:")
    for i in range(5):
        tok = ds.itos[top5_s1.indices[i].item()]
        prob = top5_s1.values[i].item()
        print(f"    {repr(tok):8s} {prob:.3f}")

    print(f"  System 2 top 5 (after {8} refinement steps):")
    for i in range(5):
        tok = ds.itos[top5_s2.indices[i].item()]
        prob = top5_s2.values[i].item()
        print(f"    {repr(tok):8s} {prob:.3f}")

    # Check if predictions changed
    if top5_s1.indices[0] != top5_s2.indices[0]:
        print(f"  -> Top prediction CHANGED!")
    else:
        prob_change = abs(top5_s2.values[0].item() - top5_s1.values[0].item())
        print(f"  -> Top prediction same, prob change: {prob_change:.4f}")

    # Check energy gap if available
    if 'energy_gap' in extras_s2:
        print(f"  Energy gap: {extras_s2['energy_gap']:.6f}")

    print()