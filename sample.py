# sample.py
import torch
from nanoebm.config import ModelConfig
from nanoebm.model import EBTLanguageModel
from nanoebm.data import CharDataset

@torch.no_grad()
def decode(idx, itos):
    return "".join(itos[i] for i in idx.tolist())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("nanoebm.pt", map_location=device)
    model_cfg = ModelConfig(**ckpt["gcfg"])
    model = EBTLanguageModel(model_cfg, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = CharDataset("shakespeare.txt", block_size=model_cfg.block_size, split="train")
    stoi, itos = ds.stoi, ds.itos

    # prompt
    prompt = "ROMEO:"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

    # choose one:
    out = model.generate_greedy(idx.clone(), max_new_tokens=200)
    # or with thinking (e.g., 4 steps, top-k=64)
    # out = model.generate_think(idx.clone(), max_new_tokens=200, steps=4, topk=64)

    txt = decode(out[0], itos)
    print(txt)

if __name__ == "__main__":
    main()
