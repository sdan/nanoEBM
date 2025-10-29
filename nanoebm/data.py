# From https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/projects/chargpt/chargpt.py#L42
import torch
from pathlib import Path


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size=256, split="train", split_ratio=0.9):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split_ratio * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # (T)
        y = chunk[1:]  # (T)
        return x, y


def get_loader(path, block_size, batch_size, split):
    ds = CharDataset(path, block_size, split)
    return (
        torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True),
        ds,
    )
