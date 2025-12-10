"""
Datasets and data loaders.

- Character-level Shakespeare dataset (original nanoEBM setup)
- GSM8K BPE dataset built with tiktoken + HuggingFace datasets
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from datasets import load_dataset


# ============================================================================
# Character-level Shakespeare (unchanged behavior for train.py)
# ============================================================================


class CharDataset(Dataset):
    """Simple character-level dataset over a single text file.

    This is kept exactly compatible with the original implementation used
    by train.py so existing experiments continue to work.
    """

    def __init__(self, path: str, block_size: int = 256, split: str = "train", split_ratio: float = 0.9):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split_ratio * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # (T)
        y = chunk[1:]  # (T)
        return x, y


def get_loader(path: str, block_size: int, batch_size: int, split: str):
    """Data loader used by the original Shakespeare training script."""
    ds = CharDataset(path, block_size, split)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True),
        ds,
    )


# ============================================================================
# GSM8K BPE dataset (for math reasoning experiments)
# ============================================================================


class Gsm8kBpeDataset(Dataset):
    """GSM8K language modeling dataset using a fixed BPE (tiktoken).

    We pack all (question, answer) pairs for a split into one long token
    stream, separated by <|endoftext|>, and train with standard next-token
    prediction over that stream. This keeps the training loop identical to
    the character-level setup while moving to a realistic math domain.
    """

    def __init__(
        self,
        split: str,
        block_size: int = 512,
        encoding_name: str = "gpt2",
        cache_dir: str | None = None,
    ):
        super().__init__()
        assert split in {"train", "test"}, "GSM8K has 'train' and 'test' splits"

        # Fixed, off-the-shelf BPE encoding (no training step required)
        self.enc = tiktoken.get_encoding(encoding_name)
        self.block_size = block_size

        ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

        # Build one big text buffer with simple separators.
        # We intentionally avoid using special tokens like "<|endoftext|>"
        # to keep tiktoken encoding straightforward.
        texts = []
        for ex in ds:
            q = ex["question"].strip()
            a = ex["answer"].strip()
            text = f"Question: {q}\nAnswer: {a}\n\n"
            texts.append(text)

        full_text = "\n\n".join(texts)

        tokens = self.enc.encode(full_text)
        self.data = torch.tensor(tokens, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_gsm8k_loader(
    split: str,
    block_size: int,
    batch_size: int,
    encoding_name: str = "gpt2",
    cache_dir: str | None = None,
) -> Tuple[DataLoader, Gsm8kBpeDataset]:
    """Return a DataLoader and underlying dataset for GSM8K BPE training."""
    ds = Gsm8kBpeDataset(
        split=split,
        block_size=block_size,
        encoding_name=encoding_name,
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )
    return loader, ds
