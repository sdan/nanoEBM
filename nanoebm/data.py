import os
import torch
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, block_size=256, split="train", split_ratio=0.9):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
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


def _list_parquet_files(data_dir: str) -> list[str]:
    try:
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    except FileNotFoundError:
        return []
    files.sort()
    return [os.path.join(data_dir, f) for f in files]


def _iter_parquet_texts(files: Iterable[str]) -> Iterable[str]:
    if pq is None:
        raise ImportError(
            "pyarrow is required to read parquet files. Please install pyarrow or provide a .txt dataset."
        )
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])  # type: ignore
            texts = rg.column("text").to_pylist()
            for t in texts:
                if not isinstance(t, str):
                    continue
                yield t


def _build_vocab_from_texts(texts: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    charset = set()
    for t in texts:
        charset.update(t)
    chars = sorted(list(charset))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class ParquetCharDataset(torch.utils.data.Dataset):
    """
    Character-level dataset over a directory of parquet shards with a 'text' column.
    By default, uses all shards except the last for train, and the last shard for val.
    Builds a shared vocabulary provided via `vocab` or computed from the union of selected shards.
    Concatenates all texts in the split into one long string and samples contiguous blocks.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 256,
        split: str = "train",
        vocab: Optional[Dict[str, int]] = None,
        max_shards: Optional[int] = None,
    ):
        assert split in {"train", "val"}
        all_files = _list_parquet_files(data_dir)
        if not all_files:
            raise FileNotFoundError(f"No .parquet files found under {data_dir}")
        if max_shards is not None and max_shards > 0:
            all_files = all_files[: max_shards]

        # Split: last shard as val for determinism
        train_files = all_files[:-1] if len(all_files) > 1 else all_files
        val_files = all_files[-1:] if len(all_files) > 1 else []
        use_files = train_files if split == "train" else val_files
        if split == "val" and not use_files:
            # Fallback: if only one shard present, use the same file for val
            use_files = all_files[-1:]

        # Build vocabulary if not provided
        if vocab is None:
            # Union of train+val characters to keep mapping consistent across splits
            vocab_files = train_files + val_files if val_files else train_files
            stoi, itos = _build_vocab_from_texts(_iter_parquet_texts(vocab_files))
        else:
            stoi = dict(vocab)
            itos = {i: ch for ch, i in stoi.items()}

        # Materialize this split's text
        pieces = []
        for t in _iter_parquet_texts(use_files):
            pieces.append(t)
        text = "\n".join(pieces)

        # Encode to indices
        data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# -----------------------------
# BPE tokenized datasets (tiktoken)
# -----------------------------
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


class BPEDataset(torch.utils.data.Dataset):
    """Token-level dataset using tiktoken on a single text file."""

    def __init__(self, path: str, block_size: int = 256, split: str = "train", split_ratio: float = 0.9, enc_name: str = "gpt2"):
        if tiktoken is None:
            raise ImportError("tiktoken is required for BPEDataset. Please install tiktoken.")
        enc = tiktoken.get_encoding(enc_name)
        text = Path(path).read_text(encoding="utf-8")
        ids = enc.encode(text)
        data = torch.tensor(ids, dtype=torch.long)
        n = int(split_ratio * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.block_size = block_size
        self.vocab_size = enc.n_vocab
        # Compatibility placeholders for code paths that expect attributes
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class ParquetBPEDataset(torch.utils.data.Dataset):
    """Token-level dataset over parquet shards with a 'text' column using tiktoken."""

    def __init__(
        self,
        data_dir: str,
        block_size: int = 256,
        split: str = "train",
        enc_name: str = "gpt2",
        max_shards: Optional[int] = None,
    ):
        assert split in {"train", "val"}
        if tiktoken is None:
            raise ImportError("tiktoken is required for ParquetBPEDataset. Please install tiktoken.")
        enc = tiktoken.get_encoding(enc_name)
        all_files = _list_parquet_files(data_dir)
        if not all_files:
            raise FileNotFoundError(f"No .parquet files found under {data_dir}")
        if max_shards is not None and max_shards > 0:
            all_files = all_files[: max_shards]
        train_files = all_files[:-1] if len(all_files) > 1 else all_files
        val_files = all_files[-1:] if len(all_files) > 1 else []
        use_files = train_files if split == "train" else (val_files if val_files else all_files[-1:])

        tokens: list[int] = []
        for t in _iter_parquet_texts(use_files):
            if not t:
                continue
            tokens.extend(enc.encode(t))
            # Separate docs with EOT if available, else newline tokenization is already baked in BPE
            if hasattr(enc, "eot_token") and getattr(enc, "eot_token") is not None:
                tokens.append(enc.eot_token)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        self.vocab_size = enc.n_vocab
        self.stoi = {}
        self.itos = {}

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_loader(
    path: str,
    block_size: int,
    batch_size: int,
    split: str,
    vocab: Optional[Dict[str, int]] = None,
    max_shards: Optional[int] = None,
    tokenizer: str = "char",
    bpe_encoding: str = "gpt2",
):
    """
    Build a DataLoader and dataset.
    - If `path` is a directory containing .parquet shards, use ParquetCharDataset.
    - Else, treat `path` as a text file and use CharDataset.
    For parquet, pass `vocab` from the train split into the val split to keep mappings identical.
    """
    if os.path.isdir(path) and _list_parquet_files(path):
        if tokenizer == "char":
            ds = ParquetCharDataset(path, block_size, split, vocab=vocab, max_shards=max_shards)
        else:
            ds = ParquetBPEDataset(path, block_size, split, enc_name=bpe_encoding, max_shards=max_shards)
    else:
        # Text file path
        if tokenizer == "char":
            ds = CharDataset(path, block_size, split)
        else:
            ds = BPEDataset(path, block_size, split, enc_name=bpe_encoding)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, ds
