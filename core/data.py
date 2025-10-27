from __future__ import annotations

import os
import numpy as np
import torch
from typing import Tuple


def memmap_dataset(data_dir: str, split: str, dtype=np.uint16):
    path = os.path.join(data_dir, f"{split}.bin")
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(
    data_dir: str, block_size: int, batch_size: int, split: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        data = memmap_dataset(data_dir, "train")
    else:
        data = memmap_dataset(data_dir, "val")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    x = x.to(device)
    y = y.to(device)
    return x, y

