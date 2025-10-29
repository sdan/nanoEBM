"""
OpenFineWeb Dataset Download Utilities

This file contains utilities for downloading the FineWeb-Edu-100B dataset:
- Download parquet files on demand with retry logic
- Parallel download with multiprocessing
- Iterate over parquet files and yield documents

Dataset: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from pathlib import Path

# -----------------------------------------------------------------------------
# Dataset configuration

# The URL where the data is hosted
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # Last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

# Default data directory (can be overridden)
DATA_DIR = os.path.expanduser("~/data/openfineweb")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Utility functions for other modules

def list_parquet_files(data_dir=None):
    """List all parquet files in a directory, sorted by name."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset in batches of underlying row_groups.

    Args:
        split: "train" or "val" (last parquet file is used for val)
        start: Starting row group index (useful for DDP, e.g., start=rank)
        step: Step size for row groups (useful for DDP, e.g., step=world_size)
        data_dir: Directory containing parquet files (defaults to DATA_DIR)

    Yields:
        List of text strings from each row group
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir)

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir or DATA_DIR}")

    # Use all but last file for train, last file for val
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


# -----------------------------------------------------------------------------
# Download functions

def download_single_file(index):
    """Download a single parquet file with retry logic and exponential backoff."""

    # Construct filepath and skip if exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"✓ Skipping {filename} (already exists)")
        return True

    # Construct remote URL
    url = f"{BASE_URL}/{filename}"
    print(f"→ Downloading {filename}...")

    # Download with retries and exponential backoff
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"✓ Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"✗ Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

            # Clean up any partial files
            for path in [temp_path, filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

            # Exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_shards(num_shards=-1, num_workers=4, data_dir=None):
    """
    Download multiple shards in parallel.

    Args:
        num_shards: Number of shards to download (-1 for all)
        num_workers: Number of parallel workers
        data_dir: Target directory (defaults to DATA_DIR)

    Returns:
        Tuple of (successful_downloads, total_shards)
    """
    global DATA_DIR
    if data_dir:
        DATA_DIR = os.path.expanduser(data_dir)
        os.makedirs(DATA_DIR, exist_ok=True)

    num = MAX_SHARD + 1 if num_shards == -1 else min(num_shards, MAX_SHARD + 1)
    ids_to_download = list(range(num))

    print(f"\n{'='*60}")
    print(f"OpenFineWeb Dataset Download")
    print(f"{'='*60}")
    print(f"Shards to download: {len(ids_to_download)}")
    print(f"Parallel workers: {num_workers}")
    print(f"Target directory: {DATA_DIR}")
    print(f"{'='*60}\n")

    with Pool(processes=num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"{'='*60}")
    print(f"✓ Downloaded: {successful}/{len(ids_to_download)} shards")
    print(f"✓ Location: {DATA_DIR}")
    print(f"{'='*60}\n")

    return successful, len(ids_to_download)


# -----------------------------------------------------------------------------
# Main CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu-100BT dataset shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 10 shards with 4 workers
  python download_data.py -n 10 -w 4

  # Download all shards (1823 total) with 8 workers
  python download_data.py -n -1 -w 8

  # Download to custom directory
  python download_data.py -n 10 -w 4 -d ~/data/custom_dir
        """
    )
    parser.add_argument(
        "-n", "--num-shards",
        type=int,
        default=10,
        help="Number of shards to download (default: 10, -1 for all 1823 shards)"
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default=None,
        help=f"Target directory for downloads (default: {DATA_DIR})"
    )

    args = parser.parse_args()

    # Download shards
    download_shards(
        num_shards=args.num_shards,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
