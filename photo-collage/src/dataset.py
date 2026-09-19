import os
import sys
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_tile_paths(max_tiles=None):
    random.seed(config.SEED)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG"]
    all_paths = []
    for ext in extensions:
        found = glob.glob(os.path.join(config.DATA_DIR, "**", ext), recursive=True)
        all_paths.extend(found)
    all_paths = sorted(set(all_paths))  # deterministic source order
    random.shuffle(all_paths)
    if max_tiles is not None:
        all_paths = all_paths[:max_tiles]
    if not all_paths:
        print("[dataset] original source images not found; using precomputed cache ordering")
        n = max_tiles if max_tiles is not None else (config.POOL_TILES or 81444)
        manifest_path = os.path.join(
            config.CACHE_DIR,
            f"basic_mean_rgb_paths_N{n}.txt",
        )
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) != n:
                raise RuntimeError(
                    f"Portable cache manifest has {len(names)} entries; expected {n}"
                )
            tile_paths = [
                os.path.join(config.DATA_DIR, os.path.basename(name))
                for name in names
            ]
            print(f"[dataset] loaded portable cache ordering: {len(tile_paths)} tiles")
            return tile_paths
        raise RuntimeError(
            f"Original dataset not found and cache ordering manifest is missing: {manifest_path}"
        )
    print(f"[dataset] found {len(all_paths)} tile images")
    return all_paths


def load_tile_rgb(path, size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    left  = (w - short) // 2
    top   = (h - short) // 2
    img   = img.crop((left, top, left + short, top + short))
    img   = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _build_cache(tile_paths, tile_size, cache_path):
    N = len(tile_paths)
    cache = np.zeros((N, tile_size, tile_size, 3), dtype=np.uint8)
    for i, path in enumerate(tqdm(tile_paths, desc=f"caching tiles ({tile_size}px)")):
        try:
            cache[i] = load_tile_rgb(path, tile_size)
        except Exception:
            pass
    np.save(cache_path, cache)
    print(f"[dataset] cache saved: {cache_path}")
    return cache


def get_tile_cache(tile_paths, tile_size):
    """Small cache at TILE_SIZE_MIN for feature extraction."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        config.CACHE_DIR,
        f"tiles_small{tile_size}_N{len(tile_paths)}.npy"
    )
    if os.path.exists(cache_path):
        print(f"[dataset] loading small cache: {cache_path}")
        return np.load(cache_path)
    print(f"[dataset] building small cache...")
    return _build_cache(tile_paths, tile_size, cache_path)


def get_hires_tile_cache(tile_paths):
    """
    Returns a memory-mapped array of the hi-res cache.
    Memory-mapped = data stays on disk, only loaded when accessed.
    This avoids the 16GB RAM problem.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    hires_size = config.TILE_SIZE_MAX
    cache_path = os.path.join(
        config.CACHE_DIR,
        f"tiles_hires{hires_size}_N{len(tile_paths)}.npy"
    )
    if os.path.exists(cache_path):
        print(f"[dataset] memory-mapping hi-res cache: {cache_path}")
        return np.load(cache_path, mmap_mode='r')
    print(f"[dataset] building hi-res cache...")
    return _build_cache(tile_paths, hires_size, cache_path)
