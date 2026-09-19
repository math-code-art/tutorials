import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset  import get_tile_paths, get_tile_cache, get_hires_tile_cache
from src.features import get_embeddings
from src.mosaic   import build_mosaic


def run_mosaic(target_path):

    print("=" * 50)
    print(f"target image: {target_path}")
    print("=" * 50)

    # ── 1. Load + resize target image ────────────────────────────────────────
    print("\n[1/5] loading target image")
    img  = Image.open(target_path).convert("RGB")
    w, h = img.size

    # Scale so the longest edge ≤ TARGET_RESOLUTION (default 2400).
    # Larger values → more detail in the split, but slower.
    target_res = getattr(config, "TARGET_RESOLUTION", 2400)
    scale = min(1.0, target_res / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    target_img = np.array(img, dtype=np.uint8)

    # Snap to a multiple of TILE_SIZE_MAX so the quadtree grid is exact
    H, W, _ = target_img.shape
    H2 = (H // config.TILE_SIZE_MAX) * config.TILE_SIZE_MAX
    W2 = (W // config.TILE_SIZE_MAX) * config.TILE_SIZE_MAX
    target_img = target_img[:H2, :W2]
    print(f"      image size after snap: {target_img.shape}")

    # ── 2. Tile paths ─────────────────────────────────────────────────────────
    print("\n[2/5] loading tile paths...")
    tile_paths = get_tile_paths(max_tiles=config.POOL_TILES)

    # ── 3. Tile caches ────────────────────────────────────────────────────────
    # small cache  → used for VGG feature extraction (fast, compact)
    # hi-res cache → used for pasting onto canvas (sharp, no upsampling blur)
    print("\n[3/5] building tile caches...")
    tile_cache_small = get_tile_cache(tile_paths, config.TILE_SIZE_MIN)
    hires_cache      = get_hires_tile_cache(tile_paths)
    print(f"      small cache : {tile_cache_small.shape}")
    print(f"      hi-res cache: {hires_cache.shape}")

    # ── 4. VGG embeddings ─────────────────────────────────────────────────────
    print("\n[4/5] extracting VGG features...")
    emb3 = get_embeddings(tile_cache_small, config.CUT3)
    emb4 = get_embeddings(tile_cache_small, config.CUT4)
    print(f"      emb3: {emb3.shape}")
    print(f"      emb4: {emb4.shape}")

    # ── 5. Build mosaic ───────────────────────────────────────────────────────
    print("\n[5/5] building mosaic...")
    canvas = build_mosaic(target_img, target_path,
                          tile_cache_small, hires_cache,
                          emb3, emb4)

    print("\n" + "=" * 50)
    print("Done")
    print("=" * 50)
    return canvas


if __name__ == "__main__":
    extensions  = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    target_paths = []

    for filename in os.listdir(config.TARGET_DIR):
        if os.path.splitext(filename)[1] in extensions:
            target_paths.append(os.path.join(config.TARGET_DIR, filename))

    target_paths.sort()

    if not target_paths:
        print(f"error: no images found in {config.TARGET_DIR}")
        sys.exit(1)

    print(f"found {len(target_paths)} target image(s)")

    for target_path in target_paths:
        print("\n\n" + "#" * 60)
        print(f"processing: {target_path}")
        print("#" * 60)
        run_mosaic(target_path)
