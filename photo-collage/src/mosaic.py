from pathlib import Path
import os
import sys
import csv
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.search   import get_blocks, build_kdtrees, find_best_tile, color_transfer, compute_variance
from src.features import VGGEmbedder, get_block_embedding


# ---------------------------------------------------------------------------
# Tile resize
# ---------------------------------------------------------------------------

def _fit_tile(hires_tile, w, h):
    """
    Resize a hi-res tile to (w, h) using LANCZOS.
    Blocks are at most TILE_SIZE_MAX, so this is usually downsampling.
    """
    img = Image.fromarray(hires_tile)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_img(arr, path, quality=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=quality)
    print(f"[save] {path}")


def _target_steps_dir(target_path):
    original_name = os.path.splitext(os.path.basename(target_path))[0]
    steps_dir = os.path.join(config.OUTPUT_DIR, f"{original_name}_steps")
    os.makedirs(steps_dir, exist_ok=True)
    return original_name, steps_dir


def _make_quadtree_overlay(target_img, blocks):
    viz = target_img.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)
    H, W = viz.shape[:2]

    for (x, y, w, h) in blocks:
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        viz[y, x:x2] = color
        viz[y2 - 1, x:x2] = color
        viz[y:y2, x] = color
        viz[y:y2, x2 - 1] = color

    return viz


def _make_block_size_map(target_img, blocks):
    """
    Visualize block size.
    Larger blocks are lighter, smaller blocks are darker.
    """
    H, W = target_img.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)

    sizes = sorted(set(w for (_, _, w, _) in blocks))
    if len(sizes) == 1:
        size_to_value = {sizes[0]: 180}
    else:
        size_to_value = {}
        for s in sizes:
            t = (s - min(sizes)) / (max(sizes) - min(sizes) + 1e-8)
            value = int(60 + 180 * t)
            size_to_value[s] = value

    for (x, y, w, h) in blocks:
        v = size_to_value[w]
        out[y:y+h, x:x+w] = [v, v, v]

    return out


def _make_block_label_map(target_img, blocks):
    """
    Random-color visualization of quadtree blocks.
    This is for debugging the spatial partition only.
    """
    H, W = target_img.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)

    for (x, y, w, h) in blocks:
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        out[y:y+h, x:x+w] = color

    return out


def _make_mean_color_reconstruction(target_img, blocks):
    """
    Reconstruct target using only average color per quadtree block.
    This shows what the partition itself preserves before tile matching.
    """
    H, W = target_img.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)

    for (x, y, w, h) in blocks:
        block = target_img[y:y+h, x:x+w]
        mean = block.astype(np.float32).reshape(-1, 3).mean(axis=0)
        out[y:y+h, x:x+w] = mean.astype(np.uint8)

    return out


def _save_blocks_manifest(target_img, blocks, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "x", "y", "w", "h", "area", "luminance_variance"])
        for i, (x, y, w, h) in enumerate(blocks):
            block = target_img[y:y+h, x:x+w]
            writer.writerow([i, x, y, w, h, w * h, compute_variance(block)])
    print(f"[save] {path}")


def _save_blocks_with_tiles_manifest(records, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_id", "x", "y", "w", "h", "area",
            "tile_index", "block_mean_r", "block_mean_g", "block_mean_b"
        ])
        for r in records:
            writer.writerow(r)
    print(f"[save] {path}")


def _make_tile_assignment_map(target_img, blocks, chosen_indices):
    """
    Random-color map showing which tile index was assigned to each block.
    Same tile index gets same color.
    """
    H, W = target_img.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)

    rng = np.random.default_rng(123)
    unique_tiles = sorted(set(int(t) for t in chosen_indices if int(t) >= 0))
    tile_to_color = {
        t: rng.integers(30, 255, size=3, dtype=np.uint8)
        for t in unique_tiles
    }

    for (x, y, w, h), tile_idx in zip(blocks, chosen_indices):
        out[y:y+h, x:x+w] = tile_to_color.get(int(tile_idx), np.array([0, 0, 0], dtype=np.uint8))

    return out


# ---------------------------------------------------------------------------
# Main mosaic builder
# ---------------------------------------------------------------------------

def build_mosaic(target_img, target_path, tile_cache_small, hires_cache,
                 emb3, emb4):
    """
    Normal quadtree photomosaic with saved intermediate steps.
    """

    original_name, steps_dir = _target_steps_dir(target_path)

    # Save preprocessed target
    _save_img(target_img, os.path.join(steps_dir, "01_target_preprocessed.jpg"))

    # ── Step 1: adaptive quadtree split ─────────────────────────────────────
    blocks = get_blocks(target_img)

    # Save split visualizations
    split_overlay = _make_quadtree_overlay(target_img, blocks)
    _save_img(split_overlay, os.path.join(steps_dir, "02_quadtree_split_overlay.jpg"))

    # Also save the old global file for compatibility
    old_split_path = os.path.join(config.OUTPUT_DIR, "quadtree_split.jpg")
    _save_img(split_overlay, old_split_path)

    block_size_map = _make_block_size_map(target_img, blocks)
    _save_img(block_size_map, os.path.join(steps_dir, "03_quadtree_block_size_map.jpg"))

    label_map = _make_block_label_map(target_img, blocks)
    _save_img(label_map, os.path.join(steps_dir, "04_quadtree_label_map.jpg"))

    mean_recon = _make_mean_color_reconstruction(target_img, blocks)
    _save_img(mean_recon, os.path.join(steps_dir, "05_mean_color_reconstruction.jpg"))

    _save_blocks_manifest(
        target_img,
        blocks,
        os.path.join(steps_dir, "blocks_manifest.csv")
    )

    # ── Step 2: build KDTrees ───────────────────────────────────────────────
    tree3, tree4 = build_kdtrees(emb3, emb4)

    # ── Step 3: feature extractors ─────────────────────────────────────────
    embedder3 = VGGEmbedder(config.CUT3)
    embedder4 = VGGEmbedder(config.CUT4)

    # ── Step 4: canvas + bookkeeping ───────────────────────────────────────
    H, W, _ = target_img.shape
    canvas    = np.zeros((H, W, 3), dtype=np.uint8)
    use_count = np.zeros(len(hires_cache), dtype=np.int32)

    tile_means = (
        hires_cache
        .astype(np.float32)
        .reshape(len(hires_cache), -1, 3)
        .mean(axis=1)
    )

    gh = H // config.TILE_SIZE_MIN
    gw = W // config.TILE_SIZE_MIN
    chosen_grid = -np.ones((gh, gw), dtype=np.int32)

    preview_path = os.path.join(config.OUTPUT_DIR, "preview.jpg")
    chosen_indices = []
    manifest_records = []

    # ── Step 5: fill every block ───────────────────────────────────────────
    for i, (x, y, w, h) in enumerate(tqdm(blocks, desc="building mosaic")):
        block = target_img[y:y+h, x:x+w]

        block_emb3 = get_block_embedding(block, embedder3)
        block_emb4 = get_block_embedding(block, embedder4)

        gi = y // config.TILE_SIZE_MIN
        gj = x // config.TILE_SIZE_MIN

        best_idx = find_best_tile(
            block, block_emb3, block_emb4,
            tree3, tree4,
            tile_cache_small,
            tile_means,
            use_count,
            chosen_grid,
            gi, gj, gh, gw
        )

        chosen_indices.append(best_idx)

        for dy in range(h // config.TILE_SIZE_MIN):
            for dx in range(w // config.TILE_SIZE_MIN):
                _gi = gi + dy
                _gj = gj + dx
                if 0 <= _gi < gh and 0 <= _gj < gw:
                    chosen_grid[_gi, _gj] = best_idx

        chosen_tile = _fit_tile(hires_cache[best_idx], w, h)

        block_mean = block.astype(np.float32).reshape(-1, 3).mean(axis=0)
        chosen_tile = color_transfer(chosen_tile, block_mean)

        canvas[y:y+h, x:x+w] = chosen_tile

        manifest_records.append([
            i, x, y, w, h, w * h, int(best_idx),
            float(block_mean[0]), float(block_mean[1]), float(block_mean[2])
        ])

        if False:  # periodic preview saving disabled
            # Old preview path
            _save_preview(canvas, preview_path, i + 1, len(blocks))

            # Per-target preview path
            step_preview_path = os.path.join(steps_dir, f"06_preview_{i+1:06d}_of_{len(blocks):06d}.jpg")
            _save_img(canvas, step_preview_path, quality=85)

    # ── Step 6: save tile assignment debug map ─────────────────────────────
    tile_assignment_map = _make_tile_assignment_map(target_img, blocks, chosen_indices)
    _save_img(tile_assignment_map, os.path.join(steps_dir, "07_tile_assignment_map.jpg"))

    _save_blocks_with_tiles_manifest(
        manifest_records,
        os.path.join(steps_dir, "blocks_manifest_with_tiles.csv")
    )

    # ── Step 7: save final result ─────────────────────────────────────────
    final_path = os.path.join(config.OUTPUT_DIR, f"{original_name}_final.jpg")
    _save_img(canvas, final_path, quality=95)

    final_steps_path = os.path.join(steps_dir, "08_final_mosaic.jpg")
    _save_img(canvas, final_steps_path, quality=95)

    summary = {
        "target_path": target_path,
        "target_shape": list(target_img.shape),
        "n_blocks": len(blocks),
        "tile_size_min": config.TILE_SIZE_MIN,
        "tile_size_max": config.TILE_SIZE_MAX,
        "split_variance_thresh": config.SPLIT_VARIANCE_THRESH,
        "pool_tiles": config.POOL_TILES,
        "topk": config.TOPK,
        "max_use": config.MAX_USE,
        "color_transfer": config.COLOR_TRANSFER,
        "color_transfer_alpha": config.COLOR_TRANSFER_ALPHA,
        "outputs": {
            "target_preprocessed": "01_target_preprocessed.jpg",
            "quadtree_split_overlay": "02_quadtree_split_overlay.jpg",
            "quadtree_block_size_map": "03_quadtree_block_size_map.jpg",
            "quadtree_label_map": "04_quadtree_label_map.jpg",
            "mean_color_reconstruction": "05_mean_color_reconstruction.jpg",
            "tile_assignment_map": "07_tile_assignment_map.jpg",
            "final_mosaic": "08_final_mosaic.jpg",
            "blocks_manifest": "blocks_manifest.csv",
            "blocks_manifest_with_tiles": "blocks_manifest_with_tiles.csv"
        }
    }

    summary_path = os.path.join(steps_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")

    # === ORIGINAL-SOURCE 300DPI PRINT MASTER ===
    from src.dataset import get_tile_paths as _get_tile_paths

    _print_tile_paths = _get_tile_paths(max_tiles=config.POOL_TILES)

    if _print_tile_paths and os.path.isfile(_print_tile_paths[0]):
        _render_original_tile_print_master(
            target_img=target_img,
            blocks=blocks,
            chosen_indices=chosen_indices,
            tile_paths=_print_tile_paths,
            original_name=original_name,
            output_dir=config.OUTPUT_DIR,
            output_long_edge=int(getattr(config, "PRINT_LONG_EDGE", 24000)),
            dpi=int(getattr(config, "PRINT_DPI", 300)),
        )
    else:
        print("[PRINT main] Original source dataset not found.")
        print("[PRINT main] Standard Split + VGG mosaic was generated from cache.")
        print("[PRINT main] Skipping full-resolution 300 DPI PRINT output.")

    print(f"[mosaic] done! saved: {final_path}")
    print(f"[mosaic] intermediate steps saved in: {steps_dir}")
    return canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_preview(canvas, path, done, total):
    Image.fromarray(canvas).save(path, quality=85)
    print(f"[mosaic] preview saved {done}/{total} blocks → {path}")


# === ORIGINAL-SOURCE 300DPI PRINT MASTER ===

def _load_original_tile_for_print(path, out_w, out_h):
    """
    Load ORIGINAL JPG/PNG, center-crop square,
    then resize directly to the final print block.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")

        w0, h0 = im.size
        side = min(w0, h0)

        left = (w0 - side) // 2
        top = (h0 - side) // 2

        im = im.crop(
            (left, top, left + side, top + side)
        )

        im = im.resize(
            (out_w, out_h),
            Image.LANCZOS
        )

        return np.asarray(im, dtype=np.uint8)


def _render_original_tile_print_master(
    target_img,
    blocks,
    chosen_indices,
    tile_paths,
    original_name,
    output_dir,
    output_long_edge=24000,
    dpi=300,
):
    """
    Re-render the finished split/VGG mosaic at museum-print resolution.

    IMPORTANT:
    Final tiles come from ORIGINAL JPG/PNG files,
    NOT tiles_hires64 / tiles_hires128.
    """

    H, W = target_img.shape[:2]

    scale = float(output_long_edge) / float(max(H, W))

    out_w = int(round(W * scale))
    out_h = int(round(H * scale))

    sx = out_w / float(W)
    sy = out_h / float(H)

    print(
        f"[PRINT main] {W}x{H} -> "
        f"{out_w}x{out_h} @ {dpi} dpi"
    )

    # Use enlarged target only as tiny-gap fallback.
    canvas = np.asarray(
        Image.fromarray(target_img).resize(
            (out_w, out_h),
            Image.LANCZOS
        ),
        dtype=np.uint8
    ).copy()

    upscale_count = 0

    for (x, y, w, h), tile_idx in tqdm(
        zip(blocks, chosen_indices),
        total=len(blocks),
        desc="rendering ORIGINAL-source PRINT master",
    ):

        tile_idx = int(tile_idx)

        if tile_idx < 0 or tile_idx >= len(tile_paths):
            continue

        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = int(round((x + w) * sx))
        y1 = int(round((y + h) * sy))

        x1 = min(out_w, max(x0 + 1, x1))
        y1 = min(out_h, max(y0 + 1, y1))

        pw = x1 - x0
        ph = y1 - y0

        source_path = tile_paths[tile_idx]

        try:
            with Image.open(source_path) as test_im:
                sw, sh = test_im.size
                if min(sw, sh) < max(pw, ph):
                    upscale_count += 1

            tile = _load_original_tile_for_print(
                source_path,
                pw,
                ph
            )

        except Exception as e:
            print(
                f"[PRINT warning] tile {tile_idx} failed: {e}"
            )
            continue

        # Keep same color-transfer behavior as normal mosaic.
        block = target_img[y:y+h, x:x+w]

        block_mean = (
            block.astype(np.float32)
            .reshape(-1, 3)
            .mean(axis=0)
        )

        tile = color_transfer(
            tile,
            block_mean
        )

        canvas[y0:y1, x0:x1] = tile

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = Path(output_dir) / (
        f"{original_name}_PRINT_"
        f"{out_w}x{out_h}_{dpi}dpi.png"
    )

    Image.fromarray(canvas).save(
        out_path,
        format="PNG",
        dpi=(dpi, dpi),
        compress_level=1,
    )

    print("[PRINT main] saved:", out_path)
    print(
        "[PRINT main] source tiles requiring upscaling:",
        f"{upscale_count}/{len(blocks)}"
    )

    return str(out_path)
