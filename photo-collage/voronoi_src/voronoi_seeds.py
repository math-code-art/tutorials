"""
方向5: Density-guided Voronoi + Error-driven Seed Insertion
============================================================

Phase A — Initial seed placement
    1. Sample ~N seeds by rejection-sampling from the density map
       (bright pixels are more likely to be chosen as seeds).
    2. Run Lloyd/CVT iterations so seeds settle at the weighted
       centroid of their own Voronoi cell.

Phase B — Error-driven refinement  (called after tile matching)
    For each iteration:
        • Render a coarse mosaic (each cell filled with its matched
          tile's average colour).
        • Compute per-cell reconstruction error (MSE vs target).
        • Insert one new seed inside each of the top-E% error cells.
        • Re-run a few Lloyd iterations to re-balance.
        • Repeat until tile_budget is reached.
"""

import numpy as np
from scipy.spatial import Voronoi, cKDTree


# ---------------------------------------------------------------------------
# Seed sampling
# ---------------------------------------------------------------------------

def sample_seeds_from_density(
    density: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rejection-sample pixel coordinates proportional to density.

    Parameters
    ----------
    density  : H × W float32 in (0, 1]
    n_seeds  : desired number of seed points
    rng      : numpy RNG (reproducibility)

    Returns
    -------
    seeds : (n_seeds, 2) float32  — (x, y) in pixel coordinates
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W = density.shape
    flat  = density.ravel()
    prob  = flat / flat.sum()          # normalise to a probability distribution

    # Draw n_seeds pixel indices according to density probability
    indices = rng.choice(len(flat), size=n_seeds, replace=False, p=prob)
    ys, xs  = np.unravel_index(indices, (H, W))

    # Add sub-pixel jitter so seeds don't lie exactly on the grid
    xs = xs.astype(np.float32) + rng.uniform(-0.5, 0.5, size=n_seeds)
    ys = ys.astype(np.float32) + rng.uniform(-0.5, 0.5, size=n_seeds)

    seeds = np.stack([xs, ys], axis=1)
    # Clip to image bounds
    seeds[:, 0] = np.clip(seeds[:, 0], 0, W - 1)
    seeds[:, 1] = np.clip(seeds[:, 1], 0, H - 1)
    return seeds


# ---------------------------------------------------------------------------
# Weighted CVT (Lloyd iteration)
# ---------------------------------------------------------------------------

def _assign_pixels_to_seeds(seeds: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Returns label map (H, W) int32: each pixel → index of nearest seed.
    Uses cKDTree for speed; O(H·W·log N) instead of O(H·W·N).
    """
    ys, xs = np.mgrid[0:H, 0:W]               # (H,W) grids
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    tree   = cKDTree(seeds)
    _, labels = tree.query(pixels, k=1, workers=-1)
    return labels.reshape(H, W).astype(np.int32)


def lloyd_iteration(
    seeds: np.ndarray,
    density: np.ndarray,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Weighted Lloyd / CVT relaxation.

    Each seed moves to the density-weighted centroid of its Voronoi cell.
    Density acts as a mass distribution — seeds in high-density regions
    converge to smaller cells (more seeds per area).

    Parameters
    ----------
    seeds   : (N, 2) float32  initial seed positions (x, y)
    density : H × W float32
    n_iter  : number of relaxation passes (5-15 is usually enough)

    Returns
    -------
    seeds : (N, 2) float32  relaxed positions
    """
    H, W     = density.shape
    seeds    = seeds.copy()
    N        = len(seeds)

    # Pixel coordinate grids (reused every iteration)
    ys_grid, xs_grid = np.mgrid[0:H, 0:W]
    xs_flat = xs_grid.ravel().astype(np.float32)
    ys_flat = ys_grid.ravel().astype(np.float32)
    w_flat  = density.ravel().astype(np.float32)     # per-pixel weight

    for it in range(n_iter):
        tree = cKDTree(seeds)
        _, labels = tree.query(
            np.stack([xs_flat, ys_flat], axis=1), k=1, workers=-1
        )

        new_seeds = np.zeros_like(seeds)
        for i in range(N):
            mask   = labels == i
            if not np.any(mask):
                # Empty cell — keep old position
                new_seeds[i] = seeds[i]
                continue
            w_cell = w_flat[mask]
            total  = w_cell.sum()
            if total < 1e-12:
                total = 1.0
            new_seeds[i, 0] = (xs_flat[mask] * w_cell).sum() / total
            new_seeds[i, 1] = (ys_flat[mask] * w_cell).sum() / total

        # Clip to image
        new_seeds[:, 0] = np.clip(new_seeds[:, 0], 0, W - 1)
        new_seeds[:, 1] = np.clip(new_seeds[:, 1], 0, H - 1)

        delta = np.max(np.linalg.norm(new_seeds - seeds, axis=1))
        seeds = new_seeds
        if delta < 0.5:          # converged
            break

    return seeds


# ---------------------------------------------------------------------------
# Voronoi label map (fast)
# ---------------------------------------------------------------------------

def build_label_map(seeds: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Return (H, W) int32 label map: each pixel → seed index.
    """
    return _assign_pixels_to_seeds(seeds, H, W)


# ---------------------------------------------------------------------------
# Error-driven seed insertion (Phase B)
# ---------------------------------------------------------------------------

def compute_cell_errors(
    target_img: np.ndarray,
    label_map: np.ndarray,
    tile_avg_colors: np.ndarray,
    cell_tile_indices: list[int],
) -> np.ndarray:
    """
    For each Voronoi cell i, compute:

        error_i = mean_over_pixels { ||target(p) - tile_color_i||² }

    Parameters
    ----------
    target_img        : H × W × 3 float32
    label_map         : H × W int32
    tile_avg_colors   : (N_tiles, 3) float32 — average RGB of each tile
    cell_tile_indices : list of length N_seeds — which tile was chosen for cell i

    Returns
    -------
    errors : (N_seeds,) float32
    """
    N_seeds = len(cell_tile_indices)
    errors  = np.zeros(N_seeds, dtype=np.float32)

    img_f = target_img.astype(np.float32)

    for i, tile_idx in enumerate(cell_tile_indices):
        mask   = label_map == i
        if not np.any(mask):
            continue
        region = img_f[mask]                          # (K, 3)
        color  = tile_avg_colors[tile_idx]            # (3,)
        mse    = np.mean(np.sum((region - color[None])**2, axis=1))
        errors[i] = mse

    return errors


def insert_seeds_by_error(
    seeds: np.ndarray,
    errors: np.ndarray,
    label_map: np.ndarray,
    density: np.ndarray,
    error_percentile: float = 80.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Insert one new seed inside each high-error cell.

    The new seed is placed at the density-weighted centroid of the
    worst pixels inside the cell (not just the cell centroid), so it
    lands in the hardest-to-reconstruct sub-region.

    Parameters
    ----------
    seeds            : (N, 2) current seeds
    errors           : (N,)   per-cell reconstruction error
    label_map        : (H, W) int32
    density          : (H, W) float32
    error_percentile : add seeds to cells above this error percentile

    Returns
    -------
    new_seeds : (N + K, 2) with K extra seeds inserted
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W = density.shape
    threshold = np.percentile(errors, error_percentile)
    high_error_cells = np.where(errors >= threshold)[0]

    ys_grid, xs_grid = np.mgrid[0:H, 0:W]
    xs_flat = xs_grid.ravel().astype(np.float32)
    ys_flat = ys_grid.ravel().astype(np.float32)
    w_flat  = density.ravel().astype(np.float32)

    extra_seeds = []
    for cell_i in high_error_cells:
        mask = (label_map == cell_i).ravel()
        if not np.any(mask):
            continue
        # Place new seed at weighted centroid of the high-density pixels
        # in this cell → lands in the visually richest sub-region
        w_cell = w_flat[mask]
        total  = w_cell.sum()
        if total < 1e-12:
            # fallback: random position inside cell
            px = rng.choice(np.where(mask)[0])
            nx = xs_flat[px] + rng.uniform(-1, 1)
            ny = ys_flat[px] + rng.uniform(-1, 1)
        else:
            nx = (xs_flat[mask] * w_cell).sum() / total
            ny = (ys_flat[mask] * w_cell).sum() / total

        nx = float(np.clip(nx, 0, W - 1))
        ny = float(np.clip(ny, 0, H - 1))
        extra_seeds.append([nx, ny])

    if not extra_seeds:
        return seeds

    return np.vstack([seeds, np.array(extra_seeds, dtype=np.float32)])


# ---------------------------------------------------------------------------
# Convenience: full initial setup
# ---------------------------------------------------------------------------

def initialise_voronoi(
    density: np.ndarray,
    n_initial_seeds: int = 500,
    lloyd_iters: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Phase A: sample seeds + Lloyd relaxation.

    Returns
    -------
    seeds     : (N, 2) float32  relaxed seed positions
    label_map : (H, W) int32    Voronoi cell assignment
    """
    H, W = density.shape
    seeds = sample_seeds_from_density(density, n_initial_seeds, rng)
    seeds = lloyd_iteration(seeds, density, n_iter=lloyd_iters)
    label_map = build_label_map(seeds, H, W)
    print(f"[voronoi] initialised {len(seeds)} seeds, "
          f"label_map shape {label_map.shape}")
    return seeds, label_map
