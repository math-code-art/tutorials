from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# BASIC PHOTOMOSAIC — VERSION 1
# ============================================================
#
# Pure baseline:
#
#   Uniform Grid
#       +
#   Mean RGB Matching
#
# NO VGG
# NO Voronoi
# NO adaptive split
# NO neural network
# NO color transfer
# NO periodic previews
#
# ============================================================


ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

import config

from src.dataset import (
    get_tile_paths,
    get_tile_cache,
    get_hires_tile_cache,
    load_tile_rgb,
)


# ============================================================
# PARAMETERS
# ============================================================

# Working target resolution.
TARGET_LONG_EDGE = 2400

# Smaller than the previous 32px baseline.
# Gives a finer but still clearly uniform grid.
CELL_SIZE = 40
# Exhibition-quality master.
PRINT_LONG_EDGE = 24000
PRINT_DPI = 300

# Controls RAM usage during RGB matching only.
MATCH_BATCH = 64

# Light appearance correction for the Basic Mosaic.
# Tile selection remains mean-RGB based; this only corrects
# the rendered tile toward the target cell color.
COLOR_TRANSFER_ALPHA = 0.72
SATURATION_BOOST = 1.12

# ============================================================
# TILE DIVERSITY / REUSE CONTROL
# ============================================================

# For each grid cell, first consider the closest source
# images in mean-RGB space.
TOP_K_CANDIDATES = 512

# Soft penalty for repeated use.
# Useful if MAX_TILE_REUSE is later changed above 1.
REUSE_PENALTY = 20.0

# Extra penalty for immediate left/up repetition.
NEIGHBOR_REPEAT_PENALTY = 100.0

# With 81k source images and only a few thousand grid cells,
# we can make every selected source image unique.
MAX_TILE_REUSE = 1



TARGET_DIR = Path(config.TARGET_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
CACHE_DIR = Path(config.CACHE_DIR)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ============================================================
# IMAGE HELPERS
# ============================================================

def center_crop_square(im: Image.Image) -> Image.Image:

    w, h = im.size

    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2

    return im.crop(
        (
            left,
            top,
            left + side,
            top + side,
        )
    )


def load_original_tile(
    path: str,
    out_w: int,
    out_h: int,
) -> tuple[Image.Image, bool]:

    with Image.open(path) as im:

        im = im.convert("RGB")

        im = center_crop_square(im)

        src_w, src_h = im.size

        was_upscaled = (
            out_w > src_w
            or out_h > src_h
        )

        im = im.resize(
            (out_w, out_h),
            Image.Resampling.LANCZOS,
        )

        return im, was_upscaled


# ============================================================
# TARGET PREPROCESSING
# ============================================================

def preprocess_target(
    path: Path,
) -> np.ndarray:

    with Image.open(path) as im:

        im = im.convert("RGB")

        w, h = im.size

        scale = (
            TARGET_LONG_EDGE
            / float(max(w, h))
        )

        new_w = max(
            CELL_SIZE,
            int(round(w * scale)),
        )

        new_h = max(
            CELL_SIZE,
            int(round(h * scale)),
        )

        im = im.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

        # ----------------------------------------------------
        # Crop slightly so both dimensions are exact multiples
        # of CELL_SIZE.
        # ----------------------------------------------------

        crop_w = (
            new_w // CELL_SIZE
        ) * CELL_SIZE

        crop_h = (
            new_h // CELL_SIZE
        ) * CELL_SIZE

        left = (
            new_w - crop_w
        ) // 2

        top = (
            new_h - crop_h
        ) // 2

        im = im.crop(
            (
                left,
                top,
                left + crop_w,
                top + crop_h,
            )
        )

        return np.asarray(
            im,
            dtype=np.uint8,
        )


# ============================================================
# CRITICAL CACHE ORDER VALIDATION
# ============================================================

def validate_cache_order(
    tile_paths: list[str],
    cache: np.ndarray,
    tile_size: int,
    label: str,
) -> bool:

    """
    CRITICAL SAFETY CHECK.

    Cache index i MUST represent tile_paths[i].

    If this is false, mean-RGB matching may select one image,
    while rendering opens a completely different image.

    That is exactly the kind of index mismatch that produces
    a visually scrambled / random mosaic.
    """

    n = len(tile_paths)

    originals_available = (
        n > 0
        and Path(tile_paths[0]).is_file()
    )

    if not originals_available:
        valid = (
            len(cache) == n
            and cache.ndim == 4
        )

        if valid:
            print(
                f"[check] {label} cache accepted: "
                "original dataset not installed; using precomputed cache"
            )
        else:
            print(
                f"[check] {label} cache rejected: "
                "cache shape/count mismatch"
            )

        return valid

    sample_ids = np.unique(
        np.linspace(
            0,
            n - 1,
            9,
            dtype=int,
        )
    )

    errors = []

    print(
        f"[check] validating {label} "
        f"tile index ordering..."
    )

    for idx in sample_ids:

        try:

            original = load_tile_rgb(
                tile_paths[int(idx)],
                tile_size,
            )

            cached = np.asarray(
                cache[int(idx)]
            )

            if (
                original.shape
                != cached.shape
            ):

                print(
                    f"[check] shape mismatch "
                    f"at tile {idx}"
                )

                return False

            err = np.abs(
                original.astype(np.int16)
                - cached.astype(np.int16)
            ).mean()

            errors.append(
                float(err)
            )

        except Exception as e:

            print(
                f"[check] skipped tile "
                f"{idx}: {e}"
            )

    if not errors:

        print(
            "[check] unable to validate cache"
        )

        return False

    mean_error = float(
        np.mean(errors)
    )

    print(
        f"[check] {label} cache/order "
        f"mean pixel error = "
        f"{mean_error:.6f}"
    )

    valid = mean_error < 1.0

    if valid:

        print(
            f"[check] {label} ordering: OK"
        )

    else:

        print()
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            f"[WARNING] {label} CACHE ORDER "
            f"DOES NOT MATCH TILE PATHS"
        )
        print(
            "This cache will NOT be used."
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print()

    return valid


# ============================================================
# MEAN RGB CACHE
# ============================================================

def get_basic_rgb_means(
    tile_paths: list[str],
    small_cache: np.ndarray,
    small_cache_valid: bool,
) -> np.ndarray:

    """
    Build one RGB vector per source image.

    The path-order file is saved alongside the mean cache.
    A cached result is reused ONLY when its source-path order
    exactly matches the current tile_paths order.
    """

    n = len(tile_paths)

    means_path = (
        CACHE_DIR
        / f"basic_mean_rgb_N{n}.npy"
    )

    order_path = (
        CACHE_DIR
        / f"basic_mean_rgb_paths_N{n}.txt"
    )

    current_order = "\n".join(
        Path(p).name
        for p in tile_paths
    )

    # --------------------------------------------------------
    # Reuse a previously verified BASIC cache.
    # --------------------------------------------------------

    if (
        means_path.exists()
        and order_path.exists()
    ):

        saved_order = (
            order_path
            .read_text()
            .rstrip("\n")
        )

        if saved_order == current_order:

            means = np.load(
                means_path
            )

            if means.shape == (n, 3):

                print(
                    f"[basic cache] loaded "
                    f"{means_path.name}"
                )

                return means.astype(
                    np.float32,
                    copy=False,
                )

    means = np.zeros(
        (n, 3),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # FAST PATH:
    # Reuse RGB thumbnail pixels from the existing dataset
    # cache ONLY after confirming the index order is correct.
    #
    # This is NOT using VGG.
    # --------------------------------------------------------

    if small_cache_valid:

        print(
            "[basic cache] computing mean RGB "
            "from verified RGB thumbnail cache"
        )

        chunk = 2048

        for start in range(
            0,
            n,
            chunk,
        ):

            stop = min(
                start + chunk,
                n,
            )

            block = np.asarray(
                small_cache[
                    start:stop
                ],
                dtype=np.float32,
            )

            means[
                start:stop
            ] = block.mean(
                axis=(1, 2)
            )

            if (
                stop % 10000 == 0
                or stop == n
            ):

                print(
                    f"  means: "
                    f"{stop}/{n}"
                )

    # --------------------------------------------------------
    # SAFE FALLBACK:
    # If old cache ordering is wrong, compute mean RGB
    # directly from original images.
    #
    # Slower, but guaranteed correct.
    # --------------------------------------------------------

    else:

        print(
            "[basic cache] building directly "
            "from ORIGINAL source images"
        )

        print(
            "[basic cache] this may take time "
            "once, but prevents index mismatch"
        )

        for i, path in enumerate(
            tile_paths
        ):

            try:

                tile = load_tile_rgb(
                    path,
                    32,
                ).astype(
                    np.float32
                )

                means[i] = tile.mean(
                    axis=(0, 1)
                )

            except Exception:

                means[i] = np.array(
                    [127.5, 127.5, 127.5],
                    dtype=np.float32,
                )

            if (
                (i + 1) % 5000 == 0
                or i + 1 == n
            ):

                print(
                    f"  means: "
                    f"{i + 1}/{n}"
                )

    np.save(
        means_path,
        means,
    )

    order_path.write_text(
        current_order + "\n"
    )

    print(
        f"[basic cache] saved "
        f"{means_path.name}"
    )

    return means


# ============================================================
# UNIFORM GRID CELL MEANS
# ============================================================

def get_target_cell_means(
    target: np.ndarray,
) -> np.ndarray:

    h, w = target.shape[:2]

    rows = h // CELL_SIZE
    cols = w // CELL_SIZE

    reshaped = (
        target
        .astype(np.float32)
        .reshape(
            rows,
            CELL_SIZE,
            cols,
            CELL_SIZE,
            3,
        )
    )

    means = reshaped.mean(
        axis=(1, 3)
    )

    return means


# ============================================================
# PURE RGB NEAREST-NEIGHBOR MATCHING
# ============================================================

def match_mean_rgb(
    target_means: np.ndarray,
    tile_means: np.ndarray,
) -> np.ndarray:

    """
    BASIC baseline matching:

        mean-RGB distance
        + reuse control

    No VGG.
    No texture feature.
    No Voronoi.
    No color transfer.

    By default MAX_TILE_REUSE = 1, so each source image
    can appear at most once in the final mosaic.
    """

    rows, cols = target_means.shape[:2]

    tiles = tile_means.astype(
        np.float32,
        copy=False,
    )

    n_tiles = len(tiles)
    n_cells = rows * cols

    chosen = np.full(
        (rows, cols),
        -1,
        dtype=np.int32,
    )

    use_count = np.zeros(
        n_tiles,
        dtype=np.int32,
    )

    tile_norm = np.sum(
        tiles * tiles,
        axis=1,
    )[None, :]

    top_k = min(
        TOP_K_CANDIDATES,
        n_tiles,
    )

    print()
    print("[matching] BASIC mean-RGB matching")
    print(f"[matching] cells: {n_cells}")
    print(f"[matching] source images: {n_tiles}")
    print(f"[matching] top-K candidates: {top_k}")
    print(f"[matching] reuse penalty: {REUSE_PENALTY}")
    print(
        f"[matching] maximum uses per source image: "
        f"{MAX_TILE_REUSE}"
    )

    # ========================================================
    # Row-by-row:
    # keeps memory moderate while allowing reuse counts
    # to update after every assignment.
    # ========================================================

    for r in range(rows):

        q = target_means[r].astype(
            np.float32
        )

        q_norm = np.sum(
            q * q,
            axis=1,
        )[:, None]

        d2 = (
            q_norm
            + tile_norm
            - 2.0 * (
                q @ tiles.T
            )
        )

        d2 = np.maximum(
            d2,
            0.0,
        )

        candidate_ids = np.argpartition(
            d2,
            kth=top_k - 1,
            axis=1,
        )[:, :top_k]

        for c in range(cols):

            candidates = (
                candidate_ids[c]
            )

            # ------------------------------------------------
            # Respect hard global reuse cap
            # ------------------------------------------------

            available_mask = (
                use_count[candidates]
                < MAX_TILE_REUSE
            )

            if np.any(
                available_mask
            ):

                candidates = (
                    candidates[
                        available_mask
                    ]
                )

            else:

                # Rare fallback:
                # if all top-K candidates are already used,
                # search all still-available source images.

                globally_available = np.flatnonzero(
                    use_count
                    < MAX_TILE_REUSE
                )

                if len(
                    globally_available
                ) > 0:

                    candidates = (
                        globally_available
                    )

                else:

                    # Only possible if number of target cells
                    # exceeds total allowed source capacity.
                    candidates = (
                        candidate_ids[c]
                    )

            color_distance = np.sqrt(
                d2[
                    c,
                    candidates,
                ]
            )

            scores = (
                color_distance
                + REUSE_PENALTY
                * use_count[candidates]
            )

            # ------------------------------------------------
            # Local spatial repetition control
            # ------------------------------------------------

            if c > 0:

                left_idx = int(
                    chosen[
                        r,
                        c - 1
                    ]
                )

                scores = (
                    scores
                    + (
                        candidates
                        == left_idx
                    )
                    * NEIGHBOR_REPEAT_PENALTY
                )

            if r > 0:

                upper_idx = int(
                    chosen[
                        r - 1,
                        c
                    ]
                )

                scores = (
                    scores
                    + (
                        candidates
                        == upper_idx
                    )
                    * NEIGHBOR_REPEAT_PENALTY
                )

            best_local = int(
                np.argmin(
                    scores
                )
            )

            winner = int(
                candidates[
                    best_local
                ]
            )

            chosen[
                r,
                c
            ] = winner

            use_count[
                winner
            ] += 1


        if (
            (r + 1) % 10 == 0
            or r + 1 == rows
        ):

            done = min(
                (r + 1) * cols,
                n_cells,
            )

            print(
                f"  matched "
                f"{done}/{n_cells}"
            )


    # ========================================================
    # Diversity diagnostics
    # ========================================================

    used_counts = use_count[
        use_count > 0
    ]

    unique_count = len(
        used_counts
    )

    maximum_reuse = (
        int(
            used_counts.max()
        )
        if unique_count
        else 0
    )

    print()
    print(
        f"[reuse] total grid cells: "
        f"{n_cells}"
    )

    print(
        f"[reuse] unique source images: "
        f"{unique_count}"
    )

    print(
        f"[reuse] maximum actual reuse: "
        f"{maximum_reuse}"
    )

    if maximum_reuse <= MAX_TILE_REUSE:
        print(
            "[reuse] reuse constraint: OK"
        )

    return chosen



# ============================================================
# PAPER PROCESS FIGURES
# ============================================================

def save_rgb(
    arr: np.ndarray,
    path: Path,
    quality: int = 95,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        np.clip(
            arr,
            0,
            255,
        ).astype(np.uint8)
    ).save(
        path,
        quality=quality,
    )


def grid_overlay(
    target: np.ndarray,
) -> np.ndarray:

    im = Image.fromarray(
        target.copy()
    )

    draw = ImageDraw.Draw(im)

    w, h = im.size

    line_color = (
        255,
        255,
        255,
    )

    for x in range(
        0,
        w + 1,
        CELL_SIZE,
    ):

        draw.line(
            [(x, 0), (x, h)],
            fill=line_color,
            width=1,
        )

    for y in range(
        0,
        h + 1,
        CELL_SIZE,
    ):

        draw.line(
            [(0, y), (w, y)],
            fill=line_color,
            width=1,
        )

    return np.asarray(im)


def expand_cell_values(
    cell_rgb: np.ndarray,
) -> np.ndarray:

    return np.repeat(
        np.repeat(
            cell_rgb,
            CELL_SIZE,
            axis=0,
        ),
        CELL_SIZE,
        axis=1,
    ).astype(
        np.uint8
    )


def rgb_error_map(
    errors: np.ndarray,
) -> np.ndarray:

    max_distance = math.sqrt(
        3.0 * 255.0 * 255.0
    )

    normalized = np.clip(
        errors
        / max_distance,
        0.0,
        1.0,
    )

    values = np.round(
        normalized * 255
    ).astype(
        np.uint8
    )

    expanded = np.repeat(
        np.repeat(
            values,
            CELL_SIZE,
            axis=0,
        ),
        CELL_SIZE,
        axis=1,
    )

    return np.stack(
        [
            expanded,
            expanded,
            expanded,
        ],
        axis=2,
    )


def apply_basic_color_adjustment(
    tile_arr: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    """
    Light post-match color correction for the Basic Mosaic.

    The selected source tile is preserved, but its mean color is shifted
    part-way toward the target cell mean. A small saturation adjustment
    keeps the target palette visually clear.
    """
    tile = tile_arr.astype(np.float32)

    src_mean = tile.mean(axis=(0, 1), keepdims=True)
    tgt_mean = np.asarray(
        target_rgb,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    adjusted = (
        tile
        + COLOR_TRANSFER_ALPHA
        * (tgt_mean - src_mean)
    )

    adjusted = np.clip(
        adjusted,
        0.0,
        255.0,
    )

    if SATURATION_BOOST != 1.0:
        gray = adjusted.mean(
            axis=2,
            keepdims=True,
        )
        adjusted = (
            gray
            + SATURATION_BOOST
            * (adjusted - gray)
        )

    return np.clip(
        adjusted,
        0.0,
        255.0,
    ).astype(np.uint8)

# ============================================================
# WORKING-RESOLUTION MOSAIC
# ============================================================

def render_working_mosaic(
    tile_paths: list[str],
    hires_cache: np.ndarray,
    hires_cache_valid: bool,
    chosen: np.ndarray,
    target_means: np.ndarray,
) -> np.ndarray:

    rows, cols = chosen.shape

    out_h = rows * CELL_SIZE
    out_w = cols * CELL_SIZE

    canvas = np.zeros(
        (
            out_h,
            out_w,
            3,
        ),
        dtype=np.uint8,
    )

    selected_cache = {}

    unique_indices = np.unique(
        chosen
    )

    print(
        f"[render] working mosaic: "
        f"{len(unique_indices)} unique tiles"
    )

    # --------------------------------------------------------
    # Load each selected source once.
    # --------------------------------------------------------

    for idx in unique_indices:

        idx = int(idx)

        if hires_cache_valid:

            tile_arr = np.asarray(
                hires_cache[idx]
            )

            tile = Image.fromarray(
                tile_arr
            ).resize(
                (
                    CELL_SIZE,
                    CELL_SIZE,
                ),
                Image.Resampling.LANCZOS,
            )

        else:

            tile = Image.fromarray(
                load_tile_rgb(
                    tile_paths[idx],
                    CELL_SIZE,
                )
            )

        selected_cache[idx] = (
            np.asarray(
                tile,
                dtype=np.uint8,
            )
        )

    # --------------------------------------------------------
    # Uniform square placement.
    # --------------------------------------------------------

    for r in range(rows):

        y0 = r * CELL_SIZE

        for c in range(cols):

            x0 = c * CELL_SIZE

            idx = int(
                chosen[r, c]
            )

            adjusted = apply_basic_color_adjustment(
                selected_cache[idx],
                target_means[r, c],
            )

            canvas[
                y0:y0 + CELL_SIZE,
                x0:x0 + CELL_SIZE,
            ] = adjusted

    return canvas


# ============================================================
# HIGH-RES ORIGINAL-SOURCE PRINT
# ============================================================

def get_print_dimensions(
    work_w: int,
    work_h: int,
) -> tuple[int, int]:

    scale = (
        PRINT_LONG_EDGE
        / float(
            max(
                work_w,
                work_h,
            )
        )
    )

    out_w = int(
        round(
            work_w * scale
        )
    )

    out_h = int(
        round(
            work_h * scale
        )
    )

    return out_w, out_h


def render_print(
    tile_paths: list[str],
    chosen: np.ndarray,
    target_means: np.ndarray,
    work_w: int,
    work_h: int,
    output_path: Path,
) -> int:

    out_w, out_h = (
        get_print_dimensions(
            work_w,
            work_h,
        )
    )

    rows, cols = chosen.shape

    print(
        f"[PRINT] "
        f"{work_w}x{work_h} "
        f"-> "
        f"{out_w}x{out_h} "
        f"@ {PRINT_DPI} dpi"
    )

    canvas = Image.new(
        "RGB",
        (
            out_w,
            out_h,
        ),
    )

    upscale_count = 0

    for r in range(rows):

        y0 = int(
            round(
                r
                * out_h
                / rows
            )
        )

        y1 = int(
            round(
                (r + 1)
                * out_h
                / rows
            )
        )

        for c in range(cols):

            x0 = int(
                round(
                    c
                    * out_w
                    / cols
                )
            )

            x1 = int(
                round(
                    (c + 1)
                    * out_w
                    / cols
                )
            )

            cell_w = max(
                1,
                x1 - x0,
            )

            cell_h = max(
                1,
                y1 - y0,
            )

            idx = int(
                chosen[r, c]
            )

            tile, was_upscaled = (
                load_original_tile(
                    tile_paths[idx],
                    cell_w,
                    cell_h,
                )
            )

            upscale_count += int(
                was_upscaled
            )

            tile_arr = apply_basic_color_adjustment(
                np.asarray(tile, dtype=np.uint8),
                target_means[r, c],
            )
            tile = Image.fromarray(tile_arr)

            canvas.paste(
                tile,
                (
                    x0,
                    y0,
                ),
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        format="PNG",
        dpi=(
            PRINT_DPI,
            PRINT_DPI,
        ),
    )

    return upscale_count


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    path: Path,
    chosen: np.ndarray,
    tile_paths: list[str],
    target_means: np.ndarray,
    tile_means: np.ndarray,
) -> None:

    rows, cols = chosen.shape

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "row",
                "col",
                "x",
                "y",
                "tile_index",
                "source_path",
                "target_mean_r",
                "target_mean_g",
                "target_mean_b",
                "source_mean_r",
                "source_mean_g",
                "source_mean_b",
                "rgb_distance",
            ]
        )

        for r in range(rows):

            for c in range(cols):

                idx = int(
                    chosen[r, c]
                )

                target_rgb = (
                    target_means[r, c]
                )

                source_rgb = (
                    tile_means[idx]
                )

                error = float(
                    np.linalg.norm(
                        target_rgb
                        - source_rgb
                    )
                )

                writer.writerow(
                    [
                        r,
                        c,
                        c * CELL_SIZE,
                        r * CELL_SIZE,
                        idx,
                        tile_paths[idx],
                        float(
                            target_rgb[0]
                        ),
                        float(
                            target_rgb[1]
                        ),
                        float(
                            target_rgb[2]
                        ),
                        float(
                            source_rgb[0]
                        ),
                        float(
                            source_rgb[1]
                        ),
                        float(
                            source_rgb[2]
                        ),
                        error,
                    ]
                )


# ============================================================
# PROCESS ONE TARGET
# ============================================================

def process_target(
    target_path: Path,
    tile_paths: list[str],
    tile_means: np.ndarray,
    hires_cache: np.ndarray,
    hires_cache_valid: bool,
) -> None:

    name = target_path.stem

    steps_dir = (
        OUTPUT_DIR
        / f"{name}_basic_mosaic_steps"
    )

    steps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove obsolete process figures from earlier runs.
    # For the paper we keep only:
    #   02_uniform_grid.jpg
    #   03_target_cell_means.jpg
    for old_name in (
        "01_target_preprocessed.jpg",
        "04_matched_tile_means.jpg",
        "05_rgb_matching_error.jpg",
    ):
        old_path = steps_dir / old_name
        if old_path.exists():
            old_path.unlink()

    print()
    print("=" * 72)
    print(
        f"BASIC MOSAIC: "
        f"{target_path.name}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Step 1: target
    # --------------------------------------------------------

    target = preprocess_target(
        target_path
    )

    h, w = target.shape[:2]

    rows = h // CELL_SIZE
    cols = w // CELL_SIZE

    n_cells = rows * cols

    print(
        f"[target] "
        f"{w}x{h}"
    )

    print(
        f"[grid] "
        f"{cols} x {rows}"
    )

    print(
        f"[grid] "
        f"{n_cells} uniform cells"
    )


    # --------------------------------------------------------
    # Step 2: uniform-grid figure
    # --------------------------------------------------------

    save_rgb(
        grid_overlay(target),
        steps_dir
        / "02_uniform_grid.jpg",
    )

    # --------------------------------------------------------
    # Step 3: target reduced to one RGB vector per cell
    # --------------------------------------------------------

    target_means = (
        get_target_cell_means(
            target
        )
    )

    target_mean_image = (
        expand_cell_values(
            target_means
        )
    )

    save_rgb(
        target_mean_image,
        steps_dir
        / "03_target_cell_means.jpg",
    )

    # --------------------------------------------------------
    # Step 4: pure mean-RGB matching
    # --------------------------------------------------------

    chosen = match_mean_rgb(
        target_means,
        tile_means,
    )

    selected_means = (
        tile_means[
            chosen.reshape(-1)
        ]
        .reshape(
            rows,
            cols,
            3,
        )
    )

    matched_mean_image = (
        expand_cell_values(
            selected_means
        )
    )


    # --------------------------------------------------------
    # VERY IMPORTANT DIAGNOSTIC
    #
    # 03 and 04 should look structurally similar.
    # If they do not, matching is wrong.
    # --------------------------------------------------------

    errors = np.linalg.norm(
        target_means
        - selected_means,
        axis=2,
    )


    print(
        f"[RGB error] mean = "
        f"{errors.mean():.3f}"
    )

    print(
        f"[RGB error] median = "
        f"{np.median(errors):.3f}"
    )

    print(
        f"[RGB error] max = "
        f"{errors.max():.3f}"
    )

    # --------------------------------------------------------
    # Final working mosaic
    # --------------------------------------------------------

    final = render_working_mosaic(
        tile_paths,
        hires_cache,
        hires_cache_valid,
        chosen,
        target_means,
    )

    final_path = (
        OUTPUT_DIR
        / f"{name}_basic_mosaic_final.jpg"
    )

    save_rgb(
        final,
        final_path,
        quality=96,
    )

    print(
        f"[save] {final_path}"
    )

    # --------------------------------------------------------
    # Reproducibility records
    # --------------------------------------------------------

    save_manifest(
        steps_dir
        / "assignments.csv",
        chosen,
        tile_paths,
        target_means,
        tile_means,
    )

    unique_tiles = len(
        np.unique(chosen)
    )

    # --------------------------------------------------------
    # PRINT MASTER
    # --------------------------------------------------------

    original_sources_available = (
        bool(tile_paths)
        and Path(tile_paths[0]).is_file()
    )

    if original_sources_available:
        print_w, print_h = (
            get_print_dimensions(
                w,
                h,
            )
        )

        print_path = (
            OUTPUT_DIR
            / (
                f"{name}_basic_mosaic_"
                f"PRINT_"
                f"{print_w}x{print_h}_"
                f"{PRINT_DPI}dpi.png"
            )
        )

        upscale_count = render_print(
            tile_paths,
            chosen,
            target_means,
            w,
            h,
            print_path,
        )

        print(
            f"[PRINT saved] "
            f"{print_path}"
        )

    else:
        print_w, print_h = get_print_dimensions(
            w,
            h,
        )
        upscale_count = 0

        print()
        print("[PRINT] Original source dataset not found.")
        print("[PRINT] Standard Basic mosaic was generated from cache.")
        print("[PRINT] Skipping full-resolution 300 DPI PRINT output.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        "BASIC PHOTOMOSAIC BASELINE\n"
        "\n"
        "Method:\n"
        "Uniform Grid + Mean RGB Matching\n"
        "\n"
        f"target={target_path.name}\n"
        f"working_size={w}x{h}\n"
        f"cell_size={CELL_SIZE}\n"
        f"grid={cols}x{rows}\n"
        f"n_cells={n_cells}\n"
        f"n_source_images={len(tile_paths)}\n"
        f"n_unique_selected_tiles="
        f"{unique_tiles}\n"
        f"mean_rgb_error="
        f"{errors.mean():.6f}\n"
        f"median_rgb_error="
        f"{np.median(errors):.6f}\n"
        f"max_rgb_error="
        f"{errors.max():.6f}\n"
        f"print_size="
        f"{print_w}x{print_h}\n"
        f"print_dpi={PRINT_DPI}\n"
        f"source_cells_requiring_upscale="
        f"{upscale_count}/{n_cells}\n"
    )

    (
        steps_dir
        / "summary.txt"
    ).write_text(
        summary
    )

    print(
        f"[done] {name}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Use EXACTLY the same source ordering as the rest of
    # the project.
    #
    # Do NOT sort these paths independently.
    # --------------------------------------------------------

    max_tiles = getattr(
        config,
        "POOL_TILES",
        None,
    )

    tile_paths = get_tile_paths(
        max_tiles
    )

    n_tiles = len(
        tile_paths
    )

    print()
    print(
        "============================================"
    )
    print(
        "BASIC PHOTOMOSAIC — VERSION 1"
    )
    print(
        "Uniform Grid + Mean RGB Matching"
    )
    print(
        "NO VGG | NO Voronoi | NO color transfer"
    )
    print(
        "============================================"
    )

    print(
        f"[dataset] source tiles: "
        f"{n_tiles}"
    )

    print(
        f"[settings] cell size: "
        f"{CELL_SIZE}px"
    )

    print(
        f"[settings] PRINT long edge: "
        f"{PRINT_LONG_EDGE}px"
    )

    # --------------------------------------------------------
    # Load the existing RGB tile caches.
    # They are only used if their index ordering is verified.
    # --------------------------------------------------------

    small_size = int(
        getattr(
            config,
            "TILE_SIZE_MIN",
            16,
        )
    )

    small_cache = get_tile_cache(
        tile_paths,
        small_size,
    )

    small_valid = (
        validate_cache_order(
            tile_paths,
            small_cache,
            small_size,
            "small",
        )
    )

    tile_means = (
        get_basic_rgb_means(
            tile_paths,
            small_cache,
            small_valid,
        )
    )

    # --------------------------------------------------------
    # Working-resolution rendering cache.
    # Again, validate its ordering before use.
    # --------------------------------------------------------

    hires_cache = (
        get_hires_tile_cache(
            tile_paths
        )
    )

    hires_size = int(
        hires_cache.shape[1]
    )

    hires_valid = (
        validate_cache_order(
            tile_paths,
            hires_cache,
            hires_size,
            "hires",
        )
    )

    # --------------------------------------------------------
    # Match mean RGB against the SAME tile representation used
    # for standard rendering.
    # --------------------------------------------------------

    print("[basic] computing mean RGB from rendering cache")

    tile_means = np.empty(
        (len(hires_cache), 3),
        dtype=np.float32,
    )

    mean_batch = 512

    for start in range(
        0,
        len(hires_cache),
        mean_batch,
    ):
        end = min(
            start + mean_batch,
            len(hires_cache),
        )

        batch = np.asarray(
            hires_cache[start:end],
            dtype=np.float32,
        )

        tile_means[start:end] = batch.mean(
            axis=(1, 2),
        )

    print(
        f"[basic] rendering-cache RGB means ready: "
        f"{tile_means.shape}"
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target_paths = sorted(
        p
        for p in TARGET_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix in IMAGE_EXTS
        )
    )

    if not target_paths:

        raise SystemExit(
            f"No target images found in "
            f"{TARGET_DIR}"
        )

    print(
        f"[targets] "
        f"{len(target_paths)}"
    )

    for target_path in target_paths:

        process_target(
            target_path,
            tile_paths,
            tile_means,
            hires_cache,
            hires_valid,
        )


if __name__ == "__main__":
    main()
