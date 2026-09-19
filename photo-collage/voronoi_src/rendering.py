"""
方向7: Polygon Tile Warping + Canvas Rendering
===============================================

Instead of pasting a rectangular tile into a square block, we:

1. For each Voronoi cell, extract its polygon boundary.
2. Resize the best-matching tile to the cell's bounding box.
3. Rasterise the polygon mask and apply it so only the tile pixels
   inside the polygon are painted.
4. Draw a 1–2 px grout line (dark/black) around each polygon boundary,
   simulating the lead came in a real glass mosaic.

Result
------
  • Each tile appears as an organic, irregular shape rather than a grid.
  • Adjacent tiles have visible separation (grout), making the individual
    pieces readable.
  • The mosaic has the handcrafted feel of a stone or glass mosaic.

Key design choice: no perspective warp
--------------------------------------
Warping the tile's internal content to fit an arbitrary polygon would
distort the artwork beyond recognition.  Instead we do a shape-mask
approach: tile content is undistorted; only its silhouette is irregular.
This keeps each tile visually readable while still giving organic shape.
"""

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Extract per-cell polygon
# ---------------------------------------------------------------------------

def extract_cell_polygons(
    label_map: np.ndarray,
    n_cells: int,
) -> list[np.ndarray | None]:
    """
    For each cell i, find the outer contour of all pixels labelled i
    and return it as an (M, 2) int32 array of (x, y) coordinates.

    Parameters
    ----------
    label_map : H × W int32
    n_cells   : number of cells

    Returns
    -------
    polygons : list of length n_cells.
               polygons[i] is (M_i, 2) int32, or None if cell is empty.
    """
    H, W     = label_map.shape
    polygons = []

    for i in range(n_cells):
        mask = (label_map == i).astype(np.uint8) * 255

        # Find external contour
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            polygons.append(None)
            continue

        # Take the largest contour (handles tiny disconnected fragments)
        c = max(contours, key=cv2.contourArea)

        # Optionally simplify: Douglas-Peucker with epsilon = 1 px
        c = cv2.approxPolyDP(c, epsilon=1.0, closed=True)

        # (M, 1, 2) → (M, 2)
        polygons.append(c.reshape(-1, 2))

    return polygons


# ---------------------------------------------------------------------------
# Fit tile to bounding box (sharp downscale or 1:1)
# ---------------------------------------------------------------------------

def _fit_tile_to_bbox(hires_tile: np.ndarray, bbox_w: int, bbox_h: int) -> np.ndarray:
    """
    Resize hires_tile to (bbox_w, bbox_h) using LANCZOS.
    Since tiles are stored at TILE_SIZE_MAX and bboxes are usually smaller,
    this is almost always a downscale (never blurry).
    """
    img = Image.fromarray(hires_tile)
    img = img.resize((bbox_w, bbox_h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Colour transfer (carried over from your existing search.py)
# ---------------------------------------------------------------------------

def color_transfer(tile: np.ndarray, target_mean: np.ndarray,
                   alpha: float = 0.55) -> np.ndarray:
    """
    Shift tile mean colour toward target_mean by alpha.
    alpha=0 → no transfer;  alpha=1 → full transfer.
    """
    tile_f    = tile.astype(np.float32)
    tile_mean = tile_f.reshape(-1, 3).mean(axis=0)
    shift     = (target_mean - tile_mean) * alpha
    return np.clip(tile_f + shift, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Render one cell
# ---------------------------------------------------------------------------

def render_cell(
    canvas: np.ndarray,
    hires_tile: np.ndarray,
    polygon: np.ndarray | None,
    label_map: np.ndarray,
    cell_idx: int,
    target_region_mean: np.ndarray,
    color_transfer_alpha: float = 0.55,
) -> None:
    """
    Paint one Voronoi cell onto the canvas in-place.

    Parameters
    ----------
    canvas              : H × W × 3 uint8  (modified in-place)
    hires_tile          : T × T × 3 uint8  (hi-res tile image)
    polygon             : (M, 2) int32 or None
    label_map           : H × W int32
    cell_idx            : which cell to render
    target_region_mean  : (3,) float32 — mean RGB of cell in target image
    color_transfer_alpha: see color_transfer()
    """
    H, W = canvas.shape[:2]

    # Fall back to rectangular paste if polygon is unavailable
    cell_mask = (label_map == cell_idx)
    if not np.any(cell_mask):
        return

    # Bounding box of the cell mask
    rows = np.where(np.any(cell_mask, axis=1))[0]
    cols = np.where(np.any(cell_mask, axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    bbox_h = y1 - y0
    bbox_w = x1 - x0

    if bbox_h < 1 or bbox_w < 1:
        return

    # Resize tile to bounding box
    tile_resized = _fit_tile_to_bbox(hires_tile, bbox_w, bbox_h)
    tile_resized = color_transfer(tile_resized, target_region_mean,
                                  color_transfer_alpha)

    # Create polygon mask within the bounding box
    poly_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    if polygon is not None:
        local_poly = polygon.copy().astype(np.int32)
        local_poly[:, 0] -= x0
        local_poly[:, 1] -= y0
        cv2.fillPoly(poly_mask, [local_poly], color=255)

    # ZERO-GAP rendering:
    # use the exact Voronoi label map as the authoritative cell mask.
    # label_map partitions the full image, so no background cracks can remain.
    cell_local = cell_mask[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1][cell_local] = tile_resized[cell_local]


# ---------------------------------------------------------------------------
# Draw grout lines
# ---------------------------------------------------------------------------

def draw_grout(
    canvas: np.ndarray,
    label_map: np.ndarray,
    grout_color: tuple[int, int, int] = (20, 20, 20),
    grout_width: int = 0,
) -> np.ndarray:
    """
    Draw a thin border around each Voronoi cell to simulate mosaic grout.

    Implementation: wherever a pixel's cell label differs from any of its
    4-neighbours, that pixel is on a cell boundary → paint grout colour.

    Parameters
    ----------
    canvas       : H × W × 3 uint8
    label_map    : H × W int32
    grout_color  : RGB tuple (default near-black)
    grout_width  : 1 = single-pixel; 2 = dilated 1-px grout

    Returns
    -------
    canvas with grout lines painted (in-place copy)
    """
    result = canvas.copy()
    H, W   = label_map.shape

    # Boundary = pixels where label ≠ label of right or bottom neighbour
    right  = np.zeros((H, W), dtype=bool)
    bottom = np.zeros((H, W), dtype=bool)
    right[:, :-1]  = label_map[:, :-1] != label_map[:, 1:]
    bottom[:-1, :] = label_map[:-1, :] != label_map[1:, :]
    boundary = right | bottom

    if grout_width > 1:
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_RECT, (grout_width, grout_width)
        )
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel) > 0

    result[boundary] = grout_color
    return result


# ---------------------------------------------------------------------------
# Full canvas renderer
# ---------------------------------------------------------------------------

def render_voronoi_mosaic(
    target_img: np.ndarray,
    hires_cache: np.ndarray,
    label_map: np.ndarray,
    cell_tile_indices: list[int],
    polygons: list[np.ndarray | None],
    color_transfer_alpha: float = 0.55,
    grout_color: tuple[int, int, int] = (20, 20, 20),
    grout_width: int = 0,
) -> np.ndarray:
    """
    Render the complete mosaic.

    Parameters
    ----------
    target_img         : H × W × 3 uint8  — reference image
    hires_cache        : (N_tiles, T, T, 3) uint8
    label_map          : H × W int32
    cell_tile_indices  : list of length N_cells — which tile each cell uses
    polygons           : list of length N_cells — polygon per cell
    color_transfer_alpha: blending strength
    grout_color        : RGB grout colour
    grout_width        : px width of grout line

    Returns
    -------
    canvas : H × W × 3 uint8  (the final mosaic)
    """
    H, W, _ = target_img.shape
    canvas = target_img.copy()
    N_cells  = len(cell_tile_indices)

    for i in range(N_cells):
        tile_idx = cell_tile_indices[i]
        if tile_idx < 0:
            continue

        # Mean RGB of the target region for colour transfer
        cell_mask = label_map == i
        if not np.any(cell_mask):
            continue
        region_mean = (target_img[cell_mask]
                       .astype(np.float32)
                       .mean(axis=0))

        render_cell(
            canvas        = canvas,
            hires_tile    = hires_cache[tile_idx],
            polygon       = polygons[i],
            label_map     = label_map,
            cell_idx      = i,
            target_region_mean = region_mean,
            color_transfer_alpha = color_transfer_alpha,
        )

    # Grout pass (after all tiles are painted)
    if grout_width > 0:
        canvas = draw_grout(canvas, label_map, grout_color, grout_width)

    return canvas

# ===========================================================================
# HIGH-RESOLUTION MUSEUM PRINT RENDERER
# Uses ORIGINAL JPG/PNG source images.
# ===========================================================================

def _load_original_tile_to_bbox(
    path: str,
    bbox_w: int,
    bbox_h: int,
):
    """
    Open original tile, center-crop square,
    resize ONCE to final print cell.
    """

    with Image.open(path) as im:

        im = im.convert("RGB")

        w0, h0 = im.size
        side = min(w0, h0)

        needed = max(bbox_w, bbox_h)
        was_upscaled = side < needed

        left = (w0 - side) // 2
        top = (h0 - side) // 2

        im = im.crop(
            (left, top, left + side, top + side)
        )

        im = im.resize(
            (bbox_w, bbox_h),
            Image.LANCZOS
        )

        return (
            np.asarray(im, dtype=np.uint8),
            was_upscaled
        )


def render_voronoi_mosaic_print(
    target_img: np.ndarray,
    tile_paths: list[str],
    label_map: np.ndarray,
    cell_tile_indices: list[int],
    polygons: list[np.ndarray | None],
    output_long_edge: int = 12000,
    color_transfer_alpha: float = 0.55,
    grout_color: tuple[int, int, int] = (20, 20, 20),
    grout_width: int = 0,
) -> np.ndarray:
    """
    Re-render existing VoroPlex geometry at print resolution.

    IMPORTANT:
    This does NOT upscale the 64/128px cache.
    It reloads each selected ORIGINAL source image.
    """

    H, W = target_img.shape[:2]

    scale = (
        float(output_long_edge)
        / float(max(H, W))
    )

    out_w = int(round(W * scale))
    out_h = int(round(H * scale))

    print(
        f"[print render] {W}x{H} -> "
        f"{out_w}x{out_h} using ORIGINAL tiles"
    )

    # Fallback base for tiny polygon rounding gaps.
    canvas = np.asarray(
        Image.fromarray(target_img).resize(
            (out_w, out_h),
            Image.LANCZOS
        ),
        dtype=np.uint8
    ).copy()

    n_cells = len(cell_tile_indices)

    flat_labels = label_map.ravel()

    counts = np.bincount(
        flat_labels,
        minlength=n_cells
    ).astype(np.float64)

    counts_safe = np.maximum(
        counts,
        1.0
    )

    region_means = np.zeros(
        (n_cells, 3),
        dtype=np.float32
    )

    for channel in range(3):

        sums = np.bincount(
            flat_labels,
            weights=target_img[:, :, channel].ravel(),
            minlength=n_cells,
        )

        region_means[:, channel] = (
            sums / counts_safe
        )

    sx = out_w / float(W)
    sy = out_h / float(H)

    scaled_polygons = []

    upscale_count = 0
    rendered_count = 0

    for i in range(n_cells):

        tile_idx = int(
            cell_tile_indices[i]
        )

        if (
            tile_idx < 0
            or tile_idx >= len(tile_paths)
        ):
            continue

        polygon = polygons[i]

        if (
            polygon is None
            or len(polygon) < 3
        ):
            continue

        poly = (
            polygon
            .astype(np.float64)
            .copy()
        )

        poly[:, 0] *= sx
        poly[:, 1] *= sy

        poly = np.round(
            poly
        ).astype(np.int32)

        poly[:, 0] = np.clip(
            poly[:, 0],
            0,
            out_w - 1
        )

        poly[:, 1] = np.clip(
            poly[:, 1],
            0,
            out_h - 1
        )

        # ====================================================
        # SMALL PRINT OVERLAP
        #
        # Keep the original polygon geometry and tile scale,
        # but extend each rendered cell slightly past its
        # boundary so tiny rasterisation cracks cannot expose
        # the background.
        #
        # 0.50 = half of one working-resolution pixel.
        # At ~10x PRINT scale this is about 5 output pixels.
        # ====================================================
        overlap_work_px = 0.50
        overlap_px = max(
            1,
            int(round(overlap_work_px * scale))
        )

        orig_x, orig_y, orig_w, orig_h = (
            cv2.boundingRect(poly)
        )

        if orig_w < 1 or orig_h < 1:
            continue

        # Expanded PRINT ROI.
        x = max(0, orig_x - overlap_px)
        y = max(0, orig_y - overlap_px)

        x_end = min(
            out_w,
            orig_x + orig_w + overlap_px
        )
        y_end = min(
            out_h,
            orig_y + orig_h + overlap_px
        )

        bbox_w = x_end - x
        bbox_h = y_end - y

        if bbox_w < 1 or bbox_h < 1:
            continue

        # Load the source tile at its ORIGINAL polygon bbox size.
        # This preserves the previous tile scale.
        try:
            tile_core, was_upscaled = (
                _load_original_tile_to_bbox(
                    tile_paths[tile_idx],
                    orig_w,
                    orig_h
                )
            )
        except Exception as e:
            print(
                f"[print warning] tile "
                f"{tile_idx} failed: {e}"
            )
            continue

        upscale_count += int(
            was_upscaled
        )
        rendered_count += 1

        tile_core = color_transfer(
            tile_core,
            region_means[i],
            alpha=color_transfer_alpha,
        )

        # Padding size needed around the original tile.
        pad_left = orig_x - x
        pad_top = orig_y - y
        pad_right = x_end - (orig_x + orig_w)
        pad_bottom = y_end - (orig_y + orig_h)

        # Extend edge pixels rather than rescaling the tile.
        # Therefore the internal image remains unchanged.
        tile = cv2.copyMakeBorder(
            tile_core,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Original polygon in the expanded ROI.
        local_poly = poly.copy()
        local_poly[:, 0] -= x
        local_poly[:, 1] -= y

        mask = np.zeros(
            (bbox_h, bbox_w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [local_poly],
            255
        )

        # Expand only the polygon coverage.
        # The tile itself is NOT resized again.
        kernel_size = 2 * overlap_px + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        roi = canvas[
            y:y+bbox_h,
            x:x+bbox_w
        ]

        inside = mask > 0
        roi[inside] = tile[inside]

        scaled_polygons.append(
            poly
        )

    if grout_width > 0:

        scaled_grout = max(
            1,
            int(round(
                grout_width * scale
            ))
        )

        for poly in scaled_polygons:

            cv2.polylines(
                canvas,
                [poly],
                True,
                grout_color,
                thickness=scaled_grout,
                lineType=cv2.LINE_AA,
            )

    print(
        f"[print render] rendered "
        f"{rendered_count}/{n_cells} cells"
    )

    print(
        f"[print render] source tiles "
        f"requiring upscaling: "
        f"{upscale_count}"
    )

    return canvas

