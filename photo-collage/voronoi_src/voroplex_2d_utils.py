import os
import cv2
import numpy as np
import torch
from PIL import Image
import voroplex
from voroplex.cell_analysis_2d import compute_polygon_cell_info_2d


def compute_topology_2d(points01: torch.Tensor, threads="auto"):
    """Recompute VoroPlex 2D topology from current generator positions."""
    points_np = np.ascontiguousarray(
        points01.detach().cpu().numpy().astype(np.float64)
    )
    return voroplex.compute_voronoi_2d(
        points_np,
        [[0.0, 1.0], [0.0, 1.0]],
        threads=threads,
        representation="topology",
    )


def reconstruct_cells_2d(topology, points01: torch.Tensor):
    """
    Reconstruct differentiable cell connectivity using voroplex's own
    global-vertex representation, so that voroplex.cell_analysis_2d.
    compute_polygon_cell_info_2d (area/perimeter/centroid/adjacency -- see
    polygon_statistics below) can be used directly instead of us
    reimplementing that geometry by hand.

    voroplex.reconstruct_vertices_2d gives a single differentiable table of
    unique vertex coordinates; voroplex.build_polygonal_cells_2d gives, per
    cell, the ordered list of indices into that table. We only need to pad
    that ragged list of lists into a fixed-width tensor ourselves -- that
    padding/masking bookkeeping isn't something the library needs to do for
    you, but the actual geometry (which voroplex.cell_analysis_2d.
    compute_polygon_cell_info_2d consumes) is voroplex's own.
    """
    device = points01.device
    vertices = voroplex.reconstruct_vertices_2d(topology, points01, device=device)
    polygons, _ = voroplex.build_polygonal_cells_2d(topology, device=device)

    n_cells = len(polygons)
    max_verts = max((len(p) for p in polygons), default=0)

    cell_vertex_ids = torch.zeros((n_cells, max_verts), dtype=torch.int64, device=device)
    cell_mask = torch.zeros((n_cells, max_verts), dtype=torch.bool, device=device)
    num_vertices_per_cell = torch.zeros(n_cells, dtype=torch.int64, device=device)
    for i, poly in enumerate(polygons):
        m = len(poly)
        if m == 0:
            continue
        cell_vertex_ids[i, :m] = torch.as_tensor(poly, dtype=torch.int64, device=device)
        cell_mask[i, :m] = True
        num_vertices_per_cell[i] = m

    return vertices, cell_vertex_ids, cell_mask, num_vertices_per_cell


def polygon_statistics(
    vertices: torch.Tensor,
    cell_vertex_ids: torch.Tensor,
    cell_mask: torch.Tensor,
    num_vertices_per_cell: torch.Tensor,
):
    """
    Area / perimeter / centroid / adjacency come straight from voroplex's
    own compute_polygon_cell_info_2d -- notably its centroid uses the real
    polygon-centroid formula (area-weighted), which is more correct than
    the naive vertex-average we used before.

    Edge NORMALS (direction perpendicular to each boundary wall, needed for
    our Kanatani fabric-tensor orientation loss) aren't something the
    library computes, since that's specific to our own loss design rather
    than general polygon geometry -- so that part is still ours, built from
    the same padded per-cell vertex gather voroplex's own function uses
    internally.
    """
    info = compute_polygon_cell_info_2d(
        vertices, cell_vertex_ids, num_vertices_per_cell, cell_mask
    )

    device = vertices.device
    dtype = vertices.dtype
    N, M = cell_vertex_ids.shape

    safe_ids = cell_vertex_ids.masked_fill(~cell_mask, 0)
    gathered = vertices[safe_ids]
    positions = torch.arange(M, device=device).unsqueeze(0)
    next_positions = torch.where(
        positions + 1 < num_vertices_per_cell[:, None],
        positions + 1,
        torch.zeros_like(positions),
    )
    next_ids = torch.gather(safe_ids, 1, next_positions)
    next_vertices = vertices[next_ids]

    edge_vec_x = next_vertices[..., 0] - gathered[..., 0]
    edge_vec_y = next_vertices[..., 1] - gathered[..., 1]
    edge_len_raw = torch.sqrt(edge_vec_x ** 2 + edge_vec_y ** 2 + 1e-12)
    inv_len = 1.0 / edge_len_raw.clamp_min(1e-12)
    edge_dir_x = edge_vec_x * inv_len
    edge_dir_y = edge_vec_y * inv_len
    # Rotate edge direction 90 deg to get the (unoriented -- sign doesn't
    # matter since it only ever appears squared) outward/inward normal.
    edge_normal = torch.stack([-edge_dir_y, edge_dir_x], dim=-1)  # (N, M, 2)

    return {
        "centroid": info["cell_centroids"],
        "area": info["cell_areas"],
        "perimeter": info["cell_perimeters"],
        "edge_normal": edge_normal,
        "edge_len": info["edge_lengths"],  # same masking/ordering convention as our edge_normal above
        "mask": cell_mask,
        "adjacency": info["adjacency_tensor"],
        "num_adjacent": info["num_adjacent"],
    }


def points01_to_pixels(points01: np.ndarray, H: int, W: int) -> np.ndarray:
    out = np.asarray(points01, dtype=np.float64).copy()
    out[:, 0] *= max(W - 1, 1)
    out[:, 1] *= max(H - 1, 1)
    return out.astype(np.float32)


def save_seed_overlay(img: np.ndarray, seeds_px: np.ndarray, path: str):
    vis = img.copy()
    for x, y in seeds_px:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 0, 0), -1)
    Image.fromarray(vis.astype(np.uint8)).save(path, quality=92)
    print(f"[save] {path}")


def save_label_visualization(label_map: np.ndarray, path: str):
    n = int(label_map.max()) + 1
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    vis = palette[label_map.clip(0, n - 1)]
    Image.fromarray(vis).save(path, quality=92)
    print(f"[save] {path}")


def _compute_cell_aspect_ratios(label_map: np.ndarray, min_pixels: int = 20):
    """Shared helper: per-cell major/minor axis ratio + major-axis unit vector
    from a PCA of each cell's pixel mask. Used by both the standalone
    heatmap and the on-photo overlay below, so the two always agree."""
    n_cells = int(label_map.max()) + 1
    ratios = np.ones(n_cells, dtype=np.float32)
    centers = np.zeros((n_cells, 2), dtype=np.float64)
    axes = np.zeros((n_cells, 2), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)

    for i in range(n_cells):
        ys, xs = np.where(label_map == i)
        counts[i] = len(xs)
        if len(xs) < min_pixels:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(len(pts), 1)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-8, None)
        ratios[i] = float(np.sqrt(vals[-1] / vals[0]))
        centers[i] = center
        axes[i] = vecs[:, int(np.argmax(vals))]

    return ratios, centers, axes, counts


def save_cell_aspect_heatmap(
    img: np.ndarray,
    label_map: np.ndarray,
    path: str,
    min_pixels: int = 20,
    ratio_cap: float = 8.0,
):
    """
    Diagnostic view: color each final cell by its measured elongation
    (major/minor axis ratio from a PCA of its pixel mask), independent of the
    random per-cell palette used elsewhere. Bright/hot = strongly elongated,
    dark = still roughly isotropic. Use this instead of the random-color
    label map to actually judge whether the directional training is working,
    since random colors make it hard to see shape at a glance.

    Prints summary stats (mean/median ratio, fraction of cells above 2x/3x)
    so you can track progress numerically run to run, not just by eye.
    """
    n_cells = int(label_map.max()) + 1
    ratios, _, _, _ = _compute_cell_aspect_ratios(label_map, min_pixels=min_pixels)

    norm = np.clip(ratios / float(ratio_cap), 0.0, 1.0)
    colors_bgr = cv2.applyColorMap(
        (norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )[:, 0, :]
    colors_rgb = colors_bgr[:, ::-1]
    vis = colors_rgb[label_map.clip(0, n_cells - 1)]

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    Image.fromarray(vis.astype(np.uint8)).save(path, quality=92)

    valid = ratios[ratios > 1.0 + 1e-4]
    if len(valid) == 0:
        valid = ratios
    print(
        f"[aspect] mean={ratios.mean():.2f} median={np.median(ratios):.2f} "
        f"max={ratios.max():.2f}  "
        f">2x={(ratios > 2.0).mean() * 100:.1f}%  "
        f">3x={(ratios > 3.0).mean() * 100:.1f}%  "
        f">{ratio_cap:.0f}x(cap)={(ratios >= ratio_cap).mean() * 100:.1f}%"
    )
    print(f"[save] {path}")


def save_cell_aspect_overlay(
    img: np.ndarray,
    label_map: np.ndarray,
    path: str,
    min_pixels: int = 20,
    ratio_cap: float = 8.0,
    heatmap_alpha: float = 0.45,
    draw_axes: bool = True,
    axis_min_ratio: float = 1.6,
    line_fraction: float = 0.65,
):
    """
    The real answer to "is the elongation actually following the fur/edges":
    blend the aspect heatmap semi-transparently OVER the real target photo
    (instead of showing it standalone, disconnected from the image content),
    and additionally draw each sufficiently-elongated cell's own major-axis
    line on top. If the method is working, the bright heatmap regions and
    the axis line directions should visibly track the mane/fur strands in
    the photo underneath, not look independent of it.
    """
    n_cells = int(label_map.max()) + 1
    ratios, centers, axes, counts = _compute_cell_aspect_ratios(
        label_map, min_pixels=min_pixels
    )

    norm = np.clip(ratios / float(ratio_cap), 0.0, 1.0)
    colors_bgr = cv2.applyColorMap(
        (norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )[:, 0, :]
    colors_rgb = colors_bgr[:, ::-1].astype(np.float32)
    heat = colors_rgb[label_map.clip(0, n_cells - 1)]

    base = img.astype(np.float32)
    a = float(np.clip(heatmap_alpha, 0.0, 1.0))
    vis = (1.0 - a) * base + a * heat
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    if draw_axes:
        for i in range(n_cells):
            if counts[i] < min_pixels or ratios[i] < axis_min_ratio:
                continue
            ys, xs = np.where(label_map == i)
            pts = np.stack([xs, ys], axis=1).astype(np.float64)
            spread = pts - centers[i]
            proj = spread @ axes[i]
            half_len = 0.5 * line_fraction * (proj.max() - proj.min())
            p1 = centers[i] - half_len * axes[i]
            p2 = centers[i] + half_len * axes[i]
            cv2.line(
                vis,
                tuple(np.round(p1).astype(int)),
                tuple(np.round(p2).astype(int)),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    Image.fromarray(vis).save(path, quality=92)
    print(f"[save] {path}")


def save_cell_axis_overlay(
    img: np.ndarray,
    label_map: np.ndarray,
    path: str,
    min_pixels: int = 20,
    line_fraction: float = 0.70,
):
    """Visualize the major axis of each rasterized final Voronoi cell."""
    vis = img.copy()
    n_cells = int(label_map.max()) + 1

    for i in range(n_cells):
        ys, xs = np.where(label_map == i)
        if len(xs) < min_pixels:
            continue

        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(len(pts), 1)

        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        major_len = 2.0 * np.sqrt(max(float(vals.max()), 1e-8))
        half = 0.5 * line_fraction * major_len

        p1 = center - half * major
        p2 = center + half * major

        cv2.line(
            vis,
            tuple(np.round(p1).astype(int)),
            tuple(np.round(p2).astype(int)),
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    Image.fromarray(vis.astype(np.uint8)).save(path, quality=92)
    print(f"[save] {path}")
