"""
Loss terms for training VoroPlex generator positions.

The orientation term follows the Kanatani fabric-tensor formulation from the
differentiable-Voronoi foam-optimization literature (Eq. 24-27 of the
referenced paper): instead of forcing each cell's vertex covariance into an
elongated target shape (which lets the optimizer "cheat" toward slivers), we
match the length-weighted second-moment tensor of the cell's own boundary
EDGE NORMALS to a target anisotropy along a local direction. A separate
isoperimetric roundness regularizer keeps cells well-conditioned; the two
terms don't fight each other the way a hard elongation target did.
"""
import math
import torch
import torch.nn.functional as F


def sample_field(field_t: torch.Tensor, points01: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a [1,C,H,W] field at Nx2 normalized [0,1] points."""
    grid = points01.clamp(0.0, 1.0) * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)
    sampled = F.grid_sample(
        field_t,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def local_fabric_tensor_loss(
    edge_normal: torch.Tensor,
    edge_len: torch.Tensor,
    target_dir: torch.Tensor,
    target_alpha: float,
    eps: float = 1e-8,
):
    """
    Per-cell Kanatani fabric tensor loss.

    `target_dir` is the local structure-tensor NORMAL direction (perpendicular
    to the image contour) at each cell's centroid. A cell elongated ALONG the
    contour (tangent) direction has its two long walls running parallel to
    the tangent, so their normals concentrate along the perpendicular
    (normal) direction -- which is exactly `target_dir` here. That's why we
    target alignment with `normal`, not `tangent`, even though the visual
    effect is elongation along the tangent/contour.

    `target_alpha` is a FIXED scalar anisotropy strength in (0.5, 1.0) --
    the same target for every cell. Confidence is NOT baked into this target
    (that waters down the target itself, e.g. average confidence ~0.4 turns
    an intended 0.8 into an effective ~0.62). Instead, confidence is used to
    WEIGHT how much each cell's loss counts (see compute_directional_training_loss),
    so uncertain cells get less gradient pressure while confident cells still
    get pushed all the way toward the real target.
    """
    u = target_dir / target_dir.norm(dim=1, keepdim=True).clamp_min(eps)
    n_perp = torch.stack([-u[:, 1], u[:, 0]], dim=1)

    a = (edge_normal * u.unsqueeze(1)).sum(dim=-1)        # (N, M)
    b = (edge_normal * n_perp.unsqueeze(1)).sum(dim=-1)    # (N, M)

    w = edge_len
    wsum = w.sum(dim=1).clamp_min(eps)

    Axx = (w * a * a).sum(dim=1) / wsum
    Ayy = (w * b * b).sum(dim=1) / wsum
    Axy = (w * a * b).sum(dim=1) / wsum

    target_diag = 2.0 * float(target_alpha) - 1.0
    per_cell = (Axx - Ayy - target_diag) ** 2 + Axy ** 2
    return per_cell, Axx, Ayy


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12):
    w = weights.clamp_min(0.0)
    return (values * w).sum() / w.sum().clamp_min(eps)


def roundness_loss(
    area: torch.Tensor,
    perimeter: torch.Tensor,
    target_q: float = math.pi / 4.0,
    eps: float = 1e-8,
):
    """
    Isoperimetric roundness regularizer (2D): q = 4*pi*area / perimeter^2,
    q=1 for a circle, q=pi/4 for a square, q->0 for a sliver.

    One-sided on purpose: only penalizes cells LESS round than target_q
    (i.e. more sliver-like). A cell that happens to be rounder than the
    target isn't punished for it -- we don't actually care about matching
    the target exactly, only about not degenerating past it.
    """
    q = 4.0 * math.pi * area / perimeter.clamp_min(eps) ** 2
    deficit = torch.relu(target_q - q)
    return torch.mean(deficit ** 2), q


def sliver_guard_loss(q: torch.Tensor, floor_q: float):
    """
    A second, much steeper guard specifically against rare severe outliers
    (e.g. one stray cell at 15-20x aspect while the rest sit around 2x).
    `roundness_loss` above is a soft, averaged penalty, so a handful of very
    bad cells barely move the mean and can slip through -- this uses a cubic
    penalty so small deficits cost almost nothing but a badly degenerate
    cell (q far below floor_q) costs a lot, specifically squashing the tail
    without dragging down cells that are already fine.
    """
    deficit = torch.relu(float(floor_q) - q)
    return torch.mean(deficit ** 3)


def size_uniformity_loss(area: torch.Tensor, eps: float = 1e-8):
    """Var(size) / mean(size)^2 -- discourages a few huge cells plus a
    swarm of tiny ones, complementary to the per-cell area anchor below."""
    mean_area = area.mean().clamp_min(eps)
    var = area.var(unbiased=False)
    return var / (mean_area ** 2)


def area_preservation_loss(area: torch.Tensor, target_area: torch.Tensor, eps: float = 1e-12):
    return torch.mean(
        (torch.log(area.clamp_min(eps)) - torch.log(target_area.clamp_min(eps))) ** 2
    )


def edge_attraction_loss(points01: torch.Tensor, confidence_t: torch.Tensor):
    """
    Pull generator points themselves toward high-confidence (strong, coherent
    edge) regions, instead of only orienting cells that happen to land there.

    This is what makes strips "hug" edges: without it, the orientation loss
    only aligns cell walls wherever a cell's centroid ends up, but does
    nothing to move the seed chain onto the edge in the first place.
    """
    point_confidence = sample_field(confidence_t, points01)[:, 0].clamp(0.0, 1.0)
    loss = torch.mean((1.0 - point_confidence) ** 2)
    return loss, point_confidence


def nearest_neighbor_separation_loss(points01: torch.Tensor, min_distance: float):
    N = points01.shape[0]
    if N < 2:
        return points01.new_tensor(0.0)

    distances = torch.cdist(points01, points01)
    eye = torch.eye(N, dtype=torch.bool, device=points01.device)
    distances = distances.masked_fill(eye, 10.0)
    nearest = distances.min(dim=1).values
    return torch.mean(torch.relu(float(min_distance) - nearest) ** 2)


def compute_directional_training_loss(
    *,
    points01: torch.Tensor,
    cell_centroid01: torch.Tensor,
    area: torch.Tensor,
    perimeter: torch.Tensor,
    edge_normal: torch.Tensor,
    edge_len: torch.Tensor,
    target_area: torch.Tensor,
    initial_points01: torch.Tensor,
    fabric_target_field_t: torch.Tensor,
    confidence_orient_t: torch.Tensor,
    confidence_edge_t: torch.Tensor,
    target_alpha: float,
    target_roundness: float,
    floor_roundness: float,
    confidence_gamma: float,
    preferred_direction_deg: float,
    preferred_alpha: float,
    lambda_orientation: float,
    lambda_preferred: float,
    lambda_roundness: float,
    lambda_sliver_guard: float,
    lambda_size_uniform: float,
    lambda_area: float,
    lambda_separation: float,
    lambda_anchor: float,
    lambda_edge_attract: float,
    min_separation: float,
):
    # Sample the fabric-tensor target direction (structure-tensor NORMAL
    # field) and confidence at each cell's differentiable centroid.
    target_dir = sample_field(fabric_target_field_t, cell_centroid01)
    confidence = sample_field(confidence_orient_t, cell_centroid01)[:, 0].clamp(0.0, 1.0)

    orient_per_cell, _, _ = local_fabric_tensor_loss(
        edge_normal, edge_len, target_dir, target_alpha
    )
    # Weight by confidence rather than diluting the target itself: cells in
    # ambiguous/flat regions contribute less gradient, but cells that ARE
    # confidently on an edge still get pushed all the way to the real
    # target_alpha instead of a watered-down effective target.
    #
    # confidence_gamma < 1 additionally broadens which cells count: with raw
    # confidence commonly sitting around ~0.4 across a busy photo, most of
    # the "weight budget" in a plain weighted mean concentrates on a small
    # sliver of the image. confidence**gamma (gamma<1) lifts mid-range
    # values so moderate-confidence regions still carry meaningful weight.
    confidence_weight = confidence.clamp_min(1e-6).pow(float(confidence_gamma))
    orientation_loss = weighted_mean(orient_per_cell, confidence_weight)

    # Global "preferred direction": the SAME fixed target axis for every
    # cell, regardless of local image content or confidence -- this is
    # exactly the paper's Fig. 9 demo (uniaxial A_tgt with a single u), and
    # is a much easier sanity check than the local-Sobel-following case:
    # with lambda_preferred set high (and lambda_orientation low), every
    # cell in the whole image should visibly line up along
    # preferred_direction_deg, which is trivial to verify by eye.
    #
    # IMPORTANT: local_fabric_tensor_loss's `target_dir` argument is the
    # WALL-NORMAL direction, not the elongation axis directly -- a cell
    # elongated ALONG some axis has its long walls' normals pointing
    # PERPENDICULAR to that axis (see that function's docstring, and how
    # the local Sobel term above correctly passes guidance["normal"], not
    # "tangent"). preferred_direction_deg is meant to be read naturally as
    # "the direction cells should elongate along", so we rotate it 90
    # degrees before building the vector that actually goes into the
    # fabric-tensor call -- forgetting this rotation is exactly what made
    # the very first validation run elongate perpendicular to the requested
    # angle (0 deg requested, ~90 deg observed).
    theta = math.radians(float(preferred_direction_deg) + 90.0)
    pref_dir = points01.new_tensor([math.cos(theta), math.sin(theta)])
    pref_dir = pref_dir.view(1, 2).expand(points01.shape[0], 2)
    pref_per_cell, _, _ = local_fabric_tensor_loss(
        edge_normal, edge_len, pref_dir, preferred_alpha
    )
    preferred_loss = pref_per_cell.mean()

    round_loss, q = roundness_loss(area, perimeter, target_q=target_roundness)
    sliver_loss = sliver_guard_loss(q, floor_roundness)
    size_loss = size_uniformity_loss(area)
    area_loss = area_preservation_loss(area, target_area)
    separation_loss = nearest_neighbor_separation_loss(points01, min_separation)
    anchor_loss = torch.mean((points01 - initial_points01) ** 2)
    edge_loss, point_confidence = edge_attraction_loss(points01, confidence_edge_t)

    total = (
        float(lambda_orientation) * orientation_loss
        + float(lambda_preferred) * preferred_loss
        + float(lambda_roundness) * round_loss
        + float(lambda_sliver_guard) * sliver_loss
        + float(lambda_size_uniform) * size_loss
        + float(lambda_area) * area_loss
        + float(lambda_separation) * separation_loss
        + float(lambda_anchor) * anchor_loss
        + float(lambda_edge_attract) * edge_loss
    )

    parts = {
        "total": total,
        "orientation": orientation_loss,
        "preferred": preferred_loss,
        "roundness": round_loss,
        "sliver_guard": sliver_loss,
        "size_uniform": size_loss,
        "area": area_loss,
        "separation": separation_loss,
        "anchor": anchor_loss,
        "edge_attract": edge_loss,
        "mean_confidence": confidence.mean(),
        "mean_point_confidence": point_confidence.mean(),
    }
    return total, parts