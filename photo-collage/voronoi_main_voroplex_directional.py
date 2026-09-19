"""Run the VoroPlex + PyTorch + Sobel directional Voronoi photomosaic."""

import math
import os
import sys
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _here not in sys.path:
    sys.path.insert(0, _here)

import config
from src.dataset import get_tile_paths, get_tile_cache, get_hires_tile_cache
from src.features import get_embeddings
from voronoi_src.voronoi_pipeline_voroplex_directional import (
    VoroPlexDirectionalMosaicPipeline,
)


# ---------------------------------------------------------------------------
# VALIDATION_MODE: quick, cheap, eyeball-verifiable sanity check.
#
# True  -> global "preferred direction" only (paper Fig. 9 style): every
#          cell in the WHOLE image is pushed toward the SAME fixed angle
#          (preferred_direction_deg), regardless of local image content.
#          Fewer seeds + fewer iters, so it runs in well under a minute.
#          Look at 09_..._final.jpg / 07b_cell_aspect_heatmap.jpg afterward
#          -- every cell should visibly lean the same way. If it doesn't,
#          the bug is in the fabric-tensor/roundness/training machinery
#          itself, not in the Sobel guidance field.
#
# False -> the real run: local Sobel-guidance-following (with edge-stopping
#          diffusion into flat interiors), full seed count and iterations.
#
# Flip this back to False once the validation run looks correct.
# ---------------------------------------------------------------------------
VALIDATION_MODE = False


VOROPLEX_DIRECTIONAL_CFG = dict(
    density_alpha=0.35,
    density_beta=0.25,
    density_gamma=0.20,
    density_delta=0.20,

    n_initial_seeds=3000,          # was 2000 - pushing resolution further
    lloyd_iters_init=8,

    guidance_blur_sigma=6.0,      # local structure-tensor smoothing (removes pixel-level noise)
    guidance_sobel_ksize=3,
    direction_diffuse_radius_px=28.0,  # was 35.0 - scaled down again to match the smaller cell size at 3000 seeds
    direction_edge_stop_k=0.10,        # smaller = harder stop at real edges; this is the actual knob for "edges stay sharp"
    guidance_mode="tangent",      # only affects the 03_* overlay visualization; training always targets the "normal" field internally

    # Differentiable VoroPlex training.
    train_iters=380,               # was 350 - more seeds = more DOF, a bit more time to settle
    train_lr=1.8e-3,
    lr_min_factor=0.05,
    voroplex_threads=1,           # keep at 1 on Mac: voroplex's own OpenMP runtime + PyTorch's MKL/OpenMP runtime commonly segfault when mixed multi-threaded

    # Fabric-tensor orientation (Kanatani tensor on the cell's own boundary
    # edge normals, matched to a local target axis) + isoperimetric
    # roundness regularizer. The orientation loss asks walls to lean toward
    # the local edge direction; roundness separately keeps cells
    # well-formed, so the two don't fight.
    target_alpha=0.88,             # anisotropy strength for the LOCAL (Sobel-following) term
    confidence_gamma=0.70,         # <1 broadens which cells get real weight in the local orientation loss
    # target_roundness is derived below from target_alpha so it matches the
    # roundness of a rectangle at the IMPLIED elongation, instead of a pure
    # square (pi/4). A pure-square target actively fights the orientation
    # loss (elongated shapes always have lower isoperimetric ratio than a
    # square).
    lambda_orientation=12.0,       # weight on the LOCAL (Sobel-following) term
    lambda_roundness=0.9,
    lambda_sliver_guard=6.0,       # was 4.0 - clamp down harder on rare severe outliers
    lambda_size_uniform=0.35,      # was 0.25 - tighter overall size consistency
    lambda_area=0.22,              # was 0.18 - tighter per-cell area anchor
    lambda_separation=0.20,        # was 0.12 - stronger push against seeds crowding into near-collinear chains along edges, which is what produces lots of 3-sided (triangle) cells
    lambda_anchor=0.02,            # decays to lambda_anchor * anchor_decay_floor over training (see below)
    anchor_decay_floor=0.08,
    lambda_edge_attract=1.2,       # was 1.6 - softened a bit: strong edge attraction is what pulls seeds into tight, near-collinear chains along a curve, which is the direct cause of the triangle-heavy look
    edge_attract_warmup_frac=0.25,
    min_separation_factor=0.32,    # was 0.25 - meaningfully more spacing headroom between seeds, giving each cell more room to have 4-6 neighbors instead of collapsing to 3
    grad_clip=0.15,                # per-seed gradient-vector clip (NOT a global norm)
    print_every=10,

    # Global "preferred direction" (paper Fig. 9 style): the SAME fixed
    # target axis for every cell, no confidence weighting. Off by default
    # (lambda_preferred=0) -- VALIDATION_MODE below turns it on and turns
    # the local term off, for the sanity check described above.
    preferred_direction_deg=0.0,   # 0 = horizontal, 90 = vertical
    preferred_alpha=0.85,
    lambda_preferred=0.00,

    w_lab=0.30,
    w_tex=0.20,
    w_vgg=0.40,
    w_pen=0.10,

    color_transfer_alpha=0.40,
    grout_color=(20, 20, 20),
    grout_width=0,
)

# Derive target_roundness from target_alpha: for a rectangle with sides
# w, h = axis_ratio*w where axis_ratio = sqrt(alpha/(1-alpha)), the
# isoperimetric ratio is q = pi*axis_ratio / (1+axis_ratio)^2. Using this
# instead of a flat pi/4 (square) target means the roundness regularizer
# stops fighting the orientation loss once cells reach roughly the intended
# elongation, and only kicks in hard against going beyond that into slivers.
_axis_ratio = (VOROPLEX_DIRECTIONAL_CFG["target_alpha"] / (1.0 - VOROPLEX_DIRECTIONAL_CFG["target_alpha"])) ** 0.5
VOROPLEX_DIRECTIONAL_CFG["target_roundness"] = math.pi * _axis_ratio / (1.0 + _axis_ratio) ** 2

# floor_roundness: the hard sliver-guard kicks in once q drops below this.
# Was 2x the intended axis ratio; tightened to 1.6x for a stricter clamp on
# outliers (previous runs still showed a long tail up to ~15-18x aspect
# despite the guard -- the averaged roundness_loss alone wasn't enough, and
# a tighter floor gives the cubic sliver_guard_loss more cells to actually
# act on).
_floor_axis_ratio = 1.6 * _axis_ratio
VOROPLEX_DIRECTIONAL_CFG["floor_roundness"] = math.pi * _floor_axis_ratio / (1.0 + _floor_axis_ratio) ** 2

if VALIDATION_MODE:
    print("=" * 72)
    print("[voroplex directional] VALIDATION_MODE=True -- running the cheap,")
    print("global preferred-direction sanity check, NOT the real local-Sobel run.")
    print("Flip VALIDATION_MODE to False in this file once this looks correct.")
    print("=" * 72)
    VOROPLEX_DIRECTIONAL_CFG.update(
        n_initial_seeds=900,   # was 400 - bumped up so cells are small enough to actually see the alignment pattern clearly; still much cheaper than the full 1300+300iters run
        train_iters=80,
        lambda_orientation=0.0,   # local term off
        lambda_preferred=8.0,     # global term on
        preferred_direction_deg=0.0,
        preferred_alpha=0.85,
    )


def run_voroplex_directional(target_path: str) -> np.ndarray:
    print("=" * 72)
    print(f"[voroplex directional] target: {target_path}")
    print("=" * 72)

    img = Image.open(target_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, getattr(config, "TARGET_RESOLUTION", 2400) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    target_img = np.array(img, dtype=np.uint8)

    H, W, _ = target_img.shape
    H2 = (H // config.TILE_SIZE_MAX) * config.TILE_SIZE_MAX
    W2 = (W // config.TILE_SIZE_MAX) * config.TILE_SIZE_MAX
    target_img = target_img[:H2, :W2]
    print(f"image size after snap: {target_img.shape}")

    print("[data] loading tile paths / caches / VGG features")
    tile_paths = get_tile_paths(max_tiles=config.POOL_TILES)
    tile_cache_small = get_tile_cache(tile_paths, config.TILE_SIZE_MIN)
    hires_cache = get_hires_tile_cache(tile_paths)
    emb_vgg = get_embeddings(tile_cache_small, config.CUT3)

    target_name = os.path.splitext(os.path.basename(target_path))[0]
    output_dir = os.path.join(
        config.OUTPUT_DIR,
        f"{target_name}_voroplex_directional_steps",
    )
    os.makedirs(output_dir, exist_ok=True)

    pipeline = VoroPlexDirectionalMosaicPipeline(
        tile_cache_small=tile_cache_small,
        hires_cache=hires_cache,
        emb_vgg=emb_vgg,
        tile_paths=tile_paths,
        **VOROPLEX_DIRECTIONAL_CFG,
    )

    return pipeline.run(
        target_img=target_img,
        output_dir=output_dir,
        save_intermediates=True,
        target_name=target_name,
    )


if __name__ == "__main__":
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    target_paths = []

    for filename in os.listdir(config.TARGET_DIR):
        if os.path.splitext(filename)[1] in extensions:
            target_paths.append(os.path.join(config.TARGET_DIR, filename))

    target_paths.sort()

    if not target_paths:
        print(f"error: no images found in {config.TARGET_DIR}")
        sys.exit(1)

    print(f"found {len(target_paths)} target image(s)")
    for tp in target_paths:
        run_voroplex_directional(tp)