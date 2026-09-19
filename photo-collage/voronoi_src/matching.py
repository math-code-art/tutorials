"""
方向6: Enhanced Tile Matching for Voronoi Cells
================================================

Matching score for cell i vs tile j:

    Score(i,j) = w_lab  · d_lab(i,j)
               + w_tex  · d_texture(i,j)
               + w_vgg  · d_vgg(i,j)
               + w_pen  · reuse_penalty(j)

Components
----------
d_lab        — L*a*b* colour distance between cell mean and tile mean.
               LAB is perceptually uniform; Euclidean distance ≈ ΔE.

d_texture    — LBP (Local Binary Pattern) histogram distance.
               LBP is a lightweight rotation-invariant texture descriptor
               that captures edge orientations and micro-patterns without
               needing a GPU.

d_vgg        — VGG deep embedding distance (reuses your existing embeddings).

reuse_penalty — exponential penalty for tiles already used many times,
               keeping the mosaic visually diverse.

All distances are independently z-score normalised across candidates
before weighting, so no single component dominates due to scale.
"""

import numpy as np
from scipy.spatial import KDTree
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------------------------
# LAB colour distance
# ---------------------------------------------------------------------------

def _rgb_to_lab_mean(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert H×W×3 uint8 → mean LAB pixel (3,).
    Pure-numpy approximation to avoid a full cv2 dependency here.
    Works fine for colour matching (we only need the mean).
    """
    import cv2
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(axis=0)


def precompute_tile_lab_means(tile_cache: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    tile_cache : (N, S, S, 3) uint8

    Returns
    -------
    lab_means : (N, 3) float32  — mean LAB for each tile
    """
    import cv2
    N  = len(tile_cache)
    out = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        lab = cv2.cvtColor(tile_cache[i], cv2.COLOR_RGB2LAB).astype(np.float32)
        out[i] = lab.reshape(-1, 3).mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# LBP texture descriptor
# ---------------------------------------------------------------------------

_LBP_RADIUS  = 2
_LBP_N_POINTS = 8 * _LBP_RADIUS    # 16
_LBP_N_BINS   = _LBP_N_POINTS + 2  # uniform LBP bins


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Compute uniform LBP histogram (L1-normalised) for a grayscale patch.
    """
    lbp  = local_binary_pattern(
        gray, _LBP_N_POINTS, _LBP_RADIUS, method="uniform"
    )
    hist, _ = np.histogram(lbp.ravel(), bins=_LBP_N_BINS,
                           range=(0, _LBP_N_BINS), density=True)
    return hist.astype(np.float32)


def precompute_tile_lbp(tile_cache: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    tile_cache : (N, S, S, 3) uint8

    Returns
    -------
    lbp_hists : (N, _LBP_N_BINS) float32
    """
    import cv2
    N   = len(tile_cache)
    out = np.zeros((N, _LBP_N_BINS), dtype=np.float32)
    for i in range(N):
        gray   = cv2.cvtColor(tile_cache[i], cv2.COLOR_RGB2GRAY)
        out[i] = _lbp_histogram(gray)
    return out


def get_block_lbp(block_rgb: np.ndarray) -> np.ndarray:
    """LBP histogram for a single query block."""
    import cv2
    gray = cv2.cvtColor(block_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return _lbp_histogram(gray)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class EnhancedMatcher:
    """
    Encapsulates all pre-computed tile features and exposes a single
    `find_best` method called once per Voronoi cell.

    Parameters
    ----------
    tile_cache_small : (N, S, S, 3) uint8   — small tiles (for LAB + LBP)
    emb_vgg          : (N, D) float32       — VGG embeddings (any cut)
    w_lab            : weight for LAB colour distance
    w_tex            : weight for LBP texture distance
    w_vgg            : weight for VGG distance
    w_pen            : weight for reuse penalty
    topk             : number of VGG nearest neighbours to retrieve first
    max_use          : hard cap on how many times a tile may be reused
    penalty_scale    : controls how fast the reuse penalty grows
    local_radius     : grid radius for spatial diversity (no same tile near neighbours)
    """

    def __init__(
        self,
        tile_cache_small: np.ndarray,
        emb_vgg: np.ndarray,
        w_lab: float = 0.30,
        w_tex: float = 0.20,
        w_vgg: float = 0.40,
        w_pen: float = 0.10,
        topk: int = 40,
        max_use: int = 6,
        penalty_scale: float = 0.4,
        local_radius: int = 3,
    ):
        self.tile_cache_small = tile_cache_small
        self.emb_vgg          = emb_vgg
        self.topk             = topk
        self.max_use          = max_use
        self.penalty_scale    = penalty_scale
        self.local_radius     = local_radius

        self.w_lab  = w_lab
        self.w_tex  = w_tex
        self.w_vgg  = w_vgg
        self.w_pen  = w_pen

        N = len(tile_cache_small)
        print("[matcher] pre-computing LAB means...", end=" ", flush=True)
        self.lab_means = precompute_tile_lab_means(tile_cache_small)
        print("done")

        print("[matcher] pre-computing LBP histograms...", end=" ", flush=True)
        self.lbp_hists = precompute_tile_lbp(tile_cache_small)
        print("done")

        print("[matcher] building VGG KDTree...", end=" ", flush=True)
        self.vgg_tree = KDTree(emb_vgg)
        print("done")

        # Runtime state
        self.use_count  = np.zeros(N, dtype=np.int32)
        self.placed     = {}         # (cell_row, cell_col) → tile_index

    # ------------------------------------------------------------------
    def find_best(
        self,
        cell_region: np.ndarray,
        cell_vgg_emb: np.ndarray,
        cell_row: int,
        cell_col: int,
    ) -> int:
        """
        Find the best matching tile for a Voronoi cell.

        Parameters
        ----------
        cell_region  : H' × W' × 3 uint8 — pixels inside cell bounding box
        cell_vgg_emb : (D,) float32
        cell_row     : approximate row index (for spatial diversity)
        cell_col     : approximate col index (for spatial diversity)

        Returns
        -------
        tile_index : int
        """
        import cv2

        # ── 1. VGG KNN pre-filter ───────────────────────────────────────
        _, idxs = self.vgg_tree.query(cell_vgg_emb[None, :], k=self.topk)
        cand    = idxs[0]                                  # (topk,)

        # ── 2. Compute LAB distance for candidates ──────────────────────
        cell_lab  = _rgb_to_lab_mean(cell_region.astype(np.uint8))
        d_lab     = np.linalg.norm(self.lab_means[cand] - cell_lab[None, :],
                                   axis=1)

        # ── 3. Compute LBP distance for candidates ──────────────────────
        cell_lbp = get_block_lbp(cell_region)
        d_tex    = np.sum(np.abs(self.lbp_hists[cand] - cell_lbp[None, :]),
                          axis=1)   # L1 histogram distance

        # ── 4. VGG distance ─────────────────────────────────────────────
        d_vgg = np.linalg.norm(self.emb_vgg[cand] - cell_vgg_emb[None, :],
                               axis=1)

        # ── 5. Z-normalise each component ───────────────────────────────
        def _znorm(v):
            mu, sd = v.mean(), v.std()
            if sd < 1e-8:
                return np.zeros_like(v)
            return (v - mu) / sd

        zlab  = _znorm(d_lab)
        ztex  = _znorm(d_tex)
        zvgg  = _znorm(d_vgg)

        score = (self.w_lab * zlab
                 + self.w_tex * ztex
                 + self.w_vgg * zvgg)

        # ── 6. Reuse penalty ────────────────────────────────────────────
        cand_use   = self.use_count[cand].astype(np.float32)
        score      = score + self.w_pen * np.exp(self.penalty_scale * cand_use)

        # ── 7. Hard cap: mask tiles used >= max_use ──────────────────────
        allowed = cand_use < self.max_use
        if np.any(allowed):
            cand2, score2 = cand[allowed], score[allowed]
        else:
            cand2, score2 = cand, score

        # ── 8. Spatial diversity: ban tiles used in nearby cells ─────────
        r = self.local_radius
        banned = set()
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                key = (cell_row + dr, cell_col + dc)
                if key in self.placed:
                    banned.add(self.placed[key])
        if banned:
            keep = np.array([c not in banned for c in cand2], dtype=bool)
            if np.any(keep):
                cand2, score2 = cand2[keep], score2[keep]

        # ── 9. Pick best (with small stochastic top-k to avoid ties) ────
        k    = min(5, len(cand2))
        best = np.argpartition(score2, k - 1)[:k]
        chosen = int(np.random.default_rng().choice(cand2[best]))

        self.use_count[chosen] += 1
        self.placed[(cell_row, cell_col)] = chosen
        return chosen
