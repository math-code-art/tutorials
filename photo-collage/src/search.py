import os
import sys
import numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Variance helper
# ---------------------------------------------------------------------------

def compute_variance(block):
    """Luminance variance — perceptually weighted."""
    arr = block.astype(np.float32) / 255.0
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(np.var(lum))


# ---------------------------------------------------------------------------
# Adaptive Quadtree
# ---------------------------------------------------------------------------

def _adaptive_split(img, x, y, size, min_size, thresh, blocks):
    H, W, _ = img.shape
    if x + size > W or y + size > H or x < 0 or y < 0:
        return

    region = img[y:y + size, x:x + size]
    var    = compute_variance(region)

    # Progressive threshold: smaller blocks need lower variance to stop splitting.
    # This keeps faces and busy areas small while letting plain backgrounds stay big.
    levels_from_min  = max(0, int(round(np.log2(size / min_size))))
    effective_thresh = thresh * (1.0 + 0.6 * levels_from_min)

    if size <= min_size:
        blocks.append((x, y, size, size))
        return

    # Large blocks (>= 4× min_size) must be VERY flat to stay unsplit.
    # This forces fine detail in faces / edges.
    if size >= min_size * 4:
        effective_thresh *= 0.35

    if var <= effective_thresh:
        blocks.append((x, y, size, size))
        return

    half = size // 2
    _adaptive_split(img, x,        y,        half, min_size, thresh, blocks)
    _adaptive_split(img, x + half, y,        half, min_size, thresh, blocks)
    _adaptive_split(img, x,        y + half, half, min_size, thresh, blocks)
    _adaptive_split(img, x + half, y + half, half, min_size, thresh, blocks)


def adaptive_quadtree(img, min_size, max_size, thresh):
    H, W, _ = img.shape
    usable_W = (W // max_size) * max_size
    usable_H = (H // max_size) * max_size

    blocks = []
    for yy in range(0, usable_H, max_size):
        for xx in range(0, usable_W, max_size):
            _adaptive_split(img, xx, yy, max_size, min_size, thresh, blocks)
    return blocks


def get_blocks(img):
    min_size = config.TILE_SIZE_MIN
    max_size = config.TILE_SIZE_MAX
    thresh   = config.SPLIT_VARIANCE_THRESH

    blocks = adaptive_quadtree(img, min_size, max_size, thresh)

    sizes = {}
    for (_, _, w, _) in blocks:
        sizes[w] = sizes.get(w, 0) + 1
    size_str = "  ".join(f"{s}px×{n}" for s, n in sorted(sizes.items(), reverse=True))
    print(f"[search] adaptive quadtree: {len(blocks)} blocks — {size_str}")
    return blocks


# ---------------------------------------------------------------------------
# KDTree
# ---------------------------------------------------------------------------

def build_kdtrees(emb3, emb4):
    print("[search] building KDTrees...")
    tree3 = KDTree(emb3)
    tree4 = KDTree(emb4)
    print("[search] KDTrees ready")
    return tree3, tree4


# ---------------------------------------------------------------------------
# Color transfer
# ---------------------------------------------------------------------------

def color_transfer(tile, block_mean):
    if not config.COLOR_TRANSFER:
        return tile
    alpha     = config.COLOR_TRANSFER_ALPHA
    tile_f    = tile.astype(np.float32)
    tile_mean = tile_f.reshape(-1, 3).mean(axis=0)
    shift     = (block_mean - tile_mean) * alpha
    return np.clip(tile_f + shift, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tile matching
# ---------------------------------------------------------------------------

def find_best_tile(block, block_emb3, block_emb4,
                   tree3, tree4, tile_cache, tile_means,
                   use_count, chosen_grid, gi, gj, gh, gw):

    block_mean = block.astype(np.float32).reshape(-1, 3).mean(axis=0)

    _, idx3 = tree3.query(block_emb3[None, :], k=config.TOPK)
    _, idx4 = tree4.query(block_emb4[None, :], k=config.TOPK)
    idx3, idx4 = idx3[0], idx4[0]

    cand = np.unique(np.concatenate([idx3, idx4]))

    d3  = np.linalg.norm(tree3.data[cand] - block_emb3[None, :], axis=1)
    d4  = np.linalg.norm(tree4.data[cand] - block_emb4[None, :], axis=1)
    col = np.sum((tile_means[cand] - block_mean[None, :]) ** 2, axis=1)

    d3n  = d3  / (float(np.mean(d3))  + 1e-8)
    d4n  = d4  / (float(np.mean(d4))  + 1e-8)
    coln = col / (float(np.mean(col)) + 1e-8)

    score     = config.W3 * d3n + config.W4 * d4n + config.W_COLOR * coln
    cand_use  = use_count[cand].astype(np.float32)
    score     = score * np.exp(config.PENALTY * cand_use)

    allowed = cand_use < config.MAX_USE
    if np.any(allowed):
        cand2, score2 = cand[allowed], score[allowed]
    else:
        cand2, score2 = cand, score

    i0 = max(0, gi - config.LOCAL_RADIUS)
    i1 = min(gh, gi + config.LOCAL_RADIUS + 1)
    j0 = max(0, gj - config.LOCAL_RADIUS)
    j1 = min(gw, gj + config.LOCAL_RADIUS + 1)
    banned = set(chosen_grid[i0:i1, j0:j1].ravel().tolist())
    banned.discard(-1)

    if banned:
        keep = np.array([c not in banned for c in cand2], dtype=bool)
        if np.any(keep):
            cand2, score2 = cand2[keep], score2[keep]

    kk    = min(10, len(cand2))
    best  = np.argpartition(score2, kk - 1)[:kk]
    chosen = int(np.random.default_rng().choice(cand2[best]))

    use_count[chosen] += 1
    return chosen
