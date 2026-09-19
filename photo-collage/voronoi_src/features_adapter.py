"""
features_adapter.py
===================

Thin wrapper that reuses your existing VGGEmbedder (from src/features.py)
to embed a single Voronoi cell region.

Why a separate file?
--------------------
voronoi_pipeline.py needs to embed arbitrary-sized cell crops on-the-fly.
Your existing code already does this — we just wire it up here so
voronoi_pipeline.py doesn't need to import from 'src' directly.

The embedder is a module-level singleton so the VGG model is loaded only
once per process (loading takes ~1 s and uses ~500 MB of RAM).
"""

import sys
import os
import numpy as np

# Allow importing from the parent project's src/ directory.
# Assumes the mosaic_project folder sits alongside the original src/ folder.
_here        = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))   # two levels up

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Lazy singleton so the GPU/MPS model is loaded only once --------------
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            import config
            from src.features import VGGEmbedder
            cut = getattr(config, "CUT3", 10)
            _embedder = VGGEmbedder(cut)
            print(f"[features_adapter] VGGEmbedder loaded (cut={cut})")
        except Exception as e:
            print(f"[features_adapter] WARNING: could not load VGGEmbedder: {e}")
            _embedder = _FallbackEmbedder()
    return _embedder


class _FallbackEmbedder:
    """
    Used when the original VGGEmbedder cannot be loaded (e.g. running
    without the parent project on path).  Returns a mean-colour feature
    vector instead, which degrades gracefully.
    """
    def embed(self, img_array: np.ndarray) -> np.ndarray:
        return img_array.astype(np.float32).reshape(-1, 3).mean(axis=0)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def get_cell_vgg_embedding(cell_region: np.ndarray) -> np.ndarray:
    """
    Compute a VGG embedding for a Voronoi cell region.

    Parameters
    ----------
    cell_region : H' × W' × 3  uint8 (any size — VGGEmbedder resizes internally)

    Returns
    -------
    embedding : (D,) float32
    """
    embedder = _get_embedder()
    return embedder.embed(cell_region).astype(np.float32)
