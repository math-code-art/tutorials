"""
方向2: Composite Density Map
============================
Combines four perceptual cues into a single density map that drives
seed placement.  Bright pixels → more / smaller tiles.

    μ(x) = α·edge(x) + β·saliency(x) + γ·texture(x) + δ·color(x)

Each component is independently normalised to [0, 1] before blending.
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Individual components
# ---------------------------------------------------------------------------

def _edge_map(gray: np.ndarray) -> np.ndarray:
    """
    Sobel gradient magnitude.  Captures hard boundaries: face outline,
    eyes, text, object silhouettes.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Mild Gaussian to avoid isolated noise pixels dominating the map
    mag = cv2.GaussianBlur(mag.astype(np.float32), (5, 5), 1.0)
    return mag


def _saliency_map(bgr: np.ndarray) -> np.ndarray:
    """
    Spectral Residual saliency (Hou & Zhang 2007).
    Works well on natural images and requires no neural network.
    
    The log-spectrum of the image is approximated; deviations from
    the average log-spectrum correspond to salient regions.
    """
    # Use OpenCV's built-in saliency if available, else fallback to
    # a simple frequency-domain implementation.
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = saliency.computeSaliency(bgr)
        if ok:
            return sal.astype(np.float32)
    except AttributeError:
        pass

    # ── Fallback: manual spectral residual ──────────────────────────────
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w   = gray.shape
    # Resize for speed
    small  = cv2.resize(gray, (64, 64))
    log_am = np.log(np.abs(np.fft.fft2(small)) + 1e-8)
    # Residual: log spectrum minus smoothed version
    avg    = uniform_filter(log_am, size=3)
    res    = log_am - avg
    sal    = np.abs(np.fft.ifft2(np.exp(res + 1j * np.angle(np.fft.fft2(small))))) ** 2
    sal    = cv2.GaussianBlur(sal.astype(np.float32), (5, 5), 0)
    sal    = cv2.resize(sal, (w, h))
    return sal


def _texture_map(gray: np.ndarray, kernel: int = 15) -> np.ndarray:
    """
    Local standard deviation — high where texture is complex
    (hair, fabric, foliage, hatching).

    We compute E[X²] - E[X]² over a sliding window.
    """
    gray_f  = gray.astype(np.float32)
    mean    = uniform_filter(gray_f,    size=kernel)
    mean_sq = uniform_filter(gray_f**2, size=kernel)
    var     = np.maximum(0.0, mean_sq - mean**2)
    return np.sqrt(var)


def _color_variation_map(lab: np.ndarray, kernel: int = 15) -> np.ndarray:
    """
    Local colour variance in LAB space.
    Captures colour boundaries missed by luminance-only edge detectors
    (e.g. red text on green background with same luminance).
    """
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    def _local_var(ch):
        m  = uniform_filter(ch,    size=kernel)
        m2 = uniform_filter(ch**2, size=kernel)
        return np.maximum(0.0, m2 - m**2)

    return np.sqrt(_local_var(l) + _local_var(a) + _local_var(b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _norm(x: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]; returns zeros if constant."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def compute_density_map(
    img_rgb: np.ndarray,
    alpha: float = 0.35,   # edge weight
    beta:  float = 0.25,   # saliency weight
    gamma: float = 0.20,   # texture weight
    delta: float = 0.20,   # colour variation weight
    blur_sigma: float = 3.0,
    min_density: float = 0.05,
) -> np.ndarray:
    """
    Parameters
    ----------
    img_rgb      : H × W × 3  uint8 RGB image
    alpha..delta : blend weights (must sum to 1 for interpretability)
    blur_sigma   : final Gaussian sigma to spatially smooth the map
    min_density  : floor value — ensures even flat areas get *some* seeds

    Returns
    -------
    density : H × W  float32 in [min_density, 1.0]
    """
    bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    edge    = _norm(_edge_map(gray))
    sal     = _norm(_saliency_map(bgr))
    texture = _norm(_texture_map(gray))
    color   = _norm(_color_variation_map(lab))

    density = alpha * edge + beta * sal + gamma * texture + delta * color

    # Spatial smoothing so seed rejection-sampling works stably
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1   # odd kernel
        density = cv2.GaussianBlur(density, (ksize, ksize), blur_sigma)

    density = _norm(density)
    density = np.clip(density, min_density, 1.0)
    return density


def visualise_density(density: np.ndarray) -> np.ndarray:
    """Return a colour-mapped uint8 image (COLORMAP_INFERNO) for debugging."""
    d8 = (density * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
