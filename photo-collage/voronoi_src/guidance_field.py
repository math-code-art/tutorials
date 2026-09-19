import os
import cv2
import numpy as np
from PIL import Image


def compute_sobel_guidance(
    target_img: np.ndarray,
    blur_sigma: float = 8.0,
    sobel_ksize: int = 3,
    eps: float = 1e-8,
):
    """
    Build a smooth unoriented guidance field from Sobel derivatives.

    Important:
    - Sobel gives local image gradients.
    - Raw per-pixel gradients are too noisy for cell-scale guidance.
    - We therefore smooth the 2x2 structure tensor built from gx, gy.
    - The dominant tensor direction gives a smooth normal line field.
    - Rotating it by 90 degrees gives the tangent line field.

    Returns normal/tangent fields with shape (H,W,2) and a confidence map.
    """
    if target_img.ndim == 3:
        gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = target_img.copy()

    gray = gray.astype(np.float32) / 255.0

    # A small image pre-smoothing suppresses single-pixel noise while leaving
    # the larger orientation smoothing to the structure tensor below.
    pre_sigma = max(0.8, float(blur_sigma) * 0.25)
    gray_smooth = cv2.GaussianBlur(
        gray,
        ksize=(0, 0),
        sigmaX=pre_sigma,
        sigmaY=pre_sigma,
    )

    gx = cv2.Sobel(gray_smooth, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_smooth, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    magnitude = np.sqrt(gx * gx + gy * gy)

    # Smoothed structure tensor:
    #     J = [[G*(gx^2), G*(gx*gy)],
    #          [G*(gx*gy), G*(gy^2)]]
    # This is the orientation filtering step. It treats theta and theta+pi
    # as the same line orientation, which is what we want for elongated cells.
    sigma = max(float(blur_sigma), 0.8)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Dominant gradient/normal orientation of a 2x2 symmetric tensor.
    theta_normal = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)

    nx = np.cos(theta_normal)
    ny = np.sin(theta_normal)
    normal = np.stack([nx, ny], axis=-1).astype(np.float32)

    # Tangent follows the local image structure/contour.
    tangent = np.stack([-ny, nx], axis=-1).astype(np.float32)

    # Tensor coherence: 0 -> locally isotropic/no reliable direction,
    # 1 -> one dominant local direction.
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    energy = jxx + jyy
    coherence = anisotropy / (energy + eps)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)

    # Also downweight very weak/flat image regions.
    strength = np.sqrt(np.maximum(energy, 0.0))
    scale = float(np.percentile(strength, 95))
    if scale < eps:
        strength_norm = np.zeros_like(strength, dtype=np.float32)
    else:
        strength_norm = np.clip(strength / scale, 0.0, 1.0).astype(np.float32)

    confidence = (coherence * strength_norm).astype(np.float32)

    return {
        "gray": gray_smooth.astype(np.float32),
        "gx": gx.astype(np.float32),
        "gy": gy.astype(np.float32),
        "magnitude": magnitude.astype(np.float32),
        "coherence": coherence,
        "normal": normal,
        "tangent": tangent,
        "confidence": confidence,
    }


def diffuse_direction_field(
    normal: np.ndarray,
    confidence: np.ndarray,
    magnitude: np.ndarray,
    diffuse_radius_px: float = 40.0,
    edge_stop_k: float = 0.12,
    downsample: int = 4,
    diffuse_step: float = 0.2,
    eps: float = 1e-8,
):
    """
    Propagate direction from confident (real-edge) pixels into their
    surrounding low-confidence (flat/textureless) interior -- but WITHOUT
    crossing real edges. Plain (isotropic) blurring can't do this: it
    spreads a strand's direction just as easily across a genuine boundary
    into the neighboring, differently-oriented strand, which dilutes the
    very edges we want to stay sharp.

    Instead this is edge-STOPPING (anisotropic) diffusion: at each pixel we
    define a "conductivity" from the local Sobel gradient magnitude --
    near 1 in flat regions (diffusion flows freely, filling the interior of
    a strand/clump with a direction inherited from its own boundary) and
    near 0 at strong edges (diffusion is blocked, so direction doesn't leak
    across into a differently-oriented neighboring region). This is the same
    idea as "Edge Tangent Flow" used in flow-based line-drawing / hatching
    algorithms.

    Direction is a LINE field (theta and theta+pi are equivalent), so we
    diffuse it via the standard double-angle trick (confidence-weighted
    exp(i*2*theta)) rather than diffusing raw vectors, which would
    incorrectly cancel opposing-but-equivalent directions to zero.

    Runs on a downsampled grid for speed (direction doesn't need full-res
    diffusion precision), then upsamples back.
    """
    H, W = confidence.shape
    ds = max(1, int(downsample))
    small_h, small_w = max(1, H // ds), max(1, W // ds)

    theta = np.arctan2(normal[..., 1], normal[..., 0])
    w = confidence.astype(np.float64)
    wc2 = w * np.cos(2.0 * theta)
    ws2 = w * np.sin(2.0 * theta)

    mag_scale = float(np.percentile(magnitude, 95)) + eps
    mag_norm = np.clip(magnitude / mag_scale, 0.0, 1.0)
    conductivity = 1.0 / (1.0 + (mag_norm / max(float(edge_stop_k), 1e-4)) ** 2)

    def _down(a):
        return cv2.resize(
            a.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA
        ).astype(np.float64)

    def _up(a):
        return cv2.resize(
            a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        ).astype(np.float64)

    wc2_s, ws2_s, w_s, cond_s = _down(wc2), _down(ws2), _down(w), _down(conductivity)

    def _shift(a, dy, dx):
        p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
        h, ww = a.shape
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + ww]

    # Effective diffusion reaches roughly diffuse_radius_px in flat regions
    # (conductivity ~= 1); real edges (conductivity ~= 0) block it almost
    # entirely regardless of iteration count.
    small_radius = max(1.0, float(diffuse_radius_px) / ds)
    iters = max(1, int((small_radius ** 2) / (4.0 * diffuse_step)))

    c_up = 0.5 * (cond_s + _shift(cond_s, -1, 0))
    c_down = 0.5 * (cond_s + _shift(cond_s, 1, 0))
    c_left = 0.5 * (cond_s + _shift(cond_s, 0, -1))
    c_right = 0.5 * (cond_s + _shift(cond_s, 0, 1))

    def _step(field):
        flux = (
            c_up * (_shift(field, -1, 0) - field)
            + c_down * (_shift(field, 1, 0) - field)
            + c_left * (_shift(field, 0, -1) - field)
            + c_right * (_shift(field, 0, 1) - field)
        )
        return field + diffuse_step * flux

    for _ in range(iters):
        wc2_s = _step(wc2_s)
        ws2_s = _step(ws2_s)
        w_s = _step(w_s)

    wc2_d, ws2_d, w_d = _up(wc2_s), _up(ws2_s), _up(w_s)

    theta_d = 0.5 * np.arctan2(ws2_d, wc2_d)
    mag = np.sqrt(wc2_d ** 2 + ws2_d ** 2)
    confidence_d = np.clip(mag / (w_d + eps), 0.0, 1.0).astype(np.float32)

    nx = np.cos(theta_d)
    ny = np.sin(theta_d)
    normal_d = np.stack([nx, ny], axis=-1).astype(np.float32)
    tangent_d = np.stack([-ny, nx], axis=-1).astype(np.float32)

    return {
        "normal": normal_d,
        "tangent": tangent_d,
        "confidence": confidence_d,
    }


def save_guidance_overlay(
    target_img: np.ndarray,
    vector_field: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    stride: int = 30,
    line_length: int = 22,
    min_confidence: float = 0.08,
):
    """Draw the filtered unoriented guidance lines over the target image."""
    vis = target_img.copy()
    H, W = confidence.shape
    half = line_length / 2.0

    for y in range(stride // 2, H, stride):
        for x in range(stride // 2, W, stride):
            if confidence[y, x] < min_confidence:
                continue

            vx = float(vector_field[y, x, 0])
            vy = float(vector_field[y, x, 1])
            norm = np.hypot(vx, vy)
            if norm < 1e-8:
                continue
            vx /= norm
            vy /= norm

            x1 = int(round(x - half * vx))
            y1 = int(round(y - half * vy))
            x2 = int(round(x + half * vx))
            y2 = int(round(y + half * vy))

            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    Image.fromarray(vis.astype(np.uint8)).save(output_path, quality=92)
    print(f"[guidance] saved: {output_path}")