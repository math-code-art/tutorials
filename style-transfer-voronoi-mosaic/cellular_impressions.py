import argparse
import os
import json
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.ndimage import sobel, gaussian_filter
from scipy.spatial import Delaunay, cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def require_file(path: str, label: str) -> None:
    if path is None:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def pil_open_rgb(path: str) -> Image.Image:
    require_file(path, "Image file")
    return Image.open(path).convert("RGB")


def pil_open_mask(path: str, size_hw=None) -> np.ndarray:
    require_file(path, "Mask file")
    m = Image.open(path).convert("L")
    if size_hw is not None:
        m = m.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    arr = np.array(m, dtype=np.float32) / 255.0
    return arr > 0.5


def save_rgb(arr_uint8: np.ndarray, path: str) -> None:
    Image.fromarray(arr_uint8).save(path)


def save_gray01(gray01: np.ndarray, path: str) -> None:
    x = np.clip(gray01, 0.0, 1.0)
    im = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(im, mode="L").save(path)


def resize_longest_side(img: Image.Image, longest: int) -> Image.Image:
    w, h = img.size
    if max(w, h) == longest:
        return img
    scale = float(longest) / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    return T.ToTensor()(img).unsqueeze(0).to(device)


def to_uint8(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w + 1e-12)


class VGGFeatures(nn.Module):
    def __init__(self, content_layer="conv4_2", style_layers=None):
        super().__init__()
        if style_layers is None:
            style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.vgg = vgg
        self.content_layer = content_layer
        self.style_layers = set(style_layers)

        self.name_map = {}
        conv_idx = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        block = 1
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.Conv2d):
                conv_idx[block] += 1
                self.name_map[i] = f"conv{block}_{conv_idx[block]}"
            elif isinstance(layer, nn.ReLU):
                self.name_map[i] = "relu"
            elif isinstance(layer, nn.MaxPool2d):
                self.name_map[i] = "pool"
                block += 1
            else:
                self.name_map[i] = f"layer_{i}"

        self.keep = set(style_layers + [content_layer])

    def forward(self, x: torch.Tensor):
        out = {}
        block = 1
        conv_in_block = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                conv_in_block += 1
                nm = f"conv{block}_{conv_in_block}"
                if nm in self.keep:
                    out[nm] = x
            elif isinstance(layer, nn.MaxPool2d):
                block += 1
                conv_in_block = 0
        return out


@dataclass
class NSTConfig:
    content_weight: float = 1.0
    style_weight: float = 2e5
    tv_weight: float = 1e-6
    steps: int = 220
    lr: float = 0.03
    content_layer: str = "conv4_2"
    style_layers: tuple = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
    use_fp16: bool = False


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())


def run_style_transfer(
    content_img: Image.Image,
    style_img: Image.Image,
    device: torch.device,
    stages_longest,
    cfg: NSTConfig
) -> torch.Tensor:
    if len(stages_longest) == 0:
        raise ValueError("stages_longest must have at least one value")

    vggf = VGGFeatures(content_layer=cfg.content_layer, style_layers=list(cfg.style_layers)).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    autocast_ok = (device.type == "cuda") and cfg.use_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_ok)

    img_out = None

    for L in stages_longest:
        cL = resize_longest_side(content_img, L)
        sL = resize_longest_side(style_img, L)
        c = to_tensor(cL, device)
        s = to_tensor(sL, device)

        c_n = (c - mean) / std
        s_n = (s - mean) / std

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=autocast_ok):
                c_feats = vggf(c_n)
                s_feats = vggf(s_n)
                target_content = c_feats[cfg.content_layer].detach()
                target_style_grams = {k: gram_matrix(v).detach() for k, v in s_feats.items()}

        if img_out is None:
            img_out = c.clone()
        else:
            prev = Image.fromarray(to_uint8(img_out))
            img_out = to_tensor(resize_longest_side(prev, L), device)

        img_out = img_out.clamp(0, 1).detach().requires_grad_(True)
        opt = torch.optim.Adam([img_out], lr=cfg.lr)

        for _ in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=autocast_ok):
                x = img_out.clamp(0, 1)
                x_n = (x - mean) / std
                feats = vggf(x_n)

                c_loss = F.mse_loss(feats[cfg.content_layer], target_content)

                s_loss = 0.0
                for k in cfg.style_layers:
                    g = gram_matrix(feats[k])
                    s_loss = s_loss + F.mse_loss(g, target_style_grams[k])
                s_loss = s_loss / float(len(cfg.style_layers))

                tv = total_variation(x)
                loss = cfg.content_weight * c_loss + cfg.style_weight * s_loss + cfg.tv_weight * tv

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        img_out = img_out.detach()

    return img_out.clamp(0, 1)

def build_density_map(rgb_uint8: np.ndarray, favor="mix", w_edge=0.85, w_lum=0.15) -> np.ndarray:
    img = rgb_uint8.astype(np.float32) / 255.0
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    gx = sobel(lum, axis=1, mode="reflect")
    gy = sobel(lum, axis=0, mode="reflect")
    edge = np.hypot(gx, gy)
    edge = edge / (edge.max() + 1e-12)

    lum_n = (lum - lum.min()) / (lum.max() - lum.min() + 1e-12)

    if favor == "edge":
        density = edge
    elif favor == "bright":
        density = lum_n
    elif favor == "dark":
        density = 1.0 - lum_n
    else:
        density = w_edge * edge + w_lum * lum_n

    density = np.clip(density, 1e-12, None)
    return density


def sample_points_from_pdf(pdf2d: np.ndarray, n: int, rng: np.random.Generator, mask: np.ndarray | None) -> np.ndarray:
    h, w = pdf2d.shape
    p = pdf2d.copy()
    if mask is not None:
        p = p * mask.astype(np.float32)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()

    idx = rng.choice(p.size, size=n, replace=True, p=p.ravel())
    ys, xs = np.divmod(idx, w)

    xs = xs.astype(np.float32) + rng.random(n).astype(np.float32)
    ys = ys.astype(np.float32) + rng.random(n).astype(np.float32)

    return np.column_stack([xs, ys]).astype(np.float32)


def sample_points_adaptive(
    density: np.ndarray,
    n_coarse: int,
    n_fine: int,
    rng: np.random.Generator,
    mask: np.ndarray | None,
    blur_sigma: float,
    density_power: float
) -> np.ndarray:

    d = density.astype(np.float32)
    d = d / (d.max() + 1e-12)

    d_blur = gaussian_filter(d, sigma=max(0.0, blur_sigma)) if blur_sigma > 0 else d
    d_blur = np.clip(d_blur, 1e-12, None)

    d_fine = np.clip(d, 1e-12, None) ** max(0.1, float(density_power))

    pts_coarse = sample_points_from_pdf(d_blur, n_coarse, rng, mask)
    pts_fine = sample_points_from_pdf(d_fine, n_fine, rng, mask)

    pts = np.concatenate([pts_coarse, pts_fine], axis=0)
    return pts


def jitter_and_clip(points_xy: np.ndarray, w: int, h: int, rng: np.random.Generator, jitter: float) -> np.ndarray:
    pts = points_xy.copy()
    if jitter > 0:
        pts += rng.normal(0.0, jitter, size=pts.shape).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1e-3))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1e-3))
    return pts


def voronoi_label_map(points_xy: np.ndarray, w: int, h: int, chunk: int = 200000) -> np.ndarray:
    tree = cKDTree(points_xy.astype(np.float32))
    xs = np.arange(w, dtype=np.int32)
    ys = np.arange(h, dtype=np.int32)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    labels = np.empty((coords.shape[0],), dtype=np.int32)
    for i in range(0, coords.shape[0], chunk):
        j = min(i + chunk, coords.shape[0])
        _, idx = tree.query(coords[i:j], k=1)
        labels[i:j] = idx.astype(np.int32)

    return labels.reshape(h, w)


def render_voronoi_mean_color(rgb_uint8: np.ndarray, points_xy: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    h, w = rgb_uint8.shape[:2]
    labels = voronoi_label_map(points_xy, w, h)

    out = np.zeros_like(rgb_uint8)
    img = rgb_uint8.reshape(-1, 3).astype(np.float32)
    lab = labels.reshape(-1)

    if mask is not None:
        m = mask.reshape(-1).astype(bool)
    else:
        m = None

    n_sites = points_xy.shape[0]
    counts = np.zeros((n_sites,), dtype=np.float32)
    sums = np.zeros((n_sites, 3), dtype=np.float32)

    if m is None:
        np.add.at(counts, lab, 1.0)
        np.add.at(sums, lab, img)
    else:
        lab_m = lab[m]
        img_m = img[m]
        np.add.at(counts, lab_m, 1.0)
        np.add.at(sums, lab_m, img_m)

    means = sums / (counts[:, None] + 1e-12)
    means_u8 = np.clip(means, 0.0, 255.0).astype(np.uint8)

    out_flat = out.reshape(-1, 3)
    if m is None:
        out_flat[:] = means_u8[lab]
    else:
        out_flat[:] = rgb_uint8.reshape(-1, 3)
        out_flat[m] = means_u8[lab[m]]

    return out


def render_tri_mosaic(rgb_uint8: np.ndarray, points_xy: np.ndarray, mask: np.ndarray | None, edge_width: float = 0.0) -> np.ndarray:
    h, w = rgb_uint8.shape[:2]
    tri = Delaunay(points_xy)

    pts = points_xy
    px = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    vcol = rgb_uint8[py, px].astype(np.float32) / 255.0

    polys = pts[tri.simplices]
    facecols = vcol[tri.simplices].mean(axis=1)

    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    if mask is None:
        pc = PolyCollection(
            polys,
            facecolors=facecols,
            edgecolors="none" if edge_width <= 0 else (0, 0, 0, 1),
            linewidths=edge_width,
            antialiased=True
        )
        ax.add_collection(pc)
    else:
        keep = []
        keep_cols = []
        for poly, fc in zip(polys, facecols):
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
            xi = int(np.clip(round(cx), 0, w - 1))
            yi = int(np.clip(round(cy), 0, h - 1))
            if mask[yi, xi]:
                keep.append(poly)
                keep_cols.append(fc)
        if len(keep) > 0:
            pc = PolyCollection(
                np.array(keep),
                facecolors=np.array(keep_cols),
                edgecolors="none" if edge_width <= 0 else (0, 0, 0, 1),
                linewidths=edge_width,
                antialiased=True
            )
            ax.add_collection(pc)

    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    out = out.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3)
    plt.close(fig)

    if out.shape[0] != h or out.shape[1] != w:
        out = np.array(Image.fromarray(out).resize((w, h), Image.LANCZOS))

    return out


def overlay_points(rgb_uint8: np.ndarray, points_xy: np.ndarray, radius: int = 2) -> np.ndarray:
    out = rgb_uint8.copy()
    h, w = out.shape[:2]
    for x, y in points_xy:
        xi = int(round(x))
        yi = int(round(y))
        x0 = max(0, xi - radius)
        x1 = min(w, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(h, yi + radius + 1)
        out[y0:y1, x0:x1] = 255
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--content", required=True, help="Path to portrait image")
    ap.add_argument("--style", required=True, help="Path to style image")
    ap.add_argument("--mask", default=None, help="Optional mask image (white=inside)")
    ap.add_argument("--outdir", default="outputs_cellular_impressions_v2")
    ap.add_argument("--seed", type=int, default=2)

    # NST
    ap.add_argument("--skip_nst", action="store_true", help="Skip NST and mosaic the content image directly")
    ap.add_argument("--stages", default="768,1024,1536", help="Comma list of longest-side sizes for progressive NST")
    ap.add_argument("--steps", type=int, default=220, help="Steps per NST stage")
    ap.add_argument("--style_weight", type=float, default=2e5)
    ap.add_argument("--content_weight", type=float, default=1.0)
    ap.add_argument("--tv_weight", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--fp16", action="store_true")

    # Adaptive sampling
    ap.add_argument("--favor", default="mix", choices=["edge", "bright", "dark", "mix"])
    ap.add_argument("--n_coarse", type=int, default=1200, help="Coarse points, makes big cells in smooth regions")
    ap.add_argument("--n_fine", type=int, default=7000, help="Fine points, makes small cells in detailed regions")
    ap.add_argument("--blur_sigma", type=float, default=6.0, help="Blur for coarse sampling density")
    ap.add_argument("--density_power", type=float, default=2.0, help="Power for fine sampling, higher means more focus on edges")
    ap.add_argument("--point_jitter", type=float, default=0.0, help="Small Gaussian jitter on points, optional")

    # Mosaic modes
    ap.add_argument("--mode", default="voronoi", choices=["voronoi", "tri"], help="voronoi = mean-cell-color fix, tri = triangulation")
    ap.add_argument("--edge_width", type=float, default=0.0, help="Only used for tri mode")

    args = ap.parse_args()

    ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)

    require_file(args.content, "Content image")
    require_file(args.style, "Style image")
    if args.mask is not None:
        require_file(args.mask, "Mask image")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    content_pil = pil_open_rgb(args.content)
    style_pil = pil_open_rgb(args.style)
    stages = [int(x.strip()) for x in args.stages.split(",") if x.strip()]

    # NST or direct
    if args.skip_nst:
        stylized_t = to_tensor(content_pil, device).clamp(0, 1)
    else:
        cfg = NSTConfig(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            steps=args.steps,
            lr=args.lr,
            use_fp16=args.fp16
        )
        stylized_t = run_style_transfer(content_pil, style_pil, device, stages, cfg)

    stylized_uint8 = to_uint8(stylized_t)
    h, w = stylized_uint8.shape[:2]

    # Mask
    mask = None
    if args.mask is not None:
        mask = pil_open_mask(args.mask, size_hw=(h, w))
        print("Mask loaded:", args.mask)

    stylized_path = os.path.join(args.outdir, "01_stylized.png")
    save_rgb(stylized_uint8, stylized_path)
    print("Saved:", stylized_path)

    # Density + points
    density = build_density_map(stylized_uint8, favor=args.favor, w_edge=0.85, w_lum=0.15)
    density_path = os.path.join(args.outdir, "02_density.png")
    save_gray01(density / (density.max() + 1e-12), density_path)
    print("Saved:", density_path)

    pts = sample_points_adaptive(
        density=density,
        n_coarse=args.n_coarse,
        n_fine=args.n_fine,
        rng=rng,
        mask=mask,
        blur_sigma=args.blur_sigma,
        density_power=args.density_power
    )
    pts = jitter_and_clip(pts, w=w, h=h, rng=rng, jitter=args.point_jitter)

    pts_overlay = overlay_points(stylized_uint8, pts, radius=1)
    pts_overlay_path = os.path.join(args.outdir, "03_points_overlay.png")
    save_rgb(pts_overlay, pts_overlay_path)
    print("Saved:", pts_overlay_path)

    # Mosaic render
    if args.mode == "voronoi":
        mosaic_uint8 = render_voronoi_mean_color(stylized_uint8, pts, mask=mask)
    else:
        mosaic_uint8 = render_tri_mosaic(stylized_uint8, pts, mask=mask, edge_width=args.edge_width)

    mosaic_path = os.path.join(args.outdir, "04_mosaic.png")
    save_rgb(mosaic_uint8, mosaic_path)
    print("Saved:", mosaic_path)

    # Create Side-by-side
    grid = np.zeros((h, w * 2, 3), dtype=np.uint8)
    grid[:, :w] = stylized_uint8
    grid[:, w:] = mosaic_uint8
    grid_path = os.path.join(args.outdir, "05_side_by_side.png")
    save_rgb(grid, grid_path)
    print("Saved:", grid_path)

    # Save points + config
    pts_path = os.path.join(args.outdir, "points_xy.npy")
    np.save(pts_path, pts)
    print("Saved:", pts_path)

    cfg_path = os.path.join(args.outdir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print("Saved:", cfg_path)


if __name__ == "__main__":
    main()
