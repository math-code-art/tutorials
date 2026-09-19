from __future__ import annotations

import json
import math
import os
import time
import numpy as np
import torch
from PIL import Image

from .density import compute_density_map, visualise_density
from .guidance_field import compute_sobel_guidance, save_guidance_overlay, diffuse_direction_field
from .matching import EnhancedMatcher
from .rendering import (
    extract_cell_polygons,
    render_voronoi_mosaic,
    render_voronoi_mosaic_print,
)
from .voronoi_seeds import initialise_voronoi, build_label_map
from .voroplex_2d_utils import (
    compute_topology_2d,
    reconstruct_cells_2d,
    polygon_statistics,
    points01_to_pixels,
    save_seed_overlay,
    save_label_visualization,
    save_cell_axis_overlay,
    save_cell_aspect_heatmap,
    save_cell_aspect_overlay,
)
from .voroplex_directional_loss import compute_directional_training_loss


DEFAULTS = dict(
    # Density-guided initial generator placement.
    density_alpha=0.35,
    density_beta=0.25,
    density_gamma=0.20,
    density_delta=0.20,
    n_initial_seeds=900,
    lloyd_iters_init=8,

    # Sobel guidance.
    guidance_blur_sigma=8.0,
    guidance_sobel_ksize=3,
    direction_diffuse_radius_px=40.0,  # how far direction can propagate INSIDE a flat region (real edges block it almost entirely regardless of this value -- see edge_stop_k)
    direction_edge_stop_k=0.12,        # smaller = edges act as a harder stop (less leakage across real boundaries); larger = softer, more like plain blurring
    guidance_mode="tangent",    # only controls which field is drawn in the 03_* overlay PNG; training always uses the "normal" field (see run(), step 4)

    # Differentiable VoroPlex training.
    train_iters=40,
    train_lr=1.0e-3,
    lr_min_factor=0.05,        # cosine-anneal the LR down to this fraction of train_lr after warmup
    voroplex_threads=1,

    # Fabric-tensor orientation (Kanatani tensor over cell boundary-edge
    # normals, matched to a local target axis) + isoperimetric roundness
    # regularizer -- see voroplex_directional_loss.py docstring.
    target_alpha=0.80,          # anisotropy strength in (0.5, 1.0); 0.5=isotropic
    target_roundness=0.7853981633974483,  # pi/4 -- "square" target, prevents degenerate slivers
    floor_roundness=0.30,       # hard guard: cells whose q falls below this get a steep, cubic penalty regardless of the averaged roundness_loss above (stops rare severe sliver outliers slipping through)
    confidence_gamma=1.0,        # <1 broadens which cells get real weight in the orientation loss (see loss module docstring)
    preferred_direction_deg=0.0,  # global "preferred direction" (paper Fig. 9 style: SAME target axis for every cell, no confidence weighting). Set lambda_preferred > 0 to use it; good as a cheap, eyeball-verifiable sanity check before trusting the local-Sobel-following result.
    preferred_alpha=0.80,         # anisotropy strength for the preferred-direction term specifically
    lambda_orientation=6.00,
    lambda_preferred=0.00,        # off by default; the local (Sobel-following) term is the main mechanism
    lambda_roundness=0.80,
    lambda_sliver_guard=4.0,
    lambda_size_uniform=0.15,
    lambda_area=0.15,
    lambda_separation=0.10,
    lambda_anchor=0.005,
    anchor_decay_floor=0.1,     # lambda_anchor cosine-decays to this fraction of itself
    lambda_edge_attract=0.00,   # pulls seeds themselves onto strong/coherent edges
    edge_attract_warmup_frac=0.25,
    min_separation_factor=0.25,
    grad_clip=2.0,
    print_every=10,

    # Existing tile matcher.
    w_lab=0.30,
    w_tex=0.20,
    w_vgg=0.40,
    w_pen=0.10,
    topk=40,
    max_use=6,
    penalty_scale=0.4,
    local_radius=3,

    # Existing renderer.
    color_transfer_alpha=0.40,
    grout_color=(20, 20, 20),
    grout_width=0,

    # Museum print master
    print_long_edge=24000,
    print_dpi=300,
)


class VoroPlexDirectionalMosaicPipeline:
    def __init__(
        self,
        tile_cache_small: np.ndarray,
        hires_cache: np.ndarray,
        emb_vgg: np.ndarray,
        tile_paths: list[str] | None = None,
        **kwargs,
    ):
        self.tile_cache_small = tile_cache_small
        self.hires_cache = hires_cache
        self.emb_vgg = emb_vgg
        self.tile_paths = tile_paths
        self.cfg = {**DEFAULTS, **kwargs}

    def run(
        self,
        target_img: np.ndarray,
        output_dir: str,
        save_intermediates: bool = True,
        rng_seed: int = 42,
        target_name: str = "target",
    ) -> np.ndarray:
        os.makedirs(output_dir, exist_ok=True)
        cfg = self.cfg
        rng = np.random.default_rng(rng_seed)
        t0 = time.time()

        if save_intermediates:
            self._save(target_img, output_dir, "01_target_preprocessed.jpg")

        print("\n[voroplex directional] step 1 - density map")
        density = compute_density_map(
            target_img,
            alpha=cfg["density_alpha"],
            beta=cfg["density_beta"],
            gamma=cfg["density_gamma"],
            delta=cfg["density_delta"],
        )
        if save_intermediates:
            self._save(visualise_density(density), output_dir, "02_density_map.jpg")

        print("\n[voroplex directional] step 2 - Sobel guidance field")
        guidance = compute_sobel_guidance(
            target_img,
            blur_sigma=cfg["guidance_blur_sigma"],
            sobel_ksize=cfg["guidance_sobel_ksize"],
        )
        mode = str(cfg["guidance_mode"]).lower()
        if mode not in {"tangent", "normal"}:
            raise ValueError("guidance_mode must be 'tangent' or 'normal'")
        overlay_field = guidance[mode]

        # Propagate direction from confident (real-edge) pixels into the
        # interior they enclose, WITHOUT crossing into a neighboring,
        # differently-oriented region -- edge-stopping (anisotropic)
        # diffusion using the Sobel gradient magnitude as the "conductivity"
        # map. This is what fills a strand/clump's interior with a coherent
        # direction while keeping real boundaries sharp. See
        # guidance_field.diffuse_direction_field docstring.
        diffused = diffuse_direction_field(
            guidance["normal"],
            guidance["confidence"],
            guidance["magnitude"],
            diffuse_radius_px=cfg["direction_diffuse_radius_px"],
            edge_stop_k=cfg["direction_edge_stop_k"],
        )
        # Orientation training targets the DIFFUSED field (fills interiors,
        # respects real edges). edge_attract (pulling seeds onto strong
        # edges specifically) keeps using the RAW, undiffused confidence --
        # that one is supposed to stay tight/local, not spread out.
        fabric_target_field = diffused["normal"]
        confidence_orient = diffused["confidence"]
        confidence_edge = guidance["confidence"]

        if save_intermediates:
            save_guidance_overlay(
                target_img,
                overlay_field,
                guidance["confidence"],
                os.path.join(output_dir, f"03_sobel_{mode}_guidance.jpg"),
            )
            save_guidance_overlay(
                target_img,
                diffused["tangent"] if mode == "tangent" else diffused["normal"],
                diffused["confidence"],
                os.path.join(output_dir, f"03b_diffused_{mode}_guidance.jpg"),
                min_confidence=0.03,
            )

        print("\n[voroplex directional] step 3 - density-guided initial generators")
        seeds_px, initial_label = initialise_voronoi(
            density,
            n_initial_seeds=cfg["n_initial_seeds"],
            lloyd_iters=cfg["lloyd_iters_init"],
            rng=rng,
        )
        if save_intermediates:
            save_seed_overlay(
                target_img,
                seeds_px,
                os.path.join(output_dir, "04_initial_seed_points.jpg"),
            )
            save_label_visualization(
                initial_label,
                os.path.join(output_dir, "05_initial_voronoi_label.jpg"),
            )

        print("\n[voroplex directional] step 4 - differentiable VoroPlex training")
        points01, history = self._optimize_generators(
            seeds_px=seeds_px,
            target_img=target_img,
            fabric_target_field=fabric_target_field,
            confidence_orient=confidence_orient,
            confidence_edge=confidence_edge,
        )

        H, W = target_img.shape[:2]
        seeds_opt_px = points01_to_pixels(points01, H, W)
        label_map = build_label_map(seeds_opt_px, H, W)

        if save_intermediates:
            save_seed_overlay(
                target_img,
                seeds_opt_px,
                os.path.join(output_dir, "06_optimized_seed_points.jpg"),
            )
            save_label_visualization(
                label_map,
                os.path.join(output_dir, "07_directional_voronoi_label.jpg"),
            )
            save_cell_axis_overlay(
                target_img,
                label_map,
                os.path.join(output_dir, "08_final_cell_major_axes.jpg"),
            )
            # Convert target_alpha (eigenvalue-space anisotropy strength) to
            # an approximate axis-length ratio for the heatmap's color scale:
            # eigenvalue ratio = alpha/(1-alpha), axis-length ratio ~ sqrt of that.
            _alpha = float(cfg["target_alpha"])
            _ratio_cap = max(1.5, (( _alpha / max(1e-3, 1.0 - _alpha)) ** 0.5))
            save_cell_aspect_heatmap(
                target_img,
                label_map,
                os.path.join(output_dir, "07b_cell_aspect_heatmap.jpg"),
                ratio_cap=_ratio_cap,
            )
            # The real check: does elongation actually track the fur/edges in
            # the photo, not just "is the average ratio high". Heatmap blended
            # over the real image + each elongated cell's own axis line drawn
            # on top -- if it's working, bright regions and axis lines should
            # visibly follow the mane/fur strands underneath.
            save_cell_aspect_overlay(
                target_img,
                label_map,
                os.path.join(output_dir, "07c_cell_aspect_overlay.jpg"),
                ratio_cap=_ratio_cap,
            )

        print("\n[voroplex directional] step 5 - existing appearance matcher")
        matcher = EnhancedMatcher(
            tile_cache_small=self.tile_cache_small,
            emb_vgg=self.emb_vgg,
            w_lab=cfg["w_lab"],
            w_tex=cfg["w_tex"],
            w_vgg=cfg["w_vgg"],
            w_pen=cfg["w_pen"],
            topk=cfg["topk"],
            max_use=cfg["max_use"],
            penalty_scale=cfg["penalty_scale"],
            local_radius=cfg["local_radius"],
        )
        cell_tile_indices = self._match_all_cells(
            target_img,
            label_map,
            len(seeds_opt_px),
            matcher,
        )

        print("\n[voroplex directional] step 6 - render matched tiles")
        polygons = extract_cell_polygons(label_map, len(cell_tile_indices))
        mosaic = render_voronoi_mosaic(
            target_img=target_img,
            hires_cache=self.hires_cache,
            label_map=label_map,
            cell_tile_indices=cell_tile_indices,
            polygons=polygons,
            color_transfer_alpha=cfg["color_transfer_alpha"],
            grout_color=cfg["grout_color"],
            grout_width=cfg["grout_width"],
        )

        self._save(mosaic, output_dir, "09_voroplex_directional_final.jpg", quality=95)
        direct_path = os.path.join(
            os.path.dirname(output_dir),
            f"{target_name}_voroplex_directional_final.jpg",
        )
        Image.fromarray(mosaic.astype(np.uint8)).save(direct_path, quality=95)
        print(f"[save] {direct_path}")

        # ------------------------------------------------------------
        # Step 7 - high-resolution print render
        # ------------------------------------------------------------

        if (
            getattr(self, "tile_paths", None)
            and os.path.isfile(self.tile_paths[0])
        ):
            print(
                "\n[voroplex directional] "
                "step 7 - high-resolution print render"
            )

            print_mosaic = render_voronoi_mosaic_print(
                target_img=target_img,
                tile_paths=self.tile_paths,
                label_map=label_map,
                cell_tile_indices=cell_tile_indices,
                polygons=polygons,
                output_long_edge=cfg["print_long_edge"],
                color_transfer_alpha=cfg["color_transfer_alpha"],
                grout_color=cfg["grout_color"],
                grout_width=cfg["grout_width"],
            )

            print_h, print_w = print_mosaic.shape[:2]

            print_path = os.path.join(
                os.path.dirname(output_dir),
                (
                    f"{target_name}_voroplex_directional_PRINT_"
                    f"{print_w}x{print_h}_300dpi.png"
                ),
            )

            Image.fromarray(
                print_mosaic.astype(np.uint8)
            ).save(
                print_path,
                format="PNG",
                dpi=(
                    cfg["print_dpi"],
                    cfg["print_dpi"],
                ),
                compress_level=1,
            )

            print(f"[save PRINT] {print_path}")
        else:
            print()
            print("[PRINT] Original source dataset not found.")
            print("[PRINT] Standard VoroPlex mosaic was generated from cache.")
            print("[PRINT] Skipping full-resolution 300 DPI PRINT output.")

        summary = {
            "target_name": target_name,
            "mode": "voroplex_pytorch_sobel_directional_photomosaic",
            "guidance_mode": mode,
            "n_cells": int(len(seeds_opt_px)),
            "target_alpha": float(cfg["target_alpha"]),
            "target_roundness": float(cfg["target_roundness"]),
            "preferred_direction_deg": float(cfg["preferred_direction_deg"]),
            "lambda_preferred": float(cfg["lambda_preferred"]),
            "train_iters": int(cfg["train_iters"]),
            "train_lr": float(cfg["train_lr"]),
            "final_training_loss": None if not history else history[-1],
            "config": cfg,
        }
        with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[voroplex directional] done in {time.time() - t0:.1f}s")
        return mosaic

    def _optimize_generators(
        self,
        *,
        seeds_px: np.ndarray,
        target_img: np.ndarray,
        fabric_target_field: np.ndarray,
        confidence_orient: np.ndarray,
        confidence_edge: np.ndarray,
    ):
        cfg = self.cfg
        H, W = target_img.shape[:2]

        points0_np = seeds_px.astype(np.float64).copy()
        points0_np[:, 0] /= max(W - 1, 1)
        points0_np[:, 1] /= max(H - 1, 1)
        points0_np = np.clip(points0_np, 1e-4, 1.0 - 1e-4)

        points0 = torch.tensor(points0_np, dtype=torch.float64)
        points = torch.nn.Parameter(points0.clone())

        field_t = torch.from_numpy(fabric_target_field).permute(2, 0, 1).unsqueeze(0)
        field_t = field_t.to(dtype=torch.float64)
        confidence_orient_t = torch.from_numpy(confidence_orient).unsqueeze(0).unsqueeze(0)
        confidence_orient_t = confidence_orient_t.to(dtype=torch.float64)
        confidence_edge_t = torch.from_numpy(confidence_edge).unsqueeze(0).unsqueeze(0)
        confidence_edge_t = confidence_edge_t.to(dtype=torch.float64)

        initial_topology = compute_topology_2d(
            points,
            threads=cfg["voroplex_threads"],
        )
        target_area = torch.as_tensor(
            np.asarray(initial_topology.areas, dtype=np.float64).copy(),
            dtype=torch.float64,
        )

        optimizer = torch.optim.Adam([points], lr=cfg["train_lr"])
        n_iters = int(cfg["train_iters"])

        # LR warmup + cosine decay: jumping straight to the peak LR lets seeds
        # take large steps before the Voronoi neighbor topology has settled,
        # which is a classic source of chaotic topology flips (points swap
        # neighbors repeatedly) that show up as jagged sliver cells, especially
        # in visually busy regions. A short warmup lets the topology find a
        # reasonable structure first; only then do we push hard on shape.
        warmup_lr_iters = max(1, int(0.10 * n_iters))
        warmup_start_factor = 0.15

        def _lr_lambda(it: int) -> float:
            if it < warmup_lr_iters:
                frac = (it + 1) / warmup_lr_iters
                return warmup_start_factor + (1.0 - warmup_start_factor) * frac
            prog = (it - warmup_lr_iters) / max(n_iters - warmup_lr_iters, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return float(cfg["lr_min_factor"]) + (1.0 - float(cfg["lr_min_factor"])) * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        min_sep = float(cfg["min_separation_factor"]) / np.sqrt(len(points0_np))
        history = []

        # Curriculum schedule:
        #  - lambda_anchor cosine-decays from its full value down to a small
        #    floor fraction, so seeds are stable at first but are free to
        #    travel onto edges as training proceeds.
        #  - lambda_edge_attract ramps UP from 0 over a short warmup, so the
        #    seed chain first settles into a reasonable local topology before
        #    being pulled hard onto high-confidence edge pixels.
        anchor_floor = float(cfg["anchor_decay_floor"])
        warmup_iters = max(1, int(float(cfg["edge_attract_warmup_frac"]) * n_iters))

        for it in range(n_iters):
            optimizer.zero_grad()

            topology = compute_topology_2d(
                points,
                threads=cfg["voroplex_threads"],
            )
            vertices, cell_vertex_ids, cell_mask, num_vertices_per_cell = reconstruct_cells_2d(topology, points)
            stats = polygon_statistics(vertices, cell_vertex_ids, cell_mask, num_vertices_per_cell)

            progress = it / max(n_iters - 1, 1)
            anchor_scale = anchor_floor + (1.0 - anchor_floor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            cur_lambda_anchor = float(cfg["lambda_anchor"]) * anchor_scale
            cur_lambda_edge_attract = float(cfg["lambda_edge_attract"]) * min(
                1.0, (it + 1) / warmup_iters
            )

            total, parts = compute_directional_training_loss(
                points01=points,
                cell_centroid01=stats["centroid"],
                area=stats["area"],
                perimeter=stats["perimeter"],
                edge_normal=stats["edge_normal"],
                edge_len=stats["edge_len"],
                target_area=target_area,
                initial_points01=points0,
                fabric_target_field_t=field_t,
                confidence_orient_t=confidence_orient_t,
                confidence_edge_t=confidence_edge_t,
                target_alpha=cfg["target_alpha"],
                target_roundness=cfg["target_roundness"],
                floor_roundness=cfg["floor_roundness"],
                confidence_gamma=cfg["confidence_gamma"],
                preferred_direction_deg=cfg["preferred_direction_deg"],
                preferred_alpha=cfg["preferred_alpha"],
                lambda_orientation=cfg["lambda_orientation"],
                lambda_preferred=cfg["lambda_preferred"],
                lambda_roundness=cfg["lambda_roundness"],
                lambda_sliver_guard=cfg["lambda_sliver_guard"],
                lambda_size_uniform=cfg["lambda_size_uniform"],
                lambda_area=cfg["lambda_area"],
                lambda_separation=cfg["lambda_separation"],
                lambda_anchor=cur_lambda_anchor,
                lambda_edge_attract=cur_lambda_edge_attract,
                min_separation=min_sep,
            )

            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite training loss at iteration {it}: {total}")

            total.backward()

            # IMPORTANT: clip each point's own 2D gradient vector, NOT the
            # global norm over all points concatenated. clip_grad_norm_([points], ...)
            # rescales the *entire* flattened gradient by one shared factor, so
            # when many seeds want to move coherently at once (exactly what strip
            # formation needs) the whole update gets crushed as seed count grows.
            # Per-point clipping keeps that coordinated motion intact while still
            # bounding any single seed's step.
            with torch.no_grad():
                grad = points.grad
                if grad is not None:
                    grad_norms = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    scale = (float(cfg["grad_clip"]) / grad_norms).clamp(max=1.0)
                    grad.mul_(scale)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                points.clamp_(1e-4, 1.0 - 1e-4)

            row = {
                key: float(value.detach().cpu())
                for key, value in parts.items()
            }
            row["iteration"] = int(it)
            row["lambda_anchor_eff"] = cur_lambda_anchor
            row["lambda_edge_attract_eff"] = cur_lambda_edge_attract
            row["lr"] = float(scheduler.get_last_lr()[0])
            history.append(row)

            if it == 0 or (it + 1) % int(cfg["print_every"]) == 0:
                print(
                    f"    iter {it + 1:4d}/{n_iters} "
                    f"total={row['total']:.6f} "
                    f"orient={row['orientation']:.6f} "
                    f"pref={row['preferred']:.6f} "
                    f"round={row['roundness']:.6f} "
                    f"sliver={row['sliver_guard']:.6f} "
                    f"edge={row['edge_attract']:.6f} "
                    f"area={row['area']:.6f} "
                    f"anchor_w={cur_lambda_anchor:.5f} "
                    f"edge_w={cur_lambda_edge_attract:.4f} "
                    f"conf={row['mean_point_confidence']:.3f}"
                )

        return points.detach().cpu().numpy(), history

    def _match_all_cells(
        self,
        target_img: np.ndarray,
        label_map: np.ndarray,
        n_cells: int,
        matcher: EnhancedMatcher,
    ) -> list[int]:
        from .features_adapter import get_cell_vgg_embedding

        H, W = label_map.shape
        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        H_approx = max(1, H // 8)
        W_approx = max(1, W // 8)
        cell_tile_indices = [-1] * n_cells

        for i in range(n_cells):
            mask = label_map == i
            if not np.any(mask):
                continue

            rows = ys_grid[mask]
            cols = xs_grid[mask]
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            x0, x1 = int(cols.min()), int(cols.max()) + 1

            region = target_img[y0:y1, x0:x1]
            cell_row = int(rows.mean()) // H_approx
            cell_col = int(cols.mean()) // W_approx

            emb = get_cell_vgg_embedding(region)
            tile_idx = matcher.find_best(region, emb, cell_row, cell_col)
            cell_tile_indices[i] = tile_idx

        assigned = sum(x >= 0 for x in cell_tile_indices)
        print(f"    matched {assigned}/{n_cells} cells")
        return cell_tile_indices

    @staticmethod
    def _save(arr: np.ndarray, folder: str, name: str, quality: int = 92):
        path = os.path.join(folder, name)
        Image.fromarray(arr.astype(np.uint8)).save(path, quality=quality)
        print(f"[save] {path}")
