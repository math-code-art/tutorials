# Photo Collage

Generate your own photomosaic from a JPG or PNG image using one of three methods:

1. **Basic Mosaic** — uniform square grid with mean RGB matching.
2. **Split + VGG Mosaic** — adaptive quadtree splitting with VGG feature matching.
3. **Directional VoroPlex Mosaic** — directional Voronoi cells with feature-based tile matching.

The repository includes precomputed caches for **81,444 source tiles**, so the original source-image dataset is not required for standard mosaic generation.

## Download

This project uses **Git LFS** for the precomputed cache files.

For the most reliable setup, clone the repository with Git:

    git lfs install
    git clone https://github.com/math-code-art/tutorials.git
    cd tutorials/photo-collage
    git lfs pull

The cache download is approximately 1.3 GB.

## Install Dependencies

From the `photo-collage` directory:

    python -m pip install -r requirements.txt

The **Directional VoroPlex Mosaic** also requires the external VoroPlex Python package to be installed separately.

The Split + VGG and VoroPlex methods use pretrained VGG features. Torchvision may download pretrained VGG weights the first time they are needed.

## Add Your Target Image

Create the target directory if needed:

    mkdir -p targets

Place your JPG or PNG image inside:

    targets/

For example:

    targets/my_photo.jpg

## Run

### Basic Mosaic

Uniform square grid with mean RGB tile matching.

    python basic_mosaic_main.py

### Split + VGG Mosaic

Adaptive quadtree splitting with VGG feature matching.

    python main.py

### Directional VoroPlex Mosaic

Directional Voronoi photomosaic.

    python voronoi_main_voroplex_directional.py

VoroPlex must be installed separately before running this method.

## Output

Generated results are saved in:

    outputs/

Each method creates its own mosaic output and process files.

## Included Precomputed Cache

The repository includes the cache files required for standard mosaic generation:

    data/cache/

Main cache files:

    basic_mean_rgb_N81444.npy
    basic_mean_rgb_paths_N81444.txt
    embeddings_cut16_N81444.npy
    embeddings_cut22_N81444.npy
    tiles_hires64_N81444.npy
    tiles_small16_N81444.npy

The cache preserves the ordering of all 81,444 source tiles.

Because this cache is included, users do not need the original image collection to generate the normal mosaic output.

## High-Resolution PRINT Output

The standard mosaic uses the included precomputed cache.

The optional high-resolution PRINT output works differently. It reloads the original JPG or PNG source tiles so that the final print is rendered from the original images instead of enlarging the cached 64 px versions.

If the original source-image dataset is not installed:

    standard mosaic        -> generated normally
    high-resolution PRINT  -> skipped automatically

To enable the high-resolution PRINT output, place the matching original source-image collection in:

    data/wikiart/

Then run the same mosaic script again.

The PRINT version is generated at approximately:

    24000 px long edge
    300 DPI

The original dataset is optional and is only required for the full-resolution PRINT output.

## Quick Start

After cloning the repository:

    cd tutorials/photo-collage
    git lfs pull
    python -m pip install -r requirements.txt
    mkdir -p targets

Put your JPG or PNG image in `targets/`.

Then choose one method:

    python basic_mosaic_main.py

or:

    python main.py

or, after installing VoroPlex:

    python voronoi_main_voroplex_directional.py

The generated mosaic will appear in `outputs/`.

## Notes

- The original 81,444-image source dataset is **not required** for standard mosaic generation.
- The included Git LFS cache is sufficient for normal rendering.
- The original dataset is only needed for the optional 300 DPI full-resolution PRINT output.
- Do not change the ordering of the cache files or the accompanying tile manifest.
