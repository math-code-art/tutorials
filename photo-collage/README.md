# Photomosaic Generation Project

This project contains three stages of a computational photomosaic pipeline, progressing from a simple color-based grid mosaic to an adaptive VGG-based mosaic and finally to a directional VoroPlex mosaic.

The three generation methods are:

1. **Basic Uniform-Grid Mosaic**
2. **Adaptive Split + VGG Mosaic**
3. **Directional VoroPlex Mosaic**

Each version can be run independently.

---

# Project Structure

```text
photo-collage/
│
├── basic_mosaic_main.py
├── main.py
├── voronoi_main_voroplex_directional.py
├── config.py
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── features.py
│   ├── mosaic.py
│   ├── pipeline.py
│   └── search.py
│
├── voronoi_src/
│   ├── __init__.py
│   ├── density.py
│   ├── features_adapter.py
│   ├── guidance_field.py
│   ├── matching.py
│   ├── rendering.py
│   ├── voronoi_pipeline_voroplex_directional.py
│   ├── voronoi_seeds.py
│   ├── voroplex_2d_utils.py
│   └── voroplex_directional_loss.py
│
└── data/
    └── README.md
```

Generated outputs, image caches, target images, and the full source-image dataset are intentionally not stored in this repository.

---

# 1. Installation

Clone the repository and enter the `photo-collage` directory.

```bash
git clone https://github.com/math-code-art/tutorials.git
cd tutorials/photo-collage
```

It is recommended to use a virtual environment.

For example, with Conda:

```bash
conda create -n photomosaic python=3.11
conda activate photomosaic
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project uses packages including:

- NumPy
- Pillow
- OpenCV
- SciPy
- scikit-image
- PyTorch
- torchvision
- tqdm

---

# 2. VoroPlex Installation

The third generation method requires the external **VoroPlex** package.

VoroPlex is not included in this repository.

Install VoroPlex separately using its upstream installation instructions before running the directional VoroPlex version.

After installation, verify that Python can import it:

```bash
python -c "import voroplex; print('VoroPlex import successful')"
```

If this command fails, the VoroPlex generation cannot run yet.

The first two mosaic versions do not require VoroPlex.

---

# 3. Dataset Setup

The source-image dataset is not included in this repository because it is very large.

Place all source images inside:

```text
data/wikiart/
```

For example:

```text
photo-collage/
└── data/
    └── wikiart/
        ├── image_000001.jpg
        ├── image_000002.jpg
        ├── image_000003.jpg
        └── ...
```

Supported image formats include:

```text
.jpg
.jpeg
.png
.JPG
.JPEG
.PNG
```

The code recursively searches the dataset directory, so images may also be organized in subfolders.

Do not manually create feature caches. The scripts will generate the necessary caches automatically.

---

# 4. Target Image Setup

Create a directory called:

```text
targets/
```

from inside `photo-collage`:

```bash
mkdir -p targets
```

Place one or more target images inside the folder.

For example:

```text
photo-collage/
└── targets/
    ├── bird1.jpg
    └── lion1.jpg
```

The scripts process target images found in this directory.

For the simplest workflow, test with only one target image first.

---

# 5. Output Directory

Generated files are written locally to:

```text
outputs/
```

You do not need to create this folder manually.

The scripts will create it automatically if necessary.

The `outputs/` directory is ignored by Git because generated mosaics, print files, and intermediate figures can be very large.

---

# Generation 1 — Basic Uniform-Grid Mosaic

## Overview

Run:

```bash
python basic_mosaic_main.py
```

This is the simplest baseline version.

The target image is divided into a **uniform square grid**.

For each grid cell:

1. The mean RGB color of the target cell is calculated.
2. Mean RGB colors are calculated for the source-image dataset.
3. Candidate source images are compared using RGB color distance.
4. A reuse constraint prevents the same source image from being repeatedly selected.
5. The selected source image is placed directly into the corresponding grid cell.

This version intentionally does **not** use:

- VGG features
- Voronoi geometry
- adaptive subdivision
- neural feature matching
- color transfer

It is intended to serve as the baseline photomosaic method.

---

## Basic Mosaic Matching

Conceptually, each target cell is represented by:

```text
Target Cell
    ↓
Mean RGB
    ↓
Nearest source-image colors
    ↓
Reuse constraint
    ↓
Selected source tile
```

The current baseline uses source-image reuse control so that excessive repetition of a single image is avoided.

---

## Basic Mosaic Process Figures

The basic version saves only the most useful process figures for analysis.

A per-target process folder is created, such as:

```text
outputs/bird1_basic_mosaic_steps/
```

Important figures include:

```text
02_uniform_grid.jpg
03_target_cell_means.jpg
```

### `02_uniform_grid.jpg`

Shows the fixed grid used to partition the target image.

This illustrates the most important structural limitation of the baseline method: all cells have the same size regardless of image content.

### `03_target_cell_means.jpg`

Shows the target after every grid cell has been replaced by its mean RGB color.

This visualizes how much spatial and texture information is removed when the baseline method represents each region with only three color values.

The folder also contains:

```text
assignments.csv
summary.txt
```

These files provide reproducibility and quantitative information about the matching process.

---

## Basic Mosaic Final Outputs

Typical files include:

```text
outputs/bird1_basic_mosaic_final.jpg
```

and a high-resolution print master:

```text
outputs/bird1_basic_mosaic_PRINT_24000x15840_300dpi.png
```

The normal JPG is useful for quick inspection.

The `PRINT` PNG is the high-resolution output intended for detailed viewing or printing.

For the print render, the program reloads the original source images rather than simply enlarging the low-resolution working mosaic.

---

# Generation 2 — Adaptive Split + VGG Mosaic

## Overview

Run:

```bash
python main.py
```

This version improves on the uniform-grid baseline in two major ways:

1. The target is divided using **adaptive splitting** rather than a fixed uniform grid.
2. Source-image matching uses **VGG visual features** in addition to image/color information.

The target therefore does not have to use identically sized cells everywhere.

Regions can be subdivided according to local image structure.

---

## Adaptive Splitting

Instead of:

```text
Uniform Grid

□ □ □ □
□ □ □ □
□ □ □ □
□ □ □ □
```

the image can be divided into different-sized rectangular regions.

Conceptually:

```text
Target Image
    ↓
Image-content analysis
    ↓
Adaptive subdivision
    ↓
Variable-sized mosaic blocks
```

This allows more detailed areas of the target to receive finer subdivisions.

---

## VGG Feature Matching

This version uses a pretrained VGG network to extract visual feature representations.

The matching process can therefore consider more than mean color.

Conceptually:

```text
Target block
    ↓
VGG feature representation
    ↓
Compare against source-image features
    ↓
Best source tile
```

The first VGG run may take additional time because pretrained model weights may need to be downloaded and feature caches must be created.

Subsequent runs are generally faster because the generated caches are reused.

---

## Running the VGG Version

Make sure that:

```text
data/wikiart/
```

contains the source-image dataset and:

```text
targets/
```

contains the target image.

Then run:

```bash
python main.py
```

The script will automatically:

1. Load the dataset.
2. Create or load source-image caches.
3. Create or load VGG feature embeddings.
4. Preprocess the target image.
5. Perform adaptive subdivision.
6. Match source images to blocks.
7. Render the working-resolution mosaic.
8. Render the high-resolution print master.

---

## VGG Outputs

Outputs are written to:

```text
outputs/
```

Typical output names include:

```text
<target>_final.jpg
```

and:

```text
<target>_PRINT_<width>x<height>_300dpi.png
```

Additional per-target process files may also be generated for analysis and reproducibility.

The `PRINT` file should be used when evaluating final image detail.

---

# Generation 3 — Directional VoroPlex Mosaic

## Overview

Run:

```bash
python voronoi_main_voroplex_directional.py
```

This is the most advanced generation method in the project.

It replaces rectangular mosaic blocks with optimized irregular cells.

The method combines:

- density-guided Voronoi initialization
- directional image guidance
- VoroPlex differentiable geometry optimization
- orientation-sensitive cell shaping
- image-feature matching
- high-resolution source-image rendering

VoroPlex must already be installed before running this version.

---

# Directional Guidance

The method calculates local directional information from the target image.

The pipeline uses image gradients and a diffused directional field to estimate dominant local orientation.

For example:

- feathers can encourage cells to align with feather direction
- fur can encourage cells to follow strands
- edges can influence local cell orientation
- flat areas can remain comparatively less directional

Conceptually:

```text
Target Image
    ↓
Image Gradients
    ↓
Directional Guidance Field
    ↓
VoroPlex Geometry Optimization
    ↓
Directional Voronoi Cells
```

---

# Density Guidance

A density map is also calculated from the target.

This controls how the mosaic cells are distributed spatially.

Different regions of the target can therefore receive different geometric attention.

---

# VoroPlex Optimization

The initial Voronoi geometry is optimized using VoroPlex.

The optimization encourages cells to respond to the local directional structure of the target while maintaining usable polygon geometry.

The resulting cells are irregular rather than rectangular.

---

# VoroPlex Process Figures

Unlike the basic baseline, the VoroPlex method intentionally saves several intermediate figures because they are useful for understanding and documenting the geometry optimization.

Depending on the current configuration, the generated process folder contains figures such as:

```text
target preprocessing
density map
directional / Sobel guidance
diffused direction field
initial seed locations
initial Voronoi structure
optimized seed locations
optimized Voronoi structure
cell orientation diagnostics
cell aspect-ratio diagnostics
final mosaic
```

These files are useful for visualizing the development of the geometry from the original target to the final directional mosaic.

---

# Running the VoroPlex Version

First verify VoroPlex:

```bash
python -c "import voroplex; print('VoroPlex import successful')"
```

Then run:

```bash
python voronoi_main_voroplex_directional.py
```

The first run may take several minutes or longer depending on:

- target resolution
- number of cells
- number of optimization iterations
- dataset size
- CPU/GPU performance
- whether VGG/cache files already exist

Do not stop the program simply because the optimization appears slow.

---

# VoroPlex Outputs

The script generates:

1. intermediate geometry/process figures
2. a normal working-resolution mosaic
3. a high-resolution print master

The high-resolution file uses a name similar to:

```text
<target>_voroplex_directional_PRINT_<width>x<height>_300dpi.png
```

Use the `PRINT` output when evaluating the final high-resolution mosaic.

---

# High-Resolution Rendering

All three methods can generate a high-resolution print master.

The print output uses a long edge of approximately:

```text
24000 pixels
```

with:

```text
300 dpi
```

metadata.

The important distinction is that the high-resolution renderer reloads the original source images for final output.

It does not simply enlarge the low-resolution preview mosaic.

Conceptually:

```text
Working Mosaic Assignment
        ↓
Selected source-image indices
        ↓
Reload original source images
        ↓
Render directly at print resolution
        ↓
High-resolution PNG
```

This preserves substantially more detail than enlarging a low-resolution composite.

---

# First Run vs. Later Runs

The first run can take significantly longer because the project may need to create:

- resized source-image caches
- mean-color caches
- VGG feature embeddings

Generated caches are stored locally under directories such as:

```text
data/cache/
```

These caches are not included in Git.

After they have been generated once, later runs usually load the existing cache instead of rebuilding it.

---

# Important: Dataset Ordering and Cache Consistency

Source-image cache indices must correspond to the same source-image path ordering used during rendering.

Do not manually reorder cache files or mix caches created from different datasets.

If the source dataset is replaced or substantially changed, delete the old generated caches and allow the scripts to rebuild them.

---

# Recommended Workflow

For a new target image:

### 1. Place the image in:

```text
targets/
```

### 2. Run the baseline:

```bash
python basic_mosaic_main.py
```

### 3. Run the adaptive VGG version:

```bash
python main.py
```

### 4. Verify VoroPlex installation:

```bash
python -c "import voroplex"
```

### 5. Run the directional VoroPlex version:

```bash
python voronoi_main_voroplex_directional.py
```

### 6. Compare the three final outputs in:

```text
outputs/
```

---

# Method Comparison

| Version | Geometry | Matching | Main Purpose |
|---|---|---|---|
| Basic Mosaic | Uniform square grid | Mean RGB + reuse control | Baseline |
| Split + VGG | Adaptive rectangular blocks | VGG-assisted visual matching | Improved visual correspondence |
| Directional VoroPlex | Directional irregular Voronoi cells | Image/VGG-assisted matching | Geometry follows image structure |

The three methods are intended to show the progression of the project rather than three unrelated algorithms.

---

# Common Issues

## No source images found

Make sure the dataset exists at:

```text
data/wikiart/
```

---

## No target images found

Create:

```text
targets/
```

and place at least one JPG or PNG target image inside.

---

## `ModuleNotFoundError: voroplex`

VoroPlex has not been installed or is not available in the current Python environment.

Install it separately according to its upstream instructions.

---

## First run is very slow

This is expected.

The first run may need to process thousands of source images and construct feature caches.

Wait until the cache-building process completes.

Later runs should be substantially faster.

---

## VGG model download

The VGG-based methods use pretrained torchvision features.

On a new environment, PyTorch/torchvision may download pretrained weights the first time the model is initialized.

An internet connection may therefore be required during the first VGG run.

---

## Final image looks lower resolution than expected

Make sure you are viewing the file containing:

```text
PRINT
```

in its filename.

For example:

```text
..._PRINT_24000x..._300dpi.png
```

The regular JPG output is primarily a working-resolution result.

---

# Reproducibility

The scripts save matching information and intermediate results where appropriate.

Because source-image selection depends on the dataset available locally, users should use the same dataset if exact reproduction of a previous mosaic is required.

Generated caches should correspond to the current dataset.

---

# Summary

The project provides three photomosaic generation approaches:

```text
Version 1
Uniform Grid
+
Mean RGB Matching
+
Reuse Control
```

```text
Version 2
Adaptive Split
+
VGG Feature Matching
```

```text
Version 3
Directional VoroPlex Geometry
+
Directional Guidance
+
VGG/Image Matching
```

The progression demonstrates how increasingly sophisticated image representation and geometry can change the visual structure of a computational photomosaic.
