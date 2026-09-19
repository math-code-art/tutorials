# Photomosaic Generation

This folder contains three photomosaic generation methods:

1. Basic Mosaic
2. Split + VGG Mosaic
3. Directional VoroPlex Mosaic

## 1. Install Dependencies

Enter the project folder:

```bash
cd photo-collage
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The VoroPlex version also requires the `voroplex` package to be installed separately.

You can check whether VoroPlex is available with:

```bash
python -c "import voroplex; print('VoroPlex installed')"
```

---

## 2. Prepare the Dataset

Create the dataset folder:

```bash
mkdir -p data/wikiart
```

Place all source images inside:

```text
data/wikiart/
```

For example:

```text
data/
└── wikiart/
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    └── ...
```

JPG, JPEG, and PNG images are supported.

---

## 3. Add a Target Image

Create the target folder:

```bash
mkdir -p targets
```

Place the image you want to convert into a photomosaic inside:

```text
targets/
```

For example:

```text
targets/
└── bird.jpg
```

For the easiest workflow, use one target image at a time.

---

## 4. Run the Basic Mosaic

```bash
python basic_mosaic_main.py
```

This generates the basic uniform-grid photomosaic.

---

## 5. Run the Split + VGG Mosaic

```bash
python main.py
```

The first run may take longer because image caches and VGG features need to be generated.

Later runs can reuse the saved caches.

---

## 6. Run the Directional VoroPlex Mosaic

Make sure VoroPlex is installed first.

Then run:

```bash
python voronoi_main_voroplex_directional.py
```

This version may take several minutes because it includes geometry optimization.

---

## 7. Results

Generated images are saved in:

```text
outputs/
```

Each method produces a regular output image and a high-resolution print version.

High-resolution files contain:

```text
PRINT
```

in the filename, for example:

```text
bird_basic_mosaic_PRINT_24000x15840_300dpi.png
```

Use the `PRINT` file for the highest-resolution result.

---

## Project Structure

```text
photo-collage/
├── README.md
├── requirements.txt
├── config.py
├── basic_mosaic_main.py
├── main.py
├── voronoi_main_voroplex_directional.py
├── src/
├── voronoi_src/
├── data/
│   └── wikiart/
└── targets/
```

## Quick Start

```bash
cd photo-collage

pip install -r requirements.txt

mkdir -p data/wikiart
mkdir -p targets
```

Add source images to:

```text
data/wikiart/
```

Add a target image to:

```text
targets/
```

Then choose one of:

```bash
python basic_mosaic_main.py
```

```bash
python main.py
```

```bash
python voronoi_main_voroplex_directional.py
```

Results will appear in:

```text
outputs/
```
## Precomputed Cache

Precomputed cache files are included in:

```text
data/cache/
```

The cache files can significantly reduce preprocessing time because they
contain precomputed resized tiles, mean-RGB values, and VGG feature embeddings.

However, **the cache files do not replace the original source-image dataset**.

The original source images are still required in:

```text
data/wikiart/
```

The dataset must correspond to the same source-image collection used to
generate the provided cache.

### Important

If you only have the cache files but do not have the original source images,
you will **not** be able to generate the full high-resolution final output
correctly.

The high-resolution `PRINT` versions reload the original source images and
render them directly at print resolution.

For example:

```text
*_PRINT_24000x..._300dpi.png
```

uses the original images from:

```text
data/wikiart/
```

rather than enlarging the low-resolution cached tiles.

Therefore:

```text
Code + original dataset + cache
    -> fastest workflow
    -> full high-resolution PRINT output

Code + original dataset only
    -> works
    -> caches will be rebuilt automatically
    -> full high-resolution PRINT output

Code + cache only
    -> not sufficient for the current pipeline
    -> original source images are still required
```

For best results, download the original dataset and place it in:

```text
photo-collage/data/wikiart/
```

Then the provided cache can be reused to avoid rebuilding the expensive
preprocessing and VGG features.
