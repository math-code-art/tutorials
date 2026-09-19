import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data/wikiart")
CACHE_DIR  = os.path.join(BASE_DIR, "data/cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TARGET_DIR = os.path.join(BASE_DIR, "targets")

POOL_TILES = 81444
SEED = 42

TILE_SIZE_MIN = 16
TILE_SIZE_MAX = 64

TARGET_RESOLUTION = 2400 #4690 for original # 最长边

SPLIT_VARIANCE_THRESH = 0.00045

EMBED_SIZE = 96
CUT3 = 16
CUT4 = 22

W3 = 0.18
W4 = 0.22
W_COLOR = 0.60

TOPK = 300
MAX_USE = 2
PENALTY = 0.40
LOCAL_RADIUS = 5

COLOR_TRANSFER = True
COLOR_TRANSFER_ALPHA = 0.38

PREVIEW_EVERY = 500
