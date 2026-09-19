import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class VGGEmbedder:
    def __init__(self, cut_layer):
        vgg = models.vgg16(pretrained=True)
        layers = list(vgg.features.children())
        self.model = nn.Sequential(*layers[:cut_layer])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((config.EMBED_SIZE, config.EMBED_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def embed(self, img_array):
        img = Image.fromarray(img_array.astype(np.uint8))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.mean(dim=[2, 3])
        return feat.cpu().numpy().flatten()

    def embed_batch(self, img_arrays, batch_size=64):
        all_embeddings = []
        for start in tqdm(range(0, len(img_arrays), batch_size),
                         desc="VGG embedding"):
            batch = img_arrays[start: start + batch_size]
            tensors = torch.stack([
                self.transform(Image.fromarray(img.astype(np.uint8)))
                for img in batch
            ]).to(self.device)
            with torch.no_grad():
                feats = self.model(tensors)
                feats = feats.mean(dim=[2, 3])
            all_embeddings.append(feats.cpu().numpy())
        return np.vstack(all_embeddings)


def get_embeddings(tile_cache, cut_layer):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        config.CACHE_DIR,
        f"embeddings_cut{cut_layer}_N{len(tile_cache)}.npy"
    )
    if os.path.exists(cache_path):
        print(f"[features] loading embedding: {cache_path}")
        return np.load(cache_path)
    print(f"[features] extracting VGG features cut_layer={cut_layer}...")
    embedder = VGGEmbedder(cut_layer)
    embeddings = embedder.embed_batch(tile_cache)
    np.save(cache_path, embeddings)
    print(f"[features] embedding saved: {cache_path}")
    return embeddings

def get_block_embedding(block_array, embedder):
    return embedder.embed(block_array)
