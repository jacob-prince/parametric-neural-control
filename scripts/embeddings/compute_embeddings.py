#!/usr/bin/env python3
"""
compute_embeddings.py — Extract ResNet-50 activations for encoding images
(PCA fit) and accentuated stimuli (PCA transform), saving compact PC scores.

Runs on cluster (GPU).  Output: data/fig1_resnet50_pc50.npz

Usage:
    python analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig1_embeddings/compute_embeddings.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.decomposition import PCA

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # was parents[6] (repo root of the cluster mirror); sys.path.insert(circuit_toolkit) removed
DATA_ROOT = Path(os.environ.get("PNC_SOURCE_DATA", PROJECT_ROOT / "data"))  # was PROJECT_ROOT / "data"

from core.layer_hook_utils import featureFetcher  # was circuit_toolkit.layer_hook_utils

# ── Inline model loading (avoids fragile prj_control_fa import chain) ────────
DEFAULT_PREPROCESS = T.Compose([
    T.Lambda(lambda x: T.ToTensor()(x) if isinstance(x, Image.Image) else x),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_resnet50(device="cuda"):
    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V1")
    model.eval().to(device)
    model.requires_grad_(False)
    return model

# ── Config ───────────────────────────────────────────────────────────────────
ENCODING_DIR = DATA_ROOT / "stimuli_encoding"  # was PROJECT_ROOT / "data" / "stimuli_encoding"
CONTROL_DIR = DATA_ROOT / "stimuli_control"  # was PROJECT_ROOT / "data" / "stimuli_control"
OUTPUT_DIR = Path(os.environ.get("PNC_PREPROCESSED_DATA", Path(__file__).resolve().parents[3] / "preproc_data"))  # was parents[3] / "preproc_data" (take9/preproc_data)
OUTPUT_FILE = OUTPUT_DIR / "fig1_resnet50_pc50.npz"

N_COMPONENTS = 50
BATCH_SIZE = 64
ACCENTUATED_SUBSAMPLE = 10  # keep every Nth image

LAYERS = [
    ".layer3.Bottleneck5",
    ".layer4.Bottleneck0",
    ".layer4.Bottleneck2",
    ".AdaptiveAvgPool2davgpool",
]


# ── Dataset ──────────────────────────────────────────────────────────────────
class ImageListDataset(Dataset):
    """Load images from an explicit list of paths."""

    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(self.paths[idx])


def collect_activations(model, fetcher, loader, layers, device):
    """Run images through model and collect flattened activations per layer."""
    layer_acts = {layer: [] for layer in layers}
    paths_out = []

    for batch_imgs, batch_paths in loader:
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            model(batch_imgs)
        for layer in layers:
            act = fetcher[layer]
            act = act.reshape(act.shape[0], -1).cpu().numpy()
            layer_acts[layer].append(act)
        paths_out.extend(batch_paths)

    for layer in layers:
        layer_acts[layer] = np.concatenate(layer_acts[layer], axis=0)

    return layer_acts, paths_out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model + hooks
    model = load_resnet50(device=device)
    fetcher = featureFetcher(
        model, input_size=(3, 224, 224), device=device, print_module=False,
        store_device="cpu",
    )
    for layer in LAYERS:
        fetcher.record(layer)

    # ── 1. Encoding images (PCA fit) ────────────────────────────────────────
    enc_paths = sorted(
        p for p in ENCODING_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Encoding images: {len(enc_paths)}")

    enc_loader = DataLoader(
        ImageListDataset(enc_paths, transform=DEFAULT_PREPROCESS),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )
    enc_acts, enc_fullpaths = collect_activations(model, fetcher, enc_loader, LAYERS, device)
    enc_filenames = np.array([Path(p).name for p in enc_fullpaths])

    for layer in LAYERS:
        print(f"  {layer}: {enc_acts[layer].shape}")

    # ── 2. Accentuated stimuli (subsample, PCA transform) ───────────────────
    acc_paths = sorted(
        p for p in CONTROL_DIR.rglob("*.png")
        if "hp_tuning" not in str(p)
    )
    # Subsample every Nth
    acc_paths = acc_paths[::ACCENTUATED_SUBSAMPLE]
    print(f"Accentuated images (1/{ACCENTUATED_SUBSAMPLE}): {len(acc_paths)}")

    acc_loader = DataLoader(
        ImageListDataset(acc_paths, transform=DEFAULT_PREPROCESS),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
    )
    acc_acts, acc_fullpaths = collect_activations(model, fetcher, acc_loader, LAYERS, device)
    # Store path relative to stimuli_control/
    acc_relpaths = np.array([
        str(Path(p).relative_to(CONTROL_DIR)) for p in acc_fullpaths
    ])

    fetcher.cleanup()

    for layer in LAYERS:
        print(f"  {layer}: {acc_acts[layer].shape}")

    # ── 3. PCA: fit on encoding, transform both ─────────────────────────────
    save_dict = {
        "enc_filenames": enc_filenames,
        "acc_relpaths": acc_relpaths,
    }

    for layer in LAYERS:
        key = layer.strip(".")
        pca = PCA(n_components=N_COMPONENTS, random_state=0)

        enc_scores = pca.fit_transform(enc_acts[layer])
        acc_scores = pca.transform(acc_acts[layer])

        save_dict[f"{key}_enc_scores"] = enc_scores.astype(np.float32)
        save_dict[f"{key}_acc_scores"] = acc_scores.astype(np.float32)
        save_dict[f"{key}_explained_var"] = pca.explained_variance_ratio_.astype(np.float32)

        print(f"  {key}: var explained = {pca.explained_variance_ratio_.sum():.3f}")

    # ── 4. Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, **save_dict)
    print(f"Saved → {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
