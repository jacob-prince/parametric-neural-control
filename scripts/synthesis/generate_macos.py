#!/usr/bin/env python3
"""
Generate MACO visualizations for all monkey x model x channel combinations.

Iterates over 5 monkeys x 10 models x 5 channels (250 total), loads the
pre-fit encoding direction (readout_vec) from the posthoc prediction pickles,
and generates one MACO image per combination using horama.

Skips combinations whose output already exists.

Usage:
    python fig_generate_macos.py                          # run all
    python fig_generate_macos.py --monkey red              # one monkey
    python fig_generate_macos.py --model resnet50          # one model
    python fig_generate_macos.py --monkey red --unit 0     # one unit
    python fig_generate_macos.py --device mps              # use Apple Silicon GPU
    python fig_generate_macos.py --image-size 512 --steps 2048  # faster/smaller
"""

import os
import sys
import glob
import re
import pickle
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from horama import maco, plot_maco
from horama.losses import dot_cossim

# Add project code to path and patch ckptroot for local use
_prj_dir = os.path.dirname(os.path.abspath(__file__))  # was join(__file__, '..', '..', '..', 'prj_control_fa'); the prj_control_fa snapshot is vendored in this folder
sys.path.insert(0, _prj_dir)

# PNC: the getpass username patch is no longer needed (models.py reads PNC_MODEL_BACKBONES)
import models as _models_mod
# Point checkpoint root to local data
_models_mod.ckptroot = os.environ.get('PNC_MODEL_BACKBONES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'model_backbones'))  # was join(__file__, '..', '..', '..', 'data', 'model_backbones')

from models import load_model
from core.layer_hook_utils import featureFetcher  # was circuit_toolkit.layer_hook_utils

# ============================================================================
# Constants
# ============================================================================

DATA_ROOT = os.environ.get('PNC_SOURCE_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data'))  # was join(__file__, '..', '..', '..', 'data')
FIGURES_ROOT = os.environ.get('PNC_MACO_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), '..', 'figures'))  # was join(__file__, '..', 'figures')
OUTPUT_DIR = FIGURES_ROOT

PCA_PROJ_PATH = os.path.join(DATA_ROOT, 'image_pca_projections')
ENCODING_MODEL_PATH = os.path.join(DATA_ROOT, 'encoding_model_outputs', 'Encoding_model_outputs')

MONKEY_DIRS = {
    'red': 'red_20250428-20250430',
    'paul': 'paul_20250428-20250430',
    'venus': 'venus_250426-250429',
    'leap': 'leap_250426-250501',
    'three0': 'three0_250426-250501',
}

MONKEY_UNITS = {
    'red': [0, 2, 9, 15, 19],
    'paul': [0, 8, 24, 40, 47],
    'venus': [9, 79, 151, 331, 355],
    'three0': [56, 74, 120, 168, 204],
    'leap': [81, 282, 286, 306, 342],
}

MODELS = [
    'AlexNet_training_seed_01',
    'clipag_vitb32',
    'dinov2_vitb14_reg',
    'radio_v2.5-b',
    'regnety_640',
    'resnet50',
    'resnet50_clip',
    'resnet50_dino',
    'resnet50_robust',
    'siglip2_vitb16',
]

# Default MACO hyperparameters (from notebook)
DEFAULT_MACO_PARAMS = dict(
    total_steps=4096,
    learning_rate=0.1,
    image_size=2048,
    model_input_size=224,
    noise=0.08,
    values_range=(0, 1),
    crops_per_iteration=12,
    box_size=(0.2, 0.25),
)


# ============================================================================
# Data loading
# ============================================================================

def load_encoding_data(monkey, monkey_dir, unit, model):
    """
    Load encoding direction from posthoc prediction pickle.

    Returns dict with keys: readout_vec, readout_bias, layer_name, config
    """
    pkl_name = f"posthoc_prediction_NSDencimg_PCA_pop_unit_{monkey_dir}_unit{unit}_{model}.pkl"
    pkl_path = os.path.join(PCA_PROJ_PATH, monkey_dir, 'posthoc_model_predict_PCA_popul_unit', pkl_name)

    if not os.path.exists(pkl_path):
        return None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    return {
        'readout_vec': data['readout_vec'],
        'readout_bias': data['readout_bias'],
        'layer_name': data['config']['layer_name'],
        'config': data['config'],
    }


def find_xtfmer_path(monkey_dir, model, unit, layer_name):
    """Find the JIT-compiled PCA transform file.

    Channel numbers have inconsistent zero-padding in filenames (Ch00 vs Ch81),
    so we always glob to find the correct file.
    """
    enc_dir = os.path.join(ENCODING_MODEL_PATH, monkey_dir)

    # Try common patterns: Ch0, Ch00, Ch000 etc
    for ch_str in [f"Ch{unit}", f"Ch{unit:02d}", f"Ch{unit:03d}"]:
        pattern = f"{monkey_dir}_{model}_{ch_str}_Xtfmer_{layer_name}_pca750_RidgeCV_JITscript.pt"
        path = os.path.join(enc_dir, pattern)
        if os.path.exists(path):
            return path

    # Broader glob fallback
    glob_pattern = f"{monkey_dir}_{model}_Ch*_Xtfmer_{layer_name}_pca750_RidgeCV_JITscript.pt"
    matches = glob.glob(os.path.join(enc_dir, glob_pattern))
    # Filter to matching unit number
    for m in matches:
        ch_match = re.search(r'_Ch(\d+)_', os.path.basename(m))
        if ch_match and int(ch_match.group(1)) == unit:
            return m

    return None


def build_readout_layer(readout_vec, readout_bias, device):
    """
    Build a single-output linear layer from the encoding direction.

    readout_vec: (750,) tensor - the encoding axis in PCA space
    readout_bias: scalar bias
    """
    if hasattr(readout_vec, 'numpy'):
        readout_vec_np = readout_vec.numpy()
    else:
        readout_vec_np = np.array(readout_vec)

    readout = nn.Linear(readout_vec_np.shape[0], 1, bias=True)
    readout.weight.data = torch.from_numpy(readout_vec_np).float().unsqueeze(0)
    readout.bias.data = torch.tensor([float(readout_bias)]).float()
    readout.eval().requires_grad_(False)
    return readout.to(device)


# ============================================================================
# MACO generation
# ============================================================================

def generate_maco(
    dnn_model, preprocess, readout, xtransform, layer_name,
    device, maco_params,
):
    """
    Generate a single MACO image for one encoding direction.

    Returns (image_tensor, alpha_tensor).
    """
    # featureFetcher only supports "cuda"/"cpu" for module name discovery,
    # so we use "cpu" then move the model back to the target device.
    fetcher = featureFetcher(dnn_model, input_size=(3, 224, 224),
                             device="cpu", print_module=False, store_device="cpu")
    fetcher.record(layer_name, ingraph=True, store_device=device)
    dnn_model.to(device)
    fetcher.device = device

    def objective(images):
        images = preprocess(images)
        images.requires_grad_(True)
        dnn_model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = xtransform(feat_tsr)
        pred = readout(feat_vec)  # (B, 1)
        return pred.mean()

    image, alpha = maco(objective, **maco_params, device=device)

    # Cleanup hooks
    fetcher.cleanup()

    return image, alpha


def save_maco_image(image, alpha, output_path):
    """Save MACO as a PNG via matplotlib (handles alpha compositing)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_maco(image, alpha)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ============================================================================
# Main loop
# ============================================================================

def get_output_path(monkey, model, unit):
    return os.path.join(OUTPUT_DIR, f"02c_{monkey}_{model}_Ch{unit}_maco.png")


def main():
    parser = argparse.ArgumentParser(description='Generate MACOs for all monkey/model/unit combos')
    parser.add_argument('--monkey', type=str, default=None,
                        choices=list(MONKEY_DIRS.keys()))
    parser.add_argument('--model', type=str, default=None,
                        choices=MODELS)
    parser.add_argument('--unit', type=int, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if omitted.')
    parser.add_argument('--image-size', type=int, default=DEFAULT_MACO_PARAMS['image_size'],
                        help=f'MACO canvas size (default: {DEFAULT_MACO_PARAMS["image_size"]})')
    parser.add_argument('--steps', type=int, default=DEFAULT_MACO_PARAMS['total_steps'],
                        help=f'Optimization steps (default: {DEFAULT_MACO_PARAMS["total_steps"]})')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if output exists')
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Build MACO params
    maco_params = dict(DEFAULT_MACO_PARAMS)
    maco_params['image_size'] = args.image_size
    maco_params['total_steps'] = args.steps

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine what to iterate over
    monkeys = [args.monkey] if args.monkey else list(MONKEY_DIRS.keys())
    models = [args.model] if args.model else MODELS

    # Count work
    total = 0
    skipped = 0
    failed = 0

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load DNN model once per model
        try:
            dnn_model, preprocess = load_model(model_name, device=device)
            dnn_model.eval().requires_grad_(False)
        except Exception as e:
            print(f"  ERROR loading model {model_name}: {e}")
            # Count all units for this model as failed
            for monkey in monkeys:
                units = [args.unit] if args.unit is not None else MONKEY_UNITS[monkey]
                failed += len(units)
            continue

        for monkey in monkeys:
            monkey_dir = MONKEY_DIRS[monkey]
            units = [args.unit] if args.unit is not None else MONKEY_UNITS[monkey]

            for unit in units:
                output_path = get_output_path(monkey, model_name, unit)

                # Skip if already generated
                if os.path.exists(output_path) and not args.force:
                    print(f"  [skip] {monkey} Ch{unit} - already exists")
                    skipped += 1
                    continue

                print(f"  Generating {monkey} Ch{unit}...", end=' ', flush=True)

                # Load encoding direction
                enc_data = load_encoding_data(monkey, monkey_dir, unit, model_name)
                if enc_data is None:
                    print("MISSING encoding data")
                    failed += 1
                    continue

                layer_name = enc_data['layer_name']

                # Load PCA transform
                xtfmer_path = find_xtfmer_path(monkey_dir, model_name, unit, layer_name)
                if xtfmer_path is None:
                    print(f"MISSING Xtfmer for layer {layer_name}")
                    failed += 1
                    continue

                try:
                    xtransform = torch.jit.load(xtfmer_path, map_location=device)
                    if hasattr(xtransform, 'to'):
                        xtransform = xtransform.to(device)
                except Exception as e:
                    print(f"ERROR loading Xtfmer: {e}")
                    failed += 1
                    continue

                # Build readout
                readout = build_readout_layer(
                    enc_data['readout_vec'], enc_data['readout_bias'], device
                )

                # Generate MACO
                try:
                    image, alpha = generate_maco(
                        dnn_model, preprocess, readout, xtransform,
                        layer_name, device, maco_params,
                    )
                    save_maco_image(image, alpha, output_path)
                    total += 1
                    print(f"OK -> {os.path.basename(output_path)}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    failed += 1
                    continue

    print(f"\n{'='*60}")
    print(f"Done! Generated: {total}, Skipped: {skipped}, Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
