#!/usr/bin/env python3
"""Figure 1 ingredient: encoding-stimulus image cloud (axis-less, for schematic embedding).

Renders the 969 encoding stimuli as thumbnails in PC1 × PC2 space (PCA fit on
ResNet50 avgpool features, from preprocessed_data/fig1_resnet50_pc50.npz) and writes
the PNG to --out.

NOT BIT-REPRODUCIBLE: the shipped preprocessed_data/fig1_image_cloud_encoding.png
(the frozen input read by fig1_framework) is a 2026-07-20 render from an uncommitted
version of this script (larger thumbnails). Rerunning this script yields a visually
equivalent but not bit-identical cloud; fig1_framework never calls it.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from pnc import paths
from pnc.utils import apply_figure_style, DATA_ROOT  # noqa: F401

apply_figure_style()

PREPROC_DATA = paths.preprocessed_data()

DATA_FILE = PREPROC_DATA / 'fig1_resnet50_pc50.npz'
STIMULI_ENCODING_DIR = Path(DATA_ROOT) / 'stimuli_encoding'

LAYER_KEY = 'AdaptiveAvgPool2davgpool'
RENDER_SIZE = 96
THUMBNAIL_FRAC = 0.020
FIGSIZE = (20, 20)
DPI = 300


def resolve_encoding_path(name):
    base = Path(name).stem
    for ext in ('.jpg', '.png', '.jpeg'):
        p = STIMULI_ENCODING_DIR / (base + ext)
        if p.exists():
            return p
    p = STIMULI_ENCODING_DIR / name
    return p if p.exists() else None


def plot_image_cloud(scores, identifiers, ax):
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]

    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    target_px = bbox.width * THUMBNAIL_FRAC
    base_zoom = target_px / RENDER_SIZE

    cx, cy = np.median(pc1), np.median(pc2)
    dist = (pc1 - cx) ** 2 + (pc2 - cy) ** 2
    order = np.argsort(-dist)

    n_plotted = 0
    for i in order:
        img_path = resolve_encoding_path(identifiers[i])
        if img_path is None:
            continue
        img = Image.open(img_path).convert('RGB').resize(
            (RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)
        im = OffsetImage(np.asarray(img), zoom=base_zoom)
        ab = AnnotationBbox(im, (pc1[i], pc2[i]), frameon=False, pad=0)
        ax.add_artist(ab)
        n_plotted += 1
    return n_plotted


def main(out):
    """Render the image cloud to the PNG path `out` and return it."""
    out = str(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    data = np.load(paths.require(DATA_FILE), allow_pickle=True)
    enc_filenames = data['enc_filenames']
    scores_enc = data[f'{LAYER_KEY}_enc_scores']
    var_exp = data[f'{LAYER_KEY}_explained_var']
    print(f'{LAYER_KEY}: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, n={len(enc_filenames)}')

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    n = plot_image_cloud(scores_enc, enc_filenames, ax)

    margin_x = (scores_enc[:, 0].max() - scores_enc[:, 0].min()) * 0.06
    margin_y = (scores_enc[:, 1].max() - scores_enc[:, 1].min()) * 0.06
    ax.set_xlim(scores_enc[:, 0].min() - margin_x, scores_enc[:, 0].max() + margin_x)
    ax.set_ylim(scores_enc[:, 1].min() - margin_y, scores_enc[:, 1].max() + margin_y)
    ax.set_aspect('equal')
    ax.set_axis_off()

    fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor='white', transparent=False)
    plt.close(fig)
    print(f'Plotted {n} thumbnails → {out}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.output_dir() / 'figures' / 'fig1_image_cloud_encoding.png'),
                    help='output PNG path (never the shipped frozen input)')
    args = ap.parse_args()
    main(args.out)
