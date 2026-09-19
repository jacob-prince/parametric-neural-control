#!/usr/bin/env python3
r"""Supp fig (gradient_gallery) - input-gradient saliency galleries by seed image.

For each of the ten natural seed images (rows), the mean input-gradient saliency of each
of the ten models (columns), averaged over all 25 recorded sites. Saliency = L2 magnitude
of the model's input-output gradient across colour channels, per-map normalized then
site-averaged.

Reads preproc_data/sup_gradient_gallery_avg.pkl (cluster-computed per-site gradient maps,
site-averaged); seed thumbnails from data/stimuli_encoding (the dog row uses the
figure-2 schematic seed crop, figures/assets/seed.png).
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_SHORT_NAMES, get_model_color, STIMULI_PATH)
apply_figure_style()

_M = _fig('gradient_gallery'); STEM = output_name('gradient_gallery')
AVG = os.path.join(str(paths.preprocessed_data()), 'sup_gradient_gallery_avg.pkl')
SEED_PNG = str(paths.ASSETS / 'seed.png')
# seed row order of sup_gradient_gallery_avg.pkl (index 0-9). seed.png is the dog crop (index 4).
SEED_IMAGES = ['shared0575_nsd43157.png', 'shared0850_nsd61798.png', 'shared0968_nsd70194.png',
               'shared0241_nsd20065.png', 'shared0160_nsd13231.png', 'shared0070_nsd07008.png',
               'shared0055_nsd05879.png', 'shared0668_nsd48623.png', 'shared0488_nsd36979.png',
               'shared0940_nsd68312.png']
DOG_SEED_IDX = 4                              # golden retriever -> use the schematic seed.png crop
SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'
FS_MODEL, FS_SEED, FS_TITLE = 6.5, 5.5, 7.5


def seed_thumb(si, px=224):
    """Real seed thumbnail for row `si`; seed.png only for the dog slot, else its own seed image."""
    p = SEED_PNG if si == DOG_SEED_IDX else os.path.join(STIMULI_PATH, SEED_IMAGES[si])
    return np.asarray(Image.open(p).convert('RGB').resize((px, px), Image.LANCZOS))


def main(out_dir):
    import pickle
    d = pickle.load(open(paths.require(AVG, hint='frozen input; python scripts/preprocessing/build_all.py'), 'rb'))
    models = d['models']; sal = d['mean_saliency']; nsites = d['n_sites']
    nseed = next(iter(sal.values())).shape[0]
    vmax = float(np.percentile(np.stack([sal[m] for m in models]), 99.0))

    ncol = 1 + len(models)                       # seed thumbnail + 10 model saliency maps
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 185 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(nseed, ncol, figure=fig, left=0.045, right=0.995, top=0.905, bottom=0.006,
                  wspace=0.05, hspace=0.02)

    for si in range(nseed):
        ax0 = fig.add_subplot(gs[si, 0]); ax0.imshow(seed_thumb(si)); ax0.set_xticks([]); ax0.set_yticks([])
        for s in ax0.spines.values():
            s.set_edgecolor('#333'); s.set_linewidth(0.5)
        if si == 0:
            ax0.set_title('seed', fontsize=FS_SEED, fontfamily=FONT_FAMILY, pad=2, color='#333')
        for mi, m in enumerate(models):
            ax = fig.add_subplot(gs[si, 1 + mi])
            ax.imshow(sal[m][si], cmap='magma', vmin=0, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():         # uniform thin frame
                s.set_edgecolor('#cccccc'); s.set_linewidth(0.3)

    # model column headers: horizontal, bold; wrap compound names to two lines to fit the column
    fig.canvas.draw()
    top_axes = fig.axes[1:1 + len(models)]        # first row: seed(0) then 10 models
    for mi, m in enumerate(models):
        bb = top_axes[mi].get_position(); xc = (bb.x0 + bb.x1) / 2
        c = get_model_color(m)   # per-model palette (robust = green, matching the rest of the paper)
        label = SHORT[m].replace('-', '-\n', 1) if '-' in SHORT[m] else SHORT[m]
        fig.text(xc, 0.91, label, ha='center', va='bottom', fontsize=FS_MODEL, linespacing=0.95,
                 color=c, fontweight='bold', fontfamily=FONT_FAMILY)

    fig.text(0.045, 0.975, f"{_M['title']}  (mean over {nsites[models[0]]} sites)",
             ha='left', va='center', fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('saved ->', out, '| sites/model:', nsites)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    main(**vars(ap.parse_args()))
