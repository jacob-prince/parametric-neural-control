#!/usr/bin/env python3
"""Figure 1 composite: paste the two data ingredients onto the closed-loop schematic.

Single entry point: main(out_dir) rebuilds the accentuation grid in-process
(fig1_grid.main -> <out_dir>/_fig1_accentuation_grid.png) and composites it with the
schematic (figures/assets/schematic_fig1.png) and the image cloud.

The image cloud (preprocessed_data/fig1_image_cloud_encoding.png) is a frozen input: the
original render came from an uncommitted 2026-07-20 version of fig1_cloud.py and cannot be
reproduced bit-exactly (`python -m figures.main.fig1_cloud --out X` regenerates a visually
equivalent, not bit-identical, cloud).

Layout: a wide landscape canvas with the square schematic centred, the image cloud
enlarged into the RIGHT margin and pushed BEHIND the schematic line-art (the schematic's
white background is colour-keyed to transparent, so step arrows/markers overlay the cloud),
and the accentuation mosaic on the LEFT as a tall 7x3 grid (levels run vertically,
suppress at the bottom -> drive at the top; seeds run across).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from pnc import paths
from pnc.manifest import MAIN_FIGURES
from pnc.utils import save_fig
from figures.main import fig1_grid

SCHEMATIC = str(paths.ASSETS / 'schematic_fig1.png')
CLOUD_PNG = paths.preprocessed_data() / 'fig1_image_cloud_encoding.png'

# wide canvas; the square schematic occupies the centre band, cloud/mosaic spill into the margins.
CANVAS = (11.8, 7.5)                 # inches; widened left so the loop centres
SCHEM_BOX = [0.2373, 0.000, 0.6339, 1.000]   # loop (step disks) centred on canvas
CLOUD_BOX = [0.6356, 0.271, 0.4102, 0.533]   # right, follows schematic + pushed out
MOSAIC_CY, MOSAIC_H, MOSAIC_CX = 0.545, 0.650, 0.176   # enlarged about a fixed centre
WHITE_THR = 244                      # >= this on all channels -> transparent (push cloud behind art)


def keyed_schematic():
    """schematic with near-white pixels made transparent, so the cloud shows through blank areas
    while the line-art (brain, arrows, markers, text, cartoons) stays opaque on top of the cloud."""
    im = np.asarray(Image.open(SCHEMATIC).convert('RGBA')).copy()
    white = (im[:, :, :3] >= WHITE_THR).all(axis=2)
    im[white, 3] = 0
    return im


def add_image(fig, box, img, zorder):
    ax = fig.add_axes(box, zorder=zorder)
    ax.imshow(img)
    ax.axis('off'); ax.patch.set_visible(False)
    return ax


def main(out_dir):
    mosaic_path = fig1_grid.main(os.path.join(out_dir, '_fig1_accentuation_grid.png'))
    cloud_path = paths.require(CLOUD_PNG, hint='frozen input (not reproducible); see fig1_cloud.py')

    cloud = Image.open(cloud_path).convert('RGB')
    mosaic = Image.open(mosaic_path).convert('RGB')
    # crop the ingredient's white margins (they are opaque and would cover schematic text)
    marr = np.asarray(mosaic)
    content = (marr < WHITE_THR).any(axis=2)
    ys, xs_ = np.where(content)
    mosaic = mosaic.crop((xs_.min(), ys.min(), xs_.max() + 1, ys.max() + 1))
    # ingredient is now natively tall (7 levels x 3 seeds, upright thumbs) - no rotation
    mw, mh = mosaic.size
    mosaic_w = MOSAIC_H * (mw / mh) * (CANVAS[1] / CANVAS[0])
    MOSAIC_BOX = [MOSAIC_CX - mosaic_w / 2, MOSAIC_CY - MOSAIC_H / 2,
                  mosaic_w, MOSAIC_H]

    fig = plt.figure(figsize=CANVAS, facecolor='white')
    add_image(fig, CLOUD_BOX, cloud, zorder=1)          # bottom: cloud
    add_image(fig, SCHEM_BOX, keyed_schematic(), zorder=2)   # middle: schematic line-art over cloud
    add_image(fig, MOSAIC_BOX, mosaic, zorder=3)        # top: mosaic (opaque, blank left area)

    out_base = os.path.join(out_dir, MAIN_FIGURES[1])
    # the imported ingredient builders applied the figure style (savefig.bbox='tight');
    # the composite is saved UNcropped, as in the original standalone script
    plt.rcParams['savefig.bbox'] = 'standard'
    out = save_fig(fig, out_base, dpi=200, bbox_inches=None)
    plt.close(fig)
    print('saved ->', out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    args = ap.parse_args()
    main(**vars(args))
