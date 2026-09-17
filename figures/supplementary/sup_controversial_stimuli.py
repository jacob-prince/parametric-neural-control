#!/usr/bin/env python3
"""Supp fig (controversial_stimuli) — full set of controversial-accentuation stimuli.

The controversial experiment (Monkey R, area aIT, 7 sites) synthesized images that DRIVE
one encoding model's prediction while SUPPRESSING the other's. Each site contributes 10
ResNet50-favoring images and 10 RN50-Robust-favoring images: 7 sites x 10 seeds x 2 target
directions = 140 stimuli. This shows the complete set as two mosaics (ResNet50-favoring |
RN50-Robust-favoring), rows = the 7 targeted aIT sites, columns = the 10 seed images.

Structure from loader.load_controversial_table(); raw stimulus PNGs from
data/stimuli_controversial/ (irreducible experiment input).
"""
import os
import argparse

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_COLORS, DATA_ROOT)
apply_figure_style()
_M = _fig('controversial_stimuli'); STEM = output_name('controversial_stimuli')

STIM_DIR = os.path.join(DATA_ROOT, 'stimuli_controversial', 'results_12-01-2025')

C_R50 = MODEL_COLORS['resnet50']            # standard ResNet50
C_ROBUST = MODEL_COLORS['resnet50_robust']  # RN50-Robust

NIMG = 10
FS_TITLE, FS_HEAD, FS_ROW, FS_COL = 9, 7, 6, 5.5


def resolve(name):
    """Resolve a controversial-stimulus basename against the local raw stimulus dir."""
    p = os.path.join(STIM_DIR, os.path.basename(str(name)))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


SAT_BOOST = 1.15  # increase stimulus-image saturation by 15%


def square(p):
    im = Image.open(p).convert('RGB')
    im = ImageEnhance.Color(im).enhance(SAT_BOOST)
    img = np.asarray(im)
    h, w = img.shape[:2]; s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def main(out_dir):
    # -- load structure -------------------------------------------------------------
    d = L.load_controversial_table()
    t = d['table']
    UNITS = [int(u) for u in d['units']]
    stim = np.asarray(t['stim']).astype(str)
    unit = np.asarray(t['unit']).astype(int)
    img_id = np.asarray(t['img']).astype(int)
    is_r50fav = np.char.find(stim, 'max_r50') >= 0
    is_robfav = np.char.find(stim, 'max_robust') >= 0
    assert is_r50fav.sum() == 70 and is_robfav.sum() == 70, 'expected 70 + 70 = 140'

    # resolve every image up front (blocked report if any missing)
    for s in stim:
        resolve(s)

    def block(fig, sub_gs, favmask, border):
        for ri, u in enumerate(UNITS):
            m = (unit == u) & favmask
            order = np.argsort(img_id[m])
            rows = stim[m][order]
            for ci in range(NIMG):
                ax = fig.add_subplot(sub_gs[ri, ci])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(border); sp.set_linewidth(0.8)
                if ci < len(rows):
                    ax.imshow(square(resolve(rows[ci])), aspect='auto')
                if ci == 0:
                    ax.text(-0.16, 0.5, f'Site {u}', transform=ax.transAxes, ha='right',
                            va='center', fontsize=FS_ROW, fontfamily=FONT_FAMILY, color='#333333')
                if ri == 0:
                    ax.set_title(f'{ci + 1}', fontsize=FS_COL, fontfamily=FONT_FAMILY,
                                 color='#777777', pad=2)

    # -- build ----------------------------------------------------------------------
    # Lay out at final size: square tiles, with the two mosaics STACKED VERTICALLY so
    # the figure fills a full portrait page and each tile is ~2x wider than the old
    # side-by-side layout. Fixed mm bands are reserved for the title + first block
    # header (top), the second block header (between blocks), and a thin bottom margin.
    LEFT, RIGHT, HSP = 0.055, 0.99, 0.10
    TOP_BAND_MM, MID_BAND_MM, BOT_BAND_MM = 16.0, 9.0, 3.0
    FIGW_MM = WIDTH_2COL_MM
    _avail_w = (RIGHT - LEFT) * FIGW_MM                         # one block spans the full width
    _tile_w = _avail_w / (NIMG + (NIMG - 1) * 0.06)             # single tile width w/ intra wspace
    _block_h = _tile_w * len(UNITS) * (1 + HSP)                 # one mosaic's image-region height
    FIGHEIGHT_MM = 2 * _block_h + TOP_BAND_MM + MID_BAND_MM + BOT_BAND_MM
    TOP = 1 - TOP_BAND_MM / FIGHEIGHT_MM
    BOT = BOT_BAND_MM / FIGHEIGHT_MM
    fig = plt.figure(figsize=(FIGW_MM * MM, FIGHEIGHT_MM * MM), facecolor=FIG_FACECOLOR)
    outer = fig.add_gridspec(2, 1, left=LEFT, right=RIGHT, top=TOP, bottom=BOT,
                             hspace=MID_BAND_MM / _block_h)
    gsT = outer[0, 0].subgridspec(len(UNITS), NIMG, wspace=0.06, hspace=HSP)
    gsB = outer[1, 0].subgridspec(len(UNITS), NIMG, wspace=0.06, hspace=HSP)
    block(fig, gsT, is_r50fav, C_R50)
    block(fig, gsB, is_robfav, C_ROBUST)

    _title_y = 1 - 4.0 / FIGHEIGHT_MM                            # ~4 mm from top edge
    _head_dy = 4.5 / FIGHEIGHT_MM                                # ~4.5 mm above each grid (clears col #s)
    fig.text(LEFT, _title_y, _M['title'] + ' (Monkey R, aIT; 7 sites x 10 seeds x 2 directions = 140)',
             fontsize=FS_TITLE, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)
    bt = outer[0, 0].get_position(fig); bb = outer[1, 0].get_position(fig)
    fig.text(0.5 * (bt.x0 + bt.x1), bt.y1 + _head_dy, 'ResNet50-favoring', ha='center', va='center',
             fontsize=FS_HEAD, fontweight='bold', fontfamily=FONT_FAMILY, color=C_R50)
    fig.text(0.5 * (bb.x0 + bb.x1), bb.y1 + _head_dy, 'RN50-Robust-favoring', ha='center', va='center',
             fontsize=FS_HEAD, fontweight='bold', fontfamily=FONT_FAMILY, color=C_ROBUST)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('n stimuli =', len(stim), '(r50fav', int(is_r50fav.sum()), '+ robfav', int(is_robfav.sum()), ')')
    print('n sites =', len(UNITS), 'sites =', UNITS)
    print('saved ->', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    main(**vars(ap.parse_args()))
