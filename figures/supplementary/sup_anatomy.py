#!/usr/bin/env python3
r"""Supp fig (anatomy) - array / Neuropixels placement across recorded animals.

Three stacked raster rows of intra-operative CT and structural-MRI sections showing the probe
tracks for the recorded sites. Row a: Monkey R, anterior IT (two CT sections). Row b: Monkey V,
V3/V4 (CT + structural MRI). Row c: Monkey T (=three0), STS (CT + structural MRI). Arrowheads
mark the electrode tip / entry. Images shown at native resolution (figures/assets/anatomical/).
S-number and output stem come from the manifest.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.utils import apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig
apply_figure_style()

_M = _fig('anatomy'); STEM = output_name('anatomy')
ANAT = paths.ASSETS / 'anatomical'
RED = str(ANAT / 'red_probe.png')
COMPOSITE = str(ANAT / 'three0_and_venus_probes.png')

FS_LETTER, FS_TITLE = 11.0, 8.0
GUTTER_MM = 20.0                 # left strip for panel letter + region title
IMG_W_MM = WIDTH_2COL_MM - GUTTER_MM
GAP_MM = 3.0                     # between-row gap
TOP_PAD_MM, BOT_PAD_MM = 2.0, 2.0


def load_rows():
    """Return [(letter, title, RGB array)] for the three raster rows."""
    red = np.asarray(Image.open(RED).convert('RGB'))
    comp = np.asarray(Image.open(COMPOSITE).convert('RGB'))
    h = comp.shape[0] // 2
    venus = comp[:h]                 # TOP half  = Monkey V / V3-V4 (CT + MRI)
    three0 = comp[h:]                # BOTTOM half = Monkey T / STS (CT + MRI)
    return [
        ('a', 'Monkey R\nanterior IT', red),
        ('b', 'Monkey V\nV3/V4', venus),
        ('c', 'Monkey T\nSTS', three0),
    ]


def main(out_dir):
    rows = load_rows()
    # each image drawn at a common display width -> height set by native aspect
    heights_mm = [IMG_W_MM * im.shape[0] / im.shape[1] for _, _, im in rows]
    fig_h_mm = TOP_PAD_MM + BOT_PAD_MM + sum(heights_mm) + GAP_MM * (len(rows) - 1)

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, fig_h_mm * MM), facecolor=FIG_FACECOLOR)

    gutter_frac = GUTTER_MM / WIDTH_2COL_MM
    img_left = gutter_frac
    img_w = IMG_W_MM / WIDTH_2COL_MM

    y = 1.0 - TOP_PAD_MM / fig_h_mm  # top edge of first image (fraction)
    for (letter, title, im), h_mm in zip(rows, heights_mm):
        h_frac = h_mm / fig_h_mm
        y0 = y - h_frac
        ax = fig.add_axes([img_left, y0, img_w, h_frac])
        ax.imshow(im, interpolation='none', aspect='auto')  # cell already matches native aspect
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('#333333'); s.set_linewidth(0.5)
        # panel letter (top of gutter) + region title just below it
        fig.text(0.006, y - 0.004, letter, ha='left', va='top',
                 fontsize=FS_LETTER, fontweight='bold', fontfamily=FONT_FAMILY)
        fig.text(0.006, y - 0.004 - 5.5 / fig_h_mm, title, ha='left', va='top',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY, color='#333333', linespacing=1.3)
        y = y0 - GAP_MM / fig_h_mm

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('saved ->', out, '| fig height mm:', round(fig_h_mm, 1))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
