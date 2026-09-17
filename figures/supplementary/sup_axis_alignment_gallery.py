#!/usr/bin/env python3
"""Supp. Fig. — Accentuation sweeps in the read-out-aligned embedding.

SPLIT 2/2 of the old axis-alignment figure (take7 figS09 panel E): the per-model
sweep gallery in the Fig 3A-style readout-aligned embedding. 4x5 grid:
rows 1-2 = most representative channel per model, rows 3-4 = most axis-aligned
channel per model — all 10 seed sweeps each. The vertical direction is the
encoding axis (teal); the horizontal + depth directions are the top two off-axis
PCs (Fig 3A-style 3D projection; the third off-axis component recedes into depth).

Reads only loader.load_axis_alignment() (preprocessed_data/axis_alignment.pkl).
Instant, numpy only.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, FIG_FACECOLOR, save_fig,
                       MM, WIDTH_2COL_MM,
                       MODEL_ORDER, MODEL_SHORT_NAMES, get_model_color, ROBUST_MODELS)
STEM = output_name('axis_alignment_gallery')

apply_figure_style()
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'      # display relabel

AXIS_TEAL = '#00C39A'     # encoding axis (Fig 3A convention)
FS_TITLE, FS_TICK, FS_ANNOT = 7, 6.5, 5.5


# ── one exemplar cell in the readout-aligned embedding ────────────────────
def _teal_axis(ax, xlim, ylim):
    base = (float(np.mean(xlim)), 0.0)
    alen = (ylim[1] - ylim[0]) * 0.30
    ax.annotate('', xy=(base[0], base[1] + alen), xytext=base,
                arrowprops=dict(arrowstyle='->', color=AXIS_TEAL, lw=1.0,
                                mutation_scale=8), zorder=1000)
    ax.scatter([base[0]], [base[1]], c=AXIS_TEAL, s=12, marker='o',
               edgecolors='black', linewidths=0.5, zorder=1000)


def exemplar_bounds(ex):
    xs, ys = list(ex['cloud_x']), list(ex['cloud_y'])
    for t in ex['trajs']:
        xs += list(t['tx']); ys += list(t['ty'])
        if 'seed_xy' in t:
            xs.append(t['seed_xy'][0]); ys.append(t['seed_xy'][1])
    return min(xs), max(xs), min(ys), max(ys)


def draw_exemplar(ax, ex, xlim, ylim):
    """Draw one cell's 10 sweeps in the given (shared-scale) window. No title —
    model names are placed at a common y in main() so the top dots align."""
    cx, cy = ex['cloud_x'], ex['cloud_y']
    cresp = ex['cloud_resp_spk']
    hull = np.column_stack([cx, cy])[ex['hull_idx']]
    ax.add_patch(Polygon(hull, closed=True, facecolor='#ECECEC', alpha=0.45,
                         edgecolor='#C0C0C0', lw=0.5, zorder=-10))
    ax.scatter(cx, cy, c='#AAAAAA', s=2, alpha=0.4, edgecolors='none', zorder=1)
    norm = Normalize(vmin=cresp.min(), vmax=cresp.max())
    cmap = plt.colormaps['RdBu_r']
    seed_x, seed_y = [], []
    for t in ex['trajs']:
        ax.plot(t['tx'], t['ty'], color='black', lw=0.35, alpha=0.4, zorder=5)
        ax.scatter(t['tx'], t['ty'], c=[cmap(norm(v)) for v in t['scores_spk']],
                   s=3.2, edgecolors='black', linewidths=0.15, zorder=10)
        if 'seed_xy' in t:
            seed_x.append(t['seed_xy'][0]); seed_y.append(t['seed_xy'][1])
    if seed_x:
        ax.scatter(seed_x, seed_y, c='white', s=11, marker='D',
                   edgecolors='black', linewidths=0.5, zorder=20)
    _teal_axis(ax, xlim, ylim)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.axis('off')


TOPC = 0.06   # content-top sits TOPC*window below the window top (constant -> aligned)


def draw_exemplar_row(fig, exs, models, y0, xs, pw, ph):
    """A row of square exemplar panels; per-panel window sized to its own content
    but top-anchored with a constant fraction so the top dots align."""
    axs = []
    for m, x in zip(models, xs):
        xl, xh, yl, yh = exemplar_bounds(exs[m])
        xc = 0.5 * (xl + xh)
        Ri = max(xh - xl, yh - yl) * 1.14
        ax = fig.add_axes([x, y0, pw, ph])
        draw_exemplar(ax, exs[m], (xc - Ri / 2, xc + Ri / 2),
                      (yh + TOPC * Ri - Ri, yh + TOPC * Ri))
        axs.append(ax)
    return axs


def place_row_names(fig, axs, models, y):
    for ax, m in zip(axs, models):
        p = ax.get_position()
        fig.text(0.5 * (p.x0 + p.x1), y, MODEL_SHORT_NAMES[m], fontsize=FS_TICK,
                 color=get_model_color(m), fontfamily=FONT_FAMILY, ha='center',
                 va='bottom', fontweight='bold' if m in ROBUST_MODELS else 'normal')


# ── build ────────────────────────────────────────────────────────────────
def main(out_dir):
    d = L.load_axis_alignment()
    exemplars = d['exemplars']
    exemplars_aligned = d['exemplars_aligned']
    top5, bot5 = MODEL_ORDER[:5], MODEL_ORDER[5:]

    FIG_W_MM, FIG_H_MM = WIDTH_2COL_MM, 158.0        # 180 mm wide; 4x5 grid + header
    FIG_W, FIG_H = FIG_W_MM * MM, FIG_H_MM * MM      # inches, for physical-square panel math
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_FACECOLOR)

    # 4x5 grid — rows 1-2 representative, rows 3-4 most-aligned
    ncol = 5
    L_, Rt, gap_c = 0.055, 0.985, 0.012
    pw = (Rt - L_ - (ncol - 1) * gap_c) / ncol
    ph = pw * FIG_W / FIG_H                          # physical square
    xs = [L_ + i * (pw + gap_c) for i in range(ncol)]
    intra, inter, bot0 = 0.014, 0.050, 0.015
    y_R4 = bot0
    y_R3 = y_R4 + ph + intra
    y_R2 = y_R3 + ph + inter
    y_R1 = y_R2 + ph + intra
    axR1 = draw_exemplar_row(fig, exemplars, top5, y_R1, xs, pw, ph)
    axR2 = draw_exemplar_row(fig, exemplars, bot5, y_R2, xs, pw, ph)
    axR3 = draw_exemplar_row(fig, exemplars_aligned, top5, y_R3, xs, pw, ph)
    axR4 = draw_exemplar_row(fig, exemplars_aligned, bot5, y_R4, xs, pw, ph)

    # terse title
    fig.text(L_, 0.968, 'Accentuation sweeps in the read-out-aligned embedding',
             fontsize=8, fontfamily=FONT_FAMILY, fontweight='bold',
             ha='left', va='center')

    # aligned model names right above the top dots of each grid row
    fig.canvas.draw()
    def ny(y0):
        return (y0 + ph) - TOPC * ph + 0.004
    place_row_names(fig, axR1, top5, ny(y_R1))
    place_row_names(fig, axR2, bot5, ny(y_R2))
    place_row_names(fig, axR3, top5, ny(y_R3))
    place_row_names(fig, axR4, bot5, ny(y_R4))

    # left group tags (rotated) spanning each 2-row block
    def blocktag(yc, desc):
        fig.text(0.018, yc, desc, rotation=90, fontsize=FS_TITLE, fontweight='bold',
                 color='#333', ha='center', va='center', fontfamily=FONT_FAMILY)
    blocktag((y_R2 + y_R1 + ph) / 2, 'most representative channel')
    blocktag((y_R4 + y_R3 + ph) / 2, 'most axis-aligned channel')

    # compact axis key (single horizontal strip below the title): vertical =
    # encoding axis (teal); horizontal = top off-axis PC
    kax = fig.add_axes([L_, 0.905, 0.46, 0.055]); kax.axis('off')
    kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    kax.annotate('', xy=(0.020, 0.92), xytext=(0.020, 0.08),                # encoding axis (vertical, teal)
                 arrowprops=dict(arrowstyle='-|>', color=AXIS_TEAL, lw=1.0, mutation_scale=8))
    kax.text(0.050, 0.50, 'vertical = encoding axis', color=AXIS_TEAL, fontsize=FS_ANNOT,
             va='center', ha='left', fontweight='bold', fontfamily=FONT_FAMILY)
    ox, oy = 0.49, 0.28                                                     # shared origin for the off-axis pair
    kax.annotate('', xy=(0.565, oy), xytext=(ox, oy),                       # horizontal arrow
                 arrowprops=dict(arrowstyle='-|>', color='#888', lw=0.9, mutation_scale=7))
    kax.text(0.575, oy, 'horizontal', color='#888', fontsize=FS_ANNOT, va='center', ha='left',
             fontfamily=FONT_FAMILY)
    kax.annotate('', xy=(0.522, 0.62), xytext=(ox, oy),                     # second SHORT arrow = depth (into page)
                 arrowprops=dict(arrowstyle='-|>', color='#888', lw=0.9, mutation_scale=5.5))
    kax.text(0.535, 0.70, 'depth', color='#888', fontsize=FS_ANNOT, va='center', ha='left',
             fontfamily=FONT_FAMILY)
    kax.text(0.70, oy, '= top 2 off-axis PCs', color='#888', fontsize=FS_ANNOT, va='center',
             ha='left', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print(f'Saved -> {out}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
