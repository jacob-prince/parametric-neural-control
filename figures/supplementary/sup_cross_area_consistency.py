#!/usr/bin/env python3
r"""Supp fig (cross_area_consistency) - cross-area consistency of the control ranking.

Per-area (5 recording areas = 5 animals) per-model mean control r. The rank order of
models by control predictivity is consistent across areas: the mean pairwise Spearman
rho over the 10 area pairs is 0.70 +/- 0.16 SD, and highest for aIT vs cIT (0.94). Number
stamped from the manifest. Self-contained on pnc.preproc.loader.

  a  Area-by-area Spearman matrix of per-model control-r rankings (10 unique pairs);
     aIT-cIT pair boxed as the strongest agreement.
  b  Per-model mean control r in each area (parallel coordinates); shared rank order.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import seaborn as sns

# perceptually uniform, colorblind-safe sequential map for rho in [0, 1]
RHO_CMAP = sns.color_palette('mako', as_cmap=True)

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS,
                       MONKEY_COLORS, get_model_color)
apply_figure_style()

_M = _fig('cross_area_consistency'); STEM = output_name('cross_area_consistency')

UNTR = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
ROBUST = set(ROBUST_MODELS)
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 11, 7, 6.5, 5.5, 5.0

# 5 recording areas = 5 animals; region label per animal
MONK = ['red', 'paul', 'venus', 'leap', 'three0']
REG = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
# area + the animal it was recorded in (distinguishes the two STS areas: Monkey L vs Monkey T)
AREA_LABEL = {'red': 'aIT (Monkey R)', 'paul': 'cIT (Monkey P)', 'venus': 'V3/V4 (Monkey V)',
              'leap': 'STS (Monkey L)', 'three0': 'STS (Monkey T)'}


def build():
    ct = L.control_table()
    piv = ct.pivot_table(index='model', columns='monkey', values='control_r', aggfunc='mean')
    piv = piv[MONK]                                     # per-area per-model mean control r
    pairs = list(combinations(range(len(MONK)), 2))
    M = np.full((len(MONK), len(MONK)), np.nan)
    for a in range(len(MONK)):
        M[a, a] = 1.0
    rhos = []
    for i, j in pairs:
        rho, _ = spearmanr(piv[MONK[i]], piv[MONK[j]])
        M[i, j] = M[j, i] = rho
        rhos.append(rho)
    rhos = np.array(rhos)
    raw = {(m, mk): d['control_r'].dropna().values
           for (m, mk), d in ct.groupby(['model', 'monkey'])}   # per-site control r
    return piv, M, rhos, pairs, raw


def panel_a(ax, M, pairs):
    n = len(MONK)
    im = ax.imshow(M, cmap=RHO_CMAP, vmin=0.0, vmax=1.0, aspect='auto', zorder=1)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            diag = (i == j)
            txt = '1' if diag else f'{v:.2f}'
            col = 'white' if v < 0.83 else '#111'
            ax.text(j, i, txt, ha='center', va='center', fontsize=FS_ANNOT,
                    color='#888' if diag else col, fontfamily=FONT_FAMILY, zorder=3)
    # box the aIT-cIT pair (red=0, paul=1) both off-diagonal cells
    for (r, c) in [(0, 1), (1, 0)]:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='#E1812C',
                               lw=1.4, zorder=4))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([AREA_LABEL[m] for m in MONK], rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_yticklabels([AREA_LABEL[m] for m in MONK], fontsize=FS_TICK)
    for tl, m in zip(ax.get_xticklabels(), MONK):
        tl.set_color(MONKEY_COLORS[m])
    for tl, m in zip(ax.get_yticklabels(), MONK):
        tl.set_color(MONKEY_COLORS[m])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'Spearman $\rho$', fontsize=FS_AX)
    cb.ax.tick_params(labelsize=FS_TICK, length=2)
    cb.outline.set_linewidth(0.4)


def panel_b(ax, piv, raw):
    # rank models by grand mean control r; parallel coordinates across the 5 areas
    order = piv.mean(axis=1).sort_values(ascending=False).index.tolist()
    xs = np.arange(len(MONK))
    rng = np.random.default_rng(0)
    for m in order:
        y = piv.loc[m, MONK].values
        c = MODEL_COLORS.get(m, get_model_color(m))
        lw = 1.6 if m in ROBUST else 0.9
        z = 5 if m in ROBUST else 3
        for xi, mk in zip(xs, MONK):                     # faint individual per-site r values
            sv = raw.get((m, mk), [])
            if len(sv):
                ax.scatter(xi + rng.uniform(-0.11, 0.11, len(sv)), sv, s=4.5, color=c,
                           alpha=0.16, edgecolors='none', zorder=z - 1)
        ax.plot(xs, y, '-o', color=c, lw=lw, ms=4.8, zorder=z,           # larger mean markers
                markeredgecolor='white', markeredgewidth=0.4,
                label=SHORT[m])
    ax.set_xticks(xs)
    ax.set_xticklabels([AREA_LABEL[m] for m in MONK], rotation=45, ha='right', fontsize=FS_TICK)
    for tl, m in zip(ax.get_xticklabels(), MONK):
        tl.set_color(MONKEY_COLORS[m])
    ax.set_ylabel('Mean control r', fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.35, len(MONK) - 0.65)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    # model legend is created in main() and placed BELOW panel a (it labels these lines,
    # so it must not float over by the heatmap)


def main(out_dir):
    piv, M, rhos, pairs, raw = build()
    mean_rho = float(rhos.mean()); sd_rho = float(rhos.std(ddof=1))   # sample SD over the 10 pairs
    aic = float(M[0, 1])                                # red(aIT)-paul(cIT)
    print(f'mean cross-area Spearman rho = {mean_rho:.3f} +/- {sd_rho:.3f} SD; aIT-cIT = {aic:.3f}')

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 90 * MM), facecolor=FIG_FACECOLOR)
    # line plot = panel a (left, wide); heatmap = panel b (right, fills its cell via aspect='auto'
    # so both panels share a top). Model legend goes BELOW panel a (it labels the panel-a lines).
    gs = GridSpec(1, 2, figure=fig, left=0.07, right=0.985, top=0.82, bottom=0.36,
                  wspace=0.30, width_ratios=[2.0, 1.0])
    ax_lines = fig.add_subplot(gs[0, 0]); ax_heat = fig.add_subplot(gs[0, 1])
    panel_b(ax_lines, piv, raw)
    panel_a(ax_heat, M, pairs)

    fig.canvas.draw()
    bb_a = ax_lines.get_position(); bb_b = ax_heat.get_position()
    top_y = max(bb_a.y1, bb_b.y1)                        # one baseline for both titles and letters
    for bb, title in ((bb_a, 'Model control r across areas'),
                      (bb_b, 'Area-by-area ranking agreement')):
        fig.text((bb.x0 + bb.x1) / 2, top_y + 0.02, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
    fig.text(bb_a.x0 - 0.048, top_y + 0.085, 'a', ha='left', va='top', fontsize=FS_LET,
             fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(bb_b.x0 - 0.022, top_y + 0.085, 'b', ha='left', va='top', fontsize=FS_LET,
             fontweight='bold', fontfamily=FONT_FAMILY)

    # model legend for the panel-a lines, centred just below panel a (clear of its rotated ticks)
    h, lbls = ax_lines.get_legend_handles_labels()
    fig.legend(h, lbls, ncol=5, loc='upper center',
               bbox_to_anchor=((bb_a.x0 + bb_a.x1) / 2, 0.155),
               fontsize=FS_ANNOT, frameon=False, handlelength=1.2, handletextpad=0.35,
               columnspacing=1.2, labelspacing=0.5)

    # cross-area summary line under the heatmap
    fig.text((bb_b.x0 + bb_b.x1) / 2, 0.10,
             rf'mean $\rho$ = {mean_rho:.2f} $\pm$ {sd_rho:.2f} SD   |   aIT-cIT $\rho$ = {aic:.2f}',
             ha='center', va='center', fontsize=FS_ANNOT + 0.5, fontfamily=FONT_FAMILY)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
