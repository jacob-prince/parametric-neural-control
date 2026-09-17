#!/usr/bin/env python3
r"""Supp fig (controllability_variation) — variation in controllability across sites and models.

Per-site parametric-control predictivity (mean Pearson control r over the 10 models) for the
25 recorded sites, sorted low-to-high and coloured by cortical area. Each site shows its 10
per-model r values (faint dots) with the site mean (black bar); the dashed line is the grand mean.
Fills the manuscript statement "SD = 0.18 in mean control r across the 25 sites, range
0.09-0.74". Self-contained on pnc.preproc.loader (control_table only).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color)

apply_figure_style()
_M = _fig('controllability_variation'); STEM = output_name('controllability_variation')

# Each monkey is one recording area; venus spans V3/V4 (per manifest region convention).
MONKEY_AREA = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
MONKEY_TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
# Colour by MONKEY (5 distinct colours) so leap (STS·L) and three0 (STS·T) are separable.
MONKEY_ORDER = ['red', 'paul', 'venus', 'leap', 'three0']

FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 8, 7, 6.5, 5.5, 5.5


def build():
    ct = L.control_table()
    sites = []
    for (mk, u), d in ct.groupby(['monkey', 'unit']):
        per_model = {m: float(r) for m, r in zip(d.model, d.control_r) if np.isfinite(r)}
        sites.append(dict(mk=mk, unit=int(u), area=MONKEY_AREA[mk],
                          per_model=per_model, mean=float(np.mean(list(per_model.values())))))
    sites.sort(key=lambda s: s['mean'])
    means = np.array([s['mean'] for s in sites])
    return sites, means


def draw(ax, sites, means):
    sd = float(np.std(means, ddof=1))
    gmean = float(means.mean())
    lo, hi = float(means.min()), float(means.max())
    x = np.arange(len(sites))
    rng = np.random.default_rng(0)

    x_gm_right = len(sites) - 1 + 0.40           # stop at the data's right edge (clears the range bracket)
    ax.plot([-0.5, x_gm_right], [gmean, gmean], color='#555555', lw=0.8, ls=(0, (4, 3)), zorder=1)

    for xi, s in zip(x, sites):
        col = MONKEY_COLORS[s['mk']]
        for m, r in s['per_model'].items():
            ax.scatter(xi + rng.uniform(-0.24, 0.24), r, s=6, color=col, alpha=0.35,
                       edgecolors='none', zorder=3)
        ax.plot([xi - 0.32, xi + 0.32], [s['mean'], s['mean']], color=col, lw=1.6,
                solid_capstyle='round', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{MONKEY_TAG[s['mk']]}{s['unit']}" for s in sites],
                       rotation=90, fontsize=FS_TICK - 0.5)
    for t, s in zip(ax.get_xticklabels(), sites):
        t.set_color(MONKEY_COLORS[s['mk']]); t.set_fontweight('bold')

    # grand-mean annotation, tucked below the mean line (clear of the legend)
    ax.annotate("grand mean", xy=(6.0, gmean),
                xytext=(6.0, gmean - 0.075), fontsize=FS_ANNOT, ha='center', va='top',
                color='#333333',
                arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6))
    # range indicator: a vertical bracket spanning the ACTUAL min-to-max span of the
    # site means (lo at the lowest site, hi at the highest), with caps at both ends.
    # Placed just to the RIGHT of the rightmost data column so it clears all data.
    xr = len(sites) - 1.0 + 0.85
    cap = 0.22
    ax.plot([xr, xr], [lo, hi], color='#666666', lw=0.9, zorder=6)
    ax.plot([xr - cap, xr + cap], [lo, lo], color='#666666', lw=0.9, zorder=6)
    ax.plot([xr - cap, xr + cap], [hi, hi], color='#666666', lw=0.9, zorder=6)
    ax.annotate(f"range {lo:.2f}$-${hi:.2f}", xy=(xr, (lo + hi) / 2),
                xytext=(xr + 0.42, (lo + hi) / 2), fontsize=FS_ANNOT,
                ha='center', va='center', color='#333333', rotation=90)
    ax.text(xr, lo - 0.055, f"SD = {sd:.2f}", fontsize=FS_ANNOT, ha='center', va='top',
            color='#333333')      # SD tucked under the range bracket

    ax.set_xlabel('Recorded site (sorted by controllability)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ax.set_ylabel('Parametric-control r', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ax.tick_params(axis='y', labelsize=FS_TICK, direction='out', length=3, width=0.6)
    ax.tick_params(axis='x', length=0)
    ax.grid(True, axis='y', alpha=0.35, lw=0.5, color='#BBBBBB')
    ax.set_axisbelow(True)
    allr = [r for s in sites for r in s['per_model'].values()]
    ax.set_xlim(-0.8, len(sites) + 0.9)
    ax.set_ylim(min(0.0, min(allr) - 0.03), max(allr) * 1.10)

    # area legend: bar marker per area
    handles = [Line2D([0], [0], color=MONKEY_COLORS[mk], lw=1.6,
                      label=f"{MONKEY_AREA[mk]} (Monkey {MONKEY_TAG[mk]})") for mk in MONKEY_ORDER]
    handles.append(Line2D([0], [0], marker='o', ls='', mfc='#888888', mec='none', ms=3,
                          alpha=0.6, label='per-model r'))
    leg = ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9,
                    edgecolor='#CCCCCC', fontsize=FS_ANNOT, ncol=1, handlelength=1.4,
                    handletextpad=0.5, borderpad=0.5, labelspacing=0.35)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)

    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.6)
    return sd, gmean, lo, hi


def main(out_dir):
    sites, means = build()
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 78 * MM), facecolor=FIG_FACECOLOR)
    ax = fig.add_axes([0.075, 0.155, 0.905, 0.72])
    sd, gmean, lo, hi = draw(ax, sites, means)

    fig.canvas.draw()
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.02, _M['title'], ha='center', va='bottom',
             fontsize=FS_TITLE, fontfamily=FONT_FAMILY)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print(f'Saved -> {path}')
    print(f'SD={sd:.4f}  mean={gmean:.4f}  range=[{lo:.4f}, {hi:.4f}]  n_sites={len(sites)}')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
