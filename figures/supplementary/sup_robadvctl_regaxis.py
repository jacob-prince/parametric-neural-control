#!/usr/bin/env python3
"""Supp. Fig. — Synthesis regularization and parametric control (within-model difficulty axis).

The automatically-selected synthesis regularization (augmentation NOISE + spectral-decay ALPHA,
recovered from the production run logs) plotted against control fidelity across all 250
(monkey, model, channel) combos, colored by model with a pooled fit line (a-d).

SPLIT 1/3 of the old robust_advantage_controls figure (old panels a-e). The between-model
decomposition and hp-adjusted robust-advantage controls live in the sibling figures
(robadvctl_controls, robadvctl_alignment).

Reads loader.control_table() + <preprocessed_data>/sup_hyperparam_hp_tuning_selected.csv
(built by scripts/preprocessing/build_hp_tuning_cache.py).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pnc import paths
from pnc.utils import (apply_figure_style, FONT_FAMILY, FIG_FACECOLOR, save_fig, MM, WIDTH_2COL_MM,
                       MODEL_ORDER, get_model_color)
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L

apply_figure_style()
STEM = output_name('robadvctl_regaxis')
PREPROC_DATA = str(paths.preprocessed_data())
HP_CSV = os.path.join(PREPROC_DATA, 'sup_hyperparam_hp_tuning_selected.csv')
HP_HINT = 'python scripts/preprocessing/build_hp_tuning_cache.py'
FS_TITLE, FS_AX, FS_TICK, FS_ANNOT, FS_LETTER = 7, 6.5, 5.5, 5, 8   # final-size-native (5-7 pt @ 180 mm)
JW = {'noise': 0.006, 'decay': 0.03}


def load():
    ct = L.control_table()
    hp = pd.read_csv(paths.require(HP_CSV, hint=HP_HINT))
    m = ct.merge(hp, left_on=['monkey', 'model', 'unit'], right_on=['monkey', 'model', 'channel'])
    return m


def _wm_r(m, pred, outcome):
    """Within-model partial correlation (predictor/outcome centered per model)."""
    d = m.copy()
    for c in (pred, outcome):
        d[c + '_c'] = d[c] - d.groupby('model')[c].transform('mean')
    return stats.pearsonr(d[pred + '_c'], d[outcome + '_c'])[0]


def _ws_r(m, pred, outcome):
    """Within-site (per monkey+channel) partial correlation."""
    d = m.dropna(subset=[pred, outcome]).copy()
    sid = d['monkey'] + '_u' + d['unit'].astype(str)
    for c in (pred, outcome):
        d[c + '_c'] = d[c] - d.groupby(sid)[c].transform('mean')
    return stats.pearsonr(d[pred + '_c'], d[outcome + '_c'])[0]


def _spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def scatter_panel(ax, m, xcol, xlabel, outcome, ylabel, title):
    rng = np.random.default_rng(0)
    for mdl in MODEL_ORDER:
        d = m[m.model == mdl]
        ax.scatter(d[xcol] + rng.uniform(-JW[xcol], JW[xcol], len(d)), d[outcome], s=6,
                   color=get_model_color(mdl), alpha=0.8, edgecolors='white', linewidths=0.2, zorder=4)
    r, p = stats.pearsonr(m[xcol], m[outcome])
    xs = np.array([m[xcol].min(), m[xcol].max()]); sl, ic = np.polyfit(m[xcol], m[outcome], 1)
    ax.plot(xs, sl * xs + ic, color='#333', lw=1.0, zorder=6)
    # stat box in the lower-right corner (clear of the up-trending point cloud)
    ax.text(0.965, 0.05, f'$r$={r:+.2f} (p={p:.0e})\nwithin-model $r$={_wm_r(m, xcol, outcome):+.2f}\n'
            f'within-site $r$={_ws_r(m, xcol, outcome):+.2f}',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.92), zorder=8)
    ax.set_xlabel(xlabel, fontsize=FS_AX); ax.set_ylabel(ylabel, fontsize=FS_AX)
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY, pad=4)
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def collinearity_panel(ax, m):
    rng = np.random.default_rng(2)
    for mdl in MODEL_ORDER:
        d = m[m.model == mdl]
        ax.scatter(d.noise + rng.uniform(-JW['noise'], JW['noise'], len(d)),
                   d.decay + rng.uniform(-JW['decay'], JW['decay'], len(d)), s=6,
                   color=get_model_color(mdl), alpha=0.8, edgecolors='white', linewidths=0.2, zorder=4)
    ax.text(0.04, 0.97, f'$r$={stats.pearsonr(m.noise, m.decay)[0]:+.2f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.85))
    ax.set_xlabel('Augmentation noise', fontsize=FS_AX)
    ax.set_ylabel('Spectral-decay exponent', fontsize=FS_AX)
    ax.set_title('Augmentation noise vs\nspectral-decay exponent', fontsize=FS_TITLE,
                 fontweight='bold', fontfamily=FONT_FAMILY, pad=4)
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def main(out_dir):
    m = load()
    # final-size-native: 180 mm wide; 2x2 scatter block + full-height collinearity panel on the right
    fig = plt.figure(figsize=(148 * MM, 128 * MM), facecolor=FIG_FACECOLOR)   # narrower + taller than 2-col; 2x2 scatters
    gs = GridSpec(2, 2, figure=fig, left=0.10, right=0.975, top=0.85, bottom=0.075,
                  wspace=0.32, hspace=0.42)
    letters = []
    a = fig.add_subplot(gs[0, 0]); scatter_panel(a, m, 'noise', 'Augmentation noise', 'control_r',
                                                 'Control $r$', 'Augmentation noise vs control $r$'); letters.append(a)
    a = fig.add_subplot(gs[0, 1]); scatter_panel(a, m, 'noise', 'Augmentation noise', 'control_slope',
                                                 'Control slope', 'Augmentation noise vs control slope'); letters.append(a)
    a = fig.add_subplot(gs[1, 0]); scatter_panel(a, m, 'decay', 'Spectral-decay exponent', 'control_r',
                                                 'Control $r$', 'Spectral-decay exponent vs control $r$'); letters.append(a)
    a = fig.add_subplot(gs[1, 1]); scatter_panel(a, m, 'decay', 'Spectral-decay exponent', 'control_slope',
                                                 'Control slope', 'Spectral-decay exponent vs control slope'); letters.append(a)
    # panel e (augmentation-noise vs spectral-decay collinearity) removed per request

    fig.canvas.draw()
    for ax, L_ in zip(letters, 'abcd'):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.030, bb.y1 + 0.018, L_, ha='left', va='bottom', fontsize=FS_LETTER,
                 fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.975, 'Synthesis regularization and parametric control',
             ha='center', va='top', fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.935, 'n = 250 site-models (10 models × 5 channels × 5 monkeys); '
             'points colored by model, line = pooled fit',
             ha='center', va='top', fontsize=FS_TICK, fontfamily=FONT_FAMILY, color='#555')

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
