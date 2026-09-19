#!/usr/bin/env python3
"""Supp fig (robadvctl_controls) - control outcomes after regressing out the effect of
the synthesis hyperparameters.

The synthesis regularization knobs (augmentation NOISE + spectral-decay ALPHA, selected
per site-model by an automated procedure) are regressed out of the control outcomes,
using coefficients from the JOINT model `outcome ~ C(model) + noise + decay + C(site)`
so model structure is never absorbed into the hp slopes; site variance stays in the
display.

  a,b Control r / control slope with hp regressed out, by model (per-model swarms +
      means, models sorted by residual mean).
  c   Summary: raw adv-vs-conventional gap (site-fixed-effects Delta among the 9 trained
      models, p = within-site paired Wilcoxon over 25 sites) vs the hp-ADJUSTED gap
      (ANCOVA `outcome ~ adv + noise + decay + C(site)`, site-clustered p). Untrained is
      shown in a,b but never enters the family tests.

Reads loader.control_table() + <preprocessed_data>/sup_hyperparam_hp_tuning_selected.csv
(built by scripts/preprocessing/build_hp_tuning_cache.py).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pnc import paths
from pnc.utils import (apply_figure_style, FONT_FAMILY, FIG_FACECOLOR, save_fig, MM,
                       WIDTH_2COL_MM, MODEL_SHORT_NAMES, get_model_color)
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.famstats import adv_gap, partial_residuals, stars as _sig_stars

apply_figure_style()
STEM = output_name('robadvctl_controls')
UNTR = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
ROBUST = {'clipag_vitb32', 'resnet50_robust'}
PREPROC_DATA = str(paths.preprocessed_data())
HP_CSV = os.path.join(PREPROC_DATA, 'sup_hyperparam_hp_tuning_selected.csv')
HP_HINT = 'python scripts/preprocessing/build_hp_tuning_cache.py'
FS_TITLE, FS_AX, FS_TICK, FS_ANNOT, FS_LETTER = 7, 6.5, 5.5, 5, 10
C_ADJ = '#E1812C'
COVARS = ('noise', 'decay')
OUTCOMES = [('control_r', 'Control $r$'), ('control_slope', 'Control slope')]


def load():
    ct = L.control_table()
    hp = pd.read_csv(paths.require(HP_CSV, hint=HP_HINT))
    m = ct.merge(hp, left_on=['monkey', 'model', 'unit'], right_on=['monkey', 'model', 'channel'])
    m['robust'] = m['robust'].astype(int)
    m['unit'] = m['unit'].astype(int)
    for y, _ in OUTCOMES:
        # joint-model partial residuals: hp effects removed, model + site structure kept
        m[y + '_res'] = partial_residuals(m, y, COVARS, keep='model', remove_site=False)
    return m


def _spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def per_model_panel(ax, m, y, ylabel):
    col = y + '_res'
    order = m.dropna(subset=[col]).groupby('model')[col].mean().sort_values().index.tolist()
    rng = np.random.default_rng(0)
    for xi, mdl in enumerate(order):
        d = m[m.model == mdl].dropna(subset=[col])
        ax.scatter(xi + rng.uniform(-0.28, 0.28, len(d)), d[col], s=6, color=get_model_color(mdl),
                   alpha=0.85, edgecolors='white', linewidths=0.2, zorder=4)
        ax.plot([xi - 0.34, xi + 0.34], [d[y].mean()] * 2, color='#b0b0b0', lw=1.2, zorder=5)
        ax.plot([xi - 0.34, xi + 0.34], [d[col].mean()] * 2, color='black', lw=1.2, zorder=6)
    ax.axhline(0, color='#888', lw=0.8, ls='--', zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[mm] for mm in order], rotation=45, ha='right', fontsize=FS_TICK)
    for t, mm in zip(ax.get_xticklabels(), order):
        t.set_color(get_model_color(mm)); t.set_fontweight('bold' if mm in ROBUST else 'normal')
    ax.set_ylabel(ylabel + ', hp regressed out', fontsize=FS_AX)
    ax.set_title(ylabel + ' residuals by model', fontsize=FS_TITLE, fontweight='bold',
                 fontfamily=FONT_FAMILY, pad=4)
    ax.tick_params(axis='y', labelsize=FS_TICK); _spines(ax)


def summary_panel(ax, m):
    """Raw (site-FE, within-site Wilcoxon) vs hp-adjusted (ANCOVA, site-clustered p),
    among the 9 trained models."""
    x = np.arange(len(OUTCOMES)); w = 0.34
    for k, (y, _) in enumerate(OUTCOMES):
        raw = adv_gap(m, y)
        adj = adv_gap(m, y, covariates=COVARS)
        for j, (g, c, lab) in enumerate([(raw, '#CCCCCC', 'Raw'),
                                         (adj, C_ADJ, 'hp-adjusted')]):
            xx = k + (j - 0.5) * w
            ax.bar(xx, g['delta'], w * 0.9, color=c, edgecolor='black', linewidth=0.4,
                   label=(lab if k == 0 else None))
            ax.text(xx, g['delta'] + 0.008, _sig_stars(g['p']),
                    ha='center', va='bottom', fontsize=FS_TICK)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in OUTCOMES], fontsize=FS_TICK)
    ax.set_ylabel(r'(Adv. $-$ conventional training)', fontsize=FS_AX)
    ax.set_title('Summary: baseline vs. hp-adjusted', fontsize=FS_TITLE, fontweight='bold',
                 fontfamily=FONT_FAMILY, pad=4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc='upper center', ncol=2, columnspacing=1.0,
              labelspacing=0.3, handlelength=1.0, handletextpad=0.35, bbox_to_anchor=(0.5, 1.02))
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def main(out_dir):
    m = load()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 72 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 3, figure=fig, left=0.07, right=0.985, top=0.775, bottom=0.30,
                  wspace=0.42, width_ratios=[1.3, 1.3, 0.85])
    letters = []
    a = fig.add_subplot(gs[0, 0]); per_model_panel(a, m, 'control_r', 'Control $r$'); letters.append(a)
    a = fig.add_subplot(gs[0, 1]); per_model_panel(a, m, 'control_slope', 'Control slope'); letters.append(a)
    a = fig.add_subplot(gs[0, 2]); summary_panel(a, m); letters.append(a)

    fig.canvas.draw()
    for ax, L_ in zip(letters, 'abc'):
        bb = ax.get_position()
        dx = 0.050 if L_ == 'c' else 0.032
        fig.text(bb.x0 - dx, bb.y1 + 0.016, L_, ha='left', va='bottom', fontsize=FS_LETTER,
                 fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.965, 'Control outcomes after regressing out the effect of synthesis hyperparameters',
             ha='center', va='top', fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.925, 'n = 250 site-models (10 models $\\times$ 5 channels $\\times$ 5 monkeys); '
             'noise & decay collinear ($r$=0.98)',
             ha='center', va='top', fontsize=FS_TICK, fontfamily=FONT_FAMILY, color='#555')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='#b0b0b0', lw=1.4, label='raw mean'),
                        Line2D([0], [0], color='black', lw=1.4, label='mean, hp regressed out')],
               fontsize=FS_ANNOT, frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.389, 0.898), handlelength=1.2, handletextpad=0.4, columnspacing=1.4)

    # console: raw vs adjusted gaps
    for y, lab in OUTCOMES:
        raw = adv_gap(m, y); adj = adv_gap(m, y, covariates=COVARS)
        print(f'{lab}: raw D={raw["delta"]:+.3f} (p={raw["p"]:.2g}, {raw["test"]}) | '
              f'hp-adjusted D={adj["delta"]:+.3f} (p={adj["p"]:.2g}, {adj["test"]})')

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
