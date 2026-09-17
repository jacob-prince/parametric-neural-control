#!/usr/bin/env python3
r"""Supp Fig (superstimuli) — robust models yield the most super-stimuli.

A super-stimulus is an accentuated image whose MEASURED control response falls beyond the
extremes of a site's natural (calibration) response distribution. Super-drive = measured
response above the site q99; super-suppress = below q01, where q01/q99 are the 1st/99th
percentiles of the natural encoding-set measured responses at that site. Number stamped from
the manifest. Self-contained on pnc.preproc.loader.

  a  Schematic of the super-drive (> q99) / super-suppress (< q01) tails of a site's natural
     response distribution.
  b  Per-model super-stimulus proportion: mean over 25 sites (bar) with per-site points; the
     two robust models carry the highest proportion.
  c  Decomposition of the mean proportion into super-drive vs super-suppress per model.
"""
import os

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats as spstats

apply_figure_style()
_M = _fig('superstimuli'); STEM = output_name('superstimuli')

UNTR = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
ROBUST = set(ROBUST_MODELS)

C_DRIVE = '#C0392B'      # upper tail
C_SUPP = '#2E6DB4'       # lower tail
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 8, 7, 6.5, 5.5, 5.5


# ---- data (self-contained) --------------------------------------------------
def build():
    cfg = L.config(); models = cfg['models']
    rows = []
    for mk in cfg['monkeys']:
        b = L.load_brain(mk); units = [int(u) for u in b['units']]
        for u in units:
            # q01 / q99 from the natural (calibration) measured response distribution
            _, em = L.encoding_cloud(mk, u, models[0], 'all')
            q01, q99 = np.percentile(em, [1, 99])
            for m in models:
                _, me = L.control_cloud(mk, u, m)
                if len(me) == 0:
                    continue
                drive = 100.0 * np.mean(me > q99)
                supp = 100.0 * np.mean(me < q01)
                rows.append(dict(monkey=mk, unit=u, model=m, robust=(m in ROBUST),
                                 pct_super=drive + supp, pct_drive=drive, pct_supp=supp))
    return pd.DataFrame(rows)


# ---- panels -----------------------------------------------------------------
def panel_a(ax):
    x = np.linspace(-4, 4, 600); y = spstats.norm.pdf(x)
    lo, hi = spstats.norm.ppf(0.01), spstats.norm.ppf(0.99)
    ax.plot(x, y, color='#555', lw=0.9, zorder=3)
    ax.fill_between(x[x <= lo], y[x <= lo], color=C_SUPP, alpha=0.55, lw=0, zorder=2)
    ax.fill_between(x[x >= hi], y[x >= hi], color=C_DRIVE, alpha=0.55, lw=0, zorder=2)
    ax.fill_between(x[(x > lo) & (x < hi)], y[(x > lo) & (x < hi)], color='#DDDDDD', lw=0, zorder=1)
    ax.axvline(lo, color=C_SUPP, ls='--', lw=0.7); ax.axvline(hi, color=C_DRIVE, ls='--', lw=0.7)
    ax.annotate('super-suppress\n(< q01)', xy=(lo - 0.4, 0.03), xytext=(-3.4, 0.30),
                fontsize=FS_ANNOT, color=C_SUPP, ha='center', va='center', fontfamily=FONT_FAMILY,
                arrowprops=dict(arrowstyle='-', color=C_SUPP, lw=0.6))
    ax.annotate('super-drive\n(> q99)', xy=(hi + 0.4, 0.03), xytext=(3.4, 0.30),
                fontsize=FS_ANNOT, color=C_DRIVE, ha='center', va='center', fontfamily=FONT_FAMILY,
                arrowprops=dict(arrowstyle='-', color=C_DRIVE, lw=0.6))
    ax.text(0, 0.16, 'natural\nrange', fontsize=FS_ANNOT, color='#777', ha='center', va='center',
            fontfamily=FONT_FAMILY)
    ax.set_xlabel('Natural response (z)', fontsize=FS_AX)
    ax.set_ylabel('Density', fontsize=FS_AX)
    ax.set_xlim(-4, 4); ax.set_ylim(0, 0.44); ax.set_yticks([])
    ax.set_xticks([-3, 0, 3])
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


def panel_b(ax, df):
    g = df.groupby('model')['pct_super'].mean()
    order = g.sort_values().index.tolist()
    xs = np.arange(len(order))
    means = [g[m] for m in order]
    cols = [get_model_color(m) for m in order]
    ax.barh(xs, means, color=cols, edgecolor='white', linewidth=0.4, height=0.72, zorder=2)
    # per-site points
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = df[df.model == m]['pct_super'].values
        jit = (rng.random(len(v)) - 0.5) * 0.42
        ax.scatter(v, np.full(len(v), i) + jit, s=3.5, color='#333333', alpha=0.5,
                   edgecolors='none', zorder=3)
    ax.set_yticks(xs)
    ax.set_yticklabels([SHORT[m] for m in order], fontsize=FS_TICK)
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(get_model_color(m))
        if m in ROBUST:
            t.set_fontweight('bold')
    ax.set_xlabel('Super-stimuli (% of accentuations)', fontsize=FS_AX)
    ax.set_xlim(0, max(df['pct_super'].max() * 1.02, 30))
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.tick_params(axis='y', length=0)
    leg = [Line2D([0], [0], marker='o', ls='', mfc='#333333', mec='none', ms=2.6, label='site')]
    ax.legend(handles=leg, fontsize=FS_ANNOT, frameon=False, loc='lower right',
              handletextpad=0.3, labelspacing=0.25, borderpad=0.2)


def panel_c(ax, df):
    g = df.groupby('model')[['pct_super', 'pct_drive', 'pct_supp']].mean()
    order = g['pct_super'].sort_values(ascending=False).index.tolist()
    xs = np.arange(len(order)); w = 0.4
    drive = [g.loc[m, 'pct_drive'] for m in order]
    supp = [g.loc[m, 'pct_supp'] for m in order]
    ax.bar(xs - w / 2, drive, w, color=C_DRIVE, edgecolor='white', linewidth=0.3, label='super-drive', zorder=2)
    ax.bar(xs + w / 2, supp, w, color=C_SUPP, edgecolor='white', linewidth=0.3, label='super-suppress', zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in order], rotation=45, ha='right', fontsize=FS_TICK)
    for t, m in zip(ax.get_xticklabels(), order):
        t.set_color(get_model_color(m))
        if m in ROBUST:
            t.set_fontweight('bold')
    ax.set_ylabel('Mean proportion (%)', fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, max(max(drive), max(supp)) * 1.15)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc='upper right', handlelength=1.0,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.2)


def main(out_dir):
    df = build()
    g = df.groupby('model')['pct_super'].mean().sort_values(ascending=False)
    top2 = list(g.index[:2])
    print('per-model mean super-stim %:')
    print(g.round(2))
    print('robust mean:', round(df[df.robust]['pct_super'].mean(), 2),
          ' non-robust(excl Untrained):',
          round(df[(~df.robust) & (df.model != UNTR)]['pct_super'].mean(), 2))
    print('top-2 models:', top2, '-> both robust:', set(top2) == ROBUST)

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 66 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 3, figure=fig, left=0.065, right=0.99, top=0.85, bottom=0.30,
                  wspace=0.55, width_ratios=[0.82, 1.15, 1.05])
    ax_a = fig.add_subplot(gs[0, 0]); panel_a(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); panel_b(ax_b, df)
    ax_c = fig.add_subplot(gs[0, 2]); panel_c(ax_c, df)

    titles = ['Super-stimulus definition', 'Super-stimulus proportion per model',
              'Super-drive vs super-suppress']
    fig.canvas.draw()
    for a, letter, title in zip([ax_a, ax_b, ax_c], 'abc', titles):
        bb = a.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.02, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
        fig.text(bb.x0 - 0.05, bb.y1 + 0.075, letter, ha='left', va='top', fontsize=FS_LET,
                 fontweight='bold', fontfamily=FONT_FAMILY)

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
