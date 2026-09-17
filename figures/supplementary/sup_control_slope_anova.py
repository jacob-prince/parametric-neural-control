#!/usr/bin/env python3
r"""Supp fig (control_slope_anova) — variance decomposition of the control slope.

Slope variant of the Fig 4 model-differences ANOVA. Repeats the population analysis using
the control-fit SLOPE (measured-vs-predicted regression slope) in place of control r, and
recovers the same picture: models differ systematically, with the two robust models highest
and the Untrained baseline lowest. Number stamped from the manifest. Self-contained on
pnc.preproc.loader.

  a  Per-model distribution of control slope across the 25 recorded sites (violin + swarm,
     models sorted by mean slope, robust highlighted, Untrained its own class). Repeated-
     measures ANOVA (model within-factor, 25 sites) annotated.
  b  Per-region RM-ANOVA F (5 sites/region) — the model effect holds within every area.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS,
                       MONKEY_COLORS, get_model_color)
apply_figure_style()

_M = _fig('control_slope_anova'); STEM = output_name('control_slope_anova')
UNTR = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
ROBUST = set(ROBUST_MODELS)
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 11, 7, 6.5, 5.5, 5.5

REGION_ORDER = ['red', 'paul', 'venus', 'three0', 'leap']
REGION_LABEL = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'three0': 'STS(T)', 'leap': 'STS(L)'}


# ---- data -------------------------------------------------------------------
def rm_anova(M):
    """One-way repeated-measures ANOVA. rows=subjects(sites), cols=conditions(models).
    Returns (F, df_cond, df_err, p)."""
    M = np.asarray(M, float)
    n, k = M.shape
    g = M.mean()
    SS_cond = n * np.sum((M.mean(0) - g) ** 2)
    SS_subj = k * np.sum((M.mean(1) - g) ** 2)
    SS_tot = np.sum((M - g) ** 2)
    SS_err = SS_tot - SS_cond - SS_subj
    d1, d2 = k - 1, (n - 1) * (k - 1)
    F = (SS_cond / d1) / (SS_err / d2)
    return F, d1, d2, float(stats.f.sf(F, d1, d2))


def build():
    tab = L.control_table()
    models = list(L.config()['models'])
    # 25 sites x 10 models matrix of control slope
    piv = tab.pivot_table(index=['monkey', 'unit'], columns='model', values='control_slope')[models]
    assert piv.shape == (25, 10) and piv.isna().sum().sum() == 0, piv.shape
    F, d1, d2, p = rm_anova(piv.values)
    per_region = {}
    for mk in REGION_ORDER:
        sub = tab[tab.monkey == mk].pivot_table(index='unit', columns='model', values='control_slope')[models]
        per_region[mk] = rm_anova(sub.values)
    return tab, piv, (F, d1, d2, p), per_region


# ---- panels -----------------------------------------------------------------
def panel_a(ax, piv, anova):
    # order models by mean slope (low -> high); Untrained stays wherever its mean lands (lowest)
    means = piv.mean(0)
    order = means.sort_values().index.tolist()
    rng = np.random.default_rng(0)
    xs = np.arange(len(order))
    for xi, m in zip(xs, order):
        v = piv[m].values
        col = get_model_color(m)
        # violin (kde) body
        vp = ax.violinplot([v], positions=[xi], widths=0.7, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor(col); b.set_edgecolor('none'); b.set_alpha(0.22); b.set_zorder(1)
        # swarm: jittered dots, robust get a dark ring
        edge = '#222222' if m in ROBUST else 'white'
        ew = 0.55 if m in ROBUST else 0.3
        jit = rng.uniform(-0.16, 0.16, size=len(v))
        ax.scatter(xi + jit, v, s=8, color=col, alpha=0.9, edgecolors=edge,
                   linewidths=ew, zorder=3)
        # mean bar
        mu = float(np.mean(v))
        ax.plot([xi - 0.28, xi + 0.28], [mu, mu], color='black', lw=1.4,
                solid_capstyle='round', zorder=5)
    ax.axhline(0, color='#bbbbbb', lw=0.5, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in order], rotation=45, ha='right', fontsize=FS_TICK)
    for t, m in zip(ax.get_xticklabels(), order):
        t.set_color(get_model_color(m))
        if m in ROBUST:
            t.set_fontweight('bold')
    ax.set_ylabel('Control slope', fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.7, len(order) - 0.3)
    # reserve a headroom band above the data for the annotation + legend (no overlap with dots)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.32 * (hi - lo))
    F, d1, d2, p = anova
    ax.text(0.015, 0.985, f'RM-ANOVA  $F$({d1},{d2}) = {F:.1f}\n$p$ = {p:.1e}  ($n$ = 25 sites)',
            transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY)
    leg = [Line2D([0], [0], marker='o', ls='', mfc='#888', mec='#222222', mew=0.55, ms=3.4, label='Adversarially trained'),
           Line2D([0], [0], color='black', lw=1.4, label='model mean')]
    lg = ax.legend(handles=leg, fontsize=FS_ANNOT, frameon=True, loc='upper right',
                   bbox_to_anchor=(1.0, 1.0), handletextpad=0.35, labelspacing=0.25, borderpad=0.4)
    lg.get_frame().set(edgecolor='#BBBBBB', facecolor='white', linewidth=0.6)
    lg.set_zorder(20)


def panel_b(ax, per_region):
    xs = np.arange(len(REGION_ORDER))
    Fs = [per_region[mk][0] for mk in REGION_ORDER]
    ps = [per_region[mk][3] for mk in REGION_ORDER]
    cols = [MONKEY_COLORS[mk] for mk in REGION_ORDER]
    ax.bar(xs, Fs, color=cols, edgecolor='white', linewidth=0.4, zorder=3)
    for x, f, p in zip(xs, Fs, ps):
        star = '***' if p < 1e-3 else ('**' if p < 1e-2 else '*')
        ax.text(x, f + 0.25, star, ha='center', va='bottom', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY)
    ax.set_xticks(xs)
    ax.set_xticklabels([REGION_LABEL[mk] for mk in REGION_ORDER], rotation=45, ha='right', fontsize=FS_TICK)
    for t, mk in zip(ax.get_xticklabels(), REGION_ORDER):
        t.set_color(MONKEY_COLORS[mk])
    d1, d2 = per_region[REGION_ORDER[0]][1], per_region[REGION_ORDER[0]][2]
    ax.set_ylabel(f'RM-ANOVA $F$({d1},{d2})', fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, max(Fs) * 1.18)


def main(out_dir):
    tab, piv, anova, per_region = build()
    F, d1, d2, p = anova
    print(f'control_slope RM-ANOVA F({d1},{d2}) = {F:.4f}  p = {p:.4e}')
    for mk in REGION_ORDER:
        f, a1, a2, pp = per_region[mk]
        print(f'  {REGION_LABEL[mk]:8s} F({a1},{a2}) = {f:.3f}  p = {pp:.3e}')

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 66 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 2, figure=fig, left=0.075, right=0.985, top=0.85, bottom=0.28,
                  wspace=0.32, width_ratios=[2.4, 1.0])
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    panel_a(ax_a, piv, anova)
    panel_b(ax_b, per_region)
    for ax in (ax_a, ax_b):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    titles = ['Control slope by model', 'Model effect by area']
    fig.canvas.draw()
    for ax, letter, title in zip((ax_a, ax_b), 'ab', titles):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.018, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
        fig.text(bb.x0 - 0.052, bb.y1 + 0.06, letter, ha='left', va='top', fontsize=FS_LET,
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
