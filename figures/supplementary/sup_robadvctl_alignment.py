#!/usr/bin/env python3
r"""Supp fig (robadvctl_alignment) - relationship between encoding-axis alignment and
neural control outcomes, and the control comparison with alignment regressed out.

Axis alignment (per-dim displacement ratio, seed-averaged per site-model) is a
post-treatment property produced by training, so the adjusted comparison is a
sensitivity analysis. Residuals use coefficients from the JOINT model
`outcome ~ C(model) + align + C(site)` so model structure is never absorbed into the
alignment slope; site variance stays in the display.

  a,b Axis alignment vs control r / control slope across the 225 trained site-models
      (pooled fit line + Pearson r).
  c,d Control r / control slope with alignment regressed out, by model (per-model
      swarms + means, all 10 models, sorted by residual mean).
  e   Summary: raw adv-vs-conventional gap (site-fixed-effects Delta among the 9
      trained models, p = within-site paired Wilcoxon over 25 sites) vs the
      alignment-ADJUSTED gap (ANCOVA `outcome ~ adv + align + C(site)`,
      site-clustered p). Untrained is shown in c,d but never enters the family tests.

Reads loader.load_axis_alignment() + control_table().
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
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_SHORT_NAMES, ROBUST_MODELS, get_model_color)
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.famstats import adv_gap, partial_residuals, stars as _sig_stars
apply_figure_style()

STEM = output_name('robadvctl_alignment')
UNTR = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
ROBUST = set(ROBUST_MODELS)
FS_TITLE, FS_AX, FS_TICK, FS_ANNOT, FS_LETTER = 7, 6.5, 5.5, 5, 10
OUTCOMES = [('control_r', 'Control $r$'), ('control_slope', 'Control slope')]
C_ADJ = '#E1812C'


def build():
    """All-10 table with per-site alignment + joint-model residuals; `robust` int flag."""
    a = L.load_axis_alignment()
    mk, un, mo = np.array(a['monkey']), np.array(a['unit']), np.array(a['model'])
    ratio = a['DM'] / (a['DOM'] / np.sqrt(a['n_off']))
    agg = {}
    for i in range(len(ratio)):
        agg.setdefault((str(mk[i]), int(un[i]), str(mo[i])), []).append(ratio[i])
    align = {k: float(np.mean(v)) for k, v in agg.items()}     # seed-averaged per site-model

    ct = L.control_table(); rows = []
    for _, r in ct.iterrows():
        k = (r['monkey'], int(r['unit']), r['model'])
        if k in align and np.isfinite(r['control_r']) and np.isfinite(r['control_slope']):
            rows.append(dict(monkey=r['monkey'], unit=int(r['unit']), model=r['model'],
                             align=align[k], control_r=float(r['control_r']),
                             control_slope=float(r['control_slope']),
                             robust=int(r['model'] in ROBUST)))
    df = pd.DataFrame(rows)
    for y, _ in OUTCOMES:
        # joint-model partial residuals: alignment effect removed, model + site structure kept
        df[y + '_res'] = partial_residuals(df, y, ('align',), keep='model', remove_site=False)
    return df


def _spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def scatter_panel(ax, tr, ycol, ylab):
    """Alignment vs control outcome across the 225 trained site-models (pooled fit)."""
    x, y = tr['align'].values, tr[ycol].values
    for _, r in tr.iterrows():
        ax.scatter(r['align'], r[ycol], s=9, color=get_model_color(r['model']),
                   edgecolors=('black' if r['robust'] else 'white'),
                   linewidths=(0.55 if r['robust'] else 0.2), alpha=0.85, zorder=3)
    b = np.polyfit(x, y, 1); xs = np.array([x.min(), x.max()])
    ax.plot(xs, np.polyval(b, xs), color='#333', lw=1.0, zorder=4)
    rr, pp = stats.pearsonr(x, y)
    ax.text(0.97, 0.04, f'$r$ = {rr:+.2f}\n$p$ = {pp:.0e}\n$n$ = {len(tr)}',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
            fontfamily=FONT_FAMILY, linespacing=1.25,
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.85))
    ax.set_xlabel('Axis alignment (per-dim displacement ratio)', fontsize=FS_AX)
    ax.set_ylabel(ylab, fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def per_model_panel(ax, df, y, ylabel):
    col = y + '_res'
    order = df.dropna(subset=[col]).groupby('model')[col].mean().sort_values().index.tolist()
    rng = np.random.default_rng(0)
    for xi, mdl in enumerate(order):
        d = df[df.model == mdl].dropna(subset=[col])
        ax.scatter(xi + rng.uniform(-0.28, 0.28, len(d)), d[col], s=6, color=get_model_color(mdl),
                   alpha=0.85, edgecolors='white', linewidths=0.2, zorder=4)
        ax.plot([xi - 0.34, xi + 0.34], [d[y].mean()] * 2, color='#b0b0b0', lw=1.2, zorder=5)
        ax.plot([xi - 0.34, xi + 0.34], [d[col].mean()] * 2, color='black', lw=1.2, zorder=6)
    ax.axhline(0, color='#888', lw=0.8, ls='--', zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[mm] for mm in order], rotation=45, ha='right', fontsize=FS_TICK)
    for t, mm in zip(ax.get_xticklabels(), order):
        t.set_color(get_model_color(mm)); t.set_fontweight('bold' if mm in ROBUST else 'normal')
    ax.set_ylabel(ylabel + ', alignment regressed out', fontsize=FS_AX)
    ax.set_title(ylabel + ' residuals by model', fontsize=FS_TITLE, fontweight='bold',
                 fontfamily=FONT_FAMILY, pad=4)
    ax.tick_params(axis='y', labelsize=FS_TICK); _spines(ax)


def summary_panel(ax, df):
    """Raw (site-FE, within-site Wilcoxon) vs alignment-adjusted (ANCOVA, site-clustered
    p), among the 9 trained models."""
    x = np.arange(len(OUTCOMES)); w = 0.34
    for k, (y, _) in enumerate(OUTCOMES):
        raw = adv_gap(df, y)
        adj = adv_gap(df, y, covariates=('align',))
        for j, (g, c, lab) in enumerate([(raw, '#CCCCCC', 'Raw'),
                                         (adj, C_ADJ, 'alignment-adjusted')]):
            xx = k + (j - 0.5) * w
            ax.bar(xx, g['delta'], w * 0.9, color=c, edgecolor='black', linewidth=0.4,
                   label=(lab if k == 0 else None))
            ax.text(xx, g['delta'] + 0.008, _sig_stars(g['p']),
                    ha='center', va='bottom', fontsize=FS_TICK)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in OUTCOMES], fontsize=FS_TICK)
    ax.set_ylabel(r'(Adv. $-$ conventional training)', fontsize=FS_AX)
    ax.set_title('Summary: baseline vs. adjusted', fontsize=FS_TITLE, fontweight='bold',
                 fontfamily=FONT_FAMILY, pad=4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc='upper center', ncol=1, columnspacing=1.0,
              labelspacing=0.3, handlelength=1.0, handletextpad=0.35, bbox_to_anchor=(0.5, 1.02))
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def main(out_dir):
    df = build()
    tr = df[df.model != UNTR]

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 124 * MM), facecolor=FIG_FACECOLOR)
    gs_top = GridSpec(1, 2, figure=fig, left=0.07, right=0.985, top=0.885, bottom=0.615,
                      wspace=0.30)
    gs_bot = GridSpec(1, 3, figure=fig, left=0.07, right=0.985, top=0.46, bottom=0.175,
                      wspace=0.42, width_ratios=[1.3, 1.3, 0.85])
    letters = []
    a = fig.add_subplot(gs_top[0, 0]); scatter_panel(a, tr, 'control_r', 'Control $r$'); letters.append(a)
    a.set_title('Axis alignment vs control $r$', fontsize=FS_TITLE, fontweight='bold',
                fontfamily=FONT_FAMILY, pad=4)
    a = fig.add_subplot(gs_top[0, 1]); scatter_panel(a, tr, 'control_slope', 'Control slope'); letters.append(a)
    a.set_title('Axis alignment vs control slope', fontsize=FS_TITLE, fontweight='bold',
                fontfamily=FONT_FAMILY, pad=4)
    a = fig.add_subplot(gs_bot[0, 0]); per_model_panel(a, df, 'control_r', 'Control $r$'); letters.append(a)
    a = fig.add_subplot(gs_bot[0, 1]); per_model_panel(a, df, 'control_slope', 'Control slope'); letters.append(a)
    a = fig.add_subplot(gs_bot[0, 2]); summary_panel(a, df); letters.append(a)

    fig.canvas.draw()
    for ax, L_ in zip(letters, 'abcde'):
        bb = ax.get_position()
        dx = 0.050 if L_ == 'e' else 0.036
        fig.text(bb.x0 - dx, bb.y1 + 0.012, L_, ha='left', va='bottom', fontsize=FS_LETTER,
                 fontweight='bold', fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.975, 'Relationship between axis alignment and neural control outcomes',
             ha='center', va='top', fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='#b0b0b0', lw=1.4, label='raw mean'),
                        Line2D([0], [0], color='black', lw=1.4, label='mean, alignment regressed out')],
               fontsize=FS_ANNOT, frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.40, 0.535), handlelength=1.2, handletextpad=0.4, columnspacing=1.4)

    # console: raw vs adjusted gaps
    for y, lab in OUTCOMES:
        raw = adv_gap(df, y); adj = adv_gap(df, y, covariates=('align',))
        print(f'{lab}: raw D={raw["delta"]:+.3f} (p={raw["p"]:.2g}, {raw["test"]}) | '
              f'alignment-adjusted D={adj["delta"]:+.3f} (p={adj["p"]:.2g}, {adj["test"]})')

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
