#!/usr/bin/env python3
r"""Supp Fig (hyperparam_selection) — automated synthesis-regularization selection.

An automated procedure identified a suitable regularization regime independently for each of
the 250 model-site combinations. It stepped up a ladder of (augmentation-noise, spectral-decay
alpha) regimes ordered least-to-most aggressive and stopped at the first rung whose synthesis
success rate exceeded 0.95. This is the SELECTION-method figure (a companion supplementary figure shows the robust
advantage survives regressing the selected regime out).

  a  The 5-rung ladder and the selected regime for all 250 model-sites (dots dodged on each rung,
     coloured by animal) - most sites land on the two least-aggressive rungs.
  b  Per-model mean selected rung (0 = least aggressive), ordered - models differ modestly in the
     aggressiveness they require.
  c  Distribution of the achieved success rate at the selected rung; the procedure targets > 0.95
     and returns the best-available regime, so most (but not all) sites clear the 0.95 line.

Number stamped from the manifest. Self-contained: reads only the cache
preprocessed_data/sup_hyperparam_hp_tuning_selected.csv (rebuilt from the production-log
recovery json by scripts/preprocessing/build_hp_tuning_cache.py).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS,
                       get_model_color, MODEL_ORDER)
apply_figure_style()

_M = _fig('hyperparam_selection'); STEM = output_name('hyperparam_selection')
CSV = os.path.join(str(paths.preprocessed_data()), 'sup_hyperparam_hp_tuning_selected.csv')

SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'   # display relabel
ROBUST = set(ROBUST_MODELS)
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
MONKEY_REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
MONKEY_NAME = {'red': 'Monkey R', 'paul': 'Monkey P', 'venus': 'Monkey V',
               'leap': 'Monkey L', 'three0': 'Monkey T'}

# The regularization ladder: (augmentation noise, spectral-decay alpha), least -> most aggressive
# (rung 0 = least aggressive = first visited by the search).
LADDER = [(0.05, 1.2), (0.10, 1.5), (0.15, 2.0), (0.25, 2.4), (0.30, 2.5)]
STOP = 0.95                                          # success-rate stopping threshold

FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 8, 7, 6.5, 5.5, 5.5


# ── data (self-contained cache) ──────────────────────────────────────────────
def load():
    df = pd.read_csv(paths.require(CSV, hint='python scripts/preprocessing/build_hp_tuning_cache.py'))
    rung = {(round(n, 2), round(d, 2)): i for i, (n, d) in enumerate(LADDER)}
    df['rung'] = df.apply(lambda r: rung[(round(r.noise, 2), round(r.decay, 2))], axis=1)
    return df


def spread_ties(vals, xcenter, halfwidth):
    """Evenly dodge points that share a discrete value so none overlap."""
    from collections import defaultdict
    xs = np.full(len(vals), float(xcenter))
    groups = defaultdict(list)
    for i, v in enumerate(vals):
        groups[round(float(v), 6)].append(i)
    for _, idxs in groups.items():
        offs = [0.0] if len(idxs) == 1 else np.linspace(-halfwidth, halfwidth, len(idxs))
        for k, i in enumerate(idxs):
            xs[i] = xcenter + offs[k]
    return xs


# ── panel a: the ladder + selected rung for all 250 sites ─────────────────────
def panel_a(ax, df, color_by='monkey', legend=True, show_yaxis=True):
    rng = np.random.default_rng(7)
    for r in range(len(LADDER)):
        sub = df[df.rung == r]
        if color_by == 'model':                          # group the dodge by model (panel a keeps its animal grouping)
            om = {m: i for i, m in enumerate(MODEL_ORDER)}
            sub = sub.iloc[np.argsort([om.get(m, 99) for m in sub['model']], kind='stable')]
        n = len(sub)
        # dodge across the rung width; jitter vertically inside the rung band
        xs = np.linspace(-0.36, 0.36, n) if n > 1 else np.array([0.0])
        xs = xs + rng.uniform(-0.006, 0.006, n)
        ys = r + rng.uniform(-0.16, 0.16, n)
        cols = ([MONKEY_COLORS[mk] for mk in sub.monkey] if color_by == 'monkey'
                else [get_model_color(m) for m in sub.model])
        ax.scatter(xs, ys, s=5.0, c=cols, alpha=0.85, edgecolors='white',
                   linewidths=0.15, zorder=4)
        ax.text(0.52, r, f'n = {n}', ha='left', va='center', fontsize=FS_ANNOT,
                fontfamily=FONT_FAMILY, color='#333', zorder=6)
    ax.set_yticks(range(len(LADDER)))
    ax.set_yticklabels([f'{i}  ({n:g}, {a:g})' for i, (n, a) in enumerate(LADDER)]
                       if show_yaxis else [], fontsize=FS_TICK)
    ax.set_ylim(-0.55, len(LADDER) - 0.4)
    ax.set_xlim(-0.6, 0.95)
    ax.invert_yaxis()                                # rung 0 (first visited) on top
    if show_yaxis:
        ax.set_ylabel('Ladder rung  (noise, ' + r'$\alpha$' + ')', fontsize=FS_AX)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=FS_TICK, length=0)
    ax.text(0.52, -0.42, 'least aggressive', ha='left', va='center', fontsize=FS_ANNOT,
            color='#999', fontfamily=FONT_FAMILY)
    ax.text(0.52, len(LADDER) - 0.62, 'most aggressive', ha='left', va='center',
            fontsize=FS_ANNOT, color='#999', fontfamily=FONT_FAMILY)
    if legend:
        if color_by == 'monkey':
            leg = [Line2D([0], [0], marker='o', ls='', mfc=MONKEY_COLORS[mk], mec='white', mew=0.2,
                          ms=2.6, label=f'{MONKEY_NAME[mk]} ({MONKEY_REGION[mk]})') for mk in MONKEYS]
            ncol = 3
        else:
            leg = [Line2D([0], [0], marker='o', ls='', mfc=get_model_color(m), mec='white', mew=0.2,
                          ms=2.6, label=SHORT[m]) for m in MODEL_ORDER]
            ncol = 2
        ax.legend(handles=leg, fontsize=FS_ANNOT - 0.5, frameon=False,
                  loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=ncol,
                  handletextpad=0.2, columnspacing=0.7, labelspacing=0.2, borderpad=0.1)


# ── panel b: per-model mean selected rung ─────────────────────────────────────
def panel_b(ax, df):
    means = df.groupby('model').rung.mean()
    order = means.sort_values().index.tolist()
    xs = np.arange(len(order))
    vals = [means[m] for m in order]
    cols = [get_model_color(m) for m in order]
    ax.bar(xs, vals, color=cols, edgecolor='white', linewidth=0.4, zorder=3)
    for m, x in zip(order, xs):
        if m in ROBUST:
            ax.scatter(x, means[m] + 0.055, marker='v', s=8, color='#E1812C',
                       edgecolors='none', zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in order], rotation=55, ha='right', fontsize=FS_TICK)
    for t, m in zip(ax.get_xticklabels(), order):
        t.set_color(get_model_color(m))
    ax.set_ylabel('Mean selected rung', fontsize=FS_AX)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.legend(handles=[Line2D([0], [0], marker='v', ls='', mfc='#E1812C', mec='none',
                              ms=3.2, label='Adversarially trained')],
              fontsize=FS_ANNOT, frameon=False, loc='upper left',
              handletextpad=0.2, borderpad=0.2)


# ── panel c: success-rate distribution at the selected rung ───────────────────
def panel_c(ax, df):
    sr = df.success_rate.values
    ax.hist(sr, bins=np.linspace(0.30, 1.0, 29), color='#3274A1', alpha=0.85,
            edgecolor='white', linewidth=0.3, zorder=3)
    ax.axvline(STOP, color='#E63946', ls='--', lw=1.0, zorder=5)
    ax.text(STOP - 0.012, ax.get_ylim()[1] * 0.98, f'target {STOP:g}', ha='right', va='top',
            fontsize=FS_ANNOT, color='#E63946', fontfamily=FONT_FAMILY, zorder=6)
    frac = np.mean(sr >= STOP)
    ax.set_xlabel('Success rate at selected rung', fontsize=FS_AX)
    ax.set_ylabel('Model-sites', fontsize=FS_AX)
    ax.set_xlim(0.30, 1.02)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.96, f'{frac*100:.0f}% $\\geq$ {STOP:g}', transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY, color='#222')
    return frac


def main(out_dir):
    df = load()
    r_noise_decay = np.corrcoef(df.noise, df.decay)[0, 1]
    n_regimes = df.groupby(['noise', 'decay']).ngroups
    print('N site-models:', len(df))
    print('distinct regimes:', n_regimes)
    print('noise-decay r:', round(r_noise_decay, 3))
    print('rung counts:', df.rung.value_counts().sort_index().to_dict())

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 68 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 4, figure=fig, left=0.055, right=0.99, top=0.85, bottom=0.38,
                  wspace=0.42, width_ratios=[1.30, 1.30, 1.0, 1.0])
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2]); axD = fig.add_subplot(gs[0, 3])
    panel_a(axA, df, color_by='monkey', legend=True)                    # a: ladder coloured by animal
    panel_a(axB, df, color_by='model', legend=True, show_yaxis=False)   # b: SAME ladder, model colours (shares a's y-axis)
    panel_b(axC, df); frac = panel_c(axD, df)             # c/d: former b/c pushed over

    for ax in (axA, axB, axC, axD):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    titles = ['Selection ladder (250 model-sites)', 'Selection ladder (by model)',
              'Aggressiveness per model', 'Success at selected rung']
    fig.canvas.draw()
    letter_dx = dict(a=0.052, b=0.018, c=0.045, d=0.045)   # b has no y-axis, so tuck its letter closer
    for ax, letter, title in zip((axA, axB, axC, axD), 'abcd', titles):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.018, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
        fig.text(bb.x0 - letter_dx[letter], bb.y1 + 0.075, letter, ha='left', va='top', fontsize=FS_LET,
                 fontweight='bold', fontfamily=FONT_FAMILY)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    print(f'fraction success>= {STOP}: {frac*100:.1f}%')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
