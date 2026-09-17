#!/usr/bin/env python3
r"""Supp fig (robust_by_site) — per-site adversarially-trained control advantage.

Control score = Pearson r between a model's accentuation scores and the measured neural
response to its accentuated stimuli (control_r in the canonical control_table). Per site
the 10 models are ranked by control_r. A robust model (RN50-Robust or CLIPAG) is the single
strongest controller at 20/25 sites (80%) and among the top two at 24/25 (96%). The per-area
robust advantage Delta r = mean(2 robust) - mean(7 non-robust CNN/ViT, Untrained excluded)
is largest in aIT (+0.37). Number stamped from the manifest. Self-contained on
pnc.preproc.loader.

  Single panel: per-site control r for all 10 models, colored by group; the best model at each
  site is ringed, and sites are grouped by animal / recorded area. (The rank-of-best-robust and
  per-area advantage summaries are still computed and printed to the console.)
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color)
apply_figure_style()
_M = _fig('robust_by_site'); STEM = output_name('robust_by_site')

UNTR = 'AlexNet_training_seed_01'
ROBUST = set(ROBUST_MODELS)
AREA = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 11, 7, 6.5, 5.5, 5.5
RING = '#111111'


def build():
    cfg = L.config()
    models = list(cfg['models']); monkeys = list(cfg['monkeys'])
    tab = L.control_table()
    piv = tab.pivot_table(index=['monkey', 'unit'], columns='model', values='control_r')[models]

    # per-site rank of each model, and rank of the best robust model
    sites = list(piv.index)                       # (monkey, unit)
    RR = {s: {m: float(piv.loc[s, m]) for m in models} for s in sites}
    is_rob = np.array([m in ROBUST for m in models])
    rank1 = top2 = 0
    rank_best_robust = []
    for s in sites:
        vals = piv.loc[s].values.astype(float)
        order = np.argsort(-vals, kind='stable')
        if is_rob[order[0]]:
            rank1 += 1
        if is_rob[order[:2]].any():
            top2 += 1
        rank_best_robust.append(next(i + 1 for i, oi in enumerate(order) if is_rob[oi]))
    rank_best_robust = np.array(rank_best_robust)

    # per-area (per-animal) advantage Delta r = mean(robust) - mean(non-robust EXCL Untrained)
    delta = {}
    for mk in monkeys:
        sub = tab[tab['monkey'] == mk]
        rob = sub[sub['robust'] == 1]['control_r'].mean()
        non = sub[(sub['robust'] == 0) & (sub['model'] != UNTR)]['control_r'].mean()
        delta[mk] = (float(rob), float(non), float(rob - non))

    n = len(sites)
    stats = dict(n=n, rank1=rank1, top2=top2,
                 pct1=100 * rank1 / n, pct2=100 * top2 / n,
                 rank_best_robust=rank_best_robust)
    return models, monkeys, sites, RR, delta, stats


# ---- panel a: per-site control r, all 10 models, best ringed ----------------
def panel_a(ax, models, monkeys, sites, RR):
    rng = np.random.default_rng(0)
    xpos, spans = {}, {}
    xi = 0.0
    order_sites = []
    for mk in monkeys:
        start = xi
        for (m, u) in sites:
            if m == mk:
                xpos[(m, u)] = xi; order_sites.append((m, u)); xi += 1
        spans[mk] = (start, xi - 1); xi += 1.0

    allr = [r for s in order_sites for r in RR[s].values()]
    ymin = min(allr)
    for s in order_sites:
        x = xpos[s]; rr = RR[s]
        ms = list(rr)
        jx = x + rng.uniform(-0.30, 0.30, len(ms))
        for m, xj in zip(ms, jx):
            rob = m in ROBUST
            ax.scatter(xj, rr[m], s=13 if rob else 9,
                       color=get_model_color(m),
                       edgecolors='black' if rob else 'white',
                       linewidths=0.5 if rob else 0.25, alpha=0.95, zorder=4)
        best = max(rr, key=rr.get)
        bx = jx[ms.index(best)]
        ax.scatter(bx, rr[best], s=42, facecolor='none', edgecolors=RING,
                   linewidths=0.9, zorder=6)

    ax.axhline(0, color='#BBBBBB', lw=0.5, zorder=1)
    tr = ax.get_xaxis_transform()
    for mk in monkeys:
        lo, hi = spans[mk]
        ax.plot([lo - 0.4, hi + 0.4], [-0.14, -0.14], transform=tr,
                color=MONKEY_COLORS[mk], lw=2.2, solid_capstyle='butt',
                clip_on=False, zorder=3)
        ax.text((lo + hi) / 2, -0.185, f'{TAG[mk]} {AREA[mk]}', transform=tr,
                ha='center', va='top', fontsize=FS_TICK, color=MONKEY_COLORS[mk],
                fontweight='bold', fontfamily=FONT_FAMILY)

    ax.set_xticks([xpos[s] for s in order_sites])
    ax.set_xticklabels([f'{TAG[m]}{u}' for (m, u) in order_sites], fontsize=FS_TICK - 1.5)
    for t, (m, u) in zip(ax.get_xticklabels(), order_sites):
        t.set_color(MONKEY_COLORS[m])
    ax.set_xlim(-0.7, max(xpos.values()) + 0.7)
    ax.set_ylim(ymin - 0.08, 1.0)
    ax.set_ylabel('Control score (Pearson r)', fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='x', length=0, pad=1)


def main(out_dir):
    models, monkeys, sites, RR, delta, stats = build()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 58 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 1, figure=fig, left=0.075, right=0.985, top=0.73, bottom=0.145)
    axA = fig.add_subplot(gs[0, 0])
    panel_a(axA, models, monkeys, sites, RR)

    # per-model color key + robust/non-robust marker style + ring (top strip)
    SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
    handles = []
    for m in models:
        rob = m in ROBUST
        handles.append(Line2D([0], [0], marker='o', ls='', mfc=get_model_color(m),
                              mec='black' if rob else 'white', mew=0.6 if rob else 0.3,
                              ms=5, label=SHORT[m]))
    handles.append(Line2D([0], [0], marker='o', ls='', mfc='none', mec=RING, mew=0.9,
                          ms=6.5, label='best at site'))
    fig.legend(handles=handles, loc='upper right', ncol=6, fontsize=FS_ANNOT,
               frameon=False, bbox_to_anchor=(0.985, 0.995), columnspacing=1.0,
               handletextpad=0.25, labelspacing=0.5)

    # single-panel figure: one centered title, no a/b/c panel letters
    fig.text(0.53, 0.81, _M['title'], fontsize=FS_TITLE + 1, fontfamily=FONT_FAMILY,
             fontweight='bold', ha='center', va='top')

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    print(f"  rank1 {stats['rank1']}/{stats['n']} ({stats['pct1']:.1f}%), "
          f"top2 {stats['top2']}/{stats['n']} ({stats['pct2']:.1f}%)")
    for mk in monkeys:
        print(f"  {mk:7s} {AREA[mk]:6s} Delta r = {delta[mk][2]:+.3f}")
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
