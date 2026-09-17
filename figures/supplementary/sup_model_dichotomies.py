#!/usr/bin/env python3
r"""Supp fig (model_dichotomies) — neural control does not dissociate by architecture, training objective, or
language alignment; only by adversarial training.

With 10 models there are many ways to carve the set. This figure asks whether any of the natural
architectural/training dichotomies separates models on parametric control the way adversarial
training does. It does not: among the 9 trained models, CNN vs Transformer, label-supervised vs
self-supervised (CLIP-family counted as self-supervised — no human labels), and language-aligned
vs not are all null; only adversarially-trained vs standard shows a large, reliable gap.

Outcome = control r (the Fig 4 headline; control slope is concordant — see printout). Each split is
tested WITHIN SITE: for every recording site the mean control r of each group's models is computed,
and the two group means are compared across the 25 sites (Wilcoxon signed-rank). The two
adversarially-trained models (RN50-Robust, CLIPAG) sit on opposite sides of every split;
because they dominate the small groups, each architecture/objective/language panel also reports the
p EXCLUDING them — the only hint of an effect (objective, control r) vanishes there.

  a  Architecture         CNN vs Transformer
  b  Training objective   supervised (labels) vs self-supervised (incl. CLIP-family)
  c  Language alignment   language-aligned vs no language
  d  Adversarial training adversarially-trained vs standard   (the one real dissociation)

--resid subtracts each site's cross-model mean (site-residualized outcome); the output stem
then carries a `_resid` suffix.
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')
from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_SHORT_NAMES, MODEL_ORDER, ROBUST_MODELS, get_model_color)
apply_figure_style()

_M = _fig('model_dichotomies')
SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'
ADV = list(ROBUST_MODELS)                       # resnet50_robust, clipag_vitb32
UNTR = 'AlexNet_training_seed_01'
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 8, 7, 6.3, 5.6, 5.4
C_SIG, C_NS = '#C0392B', '#555555'

# ---- the four dichotomies (all among the 9 TRAINED models) ------------------
PANELS = [
    ('Architecture', ('CNN', 'Transformer'),
     ['resnet50', 'regnety_640', 'resnet50_robust', 'resnet50_dino', 'resnet50_clip'],
     ['siglip2_vitb16', 'dinov2_vitb14_reg', 'radio_v2.5-b', 'clipag_vitb32']),
    ('Task supervision', ('Supervised', 'Self-supervised'),
     ['resnet50', 'regnety_640', 'resnet50_robust'],
     ['resnet50_dino', 'dinov2_vitb14_reg', 'radio_v2.5-b', 'resnet50_clip', 'siglip2_vitb16', 'clipag_vitb32']),
    ('Language alignment', ('Language-\naligned', 'No\nlanguage'),
     ['resnet50_clip', 'siglip2_vitb16', 'clipag_vitb32', 'radio_v2.5-b'],
     ['resnet50', 'regnety_640', 'resnet50_robust', 'resnet50_dino', 'dinov2_vitb14_reg']),
    ('Adversarial training', ('Adv.-\ntrained', 'Standard'),
     ['resnet50_robust', 'clipag_vitb32'],
     ['resnet50', 'regnety_640', 'resnet50_clip', 'resnet50_dino', 'dinov2_vitb14_reg',
      'radio_v2.5-b', 'siglip2_vitb16']),
]


def paired(piv, A, B):
    """Within-site Wilcoxon on the per-site group-mean difference; returns (delta, p, n_sites)."""
    d = (piv[A].mean(axis=1) - piv[B].mean(axis=1)).values
    return float(d.mean()), float(stats.wilcoxon(d)[1]), len(d)


def fmt_delta(d):
    """Signed Δ, normally 2 decimals but escalating precision so a tiny nonzero value
    (e.g. panel c) is shown with real digits rather than rounding to 0.00."""
    for prec in (2, 3, 4):
        s = f'{d:+.{prec}f}'
        if float(s) != 0:
            return s
    return f'{d:+.4f}'


def build(outcome, resid=False):
    tab = L.control_table()
    piv = tab.pivot_table(index=['monkey', 'unit'], columns='model', values=outcome)
    if resid:                    # subtract per-site mean over the 9 trained models (site-residualized)
        piv = piv.subtract(piv[[c for c in piv.columns if c != UNTR]].mean(axis=1), axis=0)
    return piv


def panel(ax, piv, title, glabels, A, B, is_adv, ylim, show_xlabels):
    rng = np.random.default_rng(0)
    xg = [0, 1]
    for xi, grp in zip(xg, (A, B)):
        n = len(grp)
        subs = xi + (np.linspace(-0.28, 0.28, n) if n > 1 else np.array([0.0]))
        for x, m in zip(subs, grp):
            vals = piv[m].values                                  # 25 site-model values
            jit = rng.uniform(-0.033, 0.033, len(vals))
            ax.scatter(x + jit, vals, s=3.2, color=get_model_color(m), alpha=0.28, lw=0, zorder=2)
            ax.scatter(x, float(np.nanmean(vals)), s=24, color=get_model_color(m), zorder=5,
                       edgecolors='white', linewidths=0.4)
        # group mean +/- 95% CI over the 25 per-site group means (the within-site test's estimate)
        sm = piv[grp].mean(axis=1).values
        gm = sm.mean(); ci = 1.96 * stats.sem(sm)
        ax.errorbar(xi, gm, yerr=ci, fmt='_', color='black', ms=18, mew=1.9,
                    elinewidth=1.3, capsize=3, capthick=1.1, zorder=6)
    dlt, p, ns = paired(piv, A, B)
    stat = {'d': dlt, 'p': p}
    col = C_SIG if p < 0.05 else C_NS
    star = '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 0.05 else 'n.s.'))
    ax.set_xticks(xg)
    ax.set_xticklabels(list(glabels) if show_xlabels else [], fontsize=FS_TICK)
    ax.set_xlim(-0.5, 1.5)
    ax.tick_params(labelsize=FS_TICK)
    ax.axhline(0, color='#cccccc', lw=0.5, zorder=0)
    ax.set_ylim(*ylim)
    txt = f'$\\Delta$ = {fmt_delta(dlt)}\n$p$ = {p:.3f}  {star}'
    if not is_adv:
        Ac = [m for m in A if m not in ADV]; Bc = [m for m in B if m not in ADV]
        _, pc, _ = paired(piv, Ac, Bc); stat['p_excl'] = pc
        txt += f'\nexcl. adv: $p$ = {pc:.2f}'
    ax.text(0.5, 0.985, txt, transform=ax.transAxes, ha='center', va='top',
            fontsize=FS_ANNOT, fontfamily=FONT_FAMILY, color=col, linespacing=1.25)
    return stat


def main(out_dir, resid=False):
    stem = output_name('model_dichotomies') + ('_resid' if resid else '')
    sfx = ' (site-resid.)' if resid else ''
    ROWS = [('Control $r$' + sfx, build('control_r', resid)), ('Control slope' + sfx, build('control_slope', resid))]
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 128 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(2, 4, figure=fig, left=0.075, right=0.985, top=0.915, bottom=0.155,
                  wspace=0.30, hspace=0.22)
    stats_out = {}
    axes = [[], []]
    letters = 'abcdefgh'
    for ri, (ylab, piv) in enumerate(ROWS):
        TRAINED = [m for m in piv.columns if m != UNTR]
        va = piv[TRAINED].values.ravel(); vlo, vhi = float(np.nanmin(va)), float(np.nanmax(va))
        ylim = (vlo - 0.05 * (vhi - vlo), vhi + 0.34 * (vhi - vlo))          # proportional annotation headroom
        for j, (title, glabels, A, B) in enumerate(PANELS):
            ax = fig.add_subplot(gs[ri, j]); axes[ri].append(ax)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=FS_AX)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            stats_out[(ylab, title)] = panel(ax, piv, title, glabels, A, B, is_adv=(j == 3),
                                             ylim=ylim, show_xlabels=True)

    fig.canvas.draw()
    for ri in range(2):
        for j, ax in enumerate(axes[ri]):
            bb = ax.get_position()
            fig.text(bb.x0 - 0.028, bb.y1 + 0.052, letters[ri * 4 + j], ha='left', va='top',
                     fontsize=FS_LET, fontweight='bold', fontfamily=FONT_FAMILY)
            if ri == 0:                                                      # dichotomy titles: top row only
                fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.016, PANELS[j][0], ha='center', va='bottom',
                         fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
    # marker-type key (upper) + per-model color key (lower)
    h = [Line2D([0], [0], marker='o', ls='', mfc='#888', mec='white', mew=0.4, ms=3.2, label='site-model (25/model)'),
         Line2D([0], [0], marker='o', ls='', mfc='#888', mec='white', mew=0.4, ms=5, label='model mean'),
         Line2D([0], [0], marker='_', ls='', mec='black', mew=1.9, ms=9, label='group mean ± 95% CI')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=FS_ANNOT, frameon=False,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.53, 0.050))
    TRAINED_ORDER = [m for m in MODEL_ORDER if m != UNTR]
    mh = [Line2D([0], [0], marker='o', ls='', mfc=get_model_color(m), mec='white',
                 mew=0.3, ms=4.6, label=SHORT[m]) for m in TRAINED_ORDER]
    fig.legend(handles=mh, loc='lower center', ncol=9, fontsize=FS_ANNOT, frameon=False,
               handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.53, 0.028))

    path = save_fig(fig, os.path.join(out_dir, stem))
    plt.close(fig)
    print('Saved ->', path)
    print('\nWithin-site paired (Wilcoxon), 25 sites, 9 trained models:')
    for (ylab, title), v in stats_out.items():
        ex = f"   excl-adv p={v['p_excl']:.3f}" if 'p_excl' in v else ''
        print(f'  {ylab.replace("$",""):12s} {title:22s} delta={v["d"]:+.3f}  p={v["p"]:.4f}{ex}')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    ap.add_argument('--resid', action='store_true',
                    help='site-residualized outcome (subtract each site\'s cross-model mean); stem gets _resid')
    args = ap.parse_args()
    main(**vars(args))
