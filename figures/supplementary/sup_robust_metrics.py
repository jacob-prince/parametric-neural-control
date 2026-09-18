#!/usr/bin/env python3
"""Supp fig (robust_metrics) - the adversarially-trained advantage is not an artefact
of the scoring metric.

The main-text control advantage was scored with Pearson r on seed-averaged sweeps.
Here the same comparison is re-run under two alternative scores and one alternative
aggregation:
  (a) linear-fit SLOPE  (measured vs predicted, accentuated control),
  (b) R2                (coefficient of determination, accentuated control),
  (c) PER-SEED Pearson r (each seed sweep scored separately, not seed-averaged).
Models are grouped by the paper's three training families - adversarially trained,
conventionally trained, untrained (figure-4 family palette); Untrained is its own
reference class, excluded from the adv-vs-conventional test.

Reads loader.control_table() (per site-model control_slope, control_R2) and
control_seed_rs() (per-seed control r).
"""
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_SHORT_NAMES, ROBUST_MODELS, MODEL_COLORS, MONKEY_COLORS)

apply_figure_style()
_M = _fig('robust_metrics')
STEM = output_name('robust_metrics')

MODEL_SHORT_NAMES = dict(MODEL_SHORT_NAMES)
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'
UNTRAINED = 'AlexNet_training_seed_01'
ROBUST = set(ROBUST_MODELS)

# Group -> color: the figure-4 family palette - adversarially trained green,
# conventionally trained magenta, Untrained deep indigo (its own reference class,
# excluded from the adv-vs-conventional test).
C_ADV = '#70C35E'
C_CONV = '#C2185B'
C_UNTR = MODEL_COLORS[UNTRAINED]
GROUPS = ['Adv', 'Conv', 'Untrained']
GROUP_COLOR = {'Adv': C_ADV, 'Conv': C_CONV, 'Untrained': C_UNTR}
# Tick labels (two-line).
TICK_LABEL = {'Adv': 'Adversarially\ntrained', 'Conv': 'Conventionally\ntrained',
              'Untrained': 'Untrained'}
# individual dots are coloured by MONKEY (faint) to expose per-animal structure in the tails
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
MK_LABEL = {'red': 'R (aIT)', 'paul': 'P (cIT)', 'venus': 'V (V3/V4)',
            'leap': 'L (STS)', 'three0': 'T (STS)'}

FS_TITLE, FS_AX, FS_TICK, FS_ANNOT, FS_LETTER = 8, 7, 6, 6, 11


def _group(model):
    if model in ROBUST:
        return 'Adv'
    if model == UNTRAINED:
        return 'Untrained'
    return 'Conv'


def _spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _sig(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def load():
    """Per site-model rows with the three scores; per-seed r pooled per group."""
    ct = L.control_table()
    ct = ct.copy()
    ct['group'] = ct['model'].map(_group)

    # per-seed control r: list per group, plus per site-model mean
    cfg = L.config()
    seed_pool = {g: [] for g in GROUPS}       # every individual seed r
    seed_pool_mk = {g: [] for g in GROUPS}    # matching monkey per seed r
    site_seedmean = []                         # (group, mean-per-seed-r)
    for mk in cfg['monkeys']:
        b = L.load_brain(mk)
        for u in b['units']:
            for m in cfg['models']:
                rs = L.control_seed_rs(mk, int(u), m)
                if not rs:
                    continue
                g = _group(m)
                seed_pool[g].extend(rs)
                seed_pool_mk[g].extend([mk] * len(rs))
                site_seedmean.append((g, float(np.mean(rs))))
    return ct, seed_pool, seed_pool_mk, site_seedmean


def _boot_median_ci(v, n_boot=5000, seed=0):
    """Percentile bootstrap 95% CI of the median (robust to the heavy negative tail)."""
    if len(v) < 2:
        return float(np.median(v)), float(np.median(v))
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def _strip(ax, groups_vals, jitter_seed, ylabel, title, central='mean'):
    """Grouped strip of per-observation values with a group central estimate + 95% CI.
    central='mean' -> mean +/- t-CI; central='median' -> median + bootstrap CI (heavy tails).
    groups_vals: dict group -> array of values. The robust-vs-non-robust test pools the
    two non-robust groups (CNN + ViT; Untrained excluded).
    Returns (p, robust_center, nonrobust_pooled_center)."""
    rng = np.random.default_rng(jitter_seed)
    xpos = {g: i for i, g in enumerate(GROUPS)}
    centers = {}

    def _center(v):
        if central == 'median':
            return float(np.median(v))
        return float(v.mean())

    for g in GROUPS:
        v, mks = groups_vals[g]
        v = np.asarray(v, float); mks = np.asarray(mks)
        ok = np.isfinite(v); v = v[ok]; mks = mks[ok]
        if len(v) == 0:
            continue
        c = GROUP_COLOR[g]
        x = xpos[g] + rng.uniform(-0.22, 0.22, len(v))
        dot_cols = [MONKEY_COLORS.get(mk, '#999999') for mk in mks]   # faint per-MONKEY colours
        ax.scatter(x, v, s=9, color=dot_cols, alpha=0.45, edgecolors='none', zorder=3)
        if central == 'median':
            m = float(np.median(v))
            lo, hi = _boot_median_ci(v, seed=jitter_seed)
            yerr = [[m - lo], [hi - m]]
        else:
            m = v.mean()
            yerr = (v.std(ddof=1) / np.sqrt(len(v)) * stats.t.ppf(0.975, len(v) - 1)
                    if len(v) > 1 else 0)
        # fig4 treatment: colored mean marker, dark neutral whisker drawn ON TOP of it
        ax.scatter(xpos[g], m, s=52, color=c, edgecolors='black', linewidth=0.8, zorder=6)
        ax.errorbar(xpos[g], m, yerr=yerr, fmt='none', ecolor='0.15', elinewidth=0.8,
                    capsize=2, capthick=0.8, zorder=7)
        centers[g] = m
    # significance: adversarially vs conventionally trained (Untrained excluded)
    rv = np.asarray(groups_vals['Adv'][0], float); rv = rv[np.isfinite(rv)]
    conv = np.asarray(groups_vals['Conv'][0], float); conv = conv[np.isfinite(conv)]
    p_conv = stats.mannwhitneyu(rv, conv, alternative='two-sided')[1]
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([TICK_LABEL[g] for g in GROUPS], fontsize=FS_TICK - 1)
    for tl, g in zip(ax.get_xticklabels(), GROUPS):
        tl.set_color(GROUP_COLOR[g]); tl.set_fontweight('bold')
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylabel(ylabel, fontsize=FS_AX)
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', fontfamily=FONT_FAMILY, pad=6)
    ax.tick_params(labelsize=FS_TICK)
    _spines(ax)
    return dict(p_conv=p_conv, c_adv=centers.get('Adv', np.nan), c_conv=_center(conv))


def _one_bracket(ax, tr, x0, x1, yb, p, delta):
    yt = yb + 0.020
    ax.plot([x0, x0, x1, x1], [yb, yt, yt, yb], transform=tr, color='black',
            lw=0.8, clip_on=False, zorder=8)
    ax.text((x0 + x1) / 2, yt + 0.006, f'{_sig(p)}  ($\\Delta$={delta:+.2f})', transform=tr,
            ha='center', va='bottom', fontsize=FS_ANNOT - 0.5, fontfamily=FONT_FAMILY, zorder=8)


def _brackets(ax, res):
    """One bracket: adversarially vs conventionally trained."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    _one_bracket(ax, tr, 0, 1, 0.90, res['p_conv'], res['c_adv'] - res['c_conv'])


def main(out_dir):
    ct, seed_pool, seed_pool_mk, site_seedmean = load()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 66 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 3, figure=fig, left=0.075, right=0.985, top=0.84, bottom=0.15,
                  wspace=0.30)

    key = {}

    # (a) slope
    ax = fig.add_subplot(gs[0, 0])
    gv = {g: (ct.loc[ct.group == g, 'control_slope'].values,
              ct.loc[ct.group == g, 'monkey'].values) for g in GROUPS}
    res = _strip(ax, gv, 1, 'Control slope (measured vs predicted)', 'Linear-fit slope')
    ax.set_ylim(-0.35, 1.5)
    _brackets(ax, res)
    key['slope'] = res
    ax_a = ax

    # (b) R2 - unclipped (heavy negative tail); symlog spacing + group MEDIANS
    ax = fig.add_subplot(gs[0, 1])
    gv = {g: (ct.loc[ct.group == g, 'control_R2'].values,
              ct.loc[ct.group == g, 'monkey'].values) for g in GROUPS}
    res = _strip(ax, gv, 2, r'Control $R^2$', r'$R^2$', central='median')
    ax.set_yscale('symlog', linthresh=1.0, linscale=0.6)
    ax.set_yticks([-100, -30, -10, -3, -1, 0, 1])
    ax.set_ylim(-140, 15)
    ax.axhline(0, color='#BBBBBB', lw=0.5, ls='--', zorder=1)
    _brackets(ax, res)
    key['R2'] = res
    ax_b = ax

    # (c) per-seed r
    ax = fig.add_subplot(gs[0, 2])
    gv = {g: (np.asarray(seed_pool[g], float), np.asarray(seed_pool_mk[g])) for g in GROUPS}
    res = _strip(ax, gv, 3, 'Per-seed control r', 'Per-seed Pearson r')
    ax.set_ylim(-0.55, 1.5)
    _brackets(ax, res)
    key['per_seed_r'] = res
    ax_c = ax

    # panel letters
    fig.canvas.draw()
    for ax, lab in zip([ax_a, ax_b, ax_c], 'abc'):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.052, bb.y1 + 0.03, lab, ha='left', va='bottom',
                 fontsize=FS_LETTER, fontweight='bold', fontfamily=FONT_FAMILY)

    # --- two-part header legend, lifted clear of the panel titles ---
    # (left, above panel a) group identity of the mean markers
    LEG_LABEL = {'Adv': 'Adversarially trained', 'Conv': 'Conventionally trained',
                 'Untrained': 'Untrained'}
    grp_handles = [Line2D([0], [0], marker='o', ls='', color=GROUP_COLOR[g], ms=6,
                          mec='black', mew=0.6, label=LEG_LABEL[g]) for g in GROUPS]
    fig.legend(handles=grp_handles, loc='upper left', ncol=3, fontsize=FS_TICK,
               frameon=False, bbox_to_anchor=(0.075, 1.0), handletextpad=0.3,
               columnspacing=1.1)

    # (right, above panel c) per-monkey colour of the individual dots
    mk_handles = [Line2D([0], [0], marker='o', ls='', color=MONKEY_COLORS[mk], ms=6,
                         mec='none', label=MK_LABEL[mk]) for mk in MONKEYS]
    fig.legend(handles=mk_handles, loc='upper right', ncol=5, fontsize=FS_TICK,
               frameon=False, bbox_to_anchor=(0.985, 1.0), handletextpad=0.3,
               columnspacing=1.0)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    print('KEY', key)
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
