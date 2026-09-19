#!/usr/bin/env python3
"""Shared panel renderers for figure 4: per-model bar panels, the family
encoding->control slopegraph, and the two-line title helper.

Reads nothing from disk; callers pass in the control_table slice.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np

from pnc.utils import MODEL_ORDER, MODEL_COLORS, MODEL_SHORT_NAMES, FONT_FAMILY  # noqa: E402

FS_TTL_BIG, FS_TTL_SUB, FS_AX, FS_TK, FS_LET = 8.5, 6.5, 7.0, 6.0, 10
SUB_PAD_PT = 2.0                               # axes top -> small second title row
TITLE_GAP_PT = SUB_PAD_PT + FS_TTL_SUB * 1.15  # axes top -> big first title row
GRID_KW = dict(alpha=0.35, lw=0.5, color='#888888')

UNTR = 'AlexNet_training_seed_01'
ADV = ['resnet50_robust', 'clipag_vitb32']
CONV = [m for m in MODEL_ORDER if m != UNTR and m not in ADV]
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
FAM = [('Adversarially\ntrained', ADV, '#70C35E'),        # CIELAB midpoint of the two ADV bars
       ('Conventionally\ntrained', CONV, '#C2185B'),      # conventional-7 color
       ('Untrained', [UNTR], MODEL_COLORS[UNTR])]         # same color as its bar in a/b
ENC_X, CTRL_X = 0.0, 1.0                                  # slopegraph x positions


def two_line_title(ax, line1, line2=None):
    """Title with a bigger top row and a smaller second row. Returns the top text artist.
    line1 always sits at the same height so the panel letters stay aligned."""
    if line2:
        ax.annotate(line2, xy=(0.5, 1.0), xycoords='axes fraction',
                    xytext=(0, SUB_PAD_PT), textcoords='offset points',
                    ha='center', va='bottom', fontsize=FS_TTL_SUB,
                    fontfamily=FONT_FAMILY, annotation_clip=False)
    return ax.annotate(line1, xy=(0.5, 1.0), xycoords='axes fraction',
                       xytext=(0, TITLE_GAP_PT), textcoords='offset points',
                       ha='center', va='bottom', fontsize=FS_TTL_BIG,
                       fontfamily=FONT_FAMILY, annotation_clip=False)


def bar_panel(ax, tab, col, title1=None, title2=None, ylabel=None,
              xlabels=True, fs=FS_TK, dot_s=3.5, bar_lw=0.9, dots=True):
    rng = np.random.default_rng(0)
    for xi, m in enumerate(MODEL_ORDER):
        d = tab.loc[tab.model == m, col].dropna().values
        mean_d = d.mean()
        ax.bar(xi, mean_d, width=0.72, color=MODEL_COLORS[m], zorder=2)
        ax.errorbar(xi, mean_d, yerr=d.std(ddof=1) / np.sqrt(len(d)), color='0.15',
                    lw=bar_lw, capsize=1.8 * bar_lw / 0.9, capthick=bar_lw, zorder=4)
        xs = xi + rng.uniform(-0.22, 0.22, len(d))
        if not dots:
            continue
        # dots that fall inside the colored bar go white so they stay visible
        inside = (d >= min(0, mean_d)) & (d <= max(0, mean_d))
        ax.scatter(xs[inside], d[inside], s=dot_s, color='white', alpha=0.7,
                   lw=0, zorder=3)
        ax.scatter(xs[~inside], d[~inside], s=dot_s, color='0.15', alpha=0.35,
                   lw=0, zorder=3)
    ax.set_xticks(range(len(MODEL_ORDER)))
    if xlabels:
        ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], fontsize=fs,
                           rotation=40, ha='right', rotation_mode='anchor')
        for t, m in zip(ax.get_xticklabels(), MODEL_ORDER):
            t.set_color(MODEL_COLORS[m])
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs)
    ttl = two_line_title(ax, title1, title2) if title1 else None
    ax.tick_params(labelsize=fs, length=2, pad=1.5)
    if not xlabels:
        ax.tick_params(axis='x', length=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', **GRID_KW)
    ax.axhline(0, color='0.7', lw=0.5)
    return ttl


def slope_panel(ax, tab, labels=True, fs=FS_TK, site_s=(9, 11), site_lw=0.5,
                mean_s=42, mean_lw=1.8, cap=2, lab_dx=0.24):
    """Per-site encoding -> control slopegraph with family means. Draws the data only;
    the caller owns the axis decoration (limits, ticks, title, grid)."""
    rng = np.random.default_rng(42)
    fam_of = {m: (name, color) for name, models, color in FAM for m in models}
    # draw conventionally-trained (magenta) first, untrained/adv-trained last so they
    # stay visible on top of the dense magenta swarm
    sw = tab.dropna(subset=['encoding_r', 'control_r']).copy()
    sw['_prio'] = np.where(sw.model.isin(CONV), 0, 1)
    sw = sw.sort_values('_prio', kind='stable')
    for _, row in sw.iterrows():
        name, c = fam_of[row.model]
        ex = ENC_X + rng.normal(0, 0.06)
        cx = CTRL_X + rng.normal(0, 0.06)
        ax.plot([ex, cx], [row.encoding_r, row.control_r], color=c, lw=site_lw,
                alpha=0.06, zorder=1)
        ax.scatter(ex, row.encoding_r, s=site_s[0], color=c, alpha=0.4,
                   edgecolors='none', zorder=3)
        ax.scatter(cx, row.control_r, s=site_s[1], color=c, alpha=0.4,
                   edgecolors='none', zorder=3)
    for name, models, color in FAM:
        sub = tab[tab.model.isin(models)].dropna(subset=['encoding_r', 'control_r'])
        em, cm_ = sub.encoding_r.mean(), sub.control_r.mean()
        es = sub.encoding_r.std() / np.sqrt(len(sub))
        cs_ = sub.control_r.std() / np.sqrt(len(sub))
        ax.plot([ENC_X, CTRL_X], [em, cm_], color=color, lw=mean_lw, zorder=5,
                solid_capstyle='round')
        for x, m_, s_ in ((ENC_X, em, es), (CTRL_X, cm_, cs_)):
            ax.scatter(x, m_, s=mean_s, marker='D', color=color, edgecolors='white',
                       linewidth=0.7, zorder=6)
            ax.errorbar(x, m_, yerr=s_, color='0.15', fmt='none', capsize=cap,
                        capthick=0.8, elinewidth=0.8, zorder=7)
        if labels:
            # n rides on the "trained" line for the two-line names, own line otherwise
            lab = f'{name} (n={len(sub)})' if '\n' in name else f'{name}\n(n={len(sub)})'
            ax.annotate(lab, xy=(CTRL_X, cm_), xytext=(CTRL_X + lab_dx, cm_),
                        fontsize=fs, fontweight='bold', color=color, va='center',
                        clip_on=False)
