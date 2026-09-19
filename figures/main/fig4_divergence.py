#!/usr/bin/env python3
"""Figure 4 - encoding vs. parametric control across models and brain areas.

a  encoding test r per model (bars = mean over the 25 sites, dots = sites).
b  control r per model (same treatment).
c  family slopegraph: encoding r -> control r per family.
d  per monkey/brain area (5 sites each): control bars with the per-model encoding
   mean +/- s.d. overlaid as a faint dash for context, plus the summary slopegraph.

Panels a/b/c are drawn by the shared renderers in fig4_panels.py, so panel d inherits
the identical bar / dot / family-color styling.

Reads only the preproc cache via loader.control_table().
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from pnc import paths
from pnc.manifest import MAIN_FIGURES
from pnc.utils import (apply_figure_style, MM, WIDTH_2COL_MM, save_fig,        # noqa: E402
                       MONKEY_COLORS, MODEL_ORDER, MODEL_COLORS, FONT_FAMILY)
from pnc.preproc import loader as L                                            # noqa: E402
from figures.main.fig4_panels import (bar_panel, slope_panel, two_line_title,  # noqa: E402
                                      GRID_KW, FS_TTL_BIG, FS_AX, FS_TK, FS_LET,
                                      ENC_X, CTRL_X, SHORT)
apply_figure_style()

FS_D = 5.0                                    # panel d tick labels
MONKEY_ORDER = ['red', 'paul', 'venus', 'three0', 'leap']
YLABEL = 'Pearson r (predicted vs. observed)'
D_ROWS = [('control_r', 'Pearson r\n(pred. vs. observed)'),
          (None, 'Summary')]


def _tint(color, f=0.55):
    """Blend a color toward white by fraction f."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * f)


def region_of(tab, monkey):
    """Brain area label for a monkey, straight from the control_table region column."""
    regs = sorted(tab.loc[tab.monkey == monkey, 'region'].unique())
    return ' / '.join(regs)


def main(out_dir):
    tab = L.control_table()[['monkey', 'unit', 'model', 'region',
                             'control_r', 'encoding_r']].copy()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 132 * MM))
    gs_top = fig.add_gridspec(1, 3, left=0.065, right=0.90, top=0.940, bottom=0.630,
                              width_ratios=[1.25, 1.25, 0.75], wspace=0.28)
    axs = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    lo = min(tab.control_r.min(), tab.encoding_r.min()) - 0.03
    hi = max(tab.control_r.max(), tab.encoding_r.max()) + 0.03
    YLIM = (lo, hi)

    # ── a / b / c ────────────────────────────────────────────────────────────
    ttl_a = bar_panel(axs[0], tab, 'encoding_r', 'Encoding score', 'Natural test images',
                      ylabel=YLABEL)
    ttl_b = bar_panel(axs[1], tab, 'control_r', 'Parametric control', 'Accentuated test images',
                      ylabel=YLABEL)
    axs[0].set_ylim(*YLIM); axs[1].set_ylim(*YLIM)

    ax = axs[2]
    slope_panel(ax, tab)
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(*YLIM)
    ax.set_xticks([ENC_X, CTRL_X])
    ax.set_xticklabels(['Encoding', 'Control'], fontsize=FS_AX, fontweight='bold')
    ax.set_ylabel(YLABEL, fontsize=FS_TK)
    ttl_c = two_line_title(ax, 'Summary')
    ax.tick_params(labelsize=FS_TK, length=2, pad=1.5)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', **GRID_KW)
    ax.axhline(0, color='0.7', lw=0.5)

    # ── measure a/b/c text extents to lay out d against them ─────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fy1 = lambda t: inv.transform((0, t.get_window_extent(renderer=renderer).y1))[1]
    fy0 = lambda t: inv.transform((0, t.get_window_extent(renderer=renderer).y0))[1]
    # x-center of a's ylabel (d row labels align to it) and right edge of c's text
    bb_ylab = axs[0].yaxis.label.get_window_extent(renderer=renderer)
    x_ylab_a = inv.transform(((bb_ylab.x0 + bb_ylab.x1) / 2, 0))[0]
    x_c_right = max(inv.transform((t.get_window_extent(renderer=renderer).x1, 0))[0]
                    for t in axs[2].texts)

    # ── d: one column per monkey, rows = control (+ encoding dashes) / summary
    gs_bot = fig.add_gridspec(2, 5, left=0.065, right=x_c_right, top=0.4525,
                              bottom=0.0885, hspace=0.40, wspace=0.24,
                              height_ratios=[22, 18])
    d_axes, hdr_texts = [], []
    for ci, mk in enumerate(MONKEY_ORDER):
        sub = tab[tab.monkey == mk]
        col_axes = []
        for ri, (col, _) in enumerate(D_ROWS):
            axd = fig.add_subplot(gs_bot[ri, ci])
            if col is not None:
                bar_panel(axd, sub, col, xlabels=False, fs=FS_D, bar_lw=0.6, dots=False)
                axd.spines['bottom'].set_visible(False)
                axd.tick_params(axis='x', bottom=False)
                axd.set_xlim(-0.8, len(MODEL_ORDER) - 0.2)
                # model names hang straight down from the bottom of the bars
                for xi, m in enumerate(MODEL_ORDER):
                    axd.text(xi, -0.035, SHORT[m], rotation=90, ha='center',
                             va='top', color=MODEL_COLORS[m], fontsize=4.3,
                             clip_on=False, zorder=6, fontfamily=FONT_FAMILY)
                # encoding mean +/- s.d. per model: faint tinted dash for context
                for xi, m in enumerate(MODEL_ORDER):
                    e = sub.loc[sub.model == m, 'encoding_r'].dropna().values
                    tc = _tint(MODEL_COLORS[m])
                    axd.plot([xi - 0.36, xi + 0.36], [e.mean()] * 2, color='white',
                             lw=2.0, alpha=0.9, zorder=3.5, solid_capstyle='butt')
                    axd.plot([xi - 0.36, xi + 0.36], [e.mean()] * 2, color=tc,
                             lw=1.0, zorder=3.6, solid_capstyle='butt')
                    axd.plot([xi, xi], [e.mean() - e.std(ddof=1), e.mean() + e.std(ddof=1)],
                             color=tc, lw=0.6, alpha=0.8, zorder=3.4)
                if ci == 0:
                    # key the two marks: stacked words, arrows leaving the right
                    # edge at 30 deg visual (x is ~5.3x denser than y here)
                    akw = dict(fontsize=4.3, color='0.45', va='center', zorder=8,
                               ha='right', fontfamily=FONT_FAMILY,
                               arrowprops=dict(arrowstyle='-|>', color='0.45',
                                               lw=0.55, mutation_scale=4.5,
                                               relpos=(1, 0.5), patchA=None,
                                               shrinkA=1.5, shrinkB=0.5))
                    axd.annotate('Encoding', xytext=(3.6, 0.545),
                                 xy=(3.6 + 9.2 * 0.11, 0.545 + 0.11), **akw)
                    axd.annotate('Control', xytext=(3.6, 0.45),
                                 xy=(3.6 + 9.2 * 0.105, 0.45 - 0.105), **akw)
            else:
                slope_panel(axd, sub, labels=False, site_s=(3.5, 4.5), site_lw=0.35,
                            mean_s=13, mean_lw=1.0, cap=1.2)
                axd.set_xlim(-0.38, 1.38)
                axd.set_xticks([ENC_X, CTRL_X])
                axd.set_xticklabels(['Encoding', 'Control'], fontsize=FS_D,
                                    fontweight='bold')
                axd.spines[['top', 'right', 'bottom']].set_visible(False)
                axd.tick_params(axis='x', bottom=False, pad=0.5)
                axd.set_axisbelow(True)
                axd.grid(True, axis='y', **GRID_KW)
                axd.axhline(0, color='0.7', lw=0.5)
            axd.set_ylim(*YLIM)
            axd.set_yticks([0.0, 0.4, 0.8])
            axd.tick_params(axis='y', labelsize=FS_D, length=2, pad=1.5)
            if ci:
                axd.set_yticklabels([])
            col_axes.append(axd)
        d_axes.append(col_axes)

        # brain area + subject above the encoding row of each column
        top_ax, c = col_axes[0], MONKEY_COLORS[mk]
        hdr_texts.append(
            top_ax.text(0.5, 1.14, region_of(tab, mk), transform=top_ax.transAxes,
                        fontsize=FS_AX + 1, fontweight='bold', color=c, ha='center',
                        va='bottom', fontfamily=FONT_FAMILY, clip_on=False))
        top_ax.text(0.5, 1.02, f'Monkey {mk[0].upper()}', transform=top_ax.transAxes,
                    fontsize=FS_D, color=c, ha='center', va='bottom',
                    fontfamily=FONT_FAMILY, clip_on=False)

    fig.canvas.draw()

    # row labels down the left edge of d, x-aligned with a's ylabel
    for (_, row_lab), axd in zip(D_ROWS, d_axes[0]):
        pos = axd.get_position()
        fig.text(x_ylab_a, pos.y0 + pos.height / 2, row_lab, fontsize=FS_TK,
                 fontfamily=FONT_FAMILY, ha='center', va='center', rotation=90)

    # d title clears the tallest column header; report the gap to the a/b tick labels
    ttl_d = fig.text((0.065 + x_c_right) / 2, max(map(fy1, hdr_texts)) + 0.015,
                     'Encoding vs. parametric control by brain area', fontsize=FS_TTL_BIG,
                     fontfamily=FONT_FAMILY, ha='center', va='bottom')
    fig.canvas.draw()
    ab_bot = min(fy0(t) for ax_ in axs[:2] for t in ax_.get_xticklabels())
    print(f'  gap a/b labels -> d title: {(ab_bot - fy1(ttl_d)) * 132:.1f} mm')

    y_top = max(fy1(t) for t in (ttl_a, ttl_b, ttl_c))
    y_top_d = fy1(ttl_d)
    for ax_, lab, y in [(axs[0], 'a', y_top), (axs[1], 'b', y_top),
                        (axs[2], 'c', y_top), (d_axes[0][0], 'd', y_top_d)]:
        bb = ax_.get_position()
        x = axs[0].get_position().x0 - 0.042 if lab == 'd' else bb.x0 - 0.042
        fig.text(x, y, lab, fontsize=FS_LET, fontweight='bold',
                 fontfamily=FONT_FAMILY, ha='left', va='top')

    out = save_fig(fig, os.path.join(out_dir, MAIN_FIGURES[4]))
    print(f'saved -> {out}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    args = ap.parse_args()
    main(**vars(args))
