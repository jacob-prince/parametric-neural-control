#!/usr/bin/env python3
"""Supp fig (controversial_results) — controversial-accentuation results across the 7
targeted aIT sites.

For each targeted aIT site the two encoding models' predictions are pitted against the
(anchorDay-standardized) measured neural response to the controversial accentuations. The
Robust ResNet50 predictions track the neural response (positive correlation); standard
ResNet50 predictions anti-track it (negative correlation). Panels a-g show the per-site
model-prediction vs neural-response clouds (both models, Lasso encoders, raw prediction
units) with least-squares fits and per-site correlations. Panel h summarizes the per-site
correlations for the two models with paired lines, group means, and the Wilcoxon signed-rank
test; grey dashed lines mark the achievable +/- noise-ceiling envelope pooled over sites.

Reads loader.load_controversial_table(). Robust-favoring vs standard-favoring stimuli are
distinguished by marker fill in the scatter panels; correlations pool both families per
site (matching the companion controversial figure).
"""
import os
import argparse

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig)
apply_figure_style()
_M = _fig('controversial_results'); STEM = output_name('controversial_results')

# exact colors + fades from the companion controversial figure so the legend markers match
C_R50, C_ROBUST = '#FF9D1E', '#8DC63F'
C_R50_FADE, C_ROBUST_FADE = '#FFD9A6', '#DCEFB4'

FS_TITLE, FS_AX, FS_TICK, FS_RVAL, FS_LEG = 7, 6, 5.5, 5.5, 5.5
FS_NC = 5.0   # per-site noise-ceiling subtitle


def _r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return stats.pearsonr(x, y)[0] if x.std() > 0 and y.std() > 0 and len(x) >= 5 else np.nan


# ── load ──────────────────────────────────────────────────────────────────────
# Table columns the panel helpers read; filled by _load() at call time (was import-time work).
UNITS = stim = unit = neural = sem = pred_r50 = pred_rob = is_r50fav = ncr = ncr_pool = None


def _load():
    global UNITS, stim, unit, neural, sem, pred_r50, pred_rob, is_r50fav, ncr, ncr_pool
    d = L.load_controversial_table()
    t = d['table']
    UNITS = [int(u) for u in d['units']]
    stim = np.asarray(t['stim']).astype(str)
    unit = np.asarray(t['unit']).astype(int)
    neural = np.asarray(t['neural'], float)
    sem = np.asarray(t['sem'], float)
    pred_r50 = np.asarray(t['RN50_Lasso'], float)       # standard-model prediction (Lasso)
    pred_rob = np.asarray(t['RN50rbst_Lasso'], float)   # robust-model prediction (Lasso)
    # robust-favoring vs standard-favoring stimulus family
    is_r50fav = np.char.find(stim, 'max_r50') >= 0
    ncr = d['ceilings']['nc_r_controversial']           # per-unit achievable ceiling (Pearson r)
    ncr_pool = d['ceilings']['nc_r_pooled']


# ── per-site scatter ────────────────────────────────────────────────────────
def scatter(ax, u, show_xlabel, show_ylabel):
    m = unit == u
    y = neural[m]; ye = sem[m]; r50fav = is_r50fav[m]   # True = ResNet50-favoring stimulus family
    # marker scheme copied from the companion controversial figure: shape encodes prediction
    # model (o=R50, s=Robust), full colour when the prediction model matches the stimulus
    # family, faded when it does not.
    for pred, mk, base, fade, match in [(pred_r50[m], 'o', C_R50, C_R50_FADE, r50fav),
                                        (pred_rob[m], 's', C_ROBUST, C_ROBUST_FADE, ~r50fav)]:
        ax.errorbar(pred, y, yerr=ye, fmt='none', ecolor='#BBBBBB', elinewidth=0.5,
                    alpha=0.7, capsize=0, zorder=2)
        ax.scatter(pred[match], y[match], s=14, facecolors=base, marker=mk, edgecolors='black',
                   linewidths=0.4, alpha=0.95, zorder=4)
        ax.scatter(pred[~match], y[~match], s=14, facecolors=fade, marker=mk, edgecolors='black',
                   linewidths=0.4, alpha=0.95, zorder=4)
        sl, b, _, _, _ = stats.linregress(pred, y)
        xs = np.linspace(pred.min(), pred.max(), 50)
        ax.plot(xs, sl * xs + b, color=base, lw=1.2, zorder=5)
    r5 = _r(pred_r50[m], y); rb = _r(pred_rob[m], y)
    ax.axhline(0, color='#E5E5E5', lw=0.6, zorder=0)
    ax.axvline(0, color='#E5E5E5', lw=0.6, zorder=0)
    tk = dict(transform=ax.transAxes, ha='right', va='top', fontsize=FS_RVAL,
              fontweight='bold', fontfamily=FONT_FAMILY, linespacing=1.25)
    ax.text(0.97, 0.97, f'r = {r5:+.2f}\nr = {rb:+.2f}', color='none', zorder=7,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#CCC',
                      lw=0.5, alpha=0.92), **tk)
    ax.text(0.97, 0.97, f'r = {r5:+.2f}\n ', color=C_R50, zorder=8, **tk)
    ax.text(0.97, 0.97, f' \nr = {rb:+.2f}', color=C_ROBUST, zorder=8, **tk)
    ax.set_title(f'Site {u} (aIT)', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=11)
    # per-site achievable noise ceiling (Pearson r over the 20 controversial stims), small
    # subtitle centred just below the panel title
    ax.text(0.5, 1.015, f'noise ceiling $r$ = {ncr[u]:.2f}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=FS_NC, color='#555555', fontfamily=FONT_FAMILY)
    ax.tick_params(labelsize=FS_TICK, length=2, pad=1)
    if show_xlabel:
        ax.set_xlabel('Predicted response (z)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    if show_ylabel:
        ax.set_ylabel('Measured neural (z)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ── summary ─────────────────────────────────────────────────────────────────
def summary(ax):
    r5 = np.array([_r(pred_r50[unit == u], neural[unit == u]) for u in UNITS])
    rb = np.array([_r(pred_rob[unit == u], neural[unit == u]) for u in UNITS])
    # achievable +/- noise-ceiling envelope (pooled over the 7 aIT sites), dashed lines only
    ax.axhline(ncr_pool, color='#AAAAAA', lw=0.6, ls=':', zorder=1)
    ax.axhline(-ncr_pool, color='#AAAAAA', lw=0.6, ls=':', zorder=1)
    for a, b in zip(r5, rb):
        ax.plot([0, 1], [a, b], color='#999999', lw=0.7, alpha=0.7, zorder=2)
    ax.scatter([0] * len(r5), r5, s=28, c=C_R50, edgecolors='black', linewidths=0.4, zorder=4)
    ax.scatter([1] * len(rb), rb, s=28, c=C_ROBUST, edgecolors='black', linewidths=0.4, zorder=4)
    ax.scatter([0], [r5.mean()], marker='D', s=70, c=C_R50, edgecolors='black', linewidths=0.8, zorder=6)
    ax.scatter([1], [rb.mean()], marker='D', s=70, c=C_ROBUST, edgecolors='black', linewidths=0.8, zorder=6)
    ax.plot([0, 1], [r5.mean(), rb.mean()], color='black', lw=1.4, zorder=5)
    # mean r annotations beside each group's diamond
    ax.annotate(f'{r5.mean():+.2f}', (0, r5.mean()), xytext=(-6, 0),
                textcoords='offset points', ha='right', va='center',
                fontsize=FS_RVAL, fontweight='bold', color=C_R50, fontfamily=FONT_FAMILY)
    ax.annotate(f'{rb.mean():+.2f}', (1, rb.mean()), xytext=(6, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=FS_RVAL, fontweight='bold', color=C_ROBUST, fontfamily=FONT_FAMILY)
    ax.axhline(0, color='#555555', lw=0.8, ls='--', zorder=1)
    # Wilcoxon signed-rank across the 7 sites
    p = stats.wilcoxon(rb, r5).pvalue
    ymax = max(r5.max(), rb.max())
    sig_y = max(ymax, ncr_pool) + 0.12
    ax.plot([0, 0, 1, 1], [sig_y - 0.04, sig_y, sig_y, sig_y - 0.04], color='black', lw=0.8)
    ax.text(0.5, sig_y + 0.015, f'p = {p:.3f}', ha='center', va='bottom',
            fontsize=FS_RVAL, fontweight='bold', fontfamily=FONT_FAMILY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['ResNet50', 'RN50-Robust'],
                                              fontsize=FS_AX, fontfamily=FONT_FAMILY)
    for tk, c in zip(ax.get_xticklabels(), [C_R50, C_ROBUST]):
        tk.set_color(c); tk.set_fontweight('bold')
    ax.set_ylabel('Correlation with\nneural response (r)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ymag = max(abs(r5.min()), abs(rb.min()), ncr_pool, sig_y) + 0.12
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-ymag, ymag)
    ax.set_title('Per-site summary', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=3)
    ax.tick_params(labelsize=FS_TICK, length=2, pad=1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return r5, rb, p


# ── build ─────────────────────────────────────────────────────────────────────
def main(out_dir):
    _load()
    FIGHEIGHT_MM = 120.0
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, FIGHEIGHT_MM * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(2, 4, figure=fig, left=0.075, right=0.985, top=0.82, bottom=0.10,
                  wspace=0.42, hspace=0.40)

    # panels a-g: 7 site scatters ; panel h: summary
    cells = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    top_axes = []
    for i, (u, (r, c)) in enumerate(zip(UNITS, cells)):
        ax = fig.add_subplot(gs[r, c])
        # bottom-row scatters and top-row col-3 (which sits above the summary, not a scatter)
        scatter(ax, u, show_xlabel=(r == 1 or (r, c) == (0, 3)), show_ylabel=(c == 0))
        if r == 0:
            top_axes.append(ax)
    ax_sum = fig.add_subplot(gs[1, 3])
    r5, rb, pval = summary(ax_sum)

    # header: title, legend, dataset-size note. Placed at provisional positions (va='top' so each
    # y is the box's TOP edge), then re-flowed so the three whitespace gaps -- title->legend,
    # legend->note, note->top-of-first-row -- are identical.
    title_txt = fig.text(0.5, 0.99, _M['title'], fontsize=FS_TITLE + 2, fontweight='bold',
                         ha='center', va='top', fontfamily=FONT_FAMILY)
    note_txt = fig.text(0.5, 0.90, r'$n$ = 140 stimuli   (7 sites $\times$ 10 seeds $\times$ 2 target models)',
                        fontsize=FS_LEG, ha='center', va='top', fontfamily=FONT_FAMILY, color='#555555')
    # legend markers + text identical to the companion controversial figure (boxed, 4 entries)
    leg = [('o', C_R50, 'R50 pred, R50 stim'), ('o', C_R50_FADE, 'R50 pred, Robust stim'),
           ('s', C_ROBUST, 'Robust pred, Robust stim'), ('s', C_ROBUST_FADE, 'Robust pred, R50 stim')]
    handles = [Line2D([0], [0], marker=mk, linestyle='none', markerfacecolor=fc, markeredgecolor='black',
                      markeredgewidth=0.6, markersize=6, label=lab) for mk, fc, lab in leg]
    lg = fig.legend(handles=handles, loc='upper center', ncol=4,
                    prop=dict(family=FONT_FAMILY, size=FS_LEG),
                    frameon=True, bbox_to_anchor=(0.5, 0.95), columnspacing=1.25,
                    handletextpad=0.4, borderaxespad=0)
    lg.get_frame().set_edgecolor('#CCC'); lg.get_frame().set_facecolor('white')

    # panel letters
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    all_cells = cells + [(1, 3)]
    top_letters = []
    for (r, c), let in zip(all_cells, letters):
        pos = gs[r, c].get_position(fig)
        lt = fig.text(pos.x0 - 0.028, pos.y1 + 0.018, let, ha='left', va='bottom',
                      fontsize=10, fontweight='bold', fontfamily=FONT_FAMILY)
        if r == 0:
            top_letters.append(lt)

    # equalise the title/legend/note/row-1 whitespace gaps from measured element heights
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer(); _inv = fig.transFigure.inverted()
    def _frac_h(obj):
        bb = obj.get_window_extent(_rend)
        return _inv.transform((0, bb.y1))[1] - _inv.transform((0, bb.y0))[1]
    def _frac_top(obj):
        return _inv.transform((0, obj.get_window_extent(_rend).y1))[1]
    row1_top = max([_frac_top(a.title) for a in top_axes] + [_frac_top(lt) for lt in top_letters])
    h_title, h_leg, h_note = _frac_h(title_txt), _frac_h(lg), _frac_h(note_txt)
    title_top = 0.99
    gap = (title_top - h_title - h_leg - h_note - row1_top) / 3.0
    legend_top = title_top - h_title - gap
    note_top = legend_top - h_leg - gap
    title_txt.set_y(title_top)
    lg.set_bbox_to_anchor((0.5, legend_top), transform=fig.transFigure)
    note_txt.set_y(note_top)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('robust-model mean r =', round(rb.mean(), 4))
    print('standard-model mean r =', round(r5.mean(), 4))
    print('Wilcoxon p =', round(pval, 4), '; n sites =', len(UNITS))
    print('saved ->', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    main(**vars(ap.parse_args()))
