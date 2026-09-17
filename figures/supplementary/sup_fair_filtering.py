#!/usr/bin/env python3
"""Supp fig (fair_filtering) — structure of synthesis failures.

A synthesized stimulus "fails" when its own post-hoc prediction (the faithful
pred_resp for the SAVED image, the prediction-cache diagonal) misses the intended
target by more than a tolerance, measured as a fraction of the channel's target
range: relerr = |target - achieved| / (per-channel target range) > tau. The target
grid is identical across models within a channel, so this is fair across channels
and models; tau = 2% / 5% of range are the two stringencies. Failures are not
uniform: they concentrate at the strongest drive levels (9, 10) AND in specific
models (panel a). Panel b shows the magnitude of the misses among failures.

The companion figure ("Fair cross-model exclusion regimes") lays out the candidate
exclusion criteria that restore a common footing, shows how each selects data at one
example site, and quantifies retention at both tau.

Masks + relerr come straight from the preproc
loader (`L.load_exclusions`, `masks[tau][criterion]`).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, DPI, FIG_FACECOLOR, save_fig,
                       MM, WIDTH_2COL_MM,
                       MODEL_ORDER, MODEL_SHORT_NAMES, MODEL_COLORS, get_model_color,
                       ROBUST_MODELS)

apply_figure_style()
STEM = output_name('fair_filtering')

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'   # display relabel
THRESHOLDS = [0.02, 0.05]           # display order: strict, then lenient
FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 7, 6.5, 5.5, 5.5   # final-size-native (5-7 pt at 180 mm)
FS_LETTER = 10                       # panel letters: clearly larger than the 7 pt titles
FAIL_C = '#E15759'


# ── data: pull straight from the precomputed masks (no rebuild) ─────────────
def load_all():
    """Concatenate the exclusion cache over all 5 monkeys. Everything the
    figure needs is precomputed in the cache — relerr, plus (gen_model, level_ord)."""
    model, level = [], []
    relerr = {t: [] for t in THRESHOLDS}
    for m in MONKEYS:
        ex = L.load_exclusions(m)
        model.append(np.asarray(ex['gen_model']))
        level.append(np.asarray(ex['level_ord']).astype(int))
        for t in THRESHOLDS:
            relerr[t].append(np.asarray(ex['relerr'], float))   # relerr is tau-independent (raw)
    model = np.concatenate(model)
    level = np.concatenate(level)
    relerr = {t: np.concatenate(v) for t, v in relerr.items()}
    return dict(model=model, level=level, relerr=relerr)


def success_matrix(D, t):
    """(model x level) success rate = relerr <= t, over all sites/seeds."""
    MO, LO = D['model'], D['level']
    NLEV = int(LO.max()) + 1
    succ = D['relerr'][t] <= t
    return np.array([[100 * succ[(MO == m) & (LO == lv)].mean()
                      for lv in range(NLEV)] for m in MODEL_ORDER])


# ── panel a: failure structure ───────────────────────────────────────────────
def draw_success_heatmap(ax, SUCC, NLEV, t, show_ylabels):
    im = ax.imshow(SUCC[t], aspect='equal', cmap='RdYlGn', vmin=0,
                   vmax=100, interpolation='nearest')
    ax.set_xticks(range(NLEV)); ax.set_xticklabels(range(NLEV), fontsize=FS_TICK)
    ax.set_yticks(range(len(MODEL_ORDER)))
    if show_ylabels:
        ax.set_yticklabels([SHORT[m] for m in MODEL_ORDER], fontsize=FS_TICK)
        for tl, m in zip(ax.get_yticklabels(), MODEL_ORDER):
            tl.set_color(get_model_color(m))
            if m in ROBUST_MODELS:
                tl.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('Feature level', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ax.set_title(f'Success rate  ({t:.0%} of range)', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=4)
    ax.tick_params(length=2)
    return im


def draw_error_hist(ax, D):
    """Magnitude of the misses among failure cases only (relerr > tau_strict); the
    success spike near 0 is excluded so the failure tail is visible."""
    tp, tr = THRESHOLDS[0], THRESHOLDS[1]
    err = D['relerr'][tp]                                 # raw relerr (tau-independent)
    fail = err[err > tp] * 100                            # as % of channel range
    hi = float(np.ceil(np.percentile(fail, 99) / 5) * 5)
    ax.hist(np.clip(fail, tp * 100, hi), bins=np.linspace(tp * 100, hi, 35),
            color=FAIL_C, edgecolor='white', linewidth=0.3, zorder=3)
    ax.axvline(tr * 100, color='#E67E22', ls='--', lw=0.8, zorder=5)
    ax.text(tr * 100, ax.get_ylim()[1] * 0.98, f' {tr:.0%} of range', color='#E67E22',
            fontsize=FS_ANNOT, ha='left', va='top', fontweight='bold', rotation=90,
            fontfamily=FONT_FAMILY)
    ax.set_xlabel('miss  |target $-$ achieved|  (% of range)', fontsize=FS_AX,
                  fontfamily=FONT_FAMILY)
    ax.set_ylabel('Failure cases', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ax.set_title('Failure-magnitude distribution', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=4)
    ax.set_xlim(tp * 100, hi); ax.tick_params(labelsize=FS_TICK)


# ── build ────────────────────────────────────────────────────────────────────
# Heatmap cells are square (aspect='equal'): NLEV columns x n-models rows, so the
# native content aspect is NLEV/len(MODEL_ORDER). We size the figure height so each
# heatmap *box* matches that content aspect exactly — no pillar-box dead bands, and
# the title/panel-letter (anchored to the box) sit tight against the map.
FIG_W_IN = WIDTH_2COL_MM * MM
LEFT, RIGHT, TOP, BOTTOM = 0.075, 0.965, 0.80, 0.17

# Two independently-spaced blocks: panel a (2 heatmaps + a dedicated colorbar column)
# and panel b (the histogram). A uniform inner wspace couples the heat|heat gap to the
# heat|colorbar gap, so we keep it small; the (larger) a|b separation is the OUTER gap.
AB_WSPACE = 0.22                                     # gap between panel a block and panel b
A_WSPACE = 0.18                                      # gap between the two heatmaps / colorbar
A_WRATIOS = [1.0, 1.0, 0.055]                        # heatmap, heatmap, colorbar
AB_WRATIOS = [2.0, 1.15]                             # panel-a block vs histogram


def build_layout(fig_h_in):
    """Lay out the grid at a given figure height and return (fig, axes, heat-box-width-in)."""
    f = plt.figure(figsize=(FIG_W_IN, fig_h_in), facecolor=FIG_FACECOLOR)
    o = GridSpec(1, 2, figure=f, left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
                 wspace=AB_WSPACE, width_ratios=AB_WRATIOS)
    g = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], wspace=A_WSPACE,
                                width_ratios=A_WRATIOS)
    a1 = f.add_subplot(g[0]); a2 = f.add_subplot(g[1]); c = f.add_subplot(g[2])
    b = f.add_subplot(o[1])
    f.canvas.draw()
    box_w_in = a1.get_position().width * FIG_W_IN     # width-only, independent of fig height
    return f, (a1, a2, c, b), box_w_in


def main(out_dir):
    D = load_all()
    NLEV = int(D['level'].max()) + 1
    SUCC = {t: success_matrix(D, t) for t in THRESHOLDS}
    CELL_ASPECT = NLEV / len(MODEL_ORDER)                # width/height of the square-cell map

    # pass 1: measure the true heatmap box width, then solve the exact height for square cells
    _probe, _, _box_w = build_layout(3.0)
    plt.close(_probe)
    FIG_H_IN = (_box_w / CELL_ASPECT) / (TOP - BOTTOM)

    # pass 2: real figure at the solved height
    fig, (axA1, axA2, cax, axB), _ = build_layout(FIG_H_IN)
    draw_success_heatmap(axA1, SUCC, NLEV, THRESHOLDS[0], True)
    imA = draw_success_heatmap(axA2, SUCC, NLEV, THRESHOLDS[1], False)
    cbA = fig.colorbar(imA, cax=cax)
    # label sits ABOVE the bar (horizontal) so it never protrudes right into panel b
    cax.set_title('% reaching\ntarget', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY,
                  pad=3, linespacing=0.95)
    cbA.set_ticks([0, 25, 50, 75, 100]); cbA.ax.tick_params(labelsize=FS_TICK)
    draw_error_hist(axB, D)

    # panel letters — anchored to the top of each subplot band, clearly larger than titles
    fig.canvas.draw()
    for ax_ref, lab in [(axA1, 'a'), (axB, 'b')]:
        x = ax_ref.get_position().x0 - 0.055
        fig.text(max(x, 0.012), ax_ref.get_position().y1 + 0.055, lab, fontsize=FS_LETTER,
                 fontweight='bold', fontfamily=FONT_FAMILY, va='top', ha='left')

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)

    print(f'Saved -> {out}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
