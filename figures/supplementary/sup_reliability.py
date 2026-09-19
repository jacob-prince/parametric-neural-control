#!/usr/bin/env python3
r"""Supp fig (reliability) - channel selection, response reliability, NSD noise ceiling.

Four panels covering how the 25 targeted electrodes were chosen and how reliable they are:

  a  Split-half reliability by channel: per monkey, every channel on the array/probe
     (grey) with the five TARGETED sites highlighted in the animal color and a median
     line. Selection kept the most reliable AND tuning-diverse channels. Reliability =
     Spearman-Brown-corrected split-half r (2r/(1+r)) from the raw encoding HDF5.
  b  Per-electrode calibration-phase noise ceiling for the 25 targeted channels: the maximum
     model-data Pearson r for the trial-averaged calibration response, computed from the SAME
     calibration data as panel a as sqrt(Spearman-Brown split-half reliability) (the NSD nc_r
     identity). All five animals have a valid ceiling (control-phase ceilings are a separate
     supplementary figure).
  c  Tuning stability: per-site cross-day (within encoding) and cross-phase (encoding vs
     control) Pearson over the shared natural images, grouped by region.
  d  Tuning correlation among the five selected channels: per monkey a 5x5 Pearson matrix
     over the encoding images - selected channels are reliable and tuning-diverse.

Numbered from the manifest. Panel c reads pnc.preproc.loader (load_tuning_stability).
Panels a/b/d read the channel-selection cache
(preprocessed_data/sup_reliability_channel_selection_cache.pkl, built by
scripts/preprocessing/build_sup_reliability_cache.py from the raw encoding HDF5 - the
all-channel grey cloud is not in the preproc cache).
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MONKEY_COLORS)
apply_figure_style()

_M = _fig('reliability'); STEM = output_name('reliability')
CS_CACHE = os.path.join(str(paths.preprocessed_data()), 'sup_reliability_channel_selection_cache.pkl')

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
CMAP = 'PuOr_r'                       # diverging, NOT the red/blue reserved for levels
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 14, 9.5, 6.5, 6.0, 5.5
FS_XTICK_A, FS_XTICK_BC = 8.5, 7.0        # bigger x-tick labels: region names (a) / per-electrode (b,c)
W = 0.36


# ---- data: panel b calibration-phase noise ceiling --------------------------
def calibration_ceilings(cs):
    """Per targeted channel: calibration-phase noise ceiling = the maximum model-data Pearson r
    for the trial-averaged calibration response = sqrt(Spearman-Brown split-half reliability)
    (the NSD nc_r identity, on the calibration data). Uses the SAME per-channel reliabilities as
    panel a, so all five animals have a valid ceiling (control-phase ceilings are a separate
    supplementary figure)."""
    out = []
    for mk in MONKEYS:
        rel = cs[mk]['rel']
        for u in cs[mk]['sel']:
            out.append(dict(monkey=mk, unit=int(u), nc=float(np.sqrt(np.clip(rel[u], 0.0, 1.0)))))
    return out


# ---- data: panels a/d (channel-selection cache; built from the raw encoding HDF5) ----
def load_channel_selection():
    path = paths.require(CS_CACHE, hint='python scripts/preprocessing/build_sup_reliability_cache.py')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---- panel a: split-half reliability by channel -----------------------------
def panel_a(ax, cs):
    rng = np.random.default_rng(1)
    for i, m in enumerate(MONKEYS):
        rel = cs[m]['rel']; rel_ok = rel[np.isfinite(rel)]
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(rel_ok)), rel_ok, s=5,
                   color='#CFCFCF', alpha=0.55, edgecolors='none', zorder=2)
        rs = rel[cs[m]['sel']]
        lo = np.nanmin(rs)
        ax.plot([i - 0.34, i + 0.34], [lo, lo], color=MONKEY_COLORS[m], lw=1.0,
                ls=':', alpha=0.8, zorder=3)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(rs)), rs, s=26,
                   color=MONKEY_COLORS[m], alpha=0.95, edgecolors='white',
                   linewidth=0.7, zorder=5)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'{TAG[m]} {REGION[m]}' for m in MONKEYS], fontsize=FS_XTICK_A)
    for t, m in zip(ax.get_xticklabels(), MONKEYS):
        t.set_color(MONKEY_COLORS[m]); t.set_fontweight('bold')
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.3, 1.0)
    ax.set_ylabel('Spearman-Brown\nsplit-half reliability (r)', fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.scatter([], [], s=26, color='#888888', edgecolors='white', linewidth=0.7,
               label='Targeted (5/monkey)')
    ax.scatter([], [], s=5, color='#CFCFCF', label='All channels')
    ax.legend(loc='lower center', fontsize=FS_ANNOT, frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=1.0, borderpad=0.2)


# ---- panel b: per-site noise ceiling ----------------------------------------
def panel_b(ax, sites):
    x, ticks, bounds, centers = 0, [], {}, {}
    for mk in MONKEYS:
        ss = [s for s in sites if s['monkey'] == mk]; start = x; c = MONKEY_COLORS[mk]
        for s in ss:
            if np.isfinite(s['nc']):
                ax.bar(x, s['nc'], 0.8, color=c, edgecolor='white', linewidth=0.4, zorder=3)
            else:
                ax.bar(x, 1.0, 0.8, color=c, alpha=0.16, edgecolor=c, linewidth=0.4,
                       hatch='////', zorder=2)
                ax.text(x, 0.03, 'n/a', ha='center', va='bottom', rotation=90,
                        fontsize=FS_ANNOT - 0.5, color=c)
            ticks.append((x, f'{TAG[mk]}{s["unit"]}', c)); x += 1
        bounds[mk] = (start - 0.5, x - 0.5); centers[mk] = (start + x - 1) / 2; x += 0.8
    ok = [s['nc'] for s in sites if np.isfinite(s['nc'])]
    mean_nc = float(np.mean(ok))
    ax.axhline(mean_nc, color='#111', ls='--', lw=0.8, zorder=4)
    ax.text(1.0, 1.015, f'mean r = {mean_nc:.2f}', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], rotation=90, fontsize=FS_XTICK_BC)
    for tk, t in zip(ax.get_xticklabels(), ticks):
        tk.set_color(t[2])
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.8, x - 0.8)
    ax.set_ylabel('Calibration noise ceiling (Pearson r)', fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    return mean_nc


# ---- panel c: tuning stability ----------------------------------------------
def panel_c(ax, ts):
    sites = ts['sites']
    x, ticks, bounds, centers = 0, [], {}, {}
    for mk in MONKEYS:
        ss = [s for s in sites if s['monkey'] == mk]; start = x; c = MONKEY_COLORS[mk]
        for s in ss:
            ax.bar(x - W / 2, s['cross_day'], W, color=c, alpha=0.42, hatch='////',
                   edgecolor=c, linewidth=0.4, zorder=3)
            ax.bar(x + W / 2, s['cross_phase'], W, color=c, edgecolor='white',
                   linewidth=0.4, zorder=3)
            ticks.append((x, f'{TAG[mk]}{s["unit"]}', c)); x += 1
        bounds[mk] = (start - 0.5, x - 0.5); centers[mk] = (start + x - 1) / 2; x += 0.8
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], rotation=90, fontsize=FS_XTICK_BC)
    for tk, t in zip(ax.get_xticklabels(), ticks):
        tk.set_color(t[2])
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.8, x - 0.8)
    ax.set_ylabel('Tuning correlation (Pearson r)', fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.legend(handles=[Patch(fc='#888', hatch='////', alpha=0.42, ec='#888',
                             label='cross-day (within encoding)'),
                       Patch(fc='#888', ec='white', label='cross-phase (encoding vs control)')],
              loc='upper right', fontsize=FS_ANNOT, frameon=False, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.2)


# ---- panel d: 5x5 tuning-correlation heatmaps -------------------------------
def panel_d(fig, gs_d, cs):
    gsD = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_d, wspace=0.42)
    im = None
    axes = []
    for i, m in enumerate(MONKEYS):
        ax = fig.add_subplot(gsD[0, i]); axes.append(ax)
        tc = cs[m]['tc']; sel = cs[m]['sel']; n = len(sel)
        im = ax.imshow(tc, cmap=CMAP, vmin=-1, vmax=1, aspect='equal')
        for r in range(n):
            for c in range(n):
                ax.text(c, r, f'{tc[r, c]:.2f}', ha='center', va='center',
                        fontsize=5.5, fontfamily=FONT_FAMILY,
                        color=('white' if abs(tc[r, c]) > 0.6 else '#222222'))
            ax.add_patch(plt.Rectangle((r - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor='#333333', lw=0.8))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(sel, fontsize=5.0, rotation=90)
        ax.set_yticklabels(sel, fontsize=5.0)
        ax.tick_params(length=0)
        ax.set_title(f'{TAG[m]} {REGION[m]}', fontsize=FS_TITLE - 0.5, fontfamily=FONT_FAMILY,
                     fontweight='bold', color=MONKEY_COLORS[m], pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
    return im, axes


def main(out_dir):
    ts = L.load_tuning_stability()
    cs = load_channel_selection()
    sites = calibration_ceilings(cs)

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 198 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(4, 1, figure=fig, left=0.085, right=0.965, top=0.95, bottom=0.055,
                  height_ratios=[1.05, 1.0, 1.0, 0.72], hspace=0.50)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[2, 0])
    panel_a(axA, cs)
    mean_nc = panel_b(axB, sites)
    panel_c(axC, ts)
    im, axesD = panel_d(fig, gs[3, 0], cs)
    for a in axesD:                     # nudge panel D down so its title clears panel C's x-tick labels
        p = a.get_position(); a.set_position([p.x0, p.y0 - 0.018, p.width, p.height])

    for a in (axA, axB, axC):
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)

    fig.canvas.draw()

    # colorbar for panel d, anchored to the right of the heatmap row
    bbD = axesD[-1].get_position()
    cax = fig.add_axes([bbD.x1 + 0.012, bbD.y0, 0.010, bbD.height])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS_ANNOT)
    cb.set_label('tuning correlation (r)', fontsize=FS_AX)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    # panel titles (over each axis / row) + bold letters
    row_titles = [(axA, 'a', 'Split-half reliability by channel'),
                  (axB, 'b', 'Calibration noise ceiling per electrode'),
                  (axC, 'c', 'Tuning stability across days and phases')]
    for a, letter, title in row_titles:
        bb = a.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.010, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
        fig.text(0.012, bb.y1 + 0.026, letter, ha='left', va='top', fontsize=FS_LET,
                 fontweight='bold', fontfamily=FONT_FAMILY)

    # panel d title + letter (over the heatmap row)
    bbL, bbR = axesD[0].get_position(), axesD[-1].get_position()
    dx0, dx1, dy1 = bbL.x0, bbR.x1, bbL.y1
    fig.text((dx0 + dx1) / 2, dy1 + 0.026, 'Tuning correlation among selected channels',
             ha='center', va='bottom', fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
    fig.text(0.012, dy1 + 0.042, 'd', ha='left', va='top', fontsize=FS_LET,
             fontweight='bold', fontfamily=FONT_FAMILY)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print(f'mean calibration noise ceiling (25 sites) = {mean_nc:.4f}')
    print('Saved ->', path)
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
