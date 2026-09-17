#!/usr/bin/env python3
"""Supp fig (layer_selection) - per-channel encoding-layer selection.

Ridge regression with cross-validated layer selection (Table 3): for each of the
250 site-model combinations the best-predicting block is chosen per channel by
validation R2 across the scanned candidate layers.

Panel a: per-model validation-R2-vs-relative-depth curves (one line per site,
coloured by recorded region), with the selected peak block marked.
Panel b: selected peak relative depth per model (swarm + median bar).
Panel c: selected relative depth (mean, range) per model (reproduces Table 3).
Panel d: read-out depth vs recorded visual area (per-model mean + grand mean).
Panel e: area-depth rank correlation (per-model ventral Spearman + significance).
Panel f: site-depth rank correlation (per-model Spearman, venus resolved into its 5 sites).

Reads only loader.load_layer_selection() (preprocessed_data/layer_selection.pkl). S-number
and output stem come from the manifest.
"""
import os

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy import stats

apply_figure_style()
_M = _fig('layer_selection'); STEM = output_name('layer_selection')
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'   # display relabel

# region maps to its monkey's canonical colour; STS (leap/three0) share leap's.
AREA_COLOR = {'V3': MONKEY_COLORS['venus'], 'V4': MONKEY_COLORS['venus'],
              'cIT': MONKEY_COLORS['paul'], 'aIT': MONKEY_COLORS['red'],
              'STS': MONKEY_COLORS['leap']}
AREA_ORDER = ['V3', 'V4', 'cIT', 'aIT', 'STS']   # posterior -> anterior, STS last
VENTRAL_AREAS = ['V3', 'V4', 'cIT', 'aIT']       # ordered posterior -> anterior

FS_TITLE, FS_AX, FS_TICK, FS_LET = 7, 6.5, 5.5, 11


def load():
    D = L.load_layer_selection()
    MODELS = list(L.config()['models'])
    keys = [k for k in D if isinstance(k, tuple)]          # (monkey, model, unit)
    area = {(mk, u): str(D[(mk, m, u)]['area'])
            for (mk, m, u) in keys}                        # region per site
    depth = {(mk, u): float(D[(mk, m, u)]['depth'])
             for (mk, m, u) in keys}                       # probe depth (a.u.) per site
    sites = sorted({(mk, u) for (mk, m, u) in keys},
                   key=lambda s: (AREA_ORDER.index(area[s]), s[0], s[1]))
    return D, MODELS, keys, sites, area, depth


# --- panel a: per-model layer-selection curves ---------------------------------
def draw_curves(ax, D, sites, area, model, show_x, show_y):
    for (mk, u) in sites:
        v = D.get((mk, model, u))
        if v is None:
            continue
        col = AREA_COLOR[area[(mk, u)]]
        ax.plot(v['reldepth'], v['curve'], '-', color=col, alpha=0.35, lw=0.5, zorder=2)
        ax.scatter(v['peak_reldepth'], v['peak_r2'], s=6, color=col, alpha=0.9,
                   edgecolors='white', linewidths=0.25, zorder=4)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 0.85)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.4, 0.8])
    ax.set_title(MODEL_SHORT_NAMES.get(model, model), fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, color=get_model_color(model),
                 fontweight='bold' if model in ROBUST_MODELS else 'normal', pad=2)
    ax.tick_params(labelsize=FS_TICK)
    if show_x:
        ax.set_xlabel('Relative layer depth', fontsize=FS_AX)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel(r'Validation $R^2$', fontsize=FS_AX)
    else:
        ax.set_yticklabels([])


# --- panel b: selected peak relative depth per model ---------------------------
def draw_peak_by_model(ax, D, sites, area, models):
    rng = np.random.default_rng(3)
    for i, m in enumerate(models):
        vals, cols = [], []
        for (mk, u) in sites:
            v = D.get((mk, m, u))
            if v:
                vals.append(v['peak_reldepth']); cols.append(AREA_COLOR[area[(mk, u)]])
        vals = np.array(vals)
        ax.scatter(i + rng.uniform(-0.26, 0.26, len(vals)), vals, s=8, c=cols,
                   alpha=0.85, edgecolors='white', linewidths=0.25, zorder=4)
        med = float(np.median(vals))
        ax.plot([i - 0.34, i + 0.34], [med, med], color='black', lw=1.8,
                solid_capstyle='round', zorder=6)
        ax.plot([i - 0.30, i + 0.30], [med, med], color=get_model_color(m), lw=1.0,
                solid_capstyle='round', zorder=7)
    _model_xticks(ax, models)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Selected relative depth', fontsize=FS_AX)
    ax.set_title('Selected depth per model', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=3)
    ax.tick_params(axis='y', labelsize=FS_TICK)


# --- panel c: selected depth mean/range per model (Table 3) --------------------
def draw_table3(ax, D, keys, models):
    rows = []
    for m in models:
        ents = [D[k] for k in keys if k[1] == m]
        reld = np.array([e['peak_reldepth'] for e in ents])
        rows.append(dict(m=m, mean=float(reld.mean()), lo=float(reld.min()),
                         hi=float(reld.max()), n=len(ents)))
    rows.sort(key=lambda r: r['mean'])
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        col = get_model_color(r['m'])
        ax.plot([r['lo'], r['hi']], [y, y], color=col, lw=1.0, alpha=0.55,
                solid_capstyle='round', zorder=2)
        ax.scatter(r['mean'], y, s=22, color=col, edgecolors='black', linewidths=0.4,
                   zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_SHORT_NAMES.get(r['m'], r['m']) for r in rows],
                       fontsize=FS_TICK)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(get_model_color(r['m']))
        if r['m'] in ROBUST_MODELS:
            t.set_fontweight('bold')
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Selected relative depth (mean, range)', fontsize=FS_AX)
    ax.set_title('Read-out depth summary', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=3)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    return rows


# --- region-resolved read-out depth (ported from take7 figS05 rows 3-4) --------
def _venus_order(sites, depth):
    """Venus's 5 sites, posterior -> anterior = deepest (V3) -> shallowest (V4)."""
    vs = [(depth[('venus', u)], u) for (mk, u) in sites if mk == 'venus']
    return [u for _, u in sorted(vs, key=lambda t: -t[0])]


def _area_mean(D, sites, area, model, a):
    vals = [D[(mk, model, u)]['peak_reldepth'] for (mk, u) in sites
            if area[(mk, u)] == a and (mk, model, u) in D]
    return np.mean(vals) if vals else np.nan


def draw_area_depth(ax, D, sites, area, models):
    """Mean selected depth per model across recorded visual areas (V3 -> STS)."""
    xr = np.arange(len(AREA_ORDER))
    allm = []
    for m in models:
        ys = [_area_mean(D, sites, area, m, a) for a in AREA_ORDER]
        allm.append(ys)
        ax.plot(xr, ys, '-', color=get_model_color(m), alpha=0.5, lw=0.8, zorder=3)
    ax.plot(xr, np.nanmean(allm, axis=0), '-o', color='#111', lw=1.8, ms=3.5, zorder=6)
    ax.axvspan(-0.4, len(VENTRAL_AREAS) - 0.6, color='#999', alpha=0.07, zorder=0)
    ax.set_xticks(xr); ax.set_xticklabels(AREA_ORDER, fontsize=FS_TICK)
    for t, a in zip(ax.get_xticklabels(), AREA_ORDER):
        t.set_color(AREA_COLOR[a]); t.set_fontweight('bold')
    ax.set_xlim(-0.4, len(AREA_ORDER) - 0.6); ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Mean selected depth', fontsize=FS_AX)
    ax.set_title('Read-out depth vs recorded visual area', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=3)
    ax.tick_params(axis='y', labelsize=FS_TICK)


# --- hierarchical correspondence (restored from take7) -------------------------
def _stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def ventral_rho(D, sites, area, model):
    """Spearman of selected depth vs coarse ventral area rank (V3<V4<cIT<aIT)."""
    xs, ys = [], []
    for (mk, u) in sites:
        a = area[(mk, u)]
        if a in VENTRAL_AREAS and (mk, model, u) in D:
            xs.append(VENTRAL_AREAS.index(a))
            ys.append(D[(mk, model, u)]['peak_reldepth'])
    return stats.spearmanr(xs, ys)


def fine_rho(D, sites, area, depth, model):
    """Spearman over ventral channels with venus's 5 sites individually ordered."""
    xs, ys = [], []
    for i, u in enumerate(_venus_order(sites, depth)):
        xs.append(i); ys.append(D[('venus', model, u)]['peak_reldepth'])
    for (mk, u) in sites:
        a = area[(mk, u)]
        if a == 'cIT':
            xs.append(5); ys.append(D[(mk, model, u)]['peak_reldepth'])
        elif a == 'aIT':
            xs.append(6); ys.append(D[(mk, model, u)]['peak_reldepth'])
    return stats.spearmanr(xs, ys)


def _draw_rho_bars(ax, rhos, title, xlabel):
    rhos = sorted(rhos, key=lambda t: t[1])
    ys = np.arange(len(rhos))
    ax.barh(ys, [r for _, r, _ in rhos],
            color=[get_model_color(m) for m, _, _ in rhos], edgecolor='white',
            linewidth=0.4, zorder=3)
    for yi, (_, r, p) in zip(ys, rhos):             # per-model Spearman significance
        ax.text(r + (0.03 if r >= 0 else -0.03), yi, _stars(p), va='center',
                ha='left' if r >= 0 else 'right', fontsize=FS_TICK - 0.5, color='#333')
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_SHORT_NAMES.get(m, m) for m, _, _ in rhos], fontsize=FS_TICK)
    for t, (m, _, _) in zip(ax.get_yticklabels(), rhos):
        t.set_color(get_model_color(m))
        if m in ROBUST_MODELS:
            t.set_fontweight('bold')
    ax.axvline(0, color='#888', lw=0.6)
    ax.set_ylim(-0.6, len(rhos) - 0.4)
    ax.set_xlim(-0.4, 1.15); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(xlabel, fontsize=FS_AX)
    ax.set_title(title, fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=3)
    ax.tick_params(axis='x', labelsize=FS_TICK)


def draw_area_rho(ax, D, sites, area, models):
    rhos = [(m,) + tuple(ventral_rho(D, sites, area, m)) for m in models]
    _draw_rho_bars(ax, rhos, 'Area-depth rank correlation',
                   r'Ventral $\rho$ (area vs depth)')


def draw_site_rho(ax, D, sites, area, depth, models):
    rhos = [(m,) + tuple(fine_rho(D, sites, area, depth, m)) for m in models]
    _draw_rho_bars(ax, rhos, 'Site-depth rank correlation',
                   r'Ventral $\rho$ (site vs depth)')


def _model_xticks(ax, models):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_SHORT_NAMES.get(m, m) for m in models], rotation=40,
                       ha='right', fontsize=FS_TICK)
    for t, m in zip(ax.get_xticklabels(), models):
        t.set_color(get_model_color(m))
        if m in ROBUST_MODELS:
            t.set_fontweight('bold')
    ax.set_xlim(-0.6, len(models) - 0.4)


# --- build ---------------------------------------------------------------------
def main(out_dir):
    D, MODELS, keys, sites, area, depth = load()

    FIG_W, FIG_H = WIDTH_2COL_MM * MM, 172 * MM      # one row dropped (venus-resolved site panel) + tighter rows
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_FACECOLOR)
    gs = GridSpec(3, 1, figure=fig, left=0.075, right=0.80, top=0.930, bottom=0.075,
                  hspace=0.44, height_ratios=[1.78, 0.92, 0.98])

    # panel a: 2 x 5 curves in one nested block, with a tight internal row gap
    gsA = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0], wspace=0.28, hspace=0.22)
    axA = []
    for i, m in enumerate(MODELS):
        r = 0 if i < 5 else 1
        ax = fig.add_subplot(gsA[r, i % 5])
        draw_curves(ax, D, sites, area, m, show_x=(r == 1), show_y=(i % 5 == 0))
        axA.append(ax)

    # panel b + c side by side
    gsBC = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.55,
                                   width_ratios=[1.25, 1.0])
    axB = fig.add_subplot(gsBC[0]); draw_peak_by_model(axB, D, sites, area, MODELS)
    axC = fig.add_subplot(gsBC[1]); rows = draw_table3(axC, D, keys, MODELS)

    # panel d (area read-out depth) + the two rank-correlation bars on ONE condensed row
    # (venus-resolved per-site line panel dropped; f & g moved up into its place)
    gsDFG = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.60,
                                    width_ratios=[1.0, 0.70, 0.70])
    axD = fig.add_subplot(gsDFG[0]); draw_area_depth(axD, D, sites, area, MODELS)
    axF = fig.add_subplot(gsDFG[1]); draw_area_rho(axF, D, sites, area, MODELS)
    axG = fig.add_subplot(gsDFG[2]); draw_site_rho(axG, D, sites, area, depth, MODELS)

    # title + panel letters
    fig.text(0.075, 0.965, 'Per-channel encoding-layer selection', fontsize=FS_TITLE + 3,
             fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)

    # region legend (top-right)
    _leg = [('aIT', AREA_COLOR['aIT']), ('cIT', AREA_COLOR['cIT']),
            ('V3/V4', AREA_COLOR['V3']), ('STS', AREA_COLOR['STS'])]
    handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='white', ms=4, label=lab)
               for lab, c in _leg]
    leg = fig.legend(handles=handles, loc='upper right', ncol=4, fontsize=FS_TICK,
                     frameon=True, bbox_to_anchor=(0.80, 0.978), columnspacing=0.9,
                     handletextpad=0.25, borderpad=0.4)
    leg.get_frame().set(facecolor='white', edgecolor='#cccccc', linewidth=0.6)

    fig.canvas.draw()
    pa = axA[0].get_position(); pb = axB.get_position(); pc = axC.get_position()
    pd = axD.get_position(); pf = axF.get_position(); pg = axG.get_position()
    for letter, p, dx in (('a', pa, 0.055), ('b', pb, 0.055), ('c', pc, 0.055),
                          ('d', pd, 0.055), ('e', pf, 0.040), ('f', pg, 0.040)):
        fig.text(p.x0 - dx, p.y1 + 0.010, letter, fontsize=FS_LET, fontweight='bold',
                 ha='left', va='bottom', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print(f'Saved -> {out}')
    print('\nTable 3 reproduction (sorted by mean depth):')
    for r in rows:
        print(f"  {MODEL_SHORT_NAMES.get(r['m'], r['m']):11s}  mean={r['mean']:.2f}  "
              f"range {r['lo']:.2f}-{r['hi']:.2f}  n={r['n']}")
    print('\nHierarchical correspondence (per-model Spearman):')
    for m in MODELS:
        ar, ap = ventral_rho(D, sites, area, m)
        sr, sp = fine_rho(D, sites, area, depth, m)
        print(f"  {MODEL_SHORT_NAMES.get(m, m):11s}  area rho={ar:+.2f} {_stars(ap):>4s}  "
              f"site rho={sr:+.2f} {_stars(sp):>4s}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
