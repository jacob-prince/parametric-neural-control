#!/usr/bin/env python3
"""Supp. Fig. (slug high_control_examples) - top high-control example sweeps.

For the highest control-r site-model combinations, show the accentuation sweep
alongside the predicted-vs-measured control scatter (per-seed connectors, dots
colored by feature level, model-colored linear fit) with the pair's r and slope.
Each example shows the TWO highest-R2 seed sweeps as a 2x6 image grid (one seed per
row, suppress -> drive) beside the scatter. Per-monkey galleries span the manifest
number block (base num FROM MANIFEST, never hardcoded). Self-contained : data via
pnc.preproc.loader; raw stimulus images from STIMULI_CONTROL_PATH.

    python -m figures.supplementary.sup_high_control_examples --out DIR [--monkey red]
"""
import os
import glob
from collections import defaultdict

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color,
                       STIMULI_CONTROL_PATH)
apply_figure_style(); _M = _fig('high_control_examples')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import stats

MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'

CTRL = STIMULI_CONTROL_PATH

MONKEY_TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
MONKEY_REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}

# One figure per monkey, each showing that monkey's top-N site-model combos by control r.
# S-numbers come from the manifest span (base = fig num); monkey index sets the offset.
_BASE_NUM = _M['num']
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
MONKEY_NUM = [(mk, _BASE_NUM + i) for i, mk in enumerate(MONKEYS)]
N_EX = 4                                     # top examples (rows) per monkey
N_STRIP = 6                                  # sweep images per seed row (suppress -> drive)
N_ROW = 2                                    # seed rows per example (2 highest-R2 seeds)


def top_examples(mk, n=N_EX):
    ct = L.control_table()
    sub = ct[(ct.monkey == mk) & ct.control_r.notna()].sort_values('control_r', ascending=False)
    return [(mk, int(r.unit), r.model) for r in sub.head(n).itertuples()]
CMAP = plt.colormaps['RdBu_r']
FS_TITLE, FS_AX, FS_TICK, FS_ANN = 7, 6.5, 6, 6


def cloud_named(mk, u, m):
    """(name, floored pred, measured z, seed, level) over the personalized accentuated sweep."""
    P = L.load_predictions(mk)
    mi = list(P['pred_models']).index(m); ui = list(P['target_units']).index(u)
    sel = (P['gen_model'] == m) & (P['gen_unit'] == u)
    resp = L._acc_response(mk, u); fl = L.firing_floor(mk, u)
    out = []
    for n, p, s, lv in zip(P['stim'][sel], P['pred'][sel, mi, ui], P['gen_seed'][sel], P['gen_level'][sel]):
        if n in resp:
            out.append((str(n), max(float(p), fl), float(resp[n]), int(s), float(lv)))
    return out


def image_dir(mk, m):
    cands = [d for d in sorted(os.listdir(CTRL)) if mk in d and d.endswith(f'_{m}_accentuation')]
    return os.path.join(CTRL, cands[0]) if cands else None


def resolve_image(mk, m, name):
    """Resolve a control stimulus name to its file on the drive (basename match)."""
    d = image_dir(mk, m)
    if d is None:
        return None
    p = os.path.join(d, name)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(d, name)))
    return hits[0] if hits else None


def _seed_r2(lst):
    """Floored-prediction R2 for one seed sweep: 1 - SS_res / SS_tot (loader convention)."""
    p = np.array([q[1] for q in lst]); me = np.array([q[2] for q in lst])
    ss = np.sum((me - me.mean()) ** 2)
    return 1.0 - np.sum((me - p) ** 2) / ss if ss > 0 else -np.inf


def top_seed_sweeps(rows, n=N_ROW):
    """The n seeds with the highest per-seed R2; each as a level-sorted [(lv, name), ...]."""
    byseed = defaultdict(list)
    for name, p, me, s, lv in rows:
        byseed[s].append((lv, p, me, name))
    ranked = sorted(byseed.items(), key=lambda kv: _seed_r2(kv[1]), reverse=True)
    return [[(q[0], q[3]) for q in sorted(lst)] for _, lst in ranked[:n]]


def render(out_dir, num, PICKS):
    stem = output_name('high_control_examples', PICKS[0][0])
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, min(232, 42 * len(PICKS) + 22) * MM),
                     facecolor=FIG_FACECOLOR)
    outer = GridSpec(len(PICKS), 1, figure=fig, left=0.035, right=0.975,
                     top=0.905, bottom=0.055, hspace=0.42)

    key_numbers = {}
    missing = []
    for ri, (mk, u, m) in enumerate(PICKS):
        rows = cloud_named(mk, u, m)
        pf = np.array([r[1] for r in rows]); mf = np.array([r[2] for r in rows])
        reg = stats.linregress(pf, mf)
        r_pair = float(reg.rvalue); slope = float(reg.slope)
        col = get_model_color(m)
        short = MODEL_SHORT_NAMES[m]
        tag = f'{short} | Monkey {MONKEY_TAG[mk]} {MONKEY_REGION[mk]} ch{u}'
        key_numbers[f'{short}_{mk}_u{u}'] = {'control_r': round(r_pair, 2),
                                             'control_slope': round(slope, 2)}

        # split the row: 2x6 image grid (left) + scatter (right, spans both rows)
        gr = outer[ri].subgridspec(1, 2, width_ratios=[N_STRIP, 2.7], wspace=0.12)

        # ---- accentuation sweeps: 2 highest-R2 seeds, one per row (suppress -> drive) ----
        sweeps = top_seed_sweeps(rows)
        all_lv = [r[4] for r in rows]
        norm_lv = mcolors.Normalize(vmin=min(all_lv), vmax=max(all_lv))
        grid = gr[0].subgridspec(N_ROW, N_STRIP, wspace=0.06, hspace=0.06)
        for rj in range(N_ROW):
            seq = sweeps[rj] if rj < len(sweeps) else []
            picks = []
            if seq:
                idx = np.linspace(0, len(seq) - 1, N_STRIP).round().astype(int)
                picks = [seq[i] for i in idx]
            for ci in range(N_STRIP):
                axi = fig.add_subplot(grid[rj, ci])
                axi.set_xticks([]); axi.set_yticks([])
                if ci < len(picks):
                    lv, name = picks[ci]
                    fp = resolve_image(mk, m, name)
                    if fp is None:
                        missing.append(f'{mk}/{m}/{name}')
                        axi.set_facecolor('#eee')
                    else:
                        axi.imshow(np.asarray(Image.open(fp).convert('RGB')))
                    for s in axi.spines.values():
                        s.set_visible(True); s.set_linewidth(1.6); s.set_edgecolor(CMAP(norm_lv(lv)))
                else:
                    axi.set_facecolor('white')
                    for s in axi.spines.values():
                        s.set_visible(False)
                if ri == 0 and rj == 0 and ci == 0:
                    axi.set_title('suppress', fontsize=FS_TICK, fontfamily=FONT_FAMILY,
                                  color='#3B4CC0', pad=2, loc='left')
                if ri == 0 and rj == 0 and ci == N_STRIP - 1:
                    axi.set_title('drive', fontsize=FS_TICK, fontfamily=FONT_FAMILY,
                                  color='#B40426', pad=2, loc='right')

        # ---- predicted vs measured scatter ----
        ax = fig.add_subplot(gr[1])
        lo = float(min(pf.min(), mf.min())); hi = float(max(pf.max(), mf.max()))
        mg = (hi - lo) * 0.08; lims = (lo - mg, hi + mg)
        norm = mcolors.Normalize(vmin=pf.min(), vmax=pf.max())
        ax.plot(lims, lims, 'k--', lw=0.6, alpha=0.35, zorder=1)
        byseed = defaultdict(list)
        for n, p, me, s, lv in rows:
            byseed[s].append((lv, p, me))
        for s, lst in byseed.items():
            lst.sort()
            ps = np.array([q[1] for q in lst]); ms = np.array([q[2] for q in lst])
            ax.plot(ps, ms, ls=':', color='#777', lw=0.5, alpha=0.6, zorder=3)
        ax.scatter(pf, mf, s=9, c=[CMAP(norm(x)) for x in pf],
                   edgecolors='black', linewidths=0.2, zorder=5)
        xf = np.linspace(pf.min(), pf.max(), 50)
        ax.plot(xf, reg.slope * xf + reg.intercept, color=col, lw=2.0, zorder=6)
        # data trend runs lower-left -> upper-right, so the upper-left corner is empty
        ax.text(0.04, 0.96, f'$r$ = {r_pair:.2f}\nslope = {slope:.2f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANN,
                fontfamily=FONT_FAMILY, zorder=14,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#cccccc', lw=0.5, alpha=0.9))
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
        ax.set_xlabel('Predicted response (z)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
        ax.set_ylabel('Measured neural (z)', fontsize=FS_AX, fontfamily=FONT_FAMILY)

        # row title spanning the block
        y0 = outer[ri].get_position(fig).y1
        fig.text(0.035, y0 + 0.010, f'{chr(97 + ri)}', fontsize=10, fontweight='bold',
                 ha='left', va='bottom', fontfamily=FONT_FAMILY)
        fig.text(0.072, y0 + 0.012, tag, fontsize=FS_TITLE, fontweight='bold',
                 ha='left', va='bottom', color=col, fontfamily=FONT_FAMILY)

    # feature-level colorbar (top-right)
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.70, 0.955, 0.26, 0.011])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[])
    cb.set_label('feature level: suppress to drive', fontsize=FS_TICK, fontfamily=FONT_FAMILY)
    cb.ax.xaxis.set_label_position('top'); cb.outline.set_linewidth(0.5)

    mk0 = PICKS[0][0]
    fig.text(0.035, 0.965, f"{_M['title']} - Monkey "
             f"{MONKEY_TAG[mk0]} ({MONKEY_REGION[mk0]})",
             fontsize=9, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, stem))
    plt.close(fig)
    print(f'{mk0:8s} S{num}  saved -> {out}')
    print('  key_numbers:', key_numbers)
    if missing:
        print('  MISSING IMAGES:', missing)
    return out, key_numbers, missing


def main(out_dir, monkey=None):
    """Render one per-monkey gallery (monkey='red'...) or, with monkey=None, all five.
    Returns the PNG path (one monkey) or the list of PNG paths (all)."""
    outs = []
    for mk, num in MONKEY_NUM:
        if monkey is not None and mk != monkey:
            continue
        outs.append(render(out_dir, num, top_examples(mk))[0])
    return outs[0] if monkey is not None else outs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    ap.add_argument('--monkey', default=None, choices=MONKEYS,
                    help='render only this monkey (default: all five)')
    args = ap.parse_args()
    main(**vars(args))
