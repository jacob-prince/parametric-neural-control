#!/usr/bin/env python3
r"""Supp. Fig. - suppress / drive extremes across every recorded site.

The strongest-suppress (level 0) and strongest-drive (level 10) accentuated stimulus for each of
the 25 channels x 4 representative models (ResNet50, DINOv2, RN50-Robust, CLIPAG). The 25 sites are
split across two side-by-side blocks (left = red/paul/venus[:3], right = leap/three0/venus[3:]) so
the 2-column page width is filled with large image cells. Within each block, columns = 4 models,
each split into a suppress (S) and drive (D) sub-column; borders are colored blue (suppress) ->
red (drive) per the paper's predicted-level convention (robust model names shown bold).
The seed image rotates across (channel, model) pairs so the grid samples the full seed set while
comparing models at each site's extremes. Self-contained: reads only the accentuated PNGs under
source_data/stimuli_control + pnc.preproc.loader for the site list. Number stamped from the manifest.
"""
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_SHORT_NAMES, ROBUST_MODELS, MONKEY_COLORS, get_model_color,
                       STIMULI_CONTROL_PATH)
apply_figure_style()

_M = _fig('sweep_extremes'); STEM = output_name('sweep_extremes')
STIM_CTRL = STIMULI_CONTROL_PATH

TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
DIRPREF = {'red': ('02-05-2025', '20250428-20250430'), 'paul': ('03-05-2025', '20250428-20250430'),
           'venus': ('05-05-2025', '250426-250429'), 'leap': ('06-05-2025', '250426-250501'),
           'three0': ('06-05-2025', '250426-250501')}
MODELS = ['resnet50', 'dinov2_vitb14_reg', 'resnet50_robust', 'clipag_vitb32']
SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'
THUMB = 224
# paper convention: border color = predicted response level (blue low -> red high)
LEVEL_CMAP = plt.get_cmap('RdBu_r')
C_SUPP, C_DRIVE = LEVEL_CMAP(0.0), LEVEL_CMAP(1.0)
FS_TITLE, FS_MODEL, FS_ROW, FS_SD = 8.0, 6.5, 6.0, 5.5
FS_MONKEY = 9.5   # rotated monkey/region block labels - larger than the per-site channel labels
BLANK = np.full((THUMB, THUMB, 3), 240, np.uint8)
MISSING = []


def acc_dir(mk, model):
    dp, dr = DIRPREF[mk]
    return os.path.join(STIM_CTRL, f'{dp}_{mk}_{dr}_{model}_accentuation')


def extreme_files(mk, model, unit, seed):
    """(suppress, drive) = level-0 and level-10 PNGs for (model, unit, seed), or (None, None)."""
    fs = sorted(glob.glob(os.path.join(acc_dir(mk, model), f'{model}_*unit_{unit}_img_{seed}_level_*.png')))
    best = {}
    for f in fs:
        m = re.search(r'_level_(-?[0-9.]+)_score_(-?[0-9.]+)\.png', os.path.basename(f))
        if not m:
            continue
        lv, sc = float(m.group(1)), float(m.group(2))
        best.setdefault(round(lv, 4), []).append((abs(sc - lv), f))
    if not best:
        return None, None
    keys = sorted(best)
    lo = min(best[keys[0]])[1]; hi = min(best[keys[-1]])[1]
    return lo, hi


def thumb(fp):
    if fp is None:
        return BLANK
    return np.asarray(Image.open(fp).convert('RGB').resize((THUMB, THUMB), Image.LANCZOS))


def main(out_dir):
    ch = L.channels_df()
    UNITS = {mk: ch[ch.monkey == mk].sort_values('unit').unit.astype(int).tolist()
             for mk in ['red', 'paul', 'venus', 'leap', 'three0']}
    nM = len(MODELS)
    # balance the two page halves: venus split 3 (left) / 2 (right) -> one empty slot only
    SIDE_BLOCKS = [
        [('red', UNITS['red']), ('paul', UNITS['paul']), ('venus', UNITS['venus'][:3])],
        [('leap', UNITS['leap']), ('three0', UNITS['three0']), ('venus', UNITS['venus'][3:])],
    ]
    BLOCK_ROWS = [5, 5, 3]                               # row slots per monkey block

    # shared row grid: monkey-blocks + inter-block gaps (right side leaves 2 empty slots)
    ROWSP, COLSP, MIDGAP = 0.20, 0.16, 1.35
    row_ratios, rowslot = [], {}
    r = 0
    for b, nr in enumerate(BLOCK_ROWS):
        for ci in range(nr):
            rowslot[(b, ci)] = r; row_ratios.append(1.0); r += 1
        if b < len(BLOCK_ROWS) - 1:
            row_ratios.append(ROWSP); r += 1
    # columns: [left 8 img + 3 gaps] [mid gap] [right 8 img + 3 gaps]
    col_ratios, lcol, rcol = [], {}, {}
    c = 0
    for mj in range(nM):
        for k in range(2):
            lcol[(mj, k)] = c; col_ratios.append(1.0); c += 1
        if mj < nM - 1:
            col_ratios.append(COLSP); c += 1
    col_ratios.append(MIDGAP); c += 1
    for mj in range(nM):
        for k in range(2):
            rcol[(mj, k)] = c; col_ratios.append(1.0); c += 1
        if mj < nM - 1:
            col_ratios.append(COLSP); c += 1
    side_col = [lcol, rcol]

    figW = WIDTH_2COL_MM * MM
    figH = figW * 0.79                                   # tall two-block gallery, within budget
    fig = plt.figure(figsize=(figW, figH), facecolor=FIG_FACECOLOR)
    gs = GridSpec(len(row_ratios), len(col_ratios), figure=fig,
                  left=0.055, right=0.995, top=0.905, bottom=0.008,
                  wspace=0.03, hspace=0.03, height_ratios=row_ratios, width_ratios=col_ratios)

    seed_ctr = 0
    for sd, blocks in enumerate(SIDE_BLOCKS):
        cmap_col = side_col[sd]
        for b, (mk, chans) in enumerate(blocks):
            for ci, u in enumerate(chans):
                grow = rowslot[(b, ci)]
                for mj, model in enumerate(MODELS):
                    seed = seed_ctr % 10; seed_ctr += 1
                    lo, hi = extreme_files(mk, model, u, seed)
                    if lo is None:
                        MISSING.append(f'{mk} u{u} {model}')
                    imgs = (thumb(lo), thumb(hi))
                    for k in range(2):
                        col = C_SUPP if k == 0 else C_DRIVE
                        ax = fig.add_subplot(gs[grow, cmap_col[(mj, k)]])
                        ax.imshow(imgs[k], aspect='auto')
                        ax.set_xticks([]); ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(1.4)
                        if b == 0 and ci == 0:          # S / D labels on the top row of each side
                            ax.text(0.5, 1.10, 'S' if k == 0 else 'D', transform=ax.transAxes,
                                    ha='center', va='bottom', fontsize=FS_SD, fontweight='bold',
                                    color=col, fontfamily=FONT_FAMILY)
                    if mj == 0:                         # channel label left of each row
                        p0 = gs[grow, cmap_col[(0, 0)]].get_position(fig)
                        fig.text(p0.x0 - 0.008, 0.5 * (p0.y0 + p0.y1), f'{TAG[mk]}{u}',
                                 ha='right', va='center', fontsize=FS_ROW, fontweight='bold',
                                 color=MONKEY_COLORS[mk], fontfamily=FONT_FAMILY)
            # monkey region label rotated left of this block (spans its filled channels)
            pt = gs[rowslot[(b, 0)], cmap_col[(0, 0)]].get_position(fig)
            pb = gs[rowslot[(b, len(chans) - 1)], cmap_col[(0, 0)]].get_position(fig)
            xlab = 0.015 if sd == 0 else pt.x0 - 0.052
            fig.text(xlab, 0.5 * (pt.y1 + pb.y0), f'{TAG[mk]} - {REGION[mk]}',
                     rotation=90, ha='center', va='center', fontsize=FS_MONKEY,
                     fontweight='bold', color=MONKEY_COLORS[mk], fontfamily=FONT_FAMILY)
        # model names above each side's top row (spanning the model's 2 sub-columns)
        for mj, model in enumerate(MODELS):
            p0 = gs[0, cmap_col[(mj, 0)]].get_position(fig)
            p1 = gs[0, cmap_col[(mj, 1)]].get_position(fig)
            fig.text(0.5 * (p0.x0 + p1.x1), p0.y1 + 0.028, SHORT.get(model, model),
                     ha='center', va='bottom', fontsize=FS_MODEL, color=get_model_color(model),
                     fontweight='bold' if model in ROBUST_MODELS else 'normal',
                     fontfamily=FONT_FAMILY)

    fig.text(0.525, 0.965, _M['title'], ha='center', va='center', fontsize=FS_TITLE,
             fontweight='bold', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('saved ->', out)
    if MISSING:
        print('MISSING:', len(MISSING)); [print('  ', x) for x in MISSING[:20]]
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
