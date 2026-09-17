#!/usr/bin/env python3
"""Supp. Figs. — accentuation-sweep galleries (one per monkey; a 5-figure span).

For each monkey, shows how all ten DNN models accentuate three of its recorded sites, each site
paired with a different seed image. Within a monkey panel: three (site, seed) groups sit side by
side; within a group, rows are the ten models and columns are four target levels along each
model's accentuation axis (strong suppress, weak suppress, weak drive, strong drive). A seed
thumbnail heads each group. The columns walk the same suppress->drive sweep for every model, so
the grid exposes how differently each model reshapes a given image to move a site's predicted
response.

The accentuation-level ordering per (model, site, seed) comes from
preproc.loader.load_predictions(mk); pixels come straight from the raw accentuated PNGs under
source_data/stimuli_control/ and the seed thumbnails from source_data/stimuli_encoding/
(irreducible raw experiment stimuli). Numbering + title + output stem are manifest-driven
(slug 'sweep_gallery'; the five monkeys occupy the reserved span: output_name('sweep_gallery', mk)).

    python -m figures.supplementary.sup_sweep_gallery --out DIR [--monkey red]   # default: all five
"""
import os, glob, re

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

from PIL import ImageEnhance

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_ORDER, MODEL_SHORT_NAMES, get_model_color,
                       STIMULI_CONTROL_PATH, STIMULI_PATH)
apply_figure_style()
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'   # spec display name
_M = _fig('sweep_gallery')

STIM_CTRL = STIMULI_CONTROL_PATH
STIM_ENC = STIMULI_PATH
SEED_DOG = str(paths.ASSETS / 'seed.png')                                    # middle-column seed
SAT = 1.15                                                                   # +15% saturation

# ── configuration ────────────────────────────────────────────────────────────
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}
TAG = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
# accentuated-stimulus directory prefix per monkey: (date, day-range)
DIRPREF = {'red': ('02-05-2025', '20250428-20250430'),
           'paul': ('03-05-2025', '20250428-20250430'),
           'venus': ('05-05-2025', '250426-250429'),
           'leap': ('06-05-2025', '250426-250501'),
           'three0': ('06-05-2025', '250426-250501')}
# three (site, seed) groups per monkey — one seed per site (face / animal / scene bases)
GROUPS_BY_MONKEY = {
    'red':    [(9, 3), (0, 4), (19, 7)],
    'paul':   [(8, 3), (24, 4), (40, 7)],
    'venus':  [(9, 3), (151, 4), (331, 7)],
    'leap':   [(81, 3), (286, 4), (342, 7)],
    'three0': [(56, 3), (120, 4), (204, 7)],
}
# seed index -> encoding base image (canonical order; index == seed number)
SEED_IMAGES = ['shared0575_nsd43157.png', 'shared0850_nsd61798.png', 'shared0968_nsd70194.png',
               'shared0241_nsd20065.png', 'shared0160_nsd13231.png', 'shared0070_nsd07008.png',
               'shared0055_nsd05879.png', 'shared0668_nsd48623.png', 'shared0488_nsd36979.png',
               'shared0940_nsd68312.png']

LEVELS = [0, 1, 6, 10]                       # of 11 ordinal levels along the axis
COL_LABELS = ['strong\nsuppress', 'weak\nsuppress', 'weak\ndrive', 'strong\ndrive']
LEVEL_CMAP = plt.get_cmap('RdBu_r')          # paper convention: blue low -> red high
COL_COLORS = [LEVEL_CMAP(v) for v in (0.06, 0.34, 0.66, 0.94)]
THUMB = 224                                  # px per rendered cell (downsample from 1024)

# font sizes authored for the ~7.1 in (180 mm) canvas -> ~5-7 pt at print size
FS_TITLE, FS_ROW, FS_COL, FS_HEAD = 7.5, 7.0, 5.0, 6.0
MISSING = []


def acc_dir(mk, model):
    dp, dr = DIRPREF[mk]
    return os.path.join(STIM_CTRL, f'{dp}_{mk}_{dr}_{model}_accentuation')


def _level_score(f):
    m = re.search(r'_level_(-?[0-9.]+)_score_(-?[0-9.]+)\.png', os.path.basename(f))
    return float(m.group(1)), float(m.group(2))


def _level(f):
    return _level_score(f)[0]


def sweep_files(mk, model, unit, seed):
    """The 11 accentuated PNGs for (model, unit, seed), sorted suppress -> drive.

    A few site/models were re-synthesised, leaving duplicate files at the same target level that
    differ only in the achieved score suffix. Dedupe per target level, keeping the synthesis whose
    achieved score is closest to its target."""
    d = acc_dir(mk, model)
    fs = sorted(glob.glob(os.path.join(d, f'{model}_*unit_{unit}_img_{seed}_level_*.png')))
    best = {}
    for f in fs:
        lv, sc = _level_score(f)
        key = round(lv, 4)
        if key not in best or abs(sc - lv) < abs(best[key][1] - key):
            best[key] = (f, sc)
    return [best[k][0] for k in sorted(best)]


def _saturate(im):
    return ImageEnhance.Color(im).enhance(SAT)


def load_thumb(fp):
    im = Image.open(fp).convert('RGB')
    if im.size != (THUMB, THUMB):
        im = im.resize((THUMB, THUMB), Image.LANCZOS)
    return np.asarray(_saturate(im))


def seed_thumb(seed, gi):
    fp = SEED_DOG if gi == 1 else os.path.join(STIM_ENC, SEED_IMAGES[seed])
    if not os.path.exists(fp):
        MISSING.append(fp); return np.full((THUMB, THUMB, 3), 240, np.uint8)
    im = Image.open(fp).convert('RGB').resize((THUMB, THUMB), Image.LANCZOS)
    return np.asarray(_saturate(im))


def build_gallery(mk, out_dir):
    """Render one monkey's sweep-gallery figure; return (out_png, span_ok)."""
    groups = GROUPS_BY_MONKEY[mk]
    models = MODEL_ORDER
    nM = len(models)

    # pre-load thumbs: thumbs[gi][mi][ci]
    thumbs = []
    span_ok = True
    for (unit, seed) in groups:
        gt = []
        for m in models:
            fs = sweep_files(mk, m, unit, seed)
            if len(fs) != 11:
                MISSING.append(f'{mk} {m} unit{unit} seed{seed}: {len(fs)}/11 levels')
                span_ok = False
                gt.append([np.full((THUMB, THUMB, 3), 240, np.uint8) for _ in LEVELS])
                continue
            # confirm the sweep spans suppress (level<0) -> drive (level>0)
            if _level(fs[0]) >= 0 or _level(fs[-1]) <= 0:
                span_ok = False
            gt.append([load_thumb(fs[c]) for c in LEVELS])
        thumbs.append(gt)

    figW = WIDTH_2COL_MM * MM
    figH = 208 * MM
    fig = plt.figure(figsize=(figW, figH), facecolor=FIG_FACECOLOR)
    outer = GridSpec(1, 1, figure=fig, left=0.12, right=0.995, top=0.808, bottom=0.006)
    parent = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.075)

    for gi, (unit, seed) in enumerate(groups):
        sub = GridSpecFromSubplotSpec(nM, 4, subplot_spec=parent[gi],
                                      wspace=0.035, hspace=0.035)
        for ri, m in enumerate(models):
            for ci in range(4):
                ax = fig.add_subplot(sub[ri, ci])
                ax.imshow(thumbs[gi][ri][ci]); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():          # red->blue level border
                    sp.set_visible(True); sp.set_edgecolor(COL_COLORS[ci]); sp.set_linewidth(0.8)
                if ri == 0:
                    ax.set_title(COL_LABELS[ci], fontsize=FS_COL, color=COL_COLORS[ci],
                                 fontweight='bold', fontfamily=FONT_FAMILY, pad=2.5,
                                 linespacing=0.9)
                if gi == 0 and ci == 0:
                    ax.set_ylabel(MODEL_SHORT_NAMES.get(m, m), fontsize=FS_ROW,
                                  rotation=0, ha='right', va='center', labelpad=5,
                                  color=get_model_color(m),
                                  fontweight='bold')

        # group header: seed thumbnail centred above the 4 columns, site label below it
        pos = parent[gi].get_position(fig)
        cx = 0.5 * (pos.x0 + pos.x1)
        th_w = 0.098                               # bigger, more-visible seed thumbnail
        th_h = th_w * figW / figH
        y_th = pos.y1 + 0.030
        axs = fig.add_axes([cx - th_w / 2, y_th, th_w, th_h])
        axs.imshow(seed_thumb(seed, gi)); axs.set_xticks([]); axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_edgecolor('#333333'); sp.set_linewidth(0.8)
        fig.text(cx, y_th - 0.007, f'{REGION[mk]} unit {unit}',
                 fontsize=FS_HEAD, fontweight='bold', ha='center', va='top',
                 fontfamily=FONT_FAMILY)

    # left-hand row label marking the top row of thumbnails as the seed images
    y_seed_mid = pos.y1 + 0.030 + 0.5 * th_h
    fig.text(0.075, y_seed_mid, 'seed\nimages', fontsize=FS_HEAD, fontweight='bold',
             ha='center', va='center', color='#333333', fontfamily=FONT_FAMILY,
             linespacing=0.95)

    fig.text(0.12, 0.960, f"{_M['title']}: Monkey {TAG[mk]} ({REGION[mk]})",
             fontsize=FS_TITLE, fontweight='bold', ha='left', va='center',
             fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, output_name('sweep_gallery', mk)))
    plt.close(fig)
    return out, span_ok


def main(out_dir, monkey=None):
    """Render the gallery for `monkey` (default None = all five). Returns the PNG path for a
    single monkey, or the list of PNG paths when rendering all five."""
    all_ok = True
    outs = []
    for mk in (MONKEYS if monkey is None else [monkey]):
        out, ok = build_gallery(mk, out_dir)
        all_ok &= ok
        outs.append(out)
        print(f'{mk:8s} span_ok={ok}  saved -> {out}')
    if MISSING:
        print('MISSING/ISSUES:'); [print('  ', x) for x in MISSING]
    print('ALL sweeps span suppress->drive:', all_ok)
    return outs[0] if monkey is not None else outs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    ap.add_argument('--monkey', default=None, choices=MONKEYS,
                    help='render one monkey only (default: all five)')
    args = ap.parse_args()
    main(**vars(args))
