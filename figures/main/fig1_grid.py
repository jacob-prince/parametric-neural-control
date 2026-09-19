#!/usr/bin/env python3
"""Figure 1 panel: 7×3 grid of accentuated stimuli (RN50-Robust, paul unit 8).

Seven feature levels (rows, DRIVE at the top → suppress at the bottom) × three
seeds (columns), thumbnails upright. Borders use the fig3 deepdive scheme:
RdBu_r mapped from each cell's ACHIEVED response score (Normalize over the min/max
achieved score across all shown cells), with the seed row in light yellow
(un-accentuated marker) and the fig3 saturation/brightness boost on thumbnails.
No text or axes - designed to be embedded tall on the left of the fig 1 schematic.

Render-time ingredient of fig1_framework: main(out_path) writes the grid PNG to the
given path (fig1_framework uses <out_dir>/_fig1_accentuation_grid.png).
"""

import os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image, ImageEnhance
import matplotlib.patheffects as pe

from pnc import paths
from pnc.utils import apply_figure_style, DATA_ROOT, light_font_properties  # noqa: F401

apply_figure_style()

STIM_DIR = Path(DATA_ROOT) / 'stimuli_control' / '03-05-2025_paul_20250428-20250430_resnet50_robust_accentuation'
SEED_DIR = Path(DATA_ROOT) / 'stimuli_encoding'

UNIT = 8
SEED_IMG_IDS = [3, 4, 7]                 # three different seeds (3 = cat: no human face -> no redaction needed)
N_COLS = 7                               # 3 suppressed + seed + 3 driven

# Per-seed hand-picked level indices (of 11 sorted ascending). Pick three
# clearly-suppressed indices for the left and three clearly-driven indices
# for the right so the perceptual sweep is monotonic on each side. Edit as
# needed after eyeballing the output.
PER_SEED_LEVELS = {
    # sid: (suppress_idx_left_to_right, drive_idx_left_to_right)
    3: ([0, 1, 3], [6, 7, 9]),
    4: ([0, 1, 2], [4, 6, 8]),
    7: ([0, 1, 2], [4, 7, 9]),
}
LEVEL_FILE_RE = re.compile(
    r'^resnet50_robust_RidgeCV_unit_(\d+)_img_(\d+)_level_'
    r'(-?\d+\.\d+)_score_(-?\d+\.\d+)\.png$')

# Map img_id → seed filename (from accentuation_configs/.../*Ch8*.yaml)
SEED_FILENAMES = {
    0: 'shared0575_nsd43157.png',
    1: 'shared0850_nsd61798.png',
    2: 'shared0968_nsd70194.png',
    3: 'shared0241_nsd20065.png',
    4: 'shared0160_nsd13231.png',
    5: 'shared0070_nsd07008.png',
    6: 'shared0055_nsd05879.png',
    7: 'shared0668_nsd48623.png',
    8: 'shared0488_nsd36979.png',
    9: 'shared0940_nsd68312.png',
}
# seed 4 (dog) uses the canonical cropped asset shared by fig2/fig5/gradient_gallery,
# rather than resolving shared0160 directly from the raw stimulus directory
SEED_IMG_OVERRIDE = {4: paths.ASSETS / 'seed.png'}

CMAP = cm.get_cmap('RdBu_r')
BORDER_LW = 4.5     # in points; rendered via spines
FIGSIZE = (3.45, 7.0)   # extra left margin for the firing-direction annotation
DPI = 300


def list_levels_for(unit, img):
    rows = []
    for fn in sorted(os.listdir(STIM_DIR)):
        m = LEVEL_FILE_RE.match(fn)
        if not m:
            continue
        u, i, lvl, sc = m.groups()
        if int(u) == unit and int(i) == img:
            rows.append((float(lvl), float(sc), STIM_DIR / fn))
    rows.sort(key=lambda r: r[0])
    return rows


def match_brightness_contrast(src, ref):
    """Linearly remap `src` so its mean and std equal `ref`'s (per channel)."""
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)
    out = np.zeros_like(src)
    for c in range(3):
        s = src[..., c]
        r = ref[..., c]
        s_std = s.std() if s.std() > 1e-3 else 1.0
        out[..., c] = (s - s.mean()) / s_std * r.std() + r.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def main(out_path):
    """Render the 7x3 accentuation grid to `out_path` (PNG) and return that path."""
    out_path = str(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    grid = []
    seed_brightness_refs = {}   # sid → reference accentuated array (level closest to 0)
    for sid in SEED_IMG_IDS:
        all_levels = list_levels_for(UNIT, sid)
        if len(all_levels) != 11:
            print(f'  WARN: unit={UNIT} img={sid} has {len(all_levels)} levels')
        suppress_idx, drive_idx = PER_SEED_LEVELS[sid]
        suppress = [all_levels[i] for i in suppress_idx]
        drive = [all_levels[i] for i in drive_idx]
        seed_path = SEED_IMG_OVERRIDE.get(sid, SEED_DIR / SEED_FILENAMES[sid])
        if not seed_path.exists():
            raise FileNotFoundError(seed_path)
        # Reference for brightness match: accentuated level closest to 0
        ref_idx = int(np.argmin([abs(lvl) for lvl, _sc, _ in all_levels]))
        seed_brightness_refs[sid] = np.asarray(
            Image.open(all_levels[ref_idx][2]).convert('RGB'))
        row = suppress + [(0.0, None, seed_path)] + drive     # score=None marks the seed
        grid.append(row)
        print(f'  seed img={sid} ({SEED_FILENAMES[sid]}): '
              f'suppress idx {suppress_idx} → '
              f'{[f"{l:+.2f}" for l,_s,_ in suppress]} | '
              f'drive idx {drive_idx} → {[f"{l:+.2f}" for l,_s,_ in drive]}')

    fig = plt.figure(figsize=FIGSIZE, facecolor='white')
    # transposed grid: rows = the 7 levels (drive at top), cols = the 3 seeds
    gs = GridSpec(N_COLS, len(SEED_IMG_IDS),
                  left=0.135, right=0.995, top=0.99, bottom=0.01,
                  wspace=0.06, hspace=0.06)

    # Border colors: fig3 deepdive scheme - RdBu_r mapped from each cell's ACHIEVED
    # score, normalized over the min/max achieved score across all shown cells; the
    # seed cell (score=None) gets the fig3 light-yellow un-accentuated marker.
    all_scores = np.array([sc for row in grid for _l, sc, _p in row if sc is not None])
    norm = mcolors.Normalize(vmin=float(all_scores.min()), vmax=float(all_scores.max()))
    SEED_RGB = (255 / 255, 236 / 255, 150 / 255)
    # one border color per LEVEL row, anchored to column 1's achieved scores so all
    # three columns share identical row colors (user-directed; per-cell coloring made
    # the columns inconsistent because the seeds reach different scores)
    ROW_BORDER = [SEED_RGB if sc is None else CMAP(norm(sc)) for _l, sc, _p in grid[0]]

    def _boost(img_arr):
        im = Image.fromarray(img_arr)
        im = ImageEnhance.Color(im).enhance(1.15)
        im = ImageEnhance.Brightness(im).enhance(1.05)
        return np.asarray(im)

    for r, row in enumerate(grid):          # r indexes the SEED (becomes the column)
        sid = SEED_IMG_IDS[r]
        for c, (_lvl, score, path) in enumerate(row):   # c indexes the LEVEL
            grow = N_COLS - 1 - c                # drive (c=6) -> top row
            ax = fig.add_subplot(gs[grow, r])
            img = np.asarray(Image.open(path).convert('RGB'))
            if c == N_COLS // 2:   # middle level: match seed to neighbours
                img = match_brightness_contrast(img, seed_brightness_refs[sid])
            img = _boost(img)
            border = ROW_BORDER[c]
            ax.imshow(img, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            for side in ('top', 'bottom', 'left', 'right'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(border)
                ax.spines[side].set_linewidth(BORDER_LW)

    # fig3-style firing-direction annotation, rotated to hug the left edge:
    # "Seed image" beside the (yellow) seed row, arrows outward to
    # "prediction: increased firing" (top) / "prediction: decreased firing" (bottom).
    AX_X = 0.075
    # true HN Light face (extracted from the system .ttc; mpl only registers Regular),
    # in the same gray as the schematic's model-name lists
    HN_LIGHT = light_font_properties(size=14)
    SIDE_KW = dict(rotation=90, ha='center', fontproperties=HN_LIGHT, color='#565656',
                   path_effects=[pe.withStroke(linewidth=0.35, foreground='#565656')])
    seed_ax = fig.axes[[grow for grow in range(len(fig.axes))][0]]  # placeholder
    # locate the seed row's centre from the actual axes bbox (col 0, grid row 3)
    seed_y = None
    for ax in fig.axes:
        bb = ax.get_position()
        # the seed row is the vertical middle row; match by centre closest to 0.5
        if seed_y is None or abs((bb.y0 + bb.y1) / 2 - 0.5) < abs(seed_y - 0.5):
            seed_y = (bb.y0 + bb.y1) / 2
    fig.text(AX_X, seed_y, 'Seed image', va='center', **SIDE_KW)
    # continuous fig3-style chain, everything inside [0.01, 0.99] so the tight crop
    # cannot shift the annotation relative to the grid
    for y0, y1 in ((seed_y + 0.095, seed_y + 0.145), (seed_y - 0.095, seed_y - 0.145)):
        fig.add_artist(mpatches.FancyArrowPatch(
            (AX_X, y0), (AX_X, y1), transform=fig.transFigure,
            arrowstyle='->', mutation_scale=14, color='black', lw=1.8))
    fig.text(AX_X, seed_y + 0.158, 'prediction: increased firing', va='bottom',
             **SIDE_KW)
    fig.text(AX_X, seed_y - 0.158, 'prediction: decreased firing', va='top',
             **SIDE_KW)

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor='white')
    plt.close(fig)
    print(f'Saved → {out_path}')
    return out_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.output_dir() / 'figures' / '_fig1_accentuation_grid.png'),
                    help='output PNG path')
    args = ap.parse_args()
    main(args.out)
