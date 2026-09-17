#!/usr/bin/env python3
"""Supp fig (calibration_mosaic) -- Calibration image set.

Full-page mosaic of the 969-image encoding/calibration stimulus set, grouped by its three
image sources: natural scenes (NSD, 649), segmented objects and animals on white backgrounds
(259), and functional-localizer images (fLoc, 61). 649 + 259 + 61 = 969.

Within each source, images are sorted by mean Monkey-R aIT firing rate (high -> low) and
each thumbnail carries a red-blue (RdBu_r) firing-rate border with a matching colorbar (spk/s).

Data: image membership + per-image source flags from loader.load_stimuli()['calibration'];
per-image firing rate (spk/s) recovered from load_brain('red') (resp_z inverted with the
cached per-unit mu/sigma, averaged over the 5 aIT sites); thumbnails read from
DATA_ROOT/stimuli_encoding. S-number and output stem come from the manifest.
"""
import sys, os, glob
from math import ceil

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, DATA_ROOT)
apply_figure_style()
_M = _fig('calibration_mosaic'); STEM = output_name('calibration_mosaic')

STIM_DIR = os.path.join(DATA_ROOT, 'stimuli_encoding')

THUMB = 96          # downsampled thumbnail edge, px
BORDER = 5          # colored firing-rate border thickness, px
OUTLINE = 1         # thin dark frame around each cell, px
GAP = 3             # white gap between cells, px
COLS = 30           # thumbnails per row
OUTLINE_RGB = (70, 70, 70)
BASE_CMAP = plt.get_cmap('RdBu_r')   # blue = low, white = mid, red = high firing rate

# The three calibration sources: flag, header label, count, block colour.
SOURCES = [
    ('is_nsd', 'NSD natural scenes', 649, '#3274A1'),
    ('is_OO',  'Segmented objects & animals', 259, '#E1812C'),
    ('is_floc', 'fLoc functional localizer', 61, '#7B2D8E'),
]


def stretched_cmap(base, white_frac, n=256):
    """Remap `base` (white at 0.5) so its white point lands at `white_frac`
    along a linear [0, 1] axis. Lets the colorbar keep even ticks 0..max while
    the color midpoint sits at the median firing rate."""
    xs = np.linspace(0, 1, n)
    s = np.where(xs <= white_frac,
                 0.5 * xs / white_frac,
                 0.5 + 0.5 * (xs - white_frac) / (1 - white_frac))
    return LinearSegmentedColormap.from_list('stretched', list(zip(xs, base(s))))


def firing_rates():
    """{stimulus_name: mean firing rate (spk/s) across the 5 Monkey-R aIT electrodes}.
    resp_z is per-channel standardized; invert with the cached per-unit mu/sigma to
    spk/s, then average across the five sites."""
    b = L.load_brain('red')
    cal = b['calibration']
    spks = np.asarray(cal['resp_z'], float) * np.asarray(b['sigma'], float) \
        + np.asarray(b['mu'], float)
    act = spks.mean(1)
    return dict(zip(np.asarray(cal['stim']).astype(str), act))


def resolve_paths(act_by_name):
    """Return {source_flag: [(abs image path, spk/s), ...]} using load_stimuli()
    membership, resolving each image_fp basename against data/stimuli_encoding and
    tagging each image with its mean firing rate (sorted high -> low within source)."""
    cal = L.load_stimuli()['calibration']
    df = pd.DataFrame({k: np.asarray(cal[k]) for k in cal}).drop_duplicates(subset='stimulus_name')
    assert len(df) == 969, f'expected 969 unique calibration images, got {len(df)}'
    disk = {os.path.basename(p): p
            for p in sorted(glob.glob(os.path.join(STIM_DIR, '*'))) if os.path.isfile(p)}
    out, missing = {}, []
    for flag, _, _, _ in SOURCES:
        sub = df[df[flag].astype(bool)]
        items = []
        for name, fp in zip(sub['stimulus_name'], sub['image_fp']):
            bn = os.path.basename(fp)
            if bn in disk:
                items.append((disk[bn], float(act_by_name[str(name)])))
            else:
                missing.append(bn)
        items.sort(key=lambda t: -t[1])          # high firing rate first
        out[flag] = items
    return out, missing


def load_thumb(path):
    im = Image.open(path).convert('RGB')
    im.thumbnail((THUMB, THUMB), Image.LANCZOS)
    a = np.asarray(im)
    # pad to square (white) so the grid is regular even for non-square sources
    h, w = a.shape[:2]
    if (h, w) != (THUMB, THUMB):
        canvas = np.full((THUMB, THUMB, 3), 255, np.uint8)
        y0, x0 = (THUMB - h) // 2, (THUMB - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = a
        a = canvas
    return a


def compose(items, norm, cmap):
    """Row-major mosaic of thumbnails, each wrapped in a firing-rate colored border
    (RdBu_r: blue = low, red = high spk/s) inside a thin dark frame, on white."""
    n = len(items)
    rows = ceil(n / COLS)
    step = THUMB + 2 * BORDER + 2 * OUTLINE + GAP
    canvas = np.full((rows * step + GAP, COLS * step + GAP, 3), 255, np.uint8)
    inner = THUMB + 2 * BORDER
    outer = inner + 2 * OUTLINE
    for k, (p, act) in enumerate(items):
        r, c = divmod(k, COLS)
        Y, X = GAP + r * step, GAP + c * step
        canvas[Y:Y + outer, X:X + outer] = OUTLINE_RGB                # dark frame
        col = (np.asarray(cmap(norm(act))[:3]) * 255).astype(np.uint8)
        yb, xb = Y + OUTLINE, X + OUTLINE
        canvas[yb:yb + inner, xb:xb + inner] = col                    # firing-rate border
        yt, xt = yb + BORDER, xb + BORDER
        canvas[yt:yt + THUMB, xt:xt + THUMB] = load_thumb(p)          # thumbnail
    return canvas, rows


def main(out_dir):
    act_by_name = firing_rates()
    by_src, missing = resolve_paths(act_by_name)
    if missing:
        print('MISSING (first 20):', missing[:20])
        sys.exit(2)

    # linear firing-rate axis 0 -> vmax (even ticks); colormap stretched so its
    # white point sits at the median spk/s.
    act = np.array(list(act_by_name.values()))
    vmax = float(np.ceil(np.percentile(act, 98) / 10.0) * 10.0)   # 40
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = stretched_cmap(BASE_CMAP, float(np.median(act)) / vmax)

    blocks = []
    for flag, label, count, color in SOURCES:
        items = by_src[flag]
        assert len(items) == count, f'{flag}: expected {count}, got {len(items)}'
        arr, rows = compose(items, norm, cmap)
        blocks.append((label, count, color, arr, rows))
    total = sum(b[1] for b in blocks)
    assert total == 969, total

    fig_w_mm = WIDTH_2COL_MM
    fig = plt.figure(figsize=(fig_w_mm * MM, 232 * MM), facecolor=FIG_FACECOLOR)
    # extra height budget per block for its header row
    HEADER = 0.9
    ratios = [b[4] + HEADER for b in blocks]
    gs = GridSpec(len(blocks), 1, figure=fig, left=0.008, right=0.992,
                  top=0.965, bottom=0.008, hspace=0.06, height_ratios=ratios)

    for i, (label, count, color, arr, rows) in enumerate(blocks):
        ax = fig.add_subplot(gs[i])
        ax.imshow(arr, aspect='equal', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.0, 1.012, f'{label}', transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9, fontweight='bold', color=color, fontfamily=FONT_FAMILY)
        ax.text(1.0, 1.012, f'n = {count}', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color=color, fontfamily=FONT_FAMILY)

    fig.text(0.008, 0.988, _M['title'], ha='left', va='bottom',
             fontsize=11, fontweight='bold', fontfamily=FONT_FAMILY)

    # firing-rate colorbar (upper band): even ticks 0..vmax, white point at median.
    # Kept clear of the right-edge per-source `n =` labels.
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.46, 0.982, 0.24, 0.007])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='max')
    cbar.set_ticks(np.arange(0, vmax + 1, 10))
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('Mean IT response (spk/s)', fontsize=8, fontfamily=FONT_FAMILY)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    cbar.outline.set_linewidth(0.5)

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('NSD 649 + segmented 259 + fLoc 61 =', total)
    print('Saved ->', out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
