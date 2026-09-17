#!/usr/bin/env python3
"""Supp fig (success_rate) — feature-accentuation success rate.

For every synthesized stimulus the achieved level (the generating model's own post-hoc
pred_resp for the SAVED image) is compared to its intended target. A generation succeeds
if the miss is within a channel-relative tolerance:

    relerr = |target - achieved| / (per-channel target range) <= tau

The target grid is identical across models within a channel, so relerr is fair across
channels of differing range. tau = 5% is the canonical default (2% is also reported).

(a) the 10 natural seed images accentuation sweeps start from;
(b) histogram of the miss as % of channel target-range (dashed lines at 2% and 5%);
(c) per-model success rate at 5% (dotted line at the overall rate);
(d) success rate at 5% vs level_ord (failures concentrate at the extreme drive levels).

Reads only loader.load_exclusions() (canonical relerr from faithful pred_resp; S-number and
output stem from the manifest) + the 10 seed thumbnails from data/stimuli_encoding (seed 4
uses the canonical cropped figures/assets/seed.png, matching every other figure that
shows this seed).
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, DPI, FIG_FACECOLOR, save_fig, MM, WIDTH_2COL_MM,
                       MODEL_ORDER, MODEL_SHORT_NAMES, MODEL_COLORS, get_model_color, STIMULI_PATH)
apply_figure_style()

# the 10 seed images (index -> filename), matching fig1_grid.py /
# sup_gradient_gallery.py; seed 4 (dog) uses the shared cropped asset, not the raw file
SEED_IMAGES = ['shared0575_nsd43157.png', 'shared0850_nsd61798.png', 'shared0968_nsd70194.png',
               'shared0241_nsd20065.png', 'shared0160_nsd13231.png', 'shared0070_nsd07008.png',
               'shared0055_nsd05879.png', 'shared0668_nsd48623.png', 'shared0488_nsd36979.png',
               'shared0940_nsd68312.png']
DOG_SEED_IDX = 4
SEED_PNG = str(paths.ASSETS / 'seed.png')


def seed_thumb(si, px=224):
    p = SEED_PNG if si == DOG_SEED_IDX else os.path.join(STIMULI_PATH, SEED_IMAGES[si])
    return np.asarray(Image.open(p).convert('RGB').resize((px, px), Image.LANCZOS))


STEM = output_name('success_rate')

MODEL_SHORT_NAMES = dict(MODEL_SHORT_NAMES)
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'      # display relabel

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
TAU, TAU2 = 0.05, 0.02          # primary / secondary success thresholds
FS_TITLE, FS_AX, FS_TICK = 7, 6.5, 5.5          # final-size-native (5-7 pt at 180 mm)


def main(out_dir):
    # ── load canonical relerr (faithful pred_resp) ───────────────────────────
    model = []; relerr = []; level = []
    for m in MONKEYS:
        ex = L.load_exclusions(m)   # per-monkey dict, 5500 records
        model.append(np.asarray(ex['gen_model'])); relerr.append(np.asarray(ex['relerr'], float))
        level.append(np.asarray(ex['level_ord']))
    model = np.concatenate(model); relerr = np.concatenate(relerr); level = np.concatenate(level)
    fin = np.isfinite(relerr); model, relerr, level = model[fin], relerr[fin], level[fin]

    succ = relerr <= TAU
    overall = succ.mean()
    overall2 = (relerr <= TAU2).mean()
    per_model = {m: succ[model == m].mean() * 100 for m in MODEL_ORDER}

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 68 * MM), facecolor=FIG_FACECOLOR)
    outer = GridSpec(2, 1, figure=fig, left=0.065, right=0.985, top=0.865, bottom=0.105,
                     hspace=0.46, height_ratios=[0.40, 1.0])

    # a: the 10 natural seed images, in accentuation-sweep order
    gsSeed = GridSpec(1, 10, figure=fig, left=0.065, right=0.985, wspace=0.10,
                      top=outer[0].get_position(fig).y1, bottom=outer[0].get_position(fig).y0)
    for si in range(10):
        axS = fig.add_subplot(gsSeed[si])
        axS.imshow(seed_thumb(si)); axS.set_xticks([]); axS.set_yticks([])
        for sp in axS.spines.values():
            sp.set_color('#999'); sp.set_linewidth(0.5)
        axS.set_title(str(si), fontsize=FS_TICK, fontfamily=FONT_FAMILY, color='#666', pad=2)
    fig.text(0.065, outer[0].get_position(fig).y1 + 0.045, 'Seed images', fontsize=FS_TITLE,
             fontfamily=FONT_FAMILY, ha='left', va='bottom')

    gs = GridSpec(1, 3, figure=fig, left=0.065, right=0.985,
                  top=outer[1].get_position(fig).y1, bottom=outer[1].get_position(fig).y0,
                  wspace=0.40, width_ratios=[1.0, 1.35, 1.25])

    # b: histogram of the channel-relative miss (% of range)
    _bins = np.linspace(0, 30, 49)
    axA = fig.add_subplot(gs[0])
    axA.hist(np.clip(relerr * 100, 0, 30), bins=_bins, color='#3274A1',
             alpha=0.85, edgecolor='white', linewidth=0.3, zorder=3)
    axA.axvline(TAU2 * 100, color='#F4A259', ls='--', lw=1.4, zorder=5, label='2% threshold')
    axA.axvline(TAU * 100, color='#E63946', ls='--', lw=1.4, zorder=5, label='5% threshold')
    axA.set_xlabel('miss  |target - achieved|  (% of channel range)', fontsize=FS_AX)
    axA.set_ylabel('Number of stimuli', fontsize=FS_AX)
    axA.set_title('Accentuation accuracy', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=6)
    axA.set_xlim(0, 30); axA.tick_params(labelsize=FS_TICK)
    axA.legend(fontsize=FS_TICK, frameon=False, loc='upper right')

    # A inset: same histogram, y-axis rescaled ~80x to expose the tail past the hump
    axA_in = inset_axes(axA, width='60%', height='42%', loc='lower right',
                        borderpad=1.1)
    axA_in.hist(np.clip(relerr * 100, 0, 30), bins=_bins, color='#3274A1',
                alpha=0.85, edgecolor='white', linewidth=0.3, zorder=3)
    axA_in.axvline(TAU2 * 100, color='#F4A259', ls='--', lw=1.1, zorder=5)
    axA_in.axvline(TAU * 100, color='#E63946', ls='--', lw=1.1, zorder=5)
    axA_in.set_xlim(0, 30); axA_in.set_ylim(0, 300)
    axA_in.set_yticks([0, 150, 300])
    axA_in.set_xticks([0, 10, 20, 30])
    axA_in.tick_params(labelsize=FS_TICK - 1, length=2, pad=1.5)
    axA_in.text(0.97, 0.93, 'tail  (y $\\times$80)', transform=axA_in.transAxes,
                ha='right', va='top', fontsize=FS_TICK - 0.5, fontfamily=FONT_FAMILY)
    for s in axA_in.spines.values():
        s.set_linewidth(0.5)

    # c: per-model success rate at 5%
    axB = fig.add_subplot(gs[1])
    xs = np.arange(len(MODEL_ORDER))
    axB.bar(xs, [per_model[m] for m in MODEL_ORDER],
            color=[get_model_color(m) for m in MODEL_ORDER], edgecolor='white',
            linewidth=0.5, zorder=3)
    axB.axhline(overall * 100, color='#555', ls=':', lw=1.1, zorder=2)
    axB.text(len(MODEL_ORDER) - 0.55, overall * 100 + 1.3, f'overall {overall*100:.1f}%',
             ha='right', va='bottom', fontsize=FS_TICK, color='#555', zorder=6)
    axB.set_xticks(xs)
    axB.set_xticklabels([MODEL_SHORT_NAMES[m] for m in MODEL_ORDER], rotation=55,
                        ha='right', fontsize=FS_TICK)
    for t, m in zip(axB.get_xticklabels(), MODEL_ORDER):
        t.set_color(get_model_color(m))
    axB.set_ylim(0, 105); axB.set_ylabel('Success rate (%)', fontsize=FS_AX)
    axB.set_title('Per-model success rate (5%)', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=6)
    axB.tick_params(axis='y', labelsize=FS_TICK)

    # d: success rate at 5% vs drive level (where failures occur)
    axC = fig.add_subplot(gs[2])
    levels = np.arange(int(level.max()) + 1)
    for m in MODEL_ORDER:
        mm = model == m
        rate = [succ[mm & (level == lv)].mean() * 100 if (mm & (level == lv)).sum() >= 10 else np.nan
                for lv in levels]
        axC.plot(levels, rate, '-', color=get_model_color(m), alpha=0.4, lw=0.9, zorder=2)
    rate_all = [succ[level == lv].mean() * 100 if (level == lv).sum() >= 10 else np.nan
                for lv in levels]
    axC.plot(levels, rate_all, '-', color='#111', lw=2.4, zorder=5, label='all models')
    axC.set_ylim(0, 105); axC.set_xlim(levels.min() - 0.3, levels.max() + 0.3)
    axC.set_xticks(levels)
    axC.set_xlabel('Feature level  (ordinal, suppress -> drive)', fontsize=FS_AX)
    axC.set_ylabel('Success rate (%)', fontsize=FS_AX)
    axC.set_title('Success rate vs feature level', fontsize=FS_TITLE - 1,
                  fontfamily=FONT_FAMILY, pad=6)
    axC.tick_params(labelsize=FS_TICK)
    axC.legend(fontsize=FS_TICK, frameon=False, loc='lower center')

    fig.suptitle('Feature-accentuation success rate',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY, fontweight='bold', y=0.98)

    # panel letters — anchored to the top-left of each subplot
    fig.canvas.draw()
    fig.text(0.065 - 0.036, outer[0].get_position(fig).y1 + 0.045 + 0.028, 'a', fontsize=12,
             fontweight='bold', fontfamily=FONT_FAMILY, va='top', ha='left')
    for ax_ref, lab, dx in [(axA, 'b', 0.036), (axB, 'c', 0.030), (axC, 'd', 0.030)]:
        p = ax_ref.get_position()
        fig.text(p.x0 - dx, p.y1 + 0.070, lab, fontsize=12, fontweight='bold',
                 fontfamily=FONT_FAMILY, va='top', ha='left')

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)

    print(f'Saved -> {out}')
    print(f'N stimuli (finite relerr): {len(relerr)}')
    print(f'Overall success @5%: {overall*100:.1f}%')
    print(f'Overall success @2%: {overall2*100:.1f}%')
    print('Per-model success rate @5%:')
    for m in MODEL_ORDER:
        print(f'  {MODEL_SHORT_NAMES[m]:12s} ({m:26s}) {per_model[m]:5.1f}%')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
