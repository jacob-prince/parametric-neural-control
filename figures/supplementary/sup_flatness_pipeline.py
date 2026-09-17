#!/usr/bin/env python3
"""Supp fig (flatness_pipeline) — feature accentuations and how gradient spectral
participation ratio (PR) is computed, for a single seed image at a single site.

Site = Monkey V V3/V4 unit 331; seed image = the cat (seed idx 3).
  a  the seed image (once) + a 2x5 mosaic of the 10 models' feature accentuations of that
     seed toward the site, matched to the highest achieved response ALL trained models
     reach (Untrained exempt -> shows its own max; achieved z printed per panel).
  b  the spectral pipeline for the same seed+site: one row per model (two blocks of five,
     in figure-4 order = control-r ascending) x four steps -- input gradient, grayscale
     (mean RGB, signed-sqrt stretched for visibility), 2D Fourier power, radial power.
     PR = (sum P)^2 / sum P^2 over frequency bins 1-111 (effective number of frequency
     bands); each profile is normalized so the arithmetic mean = 1 (dashed line) and the
     rows share one y-axis.

Gradient steps use the EXACT recipe from pnc/preproc/fft_utils.py; accentuation
thumbnails come from the accentuation-sweep PNGs. --gradsum cv renders the
coefficient-of-variation variant (filename tag _cv; the 1 +/- CoV band appears).
"""
import os, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.preproc.fft_utils import image_fourier_power, fourier_power_radial_profile_with_counts
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       get_model_color, MODEL_COLORS, MODEL_SHORT_NAMES, STIMULI_PATH,
                       STIMULI_CONTROL_PATH, MONKEY_DIRS, ACCENT_DATE_PREFIXES)
apply_figure_style()
_M = _fig('flatness_pipeline'); STEM = output_name('flatness_pipeline')
PRMODE = True                                              # gradsum 'pr' (default) | 'cv' variant; set in main()

MONKEY, UNIT, SEED = 'venus', 331, 3
SEED_FILE = 'shared0241_nsd20065.png'                      # seed idx 3 (cat)
UNTRAINED = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTRAINED] = 'Untrained'
FMIN, FMAX = 1, 112
FS_TITLE, FS_STEP, FS_AX, FS_TICK, FS_ANN, FS_ROW, FS_LET, FS_MOS = 8.0, 6.0, 6.0, 5.0, 5.0, 8.0, 11.0, 6.5

# --- canvas geometry (mm) ---
FIGW, FIGH = WIDTH_2COL_MM, 182.0
# panel B: gradient spectral pipeline -- two side-by-side blocks of 5 models
S, E_W = 16.0, 19.0                                        # square image side; radial-power width
IMG_GAP, GAP_E, ROWGAP = 2.0, 6.5, 3.0
DX = 90.0                                                  # left-block -> right-block x offset
XL = [10.0, 10.0 + (S + IMG_GAP), 10.0 + 2 * (S + IMG_GAP)]
EL_X = XL[2] + S + GAP_E
XR = [x + DX for x in XL]
ER_X = EL_X + DX
LBL_L, LBL_R = 6.5, 6.5 + DX
B_ROW0_TOP = 104.0                                         # top of the panel-B model rows
COL_TITLE = ['input gradient', 'grayscale', 'Fourier power', 'radial power']
# panel A: seed image + 10-model accentuation mosaic
A_SEED, A_SEED_X, A_SEED_BOT = 47.0, 6.0, 123.0            # seed side + lower-left
THUMB, MOS_GAP = 21.0, 3.0
MOS_X = [A_SEED_X + A_SEED + 5.0 + i * (THUMB + MOS_GAP) for i in range(5)]
ROW_BOT = [149.0, 123.0]                                   # thumb bottoms: row1 (top) / row2 (bottom)
ROW_LAB = [170.5, 145.0]                                   # model-label y per mosaic row


def accent_curve(model, unit, img_idx, monkey):
    """Sorted [(level, achieved_score, path), ...] for one model+site+seed accentuation sweep."""
    d = os.path.join(STIMULI_CONTROL_PATH,
                     f'{ACCENT_DATE_PREFIXES[monkey]}_{MONKEY_DIRS[monkey]}_{model}_accentuation')
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf'^{re.escape(model)}_RidgeCV_unit_{unit}_img_{img_idx}_level_(-?\d+\.\d+)_score_(-?\d+\.\d+)\.png$')
    out = [(float(m.group(1)), float(m.group(2)), os.path.join(d, fn))
           for fn in sorted(os.listdir(d)) if (m := pat.match(fn))]
    return sorted(out)


def load_accent_at(curve, target_resp, size=224):
    """Pick the sweep image whose achieved response is closest to target_resp; return (img, score)."""
    if not curve:
        return None, None
    lvl, score, path = min(curve, key=lambda t: abs(t[1] - target_resp))
    img = np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return img, score


def seed_thumb(px=224):
    for cand in (SEED_FILE, os.path.splitext(SEED_FILE)[0] + '.jpg'):
        p = os.path.join(STIMULI_PATH, cand)
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert('RGB').resize((px, px), Image.LANCZOS))
    return np.full((px, px, 3), 230, np.uint8)


def row_data(grad):
    hwc = grad.astype(np.float64).transpose(1, 2, 0)
    rgb = np.clip(0.5 + hwc / hwc.std(), 0, 1)             # upstream display norm
    gray = hwc.mean(2)                                     # grayscale = mean over RGB (FFT input)
    _, ps = image_fourier_power(hwc, return_shifted_spectrum=True)
    prof, _ = fourier_power_radial_profile_with_counts(ps)
    P = np.clip(prof[FMIN:FMAX], 1e-30, None)
    if PRMODE:                                             # participation ratio over the same band
        s2 = float((P ** 2).sum())
        cv = float(P.sum()) ** 2 / s2 if s2 > 0 else np.nan
    else:
        cv = P.std() / P.mean()
    return dict(rgb=rgb, gray=gray, ps=ps, prof=prof / P.mean(), cv=cv)   # arith mean -> 1


def main(out_dir, gradsum='pr'):
    global PRMODE
    PRMODE = gradsum == 'pr'
    gm = L.load_grad_maps(MONKEY, UNIT)
    models = [m for m in MODEL_COLORS if m in gm]
    data = {m: row_data(gm[m][SEED]) for m in models}
    order = models                                         # figure-4 order (control-r ascending = MODEL_COLORS)
    left, right = order[:5], order[5:]                     # panel-B blocks & panel-A mosaic rows
    freqs = np.arange(len(data[order[0]]['prof']))
    allp = np.concatenate([data[m]['prof'][FMIN:FMAX] for m in order])
    YLIM = (max(allp.min() * 0.6, 1e-3), allp.max() * 1.5)
    seed = seed_thumb()
    # accentuation target = highest achieved z reachable by ALL trained models (Untrained exempt -> its own max)
    curves = {m: accent_curve(m, UNIT, SEED, MONKEY) for m in models}
    tgt = min(max(s for _, s, _ in curves[m]) for m in models if m != UNTRAINED and curves[m])
    acc = {m: load_accent_at(curves[m], tgt) for m in models}

    def rb(r):
        return B_ROW0_TOP - S - r * (S + ROWGAP)

    fig = plt.figure(figsize=(FIGW * MM, FIGH * MM), facecolor=FIG_FACECOLOR)

    # ================= panel B: gradient spectral pipeline =================
    for mods, xs, e_x, lblx in [(left, XL, EL_X, LBL_L), (right, XR, ER_X, LBL_R)]:
        for r, m in enumerate(mods):
            d = data[m]; pb = rb(r); bot = (r == len(mods) - 1); col = get_model_color(m)
            ax = [fig.add_axes([x / FIGW, pb / FIGH, S / FIGW, S / FIGH]) for x in xs]
            ax[0].imshow(d['rgb'])
            g = np.sign(d['gray']) * np.sqrt(np.abs(d['gray']))     # signed-sqrt stretch: low-amplitude structure visible
            gv = np.percentile(np.abs(g), 98)
            ax[1].imshow(g, cmap='gray', vmin=-gv, vmax=gv)
            logp = np.log10(np.clip(d['ps'], d['ps'].max() * 1e-6, None))
            ax[2].imshow(logp, cmap='magma')
            cc = (d['ps'].shape[0] - 1) / 2.0
            for rr in (25, 60, 100):
                ax[2].add_patch(Circle((cc, cc), rr, fill=False, ec='white', lw=0.4, alpha=0.6))
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
                for s_ in a.spines.values():
                    s_.set_edgecolor('#888'); s_.set_linewidth(0.4)

            e = fig.add_axes([e_x / FIGW, pb / FIGH, E_W / FIGW, S / FIGH])
            if not PRMODE:                                 # 1 +/- CoV band is CoV-specific geometry
                e.fill_between([1, FMAX], max(1 - d['cv'], YLIM[0]), 1 + d['cv'], color=col, alpha=0.10, zorder=1)
            e.loglog(freqs[FMIN:FMAX], d['prof'][FMIN:FMAX], color=col, lw=1.0, zorder=4)
            e.axhline(1.0, color='#B40426', ls='--', lw=0.6, zorder=3)           # arith mean (= 1)
            e.set_xlim(1, FMAX); e.set_ylim(*YLIM); e.set_yticks([0.1, 1, 10])
            e.tick_params(labelsize=FS_TICK, length=2, width=0.4, which='major', pad=1.0)
            e.tick_params(which='minor', length=0)
            for s_ in ('top', 'right'):
                e.spines[s_].set_visible(False)
            for s_ in e.spines.values():
                s_.set_linewidth(0.4)
            if not bot:
                e.set_xticklabels([])
            e.text(0.95, 0.92, f'PR = {d["cv"]:.1f}' if PRMODE else f'CoV = {d["cv"]:.2f}',
                   transform=e.transAxes, ha='right', va='top',
                   fontsize=FS_ANN, fontfamily=FONT_FAMILY, color=col, fontweight='bold', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#cccccc', lw=0.4))

            fig.text(lblx / FIGW, (pb + S / 2) / FIGH, SHORT[m], rotation=90, ha='center',
                     va='center', color=col, fontsize=FS_ROW, fontweight='bold', fontfamily=FONT_FAMILY)

        xcen = [x + S / 2 for x in xs] + [e_x + E_W / 2]
        for xc, t in zip(xcen, COL_TITLE):
            fig.text(xc / FIGW, (B_ROW0_TOP + 2.0) / FIGH, t, ha='center', va='bottom',
                     fontsize=FS_STEP, fontfamily=FONT_FAMILY, fontweight='bold')
        fig.text((e_x + E_W / 2) / FIGW, (rb(4) - 4.5) / FIGH, 'spatial freq. (cyc/img)',
                 ha='center', va='center', fontsize=FS_AX - 0.5, fontfamily=FONT_FAMILY)

    # ================= panel A: seed image + accentuation mosaic =================
    axs = fig.add_axes([A_SEED_X / FIGW, A_SEED_BOT / FIGH, A_SEED / FIGW, A_SEED / FIGH])
    axs.imshow(seed); axs.set_xticks([]); axs.set_yticks([])
    for s_ in axs.spines.values():
        s_.set_edgecolor('#444'); s_.set_linewidth(0.6)
    fig.text((A_SEED_X + A_SEED / 2) / FIGW, (A_SEED_BOT - 2.5) / FIGH, 'seed image',
             ha='center', va='top', fontsize=FS_MOS, fontfamily=FONT_FAMILY)

    for ri, mods in enumerate([left, right]):
        for ci, m in enumerate(mods):
            img, sc = acc[m]; col = get_model_color(m)
            ax = fig.add_axes([MOS_X[ci] / FIGW, ROW_BOT[ri] / FIGH, THUMB / FIGW, THUMB / FIGH])
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_edgecolor(col); s_.set_linewidth(0.9)
            fig.text((MOS_X[ci] + THUMB / 2) / FIGW, ROW_LAB[ri] / FIGH, SHORT[m], ha='center',
                     va='bottom', fontsize=FS_MOS, color=col, fontweight='bold', fontfamily=FONT_FAMILY)
            if sc is not None:
                ax.text(0.05, 0.05, f'z = {sc:.1f}', transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=FS_TICK, color='#111', fontfamily=FONT_FAMILY,
                        bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none', alpha=0.65))

    # panel letters + titles
    fig.text(4 / FIGW, 177 / FIGH, 'a', fontsize=FS_LET, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)
    fig.text(11 / FIGW, 177 / FIGH, f'Feature accentuation toward Monkey V (V3/V4) unit 331; target: z = {tgt:.1f}',
             fontsize=FS_TITLE, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)
    fig.text(4 / FIGW, 113 / FIGH, 'b', fontsize=FS_LET, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)
    fig.text(11 / FIGW, 113 / FIGH,
             _M['title'] if PRMODE else 'Computing gradient spectral concentration (CoV)',
             fontsize=FS_TITLE, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)

    out = save_fig(fig, os.path.join(out_dir, STEM + ('' if PRMODE else '_cv')))
    plt.close(fig)
    print('saved ->', out)
    print(f'matched target z = {tgt:.2f}')
    print(('PR' if PRMODE else 'CoV') + ' (fig4 order):',
          {SHORT[m]: round(data[m]['cv'], 3) for m in order})
    print('accent achieved z:', {SHORT[m]: (round(acc[m][1], 2) if acc[m][1] is not None else None) for m in order})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    ap.add_argument('--gradsum', choices=['pr', 'cv'], default='pr',
                    help="gradient spectral summary: 'pr' (default) | 'cv' (filename tag _cv)")
    main(**vars(ap.parse_args()))
