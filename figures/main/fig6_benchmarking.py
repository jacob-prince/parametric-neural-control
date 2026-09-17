#!/usr/bin/env python3
"""Figure 6 - implications for benchmarking and model comparison.

  Q1  Does pooling accentuations into a larger benchmark strengthen or weaken model separation?
    a  model separation across 4 benchmark conditions (personal -> pooled ->
       pooled-minus-own -> pooled-minus-robust-generated)
    b  targeted self-prediction (SM-SC): per-model predicted-activity raincloud
    c  full pooled synthetic benchmark: per-model predicted-activity raincloud
  Q2  Can carefully selected natural images match synthesized stimuli?
    d  example pair, natural-image (ImageNet) prediction plane (RN50-CLIP x CLIPAG, red aIT)
    e  example pair, accentuated sweeps on the same plane
    f  example-pair per-image disagreement |A-B|: ImageNet best-110/model vs accentuated
    g  across 1,125 (channel x pair) combos: ImageNet best vs accentuated
    h  disagreement vs per-model image budget k

Self-contained: all data is read via `fig6_data` (the fig6_* caches in preprocessed_data/,
built by scripts/preprocessing/build_fig6_caches.py from pnc.preproc.loader -> faithful
pred_resp, clamped) plus the frozen ImageNet thumbnails preprocessed_data/fig6_imagenet_thumbs.
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy import stats

from pnc import paths
from pnc.manifest import MAIN_FIGURES
from pnc.utils import (apply_figure_style, FONT_FAMILY, FIG_FACECOLOR, save_fig,
                       STIMULI_CONTROL_PATH, MONKEY_DIRS, ACCENT_DATE_PREFIXES,
                       MODEL_COLORS, MODEL_ORDER, MODEL_SHORT_NAMES)
from figures.main import fig6_data as D
apply_figure_style()
MODEL_SHORT_NAMES = dict(MODEL_SHORT_NAMES)
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'

MODELS = list(MODEL_ORDER)

# worked-example pair (panels d/e/f): preset `rn50` (default) or `clip`
PAIR_PRESETS = {'clip': ('resnet50_clip', 'clipag_vitb32'), 'rn50': ('resnet50', 'resnet50_robust')}
CH = 2                                   # red unit 9 (aIT featured site)
COL_INET, COL_INET_BEST, COL_ACC = '#7F8C8D', 'dodgerblue', '#E1812C'
PER_MODEL_BUDGET = 110
MONKEY_ORDER = ['red', 'paul', 'venus', 'three0', 'leap']

# fonts authored for the ~14" canvas -> ~5-7 pt at 180 mm final width
FS_PANEL, FS_TITLE, FS_AXLABEL, FS_TICK = 20, 12, 10.5, 9
FS_SUB, FS_ANNOT, FS_LEG = 10.5, 9, 8.5
FS_SCAX, FS_SCTICK = 13, 11               # scatter-plane axis / tick labels (big panels)


# ============================ Q1: a / b / c =================================
def plot_violin_panel(ax, per_model, title, dot_size, dot_alpha):
    vw = 0.85
    for xi, m in enumerate(MODELS):
        arr = per_model[m]
        if len(arr) == 0:
            continue
        c = MODEL_COLORS[m]
        parts = ax.violinplot([arr], positions=[xi], showextrema=False, widths=vw)
        for body in parts['bodies']:
            v = body.get_paths()[0].vertices; v[:, 0] = np.clip(v[:, 0], -np.inf, xi)
            body.set(facecolor=c, edgecolor=c, alpha=0.55, linewidth=0.8)
        med = float(np.median(arr))
        ax.plot([xi - vw / 2 * 0.55, xi - 0.005], [med, med], color='black', lw=1.1, zorder=5)
        # Show EVERY point — no subsampling. The per-model RNG drives ONLY the horizontal jitter
        # (seeded per model so a change to one model's n never reshuffles another's dots).
        # Rasterized: the full point cloud stays lightweight in the vector PDF.
        r = np.random.default_rng(1000 + xi)
        ax.scatter(xi + r.uniform(0.04, 0.36, len(arr)), arr, s=dot_size, color=c,
                   alpha=dot_alpha, edgecolors='none', zorder=4, rasterized=True)
    ax.axhline(0, color='#888', lw=0.5, alpha=0.5, zorder=0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_SHORT_NAMES[m] for m in MODELS], rotation=45, ha='right',
                       fontsize=FS_TICK - 1.5, fontfamily=FONT_FAMILY, fontweight='bold')
    for lbl, m in zip(ax.get_xticklabels(), MODELS):
        lbl.set_color(MODEL_COLORS[m])
    ax.set_xlim(-0.7, len(MODELS) - 0.3)
    ax.set_ylabel('Predicted activity (z)', fontsize=FS_AXLABEL, fontfamily=FONT_FAMILY)
    ax.set_title(title, fontsize=FS_TITLE, fontfamily=FONT_FAMILY, fontweight='bold', pad=6)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, axis='y', alpha=0.4, lw=0.6, color='#888888'); ax.set_axisbelow(True)


def _ci95(rs):
    rs = np.asarray(rs, float)
    if len(rs) < 2:
        return (float(rs.mean()) if len(rs) else np.nan), np.nan, np.nan
    m = float(rs.mean()); sem = float(rs.std(ddof=1) / np.sqrt(len(rs)))
    lo, hi = stats.t.interval(0.95, len(rs) - 1, loc=m, scale=sem)
    return m, float(lo), float(hi)


# ---- thumbnail grids (ported) ----
def _accent_local_path(source_model, unit_id, score):
    best, best_d = None, 1e9
    for mk in MONKEY_ORDER:
        d = os.path.join(STIMULI_CONTROL_PATH, f'{ACCENT_DATE_PREFIXES[mk]}_{MONKEY_DIRS[mk]}_{source_model}_accentuation')
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, f'*unit_{unit_id}_*_score_*.png'))):
            try:
                s = float(os.path.basename(p).rsplit('_score_', 1)[1][:-4])
            except (ValueError, IndexError):
                continue
            if abs(s - score) < best_d:
                best, best_d = p, abs(s - score)
    return best if best_d < 0.05 else None


# ============================ Q2: h (budget curve) =========================
def _band(a):
    return np.nanmedian(a, axis=1), np.nanpercentile(a, 25, axis=1), np.nanpercentile(a, 75, axis=1)


def main(out_dir, preset='rn50'):
    PRESET = preset
    MODEL_A, MODEL_B = PAIR_PRESETS[PRESET]

    SEP = D.load_separation()
    EFF = D.load_efficiency()
    PAIR = D.load_pair(MODEL_A, MODEL_B, CH)
    paths.require(D.THUMB_DIR_INET, hint='frozen input: preprocessed_data/fig6_imagenet_thumbs (ImageNet val thumbnails)')

    # ============================ layout ========================================
    fig = plt.figure(figsize=(14.0, 9.6), facecolor=FIG_FACECOLOR)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.88, 1.52], hspace=0.34,
                     left=0.05, right=0.965, top=0.93, bottom=0.065)
    gs1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.42, 1.0, 1.0], wspace=0.22)
    ax_slope = fig.add_subplot(gs1[0])
    ax_smsc = fig.add_subplot(gs1[1])
    ax_full = fig.add_subplot(gs1[2], sharey=ax_smsc)
    gs2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.22, 1.22, 0.66], wspace=0.24)
    ax_nat = fig.add_subplot(gs2[0])
    ax_acc = fig.add_subplot(gs2[1])
    gs_fgh = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs2[2], hspace=0.62)
    ax_dist = fig.add_subplot(gs_fgh[0])
    ax_combos = fig.add_subplot(gs_fgh[1])
    ax_budget = fig.add_subplot(gs_fgh[2])

    # ============================ Q1: a / b / c =================================
    plot_violin_panel(ax_smsc, SEP['violin_samples']['SM-SC'], 'Model predictions: personalized accentuations',
                      dot_size=2.5, dot_alpha=0.38)
    plot_violin_panel(ax_full, SEP['violin_samples']['full'], 'Model predictions: pooled benchmark',
                      dot_size=1.5, dot_alpha=0.18)

    sep_pc = SEP['per_channel']
    COND = ['personal', 'pooled', 'pooled_minus_own', 'pooled_minus_robust']
    COND_X = [0.0, 1.0, 2.0, 3.0]
    COND_TICKS = ['Personal\n(own probes)', 'Pooled\n(all stims)', 'Pooled\n(minus own)',
                  'Pooled\n(minus robust-\ngenerated)']
    label_entries = []
    for m in MODELS:
        ys, los, his, ok = [], [], [], True
        for cond in COND:
            rs = sep_pc[cond].get(m, np.array([]))
            if len(rs) == 0:
                ok = False; break
            mm, lo, hi = _ci95(rs); ys.append(mm); los.append(lo); his.append(hi)
        if not ok:
            continue
        c = MODEL_COLORS[m]
        ax_slope.plot(COND_X, ys, color=c, lw=1.4, alpha=0.9, solid_capstyle='round', zorder=3)
        for x, mm, lo, hi in zip(COND_X, ys, los, his):
            ax_slope.errorbar([x], [mm], yerr=[[mm - lo], [hi - mm]], ecolor=c, elinewidth=1.0,
                              capsize=2.5, capthick=0.9, fmt='none', alpha=0.85, zorder=4)
        ax_slope.scatter(COND_X, ys, s=20, color=c, edgecolors='white', linewidths=0.7, zorder=5)
        label_entries.append((ys[0], ys[-1], m, c))

    label_entries.sort(key=lambda t: t[0], reverse=True)
    ys_stack = np.linspace(0.78, 0.10, len(label_entries))
    for (_ry, end_y, m, c), y in zip(label_entries, ys_stack):
        ax_slope.plot([3.03, 3.40], [end_y, y], color=c, lw=0.5, alpha=0.6, zorder=2, clip_on=False)
        ax_slope.scatter([3.42], [y], s=16, color=c, edgecolors='white', linewidths=0.5, zorder=5, clip_on=False)
        ax_slope.text(3.50, y, MODEL_SHORT_NAMES[m], fontsize=FS_ANNOT - 1, fontfamily=FONT_FAMILY,
                      fontweight='bold', color=c, va='center', ha='left', clip_on=False)

    sep_sd = {}
    for x, cond in zip(COND_X, COND):
        means = [np.nanmean(sep_pc[cond][m]) for m in MODELS if 'AlexNet' not in m and len(sep_pc[cond][m])]
        sd = float(np.nanstd(means, ddof=1)); sep_sd[cond] = sd
        ax_slope.text(x, 0.86, f'SD\n{sd:.2f}', ha='center', va='top', fontsize=FS_ANNOT - 1,
                      fontfamily=FONT_FAMILY, fontweight='bold', color='#333')
    ax_slope.set_xticks(COND_X)
    ax_slope.set_xticklabels(COND_TICKS, fontsize=FS_TICK - 1, fontfamily=FONT_FAMILY, fontweight='bold')
    ax_slope.set_xlim(-0.35, 4.55); ax_slope.set_ylim(0, 0.92)
    ax_slope.tick_params(labelsize=FS_TICK)
    ax_slope.set_ylabel('Brain predictivity r (mean $\\pm$ 95% CI)', fontsize=FS_AXLABEL, fontfamily=FONT_FAMILY)
    ax_slope.grid(True, axis='y', alpha=0.4, lw=0.6, color='#888888'); ax_slope.set_axisbelow(True)
    ax_slope.set_title('Pooled-stimulus model rankings', fontsize=FS_TITLE,
                       fontfamily=FONT_FAMILY, fontweight='bold', pad=6)

    # ============================ Q2: d / e ====================================
    iA, iB = PAIR['inet_predA'], PAIR['inet_predB']
    sweeps_all = PAIR['sweeps_A'] + PAIR['sweeps_B']
    sweep_vals = np.concatenate([s['pred_A'] for s in sweeps_all] + [s['pred_B'] for s in sweeps_all])
    inet_ext = float(max(abs(np.percentile(iA, 99.5)), abs(np.percentile(iB, 99.5)),
                         abs(np.percentile(iA, 0.5)), abs(np.percentile(iB, 0.5))))
    r = float(max(inet_ext, np.percentile(np.abs(sweep_vals), 99.5)))
    LIMS = (-r * 1.04, r * 1.04)

    def _plane(ax, title, faint):
        ax.plot(LIMS, LIMS, 'k--', lw=0.7, alpha=0.35, zorder=0)
        ax.scatter(iA, iB, s=0.7, color=COL_INET, alpha=faint, edgecolors='none', rasterized=True, zorder=3)
        ax.axhline(0, color='#CCCCCC', lw=0.4, zorder=0); ax.axvline(0, color='#CCCCCC', lw=0.4, zorder=0)
        ax.set_xlim(LIMS); ax.set_ylim(LIMS); ax.set_aspect('equal')
        ax.set_xlabel(f'{MODEL_SHORT_NAMES[MODEL_A]} predicted (z)', fontsize=FS_SCAX,
                      fontfamily=FONT_FAMILY, color=MODEL_COLORS[MODEL_A])
        ax.set_ylabel(f'{MODEL_SHORT_NAMES[MODEL_B]} predicted (z)', fontsize=FS_SCAX,
                      fontfamily=FONT_FAMILY, color=MODEL_COLORS[MODEL_B])
        ax.tick_params(labelsize=FS_SCTICK)
        ax.set_title(title, pad=6, fontsize=FS_TITLE, fontweight='bold')

    _plane(ax_nat, 'Controversial ImageNet stimulus predictions', 0.20)
    _plane(ax_acc, 'Accentuated stimulus predictions', 0.10)
    norm_b = mcolors.Normalize(vmin=-3.0, vmax=6.0); cmap_b = plt.colormaps['RdBu_r']
    for sweeps in (PAIR['sweeps_A'], PAIR['sweeps_B']):
        for sw in sweeps:
            xs, ys, sc = sw['pred_A'], sw['pred_B'], sw['score']
            ax_acc.plot(xs, ys, color='black', lw=0.5, alpha=0.45, zorder=4)
            ax_acc.scatter(xs, ys, s=12, c=[cmap_b(norm_b(v)) for v in sc], edgecolors='black',
                           linewidths=0.3, zorder=5)
            si = int(np.argmin(np.abs(sc)))
            ax_acc.scatter([xs[si]], [ys[si]], s=28, marker='D', color='white', edgecolors='black',
                           linewidths=0.8, zorder=6)
    ax_acc.text(0.04, 0.96, 'ImageNet val', transform=ax_acc.transAxes, fontsize=FS_ANNOT,
                fontfamily=FONT_FAMILY, color=COL_INET, fontweight='bold', va='top')

    def _place_thumb_grid(ax, items, corner, color, data_xy, thumb_frac=0.105, gap_frac=0.008,
                          max_thumbs=15, region_frac=0.70, thumb_px=140):
        span = LIMS[1] - LIMS[0]; tw = span * thumb_frac; step = tw + span * gap_frac; pad = span * 0.012
        dx, dy = data_xy[:, 0], data_xy[:, 1]; m = tw * 0.04
        occ = lambda xl, yb, xr, yt: bool(np.any((dx >= xl - m) & (dx <= xr + m) & (dy >= yb - m) & (dy <= yt + m)))
        n = int((region_frac * span) // step); cells = []
        for rr in range(n):
            for cc in range(n):
                if corner == 'upper-left':
                    xl = LIMS[0] + pad + cc * step; yt = LIMS[1] - pad - rr * step; yb, xr = yt - tw, xl + tw
                else:
                    xr = LIMS[1] - pad - cc * step; yb = LIMS[0] + pad + rr * step; yt, xl = yb + tw, xr - tw
                cells.append((rr + cc, xl, yb, xr, yt))
        cells.sort(); it = iter(items); placed = 0
        for _, xl, yb, xr, yt in cells:
            if placed >= max_thumbs:
                break
            if occ(xl, yb, xr, yt):
                continue
            try:
                item = next(it)
            except StopIteration:
                break
            path = item.get('thumb_path') if 'thumb_path' in item else None
            if path is not None and not os.path.exists(str(path)) and 'idx' in item:
                path = os.path.join(D.THUMB_DIR_INET, f"{item['idx']}.JPEG")   # cached abs path is folder-move stale -> rebuild
            if path is None and 'source_model' in item:
                path = _accent_local_path(item['source_model'], item['unit_id'], item['score'])
            if not (path and os.path.exists(str(path))):
                continue
            try:
                img = Image.open(str(path)).convert('RGB').resize((thumb_px, thumb_px), Image.LANCZOS)
            except Exception:
                continue
            ax.imshow(np.asarray(img), extent=(xl, xr, yb, yt), aspect='equal', zorder=12)
            ax.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb], color=color, lw=0.8, zorder=13)
            ax.scatter([item['predA']], [item['predB']], s=30, marker='s', facecolor=color,
                       edgecolors='white', linewidths=0.7, zorder=8)
            placed += 1
        return placed

    _sweep_xy = np.column_stack([np.concatenate([s['pred_A'] for s in sweeps_all]),
                                 np.concatenate([s['pred_B'] for s in sweeps_all])])
    _data_nat = np.column_stack([iA, iB]); _data_acc = np.vstack([_data_nat, _sweep_xy])
    _D_TS = dict(thumb_frac=0.092, gap_frac=0.006, max_thumbs=24, region_frac=0.78)
    _place_thumb_grid(ax_nat, PAIR['inet_bpref'], 'upper-left', MODEL_COLORS[MODEL_B], _data_nat, **_D_TS)
    _place_thumb_grid(ax_nat, PAIR['inet_apref'], 'lower-right', MODEL_COLORS[MODEL_A], _data_nat, **_D_TS)
    _place_thumb_grid(ax_acc, PAIR['acc_bpref'], 'upper-left', MODEL_COLORS[MODEL_B], _data_acc)
    _place_thumb_grid(ax_acc, PAIR['acc_apref'], 'lower-right', MODEL_COLORS[MODEL_A], _data_acc)

    # ============================ Q2: f (example-pair disagreement) =============
    inet_signed = PAIR['inet_predA_std'] - PAIR['inet_predB_std']
    inet_diff = np.abs(inet_signed)
    sw_A = np.concatenate([s['pred_A_std'] for s in sweeps_all])
    sw_B = np.concatenate([s['pred_B_std'] for s in sweeps_all])
    acc_diff = np.abs(sw_A - sw_B)
    best_idx = np.concatenate([np.argsort(inet_signed)[::-1][:PER_MODEL_BUDGET],
                               np.argsort(inet_signed)[:PER_MODEL_BUDGET]])
    data_f = [inet_diff[best_idx], acc_diff]
    labels_f = [f'ImageNet\nbest-{PER_MODEL_BUDGET}/model', f'Accentuated\n{PER_MODEL_BUDGET}/model']
    colors_f = [COL_INET_BEST, COL_ACC]
    parts = ax_dist.violinplot(data_f, positions=[0, 1], showextrema=False, widths=0.52)
    for body, col in zip(parts['bodies'], colors_f):
        body.set(facecolor=col, edgecolor=col, alpha=0.55, linewidth=0.6)
    rng_f = np.random.default_rng(0)
    for pos, arr, col in zip([0, 1], data_f, colors_f):
        ax_dist.scatter(pos + rng_f.uniform(-0.14, 0.14, len(arr)), arr, s=6, color=col, alpha=0.5,
                        edgecolors='none', zorder=3)
        mv, mx = float(np.mean(arr)), float(np.max(arr))
        ax_dist.plot([pos - 0.22, pos + 0.22], [mv, mv], color='black', lw=2.0, zorder=6)
        ax_dist.scatter([pos], [mx], marker='*', s=110, color='black', edgecolors='white', linewidths=0.7, zorder=7)
        ax_dist.text(pos + 0.30, mv, f'mean {mv:.2f}', fontsize=FS_LEG - 1, fontfamily=FONT_FAMILY,
                     fontweight='bold', ha='left', va='center')
        ax_dist.text(pos, mx + 0.10, f'max {mx:.2f}', fontsize=FS_LEG - 1.5, fontfamily=FONT_FAMILY,
                     fontweight='bold', ha='center', va='bottom')
    ax_dist.set_xticks([0, 1]); ax_dist.set_xticklabels(labels_f, fontsize=FS_LEG, fontfamily=FONT_FAMILY, fontweight='bold')
    for lbl, col in zip(ax_dist.get_xticklabels(), colors_f):
        lbl.set_color(col)
    ax_dist.set_xlim(-0.6, 1.6); ax_dist.set_ylim(-0.2, max(np.max(acc_diff), np.max(inet_diff[best_idx])) * 1.22)
    ax_dist.set_ylabel('|A $-$ B| (std units)', fontsize=FS_SUB, fontfamily=FONT_FAMILY)
    ax_dist.tick_params(labelsize=FS_TICK)
    ax_dist.set_title('Predicted disagreement', pad=5, fontsize=FS_SUB, fontweight='bold')

    # ============================ Q2: g (across 1,125 combos) ===================
    gd = [EFF['inet_group_top'], EFF['acc_group_disp']]
    parts = ax_combos.violinplot([g[~np.isnan(g)] for g in gd], positions=[0, 1], showextrema=False, widths=0.52)
    for body, col in zip(parts['bodies'], colors_f):
        body.set(facecolor=col, edgecolor=col, alpha=0.65, linewidth=0.6)
    rng_g = np.random.default_rng(0)
    for pos, arr, col in zip([0, 1], gd, colors_f):
        a = arr[~np.isnan(arr)]
        ax_combos.scatter(pos + rng_g.uniform(-0.15, 0.15, len(a)), a, s=4, color=col, alpha=0.35,
                          edgecolors='none', zorder=4, rasterized=True)
        ax_combos.errorbar(pos, float(a.mean()), yerr=float(a.std(ddof=1) / np.sqrt(len(a))), fmt='D',
                           color='black', markersize=4.5, elinewidth=0.9, capsize=2.5, zorder=6)
    frac = float(np.nanmean(np.asarray(EFF['acc_group_disp']) > np.asarray(EFF['inet_group_top'])))
    ax_combos.set_xticks([0, 1]); ax_combos.set_xticklabels(labels_f, fontsize=FS_LEG, fontfamily=FONT_FAMILY, fontweight='bold')
    for lbl, col in zip(ax_combos.get_xticklabels(), colors_f):
        lbl.set_color(col)
    ax_combos.tick_params(axis='y', labelsize=FS_TICK)
    ax_combos.set_ylabel('Mean |A $-$ B|', fontsize=FS_SUB, fontfamily=FONT_FAMILY)
    ax_combos.set_title('Summary over model pairs', pad=5, fontsize=FS_SUB, fontweight='bold')

    # ============================ Q2: h (budget curve) =========================
    m_i, lo_i, hi_i = _band(EFF['inet_budget'])
    m_a, lo_a, hi_a = _band(EFF['acc_budget'])
    k_i = np.asarray(EFF['k_grid']) / 2.0
    k_a = np.asarray(EFF['k_grid_acc']) / 2.0
    ax_budget.fill_between(k_i, lo_i, hi_i, color=COL_INET_BEST, alpha=0.18, zorder=3, lw=0)
    ax_budget.plot(k_i, m_i, color=COL_INET_BEST, lw=1.6, zorder=5, label='ImageNet best-k/model')
    ax_budget.fill_between(k_a, lo_a, hi_a, color=COL_ACC, alpha=0.18, zorder=3, lw=0)
    ax_budget.plot(k_a, m_a, color=COL_ACC, lw=1.6, zorder=5, label='Accentuated k/model')
    ax_budget.axvline(PER_MODEL_BUDGET, color='black', lw=0.5, ls=':', alpha=0.45, zorder=2)
    ax_budget.text(PER_MODEL_BUDGET, 0.50, f' k = {PER_MODEL_BUDGET}', transform=ax_budget.get_xaxis_transform(),
                   fontsize=FS_LEG, fontfamily=FONT_FAMILY, color='#444', va='center', ha='left')
    ax_budget.set_xscale('log')
    ax_budget.set_xlabel('Per-model image budget (k/model)', fontsize=FS_SUB, fontfamily=FONT_FAMILY)
    ax_budget.set_ylabel('Mean |A $-$ B|', fontsize=FS_SUB, fontfamily=FONT_FAMILY)
    ax_budget.tick_params(labelsize=FS_TICK)
    ax_budget.set_title('Effect of image budget', pad=5, fontsize=FS_SUB, fontweight='bold')
    ax_budget.legend(loc='upper right', fontsize=FS_LEG - 1, frameon=False, handlelength=1.4, handletextpad=0.5)

    # ============================ panel letters =================================
    fig.canvas.draw()
    # 6 letters: b spans both violin panels (b + pooled benchmark); c spans both scatter planes.
    # c (square scatter, centered) and d (top of the f/g/h stack) share the bottom-row top baseline.
    cd_y = outer[1].get_position(fig).y1        # true bottom-row top -> c and d letters share it
    letter_y = {'c': cd_y, 'd': cd_y}
    for ax, letter in [(ax_slope, 'a'), (ax_smsc, 'b'), (ax_nat, 'c'),
                       (ax_dist, 'd'), (ax_combos, 'e'), (ax_budget, 'f')]:
        bb = ax.get_position()
        fig.text(max(0.004, bb.x0 - 0.040), letter_y.get(letter, bb.y1) + 0.012, letter,
                 fontsize=FS_PANEL, fontfamily=FONT_FAMILY, fontweight='bold', va='bottom', ha='left')

    # default preset (rn50) -> plain benchmarking.png; the clip variant keeps its tag
    base = MAIN_FIGURES[6] if PRESET == 'rn50' else f'{MAIN_FIGURES[6]}_{PRESET}'
    path = save_fig(fig, os.path.join(out_dir, base))
    plt.close(fig)
    print(f'Saved -> {path}')
    print(f'separation SD: ' + '  '.join(f'{c}={sep_sd[c]:.3f}' for c in COND))
    print(f'acc > ImageNet-best in {frac*100:.1f}% of combos')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    ap.add_argument('--preset', default='rn50', choices=sorted(PAIR_PRESETS),
                    help='worked-example model pair for panels d/e/f')
    args = ap.parse_args()
    main(**vars(args))
