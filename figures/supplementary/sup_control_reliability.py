#!/usr/bin/env python3
r"""Supp fig (control_reliability) - control-phase reliability and ceiling-normalized control.

Control-phase noise ceiling = the canonical NSD estimator on the accentuated-stimulus
group (the maximum Pearson r attainable given the channel's trial reliability on that
stim set). The trial noise is the pooled SAME-DAY repeat variance of the encoding anchors
present in the control session. Ceilings are defined for red/paul/venus (15 site-models
per model); leap/three0 lack a control-session NSD ceiling.

  a  Repeat-count distribution across all accentuated stimuli (all sites).
  b  Control-phase noise ceiling by encoding model: per-site dots colored by animal,
     per-model medians.
  c  Ceiling-normalized control (control r / noise ceiling) per model, sorted by mean
     (best to worst); each dot = one site-model with a valid ceiling, Untrained faint.
     Sites with nc_r <= 0.1 are excluded from the normalization.

Reads loader.control_table() / ceilings_df / the brain cache; S-number and output stem
from the manifest.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR,
                       save_fig, MODEL_COLORS, MODEL_SHORT_NAMES, MODEL_ORDER, ROBUST_MODELS,
                       MONKEY_COLORS, get_model_color)
apply_figure_style()

MODEL_SHORT_NAMES = dict(MODEL_SHORT_NAMES)
MODEL_SHORT_NAMES['AlexNet_training_seed_01'] = 'Untrained'   # relabel the untrained control

_M = _fig('control_reliability'); STEM = output_name('control_reliability')

EX_MK, EX_UNIT, EX_MODEL = 'red', 9, 'resnet50'
CEIL_MONKEYS = ['red', 'paul', 'venus']               # control NSD ceiling defined here
MONKEY_TAG = {'red': 'R', 'paul': 'P', 'venus': 'V'}
MONKEY_REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4'}
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 10, 7, 6.5, 5.5, 5.5
YLABEL = 'Noise ceiling (Pearson r)'


# ---- control-session trial-level data + canonical NSD ceiling ---------------
def _acc_trials(mk, unit, model):
    """{accentuated stimulus_name: [trial z]} for one (mk, unit, model) in the control session,
    plus the (stim, day)-grouped anchor trials used to estimate pure trial noise."""
    b = L.load_brain(mk); ui = list(b['units']).index(unit); c = b['control']
    tsn = np.asarray(c['trial_stim']); days = np.asarray(c['trial_day'])
    tz = np.asarray(c['trial_z'])[:, ui]; tk = np.asarray(c['trial_kind'])
    acc_set = set(n for n, k in zip(tsn, tk)
                  if k == 'accentuated' and n.startswith(model + '_RidgeCV')
                  and f'_unit_{unit}_' in n)
    by = defaultdict(list)
    for i, s in enumerate(tsn):
        if s in acc_set and np.isfinite(tz[i]):
            by[s].append(float(tz[i]))
    return dict(by), tz, tsn, days, tk


def _anchor_trial_noise(zc, tsn, days, anchor_set, min_groups=5):
    grp = defaultdict(list)
    for i, s in enumerate(tsn):
        if s in anchor_set and np.isfinite(zc[i]):
            grp[(s, days[i])].append(zc[i])
    wd = [np.var(v, ddof=1) for v in grp.values() if len(v) >= 2]
    return float(np.mean(wd)) if len(wd) >= min_groups else np.nan


def _nsd_ceiling(by, noisevar, min_stim=8):
    means, reps = [], []
    for v in by.values():
        if len(v) >= 2:
            means.append(np.mean(v)); reps.append(len(v))
    if len(means) < min_stim or not (np.isfinite(noisevar) and noisevar > 0):
        return np.nan
    datavar = np.var(means, ddof=1); inv_n = np.mean(1.0 / np.array(reps, float))
    if not (datavar > 0):
        return np.nan
    sig = max(datavar - noisevar * inv_n, 0.0)
    ncsnr = np.sqrt(sig) / np.sqrt(noisevar)
    return float(np.sqrt(ncsnr ** 2 / (ncsnr ** 2 + inv_n)))


def example_ceiling():
    """Reproduce the stored control NSD ceiling for the Fig 3 example, self-contained."""
    by, tz, tsn, days, tk = _acc_trials(EX_MK, EX_UNIT, EX_MODEL)
    enc_set = set(tsn[tk == 'calibration'].tolist())
    tnoise = _anchor_trial_noise(tz, tsn, days, enc_set)
    return _nsd_ceiling(by, tnoise), by


def ceiling_table():
    """Long table of control-phase nc_r for every defined site-model (150 rows)."""
    frames = []
    for mk in CEIL_MONKEYS:
        d = L.ceilings_df(mk).copy(); d['monkey'] = mk
        frames.append(d[['monkey', 'unit', 'model', 'nc_r']])
    return pd.concat(frames, ignore_index=True)


def all_repeat_counts():
    """Repeat count per accentuated stimulus, pooled across every site-model (response-free:
    how many times each accentuated image was shown in each control session)."""
    from collections import Counter
    counts = []
    for mk in CEIL_MONKEYS:
        c = L.load_brain(mk)['control']
        tsn = np.asarray(c['trial_stim']); tk = np.asarray(c['trial_kind'])
        counts.extend(Counter(tsn[tk == 'accentuated'].tolist()).values())
    return np.asarray(counts, int)


# ---- panels -----------------------------------------------------------------
def panel_a(ax, by, ex_nc):
    """Ranked per-stimulus mean control response +/- trial SEM (example channel)."""
    means, sems = [], []
    for v in by.values():
        vv = np.asarray(v, float)
        means.append(vv.mean())
        sems.append(vv.std(ddof=1) / np.sqrt(len(vv)) if len(vv) >= 2 else np.nan)
    means = np.asarray(means); sems = np.asarray(sems)
    order = np.argsort(means)
    xs = np.arange(len(means))
    col = MONKEY_COLORS[EX_MK]
    ax.fill_between(xs, means[order] - sems[order], means[order] + sems[order],
                    color=col, alpha=0.22, lw=0, zorder=2)
    ax.plot(xs, means[order], color=col, lw=1.0, zorder=3)
    ax.axhline(0, color='#bbb', lw=0.4, zorder=0)
    ax.set_xlim(-1, len(means)); ax.set_xlabel('Accentuated stimulus (ranked)', fontsize=FS_AX)
    ax.set_ylabel('Mean control response (z)', fontsize=FS_AX)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.03, 0.97,
            f'{len(means)} stimuli\nmean {np.nanmean([len(v) for v in by.values()]):.1f} repeats\n'
            f'noise ceiling r = {ex_nc:.2f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
            fontfamily=FONT_FAMILY, linespacing=1.3)
    ax.text(0.97, 0.05, r'shading $\pm$ trial SEM', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=FS_ANNOT, color='#666', fontfamily=FONT_FAMILY)


def panel_b(ax, reps):
    """Repeat-count distribution pooled across ALL accentuation stimuli (every site-model)."""
    reps = np.asarray(reps, int)
    lo, hi = int(reps.min()), int(reps.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    ax.hist(reps, bins=bins, color='#3B5B7A', edgecolor='white', linewidth=0.5, zorder=3)
    ax.axvline(reps.mean(), color='#111', ls='--', lw=0.9, zorder=4)
    ax.text(reps.mean(), ax.get_ylim()[1], f' mean {reps.mean():.1f}', ha='left',
            va='top', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY)
    ax.set_xlabel('Repeats per stimulus', fontsize=FS_AX)
    ax.set_ylabel('Accentuated stimuli (count)', fontsize=FS_AX)
    ax.set_xticks(np.arange(lo, hi + 1))
    ax.tick_params(labelsize=FS_TICK)


def panel_c(ax, tab):
    """Per-model control-phase ceiling across sites; animal-colored swarm + per-model medians.
    Fig 3 example (R aIT u9, ResNet50) highlighted with an open ring. Returns per-model medians."""
    models = list(MODEL_ORDER)
    xs = np.arange(len(models))
    rng = np.random.default_rng(42)
    medians = {}
    for i, mdl in enumerate(models):
        sub = tab[(tab.model == mdl) & tab.nc_r.notna()]
        vals = sub.nc_r.to_numpy()
        cols = [MONKEY_COLORS[mk] for mk in sub.monkey]
        jit = rng.uniform(-0.26, 0.26, len(vals))
        ax.scatter(xs[i] + jit, vals, s=13, c=cols, alpha=0.8,
                   edgecolors='white', linewidths=0.35, zorder=4)
        med = float(np.median(vals)); medians[mdl] = med
        ax.plot([xs[i] - 0.36, xs[i] + 0.36], [med, med], color='black', lw=2.4,
                solid_capstyle='round', zorder=6)
        ax.plot([xs[i] - 0.31, xs[i] + 0.31], [med, med], color=get_model_color(mdl),
                lw=1.4, solid_capstyle='round', zorder=7)
        if mdl in ROBUST_MODELS:
            ax.scatter([xs[i]], [1.0], marker='v', s=16, color=get_model_color(mdl),
                       edgecolors='black', linewidths=0.4, zorder=8, clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_SHORT_NAMES.get(m, m) for m in models],
                       rotation=40, ha='right', fontsize=FS_TICK)
    for t, mdl in zip(ax.get_xticklabels(), models):
        t.set_color(get_model_color(mdl))
        if mdl in ROBUST_MODELS:
            t.set_fontweight('bold')
    ax.set_xlim(-0.6, len(models) - 0.4); ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel(YLABEL, fontsize=FS_AX)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='x', length=0)

    handles = [Line2D([0], [0], marker='o', ls='', mfc=MONKEY_COLORS[m], mec='white', mew=0.4,
                      ms=4.5, label=f'{MONKEY_TAG[m]} \u2014 {MONKEY_REGION[m]}') for m in CEIL_MONKEYS]
    lg = ax.legend(handles=handles, fontsize=FS_ANNOT, frameon=True, loc='lower right',
                   ncol=1, handletextpad=0.35, labelspacing=0.3, borderpad=0.5)
    lg.get_frame().set(facecolor='white', edgecolor='#cccccc', linewidth=0.6, alpha=0.95)
    return medians


UNTRAINED = 'AlexNet_training_seed_01'
NC_FLOOR = 0.1                            # avoid divide blow-ups in ceiling normalization


def panel_d(ax):
    """Ceiling-normalized control (control r / nc_r) per model, sorted by mean; each dot
    = one site-model with a valid ceiling (red/paul/venus); Untrained faint."""
    rng = np.random.default_rng(7)
    d = L.control_table()
    dn = d[d.nc_r.notna() & (d.nc_r > NC_FLOOR)].copy()
    dn['norm'] = dn.control_r / dn.nc_r
    ranked = dn.groupby('model')['norm'].mean().sort_values(ascending=False)
    means, ticks = {}, []
    for x, m in enumerate(ranked.index):
        faint = (m == UNTRAINED)
        col = get_model_color(m)
        vals = dn[dn.model == m].norm.to_numpy()
        jit = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x + jit, vals, s=13, color=col, alpha=0.28 if faint else 0.8,
                   edgecolors='white', linewidths=0.35, zorder=4)
        mu = float(np.mean(vals)); means[m] = mu
        ax.plot([x - 0.30, x + 0.30], [mu, mu], color='#cccccc' if faint else '#111',
                lw=1.4 if faint else 2.0, solid_capstyle='round', zorder=5)
        ax.text(x, mu + 0.03, f'{mu:.2f}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.7, fontfamily=FONT_FAMILY, zorder=6,
                color='#999999' if faint else 'black')
        ticks.append((MODEL_SHORT_NAMES.get(m, m), '#999999' if faint else col))
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels([t for t, _ in ticks], fontsize=FS_TICK - 0.5, rotation=30, ha='right')
    for t, (_, c) in zip(ax.get_xticklabels(), ticks):
        t.set_color(c)
    ax.set_xlim(-0.6, len(ranked) - 0.4)
    ax.tick_params(axis='y', labelsize=FS_TICK); ax.tick_params(axis='x', length=0)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(1.0, color='#999', ls=':', lw=0.8, zorder=1)
    ax.text(-0.55, 1.005, 'ceiling', ha='left', va='bottom', fontsize=FS_ANNOT - 0.5,
            color='#999', fontfamily=FONT_FAMILY)
    ax.set_ylabel('Control $r$ / noise ceiling', fontsize=FS_AX)
    return means


def main(out_dir):
    tab = ceiling_table()
    reps = all_repeat_counts()
    print(f'pooled accentuation repeats (n={len(reps)} stimuli across {CEIL_MONKEYS}): '
          f'mean {reps.mean():.2f}, range {reps.min()}-{reps.max()}')

    from matplotlib.gridspec import GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 118 * MM), facecolor=FIG_FACECOLOR)
    outer = GridSpec(2, 1, figure=fig, left=0.075, right=0.985, top=0.925, bottom=0.105,
                     hspace=0.52)
    gsTop = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30,
                                    width_ratios=[0.80, 1.45])
    axa = fig.add_subplot(gsTop[0, 0]); axb = fig.add_subplot(gsTop[0, 1])
    axc = fig.add_subplot(outer[1])
    panel_b(axa, reps); medians = panel_c(axb, tab)
    normmeans = panel_d(axc)
    for ax in (axa, axb, axc):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.canvas.draw()
    panels = [(axa, 'a', 'Repeats per stimulus (all sites)'),
              (axb, 'b', 'Control-phase noise ceiling by model'),
              (axc, 'c', 'Ceiling-normalized control by model')]
    for ax, letter, title in panels:
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.012, title, ha='center', va='bottom',
                 fontsize=FS_TITLE, fontfamily=FONT_FAMILY)
        fig.text(bb.x0 - 0.055, bb.y1 + 0.030, letter, ha='left', va='top', fontsize=FS_LET,
                 fontweight='bold', fontfamily=FONT_FAMILY)

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    print('per-model median control nc_r:')
    for m in MODEL_ORDER:
        print(f'  {MODEL_SHORT_NAMES[m]:12s} {medians[m]:.4f}')
    print('per-model mean ceiling-normalized control: ' +
          ', '.join(f'{MODEL_SHORT_NAMES.get(m, m)}={v:.3f}' for m, v in normmeans.items()))
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
