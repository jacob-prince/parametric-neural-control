#!/usr/bin/env python3
"""Supp fig (flatness_regime_control) -- the gradient spectral summary (participation ratio
by default; CoV via --gradsum cv) predicts
parametric control WITHIN matched synthesis-regularization regimes.

Control for the objection that gradient concentration predicts control only because low-frequency-
concentrated gradients happen to suit the Fourier-domain synthesis optimizer. We stratify the 250
site-models by their SELECTED regularization regime (the (augmentation-noise, spectral-decay)
rung, five discrete levels) and ask whether concentration still predicts control within a fixed
regime. It does: matching on regularization barely attenuates the concentration->control
correlation, with the same (positive) sign in all five regimes (values printed at run time).

(a) Gradient spectral concentration (CV) vs control score, colored by regularization regime,
    per-regime OLS fits; (b) per-regime Pearson r (control score + slope) with the pooled and
    within-regime correlations marked. Self-contained take8: control_table + load_gradient_freq
    + hp_tuning csv.
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

from pnc import paths
from pnc.manifest import output_name
from pnc.preproc import loader as L
from pnc.utils import apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig
apply_figure_style()
STEM = output_name('flatness_regime_control')
PRMODE = True                                              # gradsum 'pr' (default) | 'cv' variant (tag _cv); set in main()
PREPROC_DATA = str(paths.preprocessed_data())

FMIN, FMAX = 1, 112
FS_TITLE, FS_AX, FS_TICK, FS_ANN, FS_LEG, FS_LET = 7.5, 7, 6, 6, 5.5, 10


def _cv(p):
    P = np.clip(np.asarray(p, float)[FMIN:FMAX], 1e-30, None)
    if PRMODE:                                             # participation ratio over the same band
        s2 = float((P ** 2).sum())
        return float(P.sum()) ** 2 / s2 if s2 > 0 else np.nan
    return float(P.std() / P.mean())                       # radial-Fourier coefficient of variation


def load():
    ct = L.control_table()[['monkey', 'unit', 'model', 'control_r', 'control_slope']].copy()
    g = L.load_gradient_freq()
    fl = pd.DataFrame([dict(monkey=r['monkey'], unit=int(r['unit']), model=r['model'],
                            cv=_cv(r['profile_mean'])) for r in g['gradients']])
    hp = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, 'sup_hyperparam_hp_tuning_selected.csv'),
                                   hint='python scripts/preprocessing/build_hp_tuning_cache.py')).rename(columns={'channel': 'unit'})
    m = ct.merge(fl, on=['monkey', 'unit', 'model']).merge(
        hp[['monkey', 'model', 'unit', 'noise', 'decay']], on=['monkey', 'model', 'unit'])
    m['regime'] = list(zip(m.noise, m.decay))
    return m


def within_regime_r(m, out):
    """Site-level partial correlation of concentration with control, controlling for regularization
    regime (regime means removed from both variables). p uses df = n - #regimes - 1 -- the correct
    df once the five regime means are partialled out. Returns (r, p)."""
    d = m.copy()
    d['f_c'] = d.cv - d.groupby('regime')['cv'].transform('mean')
    d['o_c'] = d[out] - d.groupby('regime')[out].transform('mean')
    r = stats.pearsonr(d.f_c, d.o_c)[0]
    df = len(d) - d.regime.nunique() - 1
    t = r * np.sqrt(df / max(1.0 - r * r, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), df))


def _pstr(p):
    return '$p$ < 0.001' if p < 1e-3 else f'$p$ = {p:.3f}'


def main(out_dir, gradsum='pr'):
    global PRMODE
    PRMODE = gradsum == 'pr'
    m = load()
    regimes = sorted(m.regime.unique(), key=lambda t: t[1])          # order by spectral-decay alpha
    cmap = plt.colormaps['viridis']
    rc = {reg: cmap(x) for reg, x in zip(regimes, np.linspace(0.08, 0.88, len(regimes)))}

    raw_r = {o: stats.pearsonr(m.cv, m[o])[0] for o in ('control_r', 'control_slope')}
    win_r = {o: within_regime_r(m, o) for o in ('control_r', 'control_slope')}
    per = {o: {reg: stats.pearsonr(s.cv, s[o]) for reg, s in m.groupby('regime')}
           for o in ('control_r', 'control_slope')}

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 82 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], left=0.075, right=0.985,
                  top=0.86, bottom=0.145, wspace=0.30)

    # ---- panel a: flatness vs control score, colored by regime + per-regime fits ----
    axa = fig.add_subplot(gs[0, 0])
    for reg in regimes:
        s = m[m.regime == reg]
        axa.scatter(s.cv, s.control_r, s=11, color=rc[reg], edgecolors='white',
                    linewidths=0.3, alpha=0.9, zorder=3)
        if len(s) >= 8:
            xr = np.array([s.cv.min(), s.cv.max()])
            lr = stats.linregress(s.cv, s.control_r)
            axa.plot(xr, lr.slope * xr + lr.intercept, color=rc[reg], lw=1.6, zorder=4)
    axa.axhline(0, color='#cccccc', lw=0.6, zorder=1)
    axa.set_xlabel('Gradient spectral participation ratio (PR)' if PRMODE else
                   'Gradient spectral concentration (CV)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    axa.set_ylabel('Control score (Pearson $r$)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    axa.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    axa.set_title(('Participation ratio' if PRMODE else 'Concentration') +
                  ' vs control, by regularization regime', fontsize=FS_TITLE,
                  fontfamily=FONT_FAMILY, pad=4)
    ann_x, ann_ha = (0.03, 'left') if PRMODE else (0.97, 'right')   # PR mirrors x: free corner flips
    axa.text(ann_x, 0.04,
             f'site-level Pearson $r$  ($n$ = {len(m)} site$\\times$model)\n'
             f'pooled = {raw_r["control_r"]:+.2f}\n'
             f'within-regime = {win_r["control_r"][0]:+.2f}  ({_pstr(win_r["control_r"][1])})',
             transform=axa.transAxes, ha=ann_ha, va='bottom', fontsize=FS_ANN, fontfamily=FONT_FAMILY,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5, alpha=0.92))

    # ---- panel b: per-regime r (score + slope) with pooled / raw reference lines ----
    axb = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(regimes))
    axb.axhline(0, color='#cccccc', lw=0.6, zorder=1)
    for o, mk, off in [('control_r', 'o', -0.09), ('control_slope', 's', 0.09)]:
        for i, reg in enumerate(regimes):
            r_, p_ = per[o][reg]                                   # per-regime site-level Pearson (r, p)
            axb.scatter(xs[i] + off, r_, marker=mk, s=32,
                        facecolor=rc[reg] if p_ < 0.05 else 'white',
                        edgecolors='black' if p_ < 0.05 else rc[reg],
                        linewidths=0.9, zorder=5)
    axb.axhline(win_r['control_r'][0], color='#333333', ls='--', lw=1.0, zorder=2)
    axb.axhline(raw_r['control_r'], color='#999999', ls=':', lw=1.0, zorder=2)
    axb.text(len(regimes) - 0.5, win_r['control_r'][0], ' within-regime', ha='right', va='bottom',
             fontsize=FS_ANN, color='#333333', fontfamily=FONT_FAMILY)
    axb.text(len(regimes) - 0.5, raw_r['control_r'], ' pooled', ha='right', va='bottom',
             fontsize=FS_ANN, color='#999999', fontfamily=FONT_FAMILY)
    axb.set_xticks(xs)
    axb.set_xticklabels([f'{reg[0]:.2f}/\n{reg[1]:.1f}' for reg in regimes], fontsize=FS_TICK)
    axb.set_xlabel('regularization regime (noise / $\\alpha$)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    axb.set_ylabel(('PR' if PRMODE else 'concentration') + '–control $r$  (site-level)',
                   fontsize=FS_AX, fontfamily=FONT_FAMILY)
    axb.set_xlim(-0.5, len(regimes) - 0.5)
    axb.set_ylim((-0.80, 0.05) if PRMODE else (-0.05, 0.80))        # PR flips every correlation's sign
    axb.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    axb.set_title('Correlation within regularization regimes', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=4)
    # two boxed keys side by side in the clear bottom band (above y = 0): regime colors (also the
    # color key for panel a) on the left, marker/significance convention on the right. n = site count.
    reg_h = [Line2D([0], [0], marker='o', ls='', mfc=rc[reg], mec='white', mew=0.3, ms=4.5,
                    label=f'{reg[0]:.2f} / {reg[1]:.1f}  ($n$={(m.regime == reg).sum()})') for reg in regimes]
    leg_loc, leg_y = ('upper left', 0.97) if PRMODE else ('lower left', 0.03)   # clear band flips with sign
    leg_reg = axb.legend(handles=reg_h, loc=leg_loc, bbox_to_anchor=(0.0, leg_y), frameon=True,
                         fontsize=FS_LEG - 0.9, handletextpad=0.2, labelspacing=0.18, borderpad=0.3)
    leg_reg.get_frame().set(facecolor='white', edgecolor='#cccccc', linewidth=0.5, alpha=0.92)
    axb.add_artist(leg_reg)
    mk_h = [Line2D([0], [0], marker='o', ls='', mfc='white', mec='#444', mew=1.0, ms=5, label='control $r$'),
            Line2D([0], [0], marker='s', ls='', mfc='white', mec='#444', mew=1.0, ms=5, label='control slope'),
            Line2D([0], [0], marker='o', ls='', mfc='#444', mec='#444', mew=1.0, ms=5, label='closed: $p <$ 0.05')]
    leg_mk = axb.legend(handles=mk_h, loc=leg_loc, bbox_to_anchor=(0.40, leg_y), frameon=True,
                        fontsize=FS_LEG - 0.9, handletextpad=0.2, labelspacing=0.18, borderpad=0.3)
    leg_mk.get_frame().set(facecolor='white', edgecolor='#cccccc', linewidth=0.5, alpha=0.92)

    fig.text(0.02, 0.965, 'a', fontsize=FS_LET, fontweight='bold', fontfamily=FONT_FAMILY, va='top')
    fig.text(0.615, 0.965, 'b', fontsize=FS_LET, fontweight='bold', fontfamily=FONT_FAMILY, va='top')

    out = save_fig(fig, os.path.join(out_dir, STEM + ('' if PRMODE else '_cv')))
    plt.close(fig)
    print('saved ->', out)
    print('pooled:', {k: round(v, 3) for k, v in raw_r.items()},
          '| within-regime (r, p):', {k: (round(v[0], 3), round(v[1], 6)) for k, v in win_r.items()})
    for o in ('control_r', 'control_slope'):
        print(' ', o, {f'{r[0]:.2f}/{r[1]:.1f}': (round(v[0], 3), round(v[1], 4), int((m.regime == r).sum()))
                       for r, v in per[o].items()})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    ap.add_argument('--gradsum', choices=['pr', 'cv'], default='pr',
                    help="gradient spectral summary: 'pr' (default) | 'cv' (filename tag _cv)")
    main(**vars(ap.parse_args()))
