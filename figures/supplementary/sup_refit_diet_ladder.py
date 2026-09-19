#!/usr/bin/env python3
"""Supp fig (refit_diet_ladder) -- the main figure's gradient-spectra analyses redone across
the encoding TRAINING-DIET ladder: does fitting the readout on more (or cleaner) data change the
gradient spectra, or how well their spectral participation ratio (PR) predicts neural control?

Diets (per site-model pair; n fitting images stamped per column, means over the 250 pairs):
  d0 calibration only (the synthesis-time readouts, = main figure)     ~780 naturals
  d1 + own accentuations (this model x this site)                      + ~110
  d2 + site-targeted accentuations (all models, this site)             + ~1,100
  d3 + everything (all ~5.5k accentuated pairs; the pooled refit)      + ~5,475
  d4 everything MINUS this pair's own control-scored images            + ~5,365 (zero circularity)

Rows: (top) per-model mean held-out gradient spectra per diet; (middle) gradient spectral PR vs
site-residualized control slope per diet, exact main-figure recipe (site dots + model means/95% CI,
dashed site fit, bold model fit; held-out 100-NSD-image profiles; 9-trained residual reference);
(bottom-left) summary: model- and site-level r across diets for the 10/9/7 model subsets;
(bottom-right) readout predictivity along the ladder (natural test r + own-accentuation r, OOF
when the diet contains them, direct otherwise).

Reads  preproc_data/sup_diet_refit.csv (cluster-computed diet refits) +
cluster/outputs_from_cluster/{sup_diet_gradfreq_d1,_d2,_d3,_d4, fig5_heldout_gradfreq (d0)}.
"""
import os, glob, re, pickle, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pnc import paths
from pnc.manifest import output_name
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_COLORS, MODEL_ORDER, MODEL_SHORT_NAMES)
from pnc.preproc import loader as L
apply_figure_style()

STEM = output_name('refit_diet_ladder')
CLUSTER_OUT = str(paths.cluster_outputs())
HO_ORIG = os.path.join(CLUSTER_OUT, 'fig5_heldout_gradfreq')

MDISP = {**MODEL_SHORT_NAMES, 'AlexNet_training_seed_01': 'Untrained'}
FS_TTL, FS_AX, FS_TK, FS_AN, FS_LG, FS_LET = 6.8, 6.0, 5.2, 5.0, 5.2, 10
FMIN, FMAX = 1, 112
UNTR = 'AlexNet_training_seed_01'
EXTREMES = ['resnet50_robust', 'clipag_vitb32', UNTR]
DIETS = ['d0', 'd1', 'd2', 'd3', 'd4']
DIET_TITLE = {'d0': 'calibration only', 'd1': '+ own accent.', 'd2': '+ site-targeted',
              'd3': '+ everything', 'd4': 'everything $-$ own'}
GRADDIR = {'d0': HO_ORIG, 'd1': os.path.join(CLUSTER_OUT, 'sup_diet_gradfreq_d1'),
           'd2': os.path.join(CLUSTER_OUT, 'sup_diet_gradfreq_d2'),
           'd3': os.path.join(CLUSTER_OUT, 'sup_diet_gradfreq_d3'),
           'd4': os.path.join(CLUSTER_OUT, 'sup_diet_gradfreq_d4')}
SUBSET_COL = {10: '#5E35B1', 9: '#00838F', 7: '#C2185B'}       # main-figure subset label colors


def spectral_pr(p):
    """Participation ratio of the radial power spectrum over the main-figure band: (sum P)^2 / sum P^2.
    High PR = power spread across many frequency bins (flat spectrum); low PR = concentrated.
    Monotone in CoV (PR = N / (1 + CoV^2)), an exact reparameterization of spectral CoV."""
    P = np.asarray(p, float)[FMIN:FMAX]
    s2 = float((P ** 2).sum())
    return float(P.sum() ** 2 / s2) if s2 > 0 else np.nan


def normalize_row(p):
    p = np.asarray(p, float); s = p[FMIN:FMAX].sum()
    return p / s if s > 0 else p


def load_profiles(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(paths.require(d), '*.pkl'))):
        mm = re.match(r'(.+?)_unit_(\d+)_model_(.+)_grad_maps_freq_profiles\.pkl', os.path.basename(f))
        if not mm:
            continue
        key = (mm.group(1).split('_')[0], int(mm.group(2)), mm.group(3))
        with open(f, 'rb') as fh:
            out[key] = np.asarray(pickle.load(fh)['profiles'], float).mean(0)
    return out


def site_resid(df, col):
    """col minus its per-site mean over the 9 trained models (main-figure 9-trained convention)."""
    mean = df[df.model != UNTR].groupby(['monkey', 'unit'])[col].mean()
    return df[col].values - mean.reindex(pd.MultiIndex.from_frame(df[['monkey', 'unit']])).values


def pr_scatter(ax, sub, yr, show_y):
    """main-figure scatter, compact: site dots + model means (95% CI), dashed site / bold model fits."""
    for m in sub.model.unique():
        s = sub[sub.model == m]
        ax.scatter(s.pr, s.y, s=4, color=MODEL_COLORS[m], marker='D' if m in EXTREMES else 'o',
                   alpha=0.45, lw=0, zorder=2)
    pm = sub.groupby('model')[['pr', 'y']].mean()
    psem = sub.groupby('model')[['pr', 'y']].sem(); pn = sub.groupby('model').size()
    for m in pm.index:
        tc = stats.t.ppf(0.975, max(int(pn[m]) - 1, 1))
        ax.errorbar(pm.loc[m, 'pr'], pm.loc[m, 'y'], xerr=tc * psem.loc[m, 'pr'], yerr=tc * psem.loc[m, 'y'],
                    fmt='D' if m in EXTREMES else 'o', ms=3.2, mfc=MODEL_COLORS[m], mec='k', mew=0.35,
                    ecolor=MODEL_COLORS[m], elinewidth=0.5, capsize=1.0, capthick=0.5, zorder=5)
    xr = (sub.pr.min() - 0.06 * sub.pr.max(), sub.pr.max() * 1.06)
    xs = np.linspace(*xr, 40)
    bs, as_ = np.polyfit(sub.pr, sub.y, 1); ax.plot(xs, bs * xs + as_, '--', c='0.45', lw=0.8, dashes=(4, 2), zorder=4)
    bm, am = np.polyfit(pm.pr, pm.y, 1); ax.plot(xs, bm * xs + am, '-', c='0.12', lw=1.4, zorder=4)
    mr = stats.pearsonr(pm.pr, pm.y)[0]; sr = stats.pearsonr(sub.pr, sub.y)[0]
    ax.annotate(f'model r = {mr:+.2f}\nsite r = {sr:+.2f}', (0.96, 0.96), xycoords='axes fraction',
                ha='right', va='top', fontsize=FS_AN, color='0.2',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.5, alpha=0.92))
    ax.set_xlim(*xr); ax.set_ylim(*yr)
    ax.set_xlabel('gradient PR', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    if show_y:
        ax.set_ylabel('control slope\n(site resid.)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=FS_TK, length=2.5, width=0.6)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    return mr, sr


def main(out_dir):
    dr = pd.read_csv(paths.require(os.path.join(str(paths.preprocessed_data()), 'sup_diet_refit.csv'),
                                   hint='frozen input; python scripts/preprocessing/build_all.py'))
    ct = L.control_table()[['monkey', 'unit', 'model', 'control_slope']].copy()
    ct['y'] = site_resid(ct, 'control_slope')

    prof = {d: load_profiles(GRADDIR[d]) for d in DIETS}
    n_by_diet = dr.groupby('diet')[['n_fit_nat', 'n_fit_acc']].mean()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 150 * MM), facecolor=FIG_FACECOLOR)
    left, right = 0.065, 0.99
    gapx = 0.022
    w5 = (right - left - 4 * gapx) / 5
    xs5 = [left + i * (w5 + gapx) for i in range(5)]
    y_spec, h_spec = 0.735, 0.20
    y_scat, h_scat = 0.435, 0.20

    # ---- row 1: per-model mean spectra per diet ----
    x = np.arange(158)[FMIN:FMAX]
    for c, d in enumerate(DIETS):
        ax = fig.add_axes([xs5[c], y_spec, w5, h_spec])
        for m in MODEL_ORDER:
            profs = [normalize_row(v) for k, v in prof[d].items() if k[2] == m]
            if not profs:
                continue
            mean = np.mean(profs, axis=0)
            is_rob = m in ('resnet50_robust', 'clipag_vitb32')
            ax.plot(x, mean[FMIN:FMAX], color=MODEL_COLORS[m], lw=1.3 if is_rob else 0.8,
                    alpha=1.0 if is_rob else 0.85, zorder=9 if is_rob else 5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1.5e-4, 0.2)
        n = n_by_diet.loc[d]
        ax.set_title(f'{DIET_TITLE[d]}\n' + f'{n.n_fit_nat:.0f} nat + {n.n_fit_acc:.0f} acc',
                     fontsize=FS_TTL, fontfamily=FONT_FAMILY, pad=2.5)
        ax.set_xlabel('spatial freq. (cyc/img)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
        if c == 0:
            ax.set_ylabel('normalized radial power', fontsize=FS_AX, fontfamily=FONT_FAMILY)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=FS_TK, length=2.5, width=0.6, which='both')
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)

    # ---- row 2: PR -> control per diet (fig6 recipe) ----
    tabs, stats_rows = {}, []
    for d in DIETS:
        t = ct.copy()
        t['pr'] = [spectral_pr(prof[d][(mk, int(u), m)]) if (mk, int(u), m) in prof[d] else np.nan
                   for mk, u, m in zip(t.monkey, t.unit, t.model)]
        tabs[d] = t.dropna(subset=['pr', 'y'])
    yr = (min(t.y.min() for t in tabs.values()) - 0.05, max(t.y.max() for t in tabs.values()) + 0.05)
    for c, d in enumerate(DIETS):
        ax = fig.add_axes([xs5[c], y_scat, w5, h_scat])
        pr_scatter(ax, tabs[d], yr, show_y=(c == 0))
        for k in (10, 9, 7):
            ms = ([m for m in MODEL_ORDER if m != UNTR] if k == 9
                  else [m for m in MODEL_ORDER if m not in EXTREMES] if k == 7 else MODEL_ORDER)
            s = tabs[d][tabs[d].model.isin(ms)]
            pm = s.groupby('model')[['pr', 'y']].mean()
            stats_rows.append(dict(diet=d, subset=k,
                                   model_r=stats.pearsonr(pm.pr, pm.y)[0],
                                   site_r=stats.pearsonr(s.pr, s.y)[0]))
    sr = pd.DataFrame(stats_rows)

    # ---- row 3 left: subset r across diets ----
    y_bot, h_bot = 0.075, 0.24
    gap3 = 0.062                                   # wider than the 5-column rows: each panel keeps its y labels
    wbot = (right - left - 3 * gap3) / 4
    axr = {lev: fig.add_axes([left + i * (wbot + gap3), y_bot, wbot, h_bot])
           for i, lev in enumerate(['model', 'site'])}
    xd = np.arange(len(DIETS))
    for lev, ax in axr.items():
        for k in (10, 9, 7):
            v = [sr[(sr.diet == d) & (sr.subset == k)][f'{lev}_r'].iloc[0] for d in DIETS]
            ax.plot(xd, v, '-o', ms=3, lw=1.1, color=SUBSET_COL[k], label=f'{k} models')
        ax.axhline(0, color='0.75', lw=0.6, ls='--')
        ax.set_ylim(-1.06, 0.06)
        ax.set_xticks(xd); ax.set_xticklabels([DIET_TITLE[d] for d in DIETS], fontsize=FS_TK - 0.4,
                                              rotation=20, ha='right')
        ax.set_ylabel('r(PR, control slope)' if lev == 'model' else '', fontsize=FS_AX)
        ax.set_title(f'PR $\\rightarrow$ control, {lev} level', fontsize=FS_TTL, fontfamily=FONT_FAMILY, pad=2.5)
        ax.tick_params(labelsize=FS_TK, length=2.5, width=0.6)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    axr['model'].legend(fontsize=FS_LG, frameon=False, loc='upper right', handletextpad=0.4, borderpad=0.2)

    # ---- row 3 right: readout predictivity along the ladder ----
    axp = fig.add_axes([left + 2 * (wbot + gap3), y_bot, wbot, h_bot])
    g = dr.groupby('diet')[['r_nat_test', 'r_own_acc']].agg(['mean', 'sem'])
    for col, cc_, lab in [('r_nat_test', '#3B6FA0', 'natural test'), ('r_own_acc', '#B5651D', 'own accent.')]:
        ax = axp
        ax.errorbar(xd, [g.loc[d, (col, 'mean')] for d in DIETS],
                    yerr=[g.loc[d, (col, 'sem')] for d in DIETS],
                    fmt='-o', ms=3, lw=1.1, color=cc_, elinewidth=0.7, capsize=1.5, label=lab)
    axp.set_xticks(xd); axp.set_xticklabels([DIET_TITLE[d] for d in DIETS], fontsize=FS_TK - 0.4,
                                            rotation=20, ha='right')
    axp.set_ylabel('held-out r', fontsize=FS_AX)
    axp.set_title('Readout predictivity', fontsize=FS_TTL, fontfamily=FONT_FAMILY, pad=2.5)
    axp.tick_params(labelsize=FS_TK, length=2.5, width=0.6)
    for s_ in ('top', 'right'):
        axp.spines[s_].set_visible(False)
    axp.legend(fontsize=FS_LG, frameon=False, loc='lower right', handletextpad=0.4, borderpad=0.2)

    # ---- row 3 far right: model legend ----
    axl = fig.add_axes([left + 3 * (wbot + gap3), y_bot, wbot, h_bot]); axl.axis('off')
    handles = [Line2D([], [], marker='D' if m in EXTREMES else 'o', ls='none', color=MODEL_COLORS[m],
                      mec='k', mew=0.2, ms=3.6, label=MDISP[m]) for m in MODEL_ORDER]
    axl.legend(handles=handles, ncol=2, fontsize=FS_LG, frameon=False, loc='center left',
               handletextpad=0.25, columnspacing=0.8, labelspacing=0.5)

    for (xf, yf), ltr in zip([(xs5[0] - 0.048, y_spec + h_spec + 0.045),
                              (xs5[0] - 0.048, y_scat + h_scat + 0.045),
                              (left - 0.048, y_bot + h_bot + 0.045),
                              (left + 2 * (wbot + gap3) - 0.048, y_bot + h_bot + 0.045)], 'abcd'):
        fig.text(xf, yf, ltr, fontsize=FS_LET, fontweight='bold', fontfamily=FONT_FAMILY, va='top', ha='left')

    print(sr.pivot(index='diet', columns='subset', values='model_r').round(3).to_string())
    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig); print('saved ->', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    main(**vars(ap.parse_args()))
