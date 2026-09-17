#!/usr/bin/env python3
r"""Supp fig (adversarial_robustness) — attack-method invariance of the adversarial-
sensitivity measurements.

Compares the production PGD attack with single-step FGSM. Computation follows the
production analyses verbatim; each panel carries a one-line plain-language subtitle saying
what is plotted.

  a  FGSM vs PGD agreement: correlation of per-axis attack effects at each perturbation
     strength.
  b  attack effect vs perturbation strength, training-family means, both attack methods.
  c  correlation of sensitivity (log-eps AUC[0.125,16], the main-figure measure, computed
     for each attack) with control slope, per model subset, PGD vs FGSM.

Reads cluster PGD/FGSM outputs (cluster/outputs_from_cluster/{fig5_ext_heldout,
fig5_ext_heldout_fgsm}{,_lowextra}) + preproc_data/{sup_advrob_fgsm_vs_pgd_eps.csv,
fig5_heldout_table.csv}.
"""
import os, glob, argparse
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pnc import paths
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       ROBUST_MODELS, MODEL_ORDER, MODEL_SHORT_NAMES, get_model_color)
from pnc.manifest import output_name
apply_figure_style()
STEM = output_name('adversarial_robustness')
PREPROC_DATA = str(paths.preprocessed_data())
CLUSTER_OUT = str(paths.cluster_outputs())
FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANNOT, FS_SUB = 11, 7, 6.5, 5.5, 5.5, 5.0

ROBUST = list(ROBUST_MODELS); UNTR = 'AlexNet_training_seed_01'; EXTREMES = ROBUST + [UNTR]
KEY = ['model', 'monkey', 'channel']
C_PGD, C_FGSM = '#3274A1', '#E1812C'                 # PGD steel blue, FGSM burnt orange
SUBSET_COL = {10: '#5E35B1', 9: '#00838F', 7: '#C2185B'}   # subset label colours (main figure)
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
AUC_EPS_LO, AUC_EPS_HI = 0.125, 16.0                 # main-figure sensitivity integration window


def load_logauc(dirs, name):
    """Per-readout AUC of normalized swing over LOG2(eps) on the 100 held-out images —
    the main-figure adversarial-sensitivity measure, computed per attack method."""
    ho = pd.concat([pd.read_csv(p) for d in dirs
                    for p in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, d)),
                                                           'adv_robustness_*.csv')))
                    if 'smoke' not in p], ignore_index=True)
    ho = ho[ho.norm == 'linf'].copy()
    ho = ho[(ho.eps_255 >= AUC_EPS_LO - 1e-9) & (ho.eps_255 <= AUC_EPS_HI + 1e-9)]
    ho['nswing'] = (ho.adv_up - ho.adv_dn) / ho.range_q99q01
    g = ho.groupby(KEY + ['eps_255'])['nswing'].mean().reset_index()

    def _auc(d):
        d = d.sort_values('eps_255'); x = np.log2(d.eps_255.values)
        return trapz(d.nswing.values, x) / (x[-1] - x[0])
    return g.groupby(KEY).apply(_auc, include_groups=False).reset_index(name=name)


def swing_vs_eps(ext_dir):
    raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(paths.require(ext_dir), 'adv_robustness_*_*.csv')))
                     if 'smoke' not in p], ignore_index=True)
    raw = raw[raw.norm == 'linf'].copy(); raw['swing'] = (raw.adv_up - raw.adv_dn) / raw.range_q99q01
    grp = lambda m: 'Adv-trained' if m in ROBUST else ('Untrained' if m == UNTR else 'Conventional')
    raw['grp'] = raw.model.map(grp)
    return raw.groupby(['grp', 'eps_255'])['swing'].mean().reset_index()


def _spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _per_axis_swings(ext_dir):
    raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(paths.require(ext_dir), 'adv_robustness_*_*.csv')))
                     if 'smoke' not in p], ignore_index=True)
    raw = raw[raw.norm == 'linf'].copy()
    raw['nswing'] = (raw.adv_up - raw.adv_dn) / raw.range_q99q01
    return raw.groupby(['model', 'monkey', 'channel', 'eps_255'])['nswing'].mean().reset_index()


def panel_agreement(ax, er):
    # faint per-model agreement curves (25 axes each) under the pooled black curve
    p = _per_axis_swings(os.path.join(CLUSTER_OUT, 'fig5_ext_heldout'))
    f = _per_axis_swings(os.path.join(CLUSTER_OUT, 'fig5_ext_heldout_fgsm'))
    m = p.merge(f, on=['model', 'monkey', 'channel', 'eps_255'], suffixes=('_p', '_f'))
    m = m[m.eps_255.round(6).isin(er.eps_255.round(6))]
    within = {}
    for mdl in MODEL_ORDER:
        s = m[m.model == mdl]
        if s.empty:
            continue
        cur = (s.groupby('eps_255')
                .apply(lambda g: stats.pearsonr(g.nswing_p, g.nswing_f)[0], include_groups=False)
                .sort_index())
        within[mdl] = cur
        ax.plot(cur.index, cur.values, '-', lw=0.8, color=get_model_color(mdl), alpha=0.45, zorder=2)
    # mean of the within-model curves: the pooled black curve sits below it wherever the two
    # attacks disagree about BETWEEN-model structure (they re-rank models at mid/high eps)
    wm = pd.concat(within.values(), axis=1).mean(axis=1)
    ax.plot(wm.index, wm.values, '--s', color='#333333', lw=1.2, ms=3.0, mfc='white', zorder=3)
    ax.plot(er.eps_255, er.site_r, '-o', color='#333333', lw=1.7, ms=4.5, zorder=4)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'Perturbation strength $\epsilon$ (/255)', fontsize=FS_AX)
    ax.set_ylabel('FGSM vs PGD agreement ($r$)', fontsize=FS_AX)
    SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTR] = 'Untrained'
    handles = ([Line2D([0], [0], color='#333333', lw=1.7, marker='o', ms=3.5,
                       label='All axes (pooled)'),
                Line2D([0], [0], color='#333333', lw=1.2, ls='--', marker='s', ms=3.0,
                       mfc='white', label='Within-model mean')]
               + [Line2D([0], [0], color=get_model_color(mdl), lw=1.2, alpha=0.8,
                         label=SHORT.get(mdl, mdl)) for mdl in MODEL_ORDER])
    lg = ax.legend(handles=handles, fontsize=4.6, ncol=2, frameon=True, loc='lower left',
                   labelspacing=0.28, handlelength=1.1, handletextpad=0.4, columnspacing=0.8,
                   borderpad=0.4)
    lg.get_frame().set(facecolor='white', edgecolor='#ccc', lw=0.5)
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def panel_sweep(ax):
    gp = swing_vs_eps(os.path.join(CLUSTER_OUT, 'fig5_ext_heldout'))
    gf = swing_vs_eps(os.path.join(CLUSTER_OUT, 'fig5_ext_heldout_fgsm'))
    gcol = {'Untrained': '#6E6E6E', 'Conventional': '#C2185B', 'Adv-trained': '#2CA02C'}
    for grp, c in gcol.items():
        a = gp[gp.grp == grp].sort_values('eps_255'); b = gf[gf.grp == grp].sort_values('eps_255')
        ax.plot(a.eps_255, a.swing, '-o', color=c, lw=1.7, ms=3.5)
        ax.plot(b.eps_255, b.swing, '--s', color=c, lw=1.3, ms=3.0, mfc='white')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel(r'Perturbation strength $\epsilon$ (/255)', fontsize=FS_AX)
    ax.set_ylabel(r'Normalized $L_\infty$ swing', fontsize=FS_AX)
    # split legend: compact group-colour key (upper left) + attack line-style key (lower right)
    col_h = [Line2D([0], [0], color=c, lw=1.8, label=g) for g, c in gcol.items()]
    sty_h = [Line2D([0], [0], color='0.35', lw=1.7, ls='-', marker='o', ms=3.5, label='PGD'),
             Line2D([0], [0], color='0.35', lw=1.3, ls='--', marker='s', ms=3.0, mfc='white', label='FGSM')]
    lg1 = ax.legend(handles=col_h, fontsize=FS_ANNOT, loc='upper left', frameon=True,
                    labelspacing=0.22, handletextpad=0.4, borderpad=0.3)
    lg1.get_frame().set(facecolor='white', edgecolor='#ccc', lw=0.5); ax.add_artist(lg1)
    lg2 = ax.legend(handles=sty_h, fontsize=FS_ANNOT, loc='lower right', frameon=True,
                    labelspacing=0.22, handletextpad=0.4, borderpad=0.3)
    lg2.get_frame().set(facecolor='white', edgecolor='#ccc', lw=0.5)
    ax.tick_params(labelsize=FS_TICK); _spines(ax)


def panel_trend(ax, df):
    """Horizontal SIGNED-r bars: PGD vs FGSM sensitivity (log-eps AUC) correlated with control
    slope, at the model level (per-model means) and site level (site residuals), for the
    10/9/7 subsets.
    Bars oriented to the full-set (10-model) sign so the full set reads POSITIVE; a subset
    that reverses sign dips left of the dashed 0-line."""
    from matplotlib.patches import Patch
    groups = {10: list(df.model.unique()),
              9: [m for m in df.model.unique() if m != UNTR],
              7: [m for m in df.model.unique() if m not in EXTREMES]}
    attacks = [('auc_pgd', C_PGD), ('auc_fgsm', C_FGSM)]

    def _r(sub, col, level):
        if level == 'model':
            pm = sub.groupby('model')[[col, 'control_slope']].mean()
            return stats.pearsonr(pm[col], pm['control_slope'])[0]
        s = sub.dropna(subset=[col, 'control_slope']).copy()
        yr = s['control_slope'] - s.groupby(['monkey', 'channel'])['control_slope'].transform('mean')
        return stats.pearsonr(s[col], yr)[0]

    sgn = {(lev, col): (1.0 if _r(df, col, lev) >= 0 else -1.0)
           for lev in ('model', 'site') for col, _ in attacks}

    slot_y = {('model', 10): 8.0, ('model', 9): 7.0, ('model', 7): 6.0,
              ('site', 10): 2.0, ('site', 9): 1.0, ('site', 7): 0.0}
    dh, h, XNEG = 0.20, 0.34, -0.08
    for (level, k), yy in slot_y.items():
        sub = df[df.model.isin(groups[k])]
        for (col, c), off in zip(attacks, (dh, -dh)):
            v = _r(sub, col, level) * sgn[(level, col)]
            ax.barh(yy + off, max(v, XNEG), height=h, color=c, edgecolor='none', zorder=3)
    ax.axvline(0, color='0.5', lw=0.6, ls=(0, (4, 3)), zorder=2)
    NAME = {10: 'All models', 9: 'Trained models', 7: 'Conventionally trained'}
    NVAL = {'model': {10: 10, 9: 9, 7: 7}, 'site': {10: 250, 9: 225, 7: 175}}
    ax.set_yticks(list(slot_y.values()))
    ax.set_yticklabels([f'{NAME[k]}\nn = {NVAL[lev][k]}' for (lev, k) in slot_y], fontsize=FS_TICK)
    for tick, (lev, k) in zip(ax.get_yticklabels(), slot_y):
        tick.set_color(SUBSET_COL[k])
    ax.set_ylim(-0.5, 9.3); ax.set_xlim(-0.10, 1.05)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel('Correlation with control slope\n(site residuals; sign-flipped)', fontsize=FS_AX)
    ax.annotate('by model', xy=(-0.02, 8.7), xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                fontsize=FS_AX, fontweight='bold', color='0.25', fontfamily=FONT_FAMILY, annotation_clip=False)
    ax.annotate('by site', xy=(-0.02, 2.7), xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                fontsize=FS_AX, fontweight='bold', color='0.25', fontfamily=FONT_FAMILY, annotation_clip=False)
    h_leg = [Patch(fc=C_PGD, label='PGD'), Patch(fc=C_FGSM, label='FGSM')]
    lg = ax.legend(handles=h_leg, fontsize=FS_ANNOT, frameon=True, loc='lower right',
                   labelspacing=0.3, borderpad=0.4, handletextpad=0.5)
    lg.get_frame().set(facecolor='white', edgecolor='#ccc', lw=0.5)
    ax.tick_params(labelsize=FS_TICK); ax.tick_params(axis='y', length=0); _spines(ax)


def main(out_dir):
    er = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, 'sup_advrob_fgsm_vs_pgd_eps.csv'),
                                   hint='frozen input; python scripts/preprocessing/build_all.py'))
    er = er[er.norm == 'linf'].sort_values('eps_255')
    df = (pd.read_csv(paths.require(os.path.join(PREPROC_DATA, 'fig5_heldout_table.csv'),
                                    hint='python scripts/preprocessing/fig5_heldout_analysis.py'))[KEY + ['control_slope']]
          .merge(load_logauc(['fig5_ext_heldout', 'fig5_ext_heldout_lowextra'], 'auc_pgd'), on=KEY)
          .merge(load_logauc(['fig5_ext_heldout_fgsm', 'fig5_ext_heldout_fgsm_lowextra'], 'auc_fgsm'), on=KEY))

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 62 * MM), facecolor=FIG_FACECOLOR)
    gs = GridSpec(1, 3, figure=fig, left=0.07, right=0.985, top=0.80, bottom=0.17,
                  wspace=0.36)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2])
    pc = axc.get_position()                                    # nudge panel c right + narrow so its long
    axc.set_position([pc.x0 + 0.045, pc.y0, pc.width - 0.045, pc.height])  # y-labels clear panel b

    panel_agreement(axa, er)
    panel_sweep(axb)
    panel_trend(axc, df)

    # title = what is plotted; grey subtitle = plain-language guide to the panel
    titles = [
        ('Agreement between FGSM and PGD attacks',
         'black: all 250 axes pooled; colors: within each model (25 axes)'),
        ('Attack effect vs. perturbation strength',
         'training-family means; solid = PGD, dashed = FGSM'),
        ('Sensitivity-control correlation by attack',
         'sensitivity (log-ε AUC) vs control slope, per model subset'),
    ]
    axes = [axa, axb, axc]
    for a, (t, sub) in zip(axes, titles):
        a.set_title(t, fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=13)
        a.text(0.5, 1.03, sub, transform=a.transAxes, ha='center', va='bottom',
               fontsize=FS_SUB, color='0.45', style='italic', fontfamily=FONT_FAMILY)

    fig.canvas.draw()
    for a, letter in zip(axes, 'abc'):
        bb = a.get_position()
        fig.text(bb.x0 - 0.04, bb.y1 + 0.045, letter, fontsize=FS_LET, fontweight='bold',
                 fontfamily=FONT_FAMILY, ha='left', va='bottom')

    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', path)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    main(**vars(ap.parse_args()))
