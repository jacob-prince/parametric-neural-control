#!/usr/bin/env python3
"""Supp fig (adv_sensitivity_formulations) — the endpoint-dependence result ("adversarial
sensitivity correlates with control over the full model set, but the relationship vanishes
once the adversarially-trained models are removed") replicates across every alternative
formulation of adversarial sensitivity, under BOTH attack methods (each formulation is
computed for PGD and, matched exactly, for single-step FGSM), while the gradient spectral
participation ratio (PR) survives.

Everything uses the exact main-figure recipe: sensitivity measured on held-out NSD images
(never used to fit the readouts or as synthesis seeds), control slope site-residualized
against the 9 trained models' per-(monkey,channel) mean, raw predictor. Model level =
per-model means; exact p = full 7! permutation of model means. All displayed correlations
are ORIENTED: each measure is multiplied by the sign of its own all-10 correlation (per
level), so "keeps the full-set relationship" reads positive and a sign flip reads negative
— this puts the sensitivity formulations and gradient spectral PR on one common scale.

  a  adversarial-sensitivity formulations x {attack x model-subset}: oriented Pearson r
     with control slope at the model level then the site level, for {all 10 / 9 trained /
     7 standard-trained} under PGD and FGSM. Formulations span per-eps swing (0.125-64/255),
     logAUC windows, linear AUC, the L2 threat model, up-only direction, and alternative
     response normalizers; the input-gradient norms are attack-independent (masked right
     half). Black box = the main-figure default (PGD logAUC[0.125,16]).
  b  gradient spectral characteristics x {model-subset}, own panel (attack-independent, so
     no PGD/FGSM pairing), grouped concentration | location | low-freq emphasis. Black
     box = gradient spectral PR (the main-figure measure). NOTE: PR is an exact monotone
     transform of the coefficient of variation (PR = n_bins / (1 + CoV^2)); --gradsum cv
     renders the CoV-family companion.
  c  aligned exact-p strips (model then site level), segmented to match a and b: one-sided
     all-7!-relabeling p among the standard-trained 7, oriented to each measure's own
     all-10 direction.

Reads cluster outputs (fig5_ext_heldout{,_lowextra}, fig5_ext_heldout_fgsm{,_lowextra},
fig5_heldout_gradfreq) + preproc_data/{fig5_heldout_table.csv,
sup_advform_heldout_descriptors.csv}. Flags: --outcome slope|r, --gradsum pr|cv (defaults
slope, pr = the manuscript figure; non-default variants get a filename tag). The per-formulation statistics table is written next to the
figure as sup_advform_stats<tag>.csv.
"""
import os
import glob
import pickle
import re
import argparse
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pnc import paths
from pnc.utils import apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig
from pnc.manifest import output_name
apply_figure_style()

STEM = output_name('adv_sensitivity_formulations')
PREPROC_DATA = str(paths.preprocessed_data())
CLUSTER_OUT = str(paths.cluster_outputs())

FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ROW, FS_ANNOT = 11, 7, 6, 5.5, 5, 5
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
KEY = ['model', 'monkey', 'channel']
UNTR = 'AlexNet_training_seed_01'
EXTREMES = ['resnet50_robust', 'clipag_vitb32', UNTR]
SUBSET_COL = {10: '#5E35B1', 9: '#00838F', 7: '#C2185B'}   # main-figure subset label colours
# palette (CVD-validated): muted indigo <-> brick diverging heatmap (sns 'vlag': flip <-> keeps);
# the accent hues stay OFF the heatmap poles — steel/gold = the two attacks (SAME hue in every
# panel), neutral grey = attack-independent gradient norms, violet = the spectral family.
C_PGD, C_FGSM, C_SHARED, C_CV = '#3D7FB3', '#BC8F0F', '#8A8A8A', '#8746B8'
C_SENSGRP = '#3A5A72'                                      # row-group colour: adversarial sensitivity
DIV_CMAP = 'vlag'                                          # oriented r: flip <- neutral -> keeps
FMIN, FMAX = 1, 112
EPS_GRID = [0.125, 0.5, 1, 2, 4, 8, 16, 32, 64]
FIG_DEFAULT = 'logAUC[0.125,16]'


# ---------------- data: per-readout sensitivity under every formulation ----------------
def load_raw(dirs):
    return pd.concat([pd.read_csv(p) for d in dirs
                      for p in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, d)),
                                                             'adv_robustness_*.csv')))
                      if 'smoke' not in p], ignore_index=True)


def curve_table(raw, norm='linf', up_only=False, denom='range_q99q01'):
    """Per-(readout, eps) mean normalized perturbation effect."""
    r = raw[raw.norm == norm].copy()
    num = (r.adv_up - r.clean_pred) if up_only else (r.adv_up - r.adv_dn)
    r['ns'] = num / r[denom]
    return r.groupby(KEY + ['eps_255'])['ns'].mean().reset_index()


def at_eps(tab, e):
    return tab[np.isclose(tab.eps_255, e)].set_index(KEY)['ns']


def log_auc(tab, lo, hi):
    t = tab[(tab.eps_255 >= lo - 1e-9) & (tab.eps_255 <= hi + 1e-9)]
    out = {}
    for k, g in t.groupby(KEY):
        g = g.sort_values('eps_255'); x = np.log2(g.eps_255.values)
        out[k] = float(trapz(g.ns.values, x) / (x[-1] - x[0]))
    return pd.Series(out)


def lin_auc(tab, lo, hi):
    t = tab[(tab.eps_255 >= lo - 1e-9) & (tab.eps_255 <= hi + 1e-9)]
    out = {}
    for k, g in t.groupby(KEY):
        g = g.sort_values('eps_255'); x = g.eps_255.values
        out[k] = float(trapz(g.ns.values, x) / (x[-1] - x[0]))
    return pd.Series(out)


def _cov(P):
    P = np.asarray(P, float)
    return float(P.std() / P.mean())


def load_heldout_profiles():
    """(model, monkey, channel) -> mean held-out radial gradient power profile (100 NSD images)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, 'fig5_heldout_gradfreq')), '*.pkl'))):
        mm = re.match(r'(.+?)_unit_(\d+)_model_(.+)_grad_maps_freq_profiles\.pkl', os.path.basename(f))
        if not mm:
            continue
        with open(f, 'rb') as fh:
            p = np.asarray(pickle.load(fh)['profiles'], float).mean(0)
        out[(mm.group(3), mm.group(1).split('_')[0], int(mm.group(2)))] = p
    return out


def _gini(P):
    P = np.sort(np.asarray(P, float)); n = len(P)
    cum = np.cumsum(P)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _entropy(P):
    q = np.asarray(P, float); q = q / q.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q)))   # normalized Shannon entropy (0..1)


def _prat(P):
    P = np.asarray(P, float)
    d = float((P ** 2).sum())                                # participation ratio (effective #bins)
    return float(P.sum() ** 2 / d) if d > 0 else float(np.nan)


def _medfreq(P):
    P = np.asarray(P, float)
    freqs = np.arange(FMIN, FMIN + len(P))
    return float(freqs[np.searchsorted(np.cumsum(P), 0.5 * P.sum())])


def load_spectral_summaries():
    """kind -> per-readout Series for every spectral summary: the descriptor-table measures
    + the two profile-derived CoV robustness variants, extended with Gini coefficient,
    normalized spectral entropy, participation ratio, and median frequency, all computed on
    the same held-out radial profiles over the same frequency band."""
    d = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, 'sup_advform_heldout_descriptors.csv'),
                                  hint='frozen input; python scripts/preprocessing/build_all.py'))
    idx = pd.MultiIndex.from_arrays([d.model, d.monkey, d.unit.astype(int)])
    out = {k: pd.Series(d[c].values, index=idx)
           for k, c in [('cov', 'ho_cv'), ('flat', 'ho_flat'), ('decay', 'ho_decay'),
                        ('lowfrac', 'lowfrac'), ('centroid', 'ho_centroid')]}
    profs = load_heldout_profiles()
    freqs = np.arange(FMIN, FMAX)                        # cyc/img per bin of P[FMIN:FMAX]

    def _lf(P, c):                                       # power fraction below c cyc/img
        P = np.asarray(P, float)
        return float(P[freqs <= c].sum() / P.sum())

    def _lhr(P):                                         # log10 low/high power ratio
        P = np.asarray(P, float)
        return float(np.log10(P[freqs <= 8].sum() / P[freqs >= 32].sum()))

    def _invf(P):                                        # power-weighted mean of 1/f
        P = np.asarray(P, float)
        return float((P / freqs).sum() / P.sum())

    for k, fn in ([('gini', _gini), ('entropy', _entropy), ('prat', _prat), ('medfreq', _medfreq)]
                  + [(f'lf{c}', lambda p, c=c: _lf(p, c)) for c in (64, 48, 32, 24, 8, 4)]
                  + [('lhr', _lhr), ('invf', _invf),
                     ('cov_d3', lambda p: _cov(p[3:])),
                     ('cov_rb', lambda p: _cov([s.mean() for s in np.array_split(p, 20)])),
                     ('pr_d3', lambda p: _prat(p[3:])),
                     ('pr_rb', lambda p: _prat([s.mean() for s in np.array_split(p, 20)]))]):
        out[k] = pd.Series({key: fn(p[FMIN:FMAX]) for key, p in profs.items()})
    return out


# every PGD formulation has the exactly-matched FGSM formulation; the input-gradient norms and
# the gradient spectral CoV are attack-independent ('shared') and appear once
_BASE_FORMULATIONS = (
    [(f'swing @{e:g}', 'eps', e) for e in EPS_GRID]
    + [('logAUC[0.125,16]', 'logauc', (0.125, 16)),     # main-figure default
       ('logAUC[0.125,4]', 'logauc', (0.125, 4)),
       ('logAUC[0.125,64]', 'logauc', (0.125, 64)),
       ('linAUC[0.5,64]', 'linauc', (0.5, 64)),
       ('L2 swing @4', 'l2', 4),
       ('L2 logAUC[0.125,16]', 'l2log', (0.125, 16)),
       ('up-only swing @4', 'up', 4),
       ('up-only logAUC[0.125,16]', 'uplog', (0.125, 16)),
       ('swing @4 / (q95-q05)', 'q95', 4),
       ('swing @4 / std', 'std', 4),
       ('swing @4 / (max-min)', 'mm', 4),
       ('gradient norm L2', 'gradl2', None),
       ('gradient norm L1', 'gradl1', None),
       ('Gradient spectral CoV', 'cov', None),           # spectral summaries: CONCENTRATION ...
       ('Wiener flatness', 'flat', None),
       ('Gini coefficient', 'gini', None),
       ('spectral entropy', 'entropy', None),
       ('participation ratio', 'prat', None),
       ('CoV, drop 3 low bins', 'cov_d3', None),
       ('CoV, coarse rebin (20)', 'cov_rb', None),
       ('log-log decay slope', 'decay', None),           # ... then LOCATION of the power
       ('spectral centroid', 'centroid', None),
       ('median frequency', 'medfreq', None),
       ('low-freq fraction (< 64 cyc)', 'lf64', None),   # ... then explicit LOW-FREQ EMPHASIS,
       ('low-freq fraction (< 48 cyc)', 'lf48', None),   #     cutoff swept wide -> narrow
       ('low-freq fraction (< 32 cyc)', 'lf32', None),
       ('low-freq fraction (< 24 cyc)', 'lf24', None),
       ('low-freq fraction (< 16 cyc)', 'lowfrac', None),
       ('low-freq fraction (< 8 cyc)', 'lf8', None),
       ('low-freq fraction (< 4 cyc)', 'lf4', None),
       ('low/high power ratio (log)', 'lhr', None),
       ('1/f-weighted power', 'invf', None)])
SPECTRAL_KINDS = {'cov', 'flat', 'decay', 'lowfrac', 'centroid', 'gini', 'entropy', 'prat',
                  'medfreq', 'cov_d3', 'cov_rb', 'pr_d3', 'pr_rb', 'lf64', 'lf48', 'lf32', 'lf24',
                  'lf8', 'lf4', 'lhr', 'invf'}
SHARED_KINDS = {'gradl2', 'gradl1'} | SPECTRAL_KINDS


def _configure(outcome='slope', gradsum='pr'):
    """Set the variant-dependent module state (was read from env OUTCOME / GRADSUM at import).

    outcome: 'slope' (default) | 'r' (companion). gradsum: 'pr' (default, the main-figure
    metric) | 'cv': coefficient-of-variation variant. Under 'pr' the CoV-family spectral rows
    become their PR counterparts (same band; PR is featured/boxed and CoV joins the
    alternatives); PR is monotone DECREASING in CoV so raw signs flip, but every display is
    oriented by its own all-10 sign."""
    global OUTCOME, OUTC, OUTWORD, GRADSUM, TAG, SPECWORD, MAIN_SPEC_KIND, MAIN_SPEC_LABEL
    global FORMULATIONS, SENS_FORMS, SPEC_FORMS
    OUTCOME = outcome
    OUTC = {'slope': 'control_slope', 'r': 'control_r'}[OUTCOME]
    OUTWORD = {'slope': 'control slope', 'r': 'control r'}[OUTCOME]
    GRADSUM = gradsum
    TAG = ('' if OUTCOME == 'slope' else '_r') + ('' if GRADSUM == 'pr' else '_cv')
    SPECWORD = 'PR' if GRADSUM == 'pr' else 'CoV'
    MAIN_SPEC_KIND = 'prat' if GRADSUM == 'pr' else 'cov'
    MAIN_SPEC_LABEL = f'Gradient spectral {SPECWORD}'
    FORMULATIONS = list(_BASE_FORMULATIONS)
    if GRADSUM == 'pr':                                  # swap the CoV family for its PR counterparts;
        _PR_SWAP = {'Gradient spectral CoV': ('Gradient spectral PR', 'prat'),      # PR becomes the boxed
                    'participation ratio': ('coefficient of variation (CoV)', 'cov'),  # main measure, CoV
                    'CoV, drop 3 low bins': ('PR, drop 3 low bins', 'pr_d3'),        # joins the alternatives
                    'CoV, coarse rebin (20)': ('PR, coarse rebin (20)', 'pr_rb')}
        FORMULATIONS = [(*_PR_SWAP.get(lb, (lb, kd)), ar) for lb, kd, ar in FORMULATIONS]
    SENS_FORMS = [f for f in FORMULATIONS if f[1] not in SPECTRAL_KINDS]   # panel a: 20 attack + 2 grad
    SPEC_FORMS = [f for f in FORMULATIONS if f[1] in SPECTRAL_KINDS]       # panel b: 19 spectral


_configure()
SENS_BREAKS = ['logAUC[0.125,16]', 'L2 swing @4', 'up-only swing @4', 'swing @4 / (q95-q05)',
               'gradient norm L2']
SPEC_BREAKS = ['log-log decay slope', 'low-freq fraction (< 64 cyc)']  # concentration|location|low-freq
SPEC_GROUPS = ['concentration', 'location', 'low-freq\nemphasis']
ATTACK_INDEP_START = 'gradient norm L2'                  # dark rule drawn above this row (panel a)


def build_series(raw):
    """kind -> callable(args) -> per-readout Series, for one attack's raw CSVs."""
    main = curve_table(raw)
    tabs = dict(main=main, l2=curve_table(raw, norm='l2'), up=curve_table(raw, up_only=True),
                q95=curve_table(raw, denom='range_q95q05'), std=curve_table(raw, denom='std_resp'),
                mm=curve_table(raw, denom='range_minmax'))
    grads = raw[raw.norm == 'linf'].groupby(KEY)[['grad_l2', 'grad_l1']].mean()
    spec = load_spectral_summaries()
    mk = {
        'eps': lambda e: at_eps(tabs['main'], e),
        'logauc': lambda w: log_auc(tabs['main'], *w),
        'linauc': lambda w: lin_auc(tabs['main'], *w),
        'l2': lambda e: at_eps(tabs['l2'], e),
        'l2log': lambda w: log_auc(tabs['l2'], *w),
        'up': lambda e: at_eps(tabs['up'], e),
        'uplog': lambda w: log_auc(tabs['up'], *w),
        'q95': lambda e: at_eps(tabs['q95'], e),
        'std': lambda e: at_eps(tabs['std'], e),
        'mm': lambda e: at_eps(tabs['mm'], e),
        'gradl2': lambda _: grads['grad_l2'],
        'gradl1': lambda _: grads['grad_l1'],
    }
    mk.update({k: (lambda _, s=s: s) for k, s in spec.items()})
    return mk, tabs['main']


# ---------------- stats: main-figure recipe ----------------
def site_residual(ht):
    src = ht[ht.model != UNTR]
    m = src.groupby(['monkey', 'channel'])[OUTC].mean()
    return ht[OUTC].values - m.reindex(pd.MultiIndex.from_frame(ht[['monkey', 'channel']])).values


def subset_frames(d):
    return {10: d, 9: d[d.model != UNTR], 7: d[~d.model.isin(EXTREMES)]}


def model_level(sub, col):
    g = sub.groupby('model')[[col, 'y']].mean()
    return g[col].values, g['y'].values


def exact_p(x, y, s):
    """Exact one-sided permutation p over all 7! label permutations: registers only effects in
    the measure's own full-set (all-10) direction s; a sign-flipped correlation gets p near 1."""
    perms = list(permutations(range(len(y))))
    rs = np.array([stats.pearsonr(x, y[list(p)])[0] for p in perms])
    return float((s * rs >= s * rs[0] - 1e-12).mean())      # identity permutation comes first


def exact_p_site(sub, s):
    """Exact one-sided site-level permutation p: all 7! global relabelings of the model
    identities (columns of the site x model predictor matrix), recomputing the site-level
    Pearson r against the (fixed) residualized outcome each time; one-sided in the full-set
    direction s. Respects the crossed design -- exact_p is this same test on model means."""
    X = sub.pivot_table(index=['monkey', 'channel'], columns='model', values='sens')
    Y = sub.pivot_table(index=['monkey', 'channel'], columns='model', values='y')[X.columns]
    X, Y = X.values, Y.values
    keep = ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
    X, Y = X[keep], Y[keep]
    N = X.size
    sx, sy = X.sum(), Y.sum()
    dx = np.sqrt(N * (X ** 2).sum() - sx ** 2); dy = np.sqrt(N * (Y ** 2).sum() - sy ** 2)
    S = X.T @ Y                                             # S[a, b] = sum_sites X[:,a] * Y[:,b]
    m = np.arange(X.shape[1])
    rs = np.array([(N * S[list(p), m].sum() - sx * sy) / (dx * dy)
                   for p in permutations(range(X.shape[1]))])
    return float((s * rs >= s * rs[0] - 1e-12).mean())      # identity permutation comes first


def merge_series(ht, series):
    d = ht.merge(series.rename('sens').reset_index().set_axis(KEY + ['sens'], axis=1), on=KEY)
    return d.dropna(subset=['sens', 'y'])


def compute_matrix(ht, mk_pgd, mk_fgsm):
    rows = []
    for label, kind, args in FORMULATIONS:
        attacks = [('shared', mk_pgd)] if kind in SHARED_KINDS else [('PGD', mk_pgd), ('FGSM', mk_fgsm)]
        for attack, mk in attacks:
            d = merge_series(ht, mk[kind](args))
            subs = subset_frames(d)
            rec = dict(label=label, attack=attack, kind=kind)
            for k, sub in subs.items():
                rec[f'site_{k}'] = stats.pearsonr(sub.sens, sub.y)[0]
                xm, ym = model_level(sub, 'sens')
                rec[f'model_{k}'] = stats.pearsonr(xm, ym)[0]
            xm, ym = model_level(subs[7], 'sens')
            rec['p7'] = exact_p(xm, ym, np.sign(rec['model_10']))
            rec['p7_site'] = exact_p_site(subs[7], np.sign(rec['site_10']))
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------- panels ----------------
def _idx(forms, label):
    return [f[0] for f in forms].index(label)


def _oriented_grid(M, forms, level, attacks):
    """Oriented r grid: 6 cols [PGD 10/9/7 | FGSM 10/9/7] when attacks=True, else 3 cols [10/9/7].
    Each value multiplied by the sign of that measure's own all-10 correlation at that level."""
    G = np.full((len(forms), 6 if attacks else 3), np.nan)
    for i, (label, kind, _) in enumerate(forms):
        for j, attack in enumerate(['PGD', 'FGSM'] if attacks else ['shared']):
            a = 'shared' if kind in SHARED_KINDS else attack
            r = M[(M.label == label) & (M.attack == a)].iloc[0]
            s = np.sign(r[f'{level}_10'])
            G[i, 3 * j:3 * j + 3] = [s * r[f'{level}_10'], s * r[f'{level}_9'], s * r[f'{level}_7']]
    return G


def _heat(ax, G, mask, xticklabels):
    A = np.vectorize(lambda v: f'{v + 0:.2f}'.replace('-0.00', '0.00').replace('0.', '.'))(G)
    sns.heatmap(G, ax=ax, cmap=DIV_CMAP, vmin=-0.95, vmax=0.95, cbar=False, mask=mask,
                annot=A, fmt='', annot_kws={'fontsize': 4.6, 'fontfamily': FONT_FAMILY},
                xticklabels=xticklabels, yticklabels=False)
    ax.set_xticklabels(xticklabels, fontsize=FS_TICK, rotation=0)
    for tick, lab in zip(ax.get_xticklabels(), xticklabels):
        tick.set_color(SUBSET_COL[int(lab)]); tick.set_fontweight('bold')
    ax.tick_params(length=0)
    return ax.collections[0]


def _row_labels(ax, forms, color, bold):
    ax.set_yticks(np.arange(len(forms)) + 0.5)
    ax.set_yticklabels([f[0] for f in forms], fontsize=FS_ROW, rotation=0)
    for tick, (label, kind, _) in zip(ax.get_yticklabels(), forms):
        tick.set_color(color)
        if label in bold:
            tick.set_fontweight('bold')


def heat_sens(ax, M, level, title):
    """Panel a block: sensitivity formulations x {PGD 10/9/7 | FGSM 10/9/7}."""
    forms = SENS_FORMS
    shared = np.array([k in SHARED_KINDS for _, k, _ in forms])
    mask = np.zeros((len(forms), 6), bool); mask[shared, 3:] = True
    im = _heat(ax, _oriented_grid(M, forms, level, True), mask, ['10', '9', '7'] * 2)
    ax.set_title(title, fontsize=FS_TITLE, pad=13)
    for x, lab, c in [(1.5, 'PGD', C_PGD), (4.5, 'FGSM', C_FGSM)]:
        ax.text(x, -0.35, lab, fontsize=FS_ANNOT, color=c, fontweight='bold',
                ha='center', va='bottom', fontfamily=FONT_FAMILY, clip_on=False)
    ax.axvline(3, color='white', lw=2.5)
    for lb in SENS_BREAKS:
        ax.axhline(_idx(forms, lb), color='white', lw=1.6)
    ax.axhline(_idx(forms, ATTACK_INDEP_START), color='#3F3F3F', lw=1.0, zorder=5, clip_on=False)
    ax.text(4.5, _idx(forms, ATTACK_INDEP_START) + 1, 'attack-\nindependent', fontsize=FS_ANNOT,
            color='#777777', style='italic', ha='center', va='center', fontfamily=FONT_FAMILY)
    yi = _idx(forms, FIG_DEFAULT)                        # main-figure default box (PGD block)
    ax.add_patch(plt.Rectangle((0, yi), 3, 1, fill=False, edgecolor='k', lw=1.0, zorder=6))
    return im


def heat_spec(ax, M, level, header, sublabels=False):
    """Panel b block: spectral summaries x {10/9/7} (attack-independent, no PGD/FGSM pairing)."""
    forms = SPEC_FORMS
    _heat(ax, _oriented_grid(M, forms, level, False), None, ['10', '9', '7'])
    ax.text(1.5, -0.35, header, fontsize=FS_ANNOT, color='#333333', fontweight='bold',
            ha='center', va='bottom', fontfamily=FONT_FAMILY, clip_on=False)
    for lb in SPEC_BREAKS:
        ax.axhline(_idx(forms, lb), color='white', lw=1.6)
    yc = _idx(forms, MAIN_SPEC_LABEL)                    # the main-figure spectral measure
    ax.add_patch(plt.Rectangle((0, yc), 3, 1, fill=False, edgecolor='k', lw=1.0, zorder=6))
    if sublabels:                                        # sub-group names, right of block
        edges = [0] + [_idx(forms, lb) for lb in SPEC_BREAKS] + [len(forms)]
        for name, lo, hi in zip(SPEC_GROUPS, edges[:-1], edges[1:]):
            ax.text(3.35, (lo + hi) / 2, name, fontsize=FS_ANNOT, color=C_CV, style='italic',
                    ha='left', va='center', fontfamily=FONT_FAMILY, clip_on=False)


def pstrip(ax, M, forms, col, subheader=None, xlabels=False, breaks=(), dark_rule=False):
    """One exact-p strip segment, rows aligned with the given heat block."""
    for label, kind, _ in forms:
        i = _idx(forms, label) + 0.5
        if kind in SPECTRAL_KINDS:
            p = M[(M.label == label)].iloc[0][col]
            ax.plot(p, i, marker='o', ms=4.6 if kind == MAIN_SPEC_KIND else 3.0, color=C_CV, zorder=5)
            pmin = p
        elif kind in SHARED_KINDS:
            p = M[(M.label == label)].iloc[0][col]
            ax.plot(p, i, marker='o', ms=2.6, color=C_SHARED, zorder=4)
            pmin = p
        else:
            pp = {a: M[(M.label == label) & (M.attack == a)].iloc[0][col] for a in ('PGD', 'FGSM')}
            ax.plot(pp['PGD'], i, marker='o', ms=2.6, color=C_PGD, zorder=4)
            ax.plot(pp['FGSM'], i, marker='o', ms=3.6, mfc='none', mec=C_FGSM, mew=0.8, zorder=5)
            pmin = min(pp.values())
        ax.plot([pmin, 1.0], [i, i], color='#DDDDDD', lw=0.5, zorder=1)
    ax.axvline(0.05, color='#C2185B', lw=0.7, ls='--', zorder=2)
    ax.set_xscale('log'); ax.set_xlim(7e-5, 2.1)          # room for the 1/5040 floor and p = 1
    ax.set_ylim(len(forms), 0)
    ax.set_yticks([])
    ax.set_xticks([1e-3, 1e-1])
    ax.set_xticklabels(['0.001', '0.1'] if xlabels else [], fontsize=FS_TICK)
    if subheader:
        ax.text(0.5, -0.35, subheader, transform=ax.get_yaxis_transform(), fontsize=FS_ANNOT,
                color='#333333', fontweight='bold', ha='center', va='bottom',
                fontfamily=FONT_FAMILY, clip_on=False)
    for lb in breaks:
        ax.axhline(_idx(forms, lb), color='#EEEEEE', lw=0.8, zorder=0)
    if dark_rule:
        ax.axhline(_idx(forms, ATTACK_INDEP_START), color='#3F3F3F', lw=1.0, zorder=2)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


def main(out_dir, outcome='slope', gradsum='pr'):
    _configure(outcome, gradsum)
    mk_pgd, pgd_main = build_series(load_raw(['fig5_ext_heldout', 'fig5_ext_heldout_lowextra']))
    mk_fgsm, fgsm_main = build_series(load_raw(['fig5_ext_heldout_fgsm', 'fig5_ext_heldout_fgsm_lowextra']))
    ht = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, 'fig5_heldout_table.csv'),
                                   hint='python scripts/preprocessing/fig5_heldout_analysis.py'))[KEY + [OUTC]]
    ht['y'] = site_residual(ht)
    M = compute_matrix(ht, mk_pgd, mk_fgsm)
    os.makedirs(out_dir, exist_ok=True)
    M.to_csv(os.path.join(out_dir, f'sup_advform_stats{TAG}.csv'), index=False)
    print(M.to_string(float_format=lambda v: f'{v:+.3f}'))

    # panels a + b + c only (no collapse scatters / eps sweep): 0.75x the 2-column width,
    # x-positions scaled accordingly from the original full-width layout
    fig = plt.figure(figsize=(WIDTH_2COL_MM * 0.75 * MM, 132 * MM), facecolor=FIG_FACECOLOR)
    # vertical geometry: panel a = 22 sensitivity rows, panel b = 19 spectral rows, common row height
    YA0, HA = 0.493, 0.402                              # panel a block (22 rows)
    YB0, HB = 0.100, 0.348                              # panel b block (19 rows), gap 0.045
    axs = fig.add_axes([0.247, YA0, 0.197, HA])         # a: model-level heatmap (PGD | FGSM)
    axm = fig.add_axes([0.464, YA0, 0.197, HA])         # a: site-level heatmap (PGD | FGSM)
    axbm = fig.add_axes([0.296, YB0, 0.099, HB])        # b: model-level triplet (centred)
    axbs = fig.add_axes([0.513, YB0, 0.099, HB])        # b: site-level triplet (centred)
    axpm_a = fig.add_axes([0.696, YA0, 0.077, HA])      # c: exact-p model, sensitivity rows
    axps_a = fig.add_axes([0.803, YA0, 0.077, HA])      # c: exact-p site, sensitivity rows
    axpm_b = fig.add_axes([0.696, YB0, 0.077, HB])      # c: exact-p model, spectral rows
    axps_b = fig.add_axes([0.803, YB0, 0.077, HB])      # c: exact-p site, spectral rows

    im = heat_sens(axs, M, 'model', 'Model level')
    heat_sens(axm, M, 'site', 'Site level')
    _row_labels(axs, SENS_FORMS, C_SENSGRP, {FIG_DEFAULT})
    heat_spec(axbm, M, 'model', 'Model', sublabels=True)   # sub-labels sit in the gap to axbs
    heat_spec(axbs, M, 'site', 'Site')
    _row_labels(axbm, SPEC_FORMS, C_CV, {MAIN_SPEC_LABEL})

    pstrip(axpm_a, M, SENS_FORMS, 'p7', subheader='Model', breaks=SENS_BREAKS, dark_rule=True)
    pstrip(axps_a, M, SENS_FORMS, 'p7_site', subheader='Site', breaks=SENS_BREAKS, dark_rule=True)
    pstrip(axpm_b, M, SPEC_FORMS, 'p7', xlabels=True, breaks=SPEC_BREAKS)
    pstrip(axps_b, M, SPEC_FORMS, 'p7_site', xlabels=True, breaks=SPEC_BREAKS)
    PX = (0.696 + 0.803 + 0.077) / 2                    # centre of the two p columns
    # panel-c title, top-aligned with panel a's 'Model level' / 'Site level' titles
    fig.canvas.draw()
    _tb = axs.title.get_window_extent(fig.canvas.get_renderer())
    _ty = fig.transFigure.inverted().transform((0, _tb.y1))[1]
    fig.text(PX, _ty, 'Significance, 7-model subset', fontsize=FS_TITLE, fontfamily=FONT_FAMILY,
             ha='center', va='top')
    fig.text(PX, 0.058, 'One-sided exact p\n(all-10 direction,\nstandard-trained 7)\ndashed: p = 0.05',
             fontsize=FS_AX, fontfamily=FONT_FAMILY, ha='center', va='top')

    cax = fig.add_axes([0.353, 0.040, 0.227, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(f'Oriented Pearson r with {OUTWORD} (site residuals)', fontsize=FS_ANNOT, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK)
    fig.text(0.247, 0.965, 'Model subset:', fontsize=FS_ANNOT, fontfamily=FONT_FAMILY, ha='left')
    for x, (k, lab) in zip([0.353, 0.420, 0.513], [(10, 'all 10'), (9, '9 trained'), (7, '7 standard-trained')]):
        fig.text(x, 0.965, lab, fontsize=FS_ANNOT, color=SUBSET_COL[k], fontweight='bold',
                 fontfamily=FONT_FAMILY, ha='left')
    # rotated group labels, one per heat panel
    fig.text(0.040, YA0 + HA / 2, 'Adversarial sensitivity', rotation=90, fontsize=6.2,
             color=C_SENSGRP, fontweight='bold', ha='center', va='center', fontfamily=FONT_FAMILY)
    fig.text(0.040, YB0 + HB / 2, 'Gradient spectral\ncharacteristics', rotation=90, fontsize=6.2,
             color=C_CV, fontweight='bold', ha='center', va='center', fontfamily=FONT_FAMILY)

    LY_TOP = 0.955
    for x, y, letter in [(0.013, LY_TOP, 'a'), (0.013, YB0 + HB + 0.048, 'b'),
                         (0.618, LY_TOP, 'c')]:
        fig.text(x, y, letter, fontsize=FS_LET, fontweight='bold', fontfamily=FONT_FAMILY,
                 ha='left', va='top')

    path = save_fig(fig, os.path.join(out_dir, STEM + TAG))
    plt.close(fig)
    print('Saved ->', path)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    ap.add_argument('--outcome', choices=['slope', 'r'], default='slope',
                    help="control outcome: 'slope' (default) | 'r' (companion; filename tag _r)")
    ap.add_argument('--gradsum', choices=['pr', 'cv'], default='pr',
                    help="gradient spectral summary: 'pr' (default) | 'cv' (filename tag _cv)")
    main(**vars(ap.parse_args()))
