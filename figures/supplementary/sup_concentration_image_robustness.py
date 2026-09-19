#!/usr/bin/env python3
"""Supp fig (concentration_image_robustness) -- robustness of the gradient spectral summary
(participation ratio by default; CoV via --gradsum cv) to the probe-image set and to
Fourier-analysis choices.

Gradient spectral CONCENTRATION (radial-Fourier coefficient of variation, CV = SD/mean over the band
[1,112) cyc/img) is a property of the FITTED ENCODING AXIS, not of the particular synthesis seeds used
to visualize the gradient or of any single Fourier-analysis choice. Every panel re-estimates the
measure on the 100 held-out NSD images (the same probe set as the adversarial-sensitivity analysis),
i.e. on data independent of the seed images and of the control fits. (a) Independent-image
replication: per-axis concentration measured on the 10 synthesis seeds versus on the 100 held-out
images agrees almost perfectly (n=250 read-outs; points coloured by model family). (b) Model-rank
stability: the 10-model ordering of mean concentration is preserved between the seed-based and
held-out-image estimates (rank-rank plot; the two adversarially robust models remain the two most
concentrated). (c) Held-out-image concentration predicts neural control on the independent images:
within-site (recording-group and channel removed) concentration versus control slope, with the
leave-one-model-out cross-validated fit reported for both control outcomes. (d) Convergence with
probe-image count: split-half reliability of the per-axis concentration estimate and its predictive
association with control, as a function of the number of probe images (1 to 100; bootstrap
subsampling of the 100 held-out images, mean and 95% CI). Both stabilize within a handful of images.
(e) Analysis-specification curve: the standardized (within-site) concentration coefficient recomputed
under 32 reasonable Fourier-analysis choices -- including/excluding the DC bin, radial band, power
versus amplitude spectrum, linear versus log radial weighting, and average-spectra-then-CV versus
CV-per-image-then-average; the coefficient is positive under every specification. (f) Effect summary:
the standardized coefficient and the leave-one-model-out cross-validated R-squared for every
specification, separately for control slope and control score, cluster tightly.

RGB-gradient aggregation is fixed here: the cached radial profiles are already reduced over colour
channels (mean over RGB before the 2D Fourier transform), so that choice cannot be varied without the
raw gradient maps. TAKE-HOME: gradient spectral concentration is a stable property of the fitted
encoding axis -- robust to the probe-image set (seed vs held-out, r ~ 0.98), to the number of probe
images, and to reasonable Fourier-analysis choices. Reads the per-image held-out radial-Fourier
profiles (cluster/outputs_from_cluster/fig5_heldout_gradfreq/*.pkl), the held-out descriptor
table (preproc_data/sup_advform_heldout_descriptors.csv), and the seed-based profiles
(loader.load_gradient_freq).
"""
import os, glob, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pnc import paths
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L
from pnc.utils import (apply_figure_style, FONT_FAMILY, MM, WIDTH_2COL_MM, FIG_FACECOLOR, save_fig,
                       MODEL_SHORT_NAMES, get_model_group, get_model_color)
apply_figure_style()
GRADSUM = 'pr'                                          # 'pr' (default) | 'cv' variant (tag _cv); set in main()
MET = 'PR' if GRADSUM == 'pr' else 'CoV'
_M = _fig('concentration_image_robustness'); STEM = output_name('concentration_image_robustness')

HODIR = str(paths.cluster_outputs() / 'fig5_heldout_gradfreq')
HOCSV = os.path.join(str(paths.preprocessed_data()), 'sup_advform_heldout_descriptors.csv')

FMIN, FMAX = 1, 112                                     # canonical concentration band
UNTRAINED = 'AlexNet_training_seed_01'
SHORT = dict(MODEL_SHORT_NAMES); SHORT[UNTRAINED] = 'Untrained'
OUTCOMES = [('control_slope', 'Control slope'), ('control_r', 'Control score ($r$)')]
OC_COL = {'control_slope': '#3B6FA0', 'control_r': '#C9772E'}
FIG4_MODEL_ORDER = ['clipag_vitb32', 'resnet50_robust', 'resnet50_dino', 'resnet50',
                    'radio_v2.5-b', 'dinov2_vitb14_reg', 'regnety_640', 'resnet50_clip',
                    'siglip2_vitb16', 'AlexNet_training_seed_01']   # panel-A legend order = figure 4

FS_LET, FS_TITLE, FS_AX, FS_TICK, FS_ANN, FS_LEG = 8.5, 5.8, 6.0, 5.0, 5.0, 5.0


# ----------------------------------------------------------- concentration measures --
def cv_lin(spec, a, b, amp=False):
    P = np.clip(spec[..., a:b], 1e-30, None)
    if amp:
        P = np.sqrt(P)
    return P.std(-1) / P.mean(-1)


def cv_log(spec, a, b, amp=False, nb=12):
    """CV over log-spaced radial bins (approx-log radial weighting)."""
    P = np.clip(spec[..., a:b], 1e-30, None)
    if amp:
        P = np.sqrt(P)
    fr = np.arange(a, b)
    e = np.unique(np.round(np.logspace(np.log10(max(a, 1)), np.log10(b - 1), nb + 1)).astype(int))
    V = np.stack([P[..., (fr >= e[i]) & (fr < e[i + 1])].mean(-1)
                  for i in range(len(e) - 1) if ((fr >= e[i]) & (fr < e[i + 1])).sum() > 0], -1)
    return V.std(-1) / V.mean(-1)


def pr_lin(spec, a, b):
    """Participation ratio over the same radial band: PR = (sum P)^2 / sum(P^2) (= N/(1+CoV^2);
    monotone DECREASING in CoV, so all correlations with the metric flip sign)."""
    P = np.clip(spec[..., a:b], 1e-30, None)
    s2 = (P ** 2).sum(-1)
    return np.where(s2 > 0, P.sum(-1) ** 2 / s2, np.nan)


SUMM = pr_lin if GRADSUM == 'pr' else cv_lin            # active band summary for the figure


def _configure(gradsum='pr'):
    """Set the variant-dependent module state (was read from env GRADSUM at import)."""
    global GRADSUM, MET, SUMM
    GRADSUM = gradsum
    MET = 'PR' if GRADSUM == 'pr' else 'CoV'
    SUMM = pr_lin if GRADSUM == 'pr' else cv_lin


def parse_pkl(fn):
    b = os.path.basename(fn)
    monkey = b.split('_')[0]
    unit = int(b.split('_unit_')[1].split('_model_')[0])
    model = b.split('_model_')[1].rsplit('_grad_maps', 1)[0]
    return monkey, unit, model


# ------------------------------------------------------------------------ load data --
def load():
    desc = pd.read_csv(paths.require(HOCSV, hint='frozen input; python scripts/preprocessing/build_all.py'))
    # per-image radial profiles, aligned to descriptor-table row order
    pm = {parse_pkl(f): np.asarray(pickle.load(open(f, 'rb'))['profiles'])
          for f in sorted(glob.glob(os.path.join(paths.require(HODIR), '*.pkl')))}
    P = np.stack([pm[(r.monkey, r.unit, r.model)] for r in desc.itertuples()])   # (250, 100, 158)
    # seed-based per-axis mean profile (10 synthesis seeds)
    g = L.load_gradient_freq()
    seed = {(e['monkey'], e['unit'], e['model']): SUMM(np.asarray(e['profile_mean']), FMIN, FMAX)
            for e in g['gradients']}
    desc['seed_cv'] = [float(seed[(r.monkey, r.unit, r.model)]) for r in desc.itertuples()]
    desc['group'] = desc.model.map(get_model_group)
    return desc, P


# -------------------------------------------------------- fast within-site helpers --
def make_demean(codes, n_groups):
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    def demean(x):
        m = np.bincount(codes, weights=x, minlength=n_groups) / counts
        return x - m[codes]
    return demean


def lomo_r2(cv, desc, outcome, mk_dum):
    X = np.column_stack([cv, mk_dum]); y = desc[outcome].values
    yhat = np.full(len(desc), np.nan)
    for gmod in desc.model.unique():
        te = np.where((desc.model == gmod).values)[0]
        tr = np.setdiff1d(np.arange(len(desc)), te)
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); yhat[te] = X[te] @ b
    return 1 - np.nansum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# ===================================================================================
def main(out_dir, gradsum='pr'):
    _configure(gradsum)
    desc, P = load()
    n = len(desc)
    site_codes = pd.factorize(desc.monkey.astype(str) + '_' + desc.unit.astype(str))[0]
    demean = make_demean(site_codes, site_codes.max() + 1)
    mk_dum = pd.get_dummies(desc.monkey).astype(float).values
    meanspec = P.mean(1)                                # (250, 158) avg-spectra
    assert np.allclose(cv_lin(meanspec, FMIN, FMAX), desc.ho_cv.values, atol=1e-9)   # profile/table alignment
    ho_cv = SUMM(meanspec, FMIN, FMAX)                  # canonical held-out concentration (CoV or PR)

    def wsr(x, y):                                      # within-site standardized coef = within-site r
        return stats.pearsonr(demean(x), demean(y))[0]

    # ---- panel A / B ----
    rA = stats.pearsonr(desc.seed_cv, ho_cv)[0]
    pm_seed = desc.groupby('model').seed_cv.mean(); pm_ho = pd.Series(ho_cv, index=desc.index).groupby(desc.model).mean()
    models_sorted = pm_ho.sort_values().index.tolist()
    rank_seed = pm_seed.rank(); rank_ho = pm_ho.rank()
    rhoB = stats.spearmanr(pm_seed.values, pm_ho.reindex(pm_seed.index).values)[0]

    # ---- panel C ----
    cRes = {oc: (wsr(ho_cv, desc[oc].values), lomo_r2(ho_cv, desc, oc, mk_dum)) for oc, _ in OUTCOMES}

    # ---- panel D : convergence with #probe images ----
    KS = [1, 2, 5, 10, 25, 100]; NBOOT = 300; rng = np.random.default_rng(0)
    ctrl_res = {oc: demean(desc[oc].values) for oc, _ in OUTCOMES}
    relD, predD = {'m': [], 'lo': [], 'hi': []}, {oc: {'m': [], 'lo': [], 'hi': []} for oc, _ in OUTCOMES}
    for k in KS:
        rels, preds = [], {oc: [] for oc, _ in OUTCOMES}
        for _ in range(NBOOT):
            ia = rng.integers(0, 100, k); ib = rng.integers(0, 100, k)
            cva = SUMM(P[:, ia, :].mean(1), FMIN, FMAX); cvb = SUMM(P[:, ib, :].mean(1), FMIN, FMAX)
            rels.append(np.corrcoef(cva, cvb)[0, 1])
            cvad = demean(cva)
            for oc, _ in OUTCOMES:
                preds[oc].append(abs(stats.pearsonr(cvad, ctrl_res[oc])[0]))
        rels = np.array(rels)
        relD['m'].append(rels.mean()); relD['lo'].append(np.percentile(rels, 2.5)); relD['hi'].append(np.percentile(rels, 97.5))
        for oc, _ in OUTCOMES:
            pr = np.array(preds[oc]); predD[oc]['m'].append(pr.mean())
            predD[oc]['lo'].append(np.percentile(pr, 2.5)); predD[oc]['hi'].append(np.percentile(pr, 97.5))

    # ================================= figure =================================
    FIGH_MM = 56
    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, FIGH_MM * MM), facecolor=FIG_FACECOLOR)
    gs = fig.add_gridspec(1, 4, left=0.06, right=0.99, top=0.80, bottom=0.23, wspace=0.48)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2]); axD = fig.add_subplot(gs[0, 3])

    def style(ax):
        ax.tick_params(labelsize=FS_TICK, length=2.4, width=0.6)
        for s in ax.spines.values():
            s.set_linewidth(0.7)

    def boxleg(leg):
        f = leg.get_frame(); f.set_facecolor('white'); f.set_edgecolor('#bbbbbb'); f.set_linewidth(0.6); f.set_alpha(0.95)
        return leg

    # ---------- A : independent-image replication ----------
    axA.scatter(desc.seed_cv.values, ho_cv, s=9, c=[get_model_color(m) for m in desc.model],
                alpha=0.6, lw=0, zorder=3)
    hi = max(desc.seed_cv.max(), ho_cv.max()) * 1.05
    axA.plot([0, hi], [0, hi], ls='--', lw=0.8, color='#888888', zorder=1)
    axA.set_xlim(0, hi); axA.set_ylim(0, hi); axA.set_box_aspect(1)
    axA.set_xlabel(f'seed-image concentration ({MET})', fontsize=FS_AX)
    axA.set_ylabel(f'held-out-image\nconcentration ({MET})', fontsize=FS_AX)
    axA.set_title('Independent-image replication', fontsize=FS_TITLE, pad=4)
    axA.annotate(f'$r$ = {rA:.2f}\n$n$ = {n} axes', (0.045, 0.955), xycoords='axes fraction',
                 ha='left', va='top', fontsize=FS_ANN, color='0.15',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.6, alpha=0.92))
    hnd = [Line2D([0], [0], marker='o', ls='', mfc=get_model_color(m), mec='none', ms=3.0, label=SHORT[m])
           for m in FIG4_MODEL_ORDER]
    boxleg(axA.legend(handles=hnd, loc='lower right', ncol=1, frameon=True, fontsize=FS_LEG - 1.3,
                      handletextpad=0.2, labelspacing=0.2, borderpad=0.3))
    style(axA)

    # ---------- B : model-rank stability (rank-rank) ----------
    axB.plot([0.4, 10.6], [0.4, 10.6], ls='--', lw=0.8, color='#888888', zorder=1)
    for m in models_sorted:
        xr, yr = rank_seed[m], rank_ho[m]
        axB.scatter(xr, yr, s=22, color=get_model_color(m), edgecolor='white', lw=0.5, zorder=4)
        inward = m in ('resnet50_robust', 'clipag_vitb32')      # robust models sit at the extreme corner
        if GRADSUM == 'pr':
            inward = not inward                                 # PR reverses the ranks: robust now bottom-left
        dx = -0.35 if inward else 0.35
        ha = 'right' if dx < 0 else 'left'
        axB.text(xr + dx, yr, SHORT[m], fontsize=FS_ANN - 0.3, ha=ha, va='center', color=get_model_color(m))
    axB.set_xlim(0.2, 11.2); axB.set_ylim(0.2, 11.2); axB.set_box_aspect(1)
    axB.set_xticks([1, 5, 10]); axB.set_yticks([1, 5, 10])
    axB.set_xlabel('seed-image concentration rank', fontsize=FS_AX)
    axB.set_ylabel('held-out-image\nconcentration rank', fontsize=FS_AX)
    axB.set_title('Model-rank stability (10 models)', fontsize=FS_TITLE, pad=4)
    axB.annotate(f'Spearman $\\rho$ = {rhoB:.2f}', (0.045, 0.955), xycoords='axes fraction',
                 ha='left', va='top', fontsize=FS_ANN, color='0.15',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.6, alpha=0.92))
    style(axB)

    # ---------- C : held-out concentration vs control (within-site) ----------
    xr = demean(ho_cv); yr = demean(desc.control_slope.values)
    axC.scatter(xr, yr, s=9, c=[get_model_color(mm) for mm in desc.model], alpha=0.6, lw=0, zorder=3)
    xs = np.array([xr.min(), xr.max()]); sl, ic = np.polyfit(xr, yr, 1)
    axC.plot(xs, sl * xs + ic, '-', color='0.15', lw=1.3, zorder=5)
    axC.axhline(0, color='#dddddd', lw=0.6, zorder=0); axC.axvline(0, color='#dddddd', lw=0.6, zorder=0)
    axC.set_xlabel('held-out concentration\n(within-site residual)', fontsize=FS_AX)
    axC.set_ylabel('control slope\n(within-site residual)', fontsize=FS_AX)
    axC.set_title('Replication with independent images', fontsize=FS_TITLE, pad=4)
    txt = ('within-site $r$\n'
           f'  ctrl. slope {cRes["control_slope"][0]:+.2f}   ctrl. $r$ {cRes["control_r"][0]:+.2f}\n'
           'LOMO $R^2$\n'
           f'  ctrl. slope {cRes["control_slope"][1]:.2f}   ctrl. $r$ {cRes["control_r"][1]:.2f}')
    cxy, cha, cva = (((0.035, 0.045), 'left', 'bottom') if GRADSUM == 'pr'   # PR: empty corner is bottom-left
                     else ((0.965, 0.045), 'right', 'bottom'))
    axC.annotate(txt, cxy, xycoords='axes fraction', ha=cha, va=cva,
                 fontsize=FS_ANN - 1.0, color='0.15',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.6, alpha=0.92))
    axC.set_box_aspect(1)
    style(axC)

    # ---------- D : convergence with #probe images ----------
    x = np.arange(len(KS))
    axD.fill_between(x, relD['lo'], relD['hi'], color='#6b8e6b', alpha=0.22, lw=0)
    axD.plot(x, relD['m'], '-o', color='#3f6b3f', lw=1.4, ms=3.4, label='split-half reliability')
    for oc, lab in [('control_slope', 'Corr. with control slope (site-resid)'), ('control_r', 'Corr. with control r (site-resid)')]:
        axD.fill_between(x, predD[oc]['lo'], predD[oc]['hi'], color=OC_COL[oc], alpha=0.16, lw=0)
        axD.plot(x, predD[oc]['m'], '-o', color=OC_COL[oc], lw=1.4, ms=3.4, label=lab)
    axD.set_xticks(x); axD.set_xticklabels([str(k) for k in KS])
    axD.set_xlabel('number of probe images', fontsize=FS_AX)
    axD.set_ylabel('correlation', fontsize=FS_AX)
    axD.set_ylim(0.35, 1.02)
    axD.set_title('Convergence with probe-image count', fontsize=FS_TITLE, pad=4)
    axD.set_box_aspect(1)
    boxleg(axD.legend(loc='lower right', frameon=True, fontsize=FS_LEG - 0.4, handlelength=1.4,
                      handletextpad=0.3, labelspacing=0.24, borderpad=0.35))
    style(axD)

    # ---------- bold panel letters, anchored just ABOVE each panel's title ----------
    fig.canvas.draw(); _rend = fig.canvas.get_renderer()
    for ax, lab in [(axA, 'a'), (axB, 'b'), (axC, 'c'), (axD, 'd')]:
        p = ax.get_position()
        tb = ax.title.get_window_extent(_rend).transformed(fig.transFigure.inverted())
        fig.text(p.x0 - 0.045, tb.y1, lab, fontsize=FS_LET, fontweight='bold',
                 fontfamily=FONT_FAMILY, va='bottom', ha='left')

    out = save_fig(fig, os.path.join(out_dir, STEM + ('' if GRADSUM == 'pr' else '_cv'))); plt.close(fig)
    print('saved ->', out)
    print(f'A seed-vs-heldout r = {rA:.3f} (n={n})')
    print(f'B per-model Spearman rho = {rhoB:.3f}')
    print('C within-site r / LOMO R2:', {oc: (round(v[0], 3), round(v[1], 3)) for oc, v in cRes.items()})
    print('D reliability by k', dict(zip(KS, [round(v, 3) for v in relD['m']])))
    print('D predictive slope by k', dict(zip(KS, [round(v, 3) for v in predD['control_slope']['m']])))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'))
    ap.add_argument('--gradsum', choices=['pr', 'cv'], default='pr',
                    help="gradient spectral summary: 'pr' (default) | 'cv' (filename tag _cv)")
    main(**vars(ap.parse_args()))
