#!/usr/bin/env python3
"""S49 predicting_summary — compute every number, cache to preproc_data/sup_predicting_results.pkl.

PORTED 2026-08-15 verbatim from take8/supplementary/scripts/build_predicting_results.py so the
S49 pipeline is self-contained in take9 (paths only; analysis unchanged). Reads
preproc_data/sup_predicting_master.csv (the full 29-column master incl. the ImageNet
columns; formerly take8 intermediate/fig9_master.csv).

PORTED 2026-08-08 from supplementary/SX_predicting/scripts/compute.py (verbatim analysis) so the
S54 pipeline lives in supplementary/ proper; output name now matches what sup_predicting_summary.py
reads directly (no manual copy step). Spectral CoV AND spectral PR (= 111/(1+CoV^2), exact
identity over the same 1:112 band, since fig9_master's cv is SD/mean over those 111 bins) are
BOTH first-class predictor groups — leaderboard, best-subset, and Shapley. PR is a NONLINEAR
monotone transform of CoV, so carrying both lets the linear models capture that curvature.
The former GRADSUM=pr variant build (predicting_results_pr.pkl / figS54b) is retired.

Outcome = SITE-RESIDUALIZED control (per site-model minus the mean over that site's models),
re-residualized within each model family so the 10- and 7-model analyses are internally consistent.
Site-constant predictors (reliability, calibration skewness) are exactly zero under this residual and
are handled separately (the site-mean supplement); the partition therefore uses the 8 site-VARYING
predictors. Everything is cross-validated (repeated 10-fold, out-of-fold R²).

Three analyses, each run for the 10-model family and the 7-model subset (drops the two
adversarially-trained models + the untrained model), for control slope and control r:
  1. single-predictor CV R² ranking (shared slope) -> is gradient flatness the best single predictor?
  2. best per-monkey-slopes model (forward-selected) CV R² vs the split-half explainable ceiling.
  3. Shapley/LMG partition of CV R² across the site-varying predictor groups listed in GROUPS.

Plus the site-mean supplement numbers (per-site mean control vs reliability / calibration skew).

Run:
  python scripts/preprocessing/build_predicting_results.py [--exclude-untrained]
"""
import os
import sys
import pickle
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats

from pnc import paths
from pnc.preproc import loader as L

INT = str(paths.preprocessed_data())
# EXCLUDE_UNTRAINED=1: rebuild the whole analysis WITHOUT the Untrained model (family = 9
# trained models instead of 10; the '10' family key is kept so the renderer needs no rewiring).
EXCL_UNTR = os.environ.get('EXCLUDE_UNTRAINED') == '1' or '--exclude-untrained' in sys.argv
RESULTS = ('sup_predicting_results_no_untrained.pkl' if EXCL_UNTR
           else 'sup_predicting_results.pkl')    # single integrated build: CoV and PR both first-class
MK = ['red', 'paul', 'venus', 'leap', 'three0']
EXTREMES = ['resnet50_robust', 'clipag_vitb32', 'AlexNet_training_seed_01']   # 2 adv-trained + untrained

# ---- predictor definitions --------------------------------------------------
# Site-varying groups used in the Shapley partition (label -> columns); the count is len(GROUPS)
# (ten spectral/encoding/synthesis/readout/attack descriptors plus one ImageNet group).
GROUPS = [
    ('spectral CoV', ['grad_cov']),                     # CoV of the encoding-gradient power spectrum
    ('spectral PR', ['grad_pr']),                       # participation ratio = 111/(1+CoV^2), same band
    ('encoding r (calibration)', ['enc_p1']),
    ('encoding r (control)',     ['enc_p2']),
    ('accentuation-set eff. dim', ['ed_acc']),         # participation-ratio ED of the accentuated code
    ('synth. regularization',    ['noise']),
    ('readout weight norm',      ['w_norm']),          # L2 norm of the linear readout weight vector
    ('adversarial sensitivity',  ['adv_sens']),
    ('accentuation low-freq',    ['lowfrac']),
    ('diff-map flatness',        ['diff_flatness']),    # Wiener flatness of the accentuation pixel diff (29a)
    # ImageNet group: the margin canonically (the monotone FAMILY coordinate); the top-1 LEVEL
    # in the Untrained-excluded build, where the level is the natural monotone variable and
    # beats the margin head-to-head (levels are otherwise leaderboard-only: r=.99 collinear pair)
    (('ImageNet top-1', ['inet_top1']) if EXCL_UNTR
     else ('ImageNet top-5 margin', ['inet_gap'])),
]
GNAMES = [g[0] for g in GROUPS]; GCOLS = {g[0]: g[1] for g in GROUPS}
# single-predictor ranking additionally shows the frequency axes kept out of the partition
RANK = GROUPS + [e for e in (
    ('Wiener flatness', ['ho_flat']),           # convergent gradient measure (leaderboard only, NOT in Shapley)
    ('Λ high-freq energy', ['Lambda']), ('phase reliance', ['phase_reliance']),
    ('ImageNet top-1', ['inet_top1']), ('ImageNet top-5', ['inet_top5']),
    ('ImageNet top-5 margin', ['inet_gap']))
    if e[0] not in {g[0] for g in GROUPS}]      # dedupe: whichever ImageNet var is in GROUPS
OUTCOMES = [('control_slope', 'slope'), ('control_r', 'r')]


# ---- residualization + CV machinery -----------------------------------------
ALLCOLS = sorted({c for _, cs in RANK for c in cs})


def residualize(df, models):
    """Return a family sub-frame with SITE-residualized OUTCOMES and RAW predictors — exactly fig6's
    site estimator: correlate the raw predictor against the control outcome after subtracting the
    per-site mean over the 9 trained models (excl Untrained) from the outcome only. The 10- and 7-model
    families share this one 9-trained reference (resid9). Predictors are left raw (not residualized), so a
    predictor's site r attenuates by its between-site variance fraction and site-constant predictors read
    ~0 — matching fig6 exactly. (Switched 2026-07-28 to fig6's estimator; figS47 still uses within-site.)"""
    d = df[df.model.isin(models)].copy()
    ref = df[df.model != 'AlexNet_training_seed_01']               # 9 trained models = resid9 reference
    for oc, _ in OUTCOMES:
        mu = ref.groupby('site')[oc].mean()
        d[oc + '_res'] = d[oc].values - d['site'].map(mu).values   # outcome only; predictors stay raw (fig6)
    return d.reset_index(drop=True)


def zscore_cols(d, cols):
    d = d.copy()
    for c in cols:
        s = d[c].std()
        d[c] = (d[c] - d[c].mean()) / (s if s > 0 else 1.0)
    return d


def _cv_r2(X, Y, reps=5, k=10):
    out = []
    for s in range(reps):
        idx = np.random.default_rng(s).permutation(len(Y)); pred = np.zeros(len(Y))
        for f in np.array_split(idx, k):
            tr = np.setdiff1d(np.arange(len(Y)), f)
            beta, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None); pred[f] = X[f] @ beta
        ss = ((Y - Y.mean()) ** 2).sum(); out.append(1 - ((Y - pred) ** 2).sum() / ss)
    return float(np.mean(out)), float(np.std(out))


def cv_shared(d, cols, y):
    """Shared-slope CV R² (intercept + one slope per column). Base = intercept only -> ~0."""
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols]) if cols else np.ones((len(d), 1))
    return _cv_r2(X, d[y].values)[0]


def _ols_r2(d, cols, y):
    Y = d[y].values; X = np.column_stack([np.ones(len(Y))] + [d[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); pr = X @ b
    return 1 - ((Y - pr) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()


def raw_assoc(d, cols, y):
    """Bivariate within-site association of a predictor group with the outcome + a p-value.
    Single column -> signed Pearson r; multi-column -> multiple-R (sqrt R²) with an F-test."""
    Y = d[y].values; n = len(Y)
    if len(cols) == 1:
        r, p = stats.pearsonr(d[cols[0]].values, Y); return dict(mag=abs(r), signed=r, p=p)
    r2 = _ols_r2(d, cols, y); k = len(cols)
    F = (r2 / k) / ((1 - r2) / (n - k - 1)); p = float(1 - stats.f.cdf(F, k, n - k - 1))
    return dict(mag=float(np.sqrt(max(r2, 0))), signed=float(np.sqrt(max(r2, 0))), p=p)


def commonality(d, a, b, y):
    """Two-predictor commonality of control variance between predictors a and b (in-sample R²):
    unique-a, shared, unique-b, plus each predictor's bivariate r/p."""
    r2a, r2b, r2ab = _ols_r2(d, [a], y), _ols_r2(d, [b], y), _ols_r2(d, [a, b], y)
    return dict(uniq_a=r2ab - r2b, shared=r2a + r2b - r2ab, uniq_b=r2ab - r2a, r2ab=r2ab,
                ra=stats.pearsonr(d[a].values, d[y].values), rb=stats.pearsonr(d[b].values, d[y].values))


def _pm_design(d, cols):
    """Per-monkey slopes (col x monkey) + per-monkey intercepts."""
    blocks = [(d.monkey == mk).astype(float).values for mk in MK]        # per-monkey intercepts
    for c in cols:
        for mk in MK:
            blocks.append(np.where(d.monkey == mk, d[c].values, 0.0))    # per-monkey slope
    return np.column_stack(blocks)


def cv_pm(d, cols, y, reps=25):
    X = _pm_design(d, cols); return _cv_r2(X, d[y].values, reps=reps)


def kfold_oof_pm(d, cols, y, k=10, seed=0):
    X = _pm_design(d, cols); Y = d[y].values
    idx = np.random.default_rng(seed).permutation(len(Y)); pred = np.full(len(Y), np.nan)
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(np.arange(len(Y)), f)
        beta, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None); pred[f] = X[f] @ beta
    return pred


# ---- explainable ceiling (residualized) -------------------------------------
def _control_cloud_by_seed(mk, unit, model):
    """{seed -> (predicted[], measured[])} for the accentuation sweep (firing-floor clamped, as
    loader.control_cloud). Seeds (10 per readout, 11 levels each) are the experiment's repeat unit."""
    P = L.load_predictions(mk)
    mi = list(P['pred_models']).index(model); ui = list(P['target_units']).index(unit)
    sel = (P['gen_model'] == model) & (P['gen_unit'] == unit)
    resp = L._acc_response(mk, unit); fl = L.firing_floor(mk, unit)
    byseed = {}
    for n, p, s in zip(P['stim'][sel], P['pred'][sel, mi, ui], P['gen_seed'][sel]):
        if n in resp:
            a, b = byseed.setdefault(int(s), ([], []))
            a.append(max(float(p), fl)); b.append(resp[n])
    return {s: (np.array(a), np.array(b)) for s, (a, b) in byseed.items()}


def _outcome(x, y, kind):
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan
    return stats.pearsonr(x, y)[0] if kind == 'r' else stats.linregress(x, y).slope


def ceiling_resid(models, kind, mk=None, n_splits=200, seed=0):
    """Explainable-variance (R²) ceiling of the SITE-RESIDUALIZED outcome via a SEED split-half,
    Spearman-Brown corrected, AVERAGED over random 5/5 seed partitions. For each partition the control
    slope/r is recomputed on each readout's half-A seeds and half-B seeds, each half is residualized by
    its within-site (within-family) mean, the two half-outcomes are correlated across readouts, and SB-
    corrected to full length. Randomizing the split (rather than a fixed odd/even one) avoids confounding
    the partition with the specific seed images. Returns (mean_ceiling, sd) over partitions. Because CV R²
    is bounded by the outcome reliability, this ρ is the R²-ceiling directly (the correlation-r ceiling
    would be √ρ). If `mk` is given, restrict to that monkey's units (per-monkey ceiling)."""
    cfg = L.config(); rng = np.random.default_rng(seed)
    recs = []                                        # (site, {seed: (x, y)})
    seeds = set()
    for mk_ in cfg['monkeys']:
        if mk is not None and mk_ != mk:
            continue
        for u in L.load_brain(mk_)['units']:
            for mdl in models:
                bs = _control_cloud_by_seed(mk_, int(u), mdl)
                recs.append((f'{mk_}_u{u}', bs)); seeds.update(bs.keys())
    seeds = sorted(seeds); h = len(seeds) // 2
    rels = []
    for _ in range(n_splits):
        perm = rng.permutation(seeds); SA, SB = set(perm[:h].tolist()), set(perm[h:2 * h].tolist())
        rowsA, rowsB = [], []
        for site, bs in recs:
            xa = np.concatenate([bs[s][0] for s in bs if s in SA]) if any(s in SA for s in bs) else np.array([])
            ya = np.concatenate([bs[s][1] for s in bs if s in SA]) if any(s in SA for s in bs) else np.array([])
            xb = np.concatenate([bs[s][0] for s in bs if s in SB]) if any(s in SB for s in bs) else np.array([])
            yb = np.concatenate([bs[s][1] for s in bs if s in SB]) if any(s in SB for s in bs) else np.array([])
            rowsA.append((site, _outcome(xa, ya, kind))); rowsB.append((site, _outcome(xb, yb, kind)))
        A = pd.DataFrame(rowsA, columns=['site', 'v']); B = pd.DataFrame(rowsB, columns=['site', 'v'])
        A['r'] = A['v'] - A.groupby('site')['v'].transform('mean')                # residualize within site
        B['r'] = B['v'] - B.groupby('site')['v'].transform('mean')
        ok = A['r'].notna() & B['r'].notna()
        if ok.sum() < 5:
            continue
        rho = stats.pearsonr(A['r'][ok], B['r'][ok])[0]
        rels.append(2 * rho / (1 + rho))
    return float(np.mean(rels)), float(np.std(rels))


# ---- Shapley over the site-varying groups (len(GROUPS)) -----------------------
def shapley(d, y):
    # subset CV R² floored at -1 ("worse than the null by more than the total variance"):
    # small within-monkey slices can produce degenerate out-of-fold fits for near-collinear
    # subsets, whose unbounded negative scores would otherwise dominate the marginals.
    cache = {frozenset(S): max(cv_shared(d, [c for n in S for c in GCOLS[n]], y), -1.0)
             for r in range(len(GNAMES) + 1) for S in combinations(GNAMES, r)}
    k = len(GNAMES); phi = {}
    for n in GNAMES:
        others = [x for x in GNAMES if x != n]; tot = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(k - 1 - r) / factorial(k)
            for S in combinations(others, r):
                tot += w * (cache[frozenset(S) | {n}] - cache[frozenset(S)])
        phi[n] = tot
    return phi, cache[frozenset()], cache[frozenset(GNAMES)]


# ---- best per-monkey model: EXHAUSTIVE best-subset over all 2^8 predictor subsets ----
def best_model(d, y):
    """Provably-best predictor combination for the current predictor set: search every non-empty
    subset of the site-varying predictors in GROUPS, score each by per-monkey-slopes cross-validated R²,
    keep the maximum (ascending subset size, so ties resolve to the more parsimonious model). Re-run
    the winner at full repeats for the reported CV R² and the panel-B out-of-fold scatter. Because it
    ranges over GNAMES, the selection is automatically re-estimated whenever a predictor changes."""
    best_s, best_S = -np.inf, ()
    for r in range(1, len(GNAMES) + 1):
        for S in combinations(GNAMES, r):
            s = cv_pm(d, [c for n in S for c in GCOLS[n]], y, reps=8)[0]      # fast search pass
            if s > best_s:
                best_s, best_S = s, S
    chosen = list(best_S); cols = [c for n in chosen for c in GCOLS[n]]
    r2, se = cv_pm(d, cols, y, reps=25)                                       # precise re-estimate of the winner
    pred = kfold_oof_pm(d, cols, y); true = d[y].values
    return dict(chosen=chosen, cols=cols, cvr2=r2, se=se,
                pred=pred, true=true, monkey=d.monkey.values, model=d.model.values)


def _enc_p2_heldout(df):
    """control-phase encoding r over the re-presented HELD-OUT calibration images.

    The cached `enc_p2` column in sup_predicting_master.csv was computed over EVERY
    re-presented calibration image, i.e. including model-training images. The
    main text defines the cross-phase re-test on held-out validation images only
    (Fig 3c phase 2), so recompute it here to match.
    """
    cache, out = {}, []
    for mk, u, m in zip(df.monkey, df.unit, df.model):
        key = (mk, int(u), m)
        if key not in cache:
            b = L.load_brain(mk); c = b['control']; ui = list(b['units']).index(int(u))
            e = L.load_encoding(mk); mi = list(e['models']).index(m)
            ep = dict(zip(e['stim'].tolist(), e['pred'][list(e['units']).index(int(u)), mi]))
            heldout = L._test_names(mk)
            x, y = [], []
            for i, (n, k) in enumerate(zip(c['stim'], c['kind'])):
                if k == 'calibration' and n in ep and (heldout is None or n in heldout):
                    x.append(ep[n]); y.append(c['resp_z'][i, ui])
            x = np.asarray(x, float); y = np.asarray(y, float)
            ok = np.isfinite(x) & np.isfinite(y)
            cache[key] = (float(stats.pearsonr(x[ok], y[ok])[0])
                          if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan)
        out.append(cache[key])
    return out


# ---- main -------------------------------------------------------------------
def main():
    df = pd.read_csv(os.path.join(INT, 'sup_predicting_master.csv'))
    if EXCL_UNTR:
        df = df[df.model != 'AlexNet_training_seed_01'].reset_index(drop=True)
        print('EXCLUDE_UNTRAINED=1: Untrained dropped; primary family = 9 trained models')
    # both gradient summaries as first-class predictors (PR = exact identity over the
    # same 1:112 band, 111 bins, population-SD CoV — verified vs profiles)
    df['grad_cov'] = df['cv']
    df['grad_pr'] = 111.0 / (1.0 + df['cv'] ** 2)
    # control-phase encoding r: recomputed over held-out re-presented images only,
    # matching the main-text cross-phase definition (cached column mixed in training images)
    df['enc_p2'] = _enc_p2_heldout(df)
    fams = {10: sorted(df.model.unique()), 7: [m for m in df.model.unique() if m not in EXTREMES]}
    res = dict(meta=dict(GNAMES=GNAMES, GCOLS=GCOLS,
                         RANK=[r[0] for r in RANK], RANKCOLS={r[0]: r[1] for r in RANK},
                         fams={k: list(v) for k, v in fams.items()}),
               ranking={}, rawcorr={}, common={}, shapley={}, ceiling={}, ceiling_sd={},
               best={}, flat_only={}, n={})

    for fam, models in fams.items():
        d0 = residualize(df, models); res['n'][fam] = len(d0)
        d = zscore_cols(d0, ALLCOLS)
        for oc, tag in OUTCOMES:
            y = oc + '_res'
            res['ceiling'][(fam, tag)], res['ceiling_sd'][(fam, tag)] = ceiling_resid(models, tag)
            res['ranking'][(fam, tag)] = {name: cv_shared(d, cols, y) for name, cols in RANK}
            res['rawcorr'][(fam, tag)] = {name: raw_assoc(d, cols, y) for name, cols in RANK}
            res['common'][(fam, tag)] = commonality(d, 'cv', 'lowfrac', y)
            res['shapley'][(fam, tag)] = shapley(d, y)
            res['best'][(fam, tag)] = best_model(d, y)
            res['flat_only'][(fam, tag)] = cv_pm(d, ['cv'], y, reps=25)[0]   # concentration-only baseline
            print(f'[{fam}-model | {tag}]  n={len(d)}  ceiling={res["ceiling"][(fam,tag)]:.3f}  '
                  f'best CV R²={res["best"][(fam,tag)]["cvr2"]:.3f} '
                  f'({100*res["best"][(fam,tag)]["cvr2"]/res["ceiling"][(fam,tag)]:.0f}% of ceiling)  '
                  f'chosen={res["best"][(fam,tag)]["chosen"]}')

    # ---- per-monkey breakdown (panel c): best-model variance explained + top-Shapley predictor ----
    # 10-model family, control-slope residual. Best-model R² per monkey is evaluated on that monkey's
    # slice of the pooled out-of-fold predictions; the top predictor is the argmax within-monkey Shapley.
    res['per_monkey'] = {}
    d10 = zscore_cols(residualize(df, fams[10]), ALLCOLS); y = 'control_slope_res'
    best = res['best'][(10, 'slope')]; pred, true, mks = best['pred'], best['true'], best['monkey']
    for mk in MK:
        sel = mks == mk
        if sel.sum() < 4:
            continue
        t, p = true[sel], pred[sel]
        r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        dm = d10[d10.monkey == mk].reset_index(drop=True)
        phi, _, full = shapley(dm, y)                      # within-monkey Shapley (shared-slope CV)
        top = max(phi, key=lambda n: phi[n])
        ceil, _ = ceiling_resid(fams[10], 'slope', mk=mk, n_splits=120)
        res['per_monkey'][mk] = dict(r2=float(r2), ceiling=float(ceil), full=float(full),
                                     top=top, top_phi=float(phi[top]), n=int(sel.sum()))
        print(f'  per-monkey {mk}: best R²={r2:+.3f}  ceiling={ceil:.3f}  top-Shapley={top} ({phi[top]:+.3f})')

    # ---- Jacobian-pair analysis (MAIN, companion to fig10): flatness vs adv. sensitivity ----
    # Per-monkey slopes/intercepts CV R² for flatness-only, adv-sens-only, and both; the unique/shared
    # decomposition is commonality on those SAME CV R²s. Scatter uses the both-predictor per-monkey model.
    res['pair'] = {}
    for fam, models in fams.items():
        d = zscore_cols(residualize(df, models), ALLCOLS)
        for oc, tag in OUTCOMES:
            y = oc + '_res'
            cvf = cv_pm(d, ['cv'], y, reps=25)[0]
            cva = cv_pm(d, ['adv_sens'], y, reps=25)[0]
            cvb, seb = cv_pm(d, ['cv', 'adv_sens'], y, reps=25)
            # pair-panel ImageNet comparator: margin canonically; top-1 LEVEL in the
            # Untrained-excluded build (there the level is the natural monotone variable)
            mcol = 'inet_top1' if EXCL_UNTR else 'inet_gap'
            cvm = cv_pm(d, [mcol], y, reps=25)[0]
            cvbm, sebm = cv_pm(d, ['cv', mcol], y, reps=25)
            res['pair'][(fam, tag)] = dict(
                cv_flat=cvf, cv_adv=cva, cv_both=cvb, se_both=seb,
                uniq_flat=cvb - cva, uniq_adv=cvb - cvf, shared=cvf + cva - cvb,
                cv_mrg=cvm, cv_bothm=cvbm, se_bothm=sebm,
                uniq_flat_m=cvbm - cvm, uniq_mrg=cvbm - cvf, shared_m=cvf + cvm - cvbm,
                cv_covS=cv_pm(d, ['grad_cov'], y, reps=25)[0],          # both gradient summaries,
                cv_prS=cv_pm(d, ['grad_pr'], y, reps=25)[0],            # variant-independent (panel b)
                cv_bothm_cov=cv_pm(d, ['grad_cov', mcol], y, reps=25)[0],
                cv_bothm_pr=cv_pm(d, ['grad_pr', mcol], y, reps=25)[0],
                ceiling=res['ceiling'][(fam, tag)])
            print(f'  pair [{fam}|{tag}]: flat={cvf:.3f} adv={cva:.3f} both={cvb:.3f}  '
                  f'uniqF={cvb-cva:+.3f} shared={cvf+cva-cvb:+.3f} uniqA={cvb-cvf:+.3f} | '
                  f'mrg={cvm:.3f} both_m={cvbm:.3f} uniqF_m={cvbm-cvm:+.3f} '
                  f'shared_m={cvf+cvm-cvbm:+.3f} uniqM={cvbm-cvf:+.3f}')
    res['pair_scatter'] = {}
    for oc, tag in OUTCOMES:
        d = zscore_cols(residualize(df, fams[10]), ALLCOLS); y = oc + '_res'
        res['pair_scatter'][tag] = dict(pred=kfold_oof_pm(d, ['cv', 'adv_sens'], y),
                                        true=d[y].values, monkey=d.monkey.values, model=d.model.values)

    # ---- site-mean supplement: per-site mean control vs site-level factors ----
    sm = df.groupby('site').agg(monkey=('monkey', 'first'), region=('region', 'first'),
                                reliability=('reliability', 'first'), calib_skew=('calib_skew', 'first'),
                                mean_slope=('control_slope', 'mean'), mean_r=('control_r', 'mean')).reset_index()
    supp = dict(n_sites=len(sm), table=sm)
    for oc, tag in [('mean_slope', 'slope'), ('mean_r', 'r')]:
        supp[('rel', tag)] = stats.pearsonr(sm['reliability'], sm[oc])
        supp[('skew', tag)] = stats.pearsonr(sm['calib_skew'], sm[oc])
        X = np.column_stack([np.ones(len(sm)), stats.zscore(sm['reliability']), stats.zscore(sm['calib_skew'])])
        b, *_ = np.linalg.lstsq(X, sm[oc].values, rcond=None); pr = X @ b
        supp[('R2', tag)] = 1 - ((sm[oc].values - pr) ** 2).sum() / ((sm[oc].values - sm[oc].mean()) ** 2).sum()
    res['supp'] = supp

    with open(os.path.join(INT, RESULTS), 'wb') as fh:
        pickle.dump(res, fh)
    print('\nsite-mean supp: rel~mean_slope r=%.2f (p=%.3f), rel~mean_r r=%.2f (p=%.3f); '
          'R²[rel+skew] slope=%.2f r=%.2f (n=%d sites)' % (
        supp[('rel', 'slope')][0], supp[('rel', 'slope')][1], supp[('rel', 'r')][0], supp[('rel', 'r')][1],
        supp[('R2', 'slope')], supp[('R2', 'r')], supp['n_sites']))
    print('wrote', os.path.join(INT, RESULTS))


if __name__ == '__main__':
    main()
