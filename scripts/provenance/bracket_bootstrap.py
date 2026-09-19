#!/usr/bin/env python3
"""Seed-level hierarchical bootstrap p-values for the gradient-CV vs adversarial-sensitivity bracket
comparisons (fig6 panel e, per-monkey composite, S53).

Motivation: the per-model-site control score (Pearson r or slope over the accentuation sweep) is itself
a NOISY estimate whose precision depends on how many usable sweep points a site-model has and how
reliable its neural responses were. The Williams/Hotelling test treated each control score as a fixed
point (n = # model-sites), ignoring that error. This bootstrap resamples the ~10 accentuation SEEDS
within each model-site (with replacement), recomputes every model-site's control r / slope, re-
residualises (resid9), and recomputes the ORIENTED gap between the two predictors' correlations with
control -- so the per-model-site estimation error (point count + neural noise, via seed variability)
propagates into the CV-vs-sensitivity comparison.

Predictors are the held-out measures the figures plot: cv from fig9_master (held-out gradient CV),
adv_sens = figure8.load_heldout_logauc (held-out log2-eps AUC). Orientation follows each scope's own
full-sample 10-model direction (multiply by sign, not abs); no suppression on a flip. Two-sided p from
the bootstrap gap distribution crossing 0. Writes figure6/intermediate/bracket_bootstrap_p.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T8 = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, T8); sys.path.insert(0, os.path.join(T8, 'preproc'))
sys.path.insert(0, os.path.join(T8, 'supplementary/SX_predicting', 'scripts'))
sys.path.insert(0, os.path.join(T8, 'supplementary/SX_advrobustness'))
import compute as C
import figure8 as F8

MK = ['red', 'paul', 'venus', 'leap', 'three0']
UNTR = 'AlexNet_training_seed_01'; ROB = ['resnet50_robust', 'clipag_vitb32']; EXTREMES = ROB + [UNTR]
OUTDIR = os.path.join(HERE, '..', 'intermediate'); os.makedirs(OUTDIR, exist_ok=True)


def _pear(x, y):
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.pearsonr(x, y)[0]


def gather():
    mas = pd.read_csv(os.path.join(T8, 'supplementary/SX_predicting', 'intermediate', 'fig9_master.csv'))[
        ['monkey', 'unit', 'model', 'cv', 'site']]
    au = F8.load_heldout_logauc().rename(columns={'channel': 'unit', 'auc': 'adv_sens'})
    df = mas.merge(au[['monkey', 'unit', 'model', 'adv_sens']], on=['monkey', 'unit', 'model'], how='inner')
    clouds = []
    keep = []
    for i, r in df.iterrows():
        bs = C._control_cloud_by_seed(r.monkey, int(r.unit), r.model)      # {seed: (pred[], meas[])}
        cl = [(np.asarray(bs[s][0], float), np.asarray(bs[s][1], float)) for s in bs if len(bs[s][0]) >= 2]
        if len(cl) >= 2:
            clouds.append(cl); keep.append(i)
    return df.loc[keep].reset_index(drop=True), clouds


def main(reps=2000, seed=0):
    df, clouds = gather()
    n = len(df)
    site = df.site.values; model = df.model.values.astype(object); monkey = df.monkey.values
    cv = df.cv.values.astype(float); adv = df.adv_sens.values.astype(float)
    rng = np.random.default_rng(seed)

    def resid9(scores):
        d = pd.DataFrame({'site': site, 'model': model, 'v': scores})
        mu = d[d.model != UNTR].groupby('site')['v'].mean()
        return scores - d['site'].map(mu).values

    def scores_full(kind):
        return np.array([C._outcome(np.concatenate([c[0] for c in cl]),
                                    np.concatenate([c[1] for c in cl]), kind) for cl in clouds])

    subsets = {10: np.ones(n, bool), 9: (model != UNTR),
               8: ~np.isin(model, ROB), 7: ~np.isin(model, EXTREMES)}
    scopes = {'pooled': np.ones(n, bool)}
    for mk in MK:
        scopes[mk] = (monkey == mk)

    def rc_ra(y, mask, lev):
        m = mask & np.isfinite(y)
        if lev == 'model':
            mods = np.unique(model[m])
            if len(mods) < 4:
                return np.nan, np.nan
            mc = np.array([cv[m][model[m] == mm].mean() for mm in mods])
            ma = np.array([adv[m][model[m] == mm].mean() for mm in mods])
            my = np.array([y[m][model[m] == mm].mean() for mm in mods])
            return _pear(mc, my), _pear(ma, my)
        if m.sum() < 4:
            return np.nan, np.nan
        return _pear(cv[m], y[m]), _pear(adv[m], y[m])

    # orientation signs: each scope's own full-sample 10-model direction (fixed across bootstraps)
    signs = {}
    for kind, oc in (('slope', 'control_slope'), ('r', 'control_r')):
        yf = resid9(scores_full(kind))
        for scn, smask in scopes.items():
            for lev in ('site', 'model'):
                rc, ra = rc_ra(yf, smask & subsets[10], lev)
                signs[(scn, oc, lev)] = (np.sign(rc) if np.isfinite(rc) else 1.0,
                                         np.sign(ra) if np.isfinite(ra) else 1.0)

    acc = {}
    for b in range(reps):
        boot = {}
        for kind in ('slope', 'r'):
            sc = np.empty(n)
            for i, cl in enumerate(clouds):
                k = len(cl); pk = rng.integers(0, k, k)
                sc[i] = C._outcome(np.concatenate([cl[j][0] for j in pk]),
                                   np.concatenate([cl[j][1] for j in pk]), kind)
            boot[kind] = sc
        for kind, oc in (('slope', 'control_slope'), ('r', 'control_r')):
            y = resid9(boot[kind])
            for scn, smask in scopes.items():
                for lev in ('site', 'model'):
                    sf_cv, sf_adv = signs[(scn, oc, lev)][0], signs[(scn, oc, lev)][1]
                    for k, kmask in subsets.items():
                        rc, ra = rc_ra(y, smask & kmask, lev)
                        g = rc * sf_cv - ra * sf_adv if np.isfinite(rc) and np.isfinite(ra) else np.nan
                        acc.setdefault((scn, oc, lev, k), []).append(g)

    rows = []
    for (scn, oc, lev, k), gs in acc.items():
        g = np.array([x for x in gs if np.isfinite(x)])
        if len(g) < 20:
            rows.append(dict(scope=scn, outcome=oc, level=lev, subset=k, gap=np.nan, p=np.nan, nboot=len(g)))
        else:
            p = 2.0 * min((g <= 0).mean(), (g >= 0).mean())
            rows.append(dict(scope=scn, outcome=oc, level=lev, subset=k, gap=float(np.mean(g)),
                             p=float(min(p, 1.0)), nboot=len(g)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, 'bracket_bootstrap_p.csv'), index=False)
    print('wrote bracket_bootstrap_p.csv', len(rows), 'rows  (reps=%d, n=%d)' % (reps, n))
    show = out[(out.scope == 'pooled') & (out.outcome == 'control_slope')].sort_values(['level', 'subset'])
    print(show.to_string(index=False))


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(reps=reps)
