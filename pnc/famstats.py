"""Family-gap estimators for the robadvctl supplementary figures.

The estimand is the adversarially-trained minus conventionally-trained control gap
among the 9 trained models (Untrained is a reference class, never in the test).
Two principled forms:

  raw       adv_gap(df, y)             site-fixed-effects OLS `y ~ adv + C(site)`;
                                       p from the exact within-site paired Wilcoxon
                                       over per-site family means (n = sites) - the design is balanced, so the FE coefficient
                                       equals the plain family-mean gap.
  adjusted  adv_gap(df, y, covars)     ANCOVA `y ~ adv + covars + C(site)` - the
                                       covariate enters the model jointly (residualize-
                                       then-test is only valid when family and covariate
                                       are uncorrelated); p from site-clustered SEs.

partial_residuals() produces DISPLAY values consistent with the ANCOVA: covariate and
site effects removed with coefficients from the joint fit (model structure held in the
model, so family signal is not absorbed into the covariate slope).

Covariates here (synthesis-hp regime, encoding-axis alignment) are post-treatment - properties produced by or selected in response to the trained model - so the adjusted
gap is a SENSITIVITY analysis ("the advantage is not explained by X"), not confound
removal.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

UNTR = 'AlexNet_training_seed_01'


def _site(d):
    return d['monkey'].astype(str) + '_u' + d['unit'].astype(int).astype(str)


def adv_gap(df, outcome, covariates=()):
    """Adv-vs-conventional gap among the 9 trained models. Returns dict with
    delta, p, n_sites, test."""
    d = df[df.model != UNTR].dropna(subset=[outcome, *covariates]).copy()
    d['site'] = _site(d)
    d['adv'] = d['robust'].astype(int)
    rhs = ' + '.join(['adv', *covariates, 'C(site)'])
    fit = smf.ols(f'{outcome} ~ {rhs}', data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d['site']})
    delta = float(fit.params['adv'])
    if covariates:
        return dict(delta=delta, p=float(fit.pvalues['adv']),
                    n_sites=int(d['site'].nunique()), test='site-FE ANCOVA, site-clustered')
    per = d.groupby(['site', 'adv'])[outcome].mean().unstack()
    p_w = float(stats.wilcoxon(per[1], per[0])[1])
    return dict(delta=delta, p=p_w, n_sites=len(per), test='within-site paired Wilcoxon')


def partial_residuals(df, outcome, covariates, keep='model', remove_site=True):
    """Display residuals from `outcome ~ C(keep) + covariates + C(site)` fitted on ALL
    rows: covariate effects (and, when remove_site, site effects) are removed, each
    centred, so the grand mean and the `keep` structure are preserved. The covariate
    coefficients come from the JOINT model, so `keep` signal is never absorbed into the
    covariate slope. remove_site=False leaves site variance in the display while the
    covariate slopes are still estimated with site controlled."""
    d = df.dropna(subset=[outcome, *covariates]).copy()
    d['site'] = _site(d)
    rhs = ' + '.join([f'C({keep})', *covariates, 'C(site)'])
    fit = smf.ols(f'{outcome} ~ {rhs}', data=d).fit()
    removed = np.zeros(len(d))
    for c in covariates:
        removed += fit.params[c] * (d[c].values - d[c].mean())
    if remove_site:
        fe = np.array([fit.params.get(f'C(site)[T.{s}]', 0.0) for s in d['site']])
        removed += fe - fe.mean()
    out = pd.Series(d[outcome].values - removed, index=d.index)
    return out


def matched_cells_gap(df, outcome, stratum_col):
    """Exact-matching estimate among the 9 trained models: within each (site, stratum)
    cell where BOTH families are present, delta = mean(adv) - mean(conv); Wilcoxon
    signed-rank across cells. Fully nonparametric: no linearity assumption, and only
    like-for-like cells are ever compared (no extrapolation across covariate ranges)."""
    d = df[df.model != UNTR].dropna(subset=[outcome, stratum_col]).copy()
    d['site'] = _site(d)
    d['adv'] = d['robust'].astype(int)
    cells = (d.groupby(['site', stratum_col, 'adv'], observed=True)[outcome]
             .mean().unstack('adv').dropna())
    deltas = (cells[1] - cells[0]).values
    p = float(stats.wilcoxon(deltas)[1]) if len(deltas) >= 6 else np.nan
    return dict(delta=float(np.mean(deltas)), p=p, n_cells=int(len(deltas)))


def stars(p):
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 0.05 else 'n.s.'))
