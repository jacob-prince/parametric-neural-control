#!/usr/bin/env python3
"""Training-diet ladder for the encoding readouts (slug refit_diet_ladder, figS67) -- STAGE 1.

Per (site, model) pair, the readout is fit on a LADDER of training diets (same machinery as
build_pooled_refit.py: RidgeCV on the raw frozen 750-d source-layer PCA features, NO scaler,
production alpha grid -- the production convention; rho=0.5 loss balance natural-vs-accentuated
wherever accentuations are present; is_test natural images NEVER fit on, mirroring the original
fit's held-out scope):

  d0  calibration only          the original synthesis-time readout (no refit; readout_vec)
  d1  + own accentuations       naturals + THIS model's accentuations of THIS unit (~110)
  d2  + site-targeted           naturals + ALL models' accentuations of this unit (~1,100)
  d3  + everything              naturals + all ~5.5k accentuated pairs (== S66 pooled fit)
  d4  everything minus own      d3 EXCLUDING this pair's own ~110 control-scored images
                                (zero circularity vs this pair's control outcome)

Naturals = calibration-session non-test pairs + control-day re-presentations of non-test images.
Per diet we record: n's, alpha, test r on the original natural is_test split, and r on the pair's
OWN accentuations -- 5-fold OOF when they are in the diet (d1/d2/d3, plain Ridge at the export
fit's alpha per 26c), direct prediction when they are not (d0/d4).

Outputs (supplementary/intermediate/pooled_refit/):
  diet_refit.csv                1250 rows = 250 pairs x 5 diets (long format)
  diet_readouts_d{1,2,4}.npz    raw-PCA-space readouts for the gradient job
                                (d0 gradients exist in heldout_gradfreq/, d3 in gradfreq/)

Usage: python build_diet_refits.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_pooled_refit import (T8, U, L, tobool, rr, balanced_w, ALPHAS, K, OUTDIR)  # noqa: E402

CSV = os.path.join(OUTDIR, 'diet_refit.csv')
MODELS = U.MODEL_ORDER
GRAD_DIETS = ('d1', 'd2', 'd4')


def fit_export(Xn_blk, yn_blk, Xa_blk, ya_blk):
    """One diet fit on naturals + (possibly empty) accentuated block. Returns (predict_fn,
    raw-PCA readout, alpha). Balanced rho=0.5 weights when the accentuated block is non-empty."""
    X = np.vstack([Xn_blk, Xa_blk]); y = np.concatenate([yn_blk, ya_blk])
    is_acc = np.concatenate([np.zeros(len(yn_blk), bool), np.ones(len(ya_blk), bool)])
    w = balanced_w(is_acc) if is_acc.any() else None
    m = RidgeCV(alphas=ALPHAS).fit(X, y, sample_weight=w)
    return (lambda Xq: m.predict(Xq)), m.coef_, float(m.alpha_)


def oof_own(Xn_blk, yn_blk, Xa_blk, ya_blk, own_in_blk, alpha):
    """5-fold OOF predictions for the own-pair rows (boolean mask into the accentuated block),
    refitting plain Ridge at the export alpha with the fold's own rows held out."""
    idx = np.where(own_in_blk)[0]
    rng = np.random.default_rng(0)
    fold = rng.integers(0, K, len(idx))
    pred = np.full(len(idx), np.nan)
    for f in range(K):
        drop = idx[fold == f]
        keep = np.ones(len(ya_blk), bool); keep[drop] = False
        X = np.vstack([Xn_blk, Xa_blk[keep]]); y = np.concatenate([yn_blk, ya_blk[keep]])
        is_acc = np.concatenate([np.zeros(len(yn_blk), bool), np.ones(int(keep.sum()), bool)])
        m = Ridge(alpha=alpha).fit(X, y, sample_weight=balanced_w(is_acc))
        pred[fold == f] = m.predict(Xa_blk[drop])
    return pred


def compute():
    rows = []
    readouts = {d: {} for d in GRAD_DIETS}
    for mk in L.monkeys():
        B = L.load_brain(mk); P = L.load_predictions(mk)
        units = list(B['units'])
        calib, ctrl = B['calibration'], B['control']
        cmap_cal = {s: i for i, s in enumerate(np.asarray(calib['stim']).astype(str))}
        ckind = np.asarray(ctrl['kind']).astype(str)
        cstim = np.asarray(ctrl['stim']).astype(str)
        subj = U.MONKEY_DIRS[mk]
        for uu in units:
            u = int(uu); ui = units.index(uu); region = str(B['region'][ui])
            cmap_ctrl = {s: ctrl['resp_z'][i, ui] for i, s in enumerate(cstim)
                         if ckind[i] == 'accentuated'}
            y_acc_all = np.array([cmap_ctrl.get(s, np.nan) for s in np.asarray(P['stim']).astype(str)], float)
            re_pairs = [(cstim[i], float(ctrl['resp_z'][i, ui]))
                        for i in np.where(ckind == 'calibration')[0]]
            for model in MODELS:
                try:
                    enc = U.load_encoding_pickle(mk, u, model)
                    acc = U.load_accentuated_pickle(mk, u, model)
                except Exception as e:
                    print(f'skip {mk} u{u} {model}: {e}', flush=True); continue
                edf = enc['df']
                sn = edf['stimulus_name'].astype(str).values
                Xn = enc['PCA_resp'].numpy().astype(np.float64)
                yn = np.array([calib['resp_z'][cmap_cal[s], ui] if s in cmap_cal else np.nan
                               for s in sn], float)
                te = tobool(edf['is_test']) & np.isfinite(yn)
                test_imgs = set(sn[te])
                row_of = {s: i for i, s in enumerate(sn)}
                Xa = acc['PCA_resp'].numpy().astype(np.float64)
                adf = acc['df']
                good = np.isfinite(y_acc_all)
                Xa, ya = Xa[good], y_acc_all[good]
                tmod = adf['model_name'].astype(str).values[good]
                tuid = adf['unit_id'].to_numpy()[good].astype(int)
                own = (tmod == model) & (tuid == u)
                site = tuid == u

                rv = np.asarray(enc['readout_vec']).ravel()
                rb = float(np.asarray(enc['readout_bias']).ravel()[0])

                # shared natural training block: non-test calibration pairs + non-test re-presentations
                nat_i = np.where(np.isfinite(yn) & ~te)[0]
                re_ok = [(s, y) for s, y in re_pairs
                         if s in row_of and np.isfinite(y) and s not in test_imgs]
                Xn_blk = np.vstack([Xn[nat_i], Xn[[row_of[s] for s, _ in re_ok]]])
                yn_blk = np.concatenate([yn[nat_i], np.array([y for _, y in re_ok], float)])

                acc_masks = dict(d0=np.zeros(len(ya), bool), d1=own, d2=site,
                                 d3=np.ones(len(ya), bool), d4=~own)
                for diet, am in acc_masks.items():
                    if diet == 'd0':
                        pred_fn = lambda Xq: Xq @ rv + rb
                        alpha = np.nan
                    else:
                        pred_fn, w_out, alpha = fit_export(Xn_blk, yn_blk, Xa[am], ya[am])
                        if diet in GRAD_DIETS:
                            readouts[diet][f'{subj}_unit_{u}_model_{model}'] = w_out
                    r_nat = rr(yn[te], pred_fn(Xn[te]))
                    if am[own].any():                       # own rows in the diet -> OOF
                        own_in_blk = own[am]
                        p_own = oof_own(Xn_blk, yn_blk, Xa[am], ya[am], own_in_blk, alpha)
                        ownmode = 'oof'
                    else:                                    # own rows never fit on -> direct
                        p_own = pred_fn(Xa[own])
                        ownmode = 'direct'
                    rows.append(dict(monkey=mk, unit=u, region=region, model=model, diet=diet,
                                     n_fit_nat=len(yn_blk), n_fit_acc=int(am.sum()),
                                     alpha=alpha, r_nat_test=r_nat,
                                     r_own_acc=rr(ya[own], p_own), own_mode=ownmode))
                r_ = {d: [q for q in rows[-5:] if q['diet'] == d][0] for d in acc_masks}
                print(f"{mk} u{u:<3d} {model:24s} ownacc "
                      + ' '.join(f"{d}:{r_[d]['r_own_acc']:+.2f}" for d in ('d0', 'd1', 'd2', 'd3', 'd4'))
                      + f"  nat d3:{r_['d3']['r_nat_test']:+.2f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(CSV, index=False)
    for d in GRAD_DIETS:
        np.savez(os.path.join(OUTDIR, f'diet_readouts_{d}.npz'), **readouts[d])
    print(f'\nwrote {CSV} ({len(df)} rows) + diet_readouts_{{d1,d2,d4}}.npz '
          f'({[len(readouts[d]) for d in GRAD_DIETS]} readouts)')
    g = df.groupby('diet')[['n_fit_nat', 'n_fit_acc', 'r_nat_test', 'r_own_acc']].mean()
    print(g.round(3).to_string())


if __name__ == '__main__':
    compute()
