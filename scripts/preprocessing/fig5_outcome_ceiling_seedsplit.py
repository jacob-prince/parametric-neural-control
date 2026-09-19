#!/usr/bin/env python3
"""
outcome_ceiling_seedsplit.py - noise ceiling on the CORRELATION between any (near-noise-free) predictor
and the control outcome, from split-half reliability of the control-slope estimate over the 10 seeds.

Method (standard split-half noise ceiling, done per model-set and per aggregation level):
  * Split the 10 synthesis seeds into two halves of 5. For each (monkey, unit, model), fit the control
    slope (predicted -> measured, floor-clamped) on each half. Average over ALL 126 unique 5/5 splits
    (not a single even/odd partition, which would confound the split with seed image content).
  * Site-residualize each half (subtract per-(monkey,unit) mean across the 10 models), matching the
    plotted outcome. Then per model-set (10/8/7):
      - SITE level : split-half r across the site-models, Spearman-Brown to full length -> reliability
      - MODEL level: average each half to per-model means, correlate the two model-mean vectors, SB
  * The CEILING ON A CORRELATION is sqrt(reliability):  a perfect predictor X ∝ signal gives
    corr(X, O) = sd(signal)/sd(O) = sqrt(rho).  (Predictors here are ~perfectly reliable over the 100
    held-out images - flatness 0.997, sensitivity 1.000 - so sqrt(rho_pred·rho_out) ≈ sqrt(rho_out).)
  * The model-level ceiling is estimated from only 7-10 models -> report a bootstrap CI over models.

Writes preproc_data/fig5_outcome_ceiling_seedsplit*.csv:
  group, level, reliability_sb, ceiling (=sqrt reliability), ceiling_lo, ceiling_hi
Needs the pnc.preproc loader.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats

from pnc import paths
from pnc.preproc import loader as L

INT = paths.preprocessed_data()

MONKEYS = ["red", "paul", "venus", "leap", "three0"]
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
NBOOT = 1000


def measure(x, y, kind="slope"):
    """Per (monkey, unit, model) control outcome on one seed-half: control slope, or control r (Pearson)."""
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.pearsonr(x, y)[0] if kind == "r" else stats.linregress(x, y).slope


def sb(r):
    return 2 * r / (1 + r) if (np.isfinite(r) and r < 1) else (1.0 if np.isfinite(r) else np.nan)


def gather():
    """Per (monkey, unit, model): arrays (seed, x=clamped pred, y=measured resp) over accentuations."""
    cfg = L.config(); data = {}
    for mk in MONKEYS:
        P = L.load_predictions(mk)
        for u in L.load_brain(mk)["units"]:
            u = int(u); resp = L._acc_response(mk, u); fl = L.firing_floor(mk, u)
            ui = list(P["target_units"]).index(u)
            for m in cfg["models"]:
                mi = list(P["pred_models"]).index(m)
                sel = (P["gen_model"] == m) & (P["gen_unit"] == u)
                s, x, y = [], [], []
                for n, p, sd in zip(P["stim"][sel], P["pred"][sel, mi, ui], P["gen_seed"][sel]):
                    if n in resp:
                        s.append(int(sd)); x.append(max(float(p), fl)); y.append(resp[n])
                if s:
                    data[(mk, u, m)] = (np.array(s), np.array(x, float), np.array(y, float))
    return data


def main(outcome="slope", mode="site", resid="10"):
    data = gather()
    seeds = sorted(set(int(sd) for v in data.values() for sd in v[0]))
    splits = [(set(a), set(seeds) - set(a)) for a in combinations(seeds, len(seeds) // 2)]
    splits = splits[: len(splits) // 2]                       # unique complementary halves (126 for 10)
    print(f"{len(data)} site-models; {len(seeds)} seeds; {len(splits)} unique 5/5 splits; outcome={outcome}, mode={mode}")

    keys = list(data.keys())
    sid = np.array([f"{mk}_{u}" for (mk, u, m) in keys])
    model = np.array([m for (mk, u, m) in keys])
    # control-phase calibration r (model generalization) per site-model, for the 'site_calib' mode
    calib = None
    if mode == "site_calib":
        ct = L.control_table().copy(); ct["calib_r"] = np.sqrt(np.clip(ct["enc_gen_R2"].astype(float).values, 0, 1))
        cm = {(r.model, r.monkey, int(r.unit)): float(r.calib_r) for r in ct.itertuples()}
        calib = np.array([cm.get((m, mk, u), np.nan) for (mk, u, m) in keys])
        calib = np.where(np.isfinite(calib), calib, np.nanmean(calib))

    def _resid(v):
        d = pd.DataFrame({"sid": sid, "v": v, "model": model})
        if resid == "none":                                        # absolute outcome, no residualization
            mean_v = np.zeros_like(v)
        elif resid == "9trained":                                  # per-site mean over the 9 trained models
            mu = d[d.model != UNTR].groupby("sid")["v"].mean()
            mean_v = d["sid"].map(mu).values
        else:
            mean_v = d.groupby("sid")["v"].transform("mean").values
        r = v - mean_v
        if calib is not None:                                  # additionally regress out calibration r (global)
            ok = np.isfinite(r); X = np.column_stack([calib[ok], np.ones(ok.sum())])
            b, *_ = np.linalg.lstsq(X, r[ok], rcond=None); r = r.copy(); r[ok] = r[ok] - X @ b
        return r

    A = np.full((len(splits), len(keys)), np.nan); B = np.full_like(A, np.nan)
    for si, (ha, hb) in enumerate(splits):
        la, lb = list(ha), list(hb)
        va = np.array([measure(x[np.isin(s, la)], y[np.isin(s, la)], outcome) for (s, x, y) in (data[k] for k in keys)])
        vb = np.array([measure(x[np.isin(s, lb)], y[np.isin(s, lb)], outcome) for (s, x, y) in (data[k] for k in keys)])
        A[si] = _resid(va); B[si] = _resid(vb)

    groups = {10: set(model), 9: set(model) - {UNTR}, 7: set(model) - set(ROBUST) - {UNTR}}
    rng = np.random.default_rng(0)
    out = []
    for k, ms in groups.items():
        col = np.isin(model, list(ms))
        # SITE level: split-half r across site-models -> SB -> sqrt, per split; average over splits
        site_ceil = []
        for si in range(len(splits)):
            a, b = A[si, col], B[si, col]; ok = np.isfinite(a) & np.isfinite(b)
            site_ceil.append(np.sqrt(max(sb(stats.pearsonr(a[ok], b[ok])[0]), 0.0)))
        site_ceil = np.array(site_ceil)
        # MODEL level: per split -> per-model means -> corr of the two model-mean vectors -> SB -> sqrt
        mods = sorted(ms)
        mm_a = np.full((len(splits), len(mods)), np.nan); mm_b = np.full_like(mm_a, np.nan)
        for si in range(len(splits)):
            for j, mm in enumerate(mods):
                mc = model == mm
                mm_a[si, j] = np.nanmean(A[si, mc]); mm_b[si, j] = np.nanmean(B[si, mc])
        model_ceil = np.array([np.sqrt(max(sb(stats.pearsonr(mm_a[si], mm_b[si])[0]), 0.0))
                               for si in range(len(splits))])
        # bootstrap CI over models, using the SAME estimator (mean over splits of per-split SB corr)
        boot = []
        for _ in range(NBOOT):
            idx = rng.integers(0, len(mods), len(mods))
            vals = []
            for si in range(len(splits)):
                a, b = mm_a[si, idx], mm_b[si, idx]
                if np.std(a) > 0 and np.std(b) > 0:
                    vals.append(np.sqrt(max(sb(stats.pearsonr(a, b)[0]), 0.0)))
            if vals:
                boot.append(np.mean(vals))
        boot = np.array(boot)
        out.append(dict(group=k, level="site", reliability_sb=float(np.mean(site_ceil ** 2)),
                        ceiling=float(np.mean(site_ceil)),
                        ceiling_lo=float(np.percentile(site_ceil, 2.5)),
                        ceiling_hi=float(np.percentile(site_ceil, 97.5))))
        out.append(dict(group=k, level="model", reliability_sb=float(np.mean(model_ceil ** 2)),
                        ceiling=float(np.mean(model_ceil)),
                        ceiling_lo=float(np.percentile(boot, 2.5)),
                        ceiling_hi=float(np.percentile(boot, 97.5))))
        print(f"  n={k:2d}: site  ceiling={np.mean(site_ceil):.3f} "
              f"[{np.percentile(site_ceil,2.5):.3f},{np.percentile(site_ceil,97.5):.3f}]   "
              f"model ceiling={np.mean(model_ceil):.3f} "
              f"boot[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")
    fn = (f"fig5_outcome_ceiling_seedsplit{'_r' if outcome == 'r' else ''}"
          f"{ {'9trained': '_resid9', 'none': '_noresid'}.get(resid, '') }{'_calib' if mode == 'site_calib' else ''}.csv")
    pd.DataFrame(out).to_csv(INT / fn, index=False)
    print(f"wrote {INT/fn}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outcome", nargs="?", default="slope", choices=["slope", "r"])
    ap.add_argument("--resid", default="10", choices=["10", "9trained", "none"],
                    help="site residualization family (output suffix: 9trained -> _resid9, none -> _noresid)")
    ap.add_argument("--mode", default="site", choices=["site", "site_calib"])
    a = ap.parse_args()
    main(outcome=a.outcome, mode=a.mode, resid=a.resid)
