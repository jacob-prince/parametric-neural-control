#!/usr/bin/env python3
"""
item 1 - held-out-image validation. Recomputes gradient spectral flatness AND adversarial
sensitivity on 100 NSD images never used to fit the encoding models / as synthesis seeds, then:
  (2) seed-vs-held-out agreement for each measure,
  (3) per-metric split-half reliability across the 100 images (Spearman-Brown),
  (4) whether HELD-OUT flatness / sensitivity predict control (model-level 7-conventional exact
      permutation; monkey-FE + reliability-adjusted partial; leave-one-model-out CV).
Reads the cluster outputs (cluster/outputs_from_cluster/fig5_{ext_heldout, heldout_gradfreq})
+ the frozen preproc_data/fig5_verification_data.csv (join of the cluster attack outputs with
the control outcomes/predictors; its own regenerators live in take8's SX_advrobustness).
Writes preproc_data/fig5_heldout_table.csv (the merged seed+held-out table figure5 reads).
"""
import glob, os, re, pickle
from pathlib import Path
from itertools import permutations
import numpy as np, pandas as pd
from scipy import stats

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
from pnc import paths

PREPROC_DATA = paths.preprocessed_data()
CLUSTER_OUT = paths.cluster_outputs()
FMIN, FMAX = 1, 112
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
MK = ["red", "paul", "venus", "leap", "three0"]
RNG = np.random.default_rng(0)


def spectral_flatness(p):                    # take8 figure7 definition, verbatim
    P = np.clip(np.asarray(p, float)[FMIN:FMAX], 1e-30, None)
    return float(np.exp(np.mean(np.log(P))) / P.mean())


def sb(r):                                    # Spearman-Brown split-half -> full reliability
    return 2 * r / (1 + r) if r < 1 else 1.0


# ---------- held-out sensitivity (swing @4/255 L-inf = canonical; + AUC) ----------
def load_heldout_sensitivity():
    raw = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(CLUSTER_OUT / "fig5_ext_heldout" / "adv_robustness_*_*.csv")))
                     if "smoke" not in p], ignore_index=True)
    raw = raw[raw.norm == "linf"].copy()
    raw["swing"] = (raw["adv_up"] - raw["adv_dn"]) / raw["range_q99q01"]
    key = ["model", "monkey", "channel"]
    at4 = raw[raw.eps_255 == 4.0]
    canon = at4.groupby(key)["swing"].mean().rename("ho_sens")
    # AUC over eps
    per_eps = raw.groupby(key + ["eps_255"])["swing"].mean().reset_index()
    def _auc(s):
        s = s.sort_values("eps_255"); return trapz(s.swing.values, s.eps_255.values) / (s.eps_255.max() - s.eps_255.min())
    auc = per_eps.groupby(key).apply(_auc, include_groups=False).rename("ho_sens_auc")
    # split-half over images (even/odd img_idx), swing@4
    h = at4.assign(half=at4.img_idx % 2).groupby(key + ["half"])["swing"].mean().unstack("half")
    h.columns = ["ho_sens_a", "ho_sens_b"]
    return pd.concat([canon, auc, h], axis=1).reset_index()


# ---------- held-out flatness (spectral flatness of mean gradient profile; + split-half) ----------
def load_heldout_flatness():
    rows = []
    for p in sorted(glob.glob(str(CLUSTER_OUT / "fig5_heldout_gradfreq" / "*_grad_maps_freq_profiles.pkl"))):
        name = os.path.basename(p)
        m = re.match(r"(?P<subj>.+?)_unit_(?P<unit>\d+)_model_(?P<model>.+)_grad_maps_freq_profiles\.pkl", name)
        d = pickle.load(open(p, "rb")); prof = np.asarray(d["profiles"])            # (n_img, n_freq)
        rows.append(dict(monkey=m["subj"].split("_")[0], channel=int(m["unit"]), model=m["model"],
                         ho_flat=spectral_flatness(prof.mean(0)),
                         ho_flat_a=spectral_flatness(prof[0::2].mean(0)),
                         ho_flat_b=spectral_flatness(prof[1::2].mean(0)),
                         n_img=prof.shape[0]))
    return pd.DataFrame(rows)


# ---------- inference helpers (from matrix_aware_inference) ----------
def resid_by_site(df, c):
    return df[c] - df.groupby(["monkey", "channel"])[c].transform("mean")


def model_level(df, pred, outc, models):
    sub = df[df.model.isin(models)]; pm = sub.groupby("model")[[pred, outc]].mean()
    x, y = pm[pred].values, pm[outc].values
    r = np.corrcoef(x, y)[0, 1]; rho = stats.spearmanr(x, y)[0]
    rs = np.array([np.corrcoef(x, y[list(q)])[0, 1] for q in permutations(range(len(y)))])
    pex = float(np.mean(np.abs(rs) >= abs(r) - 1e-12))
    jk = [np.corrcoef(np.delete(x, i), np.delete(y, i))[0, 1] for i in range(len(x))]
    return r, rho, pex, (min(jk), max(jk))


def partial_fl(df, pred, outc, models, nperm=5000):
    sub = df[df.model.isin(models)]
    Z = np.column_stack([pd.get_dummies(sub["monkey"]).astype(float).values,
                         sub["reliability"].values[:, None], np.ones(len(sub))])
    resid = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rc, rf = resid(sub[outc].values.astype(float)), resid(sub[pred].values.astype(float))
    r = np.corrcoef(rc, rf)[0, 1]
    cnt = sum(abs(np.corrcoef(RNG.permutation(rc), rf)[0, 1]) >= abs(r) - 1e-12 for _ in range(nperm))
    return r, (cnt + 1) / (nperm + 1)


def lomo_cv(df, preds, outc):
    mk = pd.get_dummies(df["monkey"]).astype(float).values
    X = np.column_stack([df[p].values for p in preds] + [mk]); y = df[outc].values
    yhat = np.full(len(df), np.nan)
    for g in df.model.unique():
        te = np.where((df.model == g).values)[0]; tr = np.setdiff1d(np.arange(len(df)), te)
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); yhat[te] = X[te] @ b
    return 1 - np.nansum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    seed = pd.read_csv(PREPROC_DATA / "fig5_verification_data.csv").rename(
        columns={"flatness": "seed_flat", "swing.linf.q99q01.mean.v4": "seed_sens"})
    df = (seed.merge(load_heldout_sensitivity(), on=["model", "monkey", "channel"], how="inner")
              .merge(load_heldout_flatness(), on=["model", "monkey", "channel"], how="inner"))
    print(f"merged {len(df)} readouts; images/readout = {int(df.n_img.mode()[0])}")
    df.to_csv(PREPROC_DATA / "fig5_heldout_table.csv", index=False)     # merged seed+held-out table for figure5
    NORMAL7 = [m for m in df.model.unique() if m not in ROBUST + [UNTR]]

    print("\n=== (2) SEED vs HELD-OUT agreement (do fresh-image measures match the seed measures?) ===")
    for lab, s, h in [("flatness", "seed_flat", "ho_flat"), ("sensitivity", "seed_sens", "ho_sens")]:
        r_site = stats.pearsonr(df[s], df[h])[0]
        pm = df.groupby("model")[[s, h]].mean(); r_mod = stats.pearsonr(pm[s], pm[h])[0]
        print(f"  {lab:12s}: site r={r_site:+.3f}  model-mean r={r_mod:+.3f}")

    print("\n=== (3) SPLIT-HALF reliability across the 100 images (Spearman-Brown corrected) ===")
    for lab, a, b in [("flatness", "ho_flat_a", "ho_flat_b"), ("sensitivity", "ho_sens_a", "ho_sens_b")]:
        r = stats.pearsonr(df[a], df[b])[0]
        print(f"  {lab:12s}: split-half r={r:+.3f}  ->  full reliability (SB) = {sb(r):.3f}")

    print("\n=== (4) Does HELD-OUT flatness / sensitivity predict control? ===")
    for pred, lab in [("ho_flat", "held-out flatness"), ("ho_sens", "held-out sensitivity"),
                      ("seed_flat", "[seed flatness ref]")]:
        print(f"  -- {lab} --")
        for outc in ["control_slope", "control_r"]:
            r, rho, pex, jk = model_level(df, pred, outc, NORMAL7)
            pr, pp = partial_fl(df, pred, outc, NORMAL7)
            print(f"     {outc:13s}: 7-model r={r:+.2f} (rho={rho:+.2f}, exact p={pex:.4f}, jk[{jk[0]:+.2f},{jk[1]:+.2f}]) | "
                  f"monkey+reliab partial r={pr:+.2f} (p={pp:.3f})")
    print("  -- leave-one-model-out CV (unseen backbone), pooled R2 --")
    for outc in ["control_slope", "control_r"]:
        r_f = lomo_cv(df, ["ho_flat"], outc); r_s = lomo_cv(df, ["ho_sens"], outc)
        r_fs = lomo_cv(df, ["ho_flat", "ho_sens"], outc); r_seed = lomo_cv(df, ["seed_flat"], outc)
        print(f"     {outc:13s}: ho_flat={r_f:+.3f}  ho_sens={r_s:+.3f}  ho_flat+sens={r_fs:+.3f}  (seed_flat={r_seed:+.3f})")


if __name__ == "__main__":
    main()
