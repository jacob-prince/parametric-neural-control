#!/usr/bin/env python3
"""Spearman-Brown split-half reliability of the site-residualized control slope, per model
group — the reliability ceiling for the `site r` (dashed fit) in the adversarial-sensitivity
supplement's panels c/d/e.

For each of the 250 site-model experiments, split its accentuated stimuli in half and
recompute the control slope on each half (two outcome vectors). Residualize EACH half by its
per-(monkey,unit) mean over the 9 TRAINED models (the figure-5 convention, applied to both
the supplement's scatters and this ceiling), then, for each model group (all 10 / without
adv-trained / without untrained), correlate the two halves over the site-models in that
group and Spearman-Brown correct. Averaged over REPS random splits (seeded -> deterministic).

Writes preproc_data/sup_advsens_outcome_reliability.csv.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
from scipy import stats

from pnc import paths
from pnc.preproc import loader as L

OUT = os.path.join(str(paths.preprocessed_data()), 'sup_advsens_outcome_reliability.csv')
MONKEYS = ["red", "paul", "venus", "leap", "three0"]
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
REPS = 200


def main():
    cfg = L.config()
    clouds = []
    for mk in MONKEYS:
        for u in L.load_brain(mk)["units"]:
            for m in cfg["models"]:
                x, y = L.control_cloud(mk, int(u), m)
                x, y = np.asarray(x, float), np.asarray(y, float)
                if len(x) >= 8:
                    clouds.append((mk, int(u), m, x, y))
    models_all = sorted(set(c[2] for c in clouds))
    groups = {"all": models_all,
              "minus2adv": [m for m in models_all if m not in ROBUST],
              "conv7": [m for m in models_all if m not in ROBUST + [UNTR]]}
    print(f"{len(clouds)} experiments; {len(models_all)} models")

    def slope(x, y):
        if len(x) < 4 or x.std() == 0 or y.std() == 0:
            return np.nan
        return stats.linregress(x, y).slope

    def sb(r):
        return 2 * r / (1 + r) if r < 1 else 1.0

    per_group = {g: [] for g in groups}
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        rows = []
        for mk, u, m, x, y in clouds:
            idx = rng.permutation(len(x)); h = len(x) // 2
            rows.append((f"{mk}_{u}", m, slope(x[idx[:h]], y[idx[:h]]), slope(x[idx[h:]], y[idx[h:]])))
        df = pd.DataFrame(rows, columns=["sid", "model", "o1", "o2"]).dropna()
        # residualize each half by per-site mean over the 9 TRAINED models (fig-5 convention)
        tr = df[df.model != UNTR]
        for c in ("o1", "o2"):
            mean = tr.groupby("sid")[c].mean()
            df["r" + c[1]] = df[c].values - mean.reindex(df["sid"]).values
        for g, ms in groups.items():
            sub = df[df.model.isin(ms)]
            if len(sub) > 4 and sub.r1.std() > 0 and sub.r2.std() > 0:
                per_group[g].append(stats.pearsonr(sub.r1, sub.r2)[0])

    out = []
    for g in groups:
        r = float(np.mean(per_group[g]))
        out.append(dict(group=g, n_models=len(groups[g]), split_half_r=r, reliability_sb=sb(r)))
        print(f"  {g:10s} ({len(groups[g])} models): split-half r={r:+.3f} -> SB reliability={sb(r):.3f}")
    pd.DataFrame(out).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
