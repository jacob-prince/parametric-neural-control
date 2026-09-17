#!/usr/bin/env python3
"""Adversarial-sensitivity data helpers for figure 5.

Reads the cluster PGD outputs (source_data/cluster_outputs/fig5_*):
  fig5_advvis_exact/          minimal-eps adversarial-attack records per site+seed (npz)
  fig5_ext_heldout{,_lowextra}/  per-eps normalized-swing sweeps on the 100 held-out
                              NSD images (never used to fit the readouts)
and the aggregated per-readout table preprocessed_data/fig5_heldout_table.csv.
"""
import glob
import numpy as np, pandas as pd

from pnc import paths

CLUSTER_OUT = paths.cluster_outputs()
ADVVIS_DIR = CLUSTER_OUT / "fig5_advvis_exact"
HELDOUT_TABLE = paths.preprocessed_data() / "fig5_heldout_table.csv"
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

KEY = ["model", "monkey", "channel"]

# eps range the sensitivity metric (load_heldout_logauc) integrates over.
AUC_EPS_LO, AUC_EPS_HI = 0.125, 16.0


def load_heldout_table():
    """Per-readout control outcomes + reliability on the held-out probe set."""
    return pd.read_csv(paths.require(HELDOUT_TABLE, hint="python scripts/preprocessing/fig5_heldout_analysis.py"))


def _heldout_raw():
    ho = pd.concat([pd.read_csv(p) for d in ("fig5_ext_heldout", "fig5_ext_heldout_lowextra")
                    for p in sorted(glob.glob(str(paths.require(CLUSTER_OUT / d) / "adv_robustness_*.csv"))) if "smoke" not in p],
                   ignore_index=True)
    return ho[ho.norm == "linf"].copy()


def load_heldout_curves():
    """Per-(model,channel) mean normalized-swing sweep on the 100 held-out images -- the SAME data the
    quantitative sensitivity metric is read from, extended below the integration window to
    eps=2^-3/255 (ext_heldout + ext_heldout_lowextra)."""
    ho = _heldout_raw()
    ho["nswing"] = (ho.adv_up - ho.adv_dn) / ho.range_q99q01
    return ho.groupby(["model", "monkey", "channel", "eps_255"])["nswing"].mean().reset_index()


def load_heldout_logauc():
    """Per-readout AUC of normalized swing integrated over LOG2(eps) on the 100 held-out images --
    the adversarial-sensitivity measure. Weighting each eps-octave equally (log2 eps) matches the
    sweep panel's log axis. Integration window = [AUC_EPS_LO, AUC_EPS_HI] = [2^-3, 2^4]/255
    (extends into the first-order regime; excludes the super-perceptual tail 32/64)."""
    ho = _heldout_raw()
    ho = ho[(ho.eps_255 >= AUC_EPS_LO - 1e-9) & (ho.eps_255 <= AUC_EPS_HI + 1e-9)]
    ho["nswing"] = (ho.adv_up - ho.adv_dn) / ho.range_q99q01
    g = ho.groupby(KEY + ["eps_255"])["nswing"].mean().reset_index()

    def _auc(d):
        d = d.sort_values("eps_255"); x = np.log2(d.eps_255.values)
        return trapz(d.nswing.values, x) / (x[-1] - x[0])
    return g.groupby(KEY).apply(_auc, include_groups=False).reset_index(name="auc")
