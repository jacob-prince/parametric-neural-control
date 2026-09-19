#!/usr/bin/env python3
"""Supp fig (adversarial_sensitivity) - encoding-axis adversarial sensitivity and neural control.

Structural companion of the gradient-geometry supplement, with the gradient data swapped
for adversarial sensitivity. Although only two backbones were explicitly adversarially
trained, adversarial sensitivity is measured continuously for every one of the 250 fitted
encoding axes (each backbone + its scalar neural readout is a differentiable input-output
function attacked with standardized pixel PGD).

  a      mosaic: top row = per-model minimal-eps adversarial perturbation of the cat seed,
         raising the model's prediction by +2 z toward the example site (paul ch8 - the
         same records the main figure's mosaic draws). eps found by geometric bisection
         (adv_visual_attack.py --bisect) so the achieved change honestly lands at +2
         (within 0.05); minimal eps annotated under each perturbation. Bottom row = the
         perturbation itself. Columns ordered by per-model mean sensitivity.
  b      encoding-axis adversarial sensitivity vs perturbation strength (25 axes/model +
         model mean; shaded band = the eps window the AUC metric integrates over).
  c/d/e  adversarial sensitivity (log-eps AUC on the 100 held-out NSD images) vs
         site-residualized control slope - residualized against the per-site mean over the
         9 TRAINED models, the main-figure convention - for the three model subsets (all
         10 / without adv.-trained 8 / without untrained 7); dashed = site-level fit,
         bold = model-mean fit; panel c carries the per-model sensitivity inset.

Reads cluster PGD outputs (cluster/outputs_from_cluster/fig5_advvis_exact +
fig5_ext_heldout{,_lowextra}) + preproc_data/fig5_heldout_table.csv +
preproc_data/sup_advsens_outcome_reliability.csv (split-half reliability of the
site-residualized slope per model group; regenerator =
scripts/preprocessing/build_advsens_outcome_reliability.py).
"""
import os
import glob
import argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from pnc import paths
from pnc import utils as U
from pnc.utils import save_fig, get_model_color
from pnc.manifest import output_name
U.apply_figure_style()

STEM = output_name("adversarial_sensitivity")
PREPROC_DATA = str(paths.preprocessed_data())
CLUSTER_OUT = str(paths.cluster_outputs())
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

SHORT = dict(U.MODEL_SHORT_NAMES); SHORT["AlexNet_training_seed_01"] = "Untrained"
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
EXTREMES = ROBUST + [UNTR]
KEY = ["model", "monkey", "channel"]

# example site + seed for the panel-a mosaic (matches the main-figure mosaic: paul ch8, cat seed)
ACC_MONKEY, ACC_UNIT, SEED_TAG = "paul", 8, "seed3"

# eps range the sensitivity metric (load_heldout_logauc) integrates over -> shaded in panel b.
AUC_EPS_LO, AUC_EPS_HI = 0.125, 16.0


# ---- data -------------------------------------------------------------------
def load_gallery():
    """Per-model minimal-eps adversarial-attack records for the example site+seed, sorted by eps."""
    recs = []
    for p in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, "fig5_advvis_exact")),
                                           f"advvis_{ACC_MONKEY}_Ch{ACC_UNIT}_{SEED_TAG}_*.npz"))):
        d = np.load(p, allow_pickle=True)
        recs.append(dict(model=str(d["model"]), eps=float(d["eps_used"]),
                         clean=d["clean"], adv=d["adv"], delta=float(d["target_delta"]),
                         cp=float(d["clean_pred"]), ap=float(d["adv_pred"])))
    recs.sort(key=lambda r: r["eps"]); return recs


def _heldout_raw():
    ho = pd.concat([pd.read_csv(p) for d in ("fig5_ext_heldout", "fig5_ext_heldout_lowextra")
                    for p in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, d)),
                                                           "adv_robustness_*.csv")))
                    if "smoke" not in p], ignore_index=True)
    return ho[ho.norm == "linf"].copy()


def load_heldout_curves():
    """Per-(model,channel) mean normalized-swing sweep on the 100 held-out images -- the SAME data the
    quantitative sensitivity metric is read from (panels c-e), extended below the integration window
    to eps=2^-3/255."""
    ho = _heldout_raw()
    ho["nswing"] = (ho.adv_up - ho.adv_dn) / ho.range_q99q01
    return ho.groupby(["model", "monkey", "channel", "eps_255"])["nswing"].mean().reset_index()


def load_heldout_logauc():
    """Per-readout AUC of normalized swing integrated over LOG2(eps) on the 100 held-out images
    (never used to fit the readouts) -- the adversarial-sensitivity measure. Weighting each
    eps-octave equally (log2 eps) matches panel b's log axis. Integration window = [AUC_EPS_LO,
    AUC_EPS_HI] = [2^-3, 2^4]/255 (extends into the first-order regime; excludes the
    super-perceptual tail 32/64)."""
    ho = _heldout_raw()
    ho = ho[(ho.eps_255 >= AUC_EPS_LO - 1e-9) & (ho.eps_255 <= AUC_EPS_HI + 1e-9)]
    ho["nswing"] = (ho.adv_up - ho.adv_dn) / ho.range_q99q01
    g = ho.groupby(KEY + ["eps_255"])["nswing"].mean().reset_index()

    def _auc(d):
        d = d.sort_values("eps_255"); x = np.log2(d.eps_255.values)
        return trapz(d.nswing.values, x) / (x[-1] - x[0])
    return g.groupby(KEY).apply(_auc, include_groups=False).reset_index(name="auc")


def _p(x):
    s = f"{x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def _sensitivity_inset(ax, auc):
    """Per-model adversarial sensitivity (mean AUC +/- SEM over sites): square box in the top-right
    corner; colour-coded bold model labels on the left; value at each bar end; SEM whiskers."""
    per = auc.groupby("model")["auc"].agg(["mean", "sem"]).sort_values("mean")
    ax.figure.canvas.draw()
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ih = 0.235; iw = ih * (bb.height * fh) / (bb.width * fw)
    ins = ax.inset_axes([1.0 - iw + 0.035, 1.0 - ih - 0.008, iw, ih]); ins.set_box_aspect(1)
    ys = np.arange(len(per)); means = per["mean"].values; sems = per["sem"].values
    colors = [get_model_color(m) for m in per.index]
    ins.barh(ys, means, height=0.72, color=colors, alpha=0.95, edgecolor="white", linewidth=0.6, zorder=2)
    ins.errorbar(means, ys, xerr=sems, fmt="none", ecolor="0.3", elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=3)
    ins.set_yticks(ys); ins.set_yticklabels([SHORT.get(m, m) for m in per.index], fontsize=6.0)
    for tick, c in zip(ins.get_yticklabels(), colors):
        tick.set_color(c); tick.set_fontweight("bold")
    pad = 0.03 * float(means.max())
    for y, a, s in zip(ys, means, sems):
        ins.text(a + s + pad, y, f"{a:.1f}", fontsize=5.6, va="center", ha="left", color="#444", fontweight="bold")
    ins.set_xlabel("adv. sensitivity (log-ε AUC)", fontsize=7.2, labelpad=2); ins.set_xticks([])
    ins.tick_params(axis="y", length=0, pad=3)
    ins.set_ylim(len(per) - 0.4, -0.6); ins.set_xlim(0, float(means.max()) + 13 * pad)
    for s_ in ("top", "right", "bottom", "left"):
        ins.spines[s_].set_visible(True); ins.spines[s_].set_edgecolor("#aaa"); ins.spines[s_].set_linewidth(0.8)
    ins.set_facecolor((1, 1, 1, 0.95))


def build():
    gal = load_gallery()
    # log2-eps AUC on the 100 held-out images = the adversarial-sensitivity measure
    ht = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, "fig5_heldout_table.csv"),
                                   hint="python scripts/preprocessing/fig5_heldout_analysis.py"))
    auc = ht[KEY + ["control_slope", "control_r", "reliability"]].merge(load_heldout_logauc(), on=KEY)
    # panel-b sweep uses the SAME held-out images as the metric, extended below the window to eps=2^-3
    curves = load_heldout_curves()
    # SB split-half reliability of the site-residualized control slope, per model group (frozen input)
    rel_sb = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, "sup_advsens_outcome_reliability.csv"),
                                       hint="python scripts/preprocessing/build_advsens_outcome_reliability.py")
                         ).set_index("group")["reliability_sb"].to_dict()
    GROUP_KEYS = ["all", "minus2adv", "conv7"]        # panels c, d, e

    # site-residualized control slope: per-site mean over the 9 TRAINED models (the main-figure
    # convention, so every shared statistic matches the main figure exactly)
    tr_mean = auc[auc.model != UNTR].groupby(["monkey", "channel"])["control_slope"].mean()
    auc["control_slope"] = (auc["control_slope"].values
                            - tr_mean.reindex(pd.MultiIndex.from_frame(auc[["monkey", "channel"]])).values)
    OUTC, YLABEL = "control_slope", "Control slope (site residual)"

    plt.rcParams.update({"xtick.labelsize": 9.5, "ytick.labelsize": 9.5})

    fig = plt.figure(figsize=(13.7, 8.3))
    outer = GridSpec(2, 1, height_ratios=[0.92, 1.18], hspace=0.24,
                     left=0.045, right=0.99, top=0.909, bottom=0.114)
    # bottom row: b/c/d/e get equal-width columns
    gB = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], width_ratios=[1, 1, 1, 1], wspace=0.20)

    # ---- (a) gallery ----
    # columns ordered by per-model mean sensitivity (most sensitive left, adv-trained right),
    # annotated with the same metric panels c-e use on their x-axes
    per_sens = auc.groupby("model")["auc"].mean()
    gal.sort(key=lambda r: -per_sens.get(r["model"], -np.inf))
    n = len(gal)
    gA = GridSpecFromSubplotSpec(2, n + 1, subplot_spec=outer[0], hspace=0.06, wspace=0.045)
    s0 = fig.add_subplot(gA[0, 0]); s0.imshow(np.clip(np.transpose(gal[0]["clean"], (1, 2, 0)), 0, 1))
    s0.set_xticks([]); s0.set_yticks([]); s0.set_title("seed", fontsize=10, pad=14); s0.set_ylabel("adversarial", fontsize=9.5)
    s0.annotate(r"init $\rightarrow$ adv (z)", (0.5, 1.03), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=6.2, color="0.4")
    sL = fig.add_subplot(gA[1, 0]); sL.set_xticks([]); sL.set_yticks([])
    for sp in sL.spines.values(): sp.set_visible(False)
    sL.set_ylabel("perturbation", fontsize=9.5)
    sL.text(0.5, 0.5, f"target:\nchange model pred\nby +{gal[0]['delta']:.0f} z units",
            ha="center", va="center", fontsize=7, color="0.4", transform=sL.transAxes)
    for j, r in enumerate(gal):
        a0 = fig.add_subplot(gA[0, j + 1]); a0.imshow(np.clip(np.transpose(r["adv"], (1, 2, 0)), 0, 1)); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(SHORT.get(r["model"], r["model"]), fontsize=9.5, fontweight="bold", color=get_model_color(r["model"]), pad=14)
        a0.annotate(rf"${_p(r['cp'])} \rightarrow {_p(r['ap'])}$", (0.5, 1.03), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=7.2, color="0.3")
        a1 = fig.add_subplot(gA[1, j + 1]); p = r["adv"] - r["clean"]
        a1.imshow(np.transpose(np.clip(0.5 + 0.5 * p / (np.abs(p).max() + 1e-8), 0, 1), (1, 2, 0))); a1.set_xticks([]); a1.set_yticks([])
        a1.set_xlabel(f"ε={r['eps']:.3g}", fontsize=8.5, labelpad=2)

    # ---- (b) sensitivity vs perturbation strength ----
    axb = fig.add_subplot(gB[0])
    # very light shaded band = the eps window the AUC metric integrates over (agrees with panels c-e);
    # the sweep is drawn below it (eps=2^-3,2^-2) to show what is measured but not integrated.
    axb.axvspan(AUC_EPS_LO, AUC_EPS_HI, facecolor="#8a8f98", alpha=0.10, lw=0, zorder=0)
    for m in U.MODEL_ORDER:
        pmc = curves[curves.model == m]
        if pmc.empty: continue
        c = get_model_color(m)
        for _, s in pmc.groupby(["monkey", "channel"]):        # one faint line per site (25/model)
            s = s.sort_values("eps_255"); axb.plot(s.eps_255, s.nswing, "-", lw=0.3, color=c, alpha=0.22, zorder=1)
        mn = pmc.groupby("eps_255")["nswing"].mean().sort_index()
        axb.plot(mn.index, mn.values, "-", lw=1.9, color=c, label=SHORT.get(m, m), zorder=3)
    axb.set_xscale("log", base=2); axb.set_ylim(0, 6.2)
    axb.set_xlim(2 ** -3.6, 2 ** 6.6)                      # show eps=2^-3 .. 2^6 with margin
    axb.text(AUC_EPS_LO, 0.60, "AUC integration\nwindow", transform=axb.get_xaxis_transform(),
             ha="left", va="top", fontsize=7.2, color="0.55", style="italic", zorder=4)
    axb.set_xlabel("Perturbation strength  ε  (/255)", fontsize=10)
    axb.set_ylabel("Adversarial sensitivity\n" r"($\Delta$ resp., mult. of range)", fontsize=9.5)
    axb.set_title("Adversarial sensitivity\nvs perturbation strength", fontsize=12, fontweight="bold")
    lg = axb.legend(fontsize=6.4, ncol=2, frameon=True, loc="upper left", handlelength=1.0, columnspacing=0.7, labelspacing=0.25)
    lg.get_frame().set(facecolor="white", edgecolor="#cccccc", lw=0.6, alpha=0.9)

    # ---- (c–e) sensitivity ↔ control by model subset (site- and model-level correlations) ----
    subsets = [("All models", list(auc.model.unique())),
               ("Without adv.-trained", [m for m in auc.model.unique() if m not in ROBUST]),
               ("7 conventionally trained", [m for m in auc.model.unique() if m not in EXTREMES])]
    yr = (auc[OUTC].min() - 0.06, auc[OUTC].max() + 0.06)
    scat_axes = []
    for j, (name, models) in enumerate(subsets):
        sub = auc[auc.model.isin(models)]
        xpad = 0.06 * (sub["auc"].max() - sub["auc"].min())
        xr = (sub["auc"].min() - xpad, sub["auc"].max() + xpad)
        ax = fig.add_subplot(gB[j + 1]); scat_axes.append(ax)
        for m in models:
            s = sub[sub.model == m]
            ax.scatter(s["auc"], s[OUTC], s=13, color=get_model_color(m),
                       marker="D" if m in EXTREMES else "o", alpha=0.5, lw=0, zorder=2)
        pm = sub.groupby("model")[["auc", OUTC]].mean()
        psem = sub.groupby("model")[["auc", OUTC]].sem()
        pn = sub.groupby("model").size()
        for m in pm.index:
            tc = stats.t.ppf(0.975, max(int(pn[m]) - 1, 1))          # 95% CI = t * SEM over the model's sites
            ax.errorbar(pm.loc[m, "auc"], pm.loc[m, OUTC],
                        xerr=tc * psem.loc[m, "auc"], yerr=tc * psem.loc[m, OUTC],
                        fmt="D" if m in EXTREMES else "o", ms=8, mfc=get_model_color(m),
                        mec="k", mew=0.7, ecolor=get_model_color(m), elinewidth=0.9,
                        capsize=2.0, capthick=0.9, zorder=5)
        xs = np.linspace(*xr, 40)
        bs, as_ = np.polyfit(sub["auc"], sub[OUTC], 1)                 # site-level fit -> dashed
        ax.plot(xs, bs * xs + as_, "--", c="0.45", lw=1.4, dashes=(4, 2), zorder=4)
        bm, am = np.polyfit(pm["auc"], pm[OUTC], 1)                    # model-level fit -> bold solid
        ax.plot(xs, bm * xs + am, "-", c="0.12", lw=2.4, zorder=4)
        mr = stats.pearsonr(pm["auc"], pm[OUTC])[0]; sr = stats.pearsonr(sub["auc"], sub[OUTC])[0]
        rel = rel_sb[GROUP_KEYS[j]]
        ax.annotate(f"{len(sub)} channels\nmodel r = {mr:+.2f}  (bold fit)\nsite r = {sr:+.2f}  (dashed fit)",
                    (0.04, 0.04), xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5, color="0.2",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7, alpha=0.92))
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_title(f"{name} (n={len(models)})", fontsize=12.5, fontweight="bold", pad=17)
        ax.text(0.5, 1.012, f"outcome measure reliability: r = {rel:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.2, color="0.45", style="italic")
        ax.set_xlabel("Adversarial sensitivity (log-ε AUC)", fontsize=10)
        if j == 0: ax.set_ylabel(YLABEL, fontsize=10.5)
        else: ax.set_yticklabels([])
        if j == 0: _sensitivity_inset(ax, auc)          # per-model sensitivity summary, top-right of C

    # (a) mosaic title + panel letters
    fig.text((0.045 + 0.99) / 2, 0.968, "Encoding-axis adversarial perturbation per model",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(s0.get_position().x0 - 0.035, 0.968, "a", fontsize=20, fontweight="bold", va="center")
    for ax, lab in [(axb, "b")] + list(zip(scat_axes, "cde")):
        p = ax.get_position(); fig.text(p.x0 - 0.035, p.y1 + 0.008, lab, fontsize=20, fontweight="bold", va="bottom")
    return fig


def main(out_dir):
    fig = build()
    path = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print("saved ->", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", dest="out_dir", default=str(paths.output_dir() / "figures"))
    main(**vars(ap.parse_args()))
