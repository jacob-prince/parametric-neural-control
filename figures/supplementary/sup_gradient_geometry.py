#!/usr/bin/env python3
"""Supp fig (gradient_geometry) — encoding input-gradient geometry and neural control.

Structural companion of the adversarial-sensitivity supplement, with the adversarial data
swapped for input-gradient geometry (metric = gradient spectral participation ratio, the
main-figure measure).

  a      mosaic: top row = per-model input-gradient saliency of the cat seed at the
         example site (paul ch8 — the same site/seed as the main-figure mosaic); bottom
         row = the per-model accentuation of that seed toward the site, matched to a
         common achieved response (the highest response every trained model reaches;
         Untrained is capped below it and flagged). Columns ordered by per-model mean
         gradient spectral PR (annotated above each name).
  b      per-model gradient frequency spectra on the 100 held-out NSD images (thin
         per-site + bold per-model mean).
  c/d/e  held-out gradient spectral PR vs site-residualized control slope — residualized
         against the per-site mean over the 9 TRAINED models, the main-figure
         convention — for the three model subsets (all 10 / without adv.-trained 8 /
         without untrained 7); dashed = site-level fit, bold = model-mean fit; panel c
         carries the per-model PR inset.

Reads cluster outputs (cluster/outputs_from_cluster/fig5_heldout_gradfreq) +
preproc_data/{fig5_heldout_table.csv, sup_advsens_outcome_reliability.csv} + the
grad-map cache (loader.load_grad_maps) + accentuation-sweep PNGs
(data/stimuli_control).
"""
import os
import glob
import pickle
import re
import argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from pnc import paths
from pnc import utils as U
from pnc.utils import (save_fig, get_model_color, STIMULI_PATH, STIMULI_CONTROL_PATH,
                       MONKEY_DIRS, ACCENT_DATE_PREFIXES)
from pnc.manifest import output_name
from pnc.preproc import loader as L
U.apply_figure_style()

STEM = output_name("gradient_geometry")
PREPROC_DATA = str(paths.preprocessed_data())
CLUSTER_OUT = str(paths.cluster_outputs())

SHORT = dict(U.MODEL_SHORT_NAMES); SHORT["AlexNet_training_seed_01"] = "Untrained"
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
EXTREMES = ROBUST + [UNTR]; ROBUST_SET = set(U.ROBUST_MODELS)
KEY = ["model", "monkey", "channel"]
FMIN, FMAX = 1, 112

# example site + seed for the panel-a mosaic (matches the main-figure mosaic: paul ch8, cat seed)
ACC_MONKEY, ACC_UNIT, SEED_IDX = "paul", 8, 3
SEED_NAME = "shared0241_nsd20065.png"                          # seed idx 3 (cat)
MONKEY_DISP = {"red": "R", "paul": "P", "venus": "V", "leap": "L", "three0": "T"}


# ---- spectral / image helpers -------------------------------------------------
def normalize_row(p):
    p = np.asarray(p, float); s = p[FMIN:FMAX].sum()
    return p / s if s > 0 else p


def spectral_pr(p):
    """Participation ratio: effective number of frequency bands (= n / (1 + CoV^2))."""
    P = np.asarray(p, float)[FMIN:FMAX]
    d = float((P ** 2).sum())
    return float(P.sum() ** 2 / d) if d > 0 else float(np.nan)


def grad_to_saliency(grad_3hw, pct=99.0):
    mag = np.sqrt((grad_3hw.astype(np.float64) ** 2).sum(axis=0))
    hi = np.percentile(mag, pct)
    return np.clip(mag / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(mag)


def load_seed_image(name, size=224):
    p = os.path.join(STIMULI_PATH, name)
    if not os.path.exists(p):
        hits = [os.path.join(r, name) for r, _, fs in os.walk(STIMULI_PATH) if name in fs]
        p = hits[0] if hits else p
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def accent_curve(model, unit, img_idx, monkey):
    """Sorted [(level, achieved_score, path), ...] for one model+site+seed accentuation sweep."""
    d = os.path.join(STIMULI_CONTROL_PATH,
                     f"{ACCENT_DATE_PREFIXES[monkey]}_{MONKEY_DIRS[monkey]}_{model}_accentuation")
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf"^{re.escape(model)}_RidgeCV_unit_{unit}_img_{img_idx}_level_(-?\d+\.\d+)_score_(-?\d+\.\d+)\.png$")
    out = [(float(m.group(1)), float(m.group(2)), os.path.join(d, fn))
           for fn in sorted(os.listdir(d)) if (m := pat.match(fn))]
    return sorted(out)


def load_accent_at(curve, target_resp, size=224):
    """Pick the sweep image whose achieved response is closest to target_resp; return (img, score)."""
    if not curve:
        return None, None
    lvl, score, path = min(curve, key=lambda t: abs(t[1] - target_resp))
    img = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return img, score


def load_heldout_profiles():
    """(monkey, unit, model) -> mean radial gradient power profile over the 100 held-out NSD
    images (the same probe set as the adversarial-sensitivity companion; non-circular)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(paths.require(os.path.join(CLUSTER_OUT, "fig5_heldout_gradfreq")), "*.pkl"))):
        mm = re.match(r"(.+?)_unit_(\d+)_model_(.+)_grad_maps_freq_profiles\.pkl", os.path.basename(f))
        if not mm:
            continue
        with open(f, "rb") as fh:
            out[(mm.group(1).split("_")[0], int(mm.group(2)), mm.group(3))] = \
                np.asarray(pickle.load(fh)["profiles"], float).mean(0)
    return out


def _pr_inset(ax, pr):
    """Per-model gradient spectral PR (mean +/- SEM over sites): square box in the bottom-left
    corner, above the fit annotation; colour-coded bold model labels on the left; value at each
    bar end; SEM whiskers."""
    per = pr.groupby("model")["pr"].agg(["mean", "sem"]).sort_values("mean")
    ax.figure.canvas.draw()
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ih = 0.215; iw = ih * (bb.height * fh) / (bb.width * fw)
    # bottom-left, between the fit annotation (top 0.17) and the lowest robust site (0.45);
    # model labels + x label extend to ~0.07 left and ~0.035 below the box
    ins = ax.inset_axes([0.255, 0.215, iw, ih]); ins.set_box_aspect(1)
    ys = np.arange(len(per)); means = per["mean"].values; sems = per["sem"].values
    colors = [get_model_color(m) for m in per.index]
    ins.barh(ys, means, height=0.72, color=colors, alpha=0.95, edgecolor="white", linewidth=0.6, zorder=2)
    ins.errorbar(means, ys, xerr=sems, fmt="none", ecolor="0.3", elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=3)
    ins.set_yticks(ys); ins.set_yticklabels([SHORT.get(m, m) for m in per.index], fontsize=6.0)
    for tick, c in zip(ins.get_yticklabels(), colors):
        tick.set_color(c); tick.set_fontweight("bold")
    pad = 0.03 * float(means.max())
    for y, a, s in zip(ys, means, sems):
        ins.text(a + s + pad, y, f"{a:.0f}", fontsize=5.6, va="center", ha="left", color="#444", fontweight="bold")
    ins.set_xlabel("gradient spectral PR", fontsize=7.2, labelpad=2); ins.set_xticks([])
    ins.tick_params(axis="y", length=0, pad=3)
    ins.set_ylim(len(per) - 0.4, -0.6); ins.set_xlim(0, float(means.max()) + 13 * pad)
    for s_ in ("top", "right", "bottom", "left"):
        ins.spines[s_].set_visible(True); ins.spines[s_].set_edgecolor("#aaa"); ins.spines[s_].set_linewidth(0.8)
    ins.set_facecolor((1, 1, 1, 0.95))


def build():
    # per-site held-out gradient spectral PR + control slope (main-figure convention)
    hoprof = load_heldout_profiles()
    pr = pd.DataFrame([dict(monkey=mk, channel=u, model=m, pr=spectral_pr(p))
                       for (mk, u, m), p in hoprof.items()])
    ht = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, "fig5_heldout_table.csv"),
                                   hint="python scripts/preprocessing/fig5_heldout_analysis.py"))
    pr = ht[KEY + ["control_slope"]].merge(pr, on=KEY).dropna(subset=["pr", "control_slope"])
    rel_sb = pd.read_csv(paths.require(os.path.join(PREPROC_DATA, "sup_advsens_outcome_reliability.csv"),
                                       hint="python scripts/preprocessing/build_advsens_outcome_reliability.py")
                         ).set_index("group")["reliability_sb"].to_dict()
    GROUP_KEYS = ["all", "minus2adv", "conv7"]        # panels c, d, e

    # site-residualized control slope: per-site mean over the 9 TRAINED models
    tr_mean = pr[pr.model != UNTR].groupby(["monkey", "channel"])["control_slope"].mean()
    pr["control_slope"] = (pr["control_slope"].values
                           - tr_mean.reindex(pd.MultiIndex.from_frame(pr[["monkey", "channel"]])).values)
    OUTC, YLABEL = "control_slope", "Control slope (site residual)"

    # per-model mean held-out spectra + PR
    per_model_prof, per_pr = {}, pr.groupby("model")["pr"].mean()
    for m in per_pr.index:
        profs = [normalize_row(hoprof[k]) for k in hoprof if k[2] == m]
        per_model_prof[m] = np.mean(profs, axis=0)
    ranked = sorted(per_pr.index, key=lambda m: -per_pr[m])   # high PR left, adv-trained right

    # mosaic ingredients (site = paul8, cat seed) — matched on ACHIEVED response: common
    # target = min over models of each model's max achieved response, EXCLUDING Untrained
    grad_maps = L.load_grad_maps(ACC_MONKEY, ACC_UNIT)
    seed_img = load_seed_image(SEED_NAME)
    curves = {m: accent_curve(m, ACC_UNIT, SEED_IDX, ACC_MONKEY) for m in ranked}
    maxresp = {m: max((s for _, s, _ in c), default=None) for m, c in curves.items()}
    pool = [v for m, v in maxresp.items() if v is not None and m != UNTR]
    target_resp = min(pool) if pool else max(v for v in maxresp.values() if v is not None)

    plt.rcParams.update({"xtick.labelsize": 9.5, "ytick.labelsize": 9.5})

    fig = plt.figure(figsize=(13.7, 8.3))
    outer = GridSpec(2, 1, height_ratios=[0.92, 1.18], hspace=0.24,
                     left=0.045, right=0.99, top=0.909, bottom=0.114)
    gB = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], width_ratios=[1, 1, 1, 1], wspace=0.20)

    # ---- (a) mosaic: gradient maps (top) + accentuations (bottom) ----
    n = len(ranked)
    gA = GridSpecFromSubplotSpec(2, n + 1, subplot_spec=outer[0], hspace=0.06, wspace=0.045)
    s0 = fig.add_subplot(gA[0, 0]); s0.imshow(seed_img)
    s0.set_xticks([]); s0.set_yticks([]); s0.set_title("seed", fontsize=10, pad=14)
    s0.set_ylabel("input gradient", fontsize=9.5)
    sL = fig.add_subplot(gA[1, 0]); sL.set_xticks([]); sL.set_yticks([])
    for sp in sL.spines.values(): sp.set_visible(False)
    sL.set_ylabel("accentuation", fontsize=9.5)
    sL.text(0.5, 0.5, f"accentuate\ntoward\nMonkey {MONKEY_DISP[ACC_MONKEY]} ch{ACC_UNIT}\n"
            f"(matched to\nresp +{target_resp:.1f})", ha="center", va="center",
            fontsize=7.0, color="0.4", transform=sL.transAxes)
    for j, m in enumerate(ranked):
        a0 = fig.add_subplot(gA[0, j + 1]); a0.set_xticks([]); a0.set_yticks([])
        if m in grad_maps:
            a0.imshow(grad_to_saliency(grad_maps[m][SEED_IDX]), cmap="magma", vmin=0, vmax=0.65)
        else:
            a0.axis("off")
        a0.set_title(SHORT.get(m, m), fontsize=9.5, fontweight="bold", color=get_model_color(m), pad=14)
        a0.annotate(f"PR {per_pr[m]:.0f}", (0.5, 1.03), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=7.2, color="0.3")
        a1 = fig.add_subplot(gA[1, j + 1]); a1.set_xticks([]); a1.set_yticks([])
        stim, score = load_accent_at(curves[m], target_resp)
        a1.imshow(stim) if stim is not None else a1.axis("off")
        if score is not None:
            # Untrained is capped below the common target -> flag it; others land at the matched resp
            capped = (m == UNTR and maxresp[m] is not None and maxresp[m] < target_resp - 0.15)
            a1.set_xlabel(f"resp {score:+.1f}" + ("*" if capped else ""), fontsize=8.5, labelpad=2,
                          color="0.55" if capped else "0.3")

    # ---- (b) gradient frequency spectra (held-out images) ----
    x = np.arange(FMIN, FMAX)
    axb = fig.add_subplot(gB[0])
    for m in ranked:
        c = get_model_color(m); is_rob = m in ROBUST_SET
        for k in hoprof:
            if k[2] == m:
                axb.plot(x, normalize_row(hoprof[k])[FMIN:FMAX], color=c, lw=0.4, alpha=0.16, zorder=1)
        axb.plot(x, per_model_prof[m][FMIN:FMAX], color=c, lw=2.6 if is_rob else 1.7,
                 alpha=1.0 if is_rob else 0.9, zorder=9 if is_rob else 5, solid_capstyle="round")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("Spatial frequency (cyc/img)", fontsize=10)
    axb.set_ylabel("Normalized radial power", fontsize=9.5)
    axb.set_title("Gradient frequency\nspectra", fontsize=12, fontweight="bold")

    # ---- (c–e) gradient spectral PR ↔ control by model subset ----
    subsets = [("All models", list(pr.model.unique())),
               ("Without adv.-trained", [m for m in pr.model.unique() if m not in ROBUST]),
               ("Without untrained", [m for m in pr.model.unique() if m not in EXTREMES])]
    yr = (pr[OUTC].min() - 0.06, pr[OUTC].max() + 0.06)
    scat_axes = []
    for j, (name, models) in enumerate(subsets):
        sub = pr[pr.model.isin(models)]
        xpad = 0.06 * (sub["pr"].max() - sub["pr"].min())
        xr = (sub["pr"].min() - xpad, sub["pr"].max() + xpad)
        ax = fig.add_subplot(gB[j + 1]); scat_axes.append(ax)
        for m in models:
            s = sub[sub.model == m]
            ax.scatter(s["pr"], s[OUTC], s=13, color=get_model_color(m),
                       marker="D" if m in EXTREMES else "o", alpha=0.5, lw=0, zorder=2)
        pm = sub.groupby("model")[["pr", OUTC]].mean()
        psem = sub.groupby("model")[["pr", OUTC]].sem()
        pn = sub.groupby("model").size()
        for m in pm.index:
            tc = stats.t.ppf(0.975, max(int(pn[m]) - 1, 1))          # 95% CI = t * SEM over the model's sites
            ax.errorbar(pm.loc[m, "pr"], pm.loc[m, OUTC],
                        xerr=tc * psem.loc[m, "pr"], yerr=tc * psem.loc[m, OUTC],
                        fmt="D" if m in EXTREMES else "o", ms=8, mfc=get_model_color(m),
                        mec="k", mew=0.7, ecolor=get_model_color(m), elinewidth=0.9,
                        capsize=2.0, capthick=0.9, zorder=5)
        xs = np.linspace(*xr, 40)
        bs, as_ = np.polyfit(sub["pr"], sub[OUTC], 1)                 # site-level fit -> dashed
        ax.plot(xs, bs * xs + as_, "--", c="0.45", lw=1.4, dashes=(4, 2), zorder=4)
        bm, am = np.polyfit(pm["pr"], pm[OUTC], 1)                    # model-level fit -> bold solid
        ax.plot(xs, bm * xs + am, "-", c="0.12", lw=2.4, zorder=4)
        mr = stats.pearsonr(pm["pr"], pm[OUTC])[0]; sr = stats.pearsonr(sub["pr"], sub[OUTC])[0]
        rel = rel_sb[GROUP_KEYS[j]]
        ax.annotate(f"{len(sub)} channels\nmodel r = {mr:+.2f}  (bold fit)\nsite r = {sr:+.2f}  (dashed fit)",
                    (0.04, 0.04), xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5, color="0.2",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7, alpha=0.92))
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_title(f"{name} (n={len(models)})", fontsize=12.5, fontweight="bold", pad=17)
        ax.text(0.5, 1.012, f"outcome measure reliability: r = {rel:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.2, color="0.45", style="italic")
        ax.set_xlabel("Gradient freq. participation ratio", fontsize=10)
        if j == 0: ax.set_ylabel(YLABEL, fontsize=10.5)
        else: ax.set_yticklabels([])
        if j == 0: _pr_inset(ax, pr)                    # per-model PR summary, bottom-left of C

    # (a) mosaic title + panel letters
    fig.text((0.045 + 0.99) / 2, 0.968, "Encoding input-gradient geometry per model",
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
