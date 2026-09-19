#!/usr/bin/env python3
"""Figure 5 - adversarial sensitivity AND input-gradient geometry, paired, predict parametric control.

Panel a is one big integrated 4-row mosaic (columns = the 10 models, ordered by spectral flatness
descending so robust models sit on the right; no seed column):
  row 1  adversarial - per-model minimal-eps adversarial perturbation of the cat seed (+2 z) toward the site
  row 2  perturbation - the perturbation itself
  row 3  input gradient - per-model input-gradient saliency of the cat seed
  row 4  accentuation - the accentuation of the cat seed toward the site, matched to a common achieved
                          response (the highest response every non-untrained model reaches)
Row labels are on the far left; model names are the column headers.

Bottom analysis band (all 10 models):
  b  adversarial sensitivity vs perturbation strength
  c  gradient frequency spectra
  d  dual scatter: adversarial sensitivity | gradient metric, both vs control
  e  summary bars: correlation across model subsets vs the seed-split noise ceiling

Data: fig5_adv (cluster PGD outputs + heldout table), fig5_grad (gradient spectra/maps via the
loader + accentuation stimuli), preprocessed_data/fig5_* tables. Variant knobs (defaults = the
manuscript render): site=paul8, seed_img=3 (cat), outcome=slope, resid=9trained, gradmetric=pr.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from scipy import stats
from pnc import paths
from pnc import utils as U
from pnc.manifest import MAIN_FIGURES
from pnc.utils import save_fig, get_model_color
from pnc.preproc import loader as L
from figures.main import fig5_adv as F8             # adv-sensitivity data + AUC window
from figures.main import fig5_grad as F7            # gradient data + spectra/flatness/accentuation helpers
U.apply_figure_style()

PREPROC_DATA = str(paths.preprocessed_data())
CLUSTER_OUT = str(paths.cluster_outputs())
SHORT = dict(U.MODEL_SHORT_NAMES); SHORT["AlexNet_training_seed_01"] = "Untrained"
ROBUST = F7.ROBUST; UNTR = F7.UNTR; EXTREMES = F7.EXTREMES; ROBUST_SET = F7.ROBUST_SET
FMIN, FMAX = F7.FMIN, F7.FMAX
C_NATURAL = F7.C_NATURAL
KEY = ["model", "monkey", "channel"]
MONKEY_DISP = F7.MONKEY_DISP

# ---- variant configuration (take9 env knobs SITE / SEED_IMG / OUTCOME / RESID / GRADMETRIC) ----
# Module-level defaults = the manuscript render; main() re-binds them per call via _configure().
SITE = "paul8"                                     # example site for the mosaic (rows 1-4)
ACC_MONKEY, ACC_UNIT, SEED_IDX, SEED_NAME = F7.site_config(SITE, 3)
SEED_TAG = "seed" + str(SEED_IDX)
OUTCOME = "slope"                                  # "slope" | "r": control outcome for the bottom band (d, e)
OUTC = {"slope": "control_slope", "r": "control_r"}[OUTCOME]
RESID = "9trained"                                 # "10" | "9trained" | "none" (absolute outcome)
OUT_LABEL = {"slope": "Control slope", "r": "Control r"}[OUTCOME]
OUT_WORD = {"slope": "control slope", "r": "control r"}[OUTCOME]
GRADMETRIC = "pr"                                  # "flatness" | "cv" | "pr": gradient-spectrum predictor
GCOL = {"cv": "ho_cv", "flatness": "ho_flat", "pr": "ho_pr"}[GRADMETRIC]   # held-out-image metric (100 NSD imgs)
GLABEL = {"cv": "Gradient spectral CoV", "flatness": "Gradient spectral flatness",
          "pr": "Gradient freq. participation ratio"}[GRADMETRIC]
GSHORT = {"cv": "spectral CoV", "flatness": "spectral flatness", "pr": "participation ratio"}[GRADMETRIC]
GLEGEND = {"cv": "gradient CoV", "flatness": "gradient flatness", "pr": "Gradient freq. PR"}[GRADMETRIC]


def _configure(site, seed_img, outcome, resid, gradmetric):
    global SITE, ACC_MONKEY, ACC_UNIT, SEED_IDX, SEED_NAME, SEED_TAG, OUTCOME, OUTC, RESID
    global OUT_LABEL, OUT_WORD, GRADMETRIC, GCOL, GLABEL, GSHORT, GLEGEND
    SITE = site
    ACC_MONKEY, ACC_UNIT, SEED_IDX, SEED_NAME = F7.site_config(site, seed_img)
    SEED_TAG = "seed" + str(SEED_IDX)
    OUTCOME = outcome
    OUTC = {"slope": "control_slope", "r": "control_r"}[OUTCOME]
    RESID = resid
    OUT_LABEL = {"slope": "Control slope", "r": "Control r"}[OUTCOME]
    OUT_WORD = {"slope": "control slope", "r": "control r"}[OUTCOME]
    GRADMETRIC = gradmetric
    GCOL = {"cv": "ho_cv", "flatness": "ho_flat", "pr": "ho_pr"}[GRADMETRIC]
    GLABEL = {"cv": "Gradient spectral CoV", "flatness": "Gradient spectral flatness",
              "pr": "Gradient freq. participation ratio"}[GRADMETRIC]
    GSHORT = {"cv": "spectral CoV", "flatness": "spectral flatness", "pr": "participation ratio"}[GRADMETRIC]
    GLEGEND = {"cv": "gradient CoV", "flatness": "gradient flatness", "pr": "Gradient freq. PR"}[GRADMETRIC]


def _site_resid(df, col, keys):
    """col minus its per-site mean; reference = all models (RESID=='10') or the 9 trained (RESID=='9trained').
    RESID=='none' -> passthrough (absolute outcome, no residualization)."""
    if RESID == "none":
        return df[col]
    src = df[df["model"] != UNTR] if RESID == "9trained" else df
    mean = src.groupby(keys)[col].mean()
    return df[col].values - mean.reindex(pd.MultiIndex.from_frame(df[keys])).values


def spectral_cv(p):                                     # coefficient of variation of the radial power spectrum
    P = np.asarray(p, float)[FMIN:FMAX]
    return float(np.std(P) / np.mean(P))


def spectral_pr(p):                                     # participation ratio: effective number of frequency bands
    P = np.asarray(p, float)[FMIN:FMAX]                 # (= n / (1 + CoV^2), an exact transform of spectral CoV)
    return float(P.sum() ** 2 / (P ** 2).sum())


def _load_heldout_profiles():
    """(monkey, unit, model) -> mean radial gradient power profile over the 100 held-out NSD images
    (the same probe set as the adversarial-sensitivity analysis; non-circular, never used for synthesis)."""
    import pickle, re
    d = paths.require(os.path.join(CLUSTER_OUT, "fig5_heldout_gradfreq"))
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        mm = re.match(r"(.+?)_unit_(\d+)_model_(.+)_grad_maps_freq_profiles\.pkl", os.path.basename(f))
        if not mm:
            continue
        key = (mm.group(1).split("_")[0], int(mm.group(2)), mm.group(3))
        with open(f, "rb") as fh:
            out[key] = np.asarray(pickle.load(fh)["profiles"], float).mean(0)
    return out


def load_gallery(monkey, unit, seed_tag):
    """Per-model minimal-eps adversarial-attack record for one site+seed (cluster advvis_exact)."""
    recs = {}
    for p in sorted(glob.glob(os.path.join(str(paths.require(F8.ADVVIS_DIR)), f"advvis_{monkey}_Ch{unit}_{seed_tag}_*.npz"))):
        d = np.load(p, allow_pickle=True)
        recs[str(d["model"])] = dict(eps=float(d["eps_used"]), clean=d["clean"], adv=d["adv"],
                                     delta=float(d["target_delta"]), cp=float(d["clean_pred"]),
                                     ap=float(d["adv_pred"]))
    return recs


def williams_p(r_ys, r_yf, r_sf, n):
    """Two-sided p for H0: corr(control,sens) == corr(control,flat) - two DEPENDENT correlations that
    share the control variable (Williams/Hotelling test). r_sf = corr(sens, flat)."""
    if n < 5 or not np.all(np.isfinite([r_ys, r_yf, r_sf])):
        return np.nan
    det = 1 - r_ys ** 2 - r_yf ** 2 - r_sf ** 2 + 2 * r_ys * r_yf * r_sf
    rbar = (r_ys + r_yf) / 2
    denom = 2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r_sf) ** 3
    if denom <= 0:
        return np.nan
    t = (r_ys - r_yf) * np.sqrt((n - 1) * (1 + r_sf) / denom)
    return float(2 * stats.t.sf(abs(t), n - 3))


def pair_stats(comb, ycol):
    """Per subset (10/9/7) and level (model/site): signed r of sens & flat with control, and a Williams
    p for the GAP between the two predictors after ORIENTING both to the full-set (10-model) direction.
    Orientation = multiply each subset correlation by the sign of its own 10-model correlation (NOT abs):
    a predictor keeping its 10-model direction reads +|r|, one that flips reads a NEGATIVE oriented value,
    so the tested gap correctly widens (e.g. +0.26 vs a flipped -0.05 -> gap 0.31, not the abs 0.21).
    We do NOT suppress when one flips: with both on a common axis, testing +ve vs -ve is the whole point.
    comb has cols sens, flat, model, monkey, channel, <ycol>."""
    groups = {10: list(comb.model.unique()),
              9: [m for m in comb.model.unique() if m != UNTR],
              7: [m for m in comb.model.unique() if m not in EXTREMES]}

    def _rs(s, lev):
        if lev == "model":
            s = s.groupby("model")[["sens", "flat", ycol]].mean()
        return (stats.pearsonr(s.sens, s[ycol])[0], stats.pearsonr(s.flat, s[ycol])[0],
                stats.pearsonr(s.sens, s.flat)[0], len(s))

    base = {lev: _rs(comb, lev)[:2] for lev in ("site", "model")}     # full-set (10-model) signs
    out = {}
    for k, ms in groups.items():
        s = comb[comb.model.isin(ms)]; d = {}
        for lev in ("site", "model"):
            rs, rf, rsf, n = _rs(s, lev)
            ss, sf = np.sign(base[lev][0]), np.sign(base[lev][1])     # 10-model direction of each predictor
            vs, vf = rs * ss, rf * sf                                 # oriented to 10-model direction (sign-flip, not abs)
            d[f"{lev}_r_sens"] = rs; d[f"{lev}_r_flat"] = rf
            d[f"{lev}_diff_p"] = williams_p(vs, vf, ss * sf * rsf, n)  # gap in the common direction; a flip -> negative vs/vf
        out[k] = d
    return out


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


# seed-level hierarchical bootstrap bracket p-values; read if present, else Williams.
# fig5_bracket_bootstrap_p.csv is a frozen resid9-only input (its regenerator depends on the
# take8 SX_predicting pipeline and was not ported).
_BOOTP = {}


def _load_bootp():
    global _BOOTP
    _BOOTP = {}
    try:
        _bp = pd.read_csv(os.path.join(PREPROC_DATA, "fig5_bracket_bootstrap_p.csv"))
        _BOOTP = {(r.scope, r.outcome, r.level, int(r.subset)): float(r.p) for r in _bp.itertuples()}
    except Exception:
        pass


C_SENS, C_FLAT = "#8A8A8A", "#1A1A1A"       # adversarial sensitivity (grey) vs gradient flatness (black, emphasized)
SUBSET_COL = {10: "#5E35B1", 9: "#00838F", 7: "#C2185B"}   # all / w-o untrained / w-o untrained+adv (label colors)


def _flatness_inset_nonat(ax, per_model_flat, loc="bl", metric_label="spectral flatness"):
    """Per-model gradient-metric bar inset (no 'Natural' row), centred low in the spectra panel,
    model names outside the bars in black."""
    rows = sorted(list(per_model_flat), key=lambda t: t[1])
    ih, iw = 0.315, 0.44
    pos = [0.36, 0.06]                          # centred low; sits below the converged spectra
    ins = ax.inset_axes(pos + [iw, ih]); ins.set_zorder(12)   # draw over the spectra so nothing crosses it
    ys = np.arange(len(rows)); vals = np.array([r[1] for r in rows]); keys = [r[0] for r in rows]
    colors = [get_model_color(k) for k in keys]
    ins.barh(ys, vals, height=0.82, color=colors, alpha=0.97, edgecolor="white", linewidth=0.5)
    ins.set_yticks(ys); ins.set_yticklabels([SHORT.get(k, k) for k in keys], fontsize=6.0, color="black")
    for tick in ins.get_yticklabels():         # white backing so names stay legible over the spectra
        tick.set_bbox(dict(facecolor=(1, 1, 1, 0.82), edgecolor="none", pad=0.4))
    pad = 0.03 * float(vals.max())
    vfmt, headroom = (("{:.2f}", 10) if vals.max() < 10 else ("{:.1f}", 20))   # wide values need more room
    for y, a in zip(ys, vals):
        ins.text(a + pad, y, vfmt.format(a), fontsize=5.6, va="center", ha="left", color="#333", fontweight="bold")
    ins.set_xlabel(metric_label, fontsize=7, labelpad=2); ins.set_xticks([])
    ins.tick_params(axis="y", length=0, pad=2)
    ins.set_ylim(len(rows) - 0.4, -0.6); ins.set_xlim(0, float(vals.max()) + headroom * pad)
    for s_ in ("top", "right", "bottom", "left"):
        ins.spines[s_].set_visible(True); ins.spines[s_].set_edgecolor("#aaa"); ins.spines[s_].set_linewidth(0.8)
    ins.set_facecolor((1, 1, 1, 0.95))


def _scatter(ax, sub, xcol, OUTC, xr, yr, xlabel, show_y):
    """Site dots + per-model mean dots (95% CI); dashed=site fit, bold=model fit; annotation top-right."""
    for m in sub.model.unique():
        s = sub[sub.model == m]
        ax.scatter(s[xcol], s[OUTC], s=12, color=get_model_color(m),
                   marker="D" if m in EXTREMES else "o", alpha=0.5, lw=0, zorder=2)
    pm = sub.groupby("model")[[xcol, OUTC]].mean()
    psem = sub.groupby("model")[[xcol, OUTC]].sem(); pn = sub.groupby("model").size()
    for m in pm.index:
        tc = stats.t.ppf(0.975, max(int(pn[m]) - 1, 1))
        ax.errorbar(pm.loc[m, xcol], pm.loc[m, OUTC], xerr=tc * psem.loc[m, xcol], yerr=tc * psem.loc[m, OUTC],
                    fmt="D" if m in EXTREMES else "o", ms=7, mfc=get_model_color(m), mec="k", mew=0.6,
                    ecolor=get_model_color(m), elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)
    xs = np.linspace(*xr, 40)
    bs, as_ = np.polyfit(sub[xcol], sub[OUTC], 1)
    ax.plot(xs, bs * xs + as_, "--", c="0.45", lw=1.3, dashes=(4, 2), zorder=4)
    bm, am = np.polyfit(pm[xcol], pm[OUTC], 1)
    ax.plot(xs, bm * xs + am, "-", c="0.12", lw=2.2, zorder=4)
    mr = stats.pearsonr(pm[xcol], pm[OUTC])[0]; sr = stats.pearsonr(sub[xcol], sub[OUTC])[0]
    ax.annotate(f"model r = {mr:+.3f}  (solid)\nsite r = {sr:+.3f}  (dashed)", (0.045, 0.045),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8.0, color="0.2",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7, alpha=0.92))
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_xlabel(xlabel, fontsize=9.5)
    if show_y:
        ysuf = "" if RESID == "none" else " (site residuals)"
        ax.set_ylabel(f"{OUT_LABEL}{ysuf}", fontsize=10)
    else:
        ax.set_yticklabels([])


def _summary_bars(ax, cc, ceil, scope="pooled"):
    """Summary panel (horizontal bars): SIGNED correlation of adversarial sensitivity (grey) vs gradient
    flatness (black) with control, at the model and site level, for the 10/8/7 model subsets. Bars are
    plotted as -r, so the (negative) full-set relationship reads POSITIVE; a subset that reverses sign
    (does NOT generalise) dips just below zero, left of the dashed 0-line. Subset identity = label
    colour; dashed line + band = seed-split correlation ceiling."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    slot_y = {("model", 10): 8.0, ("model", 9): 7.0, ("model", 7): 6.0,
              ("site", 10): 2.0, ("site", 9): 1.0, ("site", 7): 0.0}
    dh, h, XNEG = 0.20, 0.34, -0.08                          # only a sliver below 0 -> enough to show a flip

    def _sgn(pred, lev):    # orient by the full-set (k=10) sign so full set reads +; a subset that flips goes -
        return 1.0 if cc[10][f"{lev}_r_{pred}"] >= 0 else -1.0
    for (level, k), yy in slot_y.items():
        vs = cc[k][f"{level}_r_sens"] * _sgn("sens", level)     # -r for a negative predictor, +r for a positive one
        vf = cc[k][f"{level}_r_flat"] * _sgn("flat", level)     # flatness (neg) -> -r ; spectral CV (pos) -> +r
        ax.barh(yy + dh, max(vs, XNEG), height=h, color=C_SENS, edgecolor="none", zorder=3)
        ax.barh(yy - dh, max(vf, XNEG), height=h, color=C_FLAT, edgecolor="none", zorder=3)
        for val, ysub, barcol in [(vs, yy + dh, C_SENS), (vf, yy - dh, C_FLAT)]:
            if val >= 0.20:                                    # generalises -> white value at the bar tip
                ax.text(val - 0.02, ysub, f"{val:.3f}", va="center", ha="right",
                        fontsize=7.4, color="white", fontweight="bold", zorder=6)
            elif val >= 0:                                     # weak -> value just outside, in bar colour
                ax.text(val + 0.02, ysub, f"{val:.3f}", va="center", ha="left", fontsize=7.4, color=barcol)
            else:                                              # SIGN FLIP -> bar dips left of the 0-line; signed value
                ax.text(0.015, ysub, f"{val:+.3f}", va="center", ha="left", fontsize=7.4,
                        color=barcol, fontweight="bold", zorder=6)
        c, lo, hi = ceil[(level, k)]
        ax.fill_betweenx([yy - 0.46, yy + 0.46], lo, hi, color="0.55", alpha=0.15, lw=0, zorder=1)
        ax.plot([c, c], [yy - 0.46, yy + 0.46], ls=(0, (3, 1.6)), color="0.35", lw=1.1, zorder=4)
        bp = (_BOOTP.get((scope, OUTC, level, k), np.nan)
              if RESID == "9trained" else np.nan)               # bootstrap file is resid9-only; else Williams
        st = _stars(bp if np.isfinite(bp) else cc[k].get(f"{level}_diff_p", np.nan))
        if st and max(vs, vf) > 0:                              # need >=1 predictor still in the 10-model direction
                                                                # (both-flip = neither predicts -> no 'stronger' claim)
            xb = max(vs, vf) + 0.03
            _vr = lambda v: v if v >= 0.16 else (v + 0.11 if v >= 0 else 0.145)   # right extent of each value label
            if xb < max(_vr(vs), _vr(vf)) + 0.02:               # bracket would land on the printed values (short bars):
                ax.text(max(vs, vf), yy + dh + 0.20, st, va="bottom", ha="center", fontsize=8.0,   # bump above top bar
                        fontweight="bold", color="0.15", zorder=7, clip_on=False)
            else:                                               # long bars: bracket between the two, asterisk at right
                ax.plot([xb, xb], [yy - dh, yy + dh], color="0.15", lw=0.8, zorder=7, clip_on=False)
                for ye in (yy - dh, yy + dh):
                    ax.plot([xb - 0.018, xb], [ye, ye], color="0.15", lw=0.8, zorder=7, clip_on=False)
                ax.text(xb + 0.012, yy, st, va="center", ha="left", fontsize=8.0, fontweight="bold",
                        color="0.15", zorder=7, clip_on=False)
    ax.axvline(0, color="0.5", lw=0.6, ls=(0, (4, 3)), zorder=2)          # x = 0 (thin dashed)
    NVAL = {"model": {10: 10, 9: 9, 7: 7}, "site": {10: 250, 9: 225, 7: 175}}   # site = 25 sites/model
    def _ylab(lev, k):    # wrapped so the widest line is "Conventionally" (stays out of panel d)
        n = NVAL[lev][k]
        return {10: f"All models\n(n = {n})", 9: f"Trained models\n(n = {n})",
                7: f"Conventionally\ntrained (n = {n})"}[k]
    ax.set_yticks(list(slot_y.values()))
    ax.set_yticklabels([_ylab(lev, k) for (lev, k) in slot_y], fontsize=8.2)
    for tick, (lev, k) in zip(ax.get_yticklabels(), slot_y):
        tick.set_color(SUBSET_COL[k])
    ax.set_ylim(-0.5, 9.3); ax.set_xlim(XNEG, 1.16)
    ax.set_xticks([0, 0.5, 1.0]); ax.tick_params(axis="x", labelsize=9.5); ax.tick_params(axis="y", length=0)
    xsuf = "(sign-flipped)" if RESID == "none" else "(site residuals; sign-flipped)"
    ax.set_xlabel(f"Correlation with {OUT_WORD}\n{xsuf}", fontsize=9.5)
    # level headers sit horizontally above the "all models" row of each group, right-aligned on the axis
    ax.annotate("by model", xy=(-0.02, 8.7), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=10.5, fontweight="bold", color="0.25", annotation_clip=False)
    ax.annotate("by site", xy=(-0.02, 2.7), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=10.5, fontweight="bold", color="0.25", annotation_clip=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # legend centred in the band between the by-model bars and the "by site" header -> equal whitespace
    handles = [Patch(fc=C_SENS), Patch(fc=C_FLAT),
               (Patch(fc="0.55", alpha=0.3, ec="none"),
                Line2D([0], [0], ls=(0, (3, 1.6)), color="0.35", lw=1.1))]   # dashed line over its band
    labels = ["Adversarial sensitivity", GLEGEND, "noise ceiling  (± 95%)"]
    yfrac = (4.15 - (-0.5)) / (9.3 - (-0.5))
    lg = ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                   fontsize=8.2, loc="center", frameon=True, handlelength=1.4,
                   ncol=1, labelspacing=0.32, borderpad=0.5, handletextpad=0.5, bbox_to_anchor=(0.5, yfrac))
    lg.get_frame().set(facecolor="white", edgecolor="#cccccc", lw=0.6)


def build():
    # ---------- data ----------
    _load_bootp()
    g, merged = F7.load_data()
    merged = merged.dropna(subset=[OUTC, "flatness", "slope"]).copy()
    # gradient spectra on the HELD-OUT image set (100 NSD imgs, matched to adv. sensitivity; non-circular)
    hoprof = _load_heldout_profiles()                    # (monkey,unit,model) -> mean held-out radial profile
    merged["ho_flat"] = [F7.spectral_flatness(hoprof[(mk, int(u), m)]) for mk, u, m in zip(merged.monkey, merged.unit, merged.model)]
    merged["ho_cv"] = [spectral_cv(hoprof[(mk, int(u), m)]) for mk, u, m in zip(merged.monkey, merged.unit, merged.model)]
    merged["ho_pr"] = [spectral_pr(hoprof[(mk, int(u), m)]) for mk, u, m in zip(merged.monkey, merged.unit, merged.model)]
    merged = merged.dropna(subset=[GCOL]).copy()
    merged[OUTC] = _site_resid(merged, OUTC, ["monkey", "unit"])

    x = np.asarray(g["freqs"])[FMIN:FMAX]
    per_model_prof, per_model_flat, per_model_gmetric = {}, {}, {}
    for m in set(k[2] for k in hoprof):
        profs = [F7.normalize_row(hoprof[k]) for k in hoprof if k[2] == m]
        mean = np.mean(profs, axis=0); per_model_prof[m] = mean
        per_model_flat[m] = F7.spectral_flatness(mean)
        per_model_gmetric[m] = (spectral_cv(mean) if GCOL == "ho_cv"
                                else spectral_pr(mean) if GCOL == "ho_pr" else per_model_flat[m])
    ranked = sorted(per_model_flat, key=lambda m: -per_model_flat[m])     # high flatness left, robust right

    ht = F8.load_heldout_table()
    auc = ht[KEY + ["control_slope", "control_r", "reliability"]].merge(F8.load_heldout_logauc(), on=KEY)
    auc[OUTC] = _site_resid(auc, OUTC, ["monkey", "channel"])
    curves = F8.load_heldout_curves()

    # even/odd-seed outcome noise ceiling (correlation ceiling = sqrt reliability); cached once per outcome
    _r9 = {"9trained": "_resid9", "none": "_noresid"}.get(RESID, "")
    ceil_csv = os.path.join(PREPROC_DATA, f"fig5_outcome_ceiling_seedsplit{'_r' if OUTCOME == 'r' else ''}{_r9}.csv")
    ceil_csv = paths.require(ceil_csv, hint=f"python scripts/preprocessing/fig5_outcome_ceiling_seedsplit.py "
                                            f"(outcome={OUTCOME}, resid={RESID})")
    _ct = pd.read_csv(ceil_csv)
    CEIL = {(row.level, int(row.group)): (row.ceiling, row.ceiling_lo, row.ceiling_hi)
            for row in _ct.itertuples()}

    # mosaic ingredients (site = paul8)
    gal = load_gallery(ACC_MONKEY, ACC_UNIT, SEED_TAG)
    grad_maps = F7.load_grad_maps_safe(ACC_MONKEY, ACC_UNIT)
    acc_curves = {m: F7.accent_curve(m, ACC_UNIT, SEED_IDX, ACC_MONKEY) for m in ranked}
    maxresp = {m: max((s for _, s, _ in c), default=None) for m, c in acc_curves.items()}
    pool = [v for m, v in maxresp.items() if v is not None and m != UNTR]
    target_resp = min(pool) if pool else max(v for v in maxresp.values() if v is not None)

    # ---------- layout: one big mosaic (4 rows x 10 models) + analysis band ----------
    plt.rcParams.update({"xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    n = len(ranked)
    fig = plt.figure(figsize=(14.0, 9.7))
    outer = GridSpec(2, 1, height_ratios=[1.72, 1.0], hspace=0.12,
                     left=0.055, right=0.995, top=0.945, bottom=0.075)
    gM = GridSpecFromSubplotSpec(4, n, subplot_spec=outer[0], hspace=0.035, wspace=0.035)
    gBot = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.31,
                                   width_ratios=[1.00, 1.07, 1.78, 1.14])
    gD = GridSpecFromSubplotSpec(1, 2, subplot_spec=gBot[2], wspace=0.06)   # dual scatter

    ROW_LABELS = [
        "Adversarial attack",
        "Perturbation",
        "Input gradient",
        "Accentuation",
    ]

    def cell(row, col):
        ax = fig.add_subplot(gM[row, col]); ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row], fontsize=11.5, labelpad=6)
        return ax

    for j, m in enumerate(ranked):
        # row 0: adversarial image + model-name column header
        a0 = cell(0, j)
        a0.set_title(SHORT.get(m, m), fontsize=11.5, fontweight="bold", color=get_model_color(m), pad=7)
        # row 1: perturbation
        a1 = cell(1, j)
        # row 2: input-gradient saliency
        a2 = cell(2, j)
        # row 3: accentuation toward the site
        a3 = cell(3, j)

        if m in gal:
            r = gal[m]
            a0.imshow(np.clip(np.transpose(r["adv"], (1, 2, 0)), 0, 1))
            pert = r["adv"] - r["clean"]
            a1.imshow(np.transpose(np.clip(0.5 + 0.5 * pert / (np.abs(pert).max() + 1e-8), 0, 1), (1, 2, 0)))
        else:
            a0.text(0.5, 0.5, "(pending\nadv attack)", ha="center", va="center", fontsize=6.5,
                    color="0.6", transform=a0.transAxes)

        if m in grad_maps:
            a2.imshow(F7.grad_to_saliency(grad_maps[m][SEED_IDX]), cmap="magma", vmin=0, vmax=0.65)

        stim, score = F7.load_accent_at(acc_curves[m], target_resp)
        if stim is not None:
            a3.imshow(stim)

    # ===== bottom analysis band =====
    yr = (merged[OUTC].min() - 0.05, merged[OUTC].max() + 0.05)
    # combined predictor table (sens + flat + outcome on the same site-models) for panel e
    comb = (auc[["model", "monkey", "channel", "auc", OUTC]].rename(columns={"auc": "sens", OUTC: "y"})
            .merge(merged.rename(columns={"unit": "channel"})[["model", "monkey", "channel", GCOL]]
                   .rename(columns={GCOL: "flat"}), on=["model", "monkey", "channel"], how="inner"))
    cc = pair_stats(comb, "y")

    # (b) adversarial sensitivity vs perturbation strength
    axb = fig.add_subplot(gBot[0])
    for m in U.MODEL_ORDER:
        pmc = curves[curves.model == m]
        if pmc.empty: continue
        c = get_model_color(m)
        for _, s in pmc.groupby(["monkey", "channel"]):        # one faint line per site (25/model)
            s = s.sort_values("eps_255"); axb.plot(s.eps_255, s.nswing, "-", lw=0.3, color=c, alpha=0.22, zorder=1)
        mn = pmc.groupby("eps_255")["nswing"].mean().sort_index()
        axb.plot(mn.index, mn.values, "-", lw=1.9, color=c, label=SHORT.get(m, m), zorder=3)
    axb.set_xscale("log", base=2); axb.set_ylim(0, 6.2); axb.set_xlim(2 ** -3.6, 2 ** 6.6)
    axb.set_xlabel("Perturbation strength  ε  (/255)", fontsize=10)
    axb.set_ylabel("Adversarial sensitivity\n" r"($\Delta$ resp., mult. of range)", fontsize=9.5)
    axb.set_title("Adversarial sensitivity\nvs perturbation strength", fontsize=11.5, fontweight="bold")
    lg = axb.legend(fontsize=7.5, ncol=1, frameon=True, loc="upper left", handlelength=1.2,
                    labelspacing=0.26, borderpad=0.4, handletextpad=0.5)
    lg.get_frame().set(facecolor="white", edgecolor="#cccccc", lw=0.6, alpha=0.9)

    # (c) gradient frequency spectra
    axc = fig.add_subplot(gBot[1])
    for m in ranked:
        c = get_model_color(m); is_rob = m in ROBUST_SET
        for k in hoprof:
            if k[2] == m:
                axc.plot(x, F7.normalize_row(hoprof[k])[FMIN:FMAX], color=c, lw=0.4, alpha=0.16, zorder=1)
        axc.plot(x, per_model_prof[m][FMIN:FMAX], color=c, lw=2.6 if is_rob else 1.7,
                 alpha=1.0 if is_rob else 0.9, zorder=9 if is_rob else 5, solid_capstyle="round")
    axc.set_xscale("log"); axc.set_yscale("log")
    axc.set_xlabel("Spatial frequency (cyc/img)", fontsize=10)
    axc.set_ylabel("Normalized radial power", fontsize=9.5)
    axc.set_title("Encoding input gradient\nfrequency spectra", fontsize=11.5, fontweight="bold")
    axc.set_ylim(1e-5, axc.get_ylim()[1])       # floor the view so the flatness inset sits in empty space
    _flatness_inset_nonat(axc, list(per_model_gmetric.items()), loc="bl", metric_label=GSHORT)
    # equalize the plotted x-extent of B and C (B is the narrower one -> grow it; C + its inset stay put)
    pb, pc = axb.get_position(), axc.get_position()
    if pb.width < pc.width:
        axb.set_position([pb.x0, pb.y0, pc.width, pb.height])

    # (d) dual scatter: adversarial sensitivity | gradient flatness, both vs control (all 10)
    axd_l = fig.add_subplot(gD[0]); axd_r = fig.add_subplot(gD[1])
    apad = 0.06 * (auc["auc"].max() - auc["auc"].min())
    _scatter(axd_l, auc, "auc", OUTC, (auc["auc"].min() - apad, auc["auc"].max() + apad),
             yr, "Adversarial sensitivity (AUC)", show_y=True)
    fpad = 0.06 * (merged[GCOL].max() - merged[GCOL].min())
    _scatter(axd_r, merged, GCOL, OUTC,
             (merged[GCOL].min() - fpad, merged[GCOL].max() + fpad),
             yr, GLABEL, show_y=False)

    # (e) summary bars: sensitivity vs flatness prediction of control across model subsets
    axe = fig.add_subplot(gBot[3])
    _summary_bars(axe, cc, CEIL)
    axe.set_title("Correlation across\nmodel subsets", fontsize=11.5, fontweight="bold")

    # ---------- titles + panel letters ----------
    fig.text((0.055 + 0.995) / 2, 0.988,
             "Encoding-axis adversarial sensitivity and input gradients",
             ha="center", va="center", fontsize=12, fontweight="bold")
    # a shared over-title for the dual-scatter panel d
    pdl = axd_l.get_position(); pdr = axd_r.get_position()
    dcx = (pdl.x0 + pdr.x1) / 2
    fig.text(dcx, pdl.y1 + 0.026, "Correlation with neural control",
             ha="center", va="center", fontsize=11.5, fontweight="bold")
    fig.text(dcx, pdl.y1 + 0.010, "N = 10 models    ·    n = 250 site-model axes",
             ha="center", va="center", fontsize=8.5, color="0.35")
    # panel letters - "a" aligned in the same left column as "b"
    fig.text(axb.get_position().x0 - 0.030, 0.985, "a", fontsize=20, fontweight="bold", va="top")
    for ax, lab in [(axb, "b"), (axc, "c"), (axd_l, "d"), (axe, "e")]:
        p = ax.get_position(); fig.text(p.x0 - 0.030, p.y1 + 0.010, lab, fontsize=20, fontweight="bold", va="bottom")
    return fig


def main(out_dir, site="paul8", seed_img=3, outcome="slope", resid="9trained", gradmetric="pr"):
    _configure(site, seed_img, outcome, resid, gradmetric)
    fig = build()
    # default render (paul8/slope/resid9/pr) -> plain predictors.png; variants keep a descriptive tag
    default = (SITE == "paul8" and OUTCOME == "slope" and RESID == "9trained" and GRADMETRIC == "pr")
    tag = "" if default else f"_{SITE}_{OUTCOME}{ {'9trained': '_resid9', 'none': '_noresid'}.get(RESID, '') }_{GRADMETRIC}"
    path = save_fig(fig, os.path.join(out_dir, f"{MAIN_FIGURES[5]}{tag}"))
    plt.close(fig)
    print(f"WROTE {path}  (site={SITE}, seed={SEED_IDX}, outcome={OUTCOME}, resid={RESID}, gradmetric={GRADMETRIC})")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", dest="out_dir", default=str(paths.output_dir() / "figures"))
    ap.add_argument("--site", default="paul8", choices=sorted(F7._SITES), help="example site for the mosaic")
    ap.add_argument("--seed-img", dest="seed_img", type=int, default=3, help="seed image index (3 = cat)")
    ap.add_argument("--outcome", default="slope", choices=["slope", "r"], help="control outcome for panels d/e")
    ap.add_argument("--resid", default="9trained", choices=["10", "9trained", "none"],
                    help="per-site residualization reference (none = absolute outcome)")
    ap.add_argument("--gradmetric", default="pr", choices=["flatness", "cv", "pr"],
                    help="gradient-spectrum predictor")
    args = ap.parse_args()
    main(**vars(args))
