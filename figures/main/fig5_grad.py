#!/usr/bin/env python3
"""Gradient-geometry data helpers for figure 5.

Per-(monkey,unit,model) gradient spectral descriptors merged with control outcomes
(load_data), plus the image-side ingredients for the mosaic: input-gradient saliency
maps, seed images, and accentuation sweeps.

Reads preprocessed_data via pnc.preproc.loader (gradient_freq.pkl, grad_maps/,
control_table) and raw stimuli from DATA_ROOT. The example site (paul8 default | red9)
and seed image (default idx 3 = cat) are resolved per call by site_config().
"""
import os, re
import numpy as np, pandas as pd
from PIL import Image
from scipy import stats

from pnc.utils import (STIMULI_PATH, STIMULI_CONTROL_PATH, MONKEY_DIRS,
                       ACCENT_DATE_PREFIXES, MODEL_SHORT_NAMES, ROBUST_MODELS)
from pnc.preproc import loader as L

SHORT = dict(MODEL_SHORT_NAMES); SHORT["AlexNet_training_seed_01"] = "Untrained"
ROBUST = ["resnet50_robust", "clipag_vitb32"]; UNTR = "AlexNet_training_seed_01"
EXTREMES = ROBUST + [UNTR]; ROBUST_SET = set(ROBUST_MODELS)
FMIN, FMAX = 1, 112
C_NATURAL = "#222222"

# example site: accentuation target (mosaic bottom row) + gradient-map site.
_SITES = {"red9": ("red", 9), "paul8": ("paul", 8)}
MONKEY_DISP = {"red": "R", "paul": "P", "venus": "V", "leap": "L", "three0": "T"}
SEED_NAMES = ["shared0575_nsd43157.png", "shared0850_nsd61798.png", "shared0968_nsd70194.png",
              "shared0241_nsd20065.png", "shared0160_nsd13231.png", "shared0070_nsd07008.png",
              "shared0055_nsd05879.png", "shared0668_nsd48623.png", "shared0488_nsd36979.png",
              "shared0940_nsd68312.png"]


def site_config(site="paul8", seed_img=3):
    """(acc_monkey, acc_unit, seed_idx, seed_name) for the mosaic example site + seed
    (was read from the SITE / SEED_IMG environment variables at import time)."""
    acc_monkey, acc_unit = _SITES[site]
    seed_idx = int(seed_img)                                  # 3 = shared0241 (cat)
    return acc_monkey, acc_unit, seed_idx, SEED_NAMES[seed_idx]


# ---- spectral helpers -------------------------------------------------------
def normalize_row(p):
    p = np.asarray(p, float); s = p[FMIN:FMAX].sum()
    return p / s if s > 0 else p


def spectral_flatness(p):
    P = np.clip(np.asarray(p, float)[FMIN:FMAX], 1e-30, None)
    return float(np.exp(np.mean(np.log(P))) / P.mean())


def spectral_cv(p):                                          # coefficient of variation of the radial power spectrum
    P = np.asarray(p, float)[FMIN:FMAX]
    return float(np.std(P) / np.mean(P))


def fit_log_log_slope(p):
    freqs = np.arange(FMIN, FMAX); vals = np.asarray(p, float)[FMIN:FMAX]; ok = vals > 0
    if ok.sum() < 5:
        return np.nan
    slope, *_ = stats.linregress(np.log10(freqs[ok]), np.log10(vals[ok]))
    return -float(slope)


def grad_to_saliency(grad_3hw, pct=99.0):
    mag = np.sqrt((grad_3hw.astype(np.float64) ** 2).sum(axis=0))
    hi = np.percentile(mag, pct)
    return np.clip(mag / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(mag)


# ---- data -------------------------------------------------------------------
def load_data():
    """Per-(monkey,unit,model): gradient flatness + spectral slope + control outcomes."""
    g = L.load_gradient_freq()
    rows = [dict(monkey=r["monkey"], unit=int(r["unit"]), model=r["model"],
                 flatness=spectral_flatness(r["profile_mean"]),
                 slope=fit_log_log_slope(r["profile_mean"])) for r in g["gradients"]]
    grad_df = pd.DataFrame(rows)
    ct = L.control_table()[["monkey", "unit", "model", "control_r", "control_slope"]].copy()
    ct["unit"] = ct["unit"].astype(int)
    merged = grad_df.merge(ct, on=["monkey", "unit", "model"], how="left")
    return g, merged


def load_seed_image(name, size=224):
    p = os.path.join(STIMULI_PATH, name)
    if not os.path.exists(p):
        hits = [os.path.join(r, name) for r, _, fs in sorted(os.walk(STIMULI_PATH)) if name in fs]
        p = hits[0] if hits else p
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def accent_curve(model, unit, img_idx, monkey):
    """Sorted [(level, achieved_score, path), ...] for one model+site+seed accentuation sweep."""
    d = os.path.join(STIMULI_CONTROL_PATH, f"{ACCENT_DATE_PREFIXES[monkey]}_{MONKEY_DIRS[monkey]}_{model}_accentuation")
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


def load_grad_maps_safe(monkey, unit):
    try:
        return L.load_grad_maps(monkey, unit)
    except Exception:
        return {}
