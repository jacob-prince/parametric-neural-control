"""NSD noise ceilings on the standardized control responses.

The trial-noise term is borrowed from the anchors' SAME-DAY repeats (pure trial noise),
not the accentuated stimuli's own repeats (which span days and would count day-drift as
noise, deflating the ceiling). One nc_r per channel per stimulus group.
"""
from collections import defaultdict

import numpy as np


def anchor_trial_noise(zc, tsn, days, anchor_set, min_groups=5):
    """Pooled same-day repeat variance of the anchors = pure trial noise (per channel)."""
    grp = defaultdict(list)
    for i, s in enumerate(tsn):
        if s in anchor_set and np.isfinite(zc[i]):
            grp[(s, days[i])].append(zc[i])
    wd = [np.var(v, ddof=1) for v in grp.values() if len(v) >= 2]
    return float(np.mean(wd)) if len(wd) >= min_groups else np.nan


def nsd_ceiling(zc, tsn, stimset, noisevar, min_stim=8):
    """NSD nc_r over stimset stimuli with >=2 repeats, given a trial-noise floor.
    Returns (nc_r, signal_var, noise_var, n_stim)."""
    by_stim = defaultdict(list)
    for i, s in enumerate(tsn):
        if s in stimset and np.isfinite(zc[i]):
            by_stim[s].append(zc[i])
    means, reps = [], []
    for vals in by_stim.values():
        if len(vals) >= 2:
            means.append(np.mean(vals)); reps.append(len(vals))
    if len(means) < min_stim or not (np.isfinite(noisevar) and noisevar > 0):
        return np.nan, np.nan, float(noisevar), len(means)
    datavar = np.var(means, ddof=1); inv_n = np.mean(1.0 / np.array(reps, float))
    if not (datavar > 0):
        return np.nan, 0.0, float(noisevar), len(means)
    sig = max(datavar - noisevar * inv_n, 0.0)
    ncsnr = np.sqrt(sig) / np.sqrt(noisevar)
    nc_r = np.sqrt(ncsnr ** 2 / (ncsnr ** 2 + inv_n))
    return float(nc_r), float(sig), float(noisevar), len(means)
