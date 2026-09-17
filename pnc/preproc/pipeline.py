"""Core single-trial transforms for the settled control-response pipeline.

Order: peak-window average -> outlier rejection -> per-session standardization ->
trial average. Every step is a pure function of arrays + the config knobs passed from
run_preproc.py; nothing here reads files or holds state.
"""
from collections import defaultdict

import numpy as np


def peak_window_average(resp_temporal, temporal_windows, peak_window):
    """(bins, T, U) temporal responses -> (T, U) mean over the peak response window."""
    a, b = peak_window
    pb = np.where((temporal_windows >= a) & (temporal_windows < b))[0]
    return resp_temporal[pb.min():pb.max() + 1].mean(0)


def outlier_keep_mask(raw, days, robust_z=15.0, min_trials=10, drop_if_any_channel=True):
    """Trial keep-mask. Drop a trial whose response is a giant outlier on any channel,
    robust-z = |x - median| / (1.4826*MAD) > robust_z, computed per channel per session."""
    keep = np.ones(raw.shape[0], bool)
    for ui in range(raw.shape[1]):
        for d in dict.fromkeys(days.tolist()):
            idx = np.where(days == d)[0]
            x = raw[idx, ui]; ok = np.isfinite(x)
            if ok.sum() < min_trials:
                continue
            med = np.median(x[ok]); mad = 1.4826 * np.median(np.abs(x[ok] - med))
            if mad <= 0:
                continue
            hit = idx[np.abs(x - med) / mad > robust_z]
            if drop_if_any_channel:
                keep[hit] = False
    return keep


def standardize(raw, days, anchor, method='anchorDay', anchor_min_trials=5,
                fallback='allday', calrange_pct=(1, 99), tsn=None, enc_z=None,
                train_mask=None, score_of=None):
    """Per-session single-trial z (T, U).

    method='anchorDay'  standardize on the day's calibration-anchor trials (st01 max-N).
    method='st01_train50' standardize on the day's encoding-training-set anchors.
    method='allTrial'   standardize on every trial that day (legacy).
    Fallback fires when the day has < anchor_min_trials anchors:
      'allday'   -> all that day's trials;
      'calrange' -> anchors + accentuated trials whose predicted score is inside the
                    calibration [Plo, Phi] response band (single-trial pool)."""
    z = np.empty_like(raw)
    src_mask = anchor if method != 'st01_train50' else (anchor & train_mask)
    if fallback == 'calrange':
        cal = np.concatenate([np.asarray(v).ravel() for v in enc_z.values()])
        lo, hi = np.percentile(cal, calrange_pct)
        inc = anchor.copy()
        for i in np.where(~anchor)[0]:
            s = score_of(tsn[i])
            if s is not None and lo <= s <= hi:
                inc[i] = True
    for d in dict.fromkeys(days.tolist()):
        m = days == d
        if method == 'allTrial':
            src = raw[m]
        else:
            am = m & src_mask
            if am.sum() >= anchor_min_trials:
                src = raw[am]
            elif fallback == 'calrange':
                pool = m & inc
                src = raw[pool] if pool.sum() >= 10 else raw[m]
            else:
                src = raw[m]
        mu = src.mean(0); sg = src.std(0).copy(); sg[sg == 0] = 1.0
        z[m] = (raw[m] - mu) / sg
    return z


def trial_average(tsn, z):
    """{stimulus_name: mean z (U,)} and {stimulus_name: n_reps}."""
    idx = defaultdict(list)
    for i, n in enumerate(tsn):
        idx[n].append(i)
    mean = {n: z[ii].mean(0) for n, ii in idx.items()}
    nrep = {n: len(ii) for n, ii in idx.items()}
    return mean, nrep
