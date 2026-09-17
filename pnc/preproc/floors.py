"""Firing-floor (z at 0 spk/s) per monkey/channel, computed from the calibration sessions.

Per calibration (encoding) session the z at 0 spk/s is -mu/sigma; average that across
sessions for the aggregate 0-spk/s level. Then lower it to the minimum z ever observed in
the single-trial calibration responses, so the floor never sits above a level the neuron
actually reached (physically firing >= 0, so the observed z-min tracks the true 0-spk/s of
the lowest session). Any predicted accentuation response below this floor is set to it.

    floor = min( mean_sessions(-mu_s/sigma_s),  min single-trial calibration z )
"""
import os
import glob
import pickle

import numpy as np

from pnc.utils import ENCODING_HDF5_DIR, MONKEY_RESPONSE_WINDOWS


def _sessdata_paths(monkey):
    rw = MONKEY_RESPONSE_WINDOWS[monkey]
    p = sorted(glob.glob(os.path.join(ENCODING_HDF5_DIR, f'{monkey}_*_hp0_bs0_zs1_{rw}_sessdata.pkl')))
    if not p:
        raise FileNotFoundError(f'no encoding sessdata PKLs for {monkey}')
    return p


def compute(monkey, units):
    """{unit: dict(floor, floor_avg, obs_min, n_sessions)} for the given channels.
    peak_resp is already per-session z-scored; mu/sigma give the raw-spk/s conversion."""
    per_floor = {u: [] for u in units}
    per_min = {u: [] for u in units}
    for p in _sessdata_paths(monkey):
        d = pickle.load(open(p, 'rb')); nm = d['neuron_meta']
        mu = np.asarray(nm['mu'], float); sigma = np.asarray(nm['sigma'], float)
        z = np.asarray(d['peak_resp'], float)          # (n_chan, n_trials), z per session
        for u in units:
            per_floor[u].append(-mu[u] / sigma[u])
            per_min[u].append(float(np.nanmin(z[u])))
    out = {}
    for u in units:
        favg = float(np.mean(per_floor[u])); fobs = float(np.min(per_min[u]))
        out[u] = dict(floor=min(favg, fobs), floor_avg=favg, obs_min=fobs,
                      n_sessions=len(per_floor[u]))
    return out
