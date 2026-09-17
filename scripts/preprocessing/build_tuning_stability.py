#!/usr/bin/env python3
"""Build the tuning-stability cache (preproc_data/tuning_stability.pkl) for the
reliability supp figure.

Two stability measures per site:
  cross-day    within the encoding phase — mean pairwise correlation of a site's
               per-session mean responses (over stimuli shown on every session).
               A within-phase reference: day-to-day tuning reliability.
  cross-phase  encoding vs control — correlation of responses to the natural images
               shown in BOTH phases. Lower-than-cross-day = phase-specific drift.

Also flags whether the control-phase shared responses were repeated (reliable) vs
single-trial (leap/three0 → cross-phase attenuated by noise). Torch-free; fast.
"""
import os
import glob
import pickle

import numpy as np
from collections import Counter
from scipy.stats import pearsonr

from pnc import paths
from pnc.utils import load_encoding_hdf5, load_control_hdf5, MONKEY_UNITS, DATA_ROOT
import h5py

OUT = os.path.join(str(paths.preprocessed_data()), 'tuning_stability.pkl')
ENC_DIR = os.path.join(DATA_ROOT, 'brain_data_encoding')
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
REGION = {'red': 'aIT', 'paul': 'cIT', 'venus': 'V3/V4', 'leap': 'STS', 'three0': 'STS'}


def cross_day(mk, units):
    """Day-to-day tuning stability via balanced session split-halves: partition the
    encoding sessions into two equal-ish halves (each half averaged so both estimates
    are reliable), correlate the halves per unit, over EVERY distinct partition. Mean
    and SEM are taken across partitions — reliable (multi-session) estimates, with a
    spread. (Single-session pairwise correlations attenuate by per-session noise.)"""
    from itertools import combinations
    f = sorted(glob.glob(os.path.join(ENC_DIR, f'*{mk}*vvs-encodingstimuli*.h5')))[0]
    with h5py.File(f, 'r') as h:
        tsn = np.array(h['trials']['stimulus_name'], dtype=str)
        tses = np.array(h['trials']['session_num'], dtype=str)
        tr = np.array(h['trials']['response_peak'], dtype=float)
    sessions = sorted(set(tses)); n = len(sessions); kk = n // 2
    inter = sorted(set.intersection(*[set(tsn[tses == s]) for s in sessions]))
    idx = {st: i for i, st in enumerate(inter)}
    sess_mat = {}                                   # session -> (n_inter, n_units) mean
    for s in sessions:
        m = tses == s
        sums = np.zeros((len(inter), len(units))); cnt = np.zeros(len(inter))
        for st_, row in zip(tsn[m], tr[m][:, units]):
            k = idx.get(st_)
            if k is not None:
                sums[k] += row; cnt[k] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            sess_mat[s] = sums / cnt[:, None]
    combos = list(combinations(range(n), kk))
    if n % 2 == 0:                                  # dedupe symmetric equal-size halves
        combos = [c for c in combos if 0 in c]

    def hmean(sess_list):
        return np.nanmean(np.stack([sess_mat[s] for s in sess_list]), axis=0)
    parts = [(hmean([sessions[i] for i in c]),
              hmean([sessions[i] for i in range(n) if i not in c])) for c in combos]
    mean, sem = {}, {}
    for ui, u in enumerate(units):
        rs = []
        for A, Bm in parts:
            a = A[:, ui]; b = Bm[:, ui]; ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 5 and a[ok].std() > 0 and b[ok].std() > 0:
                rs.append(pearsonr(a[ok], b[ok])[0])
        mean[u] = float(np.mean(rs)) if rs else np.nan
        sem[u] = float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    return mean, sem, len(sessions)


def main():
    sites = []
    for mk in MONKEYS:
        en, er, _, _ = load_encoding_hdf5(mk)
        cn, cr, ctn, _ = load_control_hdf5(mk)
        en = np.asarray(en); cn = np.asarray(cn)
        shared = sorted(set(en) & set(cn))
        ei = {s: i for i, s in enumerate(en)}; ci = {s: i for i, s in enumerate(cn)}
        eidx = [ei[s] for s in shared]; cidx = [ci[s] for s in shared]
        reps = Counter(np.asarray(ctn))
        med_reps = float(np.median([reps.get(s, 0) for s in shared])) if shared else 0.0
        cd, cdsem, nsess = cross_day(mk, list(MONKEY_UNITS[mk]))
        for u in MONKEY_UNITS[mk]:
            a = er[eidx, u]; b = cr[cidx, u]; ok = np.isfinite(a) & np.isfinite(b)
            cp = (pearsonr(a[ok], b[ok])[0] if ok.sum() > 5 and a[ok].std() > 0
                  and b[ok].std() > 0 else np.nan)
            sites.append(dict(monkey=mk, unit=int(u), region=REGION[mk],
                              cross_phase=float(cp), cross_day=float(cd[u]),
                              cross_day_sem=float(cdsem[u]), n_shared=int(ok.sum()),
                              ctrl_reps=med_reps, n_sessions=nsess))
        mm = [s for s in sites if s['monkey'] == mk]
        print(f'  {mk:7s} sess={nsess} shared={len(shared):3d} ctrl_reps={med_reps:.0f} '
              f'cross-day={np.nanmedian([s["cross_day"] for s in mm]):.2f} '
              f'cross-phase={np.nanmedian([s["cross_phase"] for s in mm]):.2f}')
    pickle.dump(dict(sites=sites), open(OUT, 'wb'))
    print(f'Saved -> {OUT}  ({len(sites)} sites)')


if __name__ == '__main__':
    main()
