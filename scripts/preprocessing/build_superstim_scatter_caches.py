#!/usr/bin/env python3
"""Build the two super-stimulus scatter caches read by the supplementary figures
`superstim_scatter_venus` and `superstim_scatter_all`:

    <out>/sup_superstim_scatter_venus.pkl   dict(cloud=DataFrame[pred, meas, sem],
                                                 acc=DataFrame[monkey, unit, model, is_super,
                                                 is_supersuppress, pred, meas, sem, n_reps, path])
    <out>/sup_superstim_scatter_all.pkl     dict(cloud_pool=DataFrame[pred, meas, sem], acc=...)

Both reproduce take7 build_superstimuli_scatter.py on the preprocessed loader caches
(brain_<mk>.pkl, encoding_<mk>.pkl, predictions_<mk>.pkl); no raw HDF5 is read.

  * calibration cloud pred/meas = model encoding prediction vs measured encoding-phase
    response (the grey cloud), cloud SEM = per-stimulus SEM of the encoding trial responses.
  * q99/q01 = 99th/1st percentile of the FULL encoding-phase measured distribution per unit.
  * accentuation pred = firing-floor-clamped model score (== the filename score); meas / SEM /
    n_reps from the control-session repavg resp_z + per-stimulus trial SEM/reps; a stimulus is
    kept only if |target - score| <= 0.1 (take7 THRESHOLD).
  * venus: no level exclusions (take7 excluded only red/paul low levels). all-sites: the take7
    EXCLUDED_LEVELS (red/paul strongest-drive levels) are applied.
  * `path` is the accentuated image location under STIMULI_CONTROL_PATH at build time; the
    figure scripts re-root it under the current STIMULI_CONTROL_PATH when they read the cache.

    python scripts/preprocessing/build_superstim_scatter_caches.py [--out DIR]
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from pnc import paths
from pnc.preproc import loader as L
from pnc.utils import (MODEL_ORDER, parse_stimulus_name, STIMULI_CONTROL_PATH,
                       ACCENT_DATE_PREFIXES, MONKEY_DIRS)

THRESHOLD = 0.1
MIN_REPS = 3


def _control_image_path(monkey, model, filename):
    folder = f"{ACCENT_DATE_PREFIXES[monkey]}_{MONKEY_DIRS[monkey]}_{model}_accentuation"
    return os.path.join(STIMULI_CONTROL_PATH, folder, filename)


# =============================================================================
# venus (Monkey V) cache — take9 sup_superstim_scatter_venus.build_cache
# =============================================================================
VENUS = 'venus'


def _parse(name):
    """Parse an accentuated stimulus filename -> {model, unit, seed, target, score}."""
    if 'score' not in name:
        return None
    try:
        model = name.split('_RidgeCV')[0]
        parts = name.split('_RidgeCV')[1].split('_')
        return dict(model=model, unit=int(parts[2]), seed=int(parts[4]),
                    target=float(parts[6]), score=float(parts[8].replace('.png', '')))
    except (IndexError, ValueError):
        return None


def build_venus(cache):
    """Reproduce take7 build_superstimuli_scatter.py for venus on the loader."""
    MONKEY = VENUS
    cfg = L.config(); models = list(cfg['models'])
    b = L.load_brain(MONKEY)
    e = L.load_encoding(MONKEY)
    P = L.load_predictions(MONKEY)
    units = [int(u) for u in b['units']]

    # trial-level indices for SEM
    cal = b['calibration']
    cal_tstim = np.asarray(cal['trial_stim']); cal_tz = np.asarray(cal['trial_z'])
    cal_idx = defaultdict(list)
    for i, n in enumerate(cal_tstim):
        cal_idx[str(n)].append(i)
    ctrl = b['control']
    c_tstim = np.asarray(ctrl['trial_stim']); c_tz = np.asarray(ctrl['trial_z'])
    ctrl_idx = defaultdict(list)
    for i, n in enumerate(c_tstim):
        ctrl_idx[str(n)].append(i)

    cal_names = [str(n) for n in cal['stim']]
    cal_meas = np.asarray(cal['resp_z'], float)               # (n_cal, n_unit)
    c_names = [str(n) for n in ctrl['stim']]
    c_meas = np.asarray(ctrl['resp_z'], float)                # (n_ctrl, n_unit)
    c_kind = [str(k) for k in ctrl['kind']]

    def sem_for(idx_map, trials, name, ui):
        ii = idx_map.get(name, [])
        if len(ii) > 1:
            v = trials[ii, ui]
            return float(np.std(v, ddof=1) / np.sqrt(len(v))), len(ii)
        return 0.0, len(ii)

    cloud = []                    # (pred, meas, sem) over calibration set x models
    acc = []
    e_units = list(e['units']); e_models = list(e['models'])
    for ui, unit in enumerate(units):
        col_u = list(b['units']).index(unit)
        q99 = float(np.percentile(cal_meas[:, col_u], 99))
        q01 = float(np.percentile(cal_meas[:, col_u], 1))

        # ---- grey calibration cloud: model encoding pred vs measured enc response ----
        eu = e_units.index(unit)
        cal_meas_by_name = dict(zip(cal_names, cal_meas[:, col_u]))
        cal_sem_by_name = {n: sem_for(cal_idx, cal_tz, n, col_u)[0] for n in cal_names}
        for m in models:
            mi = e_models.index(m)
            epred = dict(zip([str(s) for s in e['stim']], e['pred'][eu, mi]))
            for n, pz in epred.items():
                if n in cal_meas_by_name:
                    cloud.append((float(pz), float(cal_meas_by_name[n]), float(cal_sem_by_name[n])))

        # ---- accentuation overlay: floor-clamped score vs measured control response ----
        floor = L.firing_floor(MONKEY, unit)
        mi_p = {m: list(P['pred_models']).index(m) for m in models}   # (unused; score is per-stim)
        # index accentuated control stimuli for this unit, keyed by name
        c_meas_by_name = {c_names[i]: float(c_meas[i, col_u])
                          for i in range(len(c_names)) if c_kind[i] == 'accentuated'}
        # per-model target-bucketing then |target-score|<=0.1 filter (take7)
        by_model = defaultdict(list)
        for i, name in enumerate(c_names):
            if c_kind[i] != 'accentuated':
                continue
            pr = _parse(name)
            if pr is None or pr['unit'] != unit:
                continue
            if abs(pr['target'] - pr['score']) > THRESHOLD:
                continue
            by_model[pr['model']].append((name, pr))
        for model, items in by_model.items():
            for name, pr in items:
                if name not in c_meas_by_name:
                    continue
                mz = c_meas_by_name[name]
                sem, nrep = sem_for(ctrl_idx, c_tz, name, col_u)
                acc.append(dict(
                    monkey=MONKEY, unit=unit, model=model,
                    is_super=bool(mz > q99), is_supersuppress=bool(mz < q01),
                    pred=float(max(pr['score'], floor)), meas=mz, sem=float(sem),
                    n_reps=int(nrep), path=_control_image_path(MONKEY, model, name)))

    cloud = pd.DataFrame(cloud, columns=['pred', 'meas', 'sem'])
    rng = np.random.default_rng(0)
    if len(cloud) > 3000:
        cloud = cloud.iloc[rng.choice(len(cloud), 3000, replace=False)].reset_index(drop=True)
    acc = pd.DataFrame(acc)
    d = dict(cloud=cloud, acc=acc)
    with open(cache, 'wb') as f:
        pickle.dump(d, f)
    a3 = acc[acc['n_reps'] >= MIN_REPS]
    print(f'built cache: {len(acc)} acc ({len(a3)} with >={MIN_REPS} reps), cloud n={len(cloud)}')
    return d


# =============================================================================
# all-sites cache — take9 sup_superstim_scatter_all.build
# =============================================================================
CLOUD_POOL = 5000
EXCLUDED_LEVELS = {
    ('red', 0): {0}, ('red', 2): {0}, ('red', 9): {0},
    ('red', 15): {0}, ('red', 19): {0},
    ('paul', 0): {0}, ('paul', 8): {0}, ('paul', 24): {0, 1},
    ('paul', 40): {0}, ('paul', 47): {0},
}


def _trial_index(trial_stim):
    idx = {}
    for i, n in enumerate(trial_stim):
        idx.setdefault(str(n), []).append(i)
    return idx


def _sem_reps(idx, trial_z, ui):
    """{stim_name: (sem, n_reps)} across trials for one unit."""
    out = {}
    for n, ii in idx.items():
        if len(ii) > 1:
            v = trial_z[ii, ui]
            out[n] = (float(v.std(ddof=1) / np.sqrt(len(ii))), len(ii))
        else:
            out[n] = (0.0, len(ii))
    return out


def build_all(cache):
    rng = np.random.default_rng(0)
    all_cloud = []
    acc = []
    for monkey in L.monkeys():
        b = L.load_brain(monkey)
        units = [int(u) for u in b['units']]
        cal = b['calibration']; c = b['control']
        cal_resp = np.array(cal['resp_z'])
        cal_trial_idx = _trial_index(cal['trial_stim']); cal_trial_z = np.array(cal['trial_z'])
        ctrl_stim = [str(x) for x in c['stim']]
        ctrl_resp = np.array(c['resp_z'])
        ctrl_trial_idx = _trial_index(c['trial_stim']); ctrl_trial_z = np.array(c['trial_z'])

        for uii, unit in enumerate(units):
            q99 = np.percentile(cal_resp[:, uii], 99)
            q01 = np.percentile(cal_resp[:, uii], 1)
            fl = L.firing_floor(monkey, unit)

            # grey calibration (encoding-generalization) cloud: model prediction vs
            # measured encoding-phase response, with SEM of the measured response.
            enc_sem = _sem_reps(cal_trial_idx, cal_trial_z, uii)
            for model in MODEL_ORDER:
                ep, em = L.encoding_cloud(monkey, unit, model, 'all')
                # cloud SEM aligned by stimulus name (encoding_cloud drops names, so
                # recompute against the same stim order used by the loader)
                sm = np.array([enc_sem.get(str(n), (0.0, 0))[0] for n in cal['stim']], float)
                # encoding_cloud keeps only stims present in both preds+resp (== all here)
                for pz, mz, sz in zip(ep, em, sm):
                    all_cloud.append((float(pz), float(mz), float(sz)))

            # accentuated super-stimuli (parsed from the control stimulus names)
            ctrl_sem = _sem_reps(ctrl_trial_idx, ctrl_trial_z, uii)
            excl = EXCLUDED_LEVELS.get((monkey, unit), set())
            by_model = {}
            for i, name in enumerate(ctrl_stim):
                pr = parse_stimulus_name(name)
                if pr is None or pr['unit'] != unit:
                    continue
                if abs(pr['target'] - pr['score']) > THRESHOLD:
                    continue
                by_model.setdefault(pr['model'], []).append((i, pr, name))
            for model, items in by_model.items():
                t2i = {t: j for j, t in enumerate(sorted({p['target'] for _, p, _ in items}))}
                for i, pr, name in items:
                    if t2i[pr['target']] in excl:
                        continue
                    mz = float(ctrl_resp[i, uii])
                    sem, nrep = ctrl_sem.get(name, (0.0, 0))
                    acc.append(dict(
                        monkey=monkey, unit=unit, model=model, is_super=bool(mz > q99),
                        is_supersuppress=bool(mz < q01),
                        pred=float(max(pr['score'], fl)), meas=mz, sem=float(sem),
                        n_reps=int(nrep), path=_control_image_path(monkey, model, name)))
        print(f'  {monkey}: done')

    cloud = pd.DataFrame(all_cloud, columns=['pred', 'meas', 'sem'])
    if len(cloud) > CLOUD_POOL:
        cloud = cloud.iloc[rng.choice(len(cloud), CLOUD_POOL, replace=False)]
    cloud = cloud.reset_index(drop=True)
    acc = pd.DataFrame(acc)
    d = dict(cloud_pool=cloud, acc=acc)
    with open(cache, 'wb') as f:
        pickle.dump(d, f)
    print(f'Cached -> {cache}  (acc={len(acc)}, cloud={len(cloud)})')
    return d


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    build_venus(os.path.join(out_dir, 'sup_superstim_scatter_venus.pkl'))
    build_all(os.path.join(out_dir, 'sup_superstim_scatter_all.pkl'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.preprocessed_data()),
                    help='directory to write the two .pkl caches (default: PNC_PREPROCESSED_DATA)')
    args = ap.parse_args()
    main(**vars(args))
