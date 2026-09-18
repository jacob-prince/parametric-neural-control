"""take9 preprocessing entrypoint — the single place every hyperparameter is set.

Runs the settled pipeline once and writes the canonical caches every downstream figure
reads (via loader.py). NOTHING downstream reads repavg or raw HDF5 again.

  brain_<monkey>.pkl   control single-trial (cleaned+standardized) & trial-avg responses,
                       calibration (encoding-phase measured) responses, per (unit,model)
                       noise ceilings, per-unit region/firing-floor/reliability.
  encoding_<monkey>.pkl per (unit,model) prediction over every stimulus + linear readout.
  stimuli.pkl          calibration / accentuated / controversial stimulus tables.
  brain_controversial.pkl  red RN50-vs-robust experiment (pre-averaged .nc source).
  MANIFEST.json        frozen config + git hash + timestamp + input paths.

Run in the torch env (encoding pickles):
  KMP_DUPLICATE_LIB_OK=TRUE .../envs/monkey/bin/python run_preproc.py
"""
import os
import re
import glob
import json
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np
import h5py

from pnc import paths
from pnc.utils import (CONTROL_HDF5_PATH, MONKEY_UNITS, MODEL_ORDER, ROBUST_MODELS,
                       DATA_ROOT, MONKEY_DIRS, load_encoding_hdf5, load_encoding_mu_sigma)
from pnc.preproc import pipeline as P
from pnc.preproc import ceilings as C
from pnc.preproc import stimuli as S
from pnc.preproc import encoding as E
from pnc.preproc import floors as FL
from pnc.preproc import exclusions as X

# ============================== CONFIG ======================================
# Every knob for the whole pipeline. Edit here, re-run, everything downstream follows.
CONFIG = dict(
    monkeys=['red', 'paul', 'venus', 'leap', 'three0'],
    models=list(MODEL_ORDER),
    robust_models=sorted(ROBUST_MODELS),
    # --- outlier rejection (control phase only; encoding phase left untouched) ---
    outlier_robust_z=15.0,          # drop trial if |x-median|/(1.4826*MAD) exceeds this
    outlier_min_trials=10,          # need >= this many trials in a session to test
    outlier_drop_if_any_channel=True,
    # --- standardization ---
    standardize='anchorDay',        # 'anchorDay'(max-N) | 'st01_train50' | 'allTrial'
    anchor_min_trials=5,            # per session; below -> fallback
    fallback='allday',              # 'allday' | 'calrange'
    calrange_pct=(1, 99),           # only used when fallback == 'calrange'
    # --- noise ceiling (NSD, anchor-borrowed same-day trial noise) ---
    ceiling_min_stim=8,
    ceiling_min_anchor_groups=5,
    # --- firing-floor clamp is a downstream prediction op; recorded for provenance ---
    clamp_predictions_to_floor=True,
    experiments=['main', 'controversial'],
)
CACHE = str(paths.preprocessed_data()); os.makedirs(CACHE, exist_ok=True)
ENC_DIR = os.path.join(DATA_ROOT, 'brain_data_encoding')
CONTROV_DIR = os.path.join(DATA_ROOT, 'encoding_model_outputs', 'controversial',
                           'red_20241212-20241220')
CONTROV_PRED_VARS = ['RN50_Lasso', 'RN50_Ridge', 'RN50rbst_Lasso', 'RN50rbst_Ridge']
CONTROV_UNITS = [1, 9, 15, 16, 25, 37, 44]                 # red aIT target units
CONTROV_SESSION = 'red_20250123-20250126'                   # session used by the analysis
PRED_ROOT = os.path.join(DATA_ROOT, 'model_predictions')


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def region_map(monkey, units):
    f = sorted(glob.glob(os.path.join(ENC_DIR, f'{monkey}*_vvs-encodingstimuli_*.h5')))[0]
    with h5py.File(f, 'r') as h:
        nm = h['neuron_metadata']; nchan = h['repavg']['response_peak'].shape[1]
        ba = nm['brain_area']
        arr = (next(np.array(ba[k]) for k in ba.keys() if len(ba[k]) == nchan)
               if isinstance(ba, h5py.Group) else np.array(ba))
        areas = np.array([_dec(x).replace('l_', '') for x in arr])
    return {u: str(areas[u]) for u in units}


def _score_of(sn):
    pr = S.parse_accentuated(sn)
    return pr['score'] if pr else None


# ------------------------------ brain (main) --------------------------------
def build_brain_main(monkey, cfg):
    units = list(MONKEY_UNITS[monkey])
    mu_all, sigma_all = load_encoding_mu_sigma(monkey)     # z<->spk/s conversion per neuron
    en, er, etsn, etr = load_encoding_hdf5(monkey); en = np.asarray(en)
    enc_z = {n: er[i, units] for i, n in enumerate(en)}; enc_set = set(en.tolist())
    with h5py.File(CONTROL_HDF5_PATH, 'r') as f:
        key = [k for k in f.keys() if k.startswith(monkey)][0]
        tw = f[f'{key}/neuron_metadata/temporal_windows'][:]
        win = tuple(int(v) for v in f[f'{key}/neuron_metadata/peak_respwindow'][:])
        tsn = np.array(f[f'{key}/trials/stimulus_name'][:], dtype=str)
        days = np.array([_dec(s) for s in f[f'{key}/trials/session_dates'][:]])
        rt = f[f'{key}/trials/response_temporal'][:, :, units]
        reliab = np.array(f[f'{key}/neuron_metadata/reliability'][:])[units]
    raw = P.peak_window_average(rt, tw, win)
    keep = P.outlier_keep_mask(raw, days, cfg['outlier_robust_z'],
                               cfg['outlier_min_trials'], cfg['outlier_drop_if_any_channel'])
    n_dropped = int((~keep).sum())
    raw, tsn, days = raw[keep], tsn[keep], days[keep]
    anchor = np.isin(tsn, list(enc_set))
    train_mask = None
    if cfg['standardize'] == 'st01_train50':
        meta = E.stim_metadata(monkey, units[0], cfg['models'][0])
        train = set(meta['stimulus_name'][meta['is_train']].tolist())
        train_mask = np.isin(tsn, list(train))
    z = P.standardize(raw, days, anchor, method=cfg['standardize'],
                      anchor_min_trials=cfg['anchor_min_trials'], fallback=cfg['fallback'],
                      calrange_pct=cfg['calrange_pct'], tsn=tsn, enc_z=enc_z,
                      train_mask=train_mask, score_of=_score_of)
    mean, nrep = P.trial_average(tsn, z)
    stim = np.array(list(mean.keys()), dtype=object)
    kind = np.array([S.classify(n, enc_set) for n in stim], dtype=object)

    # per (unit, model) noise ceilings
    acc_by = defaultdict(set)
    for i in np.where(~anchor)[0]:
        pr = S.parse_accentuated(tsn[i])
        if pr and pr['unit'] in units and pr['model'] in cfg['models']:
            acc_by[(units.index(pr['unit']), pr['model'])].add(tsn[i])
    cu, cm, cnc, csig, cnv, cns = [], [], [], [], [], []
    for ui, u in enumerate(units):
        tnoise = C.anchor_trial_noise(z[:, ui], tsn, days, enc_set, cfg['ceiling_min_anchor_groups'])
        for model in cfg['models']:
            nc, sig, nv, nst = C.nsd_ceiling(z[:, ui], tsn, acc_by.get((ui, model), set()),
                                             tnoise, cfg['ceiling_min_stim'])
            cu.append(u); cm.append(model); cnc.append(nc); csig.append(sig); cnv.append(nv); cns.append(nst)

    rmap = region_map(monkey, units)
    # firing floor per channel: mean over calibration sessions of the z at 0 spk/s (-mu/sigma),
    # lowered to the minimum observed single-trial calibration z. Computed here (before the
    # accentuation cache) and stored so every prediction clamps against the same value.
    flr = FL.compute(monkey, units)
    brain = dict(
        monkey=monkey, units=np.array(units),
        region=np.array([rmap[u] for u in units], dtype=object),
        mu=mu_all[units].astype(float), sigma=sigma_all[units].astype(float),
        firing_floor=np.array([flr[u]['floor'] for u in units], float),
        firing_floor_avg=np.array([flr[u]['floor_avg'] for u in units], float),
        firing_floor_obs_min=np.array([flr[u]['obs_min'] for u in units], float),
        reliability=reliab.astype(float),
        control=dict(
            trial_stim=tsn.astype(object), trial_day=days.astype(object),
            trial_z=z.astype(np.float32),
            trial_kind=np.array([S.classify(n, enc_set) for n in tsn], dtype=object),
            stim=stim, resp_z=np.array([mean[n] for n in stim], np.float32),
            n_reps=np.array([nrep[n] for n in stim], int), kind=kind),
        calibration=dict(stim=en.astype(object), resp_z=er[:, units].astype(np.float32),
                          trial_stim=np.asarray(etsn).astype(object),
                          trial_z=np.asarray(etr)[:, units].astype(np.float32)),
        ceilings=dict(unit=np.array(cu), model=np.array(cm, dtype=object),
                      nc_r=np.array(cnc), signal_var=np.array(csig),
                      noise_var=np.array(cnv), n_stim=np.array(cns)),
        n_trials_dropped=n_dropped, config=cfg)
    pickle.dump(brain, open(os.path.join(CACHE, f'brain_{monkey}.pkl'), 'wb'))
    print(f'  brain_{monkey}: {len(stim)} stim, dropped {n_dropped} trials, '
          f'{(kind=="accentuated").sum()} accentuated, median nc_r={np.nanmedian(cnc):.3f}')


# ------------------------------ encoding ------------------------------------
def build_encoding(monkey, cfg):
    units = list(MONKEY_UNITS[monkey]); models = cfg['models']
    names = None; pred = np.full((len(units), len(models), 0), np.nan, np.float32)
    readout = {}
    for ui, u in enumerate(units):
        for mi, m in enumerate(models):
            um = E.unit_model(monkey, u, m)
            if names is None:
                names = np.array(list(um['pred'].keys()), dtype=object)
                pred = np.full((len(units), len(models), len(names)), np.nan, np.float32)
            pred[ui, mi] = np.array([um['pred'][n] for n in names], np.float32)
            readout[f'{u}|{m}'] = dict(vec=um['readout_vec'], bias=um['readout_bias'],
                                       layer=um['layer'], fit_method=um['fit_method'])
    enc = dict(monkey=monkey, units=np.array(units), models=np.array(models, dtype=object),
               stim=names, pred=pred, readout=readout)
    pickle.dump(enc, open(os.path.join(CACHE, f'encoding_{monkey}.pkl'), 'wb'))
    print(f'  encoding_{monkey}: {len(units)}x{len(models)} models, {len(names)} stim')


# -------------------- cross-model encoding predictions ----------------------
def build_predictions(monkey, cfg):
    """Every model's encoding prediction for every accentuated stimulus at every target unit
    (the cross-model / pooled matrix). RAW pred_resp (NOT floor-clamped) — the firing floor is
    applied last, only when predictions are compared to the brain (loader.control_cloud), so the
    hit/miss assessment isn't polluted by clamping. Covers the personalized diagonal
    (predicting==generating, own unit) and all off-diagonal pooled / peer-review cells. Keyed by
    the exact brain-cache stimulus name."""
    import pandas as pd
    folder = MONKEY_DIRS[monkey]
    pkl = os.path.join(PRED_ROOT, folder, 'posthoc_model_predict',
                       f'accentuated_stim_info_w_pred_resp_{folder}.pkl')
    df = pd.read_pickle(pkl)
    b = pickle.load(open(os.path.join(CACHE, f'brain_{monkey}.pkl'), 'rb'))
    units = [int(u) for u in b['units']]
    floor = {int(u): float(fl) for u, fl in zip(b['units'], b['firing_floor'])}
    models = cfg['models']
    names = np.array([os.path.basename(str(fp)) for fp in df['filepath']], dtype=object)
    S, M, U = len(df), len(models), len(units)
    pred = np.full((S, M, U), np.nan, np.float32); n_below = 0
    for mi, m in enumerate(models):
        for ui, u in enumerate(units):
            col = f'pred_resp_{m}_unit_{u}'
            if col in df.columns:
                raw = df[col].to_numpy(dtype=float)
                n_below += int(np.sum(raw < floor[u]))
                pred[:, mi, ui] = raw
    out = dict(monkey=monkey, stim=names,
               gen_model=np.array(df['model_name'].astype(str), dtype=object),
               gen_unit=np.array(df['unit_id'].astype(int)),
               gen_seed=np.array(df['img_id'].astype(int)),
               gen_level=np.array(df['level'].astype(float), np.float32),   # target level aimed for
               score=np.array(df['score'].astype(float), np.float32),       # synthesis-time (metadata only)
               target_units=np.array(units), pred_models=np.array(models, dtype=object),
               pred=pred, clamped_to_floor=False)
    pickle.dump(out, open(os.path.join(CACHE, f'predictions_{monkey}.pkl'), 'wb'))
    print(f'  predictions_{monkey}: {S} stim x {M} models x {U} units, {n_below} raw preds below floor '
          f'(clamp applied later, at brain comparison)')


# ---------------- fairness / stimulus-exclusion criteria --------------------
def build_exclusions(monkey, cfg):
    """Per-stimulus fairness criteria (relerr from pred_resp) + exclusion masks, cached early
    so every downstream analysis shares one definition of hit/miss and fair exclusion.
    permodel/drop2 are judged pre-floor (raw pred_resp); intersect (cross-model level-sharing)
    is judged post-floor, so the per-unit firing floors are passed through."""
    P = pickle.load(open(os.path.join(CACHE, f'predictions_{monkey}.pkl'), 'rb'))
    b = pickle.load(open(os.path.join(CACHE, f'brain_{monkey}.pkl'), 'rb'))
    present = {n for n, k in zip(b['control']['stim'], b['control']['kind']) if k == 'accentuated'}
    floors = {int(u): float(fl) for u, fl in zip(b['units'], b['firing_floor'])}
    out = X.compute(P, present, floors)
    pickle.dump(out, open(os.path.join(CACHE, f'exclusions_{monkey}.pkl'), 'wb'))
    tau = X.THRESHOLDS[0]; m = out['masks'][tau]
    print(f'  exclusions_{monkey}: {int(out["present"].sum())} present; @{tau:.0%} '
          f"kept none={int(m['none'].sum())} permodel={int(m['permodel'].sum())} "
          f"drop2={int(m['drop2'].sum())} intersect={int(m['intersect'].sum())}")


# ------------------------------ stimuli -------------------------------------
def build_stimuli(cfg):
    cal = defaultdict(list); acc = defaultdict(list); contr = defaultdict(list)
    for monkey in cfg['monkeys']:
        u0 = MONKEY_UNITS[monkey][0]
        meta = E.stim_metadata(monkey, u0, cfg['models'][0])
        n = len(meta['stimulus_name'])
        cal['monkey'] += [monkey] * n
        cal['stimulus_name'] += meta['stimulus_name'].tolist()
        for fl in E.STIM_FLAGS:
            cal[fl] += (meta[fl].tolist() if meta[fl] is not None else [None] * n)
        cal['image_fp'] += (meta['image_fp'].tolist() if meta['image_fp'] is not None else [None] * n)
        # accentuated stimulus catalog (metadata). `score`/`target` are synthesis-time values
        # (score = achieved, target = level aimed for) kept for the fairness criterion
        # (relerr = |target - score|). The canonical PREDICTOR is pred_resp in predictions_*.pkl.
        b = pickle.load(open(os.path.join(CACHE, f'brain_{monkey}.pkl'), 'rb'))
        for name, k in zip(b['control']['stim'], b['control']['kind']):
            if k == 'accentuated':
                pr = S.parse_accentuated(name)
                acc['monkey'].append(monkey); acc['stimulus_name'].append(name)
                for f in ('model', 'unit', 'seed', 'target', 'score'):
                    acc[f].append(pr[f])
    # controversial stimulus table from the .nc
    ncf = sorted(glob.glob(os.path.join(CONTROV_DIR, '*.nc')))
    if ncf:
        with h5py.File(ncf[0], 'r') as f:
            names = np.array(f['stimulus'][:], dtype=str); paths = np.array(f['image_path'][:], dtype=str)
        for name, ip in zip(names, paths):
            pr = S.parse_controversial(name) or {}
            contr['stimulus_name'].append(name); contr['image_path'].append(ip)
            for f in ('unit', 'img', 'score_r50', 'score_robust'):
                contr[f].append(pr.get(f))
    out = dict(calibration={k: np.array(v, dtype=object) for k, v in cal.items()},
               accentuated={k: np.array(v, dtype=object) for k, v in acc.items()},
               controversial={k: np.array(v, dtype=object) for k, v in contr.items()})
    pickle.dump(out, open(os.path.join(CACHE, 'stimuli.pkl'), 'wb'))
    print(f"  stimuli: {len(out['calibration']['stimulus_name'])} calibration, "
          f"{len(out['accentuated']['stimulus_name'])} accentuated, "
          f"{len(out['controversial']['stimulus_name'])} controversial")


# --------------------------- controversial brain ----------------------------
def build_controversial(cfg):
    sessions = {}
    for nc in sorted(glob.glob(os.path.join(CONTROV_DIR, '*.nc'))):
        label = re.search(r'(red_\d{8}-\d{8})', os.path.basename(nc)).group(1)
        with h5py.File(nc, 'r') as f:
            sessions[label] = dict(
                stim=np.array(f['stimulus'][:], dtype=str).astype(object),
                image_path=np.array(f['image_path'][:], dtype=str).astype(object),
                unit=np.array(f['unit'][:]),
                neural=np.array(f['neural_response'][:], np.float32),
                pred={v: np.array(f[v][:], np.float32) for v in CONTROV_PRED_VARS})
    out = dict(sessions=sessions, pred_vars=CONTROV_PRED_VARS, config=cfg,
               note=('pre-averaged .nc source: neural_response is already trial-averaged, '
                     'so single-trial outlier rejection / standardization / anchor noise '
                     'ceilings do not apply here (unlike the main experiment).'))
    pickle.dump(out, open(os.path.join(CACHE, 'brain_controversial.pkl'), 'wb'))
    shp = ', '.join(f'{k}:{tuple(v["neural"].shape)}' for k, v in sessions.items())
    print(f'  brain_controversial: {len(sessions)} sessions ({shp})')

    # NSD noise ceilings + per-session anchorDay standardization from the raw single-trial H5
    # (self-contained, local). std_avg/std_sem are the standardized trial-avg + SEM per stimulus.
    from pnc.preproc import controversial as CVmod
    cv, std_avg, std_sem = CVmod.build(verbose=True)

    # analysis-ready per-(stim, target-unit) table for the 7 aIT units.
    # neural = anchorDay-STANDARDIZED trial-avg (canonical, cross-day corrected); neural_raw = the
    # pre-averaged .nc repavg (reference); sem on the standardized single trials; + 4 encoding preds.
    s = sessions[CONTROV_SESSION]
    ucol = {int(u): i for i, u in enumerate(s['unit'])}
    t = defaultdict(list)
    for si, name in enumerate(s['stim']):
        pr = S.parse_controversial(name)
        if pr is None or pr['unit'] not in CONTROV_UNITS:
            continue
        ci = ucol[pr['unit']]; u = pr['unit']
        sa = std_avg.get(name); ss = std_sem.get(name)
        t['stim'].append(name); t['image_path'].append(s['image_path'][si])
        t['unit'].append(u); t['img'].append(pr['img'])
        t['score_r50'].append(pr['score_r50']); t['score_robust'].append(pr['score_robust'])
        t['neural'].append(float(sa[u]) if sa is not None else np.nan)          # standardized (canonical)
        t['neural_raw'].append(float(s['neural'][si, ci]))                      # raw .nc repavg (reference)
        t['sem'].append(float(ss[u]) if ss is not None else np.nan)            # SEM on standardized trials
        for v in CONTROV_PRED_VARS:
            t[v].append(float(s['pred'][v][si, ci]))
    tbl = {k: np.array(v, dtype=object if k in ('stim', 'image_path') else float) for k, v in t.items()}
    floor = {u: float(tbl['neural'][tbl['unit'] == u].min()) for u in CONTROV_UNITS}
    tbl['neural_floor'] = np.array([floor[u] for u in tbl['unit']], float)      # per-unit floor (std scale)
    tbl['nc_r'] = np.array([cv['nc_r_controversial'].get(int(u), np.nan) for u in tbl['unit']], float)
    sem_status = 'local: SEM on anchorDay-standardized single trials (self-contained)'
    controv = dict(session=CONTROV_SESSION, units=CONTROV_UNITS, pred_vars=CONTROV_PRED_VARS,
                   table=tbl, sem_status=sem_status, ceilings=cv,
                   neural_scale='anchorDay-standardized (neural_raw = pre-averaged .nc repavg)')
    pickle.dump(controv, open(os.path.join(CACHE, 'controversial.pkl'), 'wb'))
    print(f"  controversial table: {len(tbl['stim'])} rows over units {CONTROV_UNITS} "
          f"({', '.join(str(int((tbl['unit'] == u).sum())) for u in CONTROV_UNITS)} per unit); "
          f"neural=standardized; nc_r per unit {[round(cv['nc_r_controversial'][u], 2) for u in CONTROV_UNITS]}")


def write_manifest(cfg):
    try:
        git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(paths.REPO_ROOT)).decode().strip()
    except Exception:
        git = None
    cfg_json = {k: (list(v) if isinstance(v, (set, tuple)) else v) for k, v in cfg.items()}
    manifest = dict(created=datetime.now().isoformat(timespec='seconds'), git=git,
                    config=cfg_json,
                    inputs=dict(control_h5=os.path.relpath(CONTROL_HDF5_PATH, DATA_ROOT),      # relative to source_data
                                controversial_nc_dir=os.path.relpath(CONTROV_DIR, DATA_ROOT),
                                controversial_h5=os.path.join('brain_data_controversial',
                                'vvs_accentuate_day3_normalize_red_20250123-20250126.hdf5')),
                    # only this script's products (the cache dir may hold other builders' files)
                    outputs=sorted(f for f in os.listdir(CACHE) if f.endswith('.pkl') and (
                        f.split('_')[0] in ('brain', 'encoding', 'predictions', 'exclusions')
                        or f in ('stimuli.pkl', 'controversial.pkl'))))
    json.dump(manifest, open(os.path.join(CACHE, 'MANIFEST.json'), 'w'), indent=2)
    print(f"  MANIFEST.json written (git {git[:8] if git else 'n/a'})")


def main():
    cfg = CONFIG
    if 'main' in cfg['experiments']:
        for monkey in cfg['monkeys']:
            print(f'[{monkey}]')
            build_brain_main(monkey, cfg)      # computes + stores the firing floor
            build_encoding(monkey, cfg)
            build_predictions(monkey, cfg)     # cross-model predictions (pred_resp), clamped
            build_exclusions(monkey, cfg)      # fairness / hit-miss criteria from pred_resp
        build_stimuli(cfg)
    if 'controversial' in cfg['experiments']:
        build_controversial(cfg)
    write_manifest(cfg)
    print('done.')


if __name__ == '__main__':
    main()
