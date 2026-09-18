"""Read side of the take9 preproc caches - the ONLY entry every downstream figure and
notebook uses. Pure numpy/pandas; loads in any env (no pyarrow, no torch, no h5py).

    from pnc.preproc import loader as L
    tab = L.control_table()                 # per (monkey, unit, model) outcome table
    pred, meas = L.control_cloud('red', 9, 'resnet50')     # accentuated cloud
    D = L.load_brain('red')                 # raw processed object for a monkey

`control_table` is the canonical substrate for Fig 4 and most control figures. Outcomes are
computed on the standardized responses with predicted scores clamped to the 0-spk/s* floor.
"""
import os
import pickle
import functools

import numpy as np
import pandas as pd
from scipy import stats

from pnc import paths

CACHE = str(paths.preprocessed_data())


# ---- Leap variant deduplication --------------------------------------------------------
# Three Leap site-models (CLIPAG u81, SigLIP2 u81, SigLIP2 u342) were synthesized twice and
# BOTH variants of 277 design cells were presented (5,777 stimuli for 5,500 cells). Pixel
# audit (preprocessed_data/leap_variant_similarity.json): variant pairs with pixel r >= 0.90 are
# the same image re-synthesized (baseline: adjacent-level stimuli of the same sweep reach at
# most ~0.74) and are MERGED here as repeats of one canonical stimulus (responses pooled by
# trial count, predictions averaged). Pairs below 0.90 diverged into genuinely different
# images and are kept as unique stimuli.
LEAP_MERGE_R = 0.90


@functools.lru_cache(maxsize=None)
def _leap_merge_map():
    """{dropped stim name -> canonical stim name} for the pixel-similar Leap variant pairs.
    Canonicalization is PRESENTATION-AWARE: when exactly one variant of a pair was presented
    to the animal, that variant is canonical (so the prediction row keeps the name the brain
    data carries); when both or neither were presented, the alphabetically first is canonical
    (both-presented pairs are then pooled as repeats by load_brain)."""
    import json
    path = os.path.join(CACHE, 'leap_variant_similarity.json')
    if not os.path.exists(path):
        return {}
    # names actually shown in the Leap control sessions; decides which variant is canonical
    presented = set(pickle.load(open(os.path.join(CACHE, 'brain_leap.pkl'),
                                     'rb'))['control']['stim'].tolist())
    out = {}
    for r in json.load(open(path)):
        if r['pixel_r'] < LEAP_MERGE_R:
            continue
        a, b = sorted([r['stim_a'], r['stim_b']])
        if (a in presented) == (b in presented):
            out[b] = a                       # both or neither presented: alphabetical
        elif a in presented:
            out[b] = a                       # only a presented: a canonical
        else:
            out[a] = b                       # only b presented: b canonical
    return out


@functools.lru_cache(maxsize=None)
def load_brain(monkey):
    b = pickle.load(open(os.path.join(CACHE, f'brain_{monkey}.pkl'), 'rb'))
    mm = _leap_merge_map() if monkey == 'leap' else {}
    if mm:
        c = b['control']
        # trial level: relabel merged variants to the canonical name
        c['trial_stim'] = np.array([mm.get(n, n) for n in c['trial_stim']])
        # per-stimulus aggregates: pool the merged pair weighted by trial count
        names = list(c['stim'])
        keep = np.ones(len(names), bool)
        idx = {n: i for i, n in enumerate(names)}
        for b_name, a_name in mm.items():
            if b_name not in idx or a_name not in idx:
                continue                     # only one variant was presented: nothing to pool
            ia, ib = idx[a_name], idx[b_name]
            na, nb = c['n_reps'][ia], c['n_reps'][ib]
            c['resp_z'][ia] = (c['resp_z'][ia] * na + c['resp_z'][ib] * nb) / (na + nb)
            c['n_reps'][ia] = na + nb
            keep[ib] = False
        # drop the absorbed variants from every per-stimulus array
        for k in ('stim', 'resp_z', 'n_reps', 'kind'):
            c[k] = np.asarray(c[k])[keep]
    return b


@functools.lru_cache(maxsize=None)
def load_encoding(monkey):
    return pickle.load(open(os.path.join(CACHE, f'encoding_{monkey}.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_stimuli():
    return pickle.load(open(os.path.join(CACHE, 'stimuli.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_controversial():
    return pickle.load(open(os.path.join(CACHE, 'brain_controversial.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_predictions(monkey):
    """Cross-model predictions: pred[stim, predicting_model, target_unit], stored RAW (the
    firing floor is applied later, at the brain comparison, not baked into this cache).
    Diagonal (predicting==generating, own unit) is the personalized prediction; off-diagonal
    is the pooled / peer-review matrix. Pixel-similar Leap synthesis variants are merged
    (predictions averaged; see _leap_merge_map)."""
    P = pickle.load(open(os.path.join(CACHE, f'predictions_{monkey}.pkl'), 'rb'))
    mm = _leap_merge_map() if monkey == 'leap' else {}
    if mm:
        names = list(P['stim'])
        keep = np.ones(len(names), bool)
        idx = {n: i for i, n in enumerate(names)}
        # collapse onto the canonical variant's OWN prediction row (no averaging): the
        # canonical image is the one that was presented, and its re-encoded prediction is
        # what the stated method uses; the dropped variant's row simply vanishes.
        for b_name, a_name in mm.items():
            if b_name in idx:
                keep[idx[b_name]] = False
        for k in ('stim', 'gen_model', 'gen_unit', 'gen_seed', 'gen_level', 'score'):
            P[k] = np.asarray(P[k])[keep]
        P['pred'] = np.asarray(P['pred'])[keep]
    return P


@functools.lru_cache(maxsize=None)
def load_controversial_table():
    """Controversial R50-vs-robust analysis table (140 rows = 7 aIT units x 20 stims):
    neural + 4 encoding preds (RN50/RN50rbst x Lasso/Ridge) + synthesis scores + SEM +
    per-unit raw floor + per-unit noise ceiling `nc_r`. Raw response units. Also carries a
    `ceilings` dict (per-channel + per-target-unit NSD nc_r, anchorDay-standardized and raw)."""
    return pickle.load(open(os.path.join(CACHE, 'controversial.pkl'), 'rb'))


def controversial_ceilings():
    """NSD noise ceilings for the controversial experiment, from the raw single-trial H5 with
    per-session (cross-day) anchorDay standardization on the shared-NSD anchors. Keys: nc_r_channel
    /ncsnr_channel (64, standardized = canonical) + *_raw (unstandardized audit) + nc_r_controversial
    {unit: nc_r over its 20 controversial stims} (+ _raw) + stored_ncsnr/stored_reliability (from h5)."""
    return load_controversial_table()['ceilings']


@functools.lru_cache(maxsize=None)
def load_gradient_freq():
    """Gradient radial-Fourier profiles per (site,model): gradients[{monkey,unit,model,
    profile_mean,profile_std}] + freqs + natural profiles. Flatness = geo/arith mean of
    profile_mean over a freq band (Fig 7)."""
    return pickle.load(open(os.path.join(CACHE, 'gradient_freq.pkl'), 'rb'))


def load_grad_maps(monkey, unit):
    """Per-model 2D input-output gradient maps for one site: {model: grad_img (n_seed,3,H,W)}.
    Fig 5 gallery ingredient (cached under preprocessed_data/grad_maps/)."""
    import glob
    md = {'red': 'red_20250428-20250430', 'paul': 'paul_20250428-20250430',
          'venus': 'venus_250426-250429', 'leap': 'leap_250426-250501',
          'three0': 'three0_250426-250501'}[monkey]
    out = {}
    for p in sorted(glob.glob(os.path.join(CACHE, 'grad_maps',
                                           f'{md}_unit_{unit}_model_*_grad_maps_freq_profiles.pkl'))):
        model = os.path.basename(p).split('_model_')[1].rsplit('_grad_maps', 1)[0]
        out[model] = pickle.load(open(p, 'rb'))['grad_img']
    return out


@functools.lru_cache(maxsize=None)
def load_axis_alignment():
    return pickle.load(open(os.path.join(CACHE, 'axis_alignment.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_layer_selection():
    return pickle.load(open(os.path.join(CACHE, 'layer_selection.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_floc_selectivity():
    return pickle.load(open(os.path.join(CACHE, 'floc_selectivity.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_tuning_stability():
    return pickle.load(open(os.path.join(CACHE, 'tuning_stability.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_imagenet():
    """50k ImageNet-val encoding predictions: pred[image, site, model] (raw). Sites in
    (unit_monkeys, unit_ids) order; models in `models`. Extracted by take9/cluster/."""
    return pickle.load(open(os.path.join(CACHE, 'imagenet_predictions.pkl'), 'rb'))


@functools.lru_cache(maxsize=None)
def load_exclusions(monkey):
    """Fairness / exclusion criteria (relerr from pred_resp) + masks[tau][criterion] over the
    accentuated stimuli. criterion in {none, permodel, drop2, intersect}; tau in {0.02, 0.05}.
    none/permodel/drop2 are pre-floor (raw pred_resp); intersect is post-floor (cross-model
    level-sharing judged on firing-floored achieved). Also carries achieved/achieved_floored +
    relerr/relerr_floored. Rows are aligned to load_predictions' post-merge stimulus order
    (the frozen cache predates the Leap variant merge; dropped variants' rows are removed,
    the canonical variant's criteria kept -- variant relerrs differ by ~1e-3 at most)."""
    e = pickle.load(open(os.path.join(CACHE, f'exclusions_{monkey}.pkl'), 'rb'))
    mm = _leap_merge_map() if monkey == 'leap' else {}
    if mm:
        # reindex the frozen (pre-merge) rows to the merged prediction order; dropped variants fall out
        order = {n: i for i, n in enumerate(e['stim'])}
        idx = np.array([order[n] for n in load_predictions(monkey)['stim']])
        for k in ('stim', 'gen_model', 'gen_unit', 'gen_seed', 'target', 'achieved',
                  'achieved_floored', 'relerr', 'relerr_floored', 'level_ord', 'present'):
            e[k] = np.asarray(e[k])[idx]
        e['masks'] = {t: {c: np.asarray(v)[idx] for c, v in cm.items()}
                      for t, cm in e['masks'].items()}
    return e


def config():
    return load_brain('red')['config']


def monkeys():
    return list(config()['monkeys'])


def firing_floor(monkey, unit):
    b = load_brain(monkey)
    return float(b['firing_floor'][list(b['units']).index(unit)])


@functools.lru_cache(maxsize=None)
def _acc_response(monkey, unit):
    """{accentuated stimulus_name: measured standardized response at this unit}."""
    b = load_brain(monkey); ui = list(b['units']).index(unit); c = b['control']
    return {n: float(c['resp_z'][i, ui]) for i, n in enumerate(c['stim'])
            if c['kind'][i] == 'accentuated'}


def control_cloud(monkey, unit, model):
    """(predicted response, measured standardized response) over the (unit, model) personalized
    accentuated sweep - this model's own accentuations for this unit, scored by its own readout
    (the diagonal of the prediction cache). The firing floor is applied HERE - the last step
    before comparing to the brain - not baked into the cache."""
    P = load_predictions(monkey)
    mi = list(P['pred_models']).index(model); ui = list(P['target_units']).index(unit)
    # rows synthesized by (model, unit), scored by that same model's readout of the same unit
    sel = (P['gen_model'] == model) & (P['gen_unit'] == unit)
    resp = _acc_response(monkey, unit)
    x, y = [], []
    for n, p in zip(P['stim'][sel], P['pred'][sel, mi, ui]):
        if n in resp:
            x.append(float(p)); y.append(resp[n])
    x = np.maximum(np.array(x, float), firing_floor(monkey, unit))     # clamp at brain comparison
    return x, np.array(y, float)


def anchor_cloud(monkey, unit, model):
    """(model encoding prediction, measured standardized control response) over the control
    anchor stimuli - the encoding-generalization cloud. Encoding predictions are not clamped
    (the firing-floor clamp applies to control predictions only)."""
    b = load_brain(monkey); ui = list(b['units']).index(unit); c = b['control']
    e = load_encoding(monkey); mi = list(e['models']).index(model)
    epred = dict(zip(e['stim'].tolist(), e['pred'][list(e['units']).index(unit), mi]))
    x, y = [], []
    # anchors = calibration images re-shown in the control session, paired with their encoding prediction
    for i, (n, k) in enumerate(zip(c['stim'], c['kind'])):
        if k == 'calibration' and n in epred:
            x.append(epred[n]); y.append(c['resp_z'][i, ui])
    return np.array(x, float), np.array(y, float)


@functools.lru_cache(maxsize=None)
def _test_names(monkey):
    """Held-out encoding stimuli = NOT is_train (matches Fig 4's phase-1 test split)."""
    st = load_stimuli()['calibration']; m = st['monkey'] == monkey
    names = st['stimulus_name'][m]; tr = st['is_train'][m] if st['is_train'] is not None else None
    if tr is None:
        return None                          # no split recorded: callers fall back to every stimulus
    return set(names[~np.array([bool(x) for x in tr])].tolist())


def encoding_cloud(monkey, unit, model, split='test'):
    """(model encoding prediction, measured encoding-phase response) over the encoding stimuli.
    split='test' restricts to held-out stimuli; split='all' uses every encoding stimulus."""
    b = load_brain(monkey); e = load_encoding(monkey)
    ui = list(b['units']).index(unit); mi = list(e['models']).index(model)
    epred = dict(zip(e['stim'].tolist(), e['pred'][ui, mi]))
    keep = _test_names(monkey) if split == 'test' else None
    x, y = [], []
    for n, r in zip(b['calibration']['stim'], b['calibration']['resp_z'][:, ui]):
        if n in epred and (keep is None or n in keep):
            x.append(epred[n]); y.append(r)
    return np.array(x, float), np.array(y, float)


def shared_cloud(monkey, unit, model):
    """(model encoding prediction, measured control response) over the 'shared' calibration
    stimuli - the cross-phase set used by Fig 4's phase-2 'Encoding' panel. Prediction unclamped."""
    b = load_brain(monkey); ui = list(b['units']).index(unit); c = b['control']
    e = load_encoding(monkey); mi = list(e['models']).index(model)
    epred = dict(zip(e['stim'].tolist(), e['pred'][list(e['units']).index(unit), mi]))
    x, y = [], []
    # the shared set is identified by its stimulus names
    for i, n in enumerate(c['stim']):
        if 'shared' in n.lower() and n in epred:
            x.append(epred[n]); y.append(c['resp_z'][i, ui])
    return np.array(x, float), np.array(y, float)


def control_seed_rs(monkey, unit, model, min_per_seed=5):
    """Per-seed control r for one (unit, model): list over seeds with >= min_per_seed stimuli.
    Predictor = raw pred_resp from the predictions cache, floor-clamped here at comparison."""
    P = load_predictions(monkey)
    mi = list(P['pred_models']).index(model); ui = list(P['target_units']).index(unit)
    sel = (P['gen_model'] == model) & (P['gen_unit'] == unit)
    resp = _acc_response(monkey, unit); byseed = {}; fl = firing_floor(monkey, unit)
    for n, p, s in zip(P['stim'][sel], P['pred'][sel, mi, ui], P['gen_seed'][sel]):
        if n in resp:
            byseed.setdefault(int(s), []).append((max(float(p), fl), resp[n]))   # clamp at comparison
    out = []
    for lst in byseed.values():
        if len(lst) >= min_per_seed:
            x = np.array([a for a, _ in lst]); y = np.array([b_ for _, b_ in lst])
            if x.std() > 0 and y.std() > 0:  # pearsonr is undefined for a constant vector
                out.append(float(stats.pearsonr(x, y)[0]))
    return out


def ceilings_df(monkey):
    return pd.DataFrame(load_brain(monkey)['ceilings'])


def channels_df():
    rows = []
    for mk in monkeys():
        b = load_brain(mk)
        for i, u in enumerate(b['units']):
            rows.append(dict(monkey=mk, unit=int(u), region=str(b['region'][i]),
                             firing_floor=float(b['firing_floor'][i]),
                             reliability=float(b['reliability'][i])))
    return pd.DataFrame(rows)


def _measures(pred, meas):
    if len(pred) < 5 or pred.std() == 0 or meas.std() == 0:
        return np.nan, np.nan, np.nan
    r = stats.pearsonr(pred, meas)[0]
    slope = stats.linregress(pred, meas).slope
    ss = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - np.sum((meas - pred) ** 2) / ss if ss > 0 else np.nan
    return float(r), float(slope), float(r2)


def control_table():
    """Per (monkey, unit, model): control r / slope / R2 (accentuated), encoding-generalization
    R2 (anchors), noise ceiling, region, robust flag. The canonical downstream table.
    Predicted scores are stored raw; the firing floor is applied here at the brain comparison
    (via control_cloud), never baked into the cache."""
    cfg = config(); robust = set(cfg['robust_models']); rows = []
    for mk in cfg['monkeys']:
        b = load_brain(mk)
        cd = {(r.unit, r.model): r.nc_r for r in ceilings_df(mk).itertuples()}
        reg = {int(u): str(b['region'][i]) for i, u in enumerate(b['units'])}
        for u in b['units']:
            for m in cfg['models']:
                pr, me = control_cloud(mk, int(u), m)
                cr, cs, cR2 = _measures(pr, me)
                ax, ay = anchor_cloud(mk, int(u), m)
                _, _, egr2 = _measures(ax, ay)
                ex, ey = encoding_cloud(mk, int(u), m, 'test')
                er = stats.pearsonr(ex, ey)[0] if len(ex) >= 5 and ex.std() > 0 and ey.std() > 0 else np.nan
                rows.append(dict(monkey=mk, unit=int(u), model=m, region=reg[int(u)],
                                 robust=int(m in robust), n=len(pr), encoding_r=er,
                                 control_r=cr, control_slope=cs, control_R2=cR2,
                                 enc_gen_R2=egr2, nc_r=cd.get((int(u), m), np.nan)))
    return pd.DataFrame(rows)
