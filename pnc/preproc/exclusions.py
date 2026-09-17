"""Stimulus exclusion / fairness criteria, judged by pred_resp (the trusted predictor).

Whether a synthesized stimulus hit its target is decided by the generating model's own
post-hoc pred_resp for the SAVED image (the prediction-cache diagonal), NOT the synthesis-time
filename score:

    relerr = |target - achieved| / channel target range,   achieved = diagonal pred_resp
    success = relerr <= tau

Two-stage ordering w.r.t. the firing floor (see README):
  1. DIVERGENCE (pre-floor). Whether a stimulus hit its own target is a fact about synthesis,
     independent of the brain baseline — judged on RAW achieved. Clamping first would spuriously
     fail the clamped low-baseline sites. Feeds `permodel` (and `drop2`, a rank of the target
     level that never touches achieved at all).
  2. FLOOR, then CROSS-MODEL LEVEL-SHARING (post-floor). Once every model's achieved is clamped
     to the unit's firing floor, a (unit, seed, level) cell counts as shared iff ALL models
     succeed within tau on the FLOORED values. Feeds `intersect` — the only floor-sensitive mask.

Masks (each also requires the stimulus to be PRESENT = shown with >=1 trial):
    none       present stimuli only                              (reference)
    permodel   present & success (RAW achieved)                  (coverage differs by model)
    drop2      present & not in the 2 strongest drive levels     (equal levels per model)
    intersect  present & ALL models succeed at that (unit, seed, level) cell, judged on FLOORED
               achieved                                          (equal per model, post-floor)

Computed at tau = 5% (default) and 2% of the channel target range.
"""
from collections import defaultdict

import numpy as np

THRESHOLDS = [0.05, 0.02]     # 0.05 is the default success criterion


def _dense_rank(vals):
    order = {v: i for i, v in enumerate(sorted(set(vals)))}
    return np.array([order[v] for v in vals], int)


def compute(P, present_names, floors):
    """P = predictions cache dict; present_names = set of stimulus names shown in control;
    floors = {unit: firing_floor} for the diagonal (generating) unit.
    Returns a dict of per-stimulus arrays + masks[tau][criterion]."""
    units = list(P['target_units']); models = list(P['pred_models'])
    ui = {int(u): k for k, u in enumerate(units)}; mi = {m: k for k, m in enumerate(models)}
    stim = P['stim']; S = len(stim)
    gm = P['gen_model']; gu = P['gen_unit'].astype(int); gs = P['gen_seed'].astype(int)
    target = P['gen_level'].astype(float)
    achieved = np.array([float(P['pred'][i, mi[gm[i]], ui[gu[i]]]) for i in range(S)])
    # post-floor achieved for the cross-model level-sharing mask only (permodel/drop2 stay raw)
    achieved_floored = np.maximum(achieved, np.array([floors[int(u)] for u in gu]))
    present = np.array([n in present_names for n in stim])

    rng = {}
    for u in units:
        t = target[gu == u]; rng[u] = float(t.max() - t.min()) if len(t) and t.max() > t.min() else 1.0
    denom = np.array([rng[u] for u in gu])
    relerr = np.abs(target - achieved) / denom                    # RAW (divergence)
    relerr_floored = np.abs(target - achieved_floored) / denom    # post-floor (level-sharing)

    # level_ord within each (unit, model, seed) sweep
    level_ord = np.full(S, -1, int)
    swp = defaultdict(list)
    for i in range(S):
        swp[(gu[i], gm[i], gs[i])].append(i)
    for idxs in swp.values():
        r = _dense_rank([target[i] for i in idxs])
        for j, i in enumerate(idxs):
            level_ord[i] = r[j]
    nlev = int(level_ord.max()) + 1

    masks = {}
    for tau in THRESHOLDS:
        succ = relerr <= tau                           # pre-floor success (permodel)
        succ_floored = relerr_floored <= tau           # post-floor success (intersect)
        none = present.copy()
        permodel = present & succ
        drop2 = present & (level_ord <= nlev - 3)
        cell = defaultdict(list)                       # (unit, seed, level_ord) -> usable per stim
        for i in range(S):
            cell[(gu[i], gs[i], level_ord[i])].append(present[i] and succ_floored[i])
        intersect = np.array([all(cell[(gu[i], gs[i], level_ord[i])]) for i in range(S)])
        masks[tau] = dict(none=none, permodel=permodel, drop2=drop2, intersect=intersect)

    return dict(stim=stim, gen_model=gm, gen_unit=gu, gen_seed=gs, target=target,
                achieved=achieved.astype(np.float32),
                achieved_floored=achieved_floored.astype(np.float32),
                relerr=relerr.astype(np.float32),
                relerr_floored=relerr_floored.astype(np.float32),
                level_ord=level_ord, present=present, nlev=nlev,
                thresholds=THRESHOLDS, masks=masks)
