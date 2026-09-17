"""Controversial experiment: per-session (cross-day) standardization + NSD noise ceilings, from the
raw single-trial ephys HDF5 (synced to DATA_ROOT/brain_data_controversial/). Self-contained in take9;
reuses take9's own pipeline.py transforms.

Why this exists: the controversial session spans 4 days and every stimulus is repeated ~1x/day, so
the trial-average carries cross-day gain/offset drift. The session includes ~200 shared NSD natural
images shown every day, which serve as calibration ANCHORS -- exactly as in the main experiment. We
therefore apply the SAME `anchorDay` standardization (per-session z on the anchors) before averaging,
then estimate NSD noise ceilings from the (cross-day) repeats. build() reports the ceiling with vs
without standardization so the cross-day correction is auditable.

  ncsnr      = sqrt(signal_var)/sqrt(noise_var)
  noise_var  = mean over stimuli of within-stimulus trial variance (ddof=1)
  signal_var = var(stimulus means, ddof=1) - noise_var * mean(1/nreps)      (clipped at 0)
  nc_r       = sqrt( ncsnr^2 / (ncsnr^2 + mean(1/nreps)) )   (ceiling of the trial-averaged response)
"""
import os, re
from collections import defaultdict
import numpy as np, h5py

from pnc.utils import DATA_ROOT
from pnc.preproc import pipeline as PL

H5 = os.path.join(DATA_ROOT, 'brain_data_controversial',
                  'vvs_accentuate_day3_normalize_red_20250123-20250126.hdf5')
SESSION = 'red_20250123-20250126'
AIT_UNITS = [1, 9, 15, 16, 25, 37, 44]          # the controversial target units (all l_aIT)
OUTLIER_Z, OUTLIER_MIN, ANCHOR_MIN = 15.0, 10, 5
_ANCHOR = re.compile(r'^shared\d+_nsd')          # shared NSD naturals shown every day = anchors


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _nsd(reps):
    """reps: list of 1-D per-stimulus single-trial arrays (one channel). -> (ncsnr, nc_r, n_stim)."""
    means, nrep, nvar = [], [], []
    for v in reps:
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v) >= 2:
            means.append(v.mean()); nrep.append(len(v)); nvar.append(v.var(ddof=1))
    if len(means) < 3:
        return np.nan, np.nan, len(means)
    nrep = np.array(nrep, float); noise_var = float(np.mean(nvar)); inv_n = float(np.mean(1.0 / nrep))
    sig = max(float(np.var(means, ddof=1)) - noise_var * inv_n, 0.0)
    if noise_var <= 0:
        return np.nan, np.nan, len(means)
    ncsnr = np.sqrt(sig) / np.sqrt(noise_var)
    return float(ncsnr), float(np.sqrt(ncsnr ** 2 / (ncsnr ** 2 + inv_n))), len(means)


def _load():
    with h5py.File(H5, 'r') as f:
        g = f[SESSION]
        tn = _dec(g['trials']['stimulus_name'][()])
        sess = _dec(g['trials']['session_num'][()])
        rp = np.asarray(g['trials']['response_peak'][()], float)          # (T, 64) precomputed peak
        stored_ncsnr = np.asarray(g['neuron_metadata']['ncsnr'][()], float)
        stored_rel = np.asarray(g['neuron_metadata']['reliability'][()], float)
    return rp, tn, sess, stored_ncsnr, stored_rel


def _ceilings(data, tn, target_units):
    """per-channel (all stimuli) + per-target-unit (its controversial stimuli) NSD nc_r for a
    trials x channels response matrix `data`."""
    nchan = data.shape[1]
    by = defaultdict(list)
    for i, s in enumerate(tn):
        by[s].append(i)
    stims = list(by)
    ncr_all = np.full(nchan, np.nan); ncsnr_all = np.full(nchan, np.nan)
    for c in range(nchan):
        ncsnr_all[c], ncr_all[c], _ = _nsd([data[by[s], c] for s in stims])
    ncr_ct = {}
    for u in target_units:
        cs = [s for s in stims if s.startswith('controversial') and re.search(rf'unit_{u}_', s)]
        _, ncr_ct[u], _ = _nsd([data[by[s], u] for s in cs])
    return ncsnr_all, ncr_all, ncr_ct


def build(verbose=True):
    """Standardize (anchorDay on shared-NSD anchors) + NSD ceilings. Returns a dict for the cache,
    and prints the raw-vs-standardized ceiling comparison (the cross-day-correction audit)."""
    rp, tn, sess, stored_ncsnr, stored_rel = _load()
    keep = PL.outlier_keep_mask(rp[:, AIT_UNITS], sess, OUTLIER_Z, OUTLIER_MIN, True)
    rp, tn, sess = rp[keep], tn[keep], sess[keep]
    anchor = np.array([bool(_ANCHOR.match(n)) for n in tn])
    n_anchor_per_sess = {s: int((anchor & (sess == s)).sum()) for s in sorted(set(sess))}
    z = PL.standardize(rp, sess, anchor, method='anchorDay',
                       anchor_min_trials=ANCHOR_MIN, fallback='allday')      # per-session z on anchors

    ncsnr_raw, ncr_raw, ct_raw = _ceilings(rp, tn, AIT_UNITS)
    ncsnr_std, ncr_std, ct_std = _ceilings(z, tn, AIT_UNITS)

    # standardized trial-avg + SEM per stimulus (become the analysis table's neural/sem, self-contained)
    # + pooled controversial noise ceiling over the 140 (unit, controversial-stim) cells.
    by = defaultdict(list)
    for i, s in enumerate(tn):
        by[s].append(i)
    nchan = z.shape[1]
    std_avg = {s: z[ii].mean(0) for s, ii in by.items()}
    std_sem = {s: (z[ii].std(0, ddof=1) / np.sqrt(len(ii)) if len(ii) >= 2 else np.full(nchan, np.nan))
               for s, ii in by.items()}
    pooled = [z[by[s], u] for u in AIT_UNITS for s in by
              if s.startswith('controversial') and re.search(rf'unit_{u}_', s)]
    _, ncr_pool, _ = _nsd(pooled)

    if verbose:
        ok = np.isfinite(ncsnr_raw) & np.isfinite(stored_ncsnr)
        print(f'  controversial: {len(tn)} trials kept (dropped {int((~keep).sum())}), '
              f'anchors/session {n_anchor_per_sess}')
        print(f'  per-channel ncsnr(raw) vs stored h5 ncsnr: corr={np.corrcoef(ncsnr_raw[ok], stored_ncsnr[ok])[0,1]:.3f}')
        print(f'  median nc_r over 64 chan:  raw {np.nanmedian(ncr_raw):.3f}  ->  anchorDay-standardized {np.nanmedian(ncr_std):.3f}')
        print(f'  {"unit":>5s} {"nc_r_raw":>9s} {"nc_r_std":>9s}   (target-unit controversial ceiling)')
        for u in AIT_UNITS:
            print(f'  {u:5d} {ct_raw[u]:9.3f} {ct_std[u]:9.3f}')
    cv = dict(
        session=SESSION, units=np.arange(rp.shape[1]), ait_units=np.array(AIT_UNITS),
        n_trials_kept=int(keep.sum()), n_dropped=int((~keep).sum()),
        anchors_per_session=n_anchor_per_sess,
        nc_r_channel=ncr_std, ncsnr_channel=ncsnr_std,              # anchorDay-standardized (canonical)
        nc_r_channel_raw=ncr_raw, ncsnr_channel_raw=ncsnr_raw,      # unstandardized (for the audit)
        nc_r_controversial={int(u): float(ct_std[u]) for u in AIT_UNITS},
        nc_r_controversial_raw={int(u): float(ct_raw[u]) for u in AIT_UNITS},
        nc_r_pooled=float(ncr_pool),
        stored_ncsnr=stored_ncsnr, stored_reliability=stored_rel,
        method='anchorDay (per-session z on shared-NSD anchors)')
    # std_avg / std_sem are big per-stimulus dicts used to fill the analysis table; returned
    # separately so they are not stored in the compact `ceilings` cache entry.
    return cv, std_avg, std_sem


if __name__ == '__main__':
    build()
