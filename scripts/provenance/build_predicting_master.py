#!/usr/bin/env python3
"""Fig 9 - build the site-model master predictor table (n=250).

Canonical predictors are recomputed from the preproc cache (self-contained, identical to the
figS variance-partition loader); the three legitimately-precomputed axes are merged from their
pipelines: adversarial sensitivity (Cannon PGD; figure8/intermediate), accentuation low-frequency
content B (FFT of on-disk accentuated PNGs; 27b), and Lambda + phase reliance (27 pipeline / Cannon).

PORTED 2026-08-08 from supplementary/SX_predicting/scripts/build_master.py so the S54 pipeline
lives in supplementary/ proper; writes supplementary/scripts/intermediate/fig9_master.csv. Run:
  KMP_DUPLICATE_LIB_OK=TRUE python analyses/preprint_figures/take8/supplementary/scripts/build_predicting_master.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T8 = os.path.join(HERE, '..', '..')
sys.path.insert(0, T8); sys.path.insert(0, os.path.join(T8, 'preproc'))
from preproc import loader as L

INT = os.path.join(HERE, 'intermediate')
FIG8_INT = os.path.join(T8, 'supplementary/SX_advrobustness', 'intermediate')
SUP_INT = os.path.join(T8, 'supplementary', 'intermediate')
FREQ_RES = os.path.join(T8, '..', '..', '27_freq', 'results')
FMIN, FMAX = 1, 112
MK = ['red', 'paul', 'venus', 'leap', 'three0']


def _flat(p):
    P = np.clip(np.asarray(p, float)[FMIN:FMAX], 1e-30, None)
    return float(np.exp(np.mean(np.log(P))) / P.mean())


def _heldout_cv():
    """Held-out-image gradient spectral concentration (CV = SD/mean over the band) per axis, from the
    100 held-out NSD per-image radial profiles (same probe set as adversarial sensitivity; non-circular)."""
    import pickle, re
    rows = []
    for f in glob.glob(os.path.join(FIG8_INT, 'heldout_gradfreq', '*.pkl')):
        mm = re.match(r'(.+?)_unit_(\d+)_model_(.+)_grad_maps_freq_profiles\.pkl', os.path.basename(f))
        if not mm:
            continue
        P = np.clip(np.asarray(pickle.load(open(f, 'rb'))['profiles'], float).mean(0)[FMIN:FMAX], 1e-30, None)
        rows.append(dict(monkey=mm.group(1).split('_')[0], unit=int(mm.group(2)), model=mm.group(3),
                         cv=float(P.std() / P.mean()),
                         ho_flat=float(np.exp(np.mean(np.log(P))) / P.mean())))   # held-out Wiener flatness (convergent)
    return pd.DataFrame(rows)


def _latest(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def build():
    # --- canonical predictors from the preproc cache (verbatim with sup_variance_partition) ---
    ct = L.control_table()[['monkey', 'unit', 'model', 'region', 'robust',
                            'control_r', 'control_slope', 'encoding_r']].copy()
    ct['unit'] = ct['unit'].astype(int); ct['enc_p1'] = ct['encoding_r']

    p2 = []
    for mk in MK:
        for u in L.load_brain(mk)['units']:
            for m in L.config()['models']:
                ax, ay = L.anchor_cloud(mk, int(u), m)
                r = stats.pearsonr(ax, ay)[0] if len(ax) >= 5 and ax.std() > 0 and ay.std() > 0 else np.nan
                p2.append(dict(monkey=mk, unit=int(u), model=m, enc_p2=r))
    m = ct.merge(pd.DataFrame(p2), on=['monkey', 'unit', 'model'], how='left')

    g = L.load_gradient_freq()
    fl = pd.DataFrame([dict(monkey=r['monkey'], unit=int(r['unit']), model=r['model'],
                            flatness=_flat(r['profile_mean'])) for r in g['gradients']])
    m = m.merge(fl, on=['monkey', 'unit', 'model'], how='left')
    m = m.merge(_heldout_cv(), on=['monkey', 'unit', 'model'], how='left')   # held-out gradient concentration (cv)

    e = L.load_eigen()['df']
    # ed_acc = participation-ratio effective dim of the source-layer code over the 110-stim accentuation
    # set (take7 "accentuation-set effective dim"); the strong geometry predictor. ed_nat (calibration/
    # natural) is kept for the site-mean supplement only.
    ed = pd.DataFrame({k: e[k] for k in ['monkey', 'unit', 'model', 'ed_nat', 'ed_acc']}); ed['unit'] = ed['unit'].astype(int)
    m = m.merge(ed, on=['monkey', 'unit', 'model'], how='left')

    hp = pd.read_csv(os.path.join(SUP_INT, 'hp_tuning_selected.csv')); hp['unit'] = hp['channel'].astype(int)
    m = m.merge(hp[['monkey', 'model', 'unit', 'noise']], on=['monkey', 'model', 'unit'], how='left')

    ch = L.channels_df()[['monkey', 'unit', 'reliability']].copy(); ch['unit'] = ch['unit'].astype(int)
    ch['reliability'] = 2 * ch['reliability'] / (1 + ch['reliability'])          # Spearman-Brown
    m = m.merge(ch, on=['monkey', 'unit'], how='left')

    rows = []
    for mk in MK:
        b = L.load_brain(mk); cal = np.asarray(b['calibration']['resp_z'], float)
        for i, u in enumerate(b['units']):
            rows.append(dict(monkey=mk, unit=int(u), calib_skew=float(stats.skew(cal[:, i], nan_policy='omit'))))
    m = m.merge(pd.DataFrame(rows), on=['monkey', 'unit'], how='left')

    rows = []
    for mk in MK:
        for key, v in L.load_encoding(mk)['readout'].items():
            u, mdl = key.split('|'); vec = np.asarray(v['vec'], float)
            rows.append(dict(monkey=mk, unit=int(u), model=mdl,
                             w_norm=float(np.linalg.norm(vec)), w_kurt=float(stats.kurtosis(vec))))
    m = m.merge(pd.DataFrame(rows), on=['monkey', 'unit', 'model'], how='left')

    # --- precomputed axes (legitimate intermediates) -------------------------------
    # adversarial sensitivity: log2-eps AUC over the PGD sweep (figure8's stronger-predictor measure;
    # eps grid is 0.5..64 in powers of 2, so log2-eps weights each octave equally). Recomputed from the
    # same raw sweep figure8 reads (ext/adv_robustness_*.csv), norm=linf, nswing=(up-dn)/range_q99q01.
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    raw = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(FIG8_INT, 'ext', 'adv_robustness_*.csv'))
                     if 'smoke' not in p], ignore_index=True)
    raw = raw[raw.norm == 'linf'].copy(); raw['nswing'] = (raw.adv_up - raw.adv_dn) / raw.range_q99q01
    K = ['model', 'monkey', 'channel']
    grid = raw.groupby(K + ['eps_255'])['nswing'].mean().reset_index()

    def _log_auc(s):
        s = s.sort_values('eps_255'); x = np.log2(s.eps_255.values); y = s.nswing.values
        return float(trapz(y, x) / (x.max() - x.min()))
    adv = grid.groupby(K).apply(_log_auc, include_groups=False).rename('adv_sens').reset_index()
    adv['unit'] = adv['channel'].astype(int)
    m = m.merge(adv[['model', 'monkey', 'unit', 'adv_sens']], on=['model', 'monkey', 'unit'], how='left')

    # accentuation low-frequency content B (27b: FFT of on-disk accentuated PNGs)
    b27 = pd.read_csv(_latest(os.path.join(FREQ_RES, '27b_accentfreq_table_*.csv')))
    b27['unit'] = b27['unit'].astype(int)
    m = m.merge(b27[['monkey', 'unit', 'model', 'lowfrac_drive']].rename(columns={'lowfrac_drive': 'lowfrac'}),
                on=['monkey', 'unit', 'model'], how='left')

    # Lambda (high-freq energy) + phase reliance (27 pipeline)
    mt = pd.read_csv(_latest(os.path.join(FREQ_RES, '27_master_table_*.csv')))
    mt['unit'] = mt['unit'].astype(int)
    m = m.merge(mt[['monkey', 'unit', 'model', 'Lambda', 'phase_reliance']],
                on=['monkey', 'unit', 'model'], how='left')

    # diff-map spectral flatness: Wiener flatness of the accentuation pixel-diff spectrum (29a)
    dm = pd.read_csv(_latest(os.path.join(T8, '..', '..', '29_diffmaps', 'results', '29a_diffflat_table_*.csv')))
    dm['unit'] = dm['unit'].astype(int)
    m = m.merge(dm[['monkey', 'unit', 'model', 'diff_flatness']], on=['monkey', 'unit', 'model'], how='left')

    # ImageNet linear-probe accuracy at each site's own readout layer (analysis 30 / figS69)
    import json
    p30 = os.path.join(T8, '..', '..', '30_imagenet_probe', 'figures', 'raw')
    lm = pd.read_csv(os.path.join(p30, 'readout_layer_map.csv'))
    lps = {mdl: json.load(open(os.path.join(p30, f'probe_layers_{mdl}.json')))['layers']
           for mdl in lm.model.unique()}
    lm['inet_top1'] = [lps[r.model][r.layer]['probe_top1'] for r in lm.itertuples()]
    lm['inet_top5'] = [lps[r.model][r.layer]['probe_top5'] for r in lm.itertuples()]
    m = m.merge(lm[['monkey', 'unit', 'model', 'inet_top1', 'inet_top5']],
                on=['monkey', 'unit', 'model'], how='left')
    m['inet_gap'] = m['inet_top5'] - m['inet_top1']   # top-5 margin: the stable joint signal (top1/top5 r=.99)

    m['site'] = m['monkey'] + '_u' + m['unit'].astype(str)
    core = ['flatness', 'cv', 'ho_flat', 'enc_p1', 'enc_p2', 'ed_nat', 'ed_acc', 'noise', 'w_norm', 'w_kurt',
            'adv_sens', 'lowfrac', 'diff_flatness', 'Lambda', 'phase_reliance', 'reliability', 'calib_skew',
            'inet_top1', 'inet_top5']
    miss = {c: int(m[c].isna().sum()) for c in core if m[c].isna().sum()}
    print(f'n = {len(m)} site-models; sites = {m.site.nunique()}; models = {m.model.nunique()}')
    print('missing per predictor:', miss if miss else 'none')
    os.makedirs(INT, exist_ok=True)
    out = os.path.join(INT, 'fig9_master.csv'); m.to_csv(out, index=False)
    print('wrote', out)
    return m


if __name__ == '__main__':
    build()
