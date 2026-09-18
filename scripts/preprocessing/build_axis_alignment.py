#!/usr/bin/env python3
"""Build the axis-alignment cache for the axis_alignment supp figures (one-time engine).

On-axis / off-axis decomposition of each sweep's net latent displacement in the
750-d PCA feature space:

    d   = z_last - z_first            net displacement across the 11 levels
    a^  = readout_vec / ||readout_vec||   (unit encoding axis)
    DM  = |d . a^|                    displacement ALONG the encoding axis
    DOM = ||d - (d.a^) a^||           off-axis residual (spans the other 749 dims)

Raw DM:DOM is < 1 only because DOM aggregates 749 dimensions. Per-dimension
(DM : DOM/sqrt(749)) ~ 15x. We also record, in a FIXED off-axis basis (the
encoding-cloud residual PCs, orthogonal to a^), the sorted per-dimension
off-axis displacements, so figS09 can report the fraction of sweeps in which the
encoding axis is the single MOST-modulated representational dimension.

Additionally, for each model we find its MOST axis-aligned sweep (max DM/DOM) and
build the readout-aligned 2-D embedding of that sweep exactly as Fig 3A-right
(the readout-basis preparation): natural-image cloud + accentuation
trajectory + seed, colored by achieved response (spk/s). figS09 draws these as a
bottom row of per-model exemplars.

Reads the posthoc PCA pickles (torch tensors) via the shared loaders, so it must
run in a torch env (envs/monkey). The cache is plain numpy -> figS09 runs in psn.

Run once:  python scripts/preprocessing/build_axis_alignment.py   (needs torch to unpickle the raw PCA pickles)
"""
import os
import pickle

import numpy as np
from scipy.spatial import ConvexHull

from pnc import paths
from pnc.utils import (MONKEY_UNITS, MODEL_ORDER, load_encoding_pickle,
                       load_accentuated_pickle, load_encoding_hdf5,
                       load_encoding_mu_sigma, z_to_spks, to_numpy)

OUT = os.path.join(str(paths.preprocessed_data()), 'axis_alignment.pkl')
os.makedirs(str(paths.preprocessed_data()), exist_ok=True)

MONKEY_ORDER = ['red', 'paul', 'venus', 'leap', 'three0']
N_PC = 750
N_OFF = N_PC - 1               # 749 off-axis dimensions
READOUT_SCALE = 2.0            # Fig 3A basis scale
CLOUD_VIEW_ELEV = 12


def unit_vec(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ── Fig 3A readout-aligned embedding ─────────
def project_3d_to_2d(points_3d, azimuth=0, elevation=0):
    az, el = np.radians(azimuth), np.radians(elevation)
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    rot = points_3d @ (Rx @ Ry).T
    return rot[:, 0], rot[:, 1], rot[:, 2]


def find_best_azimuth(all_pts):
    best_az, best_vs = 0, 0
    for az in range(0, 360, 5):
        _, ty, _ = project_3d_to_2d(all_pts, azimuth=az)
        vs = ty.max() - ty.min()
        if vs > best_vs:
            best_vs, best_az = vs, az
    cands = []
    for az in range(0, 360, 5):
        _, ty, _ = project_3d_to_2d(all_pts, azimuth=az)
        if (ty.max() - ty.min()) >= best_vs * 0.95:
            tx, _, _ = project_3d_to_2d(all_pts, azimuth=az)
            cands.append((az, tx.max() - tx.min()))
    if cands:
        best_az = max(cands, key=lambda t: t[1])[0]
    return best_az


def build_exemplar(monkey, unit, model, enc_resp_cache, musig_cache):
    """Fig 3A-right readout-aligned embedding of a cell - ALL seed sweeps."""
    enc = load_encoding_pickle(monkey, unit, model)
    acc = load_accentuated_pickle(monkey, unit, model)
    PCA_resp = to_numpy(enc['PCA_resp'])
    rv = to_numpy(enc['readout_vec']).flatten()
    df_enc = enc['df']
    is_train = (df_enc['is_train'].values.astype(bool)
                if 'is_train' in df_enc.columns else np.ones(len(df_enc), bool))

    pts_c = PCA_resp[is_train] - PCA_resp[is_train].mean(0)
    centroid = PCA_resp[is_train].mean(0)
    w_hat = unit_vec(rv)
    y = pts_c @ w_hat * READOUT_SCALE
    resid = pts_c - np.outer(pts_c @ w_hat, w_hat)
    _, _, Vt = np.linalg.svd(resid, full_matrices=False)
    o1, o2 = Vt[0], Vt[1]
    points_3d = np.column_stack([pts_c @ o1, y, pts_c @ o2])

    enc_names = df_enc['stimulus_name'].values
    seed_map = {}
    for i, sp in enumerate(acc['config'].get('seed_image_paths', [])):
        m = np.where(enc_names == os.path.basename(sp))[0]
        if len(m):
            seed_map[i] = PCA_resp[m[0]]

    df_acc = acc['df']
    pca_acc = to_numpy(acc['PCA_resp'])
    mask = (df_acc['model_name'] == model) & (df_acc['unit_id'] == unit)
    df_filt = df_acc[mask]
    trajs, accent_all = [], []
    for img_id in sorted(df_filt['img_id'].unique()):
        img_df = df_filt[df_filt['img_id'] == img_id].sort_values('level')
        if len(img_df) < 2:
            continue
        centered = np.array([pca_acc[idx] for idx in img_df.index]) - centroid
        coords_3d = np.column_stack([centered @ o1, centered @ w_hat * READOUT_SCALE,
                                     centered @ o2])
        entry = dict(coords_3d=coords_3d, scores_z=img_df['score'].values,
                     img_id=int(img_id))
        if img_id in seed_map:
            sc = seed_map[img_id] - centroid
            entry['seed_3d'] = np.array([sc @ o1, sc @ w_hat * READOUT_SCALE, sc @ o2])
        trajs.append(entry)
        accent_all.append(coords_3d)
    azimuth = find_best_azimuth(np.vstack(accent_all)) if accent_all else 0

    x3, y3, _ = project_3d_to_2d(points_3d, azimuth, CLOUD_VIEW_ELEV)
    cloud_x, cloud_y = x3, y3 * 2.0
    names_train = (df_enc.loc[is_train, 'stimulus_name'].values
                   if 'is_train' in df_enc.columns else enc_names)
    enc_stims, enc_resp = enc_resp_cache[monkey]
    mu, sigma = musig_cache[monkey]
    name2z = {n: enc_resp[i, unit] for i, n in enumerate(enc_stims)}
    cloud_resp_spk = np.array([
        float(z_to_spks(np.array([name2z.get(n, 0.0)]), mu, sigma, unit)[0])
        for n in names_train])
    try:
        hull_idx = ConvexHull(np.column_stack([cloud_x, cloud_y])).vertices
    except Exception:
        hull_idx = np.arange(len(cloud_x))

    trajs2d = []
    for t in trajs:
        tx3, ty3, _ = project_3d_to_2d(t['coords_3d'], azimuth, CLOUD_VIEW_ELEV)
        entry = dict(tx=tx3, ty=ty3 * 2.0,
                     scores_spk=np.asarray(z_to_spks(np.asarray(t['scores_z']),
                                                     mu, sigma, unit)))
        if 'seed_3d' in t:
            sx3, sy3, _ = project_3d_to_2d(t['seed_3d'].reshape(1, -1), azimuth,
                                           CLOUD_VIEW_ELEV)
            entry['seed_xy'] = (float(sx3[0]), float(sy3[0] * 2.0))
        trajs2d.append(entry)
    return dict(monkey=monkey, unit=int(unit), model=model,
                cloud_x=cloud_x, cloud_y=cloud_y, cloud_resp_spk=cloud_resp_spk,
                hull_idx=hull_idx, trajs=trajs2d)


def main():
    monkey, unit_arr, model_arr, img_arr = [], [], [], []
    DM, DOM = [], []
    off_spectra = []           # per sweep: sorted-descending |d . e_off| over 749 dims

    for mk in MONKEY_ORDER:
        for uid in MONKEY_UNITS[mk]:
            for model in MODEL_ORDER:
                try:
                    enc = load_encoding_pickle(mk, uid, model)
                    acc = load_accentuated_pickle(mk, uid, model)
                except FileNotFoundError:
                    print(f'  [miss] {mk} u{uid} {model}')
                    continue

                Pe = to_numpy(enc['PCA_resp'])[:, :N_PC]
                ah = unit_vec(to_numpy(enc['readout_vec'])[:N_PC])
                mean = Pe.mean(0)
                C = Pe - mean
                Rc = C - np.outer(C @ ah, ah)
                _, _, Vr = np.linalg.svd(Rc - Rc.mean(0), full_matrices=False)
                Voff = Vr[:N_OFF]

                df = acc['df']
                P = to_numpy(acc['PCA_resp'])[:, :N_PC]
                mask = ((df['model_name'] == model) & (df['unit_id'] == uid)).values
                sub = df[mask]
                Pd = P[mask]

                for img in sorted(sub['img_id'].unique()):
                    im = (sub['img_id'] == img).values
                    lv = sub.loc[im, 'level'].values
                    if len(lv) < 3:
                        continue
                    z = Pd[im][np.argsort(lv)] - mean
                    d = z[-1] - z[0]
                    dm = abs(float(d @ ah))
                    dom = float(np.linalg.norm(d - (d @ ah) * ah))
                    off_proj = np.sort(np.abs(Voff @ d))[::-1]

                    monkey.append(mk); unit_arr.append(int(uid))
                    model_arr.append(model); img_arr.append(int(img))
                    DM.append(dm); DOM.append(dom)
                    off_spectra.append(off_proj.astype(np.float32))
        print(f'{mk}: done')

    monkey = np.array(monkey); unit_arr = np.array(unit_arr)
    model_arr = np.array(model_arr); img_arr = np.array(img_arr)
    DM = np.array(DM); DOM = np.array(DOM)
    off_spectra = np.array(off_spectra)

    # ── per-model REPRESENTATIVE channel (mean DM/DOM closest to model mean)
    #    + Fig 3A embedding of ALL 10 seed sweeps of that channel ───────────
    print('Building per-model exemplars (representative channel, all seeds)...')
    enc_cache, musig_cache = {}, {}
    for mk in MONKEY_ORDER:
        s, r, _, _ = load_encoding_hdf5(mk)
        enc_cache[mk] = (s, r)
        musig_cache[mk] = load_encoding_mu_sigma(mk)
    ratio_all = DM / DOM
    scale = np.sqrt(N_OFF)
    exemplars = {}          # representative channel (mean DM/DOM ~ model mean)
    exemplars_aligned = {}  # most-aligned channel (max mean DM/DOM)
    for model in MODEL_ORDER:
        selm = model_arr == model
        model_mean = float(ratio_all[selm].mean())
        sites = sorted(set(zip(monkey[selm], unit_arr[selm])))
        site_mean = {(mk, u): float(ratio_all[selm & (monkey == mk) & (unit_arr == u)].mean())
                     for (mk, u) in sites}
        mk, u = min(sites, key=lambda s: abs(site_mean[s] - model_mean))
        ex = build_exemplar(mk, int(u), model, enc_cache, musig_cache)
        ex['chan_mean_perdim'] = site_mean[(mk, u)] * scale
        ex['model_mean_perdim'] = model_mean * scale
        exemplars[model] = ex
        amk, au = max(sites, key=lambda s: site_mean[s])
        exa = build_exemplar(amk, int(au), model, enc_cache, musig_cache)
        exa['chan_mean_perdim'] = site_mean[(amk, au)] * scale
        exemplars_aligned[model] = exa
        print(f'  {model:26s} repr {mk} u{u} (DM/DOM={site_mean[(mk, u)]:.2f}) | '
              f'most-aligned {amk} u{au} (DM/DOM={site_mean[(amk, au)]:.2f}) | '
              f'model mean={model_mean:.2f}')

    out = dict(monkey=monkey, unit=unit_arr, model=model_arr, img=img_arr,
               DM=DM, DOM=DOM, off_spectrum=off_spectra, n_pc=N_PC, n_off=N_OFF,
               exemplars=exemplars, exemplars_aligned=exemplars_aligned)
    with open(OUT, 'wb') as f:
        pickle.dump(out, f)

    # ── sanity: reproduce the draft numbers ──────────────────────────────
    ratio = DM / (DOM / np.sqrt(N_OFF))
    frac_most = (DM > off_spectra[:, 0]).mean() * 100
    print(f'\nSaved {len(DM)} sweeps -> {OUT}')
    print(f'  raw DM:DOM               median = {np.median(DM / DOM):.2f}:1')
    print(f'  per-dim DM:(DOM/sqrt749) median = {np.median(ratio):.1f}x  '
          f'(IQR {np.percentile(ratio, 25):.0f}-{np.percentile(ratio, 75):.0f})')
    print(f'  encoding axis is the single most-modulated dim in {frac_most:.1f}% of sweeps')


if __name__ == '__main__':
    main()
