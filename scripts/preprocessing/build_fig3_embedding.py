#!/usr/bin/env python3
"""Build the Figure 3 panel-a embedding cache, preprocessed_data/fig3_data_<monkey>_unit<unit>_<model>.pkl.

The embedding (encoding + accentuated stimuli in the readout-aligned basis) is built
from RAW shared data only: the PCA-projection pickles (source_data/image_pca_projections),
the accentuation sweep metadata they carry, and the calibration responses in the brain
cache (pnc.preproc.loader.load_brain). The pickles contain torch tensors, so this must run
in an environment with torch installed; figures/main/fig3_single_site.py only reads the
resulting cache.

    python scripts/preprocessing/build_fig3_embedding.py [--monkey red --unit 9 --model resnet50] [--out DIR]
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial import ConvexHull

from pnc import paths
from pnc.utils import DATA_ROOT, MONKEY_DIRS
from pnc.preproc import loader as L

# ── embedding-basis config ──────────────────────────────────────────────
READOUT_SCALE = 2.0
CLOUD_VIEW_ELEV = 12


def _to_numpy(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def _project_3d_to_2d(points_3d, azimuth=0, elevation=0):
    az = np.radians(azimuth)
    el = np.radians(elevation)
    Ry = np.array([[np.cos(az), 0, np.sin(az)],
                   [0, 1, 0],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    rotated = points_3d @ (Rx @ Ry).T
    return rotated[:, 0], rotated[:, 1], rotated[:, 2]


def _find_best_azimuth(all_pts):
    best_az, best_vs = 0, 0
    for az in range(0, 360, 5):
        _, ty, _ = _project_3d_to_2d(all_pts, azimuth=az, elevation=0)
        vs = ty.max() - ty.min()
        if vs > best_vs:
            best_vs, best_az = vs, az
    cands = []
    for az in range(0, 360, 5):
        _, ty, _ = _project_3d_to_2d(all_pts, azimuth=az, elevation=0)
        if (ty.max() - ty.min()) >= best_vs * 0.95:
            tx, _, _ = _project_3d_to_2d(all_pts, azimuth=az, elevation=0)
            cands.append((az, tx.max() - tx.min()))
    if cands:
        best_az = max(cands, key=lambda t: t[1])[0]
    return best_az


def _raw_pca_pickle_paths(MONKEY, UNIT, MODEL):
    mdir = MONKEY_DIRS[MONKEY]
    base = os.path.join(DATA_ROOT, 'image_pca_projections', mdir,
                        'posthoc_model_predict_PCA_popul_unit')
    enc = os.path.join(base,
                       f'posthoc_prediction_NSDencimg_PCA_pop_unit_'
                       f'{mdir}_unit{UNIT}_{MODEL}.pkl')
    acc = os.path.join(base,
                       f'posthoc_prediction_PCA_pop_unit_'
                       f'{mdir}_unit{UNIT}_{MODEL}.pkl')
    return enc, acc


def _prepare_readout_basis(enc_data, accent_data, UNIT, MODEL):
    PCA_resp = _to_numpy(enc_data['PCA_resp'])
    readout_vec = _to_numpy(enc_data['readout_vec']).flatten()
    df_enc = enc_data['df']
    is_train = (df_enc['is_train'].values.astype(bool)
                if 'is_train' in df_enc.columns
                else np.ones(len(df_enc), dtype=bool))

    pts_full = PCA_resp[is_train]
    centroid = pts_full.mean(axis=0)
    pts_c = pts_full - centroid
    w_hat = readout_vec / np.linalg.norm(readout_vec)
    y_coords = pts_c @ w_hat * READOUT_SCALE
    residuals = pts_c - np.outer(pts_c @ w_hat, w_hat)
    _, _, Vt = np.linalg.svd(residuals, full_matrices=False)
    ortho1, ortho2 = Vt[0], Vt[1]
    points_3d = np.column_stack([pts_c @ ortho1, y_coords, pts_c @ ortho2])

    enc_names = df_enc['stimulus_name'].values
    seed_coords_map = {}
    for i, sp in enumerate(accent_data['config'].get('seed_image_paths', [])):
        matches = np.where(enc_names == os.path.basename(sp))[0]
        if len(matches):
            seed_coords_map[i] = PCA_resp[matches[0]]

    df_acc = accent_data['df']
    pca_acc = _to_numpy(accent_data['PCA_resp'])
    mask = (df_acc['model_name'] == MODEL) & (df_acc['unit_id'] == UNIT)
    df_filt = df_acc[mask]

    trajectories, accent_3d_all = [], []
    for img_id in sorted(df_filt['img_id'].unique()):
        img_df = df_filt[df_filt['img_id'] == img_id].sort_values('level')
        if len(img_df) < 2:
            continue
        coords_full = np.array([pca_acc[idx] for idx in img_df.index])
        centered = coords_full - centroid
        coords_3d = np.column_stack([
            centered @ ortho1, centered @ w_hat * READOUT_SCALE, centered @ ortho2])
        entry = dict(coords_3d=coords_3d,
                     scores_z=img_df['score'].values, img_id=int(img_id))
        if img_id in seed_coords_map:
            sc = seed_coords_map[img_id] - centroid
            entry['seed_3d'] = np.array([sc @ ortho1, sc @ w_hat * READOUT_SCALE,
                                         sc @ ortho2])
        trajectories.append(entry)
        accent_3d_all.append(coords_3d)

    azimuth = _find_best_azimuth(np.vstack(accent_3d_all)) if accent_3d_all else 0
    names_train = (df_enc.loc[is_train, 'stimulus_name'].values
                   if 'is_train' in df_enc.columns
                   else df_enc['stimulus_name'].values)
    return dict(points_3d=points_3d, trajectories=trajectories,
                azimuth=azimuth, names_train=names_train)


def build_embedding_cache(MONKEY, UNIT, MODEL, CACHE_PATH):
    """Build panel-a embedding from RAW shared data + calibration responses."""
    enc_path, acc_path = _raw_pca_pickle_paths(MONKEY, UNIT, MODEL)
    print(f'Building embedding from raw:\n  {enc_path}\n  {acc_path}')
    with open(enc_path, 'rb') as f:
        ed = pickle.load(f)
    with open(acc_path, 'rb') as f:
        ad = pickle.load(f)

    basis = _prepare_readout_basis(ed, ad, UNIT, MODEL)
    x3, y3, _ = _project_3d_to_2d(basis['points_3d'],
                                  azimuth=basis['azimuth'],
                                  elevation=CLOUD_VIEW_ELEV)
    cloud_x = x3
    cloud_y = y3 * 2.0

    # measured encoding-phase responses in spk/s* from the calibration cache
    b = L.load_brain(MONKEY)
    ui = list(b['units']).index(UNIT)
    mu, sg = float(b['mu'][ui]), float(b['sigma'][ui])
    cal = b['calibration']
    name_to_z = {n: float(r) for n, r in zip(cal['stim'], cal['resp_z'][:, ui])}
    cloud_resp_spk = np.array([
        max(0.0, name_to_z.get(n, 0.0) * sg + mu) for n in basis['names_train']])

    pts_2d = np.column_stack([cloud_x, cloud_y])
    try:
        hull_idx = ConvexHull(pts_2d).vertices
    except Exception:
        hull_idx = np.arange(len(pts_2d))

    traj_list = []
    for t in basis['trajectories']:
        tx3, ty3, _ = _project_3d_to_2d(t['coords_3d'], azimuth=basis['azimuth'],
                                        elevation=CLOUD_VIEW_ELEV)
        t2d = dict(tx=tx3, ty=ty3 * 2.0,
                   scores_spk=np.maximum(0.0, np.asarray(t['scores_z']) * sg + mu))
        if 'seed_3d' in t:
            sx3, sy3, _ = _project_3d_to_2d(t['seed_3d'].reshape(1, -1),
                                            azimuth=basis['azimuth'],
                                            elevation=CLOUD_VIEW_ELEV)
            t2d['seed_xy'] = (float(sx3[0]), float(sy3[0] * 2.0))
        traj_list.append(t2d)

    cache = dict(
        mu=mu, sigma=sg,
        cloud_x=cloud_x, cloud_y=cloud_y,
        cloud_resp_spk=cloud_resp_spk,
        cloud_names=basis['names_train'],
        cloud_hull_idx=hull_idx,
        cloud_azimuth=basis['azimuth'],
        cloud_traj_2d=traj_list,
    )
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_PATH)), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Cached embedding -> {CACHE_PATH}')
    return cache


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--monkey', default='red')
    ap.add_argument('--unit', type=int, default=9)
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--out', default=str(paths.preprocessed_data()),
                    help='directory that receives fig3_data_<monkey>_unit<unit>_<model>.pkl')
    args = ap.parse_args()
    build_embedding_cache(args.monkey, args.unit, args.model,
                          os.path.join(args.out, f'fig3_data_{args.monkey}_unit{args.unit}_{args.model}.pkl'))
