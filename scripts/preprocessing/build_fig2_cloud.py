#!/usr/bin/env python3
"""Build preprocessed_data/fig2_cloud.pkl - the Fig 2 panel D 3D accentuation-dataset cloud.

Each (monkey, unit, model) is aligned into a unified space (residual PC1, residual
PC2, predicted z along the encoding axis), z-scored to that combo's encoding stats
so every combo contributes equally, then pooled. The 80 panel-E gallery stimuli
(5 sites x 4 models x 4 levels) are tagged for highlighting.

Reads the raw encoding/accentuated PCA-projection pickles via pnc.utils (these contain
torch tensors, so this must run in an environment with torch installed) and writes
<out>/fig2_cloud.pkl. The figure script (figures/main/fig2_accentuation.py) only reads it.

    python scripts/preprocessing/build_fig2_cloud.py [--out DIR]
"""
import argparse
import os
import pickle
import warnings

import numpy as np

from pnc import paths
from pnc.utils import (load_encoding_pickle, load_accentuated_pickle, to_numpy,
                       MONKEY_UNITS, ALL_MODELS)

# panel-E gallery stimuli to highlight (must match fig2_accentuation.py GRID_SITES / MODELS / LEVEL_PICK)
PANEL_E_SITES = {('red', 19): 1, ('paul', 8): 2, ('venus', 331): 4,
                 ('venus', 151): 8, ('three0', 120): 5}
PANEL_E_MODELS = ['resnet50', 'dinov2_vitb14_reg', 'resnet50_robust', 'clipag_vitb32']
PANEL_E_LEVEL_IDX = [0, 1, 7, 10]
ENC_SUBSAMPLE = 20


def align_combo_3d(monkey, unit, model):
    """Align (monkey,unit,model) into (residual PC1, residual PC2, predicted z),
    z-scored to encoding stats. Returns enc_xyz, trajectories, levels, seeds."""
    try:
        enc = load_encoding_pickle(monkey, unit, model)
        acc = load_accentuated_pickle(monkey, unit, model)
    except FileNotFoundError:
        return None, [], [], []
    enc_pca = to_numpy(enc['PCA_resp'])
    readout_vec = to_numpy(enc['readout_vec']).flatten()
    readout_bias = float(to_numpy(enc['readout_bias']))
    acc_pca = to_numpy(acc['PCA_resp'])
    acc_df = acc['df']

    enc_z = enc_pca @ readout_vec + readout_bias
    acc_z = acc_pca @ readout_vec + readout_bias
    r_hat = readout_vec / np.linalg.norm(readout_vec)
    enc_resid = enc_pca - np.outer(enc_pca @ r_hat, r_hat)
    _, _, Vt = np.linalg.svd(enc_resid, full_matrices=False)
    v1, v2 = Vt[0], Vt[1]
    enc_x, enc_y = enc_resid @ v1, enc_resid @ v2
    acc_resid = acc_pca - np.outer(acc_pca @ r_hat, r_hat)
    acc_x, acc_y = acc_resid @ v1, acc_resid @ v2

    x_mu, x_sd = enc_x.mean(), enc_x.std()
    y_mu, y_sd = enc_y.mean(), enc_y.std()
    z_mu, z_sd = enc_z.mean(), enc_z.std()
    if x_sd < 1e-10 or y_sd < 1e-10 or z_sd < 1e-10:
        return None, [], [], []
    enc_x = (enc_x - x_mu) / x_sd; enc_y = (enc_y - y_mu) / y_sd; enc_z = (enc_z - z_mu) / z_sd
    acc_x = (acc_x - x_mu) / x_sd; acc_y = (acc_y - y_mu) / y_sd; acc_z = (acc_z - z_mu) / z_sd
    enc_xyz = np.column_stack([enc_x, enc_y, enc_z])

    mask = (acc_df['model_name'] == model) & (acc_df['unit_id'] == unit)
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return enc_xyz, [], [], []
    sub_df = acc_df.iloc[idx]
    sub_x, sub_y, sub_z = acc_x[idx], acc_y[idx], acc_z[idx]
    trajectories, traj_levels, traj_seeds = [], [], []
    for img_id in sorted(sub_df['img_id'].unique()):
        smask = sub_df['img_id'].values == img_id
        lv = sub_df['level'].values[smask]; order = np.argsort(lv)
        trajectories.append(np.column_stack([sub_x[smask][order], sub_y[smask][order], sub_z[smask][order]]))
        traj_levels.append(lv[order]); traj_seeds.append(int(img_id))
    return enc_xyz, trajectories, traj_levels, traj_seeds


def build_cloud(cache):
    enc_parts, all_trajectories, all_traj_levels = [], [], []
    gallery_points, gallery_levels, gallery_trajectories, gallery_traj_levels = [], [], [], []
    n_combos = 0
    for monkey in MONKEY_UNITS:
        for unit in MONKEY_UNITS[monkey]:
            for model in ALL_MODELS:
                ex, trajs, tlevels, tseeds = align_combo_3d(monkey, unit, model)
                if ex is None:
                    continue
                enc_parts.append(ex); n_combos += 1
                all_trajectories.extend(trajs); all_traj_levels.extend(tlevels)
                if (monkey, unit) in PANEL_E_SITES and model in PANEL_E_MODELS:
                    tgt = PANEL_E_SITES[(monkey, unit)]
                    for traj, levels, seed in zip(trajs, tlevels, tseeds):
                        if seed != tgt:
                            continue
                        gallery_trajectories.append(traj); gallery_traj_levels.append(levels)
                        for li in PANEL_E_LEVEL_IDX:
                            if li < len(traj):
                                gallery_points.append(traj[li]); gallery_levels.append(levels[li])
    enc_xyz = np.vstack(enc_parts)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(enc_xyz), size=len(enc_xyz) // ENC_SUBSAMPLE, replace=False)
    out = dict(enc=enc_xyz[sub],
               gallery=(np.vstack(gallery_points) if gallery_points else np.zeros((0, 3))),
               gallery_levels=np.array(gallery_levels, float),
               gallery_trajectories=gallery_trajectories, gallery_traj_levels=gallery_traj_levels,
               trajectories=all_trajectories, traj_levels=all_traj_levels)
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(out, f)
    print(f'combos={n_combos} enc={len(enc_xyz)}->{len(out["enc"])} '
          f'trajectories={len(all_trajectories)} gallery={len(out["gallery"])} (expect 80)')
    print(f'cached -> {cache}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.preprocessed_data()),
                    help='directory that receives fig2_cloud.pkl (default: preprocessed_data)')
    args = ap.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    build_cloud(os.path.join(args.out, 'fig2_cloud.pkl'))
