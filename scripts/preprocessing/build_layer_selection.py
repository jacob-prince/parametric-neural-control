#!/usr/bin/env python3
"""Build the encoding-layer-selection cache for the layer_selection supp figure (one-time engine).

For every (monkey, model, target site) it reads the cross-validated per-unit test
R^2 across all candidate layers (the RidgeCV layer sweep, pca750 dim-reduction) and
records the full layer curve, the selected peak layer (argmax test R^2 = exactly
the production selection rule, verified against the fitted .pt filenames), and the
peak's relative depth in the model.

Source data (synced from a GPU cluster,
/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subj}/model_outputs_pca4all/
*_pred_meta.pkl -> D2_per_unit_test_dict), extracted per-unit into
data/model_features/_layer_scores/. Torch-free. Run once:
    python build_layer_selection.py
"""
import os
import pickle
import glob

import h5py
import numpy as np

from pnc import paths

DATA = str(paths.source_data())          # was /Volumes/CORSAIR/AccentuateVVS/data
LS = os.path.join(DATA, 'model_features', '_layer_scores')
ENC = os.path.join(DATA, 'brain_data_encoding')
OUT = os.path.join(str(paths.preprocessed_data()), 'layer_selection.pkl')

SUBJ = {'red': 'red_20250428-20250430', 'paul': 'paul_20250428-20250430',
        'venus': 'venus_250426-250429', 'leap': 'leap_250426-250501',
        'three0': 'three0_250426-250501'}
TARGET = {'red': [0, 2, 9, 15, 19], 'paul': [0, 8, 24, 40, 47],
          'venus': [9, 79, 151, 331, 355], 'leap': [81, 282, 286, 306, 342],
          'three0': [56, 74, 120, 168, 204]}
MODELS = ['AlexNet_training_seed_01', 'siglip2_vitb16', 'resnet50_clip',
          'regnety_640', 'dinov2_vitb14_reg', 'radio_v2.5-b', 'resnet50',
          'resnet50_dino', 'resnet50_robust', 'clipag_vitb32']
DIMRED = 'pca750'
# ventral-stream posterior->anterior ordinal (from recorded per-electrode area);
# STS is a higher-order region off the ventral chain -> no ordinal.
VENTRAL_ORD = {'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'pIT': 5, 'cIT': 6, 'aIT': 7}


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_areas():
    """Per-target-channel recorded brain area (canonical) + channel depth, and the
    full per-probe (depth, area) profile for Neuropixels probes, from the encoding
    HDF5. Each monkey's target channels index a single probe (red/paul plexon FMA,
    venus/leap/three0 Neuropixels); pick the array whose length == n_channels."""
    areas, depths, profiles = {}, {}, {}
    for mk, subj in SUBJ.items():
        f = sorted(glob.glob(os.path.join(ENC, f'{mk}*_vvs-encodingstimuli_*.h5')))[0]
        with h5py.File(f, 'r') as h:
            nm = h['neuron_metadata']
            nchan = h['repavg']['response_peak'].shape[1]
            ba = nm['brain_area']
            if isinstance(ba, h5py.Group):
                arr = next(np.array(ba[k]) for k in ba.keys() if len(ba[k]) == nchan)
            else:
                arr = np.array(ba)
            areas_all = np.array([_dec(x).replace('l_', '') for x in arr])
            depth = np.array(nm['channel_depth']) if 'channel_depth' in nm else None
            for ch in TARGET[mk]:
                areas[(mk, ch)] = areas_all[ch]
                depths[(mk, ch)] = float(depth[ch]) if depth is not None else np.nan
            if depth is not None:  # Neuropixels probe - keep full depth/area profile
                profiles[mk] = {'depth': depth.astype(float), 'area': areas_all}
    return areas, depths, profiles


def ordered_layers(d2test):
    """Layers for DIMRED, in sweep (architecture) order."""
    seen = []
    for (layer, dimred), reg in d2test.keys():
        if dimred == DIMRED and layer not in seen:
            seen.append(layer)
    return seen


def main():
    areas, depths, profiles = load_areas()
    out = {'_probe_profiles': profiles}
    for mk, subj in SUBJ.items():
        for model in MODELS:
            fs = sorted(glob.glob(os.path.join(
                LS, f'{subj}__*_{model}_sweep_regressors_layers_D2perunit.pkl')))
            if not fs:
                print(f'  MISSING {mk} {model}')
                continue
            d2 = pickle.load(open(fs[0], 'rb'))['D2_per_unit_test_dict']
            layers = ordered_layers(d2)
            nL = len(layers)
            reldepth = np.arange(nL) / (nL - 1)
            M = np.array([d2[((L, DIMRED), 'RidgeCV')] for L in layers])  # (nL,64)
            for ch in TARGET[mk]:
                curve = M[:, ch].astype(float)
                pk = int(np.argmax(curve))
                area = areas[(mk, ch)]
                out[(mk, model, ch)] = {
                    'layers': layers, 'reldepth': reldepth, 'curve': curve,
                    'peak_idx': pk, 'peak_reldepth': float(reldepth[pk]),
                    'peak_layer': layers[pk], 'peak_r2': float(curve[pk]),
                    'n_layers': nL, 'area': area, 'depth': depths[(mk, ch)],
                    'ventral_ord': VENTRAL_ORD.get(area, np.nan),
                }
    with open(OUT, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved {len(out)} (monkey,model,site) layer curves -> {OUT}')
    # sanity summary
    for model in MODELS:
        rds = [v['peak_reldepth'] for k, v in out.items() if k[1] == model]
        print(f'  {model:26s} peak rel-depth median={np.median(rds):.2f} '
              f'[{min(rds):.2f},{max(rds):.2f}]  n={len(rds)}')


if __name__ == '__main__':
    main()
