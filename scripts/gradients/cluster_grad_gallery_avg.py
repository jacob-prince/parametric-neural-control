#!/usr/bin/env python3
r"""Cluster stage for Supp Fig S35: average input-gradient saliency per (seed, model) across all 25
sites. Reads Binxu's precomputed grad_img (10 seeds x 3 x 224 x 224) for each of the 250 site-models,
converts each seed-gradient to an L2 saliency map (magnitude over colour channels), normalizes per map
(by its 99th percentile), and averages over the 25 sites for each (model, seed). Writes a compact
gradient_gallery_avg.pkl -> {models, seeds, mean_saliency[model] = (10, 224, 224), n_sites[model]}.
Pure numpy; runs on a shared CPU node.
"""
import os
import pickle
import argparse
import numpy as np

ENC = os.environ.get('PNC_ENCODING_MODEL_ROOT', '/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models')  # was '/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models'
SUB = 'posthoc_model_predict/encoding_gradient_map_fourier_spectra'
# take8 canonical mdir + units per monkey (matches preproc.loader.load_grad_maps)
MD = {'red': ('red_20250428-20250430', [0, 2, 9, 15, 19]),
      'paul': ('paul_20250428-20250430', [0, 8, 24, 40, 47]),
      'venus': ('venus_250426-250429', [9, 79, 151, 331, 355]),
      'leap': ('leap_250426-250501', [81, 282, 286, 306, 342]),
      'three0': ('three0_250426-250501', [56, 74, 120, 168, 204])}
MODELS = ['AlexNet_training_seed_01', 'siglip2_vitb16', 'resnet50_clip', 'regnety_640',
          'dinov2_vitb14_reg', 'radio_v2.5-b', 'resnet50', 'resnet50_dino', 'resnet50_robust', 'clipag_vitb32']


def saliency(grad_img):
    """(n_seed,3,H,W) -> (n_seed,H,W) L2 magnitude, each map normalized by its 99th pct."""
    mag = np.sqrt((grad_img.astype(np.float64) ** 2).sum(axis=1))          # (n_seed,H,W)
    out = np.empty_like(mag)
    for i in range(mag.shape[0]):
        hi = np.percentile(mag[i], 99.0)
        out[i] = np.clip(mag[i] / hi, 0, 1) if hi > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    acc = {m: None for m in MODELS}; cnt = {m: 0 for m in MODELS}; missing = []
    for mk, (md, units) in MD.items():
        for u in units:
            for m in MODELS:
                p = os.path.join(ENC, md, SUB, f'{md}_unit_{u}_model_{m}_grad_maps_freq_profiles.pkl')
                if not os.path.exists(p):
                    missing.append(f'{mk}/{u}/{m}'); continue
                gi = pickle.load(open(p, 'rb'))['grad_img']                 # (10,3,224,224)
                s = saliency(np.asarray(gi))                                # (10,224,224)
                acc[m] = s if acc[m] is None else acc[m] + s
                cnt[m] += 1
    mean = {m: (acc[m] / cnt[m]).astype(np.float32) for m in MODELS if cnt[m] > 0}
    pickle.dump(dict(models=MODELS, mean_saliency=mean, n_sites={m: cnt[m] for m in MODELS},
                     seed_order='take8 SEED_IMAGES index 0-9', img_size=224),
                open(a.out, 'wb'))
    print('sites averaged per model:', {m: cnt[m] for m in MODELS})
    if missing:
        print('MISSING:', len(missing), missing[:10])
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
