"""Fold synced per-model ImageNet-prediction pkls into one cache.

Reads outputs_from_cluster/fig6_imagenet_out/imagenet_pred_<model>.pkl (full runs, no _lim)
for all models and writes preproc_data/imagenet_predictions.pkl: pred[image, site, model]
+ site/model metadata. Run after `sync.sh pull` once the full extraction completes. psn env.
"""
import os, glob, pickle
import numpy as np

from pnc import paths

OUTD = os.path.join(str(paths.cluster_outputs()), 'fig6_imagenet_out')
CACHE = os.path.join(str(paths.preprocessed_data()), 'imagenet_predictions.pkl')
MODELS = ['AlexNet_training_seed_01', 'siglip2_vitb16', 'resnet50_clip', 'regnety_640',
          'dinov2_vitb14_reg', 'radio_v2.5-b', 'resnet50', 'resnet50_dino',
          'resnet50_robust', 'clipag_vitb32']


def main():
    mats, have = [], []
    ref = None
    for m in MODELS:
        p = os.path.join(OUTD, f'imagenet_pred_{m}.pkl')
        if not os.path.exists(p):
            print(f'  MISSING full run: {m}'); continue
        d = pickle.load(open(p, 'rb'))
        mats.append(d['imagenet_predictions'].astype(np.float32)); have.append(m)
        ref = d
    if len(have) != len(MODELS):
        print(f'  only {len(have)}/{len(MODELS)} models present - not writing cache yet'); return
    pred = np.stack(mats, axis=-1)                                   # (50000, 25, 10)
    out = dict(imagenet_predictions=pred, models=np.array(have, dtype=object),
               unit_monkeys=ref['unit_monkeys'], unit_ids=ref['unit_ids'])
    pickle.dump(out, open(CACHE, 'wb'))
    print(f'  wrote {CACHE}: pred {pred.shape} (image x site x model)')


if __name__ == '__main__':
    main()
