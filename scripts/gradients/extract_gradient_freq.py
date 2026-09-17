#!/usr/bin/env python3
"""Aggregate input-output gradient radial-Fourier profiles per (site, model) — Fig 7 flatness.

Reads Binxu's precomputed per-(site,model) gradient freq-profile pkls on the cluster
(250 = 5 monkeys x 5 units x 10 models), takes the per-seed mean/std profile, and computes a
natural-image reference radial profile from the 969 encoding stimuli. Pure numpy/PIL — runs on
a cluster login node (no GPU). Self-contained within cluster/.

Output: gradient_freq.pkl with schema:
  gradients [{monkey,unit,model,profile_mean,profile_std,n_seeds}], freqs, bincounts,
  natural_profile, natural_profiles_per_image, img_size.

  python extract_gradient_freq.py --enc-models <cluster Encoding_models> --stim <encstim dir> --out <path>
"""
import os
import glob
import pickle
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MONKEY_DIRS = {'red': 'red_20250428-20250430', 'paul': 'paul_20250428-20250430',
               'venus': 'venus_250426-250429', 'leap': 'leap_250426-250501',
               'three0': 'three0_250426-250501'}
CHANNELS = {'red': [0, 2, 9, 15, 19], 'paul': [0, 8, 24, 40, 47],
            'venus': [9, 79, 151, 331, 355], 'leap': [81, 282, 286, 306, 342],
            'three0': [56, 74, 120, 168, 204]}
MODELS = ['AlexNet_training_seed_01', 'siglip2_vitb16', 'resnet50_clip', 'regnety_640',
          'dinov2_vitb14_reg', 'radio_v2.5-b', 'resnet50', 'resnet50_dino',
          'resnet50_robust', 'clipag_vitb32']
IMG = 224


def _radial(power2d):
    h, w = power2d.shape; cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.int64)
    return np.bincount(R.ravel(), power2d.ravel()) / np.maximum(np.bincount(R.ravel()), 1)


def _img_power(arr):
    x = arr.astype(np.float32)
    if x.ndim == 3:
        x = x.mean(-1)
    x = x - x.mean()
    return _radial(np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc-models', required=True)   # cluster Encoding_models root
    ap.add_argument('--stim', required=True)          # encoding stimuli dir (969 images)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    records, freqs, bincounts = [], None, None
    for mk, mdir in MONKEY_DIRS.items():
        gdir = Path(a.enc_models) / mdir / 'posthoc_model_predict' / 'encoding_gradient_map_fourier_spectra'
        n = 0
        for unit in CHANNELS[mk]:
            for model in MODELS:
                p = gdir / f'{mdir}_unit_{unit}_model_{model}_grad_maps_freq_profiles.pkl'
                if not p.exists():
                    print(f'MISSING {p.name}'); continue
                d = pickle.load(open(p, 'rb')); prof = np.asarray(d['profiles'])
                if freqs is None:
                    freqs = np.asarray(d['freqs']); bincounts = np.asarray(d['bincounts'])
                records.append(dict(monkey=mk, unit=unit, model=model,
                                    profile_mean=prof.mean(0), profile_std=prof.std(0),
                                    n_seeds=int(prof.shape[0])))
                n += 1
        print(f'  {mk}: {n} records')

    imgs = sorted(p for p in Path(a.stim).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    print(f'natural ref from {len(imgs)} images...')
    nat = []
    for i, p in enumerate(imgs):
        try:
            arr = np.asarray(Image.open(p).convert('RGB').resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0
        except Exception:
            continue
        pr = _img_power(arr)
        pr = pr[:len(freqs)] if len(pr) >= len(freqs) else np.pad(pr, (0, len(freqs) - len(pr)))
        nat.append(pr)
    nat = np.stack(nat, 0)
    pickle.dump(dict(gradients=records, freqs=freqs, bincounts=bincounts,
                     natural_profile=nat.mean(0), natural_profiles_per_image=nat, img_size=IMG),
                open(a.out, 'wb'), protocol=4)
    print(f'wrote {a.out}: {len(records)} records, natural {nat.shape}')


if __name__ == '__main__':
    main()
