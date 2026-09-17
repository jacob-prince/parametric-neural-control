#!/usr/bin/env python3
"""Figure 6 cache readers (read-only).

The three datasets Fig 6 consumes are preprocessed caches built by
scripts/preprocessing/build_fig6_caches.py (which reads only pnc.preproc.loader):

  load_separation() -> preprocessed_data/fig6_separation.pkl   (Q1 model separation)
  load_efficiency() -> preprocessed_data/fig6_efficiency.pkl    (Q2 natural-vs-accentuated)
  load_pair(a, b, ch) -> preprocessed_data/fig6_pair_<A>__<B>__ch<ch>.pkl  (worked example)

THUMB_DIR_INET (preprocessed_data/fig6_imagenet_thumbs) is a frozen input: the ImageNet
validation thumbnails the worked-example panels paste onto the prediction plane.
"""
import os
import pickle

from pnc import paths

PREPROC_DATA = str(paths.preprocessed_data())
THUMB_DIR_INET = os.path.join(PREPROC_DATA, 'fig6_imagenet_thumbs')
_HINT = 'python scripts/preprocessing/build_fig6_caches.py'


def _read(cache):
    with open(paths.require(cache, hint=_HINT), 'rb') as f:
        return pickle.load(f)


def load_separation():
    return _read(os.path.join(PREPROC_DATA, 'fig6_separation.pkl'))


def load_efficiency():
    return _read(os.path.join(PREPROC_DATA, 'fig6_efficiency.pkl'))


def load_pair(model_a, model_b, ch=2):
    return _read(os.path.join(PREPROC_DATA, f'fig6_pair_{model_a}__{model_b}__ch{ch}.pkl'))
