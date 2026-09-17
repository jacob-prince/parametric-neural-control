#!/usr/bin/env python3
"""Fig 2 panel D data — the 3D accentuation-dataset cloud (read-only).

The cloud is a preprocessed cache, preprocessed_data/fig2_cloud.pkl, built by
scripts/preprocessing/build_fig2_cloud.py (each (monkey, unit, model) aligned into a
unified space — residual PC1, residual PC2, predicted z along the encoding axis — z-scored
to that combo's encoding stats and pooled; the 80 panel-E gallery stimuli tagged).
This module only loads it.
"""
import pickle

from pnc import paths

CACHE = paths.preprocessed_data() / 'fig2_cloud.pkl'


def load_cloud():
    path = paths.require(CACHE, hint='python scripts/preprocessing/build_fig2_cloud.py')
    return pickle.load(open(path, 'rb'))
