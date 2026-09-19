#!/usr/bin/env python3
"""Build the channel-selection cache for the reliability supp figure
(preprocessed_data/sup_reliability_channel_selection_cache.pkl).

Per monkey, from the raw encoding HDF5 (source_data/brain_data_encoding):
  rel  Spearman-Brown-corrected split-half reliability (2r/(1+r)) of EVERY channel on the
       array/probe (the all-channel grey cloud of panel a; not in the preproc cache),
  sel  the five targeted units (utils.MONKEY_UNITS),
  tc   5x5 Pearson tuning-correlation matrix of the targeted units over the encoding images.

Torch-free; runs in seconds. Same logic as the former `--recompute` path of the take9
sup_reliability.py script.

    python scripts/preprocessing/build_sup_reliability_cache.py [--out DIR]
"""
import os
import argparse
import pickle

import numpy as np
import h5py

from pnc import paths
from pnc.utils import MONKEY_UNITS, ENCODING_HDF5_DIR, ENCODING_HDF5_FILES, load_encoding_hdf5

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
CACHE_NAME = 'sup_reliability_channel_selection_cache.pkl'


def spearman_brown(r):
    r = np.asarray(r, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = 2 * r / (1 + r)
    sb[~np.isfinite(sb) | (r <= -0.99)] = np.nan
    return sb


def compute_channel_selection():
    out = {}
    for m in MONKEYS:
        with h5py.File(os.path.join(ENCODING_HDF5_DIR, ENCODING_HDF5_FILES[m]), 'r') as fe:
            raw_rel = np.array(fe['neuron_metadata']['reliability'], float)
        _, enc_resp, _, _ = load_encoding_hdf5(m)
        nchan = enc_resp.shape[1]
        rel = spearman_brown(raw_rel[:nchan])
        sel = list(MONKEY_UNITS[m])
        tuning = np.column_stack([enc_resp[:, c] for c in sel])   # (n_stim, 5)
        out[m] = dict(rel=rel, sel=sel, tc=np.corrcoef(tuning.T))
    return out


def main(out):
    d = compute_channel_selection()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, CACHE_NAME)
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    print(f'Cached -> {path}')
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(paths.preprocessed_data()),
                    help='directory for the cache (default: PNC_PREPROCESSED_DATA)')
    main(**vars(ap.parse_args()))
