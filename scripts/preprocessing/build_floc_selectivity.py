#!/usr/bin/env python3
"""Build the fLoc category-selectivity cache for the floc_selectivity supp figure.

For each of the 25 target sites, loads encoding-session responses to the 61 fLoc
stimuli (tagged by category domain) and to the natural (non-fLoc) calibration
images, and caches them to ``preproc_data/floc_selectivity.pkl``. The floc supp figure then
computes KDEs / selectivity stats and draws - no HDF5 touched at plot time.

Mirrors analyses/10_fLoc_analysis (domain map, natural filter, file selection).
Torch-free (encoding HDF5 via h5py). Run once:
    python build_floc_selectivity.py
"""
import os
import pickle
import re
from pathlib import Path

import h5py
import numpy as np

from pnc import paths

DATA = Path(paths.source_data())          # was /Volumes/CORSAIR/AccentuateVVS/data
ENC_DIR = DATA / 'brain_data_encoding'
OUT = os.path.join(str(paths.preprocessed_data()), 'floc_selectivity.pkl')

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
TARGET_CHANNELS = {
    'red':    [0, 2, 9, 15, 19],
    'paul':   [0, 8, 24, 40, 47],
    'venus':  [9, 79, 151, 331, 355],
    'leap':   [81, 282, 286, 306, 342],
    'three0': [56, 74, 120, 168, 204],
}
DOMAIN_MAP = {
    'adult': 'Faces', 'child': 'Faces', 'body': 'Bodies', 'limb': 'Bodies',
    'car': 'Objects', 'instrument': 'Objects', 'corridor': 'Scenes',
    'house': 'Scenes', 'number': 'Characters', 'word': 'Characters',
    'scrambled': 'Scrambled',
}


def parse_floc(sn):
    m = re.match(r'fLoc_subset_\d+_([a-z]+)-(\d+)\.jpg', sn)
    return m.group(1) if m else None


def is_natural(sn):
    if 'fLoc' in sn:
        return False
    if 'level' in sn and 'score' in sn:
        return False
    return True


def enc_file(monkey):
    cands = sorted(ENC_DIR.glob(f'{monkey}*_vvs-encodingstimuli_*.h5'))
    return cands[0] if cands else None


def main():
    out = {}
    for mk in MONKEYS:
        fpath = enc_file(mk)
        chans = TARGET_CHANNELS[mk]
        with h5py.File(fpath, 'r') as f:
            names = [s.decode() if isinstance(s, bytes) else s
                     for s in f['repavg/stimulus_name'][()]]
            resp = f['repavg/response_peak'][()]
        floc_idx, floc_dom = [], []
        for i, sn in enumerate(names):
            cat = parse_floc(sn)
            if cat is not None:
                floc_idx.append(i)
                floc_dom.append(DOMAIN_MAP[cat])
        nat_idx = [i for i, sn in enumerate(names) if is_natural(sn)]
        out[mk] = {
            'channels': chans,
            'floc_resp': resp[np.array(floc_idx)][:, chans],   # (n_floc, 5)
            'floc_domains': np.array(floc_dom),                # (n_floc,)
            'nat_resp': resp[np.array(nat_idx)][:, chans],     # (n_nat, 5)
            'enc_file': fpath.name,
        }
        print(f'  {mk:7s} {fpath.name}  fLoc={len(floc_idx)}  nat={len(nat_idx)}')

    with open(OUT, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
