"""Stimulus-name parsing and classification.

Three kinds appear in the control sessions:
  calibration  - encoding/anchor stimuli re-shown in control (no score in the name)
  accentuated  - synthesized sweeps: '<model>_RidgeCV_..._<unit>_..._<seed>_..._<score>.png'
  controversial- 'controversial_max_r50_..._unit_<u>_img_<i>_srobust_<a>_sr50_<b>.png'
"""
import re


def parse_accentuated(sn):
    if 'score' not in sn or '_RidgeCV' not in sn:
        return None
    try:
        p = sn.split('_RidgeCV')[1].split('_')
        return dict(model=sn.split('_RidgeCV')[0], unit=int(p[2]), seed=int(p[4]),
                    target=float(p[6]), score=float(p[8].replace('.png', '')))
    except (IndexError, ValueError):
        return None


_CONTR = re.compile(r'unit_(\d+)_img_(\d+)_srobust_(-?\d+\.?\d*)_sr50_(-?\d+\.?\d*)')


def parse_controversial(sn):
    if not sn.startswith('controversial'):
        return None
    m = _CONTR.search(sn)
    if not m:
        return None
    return dict(unit=int(m.group(1)), img=int(m.group(2)),
                score_robust=float(m.group(3)), score_r50=float(m.group(4)))


def classify(sn, anchor_set):
    if sn.startswith('controversial'):
        return 'controversial'
    if '_RidgeCV' in sn and 'score' in sn:
        return 'accentuated'
    if sn in anchor_set:
        return 'calibration'
    return 'other'
