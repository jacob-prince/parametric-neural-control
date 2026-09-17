"""Encoding-model extraction: per (unit, model) prediction over every stimulus, plus the
linear readout and its config. Features (PCA_resp) are left in the source pickles -- only
the fitted model + its per-stimulus predictions are cached."""
import numpy as np

from pnc.utils import load_encoding_pickle

STIM_FLAGS = ['is_train', 'is_test', 'is_normalizer', 'is_nsd', 'is_floc', 'is_OO']


def unit_model(monkey, unit, model):
    """{stimulus_name: prediction}, readout vec/bias, layer, fit method for one encoding model."""
    ed = load_encoding_pickle(monkey, unit, model)
    names = np.asarray(ed['df']['stimulus_name'].values, dtype=object)
    pred = np.asarray(ed['target_unit_resp'], np.float32).ravel()
    cfg = ed['config']
    return dict(pred=dict(zip(names.tolist(), pred.tolist())),
                readout_vec=np.asarray(ed['readout_vec'], np.float32),
                readout_bias=float(np.ravel(ed['readout_bias'])[0]),
                layer=cfg.get('layer_name'), fit_method=cfg.get('fit_method_name'))


def _flag_to_bool(values):
    """Parse a per-stimulus flag column. The upstream pickles mix real bools with the
    STRINGS 'True'/'False'/'1'/'0' (dtype=object); a plain .astype(bool) marks every
    non-empty string True, which silently degraded is_test/is_normalizer to all-True."""
    arr = np.asarray(values, dtype=object)
    out = np.empty(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ('true', '1', '1.0'):
                out[i] = True
            elif s in ('false', '0', '0.0'):
                out[i] = False
            else:
                raise ValueError(f'unparseable stimulus flag {v!r}')
        else:
            out[i] = bool(v)
    return out


def stim_metadata(monkey, unit, model):
    """Per-stimulus flag table for a monkey (identical across units/models); plain numpy."""
    ed = load_encoding_pickle(monkey, unit, model)
    df = ed['df']
    out = {'stimulus_name': np.asarray(df['stimulus_name'].values, dtype=object)}
    for c in STIM_FLAGS:
        out[c] = _flag_to_bool(df[c].values) if c in df.columns else None
    out['image_fp'] = (np.asarray(df['image_fps'].values, dtype=object)
                       if 'image_fps' in df.columns else None)
    return out
