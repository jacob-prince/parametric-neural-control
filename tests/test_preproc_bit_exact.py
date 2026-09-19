"""preprocessed_data/ is a pure function of source_data/ and the scripts in scripts/preprocessing.

Two checks:
  * the shipped preprocessed_data matches tests/reference/preprocessed_data_hashes.json byte for byte;
  * (marker `preproc`, slow, needs source_data) scripts/preprocessing/build_all.py rebuilds the tree
    into a temporary directory and every file compares equal -- level 1: bytes; level 2: structured
    exact (dict keys in order, arrays equal incl. NaN positions, DataFrames exact, csv cell-wise,
    json minus the volatile created/git fields); level 3: allclose(rtol=1e-12). Level 1 is required
    in the reference environment for every file that is not a pickled DataFrame; the level reached is
    reported for each file.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

REFERENCE = REPO / 'tests' / 'reference' / 'preprocessed_data_hashes.json'
VOLATILE = {'MANIFEST.json': ('created', 'git'), 'BUILD_MANIFEST.json': None}


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def test_shipped_preprocessed_data_matches_reference_hashes(cache):
    if not REFERENCE.exists():
        pytest.skip('reference hashes not built yet')
    ref = json.loads(REFERENCE.read_text())['files']
    bad = [rel for rel, h in ref.items() if not (cache / rel).exists() or _sha(cache / rel) != h['sha256']]
    assert not bad, f'{len(bad)} file(s) differ from the reference: {bad[:10]}'


def _structured_equal(a, b, path='', exact=True):
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return False, f'{path}: key order/set differs'
        for k in a:
            ok, why = _structured_equal(a[k], b[k], f'{path}/{k}', exact)
            if not ok:
                return ok, why
        return True, ''
    if isinstance(a, pd.DataFrame) or isinstance(a, pd.Series):
        try:
            # check_dtype=False: a pickle written by an older pandas stores text columns as object
            # arrays, pandas 3 as string arrays; the values must still be identical.
            (pd.testing.assert_frame_equal if isinstance(a, pd.DataFrame) else pd.testing.assert_series_equal)(
                a, b, check_exact=exact, check_dtype=False, check_index_type=False, check_column_type=False)
            return True, ''
        except AssertionError as e:
            return False, f'{path}: {str(e)[:200]}'
    if isinstance(a, pd.api.extensions.ExtensionArray) or isinstance(b, pd.api.extensions.ExtensionArray):
        a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if isinstance(a, np.ndarray):
        b = np.asarray(b)
        if a.dtype.kind in 'OUS' and b.dtype.kind in 'OUS':
            a, b = a.astype(object), b.astype(object)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, f'{path}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}'
        if a.dtype.kind in 'fc':
            same = np.array_equal(a, b, equal_nan=True) if exact else np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        return (True, '') if same else (False, f'{path}: array values differ')
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f'{path}: length {len(a)} vs {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            ok, why = _structured_equal(x, y, f'{path}[{i}]', exact)
            if not ok:
                return ok, why
        return True, ''
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True, ''
        return bool(a == b if exact else np.isclose(a, b, rtol=1e-12, atol=0)), f'{path}: {a} vs {b}'
    if type(a) is not type(b) and not (isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))):
        return False, f'{path}: type {type(a).__name__} vs {type(b).__name__}'
    try:
        return bool(np.all(a == b)), f'{path}: {a!r:.80} vs {b!r:.80}'
    except Exception as e:  # objects without a sensible ==
        return False, f'{path}: cannot compare ({e})'


def compare_file(new, old):
    """Return the comparison level reached (1 bytes, 2 structured exact, 3 allclose) or 0."""
    if new.read_bytes() == old.read_bytes():
        return 1, ''
    if new.suffix == '.pkl':
        a, b = pickle.load(open(new, 'rb')), pickle.load(open(old, 'rb'))
        ok, why = _structured_equal(a, b)
        if ok:
            return 2, ''
        ok, why = _structured_equal(a, b, exact=False)
        return (3, '') if ok else (0, why)
    if new.suffix == '.csv':
        a, b = pd.read_csv(new, dtype=str, keep_default_na=False), pd.read_csv(old, dtype=str, keep_default_na=False)
        if a.equals(b):
            return 2, ''
        try:
            pd.testing.assert_frame_equal(pd.read_csv(new), pd.read_csv(old), check_exact=False, rtol=1e-12)
            return 3, ''
        except AssertionError as e:
            return 0, str(e)[:200]
    if new.suffix == '.json':
        a, b = json.loads(new.read_text()), json.loads(old.read_text())
        for k in VOLATILE.get(new.name) or ():
            a.pop(k, None); b.pop(k, None)
        return (2, '') if a == b else (0, 'json differs')
    if new.suffix == '.npz':
        a, b = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
        ok, why = _structured_equal({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
        return (2, '') if ok else (0, why)
    return 0, 'bytes differ (opaque format)'


@pytest.mark.preproc
@pytest.mark.slow
def test_build_all_regenerates_preprocessed_data(source, cache, tmp_path_factory, reference_env):
    out = tmp_path_factory.mktemp('preproc_build')
    r = subprocess.run([sys.executable, str(REPO / 'scripts' / 'preprocessing' / 'build_all.py'), '--out', str(out)],
                       cwd=str(REPO), text=True, capture_output=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    levels, failures = {}, []
    for new in sorted(p for p in out.rglob('*') if p.is_file() and not p.name.startswith('.') and '.build_logs' not in p.parts):
        rel = new.relative_to(out)
        if new.name in VOLATILE and VOLATILE[new.name] is None:
            continue
        old = cache / rel
        if not old.exists():
            failures.append(f'{rel}: not in shipped preprocessed_data'); continue
        level, why = compare_file(new, old)
        levels[str(rel)] = level
        if level == 0:
            failures.append(f'{rel}: {why}')
    print('comparison levels:', json.dumps(levels, indent=1))
    assert not failures, '\n'.join(failures)
    is_ref, _ = reference_env
    if is_ref:
        not_bytes = [rel for rel, lv in levels.items() if lv > 1 and not rel.endswith('MANIFEST.json')
                     and rel not in json.loads(REFERENCE.read_text()).get('known_pickle_only_semantic', [])]
        assert not not_bytes, f'expected byte identity in the reference env: {not_bytes}'
