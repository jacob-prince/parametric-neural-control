"""Execute the notebooks. Figure notebooks must reproduce the manuscript PNG (pixel hash in the
reference environment); demo notebooks must run end to end at their reduced default scale."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import REPO

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')

sys.path.insert(0, str(REPO / 'figures'))
import render_all  # noqa: E402

FIG_NBS = sorted((REPO / 'notebooks' / 'figures').glob('*.ipynb'))
DEMO_NBS = sorted((REPO / 'notebooks' / 'demos').glob('*.ipynb'))
ORACLE = json.loads((REPO / 'tests' / 'reference' / 'figure_pixel_hashes.json').read_text())['figures']


def _execute(path, out_dir, timeout=1800):
    nb = nbformat.read(path, as_version=4)
    env_backup = dict(os.environ)
    os.environ.update({k: v for k, v in render_all.render_env(out_dir).items() if k != 'PATH'})
    os.environ['PNC_OUTPUT'] = str(out_dir)
    try:
        client = nbclient.NotebookClient(nb, kernel_name=os.environ.get('PNC_KERNEL', 'python3'), timeout=timeout,
                                         resources={'metadata': {'path': str(REPO)}})
        client.execute()
    finally:
        os.environ.clear(); os.environ.update(env_backup)
    return nb


@pytest.mark.notebooks
@pytest.mark.pixel
@pytest.mark.parametrize('path', FIG_NBS, ids=[p.stem for p in FIG_NBS])
def test_figure_notebook_reproduces_manuscript_png(path, tmp_path_factory, cache, reference_env):
    out_dir = tmp_path_factory.mktemp('nb')
    _execute(path, out_dir)
    stem = re.sub(r'^\d\d_', '', path.stem)
    png = out_dir / 'figures' / f'{stem}.png'
    assert png.exists(), f'{path.name} did not write {png}'
    arr = np.asarray(Image.open(png).convert('RGBA'))
    is_ref, _ = reference_env
    if is_ref:
        assert hashlib.sha256(arr.tobytes()).hexdigest() == ORACLE[stem]['sha256']
    else:
        assert abs(arr.shape[0] - ORACLE[stem]['shape'][0]) <= 2 and abs(arr.shape[1] - ORACLE[stem]['shape'][1]) <= 2


@pytest.mark.notebooks
@pytest.mark.slow
@pytest.mark.parametrize('path', DEMO_NBS, ids=[p.stem for p in DEMO_NBS])
def test_demo_notebook_runs(path, tmp_path_factory, source, cache):
    _execute(path, tmp_path_factory.mktemp('demo'), timeout=3600)
