"""Every manuscript figure re-renders from preprocessed_data and matches the submitted PNG.

Reference environment (see conftest.reference_env_status): the RGBA pixel array of the trimmed
render must hash-equal tests/reference/figure_pixel_hashes.json. Elsewhere: same size within
2 px, fewer than 1% of pixels differ and mean |delta| < 0.5 over the common region; a diff image
is written to <PNC_OUTPUT>/pixel_diffs/ on failure so font substitutions can be inspected.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import REPO

sys.path.insert(0, str(REPO / 'figures'))
import render_all  # noqa: E402

ORACLE = json.loads((REPO / 'tests' / 'reference' / 'figure_pixel_hashes.json').read_text())['figures']
ITEMS = render_all.registry()
assert [it[0] for it in ITEMS] == list(ORACLE), 'render registry and oracle disagree'


@pytest.fixture(scope='module')
def out_dir(tmp_path_factory, cache):
    return tmp_path_factory.mktemp('figures')


def _rgba(p):
    return np.asarray(Image.open(p).convert('RGBA'))


@pytest.mark.pixel
@pytest.mark.parametrize('stem,module,extra', ITEMS, ids=[it[0] for it in ITEMS])
def test_figure_matches_manuscript(stem, module, extra, out_dir, reference_env, paths):
    rec = render_all.render(stem, module, extra, out_dir, render_all.render_env(out_dir))
    assert rec['returncode'] == 0, (out_dir / 'logs' / f'{stem}.log').read_text()[-4000:]
    ref = ORACLE[stem]
    is_ref, details = reference_env
    if is_ref:
        assert rec['shape'] == ref['shape'], f'{stem}: shape {rec["shape"]} vs manuscript {ref["shape"]}'
        assert rec['sha256'] == ref['sha256'], f'{stem}: pixels differ from the manuscript PNG'
        return
    # tolerance mode (non-reference fonts / library versions)
    a = _rgba(out_dir / f'{stem}.png')
    assert abs(a.shape[0] - ref['shape'][0]) <= 2 and abs(a.shape[1] - ref['shape'][1]) <= 2, \
        f'{stem}: shape {a.shape} vs manuscript {ref["shape"]} (tolerance mode)'
    manuscript = paths['tests'] / 'reference' / 'manuscript_png' / f'{stem}.png'
    if not manuscript.exists():
        pytest.skip('tolerance mode needs the manuscript PNGs (python data/download_data.py --tier preprocessed fetches them)')
    b = _rgba(manuscript)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    d = np.abs(a[:h, :w].astype(np.int16) - b[:h, :w].astype(np.int16))
    frac = float((d.max(axis=2) > 0).mean()); mean = float(d.mean())
    if frac >= 0.01 or mean >= 0.5:
        diff_dir = paths['output'] / 'pixel_diffs'
        diff_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(d.max(axis=2) * 4, 0, 255).astype(np.uint8)).save(diff_dir / f'{stem}.png')
    assert frac < 0.01 and mean < 0.5, f'{stem}: {frac:.2%} pixels differ, mean |delta| {mean:.3f} (tolerance mode)'


@pytest.mark.pixel
def test_untrimmed_render_trims_to_the_trimmed_output(out_dir, cache):
    """save_fig(do_trim=False) followed by pnc.trim.trim gives the same pixels as the default."""
    from pnc import trim
    stem, module, extra = ITEMS[3]  # divergence: fast, tight-bbox
    rec = render_all.render(stem, module, extra, out_dir, render_all.render_env(out_dir))
    assert rec['returncode'] == 0
    raw_dir = out_dir / 'raw'
    env = render_all.render_env(raw_dir)
    r = subprocess.run([sys.executable, '-m', module, '--out', str(raw_dir), '--no-trim', *extra], cwd=str(REPO),
                       env=env, text=True, capture_output=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    trim.trim(raw_dir / f'{stem}.png')
    assert np.array_equal(_rgba(raw_dir / f'{stem}.png'), _rgba(out_dir / f'{stem}.png'))
