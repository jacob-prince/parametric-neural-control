"""Non-default figure variants (env-var switches in the original pipeline, now keyword arguments)
still run end to end. Only the defaults are pixel-checked; these variants are not in the paper."""
import subprocess
import sys

import pytest

from conftest import REPO

VARIANTS = [
    ('figures.main.fig3_single_site', ['--p2set', 'allcal']),   # other sites need build_fig3_embedding.py first
    ('figures.main.fig5_predictors', ['--outcome', 'r']),
    ('figures.main.fig6_benchmarking', ['--preset', 'clip']),
    ('figures.supplementary.sup_model_dichotomies', ['--resid']),
    ('figures.supplementary.sup_flatness_pipeline', ['--gradsum', 'cv']),
]


@pytest.mark.slow
@pytest.mark.parametrize('module,args', VARIANTS, ids=[m.rsplit('.', 1)[-1] + ' ' + ' '.join(a) for m, a in VARIANTS])
def test_variant_renders(module, args, tmp_path, cache):
    sys.path.insert(0, str(REPO / 'figures'))
    import render_all
    r = subprocess.run([sys.executable, '-m', module, '--out', str(tmp_path), *args], cwd=str(REPO),
                       env=render_all.render_env(tmp_path), text=True, capture_output=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    pngs = list(tmp_path.glob('*.png'))
    assert pngs, 'variant wrote no PNG'
