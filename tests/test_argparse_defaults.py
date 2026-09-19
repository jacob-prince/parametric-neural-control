"""The keyword defaults of every figure's main() equal the environment-variable defaults of the
original pipeline (FIG3_*, SITE, SEED_IMG, OUTCOME, RESID, GRADMETRIC, GRADSUM, EXCLUDE_UNTRAINED,
INCLUDE_UNTRAINED), so the default render is the manuscript figure. Every figure module must
expose main(out_dir, ...) and a CLI whose --out maps onto out_dir."""
import importlib
import inspect
import os
import subprocess
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / 'figures'))
import render_all  # noqa: E402

EXPECTED = {
    'figures.main.fig3_single_site': dict(monkey='red', unit=9, model='resnet50', channel_label='2', p2set='heldout'),
    'figures.main.fig5_predictors': dict(site='paul8', seed_img=3, outcome='slope', resid='9trained', gradmetric='pr'),
    'figures.main.fig6_benchmarking': dict(preset='rn50'),
    'figures.supplementary.sup_adv_sensitivity_formulations': dict(outcome='slope', gradsum='pr'),
    'figures.supplementary.sup_flatness_pipeline': dict(gradsum='pr'),
    'figures.supplementary.sup_concentration_image_robustness': dict(gradsum='pr'),
    'figures.supplementary.sup_flatness_regime_control': dict(gradsum='pr'),
    'figures.supplementary.sup_predicting_cv_trained': dict(include_untrained=False),
    'figures.supplementary.sup_predicting_summary': dict(exclude_untrained=True),
    'figures.supplementary.sup_model_dichotomies': dict(resid=False),
    'figures.supplementary.sup_sweep_gallery': dict(monkey=None),
    'figures.supplementary.sup_high_control_examples': dict(monkey=None),
}
MODULES = sorted({m for _, m, _ in render_all.registry()})


@pytest.mark.parametrize('module', MODULES, ids=[m.rsplit('.', 1)[-1] for m in MODULES])
def test_main_signature(module):
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    mod = importlib.import_module(module)
    sig = inspect.signature(mod.main)
    params = list(sig.parameters)
    assert params and params[0] == 'out_dir', f'{module}: main() must take out_dir first'
    for name, default in EXPECTED.get(module, {}).items():
        assert name in sig.parameters, f'{module}: main() lacks {name}'
        assert sig.parameters[name].default == default, f'{module}: {name} default {sig.parameters[name].default!r} != {default!r}'


@pytest.mark.parametrize('module', MODULES, ids=[m.rsplit('.', 1)[-1] for m in MODULES])
def test_cli_help(module):
    r = subprocess.run([sys.executable, '-m', module, '--help'], cwd=str(REPO), text=True, capture_output=True,
                       env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'MPLBACKEND': 'Agg'})
    assert r.returncode == 0, r.stderr[-1500:]
    assert '--out' in r.stdout
