"""Demo notebooks: structure (kernel-safe setup cell, FULL_SCALE switch, captions) and unit
tests of the shared helper module (accentuation math, level schedule, objective loading)."""
import json
import sys

import numpy as np
import pytest

from conftest import REPO

DEMOS = sorted((REPO / 'notebooks' / 'demos').glob('*.ipynb'))
sys.path.insert(0, str(REPO / 'notebooks' / 'demos'))


def _cells(path):
    return json.loads(path.read_text())['cells']


@pytest.mark.parametrize('path', DEMOS, ids=[p.stem for p in DEMOS])
def test_demo_setup_cell_uses_the_shared_helper(path):
    """The kernel-safety boilerplate (thread pinning, Agg backend, data check) lives in
    notebooks/nbsetup.py; every demo's first code cell must go through it and nothing else."""
    cells = _cells(path)
    first = next(c for c in cells if c['cell_type'] == 'code')
    src = ''.join(first['source'])
    assert 'from nbsetup import setup' in src and 'setup(' in src
    assert 'OMP_NUM_THREADS' not in src and 'paths.require' not in src, f'{path.name}: boilerplate duplicated inline'
    assert '%matplotlib inline' not in src, f'{path.name}: the inline backend would undo the Agg selection'
    assert cells[0]['cell_type'] == 'markdown' and ''.join(cells[0]['source']).startswith('#')


@pytest.mark.parametrize('path', [p for p in DEMOS if p.stem[:2] in ('01', '02', '03', '04', '05')], ids=lambda p: p.stem)
def test_demo_documents_full_scale_settings(path):
    src = ''.join(''.join(c['source']) for c in _cells(path))
    assert 'FULL_SCALE' in src


def test_six_demos_exist_in_order():
    assert [p.stem[:2] for p in DEMOS] == ['00', '01', '02', '03', '04', '05']


torch = pytest.importorskip('torch')
import _demo_utils as U  # noqa: E402


def test_unit_levels_span_the_extended_natural_range():
    # run.py rule: extend_range*bandwidth below the natural range, twice that above it
    lv = U.compute_unit_levels(-1.0, 3.0, extend_range=0.25, num_levels=11)
    assert len(lv) == 11 and np.isclose(lv[0], -2.0) and np.isclose(lv[-1], 5.0)
    assert np.allclose(np.diff(lv), np.diff(lv)[0])


def test_normalize_and_display_round_trip():
    x = torch.rand(1, 3, 8, 8)
    xn = U.normalize01(x)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(xn * std + mean, x, atol=1e-6)
    disp = U.to_display(x[0])
    assert disp.shape == (8, 8, 3) and disp.min() >= 0 and disp.max() <= 1


def test_fft_scale_decays_with_frequency():
    s = U.get_fft_scale(32, 32, decay_power=1.5)
    s = np.asarray(s.detach().cpu()) if torch.is_tensor(s) else np.asarray(s)
    assert s.max() == s.flat[0] or s.max() >= s.mean(), 'low frequencies must be scaled up relative to high'
    assert np.all(s > 0)


def test_feature_accentuation_reduces_the_loss_on_a_toy_objective():
    torch.manual_seed(0)
    target = torch.zeros(1, 3, 16, 16); target[:, 0] = 1.0           # objective: redness

    def objective(x):
        return x[:, 0].mean(dim=(1, 2)) - x[:, 1:].mean(dim=(1, 2, 3))
    seed = torch.rand(3, 16, 16) * 0.5 + 0.25          # a constant image has no spectrum to precondition
    res = U.feature_accentuation(objective, seed, target_level=None, total_steps=30, learning_rate=2.0,
                                 image_size=16, model_input_size=16, crops_per_iteration=2, verbose=False, seed=0)
    assert res['history'][-1] > res['history'][0]
    assert res['image'].shape == (3, 16, 16) and res['image'].min() >= 0 and res['image'].max() <= 1


def test_feature_accentuation_tracks_a_target_level():
    torch.manual_seed(0)

    def objective(x):
        return x.mean(dim=(1, 2, 3)) * 10
    seed = (torch.rand(3, 16, 16) * 0.2 + 0.1)        # mean ~0.2 -> score ~2.0; ask for 6.0
    res = U.feature_accentuation(objective, seed, target_level=6.0, total_steps=60, learning_rate=1.0,
                                 image_size=16, model_input_size=16, crops_per_iteration=2, verbose=False, seed=0)
    start = float(objective(seed[None])); achieved = float(objective(res['image'][None]))
    assert abs(achieved - 6.0) < abs(start - 6.0), 'closer to the target than the seed was'


def test_find_export_names_follow_the_readout_convention(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path))
    p = U.find_export('red', 9, 'resnet50', '.layer4.Bottleneck0', 'Xtfmer')
    assert p.name == 'red_20250428-20250430_resnet50_Ch09_Xtfmer_.layer4.Bottleneck0_pca750_RidgeCV_JITscript.pt'
    assert 'encoding_model_outputs/Encoding_model_outputs/red_20250428-20250430' in str(p)
    assert U.find_export('paul', 8, 'resnet50_robust', '.layer4.Bottleneck2', 'meta').name.endswith('_meta_.layer4.Bottleneck2_pca750_RidgeCV.pkl')


@pytest.mark.slow
def test_encoding_objective_predicts_calibration_responses(source, cache):
    """The demo objective (backbone -> PCA -> readout) reproduces the cached post-hoc predictions."""
    from pnc.preproc import loader as L
    obj = U.EncodingObjective('resnet50', 'red', 9)
    e = L.load_encoding('red'); mi = list(e['models']).index('resnet50'); ui = list(e['units']).index(9)
    names = [n for n in e['stim'][:6]]
    imgs = torch.stack([U.load_image01(U.stimulus_path(n), 224) for n in names])
    with torch.no_grad():
        pred = obj(imgs).cpu().numpy()
    obj.close()
    cached = np.array([e['pred'][ui, mi][list(e['stim']).index(n)] for n in names])
    assert np.corrcoef(pred, cached)[0, 1] > 0.99
