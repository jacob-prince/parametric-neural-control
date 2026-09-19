"""Unit tests for the small pnc helpers: trim, manifest, paths, style constants, name parsing,
family-gap statistics. No data needed."""
import os
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pnc import manifest, paths, trim, utils, famstats


# ---------------------------------------------------------------- trim
def _png(tmp_path, arr, name='x.png'):
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_trim_crops_white_border_and_keeps_pad(tmp_path):
    arr = np.full((100, 120, 3), 255, np.uint8)
    arr[30:50, 40:70] = 0                       # content box rows 30-49, cols 40-69
    p = _png(tmp_path, arr)
    assert trim.trim(p) is True
    out = np.asarray(Image.open(p))
    assert out.shape[:2] == (20 + 2 * trim.PAD, 30 + 2 * trim.PAD)
    assert (out[trim.PAD:-trim.PAD, trim.PAD:-trim.PAD, :3] == 0).all()


def test_trim_is_idempotent(tmp_path):
    arr = np.full((60, 60, 3), 255, np.uint8)
    arr[20:30, 20:30] = 10
    p = _png(tmp_path, arr)
    trim.trim(p)
    first = np.asarray(Image.open(p)).copy()
    assert trim.trim(p) is False
    assert np.array_equal(first, np.asarray(Image.open(p)))


def test_trim_treats_transparent_and_near_white_as_background(tmp_path):
    arr = np.zeros((50, 50, 4), np.uint8)
    arr[..., :3] = 252; arr[..., 3] = 255       # near-white opaque (>= THRESH) is background
    arr[10:20, 10:20] = (200, 200, 200, 255)    # content
    arr[40:50, :, 3] = 0                        # transparent strip is background
    p = _png(tmp_path, arr)
    assert trim.trim(p)
    assert np.asarray(Image.open(p)).shape[:2] == (10 + 2 * trim.PAD, 10 + 2 * trim.PAD)


def test_trim_leaves_blank_and_full_images_alone(tmp_path):
    blank = _png(tmp_path, np.full((20, 20, 3), 255, np.uint8), 'blank.png')
    assert trim.trim(blank) is False
    full = _png(tmp_path, np.zeros((20, 20, 3), np.uint8), 'full.png')
    assert trim.trim(full) is False
    assert np.asarray(Image.open(full)).shape[:2] == (20, 20)


# ---------------------------------------------------------------- manifest
def test_manifest_numbering_is_contiguous_and_totals_49():
    entries = manifest.all_entries()
    nums = []
    for e in entries:
        nums += list(range(e['num'], e['num'] + e.get('count', 1)))
    assert nums == list(range(1, 50))
    assert manifest.TOTAL == 49
    assert len({e['slug'] for e in entries}) == len(entries)


def test_manifest_output_names_are_unique_manuscript_stems():
    names = manifest.all_output_names()
    assert len(names) == 55 and len(set(names)) == 55
    assert names[:6] == ['framework', 'accentuation', 'single_site', 'divergence', 'predictors', 'benchmarking']
    for n in names[6:]:
        assert re.fullmatch(r's\d\d_[a-z0-9_]+', n), n
    assert [int(n[1:3]) for n in names[6:]] == list(range(1, 50))


def test_span_entries_need_a_monkey_and_map_to_consecutive_numbers():
    with pytest.raises(ValueError):
        manifest.output_name('sweep_gallery')
    stems = [manifest.output_name('sweep_gallery', mk) for mk in manifest.SPAN_MONKEYS]
    assert stems == [f's{n:02d}_sweep_gallery_{mk}' for n, mk in zip(range(12, 17), manifest.SPAN_MONKEYS)]
    with pytest.raises(ValueError):
        manifest.output_name('sweep_gallery', 'not_a_monkey')
    e = manifest.fig('high_control_examples')
    assert e['span'] == (22, 26) and e['prefix'] == 'figS22_high_control_examples'


def test_manifest_unknown_slug_raises():
    with pytest.raises(KeyError):
        manifest.fig('no_such_figure')


# ---------------------------------------------------------------- paths
def test_paths_follow_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 'src'))
    monkeypatch.setenv('PNC_PREPROCESSED_DATA', str(tmp_path / 'pre'))
    monkeypatch.setenv('PNC_OUTPUT', str(tmp_path / 'out'))
    assert paths.source_data() == tmp_path / 'src'
    assert paths.preprocessed_data() == tmp_path / 'pre'
    assert paths.output_dir() == tmp_path / 'out'
    assert paths.cluster_outputs() == tmp_path / 'src' / 'cluster_outputs'
    assert paths.frozen_inputs() == tmp_path / 'src' / 'frozen_inputs'


def test_paths_defaults_are_repo_relative(monkeypatch):
    for v in ('PNC_SOURCE_DATA', 'PNC_PREPROCESSED_DATA', 'PNC_OUTPUT'):
        monkeypatch.delenv(v, raising=False)
    assert paths.source_data() == paths.REPO_ROOT / 'source_data'
    assert paths.preprocessed_data() == paths.REPO_ROOT / 'preprocessed_data'
    assert paths.ASSETS.is_dir()


def test_require_reports_tier_and_hint(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 'src'))
    monkeypatch.setenv('PNC_PREPROCESSED_DATA', str(tmp_path / 'pre'))
    with pytest.raises(paths.MissingDataError) as e:
        paths.require(tmp_path / 'src' / 'x.h5')
    assert '--tier source' in str(e.value)
    with pytest.raises(paths.MissingDataError) as e:
        paths.require(tmp_path / 'pre' / 'y.pkl', hint='python scripts/preprocessing/build_y.py')
    assert '--tier preprocessed' in str(e.value) and 'build_y.py' in str(e.value)
    (tmp_path / 'ok').write_text('')
    assert paths.require(tmp_path / 'ok') == tmp_path / 'ok'


# ---------------------------------------------------------------- utils: study constants and style
def test_model_tables_are_consistent():
    models = set(utils.ALL_MODELS)
    assert len(models) == 10
    assert set(utils.MODEL_ORDER) == models and len(utils.MODEL_ORDER) == 10
    assert set(utils.MODEL_COLORS) == models and set(utils.MODEL_SHORT_NAMES) == models
    groups = set(utils.ROBUST_MODELS) | set(utils.NON_ROBUST_CNNS) | set(utils.NON_ROBUST_TRANSFORMERS)
    assert groups == models
    assert not set(utils.ROBUST_MODELS) & set(utils.NON_ROBUST_CNNS)
    assert all(len(utils.MONKEY_UNITS[m]) == 5 for m in utils.MONKEY_DIRS)
    assert set(utils.MONKEY_DIRS) == set(utils.MONKEY_COLORS) == set(utils.ENCODING_HDF5_FILES) == set(utils.MONKEY_RESPONSE_WINDOWS)
    for m in utils.MODEL_ORDER:
        assert re.fullmatch(r'#[0-9A-Fa-f]{6}', utils.get_model_color(m))
        assert utils.get_model_group(m) in utils.GROUP_ORDER
        assert utils.get_model_group_color(m) == utils.GROUP_COLORS[utils.get_model_group(m)]


def test_apply_figure_style_sets_the_publication_rcparams():
    import matplotlib.pyplot as plt
    utils.apply_figure_style()
    rc = plt.rcParams
    assert rc['font.sans-serif'][:4] == ['Helvetica Neue', 'Helvetica', 'Arial', 'Avenir']
    assert rc['savefig.dpi'] == 300 and rc['figure.dpi'] == 100
    assert rc['savefig.bbox'] == 'tight' and rc['savefig.pad_inches'] == 0.05
    assert rc['pdf.fonttype'] == 42 and rc['ps.fonttype'] == 42 and rc['svg.fonttype'] == 'none'
    assert rc['axes.spines.top'] is False and rc['axes.spines.right'] is False
    assert rc['legend.frameon'] is False and rc['axes.grid'] is False


def test_save_fig_trims_by_default_and_not_with_flag(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    utils.apply_figure_style()
    fig = plt.figure(figsize=(2, 2)); fig.text(0.5, 0.5, 'x'); plt.axis('off')
    raw = utils.save_fig(fig, tmp_path / 'raw', do_trim=False, bbox_inches=None)
    trimmed = utils.save_fig(fig, tmp_path / 'trimmed', bbox_inches=None)
    plt.close(fig)
    a, b = np.asarray(Image.open(raw)), np.asarray(Image.open(trimmed))
    assert a.shape[0] > b.shape[0] and a.shape[1] > b.shape[1]
    assert trim.trim(raw) and np.array_equal(np.asarray(Image.open(raw)), b)


def test_z_to_spks_clamps_at_zero():
    mu, sigma = np.array([5.0, 2.0]), np.array([2.0, 1.0])
    assert np.array_equal(utils.z_to_spks(np.array([-10.0, 1.0]), mu, sigma), np.array([0.0, 3.0]))
    assert utils.z_to_spks(-1.0, mu, sigma, unit=0) == 3.0
    assert utils.z_to_spks(-3.0, mu, sigma, unit=0) == 0.0


@pytest.mark.parametrize('name,expected', [
    ('resnet50_RidgeCV_unit_9_img_3_level_-1.25_score_-1.1.png',
     dict(model='resnet50', unit=9, seed=3, target=-1.25, score=-1.1)),
    ('clipag_vitb32_RidgeCV_unit_81_img_0_level_2.5_score_2.49', dict(model='clipag_vitb32', unit=81, seed=0, target=2.5, score=2.49)),
    ('AlexNet_training_seed_01_RidgeCV_unit_0_img_7_level_0.0_score_0.1.png',
     dict(model='AlexNet_training_seed_01', unit=0, seed=7, target=0.0, score=0.1)),
])
def test_parse_stimulus_name_round_trips(name, expected):
    assert utils.parse_stimulus_name(name) == expected


@pytest.mark.parametrize('name', ['shared0241_nsd20065.png', 'resnet50_RidgeCV_unit_x_img_3_level_1_score_1.png', 'no_score_here'])
def test_parse_stimulus_name_rejects_non_accentuated(name):
    assert utils.parse_stimulus_name(name) is None


def test_trial_index_and_sem():
    names = np.array(['a', 'b', 'a', 'a'])
    idx = utils.build_trial_index(names)
    assert idx == {'a': [0, 2, 3], 'b': [1]}
    resp = np.array([[1.0], [5.0], [3.0], [2.0]])
    sems, n = utils.compute_sem_for_unit(['a', 'b'], idx, resp, 0)
    assert n.tolist() == [3, 1] and sems[1] == 0
    assert np.isclose(sems[0], np.std([1, 3, 2], ddof=1) / np.sqrt(3))


def test_light_font_properties_falls_back_without_the_private_ttf(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_HN_LIGHT_TTF', str(tmp_path / 'missing.ttf'))
    monkeypatch.setattr(paths, 'ASSETS', tmp_path)       # no private/ dir here
    fp = utils.light_font_properties(size=9)
    assert fp.get_size() == 9 and fp.get_file() is None and fp.get_weight() == 'light'


# ---------------------------------------------------------------- famstats
def _balanced_design(gap=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mk in ['red', 'paul']:
        for u in range(3):
            base = rng.normal()
            for m in utils.MODEL_ORDER:
                rob = int(m in utils.ROBUST_MODELS)
                rows.append(dict(monkey=mk, unit=u, model=m, robust=rob, y=base + gap * rob + 0.01 * rng.normal()))
    return pd.DataFrame(rows)


def test_adv_gap_equals_family_mean_gap_on_a_balanced_design():
    df = _balanced_design()
    res = famstats.adv_gap(df, 'y')
    d = df[df.model != famstats.UNTR]
    per = d.groupby(['monkey', 'unit', 'robust'])['y'].mean().unstack()
    assert np.isclose(res['delta'], (per[1] - per[0]).mean(), atol=1e-9)
    assert res['n_sites'] == 6 and res['test'].startswith('within-site') and 0 <= res['p'] <= 1
    assert res['delta'] > 0.25


def test_adv_gap_with_covariate_reports_ancova():
    df = _balanced_design()
    df['cov'] = np.random.default_rng(1).normal(size=len(df))
    res = famstats.adv_gap(df, 'y', covariates=('cov',))
    assert res['test'].startswith('site-FE ANCOVA') and np.isfinite(res['delta'])


def test_stars_thresholds():
    assert famstats.stars(0.5) == 'n.s.' or famstats.stars(0.5) in ('ns', 'n.s.', '')
    assert famstats.stars(1e-4).count('*') >= 3 >= famstats.stars(0.03).count('*') >= 1
