"""notebooks/nbsetup.py: the setup shared by every notebook."""
import os
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / 'notebooks'))
import nbsetup  # noqa: E402
from pnc import manifest  # noqa: E402


def test_setup_pins_threads_selects_agg_and_returns_the_figure_dir(monkeypatch, tmp_path):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'KMP_DUPLICATE_LIB_OK'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv('PNC_OUTPUT', str(tmp_path / 'out'))
    out = nbsetup.setup(needs=None)
    import matplotlib
    assert out == tmp_path / 'out' / 'figures' and out.is_dir()
    assert os.environ['OMP_NUM_THREADS'] == '1' and os.environ['KMP_DUPLICATE_LIB_OK'] == 'TRUE'
    assert matplotlib.get_backend().lower() == 'agg'
    assert str(REPO) in sys.path


def test_setup_reports_missing_data_with_the_download_hint(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_PREPROCESSED_DATA', str(tmp_path / 'nothing'))
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 'nothing'))
    monkeypatch.setenv('PNC_OUTPUT', str(tmp_path / 'out'))
    from pnc.paths import MissingDataError
    with pytest.raises(MissingDataError, match='--tier preprocessed'):
        nbsetup.setup(needs='preprocessed')
    with pytest.raises(MissingDataError, match='--tier source'):
        nbsetup.setup(needs='source')


def test_setup_source_tier_points_backbones_at_source_data(monkeypatch, tmp_path):
    (tmp_path / 'src' / 'stimuli_encoding').mkdir(parents=True)
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 'src'))
    monkeypatch.setenv('PNC_OUTPUT', str(tmp_path / 'out'))
    monkeypatch.delenv('PNC_MODEL_BACKBONES', raising=False)
    nbsetup.setup(needs='source')
    assert os.environ['PNC_MODEL_BACKBONES'] == str(tmp_path / 'src' / 'model_backbones')


@pytest.mark.parametrize('stem', ['framework', 'divergence', 's12_sweep_gallery_red', 's49_predicting_summary'])
def test_caption_lookup_covers_main_and_supplementary(stem):
    cap = nbsetup.caption(stem)
    assert cap.startswith('**') and len(cap) > 80
    label = 'Figure' if stem in manifest.MAIN_FIGURES.values() else 'Supplementary Figure S'
    assert cap.startswith(f'**{label}')


def test_caption_unknown_stem_raises():
    with pytest.raises(KeyError):
        nbsetup.caption('no_such_figure')


def test_every_manuscript_stem_has_a_caption():
    for stem in manifest.all_output_names():
        nbsetup.caption(stem)


def test_show_figure_emits_a_preview_and_the_caption(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    png = tmp_path / 'divergence.png'
    Image.fromarray(np.full((3000, 4000, 3), 200, np.uint8)).save(png)
    shown = []
    import IPython.display as ipd
    monkeypatch.setattr(ipd, 'display', lambda obj: shown.append(obj))
    nbsetup.show_figure(png)
    assert len(shown) == 2
    img, md = shown
    assert getattr(img, 'width', None) == 900 and len(img.data) < 200_000, 'preview must be small'
    assert 'Figure 4.' in md.data
