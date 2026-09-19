"""figures/render_all.py: registry, isolated environment, pixel hashing, selection."""
import hashlib
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import REPO

sys.path.insert(0, str(REPO / 'figures'))
import render_all  # noqa: E402
from pnc import manifest  # noqa: E402


def test_registry_covers_every_manuscript_figure_in_order():
    items = render_all.registry()
    assert [it[0] for it in items] == manifest.all_output_names()
    assert len({it[1] for it in items}) == 47, 'one module per script (6 main + 41 supplementary)'
    for stem, module, extra in items:
        assert (REPO / (module.replace('.', '/') + '.py')).exists(), module
        if '_gallery_' in stem or 'high_control_examples_' in stem:
            assert extra == ['--monkey', stem.rsplit('_', 1)[-1]]
        else:
            assert extra == []


def test_render_env_is_built_from_scratch(monkeypatch, tmp_path):
    monkeypatch.setenv('TAKE9_FACE_MODE', 'masked')
    monkeypatch.setenv('FIG3_MONKEY', 'paul')
    monkeypatch.setenv('GRADSUM', 'cv')
    monkeypatch.setenv('PNC_SOURCE_DATA', '/data/src')
    env = render_all.render_env(tmp_path)
    for k in ('TAKE9_FACE_MODE', 'FIG3_MONKEY', 'GRADSUM', 'SITE', 'OUTCOME', 'RESID'):
        assert k not in env
    assert env['MPLBACKEND'] == 'Agg' and env['PYTHONHASHSEED'] == '0'
    assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '1'
    assert env['PNC_SOURCE_DATA'] == '/data/src'
    assert env['MPLCONFIGDIR'] == str(tmp_path / '.mplconfig') and (tmp_path / '.mplconfig').is_dir()


def test_pixel_hash_is_the_sha256_of_the_rgba_array(tmp_path):
    arr = np.random.default_rng(0).integers(0, 255, (8, 9, 3), dtype=np.uint8)
    p = tmp_path / 'x.png'
    Image.fromarray(arr).save(p)
    h, shape = render_all.pixel_hash(p)
    rgba = np.dstack([arr, np.full((8, 9), 255, np.uint8)])
    assert h == hashlib.sha256(rgba.tobytes()).hexdigest() and shape == [8, 9, 4]


def test_versions_report_the_libraries_that_matter():
    v = render_all.versions()
    for k in ('python', 'matplotlib', 'numpy', 'pillow', 'freetype', 'platform', 'machine', 'helvetica_neue'):
        assert k in v


def test_only_filter_accepts_stems_and_slugs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(render_all, 'render', lambda stem, module, extra, out, env: (calls.append(stem) or
                        dict(name=stem, module=module, args=extra, returncode=0, duration_seconds=0.0)))
    assert render_all.main(['--only', 'divergence,sweep_gallery', '--out', str(tmp_path)]) == 0
    assert calls == ['divergence'] + [f's{n:02d}_sweep_gallery_{mk}' for n, mk in zip(range(12, 17), manifest.SPAN_MONKEYS)]
    assert (tmp_path / 'render_manifest.json').exists()
    with pytest.raises(SystemExit):
        render_all.main(['--only', 'no_such_figure', '--out', str(tmp_path)])


def test_check_mode_flags_a_differing_hash(monkeypatch, tmp_path):
    def fake_render(stem, module, extra, out, env):
        return dict(name=stem, module=module, args=extra, returncode=0, duration_seconds=0.0,
                    sha256='0' * 64, shape=[1, 1, 4])
    monkeypatch.setattr(render_all, 'render', fake_render)
    assert render_all.main(['--only', 'divergence', '--check', '--out', str(tmp_path)]) == 1
