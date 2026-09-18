"""Shared pytest setup.

Markers: pixel (renders figures; needs preprocessed_data), preproc (regenerates caches; needs
source_data), notebooks, slow, gpu, raw (large raw HDF5 checks; needs --run-raw and source_data).
Data locations come from pnc.paths (PNC_SOURCE_DATA / PNC_PREPROCESSED_DATA); tests that need
data skip cleanly when it is absent. The suite is read-only: a session-scoped guard hashes
preprocessed_data before and after the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MPLCONFIGDIR', str(TESTS / '.mplconfig'))
Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pnc import paths as _paths  # noqa: E402

# Exact versions of the reference (pixel-exact) environment; see environment.yml.
REFERENCE_VERSIONS = dict(python='3.12.12', numpy='2.4.1', matplotlib='3.10.8', pillow='12.1.0',
                          freetype='2.14.1')


def pytest_addoption(parser):
    parser.addoption('--run-raw', action='store_true', default=False,
                     help='also validate the large raw HDF5 / session sources (needs source_data)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'manuscript: invariant stated in the manuscript or methods')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-raw'):
        return
    skip = pytest.mark.skip(reason='pass --run-raw to validate the large raw inputs')
    for item in items:
        if 'raw' in item.keywords:
            item.add_marker(skip)


def reference_env_status():
    """(is_reference, details). Reference = exact version tuple + system Helvetica Neue + arm64 macOS."""
    import matplotlib, numpy, PIL
    from matplotlib import font_manager, ft2font
    have = dict(python=platform.python_version(), numpy=numpy.__version__, matplotlib=matplotlib.__version__,
                pillow=PIL.__version__, freetype=ft2font.__freetype_version__)
    try:
        font = font_manager.findfont('Helvetica Neue', fallback_to_default=False)
    except ValueError:
        font = None
    ok = have == REFERENCE_VERSIONS and font is not None and font.endswith('HelveticaNeue.ttc') \
        and platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return ok, dict(have=have, expected=REFERENCE_VERSIONS, helvetica_neue=font, machine=platform.machine())


@pytest.fixture(scope='session')
def reference_env():
    return reference_env_status()


@pytest.fixture(scope='session')
def paths():
    """Data roots. `cache` is preprocessed_data (the caches every figure reads)."""
    return dict(repo=REPO, tests=TESTS, source=_paths.source_data(), cache=_paths.preprocessed_data(),
                output=_paths.output_dir(), cluster=_paths.cluster_outputs(), frozen=_paths.frozen_inputs())


def _need(path, what):
    if not Path(path).exists():
        pytest.skip(f'{what} not present ({path}); run data/download_data.py')


@pytest.fixture(scope='session')
def cache(paths):
    _need(paths['cache'] / 'brain_red.pkl', 'preprocessed_data')
    return paths['cache']


@pytest.fixture(scope='session')
def source(paths):
    _need(paths['source'] / 'brain_data_encoding', 'source_data')
    return paths['source']


@pytest.fixture(scope='session')
def load_pickle_file():
    def load(path):
        with Path(path).open('rb') as fh:
            return pickle.load(fh)
    return load


@pytest.fixture(scope='session')
def config(load_pickle_file, cache):
    return load_pickle_file(cache / 'brain_red.pkl')['config']


@pytest.fixture(scope='session')
def monkeys(config):
    return tuple(config['monkeys'])


@pytest.fixture(scope='session')
def models(config):
    return tuple(config['models'])


def _cached_loader(cache, prefix, load_pickle_file):
    store = {}

    def load(monkey):
        if monkey not in store:
            store[monkey] = load_pickle_file(cache / f'{prefix}_{monkey}.pkl')
        return store[monkey]
    return load


@pytest.fixture(scope='session')
def brain(load_pickle_file, cache):
    return _cached_loader(cache, 'brain', load_pickle_file)


@pytest.fixture(scope='session')
def encoding(load_pickle_file, cache):
    return _cached_loader(cache, 'encoding', load_pickle_file)


@pytest.fixture(scope='session')
def predictions(load_pickle_file, cache):
    return _cached_loader(cache, 'predictions', load_pickle_file)


def _tree_digest(root):
    h = hashlib.sha256()
    root = Path(root)
    if not root.exists():
        return None
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.name.startswith('.') and '.build_logs' not in p.parts:
            st = p.stat()
            h.update(f'{p.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}'.encode())
    return h.hexdigest()


@pytest.fixture(scope='session', autouse=True)
def preprocessed_data_is_read_only(paths):
    """Nothing in the suite may write into preprocessed_data."""
    before = _tree_digest(paths['cache'])
    yield
    after = _tree_digest(paths['cache'])
    assert before == after, 'the test session modified preprocessed_data/'


@pytest.fixture(scope='session')
def pnc_kernel():
    """Name of a Jupyter kernel that runs THIS interpreter. A user-level 'python3' kernelspec
    (~/Library/Jupyter or ~/.local/share/jupyter) would otherwise take precedence and execute
    the notebooks under a different Python; JUPYTER_DATA_DIR overrides that lookup."""
    tmp = tempfile.mkdtemp(prefix='pnc-kernel-')
    spec = Path(tmp) / 'kernels' / 'pnc'
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python 3 (pnc)', 'language': 'python'}))
    old = os.environ.get('JUPYTER_DATA_DIR')
    os.environ['JUPYTER_DATA_DIR'] = tmp
    yield 'pnc'
    if old is None:
        os.environ.pop('JUPYTER_DATA_DIR', None)
    else:
        os.environ['JUPYTER_DATA_DIR'] = old
