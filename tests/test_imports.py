"""Every module in the repo imports (third-party dependencies of scripts/ may be absent in the
figure environment; those modules are skipped, not failed, when the missing package is one of
the GPU-stage extras listed in requirements-scripts.txt)."""
import importlib
import pkgutil
import re

import pytest

from conftest import REPO

EXTRAS = set(re.findall(r'^([A-Za-z0-9_-]+)', (REPO / 'requirements-scripts.txt').read_text(), flags=re.M)) \
    | {'clip', 'open_clip', 'yaml', 'skimage', 'cuml', 'einops', 'dill', 'imageio'}


def _modules(pkg):
    mod = importlib.import_module(pkg)
    return [pkg] + [m.name for m in pkgutil.walk_packages(mod.__path__, pkg + '.')]


@pytest.mark.parametrize('name', _modules('pnc') + _modules('core') + _modules('neural_regress'))
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split('.')[0].replace('-', '_') in {x.replace('-', '_') for x in EXTRAS}:
            pytest.skip(f'{name}: optional dependency {e.name} not installed')
        raise
