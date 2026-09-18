"""Static invariants of the ported figure scripts: package imports only, no legacy environment
switches, no path hacks, no writes outside the output directory."""
import re
from pathlib import Path

import pytest

from conftest import REPO

SCRIPTS = sorted((REPO / 'figures' / 'main').glob('*.py')) + sorted((REPO / 'figures' / 'supplementary').glob('*.py'))
SCRIPTS = [p for p in SCRIPTS if p.name != '__init__.py']
LEGACY = ['TAKE9_', 'FIG3_', "'SITE'", "'SEED_IMG'", "'OUTCOME'", "'RESID'", "'GRADMETRIC'", "'GRADSUM'",
          "'EXCLUDE_UNTRAINED'", "'INCLUDE_UNTRAINED'"]


def _code(path):
    """Source with comments and docstrings blanked."""
    src = path.read_text()
    src = re.sub(r'"""[\s\S]*?"""', '""', src)
    src = re.sub(r"'''[\s\S]*?'''", "''", src)
    return '\n'.join(l.split('#', 1)[0] for l in src.splitlines())


@pytest.mark.parametrize('path', SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_no_legacy_switches_or_path_hacks(path):
    code = _code(path)
    assert 'sys.path.insert' not in code and 'sys.path.append' not in code
    assert 'face_masking' not in code and 'ingredient_path' not in code
    assert '/Volumes/' not in code and '/Users/' not in code and '/n/holylabs' not in code
    for k in LEGACY:
        assert k not in code, f'{path.name}: legacy environment switch {k}'
    assert 'shutil.move' not in code, 'z-old archiving must not survive'
    assert 'datetime.now' not in code and 'strftime' not in code, 'timestamps in output names'


@pytest.mark.parametrize('path', SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_imports_go_through_pnc(path):
    code = _code(path)
    assert re.search(r'^from pnc(\.\w+)? import|^from pnc\.\w+ import|^import pnc', code, flags=re.M), path.name
    assert not re.search(r'^from utils import|^import utils$|^from _manifest|^from _famstats|^from preproc_helpers', code, flags=re.M)


@pytest.mark.parametrize('path', SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_writes_only_into_out_dir(path):
    code = _code(path)
    for m in re.finditer(r'\.(?:to_csv|to_pickle)\(([^,\n)]*)|pickle\.dump\([^,]+,\s*open\(([^,]+)|np\.save\w*\(([^,]+)|json\.dump\([^,]+,\s*open\(([^,]+)', code):
        target = next(g for g in m.groups() if g)
        assert 'out_dir' in target or 'out' in target.lower(), f'{path.name}: write target {target.strip()!r} is not under out_dir'
    assert 'preprocessed_data()' not in re.sub(r'paths\.require\(', '', code) or 'require' in code


@pytest.mark.parametrize('path', SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_entry_point_shape(path):
    src = path.read_text()
    if path.stem in ('fig1_grid', 'fig1_cloud', 'fig2_data', 'fig4_panels', 'fig5_adv', 'fig5_grad', 'fig6_data'):
        pytest.skip('helper module')
    assert re.search(r'^def main\(out_dir', src, flags=re.M), f'{path.name}: main(out_dir, ...) missing'
    assert "if __name__ == '__main__':" in src or 'if __name__ == "__main__":' in src
    assert "'--out'" in src or '"--out"' in src
    assert "matplotlib.use('Agg')" in src or 'matplotlib.use("Agg")' in src or 'import matplotlib' not in src.split('main(')[0]
