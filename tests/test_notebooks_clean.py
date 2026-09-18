"""Committed notebooks carry their rendered outputs (a downscaled preview for each figure
notebook, real outputs for the demos), and their sources are exactly what the generator
produces."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

FIGURE_NBS = sorted((REPO / 'notebooks' / 'figures').glob('*.ipynb'))
DEMO_NBS = sorted((REPO / 'notebooks' / 'demos').glob('*.ipynb'))


def _code_cells(path):
    return [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']


@pytest.mark.parametrize('path', FIGURE_NBS, ids=[p.name for p in FIGURE_NBS])
def test_figure_notebook_carries_a_rendered_preview(path):
    cells = _code_cells(path)
    show = [c for c in cells if c.get('id', '').endswith('-show')]
    assert len(show) == 1
    outs = show[0].get('outputs', [])
    assert any(k.startswith('image/') for o in outs for k in o.get('data', {})), f'{path.name}: no rendered preview'
    assert not any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', [])), f'{path.name}: error output'
    b64 = next(v for o in outs for k, v in o.get('data', {}).items() if k.startswith('image/'))
    assert len(b64) < 1_500_000, f'{path.name}: preview too large for git'


@pytest.mark.parametrize('path', DEMO_NBS, ids=[p.name for p in DEMO_NBS])
def test_demo_notebook_was_executed(path):
    cells = _code_cells(path)
    assert any(c.get('outputs') for c in cells), f'{path.name}: no outputs'
    assert not any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', [])), f'{path.name}: error output'
    assert all(c.get('execution_count') for c in cells if c['source']), f'{path.name}: a cell was not executed'


def test_figure_notebooks_match_generator():
    r = subprocess.run([sys.executable, str(REPO / 'notebooks' / '_make_figure_notebooks.py'), '--check'],
                       cwd=str(REPO), text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_one_notebook_per_manuscript_figure():
    from pnc import manifest
    stems = {p.stem for p in FIGURE_NBS}
    expected = {f'{i:02d}_{n}' for i, n in manifest.MAIN_FIGURES.items()} | set(manifest.all_output_names()[6:])
    assert stems == expected
