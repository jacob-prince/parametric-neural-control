"""Committed notebooks carry no outputs and are exactly what the generator produces."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

NOTEBOOKS = sorted((REPO / 'notebooks').rglob('*.ipynb'))


@pytest.mark.parametrize('path', NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            assert cell.get('outputs', []) == [], f'{path.name}: cell has outputs'
            assert cell.get('execution_count') is None, f'{path.name}: cell has an execution count'
    assert 'widgets' not in nb.get('metadata', {})


def test_figure_notebooks_match_generator():
    r = subprocess.run([sys.executable, str(REPO / 'notebooks' / '_make_figure_notebooks.py'), '--check'],
                       cwd=str(REPO), text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_one_notebook_per_manuscript_figure():
    from pnc import manifest
    stems = {p.stem for p in (REPO / 'notebooks' / 'figures').glob('*.ipynb')}
    expected = {f'{i:02d}_{n}' for i, n in manifest.MAIN_FIGURES.items()} | set(manifest.all_output_names()[6:])
    assert stems == expected
