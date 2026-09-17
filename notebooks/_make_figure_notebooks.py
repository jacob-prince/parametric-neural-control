#!/usr/bin/env python3
"""Generate one notebook per manuscript figure under notebooks/figures/ (55 notebooks).

    python notebooks/_make_figure_notebooks.py            # (re)write all notebooks
    python notebooks/_make_figure_notebooks.py --check    # exit 1 if any committed notebook is stale

Each notebook has a fixed structure and fixed cell ids, so regenerating is a no-op diff:
title + caption (from figures/CAPTIONS.md), a setup cell that checks the data are present,
one cell that calls the figure script's main(), and one that displays the PNG. Notebooks are
committed without outputs (see tests/test_notebooks_clean.py).
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnc import manifest  # noqa: E402

OUT = REPO / 'notebooks' / 'figures'
KERNEL = dict(name='python3', display_name='Python 3 (pnc)', language='python')


def captions():
    text = (REPO / 'figures' / 'CAPTIONS.md').read_text()
    out = {}
    for block in re.split(r'^### ', text, flags=re.M)[1:]:
        head, _, body = block.partition('\n')
        stem = re.search(r'`([^`]+)`', head).group(1)
        out[stem] = (head.strip(), body.strip())
    return out


def notebook_spec():
    """[(notebook file stem, output stem, module, kwargs)] in manuscript order."""
    spec = []
    for i, name in manifest.MAIN_FIGURES.items():
        spec.append((f'{i:02d}_{name}', name, f'figures.main.fig{i}_{name}', {}))
    for e in manifest.all_entries():
        mod = f"figures.supplementary.sup_{e['slug']}"
        if e['span']:
            for mk in manifest.SPAN_MONKEYS:
                stem = manifest.output_name(e['slug'], mk)
                spec.append((stem, stem, mod, {'monkey': mk}))
        else:
            stem = manifest.output_name(e['slug'])
            spec.append((stem, stem, mod, {}))
    return spec


def build(nb_stem, stem, module, kwargs, cap):
    head, body = cap
    kw = ''.join(f', {k}={v!r}' for k, v in kwargs.items())
    cells = [
        new_markdown_cell(f'# {head}\n\n{body}\n\n'
                          f'Rendered by `{module.replace(".", "/")}.py`; the PNG written below is pixel-identical '
                          f'to the manuscript file `{stem}.png` in the reference environment (see README, '
                          f'"Pixel-exact policy").', id=f'{stem}-title'),
        new_code_cell('%matplotlib inline\n'
                      'import os, sys\n'
                      'from pathlib import Path\n'
                      "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
                      "REPO = Path.cwd() if (Path.cwd() / 'pnc').exists() else Path.cwd().parents[1]\n"
                      'sys.path.insert(0, str(REPO))\n'
                      'from pnc import paths\n'
                      "paths.require(paths.preprocessed_data() / 'brain_red.pkl')  # python data/download_data.py --tier preprocessed\n"
                      "out_dir = paths.output_dir() / 'figures'\n"
                      'out_dir.mkdir(parents=True, exist_ok=True)', id=f'{stem}-setup'),
        new_code_cell(f'from {module} import main\n'
                      f'png = main(str(out_dir){kw})\n'
                      'print(png)', id=f'{stem}-render'),
        new_code_cell('from IPython.display import Image, display\n'
                      'display(Image(filename=png, width=900))', id=f'{stem}-show'),
    ]
    nb = new_notebook(cells=cells, metadata={'kernelspec': KERNEL, 'language_info': {'name': 'python'}})
    return nb


def render_all():
    caps = captions()
    out = {}
    for nb_stem, stem, module, kwargs in notebook_spec():
        nb = build(nb_stem, stem, module, kwargs, caps[stem])
        nbformat.validate(nb)
        out[OUT / f'{nb_stem}.ipynb'] = nbformat.writes(nb) + '\n'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for path, text in render_all().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
    if args.check:
        if stale:
            print('stale notebooks (run notebooks/_make_figure_notebooks.py):', stale)
            return 1
        print('all figure notebooks up to date')
        return 0
    print(f'wrote {len(render_all())} notebooks -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
