#!/usr/bin/env python3
"""Generate one notebook per manuscript figure under notebooks/figures/ (55 notebooks).

    python notebooks/_make_figure_notebooks.py            # (re)write all notebooks
    python notebooks/_make_figure_notebooks.py --check    # exit 1 if any committed notebook is stale

Each notebook has a fixed structure and fixed cell ids, so regenerating is a no-op diff:
title + caption (from figures/CAPTIONS.md), a setup cell that checks the data are present (kernel 'pnc': see README, Install),
one cell that calls the figure script's main(), and one that shows a downscaled preview with the
caption underneath. Everything the notebooks share (kernel setup, data check, preview) lives in
notebooks/nbsetup.py. The committed notebooks carry the rendered preview as their output (the
full-resolution file is written to outputs/figures/); --check compares cell sources only.
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
KERNEL = dict(name='pnc', display_name='Python 3 (pnc)', language='python')


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
        new_markdown_cell(f'# {head}\n\n'
                          f'Rendered by `{module.replace(".", "/")}.py`. In the reference environment the PNG written '
                          f'below is pixel-identical to the manuscript file `{stem}.png` (see docs/REPRODUCIBILITY.md); '
                          f'the preview at the end is downscaled.', id=f'{stem}-title'),
        new_code_cell('import sys\n'
                      "sys.path[:0] = ['..', 'notebooks']        # notebooks/nbsetup.py holds the setup shared by every notebook\n"
                      'from nbsetup import setup, show_figure\n'
                      '\n'
                      "out_dir = setup()                          # single-thread BLAS, Agg backend, repo on the path, data check", id=f'{stem}-setup'),
        new_code_cell(f'from {module} import main\n'
                      '\n'
                      '# main() writes <out_dir>/<manuscript name>.png (border-trimmed, so it matches the submitted file)\n'
                      '# and returns its path; keyword arguments select the variants documented in the script header.\n'
                      f'png = main(str(out_dir){kw})\n'
                      'print(png)', id=f'{stem}-render'),
        new_code_cell('show_figure(png)                          # preview of the rendered figure, caption underneath', id=f'{stem}-show'),
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


def skeleton(nb):
    """What --check compares: cell ids, types, sources and the kernelspec (never outputs)."""
    return ([(c.id, c.cell_type, c.source) for c in nb.cells], nb.metadata.get('kernelspec'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for path, text in render_all().items():
        if args.check:
            # compare the generated skeleton (sources, ids, kernel) with the committed notebook; outputs are ignored
            if not path.exists() or skeleton(nbformat.reads(text, as_version=4)) != skeleton(nbformat.read(path, as_version=4)):
                stale.append(path.name)
        elif path.exists():
            # keep the committed outputs (rendered previews) and refresh only the skeleton
            nb = nbformat.read(path, as_version=4); gen = nbformat.reads(text, as_version=4)
            if skeleton(nb) != skeleton(gen):
                by_id = {c.id: c for c in nb.cells}
                for c in gen.cells:
                    old = by_id.get(c.id)
                    if old is not None and old.cell_type == 'code' and old.source == c.source:
                        c.outputs, c.execution_count = old.outputs, old.execution_count
                nbformat.write(gen, path)
        else:
            path.write_text(text)
    if args.check:
        if stale:
            print('stale notebooks (run notebooks/_make_figure_notebooks.py):', stale)
            return 1
        print('all figure notebooks up to date')
        return 0
    print(f'wrote {len(render_all())} notebook skeletons -> {OUT} (existing outputs kept where the source is unchanged)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
