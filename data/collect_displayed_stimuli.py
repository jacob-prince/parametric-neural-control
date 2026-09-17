#!/usr/bin/env python3
"""Maintainer tool: list every stimulus image the figure scripts open, so that source_data can
ship only the displayed subset of the (83 GB) accentuated-stimulus and controversial-stimulus
directories.

    python data/collect_displayed_stimuli.py --out data/displayed_stimuli.txt

Runs every figure in-process with PIL.Image.open and matplotlib.image.imread wrapped, and
writes the sorted set of opened paths under stimuli_control/ and stimuli_controversial/ relative
to source_data. Renders are written to a temporary directory and discarded; pixels are not
checked here (that is render_all.py --check).
"""
import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pnc import manifest, paths  # noqa: E402

OPENED = set()


def _hook():
    import PIL.Image
    import matplotlib.image
    real_open, real_imread = PIL.Image.open, matplotlib.image.imread

    def open_(fp, *a, **k):
        if isinstance(fp, (str, os.PathLike)):
            OPENED.add(os.path.abspath(os.fspath(fp)))
        return real_open(fp, *a, **k)

    def imread(fname, *a, **k):
        if isinstance(fname, (str, os.PathLike)):
            OPENED.add(os.path.abspath(os.fspath(fname)))
        return real_imread(fname, *a, **k)
    PIL.Image.open, matplotlib.image.imread = open_, imread


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=REPO / 'data' / 'displayed_stimuli.txt')
    ap.add_argument('--only', help='comma-separated output stems')
    args = ap.parse_args(argv)
    _hook()
    src = paths.source_data().resolve()
    wanted = {w.strip() for w in args.only.split(',')} if args.only else None
    items = [(n, f'figures.main.fig{i}_{n}', {}) for i, n in manifest.MAIN_FIGURES.items()]
    for e in manifest.all_entries():
        mod = f"figures.supplementary.sup_{e['slug']}"
        if e['span']:
            items += [(manifest.output_name(e['slug'], mk), mod, {'monkey': mk}) for mk in manifest.SPAN_MONKEYS]
        else:
            items.append((manifest.output_name(e['slug']), mod, {}))
    with tempfile.TemporaryDirectory() as tmp:
        for stem, mod, kw in items:
            if wanted and stem not in wanted:
                continue
            before = len(OPENED)
            print(f'{stem} ...', end=' ', flush=True)
            importlib.import_module(mod).main(tmp, **kw)
            print(f'{len(OPENED) - before} new opens', flush=True)
    rel = sorted(os.path.relpath(p, src) for p in OPENED
                 if Path(p).resolve().is_relative_to(src) or p.startswith(str(src)))
    keep = [r for r in rel if r.split(os.sep)[0] in ('stimuli_control', 'stimuli_controversial')]
    args.out.write_text('\n'.join(keep) + '\n')
    print(f'{len(keep)} displayed stimulus files (of {len(rel)} source_data files opened) -> {args.out}')


if __name__ == '__main__':
    main()
