#!/usr/bin/env python3
"""Rebuild tests/reference/figure_pixel_hashes.json from the manuscript PNGs in
tests/reference/manuscript_png/ (the 55 figures as submitted; downloaded with the preprocessed
data tier). Run only when the manuscript figures are intentionally updated."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pnc import manifest  # noqa: E402


def main():
    src = HERE / 'manuscript_png'
    out = {'_comment': "sha256 of the RGBA pixel array (np.asarray(Image.open(p).convert('RGBA')).tobytes()) "
                       "of each manuscript PNG (already border-trimmed). Regenerate with tests/reference/make_reference.py.",
           'figures': {}}
    for i, name in manifest.MAIN_FIGURES.items():
        a = np.asarray(Image.open(src / f'{name}.png').convert('RGBA'))
        out['figures'][name] = dict(kind='main', number=i, script=f'figures/main/fig{i}_{name}.py',
                                    shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())
    for e in manifest.all_entries():
        stems = [(manifest.output_name(e['slug'], mk), mk) for mk in manifest.SPAN_MONKEYS] if e['span'] \
            else [(manifest.output_name(e['slug']), None)]
        for stem, mk in stems:
            a = np.asarray(Image.open(src / f'{stem}.png').convert('RGBA'))
            d = dict(kind='supplementary', number=int(stem[1:3]), slug=e['slug'],
                     script=f"figures/supplementary/sup_{e['slug']}.py", shape=list(a.shape),
                     sha256=hashlib.sha256(a.tobytes()).hexdigest())
            if mk:
                d['monkey'] = mk
            out['figures'][stem] = d
    (HERE / 'figure_pixel_hashes.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f"{len(out['figures'])} figures hashed -> {HERE / 'figure_pixel_hashes.json'}")


if __name__ == '__main__':
    main()
