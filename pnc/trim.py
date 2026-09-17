"""Trim near-white (and transparent) borders from a figure PNG, leaving a small uniform
pad. This is the exact operation the manuscript build applies to every figure at ingest
(02-Nature-Submission/take1/trim_png.py), so a trimmed render is pixel-identical to the
manuscript file.
"""
import numpy as np
from PIL import Image

THRESH = 250   # channel value above which a pixel counts as background white
PAD = 8        # pixels of border to keep on every side after the trim


def trim(path, thresh=THRESH, pad=PAD):
    """Trim `path` in place. Returns True if the file changed."""
    im = Image.open(path)
    arr = np.asarray(im.convert('RGBA'))
    rgb, alpha = arr[..., :3], arr[..., 3]
    content = ~((rgb.min(axis=2) >= thresh) | (alpha == 0))
    rows, cols = np.any(content, axis=1), np.any(content, axis=0)
    if not rows.any():
        return False  # blank image; leave untouched
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(arr.shape[0], r1 + pad), min(arr.shape[1], c1 + pad)
    if (r0, c0) == (0, 0) and (r1, c1) == arr.shape[:2]:
        return False  # nothing to trim
    im.crop((c0, r0, c1, r1)).save(path)
    return True
