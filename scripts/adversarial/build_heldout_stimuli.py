#!/usr/bin/env python3
"""
build_heldout_stimuli.py - resolve the 88 NSD shared1000 images that were never used
anywhere in the study (item 1 held-out validation set) to full cluster paths, and render a
contact-sheet montage for a diversity check.

Runs on the cluster (the shared1000 master dir is cluster-only). Writes into
  <figure8>/intermediate/validation_stimuli/{heldout_paths.txt, heldout88_montage.png}
"""
import glob, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

SHARED_DIR = os.environ.get("PNC_SHARED1000_DIR", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/shared1000")  # was "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/shared1000"

# the 88 shared indices untouched by encoding fit/test, synthesis seeds, and BOTH
# natural-controversial sessions (strict "not used in the study whatsoever").
UNUSED_IDX = [
    0, 15, 34, 37, 46, 49, 52, 69, 82, 84, 85, 86, 121, 122, 124, 132, 150, 154, 191, 197,
    213, 231, 256, 276, 282, 303, 304, 319, 343, 348, 349, 351, 387, 403, 414, 426, 432, 439,
    449, 470, 473, 483, 487, 510, 512, 522, 526, 527, 566, 603, 605, 608, 627, 632, 637, 650,
    686, 691, 700, 708, 711, 712, 721, 747, 761, 772, 775, 791, 792, 797, 810, 812, 820, 825,
    826, 837, 851, 906, 909, 910, 933, 935, 970, 977, 978, 992, 996, 997,
]
# The strict-unused pool is COCO-scene-heavy and has NO large human faces. These 13 shared1000
# images carry visible human faces and are unused by the encoding fit/test AND synthesis seeds
# (they DID appear in the red natural-controversial session, so they are held out w.r.t. the
# encoding models being validated, but are not "untouched by the whole study"). Added for
# face-category coverage -> 87 strict + 13 face = 100 held-out validation images.
FACE_AUGMENT_IDX = [22, 144, 306, 385, 453, 525, 528, 731, 786, 798, 817, 936, 956]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    def resolve(indices):
        got, miss = [], []
        for i in indices:
            hits = sorted(glob.glob(f"{SHARED_DIR}/shared{i:04d}_nsd*.png"))
            (got.append(hits[0]) if hits else miss.append(i))
        return got, miss

    strict, miss_s = resolve(UNUSED_IDX)
    faces, miss_f = resolve(FACE_AUGMENT_IDX)
    paths = strict + faces
    print(f"strict {len(strict)}/{len(UNUSED_IDX)} (missing {miss_s}); "
          f"face {len(faces)}/{len(FACE_AUGMENT_IDX)} (missing {miss_f}); total {len(paths)}", flush=True)
    (out / "heldout_paths.txt").write_text("\n".join(paths) + "\n")

    th, cols = 110, 10
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * th, rows * th), (255, 255, 255))
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((th, th), Image.BICUBIC)
        canvas.paste(im, ((k % cols) * th, (k // cols) * th))
    canvas.save(out / "heldout100_montage.png")
    print(f"WROTE {out/'heldout_paths.txt'} ({len(paths)} imgs) and {out/'heldout100_montage.png'}", flush=True)


if __name__ == "__main__":
    main()
