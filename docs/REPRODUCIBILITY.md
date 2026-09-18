# Reproducibility guide

This page is for anyone who wants to check, rather than just use, the repository: how the
figures are verified against the submitted manuscript, how `preprocessed_data/` is regenerated
byte for byte, what the tests cover, and how the data are packaged for Zenodo.

## Reference environment and the pixel-exact policy

`environment.yml` pins the exact library versions the manuscript figures were rendered with
(Python 3.12.12, numpy 2.4.1, matplotlib 3.10.8, Pillow 12.1.0, freetype 2.14.1, pandas 3.0.0,
scipy 1.17.0, torch 2.10.0 CPU). The figures depend on text rendering, so "pixel-exact" is
defined on the **reference environment**: macOS (arm64) with the system *Helvetica Neue* font
and those pins. There, `python figures/render_all.py --check` and `pytest -m pixel` require the
RGBA pixel hash of every render to equal the hash of the submitted PNG
(`tests/reference/figure_pixel_hashes.json`). On other platforms or fonts the tests switch to
**tolerance mode** automatically: same size within 2 px, fewer than 1 % of pixels differing,
mean absolute difference below 0.5, with a diff image written to `outputs/pixel_diffs/` on
failure. `pytest tests/test_env.py -s` prints which mode applies.

One label in Figure 1 uses the *Light* face of Helvetica Neue. Apple's fonts cannot be
redistributed, so the `.ttf` is not in the repo: `pnc.utils.light_font_properties` looks for
`$PNC_HN_LIGHT_TTF`, then `figures/assets/private/HelveticaNeue-Light.ttf` (git-ignored;
extract it from `/System/Library/Fonts/HelveticaNeue.ttc` on macOS), and otherwise falls back to
the installed family at weight *light* (tolerance mode).

Saved PNGs are border-trimmed exactly as the manuscript build trims them (`pnc/trim.py`), so a
render *is* the manuscript file; set `PNC_NO_TRIM=1` for the raw matplotlib canvas.

`figures/render_all.py` runs each script in its own subprocess with a fixed environment (Agg
backend, private matplotlib config, `PYTHONHASHSEED=0`, single-threaded BLAS) and writes
`render_manifest.json` (script, duration, shape, pixel hash, library versions).

## Regenerating `preprocessed_data/`

```bash
python scripts/preprocessing/build_all.py --out /tmp/preproc      # ~1 h on a laptop
python -m pytest tests/test_preproc_bit_exact.py -m preproc         # rebuild + compare
```

`scripts/preprocessing/run_preproc.py` turns the raw recordings and model predictions into the
per-monkey caches (`brain_*.pkl`, `encoding_*.pkl`, `predictions_*.pkl`, `exclusions_*.pkl`,
`stimuli.pkl`, the controversial-experiment tables); every preprocessing knob lives in its
`CONFIG` dict. The other builders derive the figure-specific caches (layer selection, fLoc
selectivity, tuning stability, axis alignment, held-out attack tables, outcome ceilings, the
figure 2/3/6 embeddings, ...). `build_all.py` first copies the *frozen inputs* (files with no
public producer; see `data/FROZEN_INPUTS.md`), then runs the builders in dependency order in a
pinned environment and records a `BUILD_MANIFEST.json` of SHA-256s.

On the reference environment the rebuilt tree is byte-identical to the shipped one for every
cache (the only exceptions are the `created`/`git` stamps in `MANIFEST.json`); the shipped tree
is recorded in `tests/reference/preprocessed_data_hashes.json`. Elsewhere the comparison falls
back to structured equality (same keys, arrays equal including NaN positions, DataFrames equal
up to string dtype) and reports the level reached per file.

## Data tiers

| tier | contents | size |
|---|---|---|
| `preprocessed` | the caches every figure reads, plus the 55 submitted PNGs (pixel oracle) | ~0.5 GB |
| `source` (Tier A) | trial-level recordings (HDF5), post-hoc predictions, layer scores, accentuation configs, cluster outputs (attacks, gradient spectra), calibration images, every displayed accentuation sweep, the robust ResNet-50, the two demo readouts, frozen inputs | ~35 GB |

Not included: the remaining accentuated stimuli (83 GB), the exported readout weights of all
250 model-site pairs, ImageNet-val. `scripts/README.md` marks which stages need them.

`data/download_data.py` resumes interrupted downloads, verifies SHA-256 and MD5 of every
archive and every extracted member, never overwrites a file whose checksum already matches,
and refuses to replace a differing file without `--force`. `tests/reference/source_data_hashes.json`
lists every Tier A file; `python data/download_data.py --verify-only` checks a tree against it.

## Packaging and Zenodo

`data/build_manifest.py --archives DIR` writes the archives: the large HDF5 files are uploaded
as they are (Zenodo serves range requests), every many-file directory becomes one deterministic
tar (sorted members, zeroed mtimes and owners, a `MEMBERS.json` of per-file checksums first).
`data/zenodo_upload.py --archives DIR` creates or updates a **draft** deposit; the token is read
from `ZENODO_TOKEN` only and the script cannot publish. Publishing, which mints the DOI, is done
in the Zenodo web interface; the record id is then written into `data/zenodo_manifest.json`.
The software release is archived separately through Zenodo's GitHub integration.

## Tests

```bash
python -m pytest -q                                   # everything the present data allow
python -m pytest -q -m "not pixel and not preproc and not notebooks and not slow"   # no data needed
python -m pytest -q -m pixel                          # 55 figures vs the manuscript
python -m pytest -q -m notebooks                      # execute every notebook
python -m pytest -q -m preproc                        # rebuild preprocessed_data and compare
python -m pytest -q --run-raw                         # also validate the raw HDF5 sources
```

What the suite covers:

- **figures**: every render hash-matches the manuscript; the untrimmed canvas trims to the same
  pixels; every figure CLI keeps the original defaults; non-default variants still render.
- **caches**: the shipped tree matches its hash inventory; a full rebuild compares at three
  levels (bytes, structured exact, allclose).
- **scientific validation** (ported from the analysis pipeline): numerical kernels (outlier
  rejection, anchor-day standardization, noise ceilings, spectral flatness), cache integrity
  (study dimensions, partitions, target schedules, exclusion masks recomputed from their
  definitions), loader vs independent formulas, train/test isolation, gradient-cache
  consistency, raw HDF5 sources. Eight invariants the original suite deliberately kept red are
  strict `xfail`s with the reason spelled out.
- **notebooks**: the figure notebooks reproduce the manuscript PNGs, the demos execute end to
  end, and the committed notebooks carry their rendered outputs (a downscaled preview per figure).
- **tooling and docs**: deterministic archives, resumable downloads (against a local range
  server), the draft-only uploader, captions per figure, README paths, citation metadata, no
  private paths or legacy switches in the ported code.

Tests that need data skip cleanly when it is absent; the suite never writes into
`preprocessed_data/`.

## Frozen inputs and provenance

`data/FROZEN_INPUTS.md` lists every file that is shipped verbatim because its producer cannot be
re-run from the public data (cluster-only GPU jobs, the previous analysis tree, logs), with the
code that produced it. `scripts/synthesis/PROVENANCE.md` records the vendored accentuation
engine (upstream commit and every edit). Original cluster paths survive in the scripts only as
comments or as overridable `PNC_*` environment defaults.
