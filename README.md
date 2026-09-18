# parametric-neural-control

Code, data pipeline and figure sources for

> Prince J.S.\*, Wang B.\*, Fel T., Jagadeesh A.V., Vaziri P.A., Alvarez G.A., Livingstone M.S. & Konkle T. (2026).
> **Parametric neural control differentiates top neural network models of primate visual cortex.**

Everything in the paper can be rebuilt from this repository plus two data tiers hosted on Zenodo:

| what you want | what you need | command |
|---|---|---|
| every main and supplementary figure, pixel-identical to the submitted PNGs | `preprocessed_data/` (~0.5 GB) | `python figures/render_all.py --check` |
| regenerate `preprocessed_data/` byte-for-byte from the raw inputs | `source_data/` (~35 GB) | `python scripts/preprocessing/build_all.py` |
| fit an encoding model, accentuate an image, run an attack, compute a gradient spectrum | `source_data/` | `notebooks/demos/` |
| re-run the upstream GPU stages as we ran them | `source_data/` + a GPU + the full stimulus set | `scripts/` (see the README in each folder) |

Data record: Zenodo DOI **10.5281/zenodo.YYYYYYY** (placeholder until the deposit is published).
Software archive of this release: Zenodo DOI **10.5281/zenodo.XXXXXXX** (placeholder).

---

## 1. Install

```bash
git clone https://github.com/jacob-prince/parametric-neural-control.git
cd parametric-neural-control
conda env create -f environment.yml      # the reference environment (exact pins, CPU only)
conda activate pnc
pip install -e .
```

`environment.yml` pins the exact library versions the manuscript figures were rendered with
(Python 3.12.12, numpy 2.4.1, matplotlib 3.10.8, Pillow 12.1.0, freetype 2.14.1, pandas 3.0.0,
scipy 1.17.0, torch 2.10.0 CPU). Any recent Python with the packages in `pyproject.toml` will
run the code; only pixel-exactness needs the pins (section 5).

The GPU stages under `scripts/` need extra packages: `pip install -r requirements-scripts.txt`.

## 2. Download the data

```bash
python data/download_data.py --tier preprocessed    # ~0.5 GB: caches + the 55 manuscript PNGs
python data/download_data.py --tier source          # ~25 GB: raw inputs (Tier A, see below)
python data/download_data.py --verify-only          # re-check what is on disk
```

Files land in `preprocessed_data/` and `source_data/` (override with `PNC_PREPROCESSED_DATA`,
`PNC_SOURCE_DATA`; outputs go to `outputs/`, override with `PNC_OUTPUT`). Downloads resume,
every file is checked against the SHA-256 recorded in `data/zenodo_manifest.json`, and a file
already present with the right checksum is never re-downloaded or overwritten.

**Tier A source data** is everything needed to regenerate `preprocessed_data/` and to run the
demos: trial-level neural recordings (HDF5) for the five macaques, the post-hoc encoding-model
predictions, layer-selection scores, accentuation configs, the cluster analysis outputs (attacks,
gradient spectra), the calibration image set, the accentuated and controversial stimuli that
appear in figures (complete sweeps), the robust ResNet-50 backbone, the exported readouts of the
two model-site pairs used by the demos, and a small set of *frozen inputs* whose
producers cannot be re-run publicly (`source_data/frozen_inputs/PROVENANCE.md`). It does not
include the complete set of 27,720 accentuated stimuli, the exported readout weights, or
ImageNet-val; the scripts that need those are marked in `scripts/README.md`.

## 3. Reproduce the figures

```bash
python figures/render_all.py                   # all 55 -> outputs/figures/<manuscript name>.png
python figures/render_all.py --check           # ...and compare with the submitted PNGs
python figures/render_all.py --only divergence,s17_control_slope_anova
python -m figures.main.fig4_divergence --out outputs/figures        # one figure, one script
```

`figures/main/fig<N>_<name>.py` and `figures/supplementary/sup_<slug>.py` each expose
`main(out_dir, **variant)` and a CLI with the same keyword arguments (variants such as
`--outcome r` or `--monkey paul` reproduce the alternative renders we looked at; only the
defaults are in the paper). Numbering of the supplementary figures comes from one place,
`pnc/manifest.py`; output names are the manuscript names (`s17_control_slope_anova.png`).

`figures/render_all.py` runs each script in its own subprocess with a fixed environment (Agg
backend, private matplotlib config, hash seed 0, single-threaded BLAS), writes
`render_manifest.json` (script, duration, shape, pixel hash, library versions) and, with
`--check`, compares the RGBA pixel array of each render with `tests/reference/figure_pixel_hashes.json`.

Saved PNGs are border-trimmed exactly as the manuscript build trims them (`pnc/trim.py`), so a
render *is* the manuscript file; pass `--no-trim` for the raw matplotlib canvas.

`notebooks/figures/` has one notebook per figure (55) that calls the same `main()` and shows
the result with its caption (`figures/CAPTIONS.md`).

## 4. Regenerate `preprocessed_data/`

```bash
python scripts/preprocessing/build_all.py --out /tmp/preproc      # ~15 min on a laptop
python -m pytest tests/test_preproc_bit_exact.py -m preproc         # rebuild + compare
```

`scripts/preprocessing/run_preproc.py` turns the raw recordings and model predictions into the
per-monkey caches (`brain_*.pkl`, `encoding_*.pkl`, `predictions_*.pkl`, `exclusions_*.pkl`,
`stimuli.pkl`, the controversial-experiment tables); every preprocessing knob lives in its
`CONFIG` dict. The other builders derive the figure-specific caches (layer selection, fLoc
selectivity, tuning stability, axis alignment, held-out attack tables, outcome ceilings, ...).
`build_all.py` runs them in dependency order in a pinned environment and records a
`BUILD_MANIFEST.json` of SHA-256s.

On the reference environment the rebuilt tree is byte-identical to the shipped one for every
cache (the only exceptions are the `created`/`git` stamps in `MANIFEST.json`). Elsewhere the
comparison in `tests/test_preproc_bit_exact.py` falls back to structured equality
(same keys, arrays equal including NaN positions) and reports the level reached per file.

## 5. Pixel-exact policy

The figures depend on text rendering, so "pixel-exact" is defined on the **reference
environment**: macOS (arm64) with the system *Helvetica Neue* font and the versions pinned in
`environment.yml`. There, `render_all.py --check` and `pytest -m pixel` require the pixel
hashes to match. On other platforms (or other fonts) the tests switch to **tolerance mode**
automatically: same size within 2 px, fewer than 1 % of pixels differing, mean absolute
difference below 0.5, with a diff image written to `outputs/pixel_diffs/` on failure. The mode
in use is printed by `pytest tests/test_env.py -s`.

One label in Figure 1 uses the *Light* face of Helvetica Neue. Apple's fonts cannot be
redistributed, so the `.ttf` is not in the repo: `pnc.utils.light_font_properties` looks for
`$PNC_HN_LIGHT_TTF`, then `figures/assets/private/HelveticaNeue-Light.ttf` (git-ignored; extract
it from `/System/Library/Fonts/HelveticaNeue.ttc` on macOS), and otherwise falls back to the
installed family at weight *light* (tolerance mode).

## 6. The pipeline behind the figures (`scripts/`)

| paper step | folder | runs from Tier A? |
|---|---|---|
| encoding-model fitting (features, PCA-750, RidgeCV layer sweep), layer selection, readout export, post-hoc prediction | `scripts/encoding/` | no (needs the calibration recordings + a GPU; readouts not shipped) |
| feature accentuation (parametric sweeps), MACO superstimuli, controversial accentuation | `scripts/synthesis/` | yes for the demos' reduced settings; full runs need a GPU |
| adversarial sensitivity of the encoding axes (PGD / FGSM, L-inf / L2), minimal-perturbation visualizations | `scripts/adversarial/` | attack outputs are shipped in `cluster_outputs/`; re-running needs a GPU |
| input-gradient maps and their radial Fourier spectra, participation ratio, refit diet ladder | `scripts/gradients/` | spectra are shipped; the 30 full gradient-map pickles are frozen inputs |
| ImageNet-val predictions through the 250 readouts | `scripts/imagenet/` | no (ImageNet-val); the cache is a frozen input |
| ResNet-50 embedding cloud (Fig. 1) | `scripts/embeddings/` | no (needs the full accentuated stimulus set) |
| producers of frozen csv inputs from the previous analysis tree | `scripts/provenance/` | partly (see its README) |
| preprocessing into `preprocessed_data/` | `scripts/preprocessing/` | **yes** |

Each folder's README lists, per script, what it computes, its inputs and outputs, which
`preprocessed_data`/`cluster_outputs` files it produced, and its GPU/SLURM requirements. The
scripts are the code we ran, with only path plumbing changed (original cluster paths are kept in
comments). `core/` and `neural_regress/` are the library code they import (feature hooks, model
zoo, GPU PCA/SRP, ridge regression, sklearn-to-torch conversion).

## 7. Demos (`notebooks/demos/`)

CPU-runnable at reduced scale with a `FULL_SCALE` switch documenting the paper settings:
`00_data_tour`, `01_fit_encoding_model`, `02_feature_accentuation`, `03_controversial_accentuation`,
`04_adversarial_robustness`, `05_gradient_spectra`.

## 8. Tests

```bash
python -m pytest -q                                   # everything that the present data allow
python -m pytest -q -m "not pixel and not preproc and not notebooks and not slow"   # no data needed
python -m pytest -q -m pixel                          # 55 figures vs the manuscript
python -m pytest -q -m preproc                        # rebuild preprocessed_data and compare
python -m pytest -q --run-raw                         # also validate the raw HDF5 sources
```

Beyond the figure and cache checks, `tests/` carries the scientific-validation suite of the
analysis pipeline: numerical kernels (outlier rejection, anchor-day standardization, noise
ceilings, spectral flatness), cache integrity (study dimensions, partitions, target schedules,
exclusion masks recomputed from their definitions), loader/statistic agreement, train/test
isolation, and gradient-cache consistency. Tests that need data skip cleanly when it is absent;
the suite never writes into `preprocessed_data/`.

## 9. Layout

```
pnc/                figure-side library: paths, study config + style (utils), trim, manifest, preproc loader + transforms
core/ neural_regress/   model zoo, feature hooks, GPU PCA/SRP, ridge fitting, sklearn->torch (library code)
figures/            main/ and supplementary/ figure scripts, assets/, CAPTIONS.md, render_all.py
notebooks/          figures/ (one per figure) and demos/
scripts/            preprocessing, encoding, synthesis, adversarial, gradients, imagenet, embeddings, provenance, cluster
data/               download_data.py, build_manifest.py, zenodo_upload.py, zenodo_manifest.json
tests/              pytest suite + reference/ (pixel hashes, cache hashes, frozen pnc API)
source_data/ preprocessed_data/ outputs/    data and renders (git-ignored)
```

## 10. Data hosting and citation

The data record on Zenodo is created as a draft by `data/zenodo_upload.py` (token from
`ZENODO_TOKEN`; the script never publishes) and published from the Zenodo web interface; the
record id is then written into `data/zenodo_manifest.json`. The software release is archived
separately through Zenodo's GitHub integration. Both DOIs and the citation are in `CITATION.cff`.
Data are released under CC BY 4.0, code under the MIT license (`LICENSE`, which also carries the
third-party notices).
