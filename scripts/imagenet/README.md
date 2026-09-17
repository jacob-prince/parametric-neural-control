# scripts/imagenet -- ImageNet-val predictions of the 250 encoding readouts

Source: `take9/cluster/scripts_to_cluster/fig6_imagenet/` (its own README is kept as
`UPSTREAM_README.md`). `models.py` / `models_utils.py` are the same vendored `prj_control_fa`
files as in `scripts/synthesis/` (same `PNC_MODEL_BACKBONES` edit; see `../synthesis/PROVENANCE.md`);
`layer_hook_utils.py` is the vendored `featureFetcher` (differs from `core/layer_hook_utils.py`
only by an `input_size` pass-through). `build_imagenet_cache.py` was already here and is untouched.

`submit_extract.sh` env vars: `PNC_PYTHON`, `PNC_IMAGENET_STAGE` (dir holding the script and the
`encoding_<monkey>.pkl` caches), `PNC_READOUT_EXPORT_ROOT`, `PNC_MODEL_BACKBONES`, `PNC_IMAGENET_ROOT`.

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `extract_imagenet_predictions.py` | runs all 50k ImageNet-val images through the 25 readouts of one backbone (backbone -> hooked layer -> Xtfmer JIT -> readout) | `--enc-cache` (`encoding_<monkey>.pkl`: readout vec/bias + layer), `--enc-outputs` (Xtfmer JIT, not public), `--imagenet-root` (ImageNet val, not public), `--backbones` | `imagenet_pred_<model>.pkl` (50000 x 25) | GPU, `submit_extract.sh` (array over 10 models) | `cluster_outputs/fig6_imagenet_out/*` -> `build_imagenet_cache.py` -> `frozen_inputs/imagenet_predictions.pkl` | no: needs ImageNet-val and the Xtfmer transforms; outputs are shipped |
| `submit_extract.sh` | SLURM array wrapper (`LIMIT=N` for a validation subset) | - | - | SLURM | - | - |
| `build_imagenet_cache.py` (pre-existing) | folds the per-model pickles into the preprocessed cache | `cluster_outputs/fig6_imagenet_out` | preprocessed data | no | - | yes |
