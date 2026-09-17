# scripts/gradients -- input-gradient Fourier spectra (flatness) of the encoding readouts

Sources: `take9/cluster/scripts_to_cluster/fig5_gradmaps/` (`encoding_model_gradmap_freq_analysis.py`,
`extract_gradient_freq.py`, `submit_gradient.sh`), `take8/cluster/` (`cluster_grad_gallery_avg.py`,
`pooled_gradmap_freq.py`) and `take8/supplementary/scripts/build_diet_refits.py`.
`build_gradient_cache.py` was already in this folder and is untouched.

Env vars: `PNC_ENCODING_MODEL_ROOT` (`<subj>/posthoc_model_predict/encoding_gradient_map_fourier_spectra/`),
`PNC_ACCENT_CONFIG_ROOT`, `PNC_STIMULI_ROOT` (dir containing `shared1000/`), `PNC_MODEL_BACKBONES`,
`PNC_ENCODING_STIM_DIR`; `submit_gradient.sh` also takes `PNC_PYTHON`, `PNC_GRAD_STAGE`.

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `encoding_model_gradmap_freq_analysis.py` | for every accentuation config (site x model): d(pred)/d(image) on the 10 natural seed images, then radial Fourier power profiles; saves `{profiles, freqs, bincounts, grad_img}` | accentuation configs (Tier A), readouts/Xtfmer (not public), backbones, seed images (`shared1000/`) | `<subj>/posthoc_model_predict/encoding_gradient_map_fourier_spectra/*_grad_maps_freq_profiles.pkl` | GPU | upstream of `cluster_outputs/fig5_grad_out` and `frozen_inputs/grad_maps` | no (readouts) |
| `extract_gradient_freq.py` | aggregates the 250 per-(site, model) profile pickles to mean/std per seed and adds the natural-image reference spectrum of the 969 encoding images | the pickles above, encoding stimuli | `gradient_freq.pkl` | CPU node, `submit_gradient.sh` | `cluster_outputs/fig5_grad_out/gradient_freq.pkl` -> `build_gradient_cache.py` | no (the per-site pickles are not public); output shipped |
| `cluster_grad_gallery_avg.py` | per (model, seed) average L2 saliency map over the 25 sites, normalised per map | the per-site `grad_img` pickles | `gradient_gallery_avg.pkl` | CPU | `frozen_inputs/sup_gradient_gallery_avg.pkl` | no; output shipped |
| `pooled_gradmap_freq.py` | same recipe as `adversarial/heldout_gradmap_freq.py` but with the pooled/diet-refit readouts (`--readouts` npz, raw PCA space) on the 100 held-out images | configs, readouts/Xtfmer, backbones, `diet_readouts_d{1,2,4}.npz` / pooled readouts, held-out images | `<subj>_unit<u>_<model>_heldout_gradfreq.pkl` | GPU | `cluster_outputs/sup_diet_gradfreq_d{1,2,3,4}` | no; outputs shipped |
| `build_diet_refits.py` | training-diet ladder (d0..d4) of RidgeCV refits of each readout on the raw 750-d PCA features; writes the ladder table and the refit readouts used by `pooled_gradmap_freq.py` | take8 preproc cache via `build_pooled_refit` (`preproc.loader`, `utils`) -- **not in this repo** | `diet_refit.csv`, `diet_readouts_d{1,2,4}.npz` | CPU | `frozen_inputs/sup_diet_refit.csv` | no: shipped for provenance; it imports the take8 `build_pooled_refit`/`preproc.loader` modules that are not part of the public repo |
| `submit_gradient.sh` | SLURM wrapper for `extract_gradient_freq.py` | - | - | SLURM | - | - |
| `build_gradient_cache.py` (pre-existing) | folds `gradient_freq.pkl` into the preprocessed cache | `cluster_outputs/fig5_grad_out` | preprocessed data | no | - | yes |
