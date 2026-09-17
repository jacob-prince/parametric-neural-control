# scripts/ -- the as-run pre-figure pipeline

Every stage that produced the released data (`$PNC_SOURCE_DATA`) or the preprocessed caches, in
the order the paper's pipeline ran. All files are the code as it was run, with **plumbing edits
only**: notebook magics and `sys.path.append(<cluster clone>)` removed, `circuit_toolkit.*` ->
`core.*`, hard-coded cluster paths routed through `PNC_*` environment variables / argparse (the
original path is always kept in a trailing `# was ...` comment), and the `getpass.getuser()` gate of
the vendored `models.py` replaced by `PNC_MODEL_BACKBONES`. Each folder's `README.md` has a
per-script table (what it computes, inputs, outputs, GPU/SLURM, which public file it produced, and
whether it can be re-run from the public data).

| folder | stage | main outputs | re-runnable from the public (Tier A) data? |
|---|---|---|---|
| `encoding/` | encoding-model fitting (layer sweep, RidgeCV on PCA-750 features), layer selection + readout/Xtfmer export + accentuation-config generation, post-hoc predictions and PCA projections, layer-score extraction | `model_features/_layer_scores`, `accentuation_configs`, `image_pca_projections`, `model_predictions` | fit: yes in principle (GPU + backbone weights); export / post-hoc stages: no (need the private readout exports and all accentuated stimuli); `extract_layer_scores.py`: yes for the shipped `pred_meta` |
| `synthesis/` | feature-accentuation synthesis (vendored `prj_control_fa`, see `PROVENANCE.md`), MACO super-stimuli, controversial stimuli | `stimuli_control`, MACO images, `stimuli_controversial` | no (readouts/Xtfmer not public) |
| `adversarial/` | PGD / FGSM attacks on the 250 readouts on 100 held-out images, attack galleries, held-out gradient spectra | `cluster_outputs/fig5_ext_heldout*`, `fig5_advvis_exact`, `fig5_heldout_gradfreq` | no (readouts/Xtfmer); `test_pgd_logic.py` yes |
| `gradients/` | input-gradient Fourier spectra (flatness), pooled/diet refit spectra, saliency gallery | `cluster_outputs/fig5_grad_out`, `sup_diet_gradfreq_d*`, `frozen_inputs/sup_gradient_gallery_avg.pkl`, `sup_diet_refit.csv` | no (readouts / per-site pickles / take8 loader); `build_gradient_cache.py` yes |
| `imagenet/` | ImageNet-val predictions of every readout | `cluster_outputs/fig6_imagenet_out`, `frozen_inputs/imagenet_predictions.pkl` | no (ImageNet-val + Xtfmer); `build_imagenet_cache.py` yes |
| `embeddings/` | Fig. 1 ResNet-50 PCA image cloud | `frozen_inputs/fig1_resnet50_pc50.npz` | partly (encoding images only) |
| `provenance/` | producers of three frozen csv inputs | `fig5_bracket_bootstrap_p.csv`, `sup_predicting_master.csv`, `sup_advrob_fgsm_vs_pgd_eps.csv` | `fgsm_vs_pgd.py` yes (byte-identical); the other two need the private take8 loader |
| `cluster/` | SLURM/rsync plumbing (`config.example.sh`, `setup.sh`, `run.sh`, `requirements.txt`) | - | infrastructure |
| `preprocessing/` | (maintained separately) builds the preprocessed caches from the public data | preprocessed data | yes |

Tier A (public release): brain HDF5s, `image_pca_projections`, `model_predictions`, `_layer_scores`,
accentuation configs, `cluster_outputs`, frozen inputs, `stimuli_encoding`, the displayed
`stimuli_control` images, and the `imagenet_linf_8_pure.pt` backbone. **Not** public: the 83 GB of
all accentuated stimuli, the exported readouts / Xtfmer JIT transforms / meta pickles, and
ImageNet-val. Consequently every GPU stage downstream of the readout export can be *read* but not
re-run from the release; re-running them requires re-fitting the encoding models
(`encoding/neural_regression_massprod_*.py`) and re-exporting (`encoding/neural_regress_yaml_export.py`).

Environment variables used across the folders:

| variable | meaning |
|---|---|
| `PNC_SOURCE_DATA` | public data root (repo convention, `pnc/paths.py`) |
| `PNC_PREPROCESSED_DATA`, `PNC_OUTPUT` | preprocessed-cache / output roots (repo convention) |
| `PNC_EPHYS_DATA_ROOT`, `PNC_ENCODING_STIM_DIR`, `PNC_ENCODING_MODEL_ROOT`, `PNC_READOUT_EXPORT_ROOT`, `PNC_ACCENT_CONFIG_ROOT`, `PNC_ACCENT_OUTPUT_ROOT`, `PNC_ACCENT_TEMPLATE_YAML` | the cluster storage roots of the encoding / accentuation pipeline |
| `PNC_MODEL_BACKBONES` | backbone checkpoint dir (replaces the username gate in `models.py`) |
| `PNC_SHARED1000_DIR`, `PNC_STIMULI_ROOT` | NSD shared1000 seed/held-out images |
| `PNC_ADV_OUT`, `PNC_GRAD_STAGE`, `PNC_IMAGENET_STAGE`, `PNC_IMAGENET_ROOT`, `PNC_PYTHON`, `PNC_REPO_ROOT`, `PNC_MACO_OUTPUT_DIR`, `PNC_READOUT_EXPORT_ROOT_2024` | per-job-script settings required by the `.sh` wrappers (`${PNC_X:?...}`; the as-run values are in `# was` comments) |

Dependencies beyond `requirements-scripts.txt`: the GPU stages need `timm`, `open_clip_torch`,
`clip` (OpenAI), `horama`, `pyyaml`, `einops`, `opencv-python`, `boto3` (`cluster/requirements.txt`).
