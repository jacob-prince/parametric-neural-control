# scripts/encoding -- encoding-model fitting, layer selection, readout export, post-hoc prediction

As-run code from `Closed-loop-visual-insilico/scripts/` (Binxu Wang), edited only for plumbing:
notebook magics removed, `sys.path.append(<cluster clone>)` removed, `circuit_toolkit.*` imports
redirected to the repo's `core/` package, and every hard-coded cluster path routed through an
environment variable (`os.environ.get("PNC_*", <original path>)`, original kept in a `# was` comment).
The scripts are notebook-style (`# %%` cells, no CLI): edit the subject list / env vars and run
`python <script>.py` from the repo root with `core/` and `neural_regress/` importable.

| env var | meaning | as-run value |
|---|---|---|
| `PNC_EPHYS_DATA_ROOT` | dir of the `<subj>_vvs-encodingstimuli_*.h5` brain files | `/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data` |
| `PNC_ENCODING_STIM_DIR` | encoding stimuli + `encoding_stimuli_split_seed0*.csv` | `.../VVS_Accentuation/Stimuli/encodingstimuli_apr2025` |
| `PNC_ENCODING_MODEL_ROOT` | `<subj>/model_outputs_pca4all`, `<subj>/posthoc_model_predict*` | `.../VVS_Accentuation/Encoding_models` |
| `PNC_READOUT_EXPORT_ROOT` | exported readouts / Xtfmer JIT / meta | `.../Accentuate_VVS/Encoding_model_outputs` |
| `PNC_ACCENT_CONFIG_ROOT` | accentuation yaml configs | `.../Accentuate_VVS/accentuation_configs` |
| `PNC_ACCENT_OUTPUT_ROOT` | accentuated stimuli (`*_accentuation/` folders) | `.../Accentuate_VVS/accentuation_outputs` |
| `PNC_ACCENT_TEMPLATE_YAML` | config template | shipped here as `accentuation_template.yaml` |

Tier A = the public data release (brain HDF5s, `image_pca_projections`, `model_predictions`,
`_layer_scores`, accentuation configs, `cluster_outputs`, frozen inputs, `stimuli_encoding`, the
displayed `stimuli_control` images, `imagenet_linf_8_pure.pt`). It does **not** contain the
exported readouts/Xtfmer, the full set of accentuated stimuli, or the raw `model_outputs_pca4all`
fit outputs (only one `*_pred_meta.pkl`, red/resnet50, is shipped as a worked example).

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `neural_regression_massprod_VVS_multimonkey_20250428-20250430_PCA4All.py` | the production encoding fit: per subject x backbone, records every candidate layer's features on the 969 encoding images, reduces them (PCA-750 / SRP), fits RidgeCV per layer, scores per-unit train/test R^2 | brain HDF5 (`PNC_EPHYS_DATA_ROOT`), encoding stimuli + split csv, backbone weights (`core.model_load_utils`) | `<subj>/model_outputs_pca4all/<subj>_<model>_sweep_regressors_layers_{pred_meta.pkl, sweep_RidgeCV_df.pkl, ...}` | GPU | `model_features/<subj>/model_outputs_pca4all/*` (upstream of `_layer_scores`) | in principle yes: brain HDF5s + encoding stimuli are Tier A; needs a GPU and the backbone weights (only the robust ResNet-50 checkpoint is shipped; torchvision/timm/DINOv2/SigLIP/RADIO weights download; CLIPAG and Brain-Score AlexNet need their checkpoints) |
| `extract_layer_scores.py` (new, torch-free CLI) | extracts `{D2_per_unit_test_dict, D2_per_unit_train_dict}` from each `*_pred_meta.pkl` | `model_outputs_pca4all` dir(s) | `<subj>__<subj>_<model>_sweep_regressors_layers_D2perunit.pkl` | no | `model_features/_layer_scores/*` (verified array-equal on the shipped red/resnet50 pair) | yes for the shipped red/resnet50 `pred_meta.pkl`; the other 49 need the fit re-run |
| `neural_regress_yaml_export.py` | layer-selection rule (argmax per-unit test R^2 over layers, RidgeCV/pca750) for the 5 target units per subject, re-fits/exports the readout (`.pth`), the JIT-scripted feature transform (`Xtfmer_*.pt`) and meta stats, and writes one accentuation yaml per unit x model from `accentuation_template.yaml` | `model_outputs_pca4all` (`pred_meta` + regressor pickles), brain HDF5, encoding stimuli, backbone | `Encoding_model_outputs/<subj>/*_{readout,Xtfmer,meta}_*`, `accentuation_configs/<subj>/*.yaml` | GPU | `accentuation_configs/*` (Tier A); the readout exports (not public) | no: needs the full `model_outputs_pca4all` fit output; the resulting configs are shipped |
| `posthoc_prediction_unit_population_PCA_extraction.py` | fits the unit-population PCA basis on the encoding-image features and projects the accentuated stimuli; writes the per-(unit, model) `posthoc_prediction_*_PCA_pop_unit_*.pkl` (readout_vec, readout_bias, config, PCA scores) | accentuation configs + readouts/Xtfmer, all accentuated stimuli, brain HDF5, encoding stimuli | `<subj>/posthoc_model_predict_PCA_popul_unit/*.pkl` | GPU | `image_pca_projections/<subj>/posthoc_model_predict_PCA_popul_unit/*` (Tier A) | no: needs readouts/Xtfmer and the 83 GB accentuated-stimulus set |
| `posthoc_prediction_natimg_unit_population_PCA.py` | same PCA projection for the natural encoding images (`*_NSDencimg_PCA_pop_unit_*.pkl`, the pickles read by `generate_macos.py` and `adv_robustness_attack.py`) | configs + readouts/Xtfmer, encoding stimuli | `<subj>/posthoc_model_predict_PCA_popul_unit/posthoc_prediction_NSDencimg_PCA_pop_unit_*.pkl` | GPU | `image_pca_projections/*` (Tier A) | no (readouts/Xtfmer) |
| `feature_accentuation_model_posthoc_prediction.py` | runs every exported readout over every accentuated stimulus set -> predicted responses table per (subject, unit, model) | configs + readouts/Xtfmer, all accentuated stimuli, encoding stimuli | `<subj>/posthoc_model_predict/*.pkl` | GPU | `model_predictions/*` (Tier A) | no (readouts + all accentuated stimuli) |
| `neural_regress_posthoc_synopsis.py` | synopsis tables/plots of the layer sweep: per-unit best layer, R^2 vs reliability, cross-model comparison | `model_outputs_pca4all/*_pred_meta.pkl` for all models, brain HDF5 | `<subj>/synopsis/*_result_df_synopsis_*.csv` + figures | no (CPU) | `model_features/<subj>/synopsis/*.csv` | no: needs all `*_pred_meta.pkl` (only red/resnet50 shipped) |
| `accentuation_template.yaml` | template consumed by `create_accentuation_config()` in `neural_regress_yaml_export.py` | - | - | - | - | - |

Imports that could not be redirected: `circuit_toolkit.CNN_scorers.TorchScorer` and
`circuit_toolkit.GAN_utils.{upconvGAN, Caffenet}` have no counterpart in `core/`. They were unused
in the three post-hoc scripts that imported them, so the import lines are commented out with a note.
