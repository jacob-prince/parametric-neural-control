# scripts/synthesis -- accentuation, MACO super-stimuli, controversial stimuli

See `PROVENANCE.md` for the origin (private upstream `fel-thomas/prj_control_fa`, commit
`7cd2e4b3`, no license file) and the exact list of edits. Everything needs a GPU and the
`horama` feature-visualisation package (plus `timm`, `open_clip`, `clip`, `pyyaml`, `einops`,
`opencv-python`; see `scripts/cluster/requirements.txt`).

| env var | meaning |
|---|---|
| `PNC_MODEL_BACKBONES` | dir with `imagenet_linf_8_pure.pt`, `CLIPAG_ViTB32.pt`, Brain-Score AlexNet weights, ... (replaces the `getpass.getuser()` gate in `models.py`/`models_utils.py`) |
| `PNC_SOURCE_DATA` | public data root (`generate_macos.py` reads `image_pca_projections/` and `encoding_model_outputs/Encoding_model_outputs/` under it) |
| `PNC_MACO_OUTPUT_DIR` | where `generate_macos.py` writes `02c_<monkey>_<model>_Ch<u>_maco.png` |
| `PNC_READOUT_EXPORT_ROOT_2024`, `PNC_SHARED1000_DIR` | defaults for `controversial_synthesis.py` (Dec-2024 readout export dir; NSD shared1000 png dir) |

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `run.py` (+ `feature_viz.py`, `models.py`, `models_utils.py`, `transforms_utils.py`) | the accentuation synthesis used for every control experiment: `python run.py --config <yaml>` loads backbone + exported readout/Xtfmer/meta, tunes the FA hyper-parameters (`hp_tuning`), then for each unit x seed image optimises 11 target response levels (`num_levels`, range extended by `extend_range`) with the Fourier-parameterised Feature Accentuation objective; writes pngs (+ gifs, logs) | an accentuation yaml (see `example_config.yaml`): readout `.pth`, `Xtfmer_*_JITscript.pt`, `meta_*.pkl`, backbone checkpoint, 10 seed images (NSD shared1000 pngs) | `<result_folder>/<subj>_<model>_Ch<u>_*` accentuated images | GPU (one SLURM job per config; `scripts/cluster/`) | `stimuli_control/*` (Tier A ships only the displayed images) and, downstream, `model_predictions`, `image_pca_projections` | no: the exported readouts/Xtfmer/meta are not public (they are re-created by `scripts/encoding/neural_regress_yaml_export.py` after re-fitting) |
| `example_config.yaml` | verbatim production config (red, robust ResNet-50, unit 9) with a header comment; cluster paths inside must be edited | - | - | - | - | - |
| `generate_macos.py` | unconstrained MACO super-stimuli: for each monkey x model x unit builds the readout from the `posthoc_prediction_NSDencimg_PCA_pop_unit_*.pkl` encoding direction, hooks the layer, and runs `horama.maco` (4096 steps, 2048 px canvas, `DEFAULT_MACO_PARAMS`) | `image_pca_projections/*` (Tier A), `Xtfmer_*_JITscript.pt` (not public), backbone weights | `02c_<monkey>_<model>_Ch<u>_maco.png` | GPU | the MACO super-stimulus images in the figures | no (needs the Xtfmer JIT transforms) |
| `controversial_synthesis.py` | controversial stimuli for monkey red: FA optimisation of `y_ResNet50 - y_robustResNet50` for 7 units x 10 seed images (notebook `015 Contreversial 11-01-2025 Lasso BIS.ipynb`, converted; objective + hyper-parameters verbatim, argparse for paths/units/seeds) | Dec-2024 MultiTaskLassoCV readouts + PCA pickles for resnet50 / resnet50_robust (session red_20241212-20241220), `imagenet_linf_8_pure.pt`, shared1000 pngs | `results_12-01-2025/controversial_max_r50_MultiLassoCV_unit_<u>_img_<i>_srobust_<s>_sr50_<s>.png` | GPU | `stimuli_controversial/*` | no: the Dec-2024 readouts are not part of the release (the robust backbone is) |
