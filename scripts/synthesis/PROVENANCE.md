# Provenance of `scripts/synthesis/`

## Vendored feature-accentuation code (`run.py`, `feature_viz.py`, `models.py`, `models_utils.py`, `transforms_utils.py`)

| item | value |
|---|---|
| upstream repository | `https://github.com/fel-thomas/prj_control_fa` (Thomas Fel) |
| how it was used | git submodule `prj_control_fa/` of the private analysis repository (`AccentuateVVS`) |
| submodule commit | `7cd2e4b3a2b9b992c93844c56c457b525fce8e50` (`heads/main`, "add channel name to avoid name collision!") |
| `git submodule status` | ` 7cd2e4b3a2b9b992c93844c56c457b525fce8e50 prj_control_fa (heads/main)` |
| upstream visibility | **private** on GitHub (`gh repo view`: `isPrivate: true`, `licenseInfo: null`) |
| license | **none.** The submodule checkout contains no `LICENSE`/`COPYING` file, `readme.md` has no license statement, and GitHub reports no license for the repository. The files are therefore redistributed here only with the permission of their author (T. Fel, co-author of the paper); they are not covered by this repository's LICENSE unless the author agrees to that. |

### Local, uncommitted modifications that were present in the as-run working tree

The snapshot copied here is the *working tree* of the submodule, not the bare commit. `git diff`
against `7cd2e4b` showed these as-run changes (all in the original files, kept verbatim):

* `models.py`: `elif USERNAME == "jacobprince": raise NotImplementedError(...)` replaced by
  `ckptroot = "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones"`;
  `torch.load(...)` of `imagenet_linf_8_pure.pt` and `CLIPAG_ViTB32.pt` given `weights_only=False`.
* `models_utils.py`: `torch.load(...)` of the ReAlnet and Brain-Score AlexNet weights given
  `weights_only=False`.
* an untracked `neural_regress/` directory inside the submodule (a copy of the repo's
  `neural_regress/` package, used for the `Xtfmer` JIT export) -- not copied; the repo's
  `neural_regress/` package is the same code.

### Edits made for this public repository (plumbing only)

* `models.py`: added `import os`; the `getpass.getuser()` gate now takes `ckptroot` from
  `$PNC_MODEL_BACKBONES` when set, and otherwise falls back to the original per-user table; the
  `else: raise ValueError` message now tells the user to set `PNC_MODEL_BACKBONES`.
* `models_utils.py`: same `$PNC_MODEL_BACKBONES` override in front of the per-user `save_root` table
  (the as-run value for user `jacobprince` was the shared `model_backbones` directory).
* `feature_viz.py`: `from circuit_toolkit.layer_hook_utils import featureFetcher` ->
  `from core.layer_hook_utils import featureFetcher` (the repo's `core/` package vendors this file).
* `run.py`, `transforms_utils.py`: unchanged.

The identical `models.py` / `models_utils.py` in `scripts/imagenet/` (vendored there by the
ImageNet stage) received the same two edits.

## Other files in this folder

* `example_config.yaml` -- verbatim production accentuation config
  (`accentuation_configs/red_20250428-20250430/red_20250428-20250430_resnet50_robust_Ch9_accentuation_config.yaml`)
  with a comment header added; the absolute cluster paths inside are left as they were run.
* `generate_macos.py` -- `analyses/02_feature_accentuation_method/scripts/02c_generate_macos.py`
  (J. Prince). Edits: `_prj_dir` now points at this folder (vendored snapshot) instead of
  `../../../prj_control_fa`; `ckptroot`, `DATA_ROOT`, `FIGURES_ROOT` overridable by
  `PNC_MODEL_BACKBONES`, `PNC_SOURCE_DATA`, `PNC_MACO_OUTPUT_DIR`; `circuit_toolkit` -> `core`.
* `controversial_synthesis.py` -- script conversion of the code cells of
  `notebooks/015 Contreversial 11-01-2025 Lasso BIS.ipynb` (T. Fel). Objective, FA optimiser and
  hyper-parameters verbatim; paths/units/seed images via argparse; see its docstring for the list
  of notebook cells that were not carried over (exploratory range check, hp preview, `!zip`/`!cp`).
