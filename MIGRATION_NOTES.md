# Migration notes

The initial camera-ready migration preserves the original module organization,
filenames, and behavior from `Closed-loop-visual-insilico` at commit `48ca561`.

## Deferred model and data path work

The following paths and download behavior are intentionally unchanged for now:

- `core/model_load_utils.py`
  - The robust ResNet-50 checkpoint is loaded from a hard-coded Kempner filesystem path.
  - The CLIPAG checkpoint directory is a hard-coded Kempner filesystem path.
  - The CLIPAG loader passes `device="cuda"` directly rather than using the requested device.
  - DINO, DINOv2, RADIO, SigLIP2, torchvision, CLIP, and RegNetY may download weights dynamically.
  - The AlexNet and ReAlNet branches append the legacy repository path to `sys.path`.
- `core/brainscore_model_utils.py`
  - The BrainScore cache root is hard-coded to the Kempner filesystem.
  - Missing AlexNet/ReAlNet weights are downloaded from S3.
  - The module appends the legacy repository path to `sys.path`.
- `core/posthoc_prediction_utils.py`
  - The module appends the legacy repository path to `sys.path`.
- `core/encoding_eval_utils.py`
  - Ephys/model roots and the default figure directory are hard-coded cluster paths.
  - `load_encoding_data()` currently constructs the `rw100-400` filename for every session.

Follow-up work should make these paths configurable, record checkpoint identifiers and
hashes, pin remote model revisions where possible, and preserve compatibility with the
existing exported artifacts.

## Deliberately excluded

- `core/ReAlNet.py`: ReAlNet is not in the active fitting, synopsis, or export model lists.
- `core/GAN_utils_hf.py`: legacy GAN visualization code is outside the current encoding pipeline.
