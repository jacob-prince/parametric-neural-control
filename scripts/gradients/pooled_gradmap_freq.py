#!/usr/bin/env python3
"""
pooled_gradmap_freq.py - gradient radial Fourier profiles for the POOLED-REFIT readouts
(figS66, slug pooled_refit), on the same 100 held-out images and with the same recipe as
heldout_gradmap_freq.py, so pooled-refit spectral CoV is directly comparable to the original
encodings' held-out CoV.

Identical to heldout_gradmap_freq.py except the readout: instead of the synthesis-time readout
inside get_predictor_from_config's predictor, the prediction is Xtransform(features) @ w_new,
where w_new is the pooled-refit readout in RAW PCA space (coef_/scale_ from
build_pooled_refit.py; bias omitted - constant, no gradient). Everything upstream (backbone,
layer, featureFetcher, PCA transform, image transform) comes from the accentuation config
exactly as before.

Extra arg vs heldout script: --readouts (npz keyed '<subject>_unit_<u>_model_<model>').
"""
import argparse, glob, os, sys
from os.path import join
import numpy as np
import torch
import yaml
from PIL import Image
import pickle as pkl

# PNC: the Closed-loop-visual-insilico sys.path loop was removed; core/ is importable from the repo.
# posthoc_prediction_utils imports seaborn only for (unused-here) plotting; the accentuate env
# lacks it, so stub it with a no-op mock so the predictor path imports cleanly.
from unittest.mock import MagicMock
sys.modules.setdefault("seaborn", MagicMock())
from core.fft_utils import image_fourier_power, fourier_power_radial_profile_with_counts
from core.posthoc_prediction_utils import get_predictor_from_config

# Same weight-path fixes as heldout_gradmap_freq.py (brainscore AlexNet cache dir + hardcoded
# robust/CLIPAG weight paths in Binxu's unwritable model_backbones).
import core.brainscore_model_utils as _bsu
_MBK = os.environ.get("PNC_MODEL_BACKBONES", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones")  # was "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones"
_orig_alexbs = _bsu.build_alexnet_brainscore
_bsu.build_alexnet_brainscore = lambda identifier="training_seed_01", save_root=None: \
    _orig_alexbs(identifier=identifier, save_root=_MBK)

_UPSTREAM_MBK = os.environ.get("PNC_UPSTREAM_MODEL_BACKBONES", "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones")  # was _BINXU_MBK; match-prefix key of the paths baked into core/model_load_utils, redirected to _MBK
_orig_torch_load = torch.load
def _redirect_load(f, *a, **k):
    try:
        fs = os.fspath(f)
        if fs.startswith(_UPSTREAM_MBK):
            cand = os.path.join(_MBK, os.path.basename(fs))
            if os.path.exists(cand):
                f = cand
    except TypeError:
        pass
    return _orig_torch_load(f, *a, **k)
torch.load = _redirect_load

SUBJECTS = ['leap_250426-250501', 'paul_20250428-20250430', 'red_20250428-20250430',
            'three0_250426-250501', 'venus_250426-250429']
CONFIG_ROOT = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"


def grad_for_images(predictor, tfm, paths, chunk=24):
    """d(pred)/d(image) per image, chunked over images to bound GPU memory."""
    grads = []
    for s in range(0, len(paths), chunk):
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[s:s + chunk]])
        imgs = imgs.clone().detach().cuda().requires_grad_(True)
        predictor(imgs).sum().backward()
        grads.append(imgs.grad.detach().cpu())
    return torch.cat(grads, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-list", required=True)
    ap.add_argument("--readouts", required=True, help="pooled_readouts.npz from build_pooled_refit.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--overwrite", action="store_true", help="recompute even if the output pkl exists")
    args = ap.parse_args()

    paths = [ln.strip() for ln in open(args.image_list) if ln.strip()]
    readouts = np.load(args.readouts)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(paths)} held-out images, {len(readouts.files)} pooled readouts -> {args.out_dir}", flush=True)

    for subject_id in args.subjects.split(","):
        cfgs = sorted(glob.glob(join(CONFIG_ROOT, subject_id, "*.yaml")))
        print(f"{subject_id}: {len(cfgs)} configs", flush=True)
        ok = 0
        for cf in cfgs:
            try:
                acc = yaml.safe_load(open(cf))
                unit = acc['unit_ids'][0]; model_name = acc['model_name']
                layer_name = acc['layer_name']
                key = f"{subject_id}_unit_{unit}_model_{model_name}"
                out = join(args.out_dir, f"{key}_grad_maps_freq_profiles.pkl")
                if os.path.exists(out) and not args.overwrite:
                    ok += 1; continue
                if key not in readouts:
                    print(f"  [SKIP] no pooled readout for {key}", flush=True)
                    continue
                _, _, model, tfm, fetcher, Xtransform, _readout = \
                    get_predictor_from_config(cf, device="cuda")
                w = torch.tensor(np.asarray(readouts[key]), dtype=torch.float32, device="cuda")

                def pooled_predictor(images):
                    model(images)
                    feat_vec = Xtransform(fetcher[layer_name])
                    return feat_vec @ w

                g = grad_for_images(pooled_predictor, tfm, paths)      # (n_img, 3, 224, 224)
                profiles, bincounts = [], None
                for i in range(len(g)):
                    _, power_shift = image_fourier_power(g[i].permute(1, 2, 0).numpy(),
                                                         return_shifted_spectrum=True)
                    prof, bincounts = fourier_power_radial_profile_with_counts(power_shift)
                    profiles.append(prof)
                profiles = np.stack(profiles, 0)
                pkl.dump({"profiles": profiles, "freqs": np.arange(profiles.shape[1]),
                          "bincounts": bincounts, "image_paths": paths}, open(out, "wb"))
                ok += 1
            except Exception as e:
                print(f"  [FAIL] {os.path.basename(cf)}: {type(e).__name__}: {e}", flush=True)
                continue
        print(f"  {subject_id} done ({ok}/{len(cfgs)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
