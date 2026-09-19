#!/usr/bin/env python3
"""
heldout_gradmap_freq.py - recompute the gradient radial Fourier profiles (-> spectral
flatness) on the HELD-OUT image set (item 1), using the SAME upstream recipe as the
seed-based caches so held-out flatness is directly comparable to the seed flatness in
verification_data.

Adapted from take8/cluster/encoding_model_gradmap_freq_analysis.py (provenance copy of
Binxu Wang's Closed-loop-visual-insilico generator): ONLY the image list and the output dir
change (+ image chunking for memory, + plotting dropped). Per-image profiles are saved so a
split-half image reliability of flatness can be computed downstream.

Runs on the cluster (needs Closed-loop-visual-insilico + fitted models + accentuation configs).
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

# AlexNet_training_seed_01 loads via brainscore, whose default cache dir (Binxu's holylfs06) is
# unwritable -> PermissionError on first config. The weight already lives in model_backbones, so
# force build_alexnet_brainscore to read it from there (no S3), matching the 19_bsf fix. Patch the
# caller's bound reference.
# load_model_transform does a local `from core.brainscore_model_utils import build_alexnet_brainscore`
# at call time, so patch it at the source module.
import core.brainscore_model_utils as _bsu
_MBK = os.environ.get("PNC_MODEL_BACKBONES", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones")  # was "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones"
_orig_alexbs = _bsu.build_alexnet_brainscore
_bsu.build_alexnet_brainscore = lambda identifier="training_seed_01", save_root=None: \
    _orig_alexbs(identifier=identifier, save_root=_MBK)

# resnet50_robust / clipag_vitb32 load their weights from hardcoded (string-literal) paths in
# Binxu's unwritable holylfs06 model_backbones. The same files exist (group-readable) in the
# accentuate model_backbones, so redirect any torch.load of that dir to the local copy.
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--overwrite", action="store_true", help="recompute even if the output pkl exists")
    args = ap.parse_args()

    paths = [ln.strip() for ln in open(args.image_list) if ln.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(paths)} held-out images -> {args.out_dir}", flush=True)

    for subject_id in args.subjects.split(","):
        cfgs = sorted(glob.glob(join(CONFIG_ROOT, subject_id, "*.yaml")))
        print(f"{subject_id}: {len(cfgs)} configs", flush=True)
        ok = 0
        for cf in cfgs:
            try:
                acc = yaml.safe_load(open(cf))
                unit = acc['unit_ids'][0]; model_name = acc['model_name']
                out = join(args.out_dir,
                           f"{subject_id}_unit_{unit}_model_{model_name}_grad_maps_freq_profiles.pkl")
                if os.path.exists(out) and not args.overwrite:
                    ok += 1; continue
                _, target_unit_predictor, _model, tfm, _, _, _ = get_predictor_from_config(cf, device="cuda")
                g = grad_for_images(target_unit_predictor, tfm, paths)      # (n_img, 3, 224, 224)
                profiles, bincounts = [], None
                for i in range(len(g)):
                    _, power_shift = image_fourier_power(g[i].permute(1, 2, 0).numpy(),
                                                         return_shifted_spectrum=True)
                    prof, bincounts = fourier_power_radial_profile_with_counts(power_shift)
                    profiles.append(prof)
                profiles = np.stack(profiles, 0)
                out = join(args.out_dir,
                           f"{subject_id}_unit_{unit}_model_{model_name}_grad_maps_freq_profiles.pkl")
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
