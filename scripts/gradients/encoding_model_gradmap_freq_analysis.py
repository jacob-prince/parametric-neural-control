# ============================================================================
# PROVENANCE COPY (verbatim) of the upstream generator that produced the
# per-(site,model) gradient Fourier-profile caches:
#   Closed-loop-visual-insilico/scripts/encoding_model_gradmap_freq_analysis.py
#     (Binxu Wang)
#
# It computes, for each of the 10 natural seed images, the input gradient of the
# site's encoding readout d(pred)/d(image), then its radial Fourier power profile
# (grayscale = mean over channels; |fftshift(FFT)|^2; radial mean about the image
# center), and writes {profiles, freqs, bincounts, grad_img} per site-model to
#   .../posthoc_model_predict/encoding_gradient_map_fourier_spectra/*_grad_maps_freq_profiles.pkl
# whose grad_img + profiles are cached in take8 under preproc/cache/grad_maps/.
#
# RE-RUNNING THIS requires the Closed-loop-visual-insilico repo (core.fft_utils,
# core.posthoc_prediction_utils, circuit_toolkit), torch + a GPU, the accentuation
# configs, and the fitted encoding models -- i.e. it is NOT standalone in take8.
# The take8-standalone, reproducible part is preproc/fft_utils.py, which vendors the
# exact profile recipe and reproduces the cached `profiles` from the cached grad_img
# to ~1e-8 (grad_img -> gradient_radial_profile -> spectral_flatness).
# ============================================================================

import glob
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
import pickle as pkl
from tqdm.auto import tqdm

# Add toolkit root to path before imports that require it
# PNC: sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico") removed; core/ is importable from the repo.

from core.fft_utils import image_fourier_power, fourier_power_radial_profile_with_counts
from core.posthoc_prediction_utils import get_predictor_from_config
from core.plot_utils import saveallforms, to_imgrid  # was circuit_toolkit.plot_utils

seed_image_paths = [
    "shared1000/shared0575_nsd43157.png",
    "shared1000/shared0850_nsd61798.png",
    "shared1000/shared0968_nsd70194.png",
    "shared1000/shared0241_nsd20065.png",
    "shared1000/shared0160_nsd13231.png",
    "shared1000/shared0070_nsd07008.png",
    "shared1000/shared0055_nsd05879.png",
    "shared1000/shared0668_nsd48623.png",
    "shared1000/shared0488_nsd36979.png",
    "shared1000/shared0940_nsd68312.png"
]
shared_dir = os.environ.get("PNC_STIMULI_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/")  # was "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/" (must contain shared1000/)
seed_images_fullpaths = [join(shared_dir, path) for path in seed_image_paths]


subject_id = "red_20250428-20250430"
for subject_id in [
    'leap_250426-250501',
    'paul_20250428-20250430',
    'red_20250428-20250430',
    'three0_250426-250501',
    'venus_250426-250429'
]:
    config_root = os.environ.get("PNC_ACCENT_CONFIG_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs")  # was r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
    config_dir = join(config_root, subject_id)
    config_files = sorted(glob.glob(join(config_dir, "*.yaml")))
    print(subject_id, len(config_files))
    
    subject_dir = join(os.environ.get("PNC_ENCODING_MODEL_ROOT", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"), subject_id)  # was f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subject_id}"
    fft_dir = join(subject_dir, "posthoc_model_predict", "encoding_gradient_map_fourier_spectra")
    os.makedirs(fft_dir, exist_ok=True)
    for config_file in tqdm(config_files):
        # config_file = f'/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs/{subject_id}/{subject_id}_resnet50_clip_Ch2_accentuation_config.yaml'
        acc_config = yaml.safe_load(open(config_file))
        unit_ids = acc_config['unit_ids']
        assert len(unit_ids) == 1, "Only one unit is supported for now"
        unit_ids = unit_ids[0]
        model_name = acc_config['model_name']
        layer_name = acc_config['layer_name']
        population_predictor, target_unit_predictor, \
            model, transforms_pipeline, _, _, _ \
                = get_predictor_from_config(config_file, device="cuda")

        seed_images_torch = torch.stack([
            transforms_pipeline(Image.open(path).convert("RGB"))
            for path in seed_images_fullpaths
        ])
        # Ensure gradients flow from prediction back to the images:
        seed_images_torch = seed_images_torch.clone().detach().cuda()
        seed_images_torch.requires_grad_(True);
        pred_scores = target_unit_predictor(seed_images_torch)
        obj = pred_scores.sum()  # example objective (sum over features, mean over batch)
        obj.backward()
        grad_img_t = seed_images_torch.grad.detach().cpu()  # shape: (batch, 3, 224, 224)
        gradmtg = to_imgrid(0.5 + grad_img_t / grad_img_t.std(dim=(1,2,3), keepdim=True), nrow=5)
        # display(gradmtg)
        gradmtg.save(join(fft_dir, f"{subject_id}_unit_{unit_ids}_model_{model_name}_grad_maps.png"))

        profiles = []
        for i in range(len(grad_img_t)):
            grad_img_np = grad_img_t[i].permute(1, 2, 0).numpy()
            _, power_shift = image_fourier_power(grad_img_np, return_shifted_spectrum=True)
            prof, bincounts = fourier_power_radial_profile_with_counts(power_shift)
            profiles.append(prof)
        profiles = np.stack(profiles, axis=0)
        freqs = np.arange(profiles.shape[1])

        pkl.dump({
            "profiles": profiles,
            "freqs": freqs,
            "bincounts": bincounts,
            "grad_img": grad_img_t.numpy(),
        }, open(join(fft_dir, f"{subject_id}_unit_{unit_ids}_model_{model_name}_grad_maps_freq_profiles.pkl"), "wb"))

        # Plot the radial power profiles for all gradients
        plt.figure(figsize=(6, 5))
        for prof in profiles:
            plt.plot(freqs[1:], prof[1:], alpha=0.3, color="blue")
        plt.plot(freqs[1:], profiles.mean(axis=0)[1:], color="red", label="Mean profile", linewidth=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Radius (frequency bin)")
        plt.ylabel("Power")
        plt.title(f"Radial Fourier Power Profiles of Gradients\n{subject_id} unit {unit_ids} \n{model_name}  {layer_name}")
        plt.legend()
        plt.tight_layout()
        saveallforms(fft_dir, f"{subject_id}_unit_{unit_ids}_model_{model_name}_grad_maps_freq_profiles")
        # plt.show()
        plt.close("all")
