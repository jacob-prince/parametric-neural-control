#!/usr/bin/env python3
"""Controversial-stimulus synthesis: maximise (ResNet-50 prediction - robust ResNet-50 prediction).

Script conversion of the code cells of
    notebooks/015 Contreversial 11-01-2025 Lasso BIS.ipynb   (Thomas Fel, Jan 2025)
which produced the controversial stimuli shown to monkey red. The two encoding readouts
(ImageNet ResNet-50 and the L-inf adversarially-robust ResNet-50, layer4.Bottleneck1, PCA-1000 +
MultiTaskLassoCV, session red_20241212-20241220) are loaded, and for each target unit and each of
the 10 natural seed images a Feature-Accentuation optimisation (Hamblin et al.) is run with the
objective y_r50 - y_robust, i.e. images the standard model predicts to drive the site strongly
while the robust model predicts it will not.

Conversion notes (plumbing only; the objective, the FA optimiser and every hyperparameter are
verbatim from the notebook):
  * paths / units / seed images come from argparse instead of the notebook's literals; the
    notebook literals are the defaults (cluster paths -> PNC_* env vars, see below);
  * `from circuit_toolkit.layer_hook_utils import ...` -> `from core.layer_hook_utils import ...`;
  * the notebook's exploratory cells are not reproduced: the "Step 2" natural-image response
    range (Preds over all shared1000 images; computed but not used by the generation loop) is
    kept behind --natural-range, the "Step 4" single-unit hp-tuning preview loop (identical
    objective, unit best_unit_ids[0], no files written) is omitted, and the trailing shell cells
    (`!zip`, `!cp` to the lab share) are omitted;
  * figures are saved through matplotlib's Agg backend (the notebook used inline plotting).

Usage:
    python controversial_synthesis.py --shared1000 <dir with shared1000 pngs> \
        --robust-ckpt <model_backbones>/imagenet_linf_8_pure.pt --out-dir results_12-01-2025
"""
import argparse
import os

import cv2
import torch
import torch as th
import torch.nn as nn
import numpy as np
import pickle as pkl
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import resnet50
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.layer_hook_utils import featureFetcher_module, featureFetcher, get_module_names  # was circuit_toolkit.layer_hook_utils

from tqdm import tqdm
from functools import lru_cache

from einops import rearrange
from horama.common import recorrelate_colors, standardize
from horama.fourier_fv import optimization_step, fft_2d_freq, get_fft_scale
from horama import *
from horama.plots import *
from horama.plots import show


def set_size(w,h):
  """Set matplot figure size"""
  plt.rcParams["figure.figsize"] = [w,h]


# ---------------------------------------------------------------------------------------------
# Notebook literals (defaults of the CLI). The readouts came from the Dec-2024 MultiTaskLassoCV
# export of session red_20241212-20241220 on the cluster.
# ---------------------------------------------------------------------------------------------
_EXPORT_ROOT = os.environ.get("PNC_READOUT_EXPORT_ROOT_2024", "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_model_outputs")  # was the literal cluster path (Dec-2024 export dir)
_SESSION = "red_20241212-20241220"
DEFAULTS = dict(
    xtransform_r50_path=f"{_EXPORT_ROOT}/{_SESSION}/{_SESSION}_resnet50_Xtfmer_.layer4.Bottleneck1_pca1000_MultiTaskLassoCV.pkl",
    readout_r50_path=f"{_EXPORT_ROOT}/{_SESSION}/{_SESSION}_resnet50_readout_.layer4.Bottleneck1_pca1000_MultiTaskLassoCV.pth",
    xtransform_robust_path=f"{_EXPORT_ROOT}/{_SESSION}/{_SESSION}_resnet50_robust_Xtfmer_.layer4.Bottleneck1_pca1000_MultiTaskLassoCV.pkl",
    readout_robust_path=f"{_EXPORT_ROOT}/{_SESSION}/{_SESSION}_resnet50_robust_readout_.layer4.Bottleneck1_pca1000_MultiTaskLassoCV.pth",
    robust_ckpt=os.path.join(os.environ.get("PNC_MODEL_BACKBONES", "."), "imagenet_linf_8_pure.pt"),  # was th.load("imagenet_linf_8_pure.pt") from the cwd
    shared1000=os.environ.get("PNC_SHARED1000_DIR", "shared1000"),  # was the relative "shared1000" dir
)
layer_name = ".layer4.Bottleneck1"
best_unit_ids = [44,  1, 25, 15,  9, 16, 37]
fit_method_name = "MultiLassoCV"
day_string = "12-01-2025"

# Images used in the experiment
to_load = [
  'shared0631_nsd46161.png',
  'shared0131_nsd11160.png',
  'shared0196_nsd16467.png',
  'shared0147_nsd12066.png',
  'shared0491_nsd37225.png',
  'shared0485_nsd36911.png',
  'shared0862_nsd62480.png',
  'shared0255_nsd21193.png',
  'shared0974_nsd70506.png',
  'shared0189_nsd15794.png',
]

# FA hyperparameters of the production loop (notebook cell "Step 4" / generation cell)
noise, decay = 0.3, 2.5
lr = 5.0
FA_KWARGS = dict(total_steps=2048, image_size=768, model_input_size=224,
                 values_range=(-4.0, 4.0), crops_per_iteration=8, box_size=(0.90, 0.95), penalty=0.0)


# ---------------------------------------------------------------------------------------------
# Step 1 : Load the models and neural proxy
# ---------------------------------------------------------------------------------------------
class PCA_torch(torch.nn.Module):
    def __init__(self, pca: PCA, device="cpu"):
        super(PCA_torch, self).__init__()
        self.n_features = pca.n_features_in_
        self.n_components = pca.n_components
        self.mean = torch.from_numpy(pca.mean_).float().to(device)  # (n_features,)
        self.components = torch.from_numpy(pca.components_).float().to(device)  # (n_components, n_features)

    def forward(self, X):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        X = X - self.mean
        return torch.mm(X, self.components.T)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        return self


def load_models_and_readouts(xtransform_r50_path, readout_r50_path, xtransform_robust_path,
                             readout_robust_path, robust_ckpt, device):
    robust = resnet50(pretrained=False)
    robust.load_state_dict(th.load(robust_ckpt))
    robust = robust.eval().to(device)
    robust.requires_grad_(False)

    r50 = resnet50(pretrained=True)
    r50 = r50.eval().to(device)
    r50.requires_grad_(False)

    fetcher_robust = featureFetcher(robust, input_size=(3, 224, 224), print_module=False)
    fetcher_robust.record(layer_name,  ingraph=True, store_device=device)

    fetcher_r50 = featureFetcher(r50, input_size=(3, 224, 224), print_module=False)
    fetcher_r50.record(layer_name,  ingraph=True, store_device=device)

    state_dict = torch.load(readout_robust_path)
    readout_robust = nn.Linear(state_dict['weight'].shape[1], state_dict['weight'].shape[0], bias=True).to(device)
    readout_robust.load_state_dict(state_dict)

    state_dict = torch.load(readout_r50_path)
    readout_r50 = nn.Linear(state_dict['weight'].shape[1], state_dict['weight'].shape[0], bias=True).to(device)
    readout_r50.load_state_dict(state_dict)

    pca_robust = pkl.load(open(xtransform_robust_path, "rb"))
    xtransform_robust = PCA_torch(pca_robust, device=device)

    pca_r50 = pkl.load(open(xtransform_r50_path, "rb"))
    xtransform_r50 = PCA_torch(pca_r50, device=device)

    print('everything loaded correctly.')
    return dict(robust=robust, r50=r50, fetcher_robust=fetcher_robust, fetcher_r50=fetcher_r50,
                readout_robust=readout_robust, readout_r50=readout_r50,
                xtransform_robust=xtransform_robust, xtransform_r50=xtransform_r50)


# ---------------------------------------------------------------------------------------------
# Step 2: Get the range of values for each units using real images (notebook; optional here)
# ---------------------------------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = np.array(img)[...,::-1]
    img = img - img.min()
    img = img / img.max()
    img = torch.tensor(img).permute(2,0,1).float()
    return img


def load_seed_images(shared1000, names):
    imgs = []
    for p in names:
      imgs.append(load_image(f"{shared1000}/{str(p)}"))
    return imgs


def natural_image_predictions(M, shared1000, device):
    all_imgs = []
    for p in os.listdir(shared1000):
      try:
        all_imgs.append(load_image(f"{shared1000}/{str(p)}"))
      except:
        pass

    def all_neurons_predictions(images):
        M['robust'](images)
        feat_tsr = M['fetcher_robust'][layer_name]
        feat_vec = M['xtransform_robust'](feat_tsr)
        return M['readout_robust'](feat_vec)

    def normalize_imagenet(tensor):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1).to(device)
        return (tensor - imagenet_mean) / imagenet_std

    Preds = []
    for batch in range(0, len(all_imgs), 10):
        images = torch.stack(all_imgs[batch:batch+10]).to(device)
        images = normalize_imagenet(images)
        predictions = all_neurons_predictions(images)
        Preds += list(predictions.cpu().detach().numpy())

    Preds = np.array(Preds)
    return Preds


# ---------------------------------------------------------------------------------------------
# Step 3: Feature Accentuation loss (verbatim)
# ---------------------------------------------------------------------------------------------
def get_fft_scale2(width, height, decay_power=1.0):
    frequencies = fft_2d_freq(width, height)

    fft_scale = (1.0 / (frequencies+1e-10)) ** decay_power
    fft_scale = fft_scale * torch.sqrt(torch.tensor(width * height).float())

    return fft_scale.to(torch.complex64)[None, :, :]

@lru_cache(maxsize=8)
def get_color_correlation_svd_sqrt(device):
    return torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32, device=device
    )

class SpectrumBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spectrum_scaler):
        ctx.save_for_backward(spectrum_scaler)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (spectrum_scaler,) = ctx.saved_tensors
        g = grad_output * spectrum_scaler[None,:,:]
        return g, g

class SpectrumBackwardScaler(torch.nn.Module):
    def __init__(self, spectrum_scaler):
        super(SpectrumBackwardScaler, self).__init__()
        self.spectrum_scaler = spectrum_scaler.cuda()

    def forward(self, x):
        return SpectrumBackwardFunction.apply(x, self.spectrum_scaler)


def fa_preconditionner(spectrum, spectrum_backward_scaler, values_range, device):
    assert spectrum.shape[0] == 3

    spec_scaled = spectrum
    spec_scaled = spectrum_backward_scaler(spec_scaled)

    spatial_image = torch.fft.irfft2(spec_scaled)
    spatial_image = spatial_image - spatial_image.mean()

    color_recorrelated_image = recorrelate_colors(spatial_image, device)

    image = torch.sigmoid(color_recorrelated_image)
    image = image * (values_range[1] - values_range[0]) + values_range[0]

    return image


def inverse_image_spectrum(image, device=None):
    if device is None:
        device = image.device

    img_max = torch.amax(image)
    img_min = torch.amin(image)

    image = (image - img_min) / (img_max - img_min)

    eps = 1e-3
    inv_sigmoid = torch.log((image+eps) / (1.0 - image + eps))

    inv_corr_matrix = torch.linalg.pinv(get_color_correlation_svd_sqrt(device))
    inv_sigmoid = rearrange(inv_sigmoid, 'c h w -> h w c')

    inv_recorrelate = torch.matmul(inv_sigmoid.contiguous().view(-1, 3),
                                  inv_corr_matrix)
    inv_recorrelate = rearrange(inv_recorrelate, '(h w) c -> c h w', c=inv_sigmoid.shape[-1], h=inv_sigmoid.shape[0], w=inv_sigmoid.shape[1])
    spectrum = torch.fft.rfft2(inv_recorrelate)

    return spectrum


def fa(objective_function, objective_function2, image_seed, decay_power=1.5, total_steps=1000, learning_rate=1.0, image_size=1280,
        model_input_size=224, noise=0.05, values_range=(-2.5, 2.5),
        crops_per_iteration=6, box_size=(0.20, 0.25), penalty=1.0,
        device='cuda', objective_score = None):
    # perform the Feature Accentuation (Hamblin & al) optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    if image_seed.shape[1] != image_size:
        image_seed = torch.nn.functional.interpolate(image_seed.unsqueeze(0), size=(image_size, image_size), mode='bilinear', antialias=True)[0]

    spectrum = inverse_image_spectrum(image_seed, device)
    scaler = get_fft_scale(image_size, image_size, decay_power)
    scaler = scaler

    spectrum_scaler = SpectrumBackwardScaler(scaler)

    spectrum = spectrum.to(device)
    spectrum.requires_grad = True
    spectrum_scaler = spectrum_scaler.to(device)

    optimizer = torch.optim.NAdam([spectrum], lr=learning_rate, betas=(0.95, 0.9))
    transparency_accumulator = torch.zeros((3, image_size, image_size)).to(device)

    image_seed = fa_preconditionner(spectrum, spectrum_scaler, values_range, device).detach()

    model1_loss = []
    model2_loss = []

    best_image = None
    best_score = np.inf

    for step_id in tqdm(range(total_steps)):
        optimizer.zero_grad()

        image = fa_preconditionner(spectrum, spectrum_scaler, values_range, device)
        loss, img = optimization_step(objective_function, image, box_size,
                                      noise, crops_per_iteration, model_input_size)


        image_resized = torch.nn.functional.interpolate(img.unsqueeze(0), size=(model_input_size, model_input_size),
                                                        mode='bilinear', antialias=True)

        m1_score, m2_score = objective_function2(image_resized)
        model1_loss.append(m1_score.item())
        model2_loss.append(m2_score.item())

        if 1 < 2: # always save last one
            best_score = loss.item()
            best_image = image_resized

        loss.backward()
        transparency_accumulator += torch.abs(img.grad)

        optimizer.step()

    return best_image, transparency_accumulator, np.array(model1_loss), np.array(model2_loss)


# ---------------------------------------------------------------------------------------------
# Generation loop (notebook generation cell, verbatim objective + hyperparameters)
# ---------------------------------------------------------------------------------------------
def run(M, imgs, unit_ids, result_folder, device):
    robust, r50 = M['robust'], M['r50']
    fetcher_robust, fetcher_r50 = M['fetcher_robust'], M['fetcher_r50']
    readout_robust, readout_r50 = M['readout_robust'], M['readout_r50']
    xtransform_robust, xtransform_r50 = M['xtransform_robust'], M['xtransform_r50']

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for unit_id in unit_ids:

      def max_r50_objective(images, i=unit_id):
        # get robust readout
        robust(images)
        feat_tsr = fetcher_robust[layer_name]
        feat_vec = xtransform_robust(feat_tsr)
        y_robust = readout_robust(feat_vec)[:, i].mean()
        # get r50 readout
        r50(images)
        feat_tsr = fetcher_r50[layer_name]
        feat_vec = xtransform_r50(feat_tsr)
        y_r50 = readout_r50(feat_vec)[:, i].mean()
        return y_r50 - y_robust

      def objective_function2(images, i=unit_id):
          # get robust readout
          robust(images)
          feat_tsr = fetcher_robust[layer_name]
          feat_vec = xtransform_robust(feat_tsr)
          y_robust = readout_robust(feat_vec)[:, i].mean()
          # get r50 readout
          r50(images)
          feat_tsr = fetcher_r50[layer_name]
          feat_vec = xtransform_r50(feat_tsr)
          y_r50 = readout_r50(feat_vec)[:, i].mean()
          return y_robust, y_r50

      set_size(15, 15)
      for img_id in range(len(imgs)):

        img = torch.tensor(imgs[img_id])

        fv, alpha, rob_loss, r50_loss = fa(max_r50_objective, objective_function2, img.to(device), decay_power=decay,
                                  total_steps=FA_KWARGS['total_steps'], learning_rate=lr, image_size=FA_KWARGS['image_size'],
                                  model_input_size=FA_KWARGS['model_input_size'], noise=noise, values_range=FA_KWARGS['values_range'],
                                  crops_per_iteration=FA_KWARGS['crops_per_iteration'], box_size=FA_KWARGS['box_size'], penalty=FA_KWARGS['penalty'],
                                  device=device, objective_score = None)

        fv = fv.double()
        fv = fv.detach().cpu().numpy()
        fv = np.clip(fv, np.percentile(fv, 0.1), np.percentile(fv, 99.9))
        show(fv[0])

        filename = f'{result_folder}/controversial_max_r50_{fit_method_name}_unit_{unit_id}_img_{img_id}_srobust_{rob_loss[-1]}_sr50_{r50_loss[-1]}.png'
        plt.savefig(filename, dpi=350, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

      print('done for image', img_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtransform-r50", default=DEFAULTS["xtransform_r50_path"], help="sklearn PCA pickle (ResNet-50)")
    ap.add_argument("--readout-r50", default=DEFAULTS["readout_r50_path"], help="nn.Linear state_dict (.pth) (ResNet-50)")
    ap.add_argument("--xtransform-robust", default=DEFAULTS["xtransform_robust_path"], help="sklearn PCA pickle (robust ResNet-50)")
    ap.add_argument("--readout-robust", default=DEFAULTS["readout_robust_path"], help="nn.Linear state_dict (.pth) (robust ResNet-50)")
    ap.add_argument("--robust-ckpt", default=DEFAULTS["robust_ckpt"], help="imagenet_linf_8_pure.pt (L-inf eps=8/255 robust ResNet-50)")
    ap.add_argument("--shared1000", default=DEFAULTS["shared1000"], help="directory with the NSD shared1000 pngs")
    ap.add_argument("--seed-images", default=",".join(to_load),
                    help="comma list of seed image filenames inside --shared1000, or a .txt with one per line "
                         "(default: the 10 images used in the experiment)")
    ap.add_argument("--units", default=",".join(map(str, best_unit_ids)),
                    help=f"comma list of readout unit ids (default: {best_unit_ids})")
    ap.add_argument("--out-dir", default=f"results_{day_string}", help="output folder (notebook: results_<day_string>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--natural-range", action="store_true",
                    help="also compute the robust-readout predictions over ALL shared1000 images "
                         "(notebook Step 2; saved to <out-dir>/natural_preds.npy, not used by the synthesis)")
    args = ap.parse_args()

    if os.path.isfile(args.seed_images):
        names = [l.strip() for l in open(args.seed_images) if l.strip()]
    else:
        names = [s.strip() for s in args.seed_images.split(",") if s.strip()]
    unit_ids = [int(u) for u in args.units.split(",")]

    M = load_models_and_readouts(args.xtransform_r50, args.readout_r50, args.xtransform_robust,
                                 args.readout_robust, args.robust_ckpt, args.device)
    imgs = load_seed_images(args.shared1000, names)
    print(len(imgs), "seed images")

    if args.natural_range:
        os.makedirs(args.out_dir, exist_ok=True)
        Preds = natural_image_predictions(M, args.shared1000, args.device)
        np.save(os.path.join(args.out_dir, "natural_preds.npy"), Preds)
        print("natural-image predictions:", Preds.shape)

    run(M, imgs, unit_ids, args.out_dir, args.device)


if __name__ == "__main__":
    main()
