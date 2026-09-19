"""Shared helpers for the demo notebooks (notebooks/demos/*.ipynb). CPU-only, plain torch.

Three things live here so that the notebooks stay readable:

1. Readout loading + a differentiable encoding objective
   image [0,1] -> ImageNet normalize -> frozen backbone -> hooked layer -> PCA-750 (JIT) ->
   readout_vec @ . + bias  =  predicted standardized response of one site.
   The readout vector / bias / layer name come from the post-hoc prediction pickle
   (source_data/image_pca_projections/...); the PCA basis is the exported JIT transform
   (source_data/encoding_model_outputs/Encoding_model_outputs/<session>/..._JITscript.pt), the same
   file `scripts/adversarial/adv_robustness_attack.py` and `scripts/synthesis/generate_macos.py` load.

2. Feature accentuation (Hamblin et al.) in plain torch. `scripts/synthesis/feature_viz.py` is the
   production code, but it depends on the `horama` package (not installed in the reference env).
   `feature_accentuation` below is a line-by-line port of FeatureVisualizer._fa together with the
   three horama helpers it calls (get_fft_scale, recorrelate_colors, optimization_step), and `maco`
   ports horama.maco (the unconstrained super-stimulus used by scripts/synthesis/generate_macos.py).

3. Small image / target-level utilities shared by several notebooks.
"""
import os
import glob
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

from pnc import paths
from pnc.utils import MONKEY_DIRS, load_encoding_pickle

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# The 10 NSD seed images every accentuation sweep started from (scripts/synthesis/example_config.yaml).
SEED_NAMES = ["shared0575_nsd43157.png", "shared0850_nsd61798.png", "shared0968_nsd70194.png",
              "shared0241_nsd20065.png", "shared0160_nsd13231.png", "shared0070_nsd07008.png",
              "shared0055_nsd05879.png", "shared0668_nsd48623.png", "shared0488_nsd36979.png",
              "shared0940_nsd68312.png"]

DOWNLOAD_HINT = "python data/download_data.py --tier source   (README section 2)"


# ----------------------------------------------------------------------------------------------
# paths / images
# ----------------------------------------------------------------------------------------------
def stimulus_path(name):
    """Resolve a calibration stimulus name (as stored in the HDF5, e.g. 'BigAnimate_20.png') to the
    file in source_data/stimuli_encoding (some are shipped as .jpg)."""
    d = paths.source_data() / 'stimuli_encoding'
    stem = os.path.splitext(name)[0]
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG'):
        p = d / (stem + ext)
        if p.exists():
            return p
    hits = glob.glob(str(d / (stem + '.*')))
    if not hits:
        raise FileNotFoundError(f'{name} not found in {d}')
    return hits[0]


def load_image01(path, size=224):
    """PIL -> float tensor (3, size, size) in [0, 1], bicubic resize -- the seed-image loader of
    scripts/synthesis/run.py (load_seed_images)."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32).copy() / 255.0).permute(2, 0, 1).contiguous()


def normalize01(x):
    """[0,1] image batch -> ImageNet-normalized model input (differentiable)."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def to_display(x):
    """(3,H,W) or (1,3,H,W) tensor in [0,1] -> HWC numpy for imshow."""
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------------------------
# readouts and the encoding objective
# ----------------------------------------------------------------------------------------------
def export_dir(monkey):
    return paths.source_data() / 'encoding_model_outputs' / 'Encoding_model_outputs' / MONKEY_DIRS[monkey]


def find_export(monkey, unit, model, layer, kind):
    """kind='Xtfmer' -> the JIT PCA transform; kind='meta' -> the response-statistics pickle.
    Channel zero-padding differs between sessions (Ch09 vs Ch81), so glob like generate_macos.py."""
    d = export_dir(monkey)
    suffix = {'Xtfmer': f'Xtfmer_{layer}_pca750_RidgeCV_JITscript.pt',
              'meta': f'meta_{layer}_pca750_RidgeCV.pkl'}[kind]
    for ch in (f'Ch{unit:02d}', f'Ch{unit}', f'Ch{unit:03d}'):
        p = d / f'{MONKEY_DIRS[monkey]}_{model}_{ch}_{suffix}'
        if p.exists():
            return p
    return d / f'{MONKEY_DIRS[monkey]}_{model}_Ch{unit:02d}_{suffix}'      # -> paths.require will explain


def load_readout(monkey, unit, model, device='cpu'):
    """Everything that defines one (site, model) encoding axis, exactly as the production scripts
    load it: readout vector/bias + layer (post-hoc pickle), PCA basis (JIT Xtfmer), natural-response
    statistics (meta: q01/q99 etc. over the 969 calibration images, indexed by channel)."""
    d = load_encoding_pickle(monkey, unit, model)
    layer = d['config']['layer_name']
    xt_path = paths.require(find_export(monkey, unit, model, layer, 'Xtfmer'), DOWNLOAD_HINT)
    meta_path = paths.require(find_export(monkey, unit, model, layer, 'meta'), DOWNLOAD_HINT)
    xt = torch.jit.load(str(xt_path), map_location=device).eval()
    meta = torch.load(str(meta_path), weights_only=False, map_location='cpu')
    stats = {k: float(np.asarray(meta[k]).ravel()[unit]) for k in
             ('q01_resp', 'q99_resp', 'q05_resp', 'q95_resp', 'min_resp', 'max_resp',
              'mean_resp', 'std_resp', 'D2_per_unit_test', 'reliability')}
    return dict(monkey=monkey, unit=unit, model=model, layer=layer, xtransform=xt,
                readout_vec=d['readout_vec'].to(device).float().ravel(),
                readout_bias=float(d['readout_bias']), stats=stats, config=d['config'])


def load_backbone(model_name, device='cpu'):
    """core.model_load_utils.load_model_transform for 'resnet50' / 'resnet50_robust'; the robust
    checkpoint is looked up in source_data/model_backbones unless PNC_MODEL_BACKBONES is set."""
    from core.model_load_utils import load_model_transform
    if model_name == 'resnet50_robust':
        os.environ.setdefault('PNC_MODEL_BACKBONES', str(paths.source_data() / 'model_backbones'))
        paths.require(os.path.join(os.environ['PNC_MODEL_BACKBONES'], 'imagenet_linf_8_pure.pt'), DOWNLOAD_HINT)
    model, transform = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)                     # frozen: only the input image ever receives gradients
    return model, transform


class EncodingObjective:
    """Differentiable map: [0,1] image batch (B,3,224,224) -> predicted response (B,) of one site
    under one DNN encoding model. Holds the backbone, the ingraph layer hook (core.layer_hook_utils
    .featureFetcher), the JIT PCA transform and the linear readout."""

    def __init__(self, model_name, monkey, unit, device='cpu', backbone=None):
        from core.layer_hook_utils import featureFetcher
        self.device = device
        self.r = load_readout(monkey, unit, model_name, device)
        self.layer = self.r['layer']
        self.model = backbone if backbone is not None else load_backbone(model_name, device)[0]
        self.fetcher = featureFetcher(self.model, input_size=(3, 224, 224), device=device,
                                      print_module=False, store_device=device)
        self.fetcher.record(self.layer, ingraph=True, store_device=device)
        self.name = model_name

    def from_normalized(self, xn):
        """Prediction from an already-normalized model input (B,3,224,224)."""
        self.model(xn)
        feat = self.fetcher[self.layer]
        return self.r['xtransform'](feat) @ self.r['readout_vec'] + self.r['readout_bias']

    def __call__(self, x01):
        if x01.shape[-1] != 224 or x01.shape[-2] != 224:   # any canvas size in; the backbone always sees 224 px
            x01 = F.interpolate(x01, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        return self.from_normalized(normalize01(x01))

    @property
    def stats(self):
        return self.r['stats']

    def close(self):
        self.fetcher.cleanup()


def compute_unit_levels(lower, upper, extend_range=0.25, num_levels=11):
    """Target levels of a sweep, verbatim from scripts/synthesis/run.py: the natural range
    [q01, q99] is extended by extend_range*bandwidth below and 2*extend_range*bandwidth above."""
    bandwidth = upper - lower
    # equally spaced levels; the top end overshoots q99 twice as far as the bottom undershoots q01,
    # so a sweep probes super-natural responses more than sub-natural ones
    return np.linspace(lower - extend_range * bandwidth, upper + extend_range * bandwidth * 2, num_levels)


# ----------------------------------------------------------------------------------------------
# Feature accentuation -- plain-torch port of scripts/synthesis/feature_viz.py (+ horama helpers)
# ----------------------------------------------------------------------------------------------
COLOR_CORRELATION_SVD_SQRT = torch.tensor(
    [[0.56282854, 0.58447580, 0.58447580],
     [0.19482528, 0.00000000, -0.19482528],
     [0.04329450, -0.10823626, 0.06494176]], dtype=torch.float32)


def recorrelate_colors(image):
    """horama.common.recorrelate_colors: (3,H,W) decorrelated -> RGB-correlated."""
    C = COLOR_CORRELATION_SVD_SQRT.to(image.device)
    flat = image.permute(1, 2, 0).reshape(-1, 3)
    return (flat @ C).reshape(image.shape[1], image.shape[2], 3).permute(2, 0, 1)


def get_fft_scale(width, height, decay_power=1.0):
    """horama.fourier_fv.get_fft_scale: 1/f^decay scaler over the rfft2 grid."""
    freq_y = torch.fft.fftfreq(height).unsqueeze(1)
    freq_x = torch.fft.fftfreq(width)[:width // 2 + 1 + int(width % 2 == 1)]
    freqs = torch.sqrt(freq_x ** 2 + freq_y ** 2)
    scale = 1.0 / torch.maximum(freqs, torch.tensor(1.0 / max(width, height))) ** decay_power   # floor at the lowest resolvable frequency (no 1/0 at DC)
    scale = scale * math.sqrt(width * height)         # keeps the spatial amplitude independent of the canvas size
    return scale.to(torch.complex64)[None, :, :]


class _SpectrumBackward(torch.autograd.Function):
    """Identity forward; gradient multiplied by the 1/f scaler (feature_viz.SpectrumBackwardFunction).
    This is what makes accentuation a *preconditioned* ascent rather than a change of variables."""
    @staticmethod
    def forward(ctx, x, scaler):
        ctx.save_for_backward(scaler)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (scaler,) = ctx.saved_tensors
        # low frequencies receive the larger update, so the ascent favours coarse, natural-looking changes
        # over high-frequency adversarial texture; the forward image is untouched
        return grad_output * scaler, None


def fa_preconditioner(spectrum, scaler, values_range=(0.0, 1.0)):
    """feature_viz.fa_preconditionner: spectrum -> image in values_range."""
    spatial = torch.fft.irfft2(_SpectrumBackward.apply(spectrum, scaler))
    spatial = spatial - spatial.mean()               # zero-mean so the sigmoid is centred on mid-gray
    image = torch.sigmoid(recorrelate_colors(spatial))
    return image * (values_range[1] - values_range[0]) + values_range[0]


def inverse_image_spectrum(image):
    """feature_viz.inverse_image_spectrum: seed image (3,H,W) -> spectrum whose preconditioned
    reconstruction is (close to) the seed, so the optimization starts *at* the natural image."""
    image = (image - image.min()) / (image.max() - image.min())
    eps = 1e-3                                       # keeps the inverse sigmoid finite at 0 and 1
    inv_sigmoid = torch.log((image + eps) / (1.0 - image + eps))
    inv_C = torch.linalg.pinv(COLOR_CORRELATION_SVD_SQRT.to(image.device))
    flat = inv_sigmoid.permute(1, 2, 0).reshape(-1, 3) @ inv_C
    decorrelated = flat.reshape(image.shape[1], image.shape[2], 3).permute(2, 0, 1)
    return torch.fft.rfft2(decorrelated)


def optimization_step(objective, image, box_size, noise, crops_per_iteration, model_input_size):
    """horama.common.optimization_step: random crops (box_size fraction of the canvas, jittered
    centre) -> resize to the model input -> gaussian + uniform noise -> objective; loss = -mean score."""
    device = image.device
    image.retain_grad()                              # image is not a leaf (it comes from the spectrum); maco reads its grad
    n = crops_per_iteration
    # crops covering most of the canvas, with jittered centre and size: the score has to hold under small
    # shifts and rescales, which keeps the optimizer from exploiting one exact pixel alignment
    x0 = 0.5 + torch.randn(n, device=device) * 0.15
    y0 = 0.5 + torch.randn(n, device=device) * 0.15
    dx = torch.rand(n, device=device) * (box_size[1] - box_size[0]) + box_size[1]
    boxes = torch.stack([torch.zeros(n, device=device), x0 - dx / 2, y0 - dx / 2, x0 + dx / 2, y0 + dx / 2], 1) * image.shape[1]
    crops = roi_align(image.unsqueeze(0), boxes, output_size=(model_input_size * 2, model_input_size * 2)).squeeze(0)   # crop at 2x, then antialiased downsample
    crops = F.interpolate(crops, size=(model_input_size, model_input_size), mode='bicubic', align_corners=True, antialias=True)
    crops = crops + torch.randn_like(crops) * noise + (torch.rand_like(crops) - 0.5) * noise   # pixel noise: the score must also survive perturbation
    return -objective(crops).mean(), image


def feature_accentuation(objective, image_seed, target_level=None, noise=0.1, decay=1.5, total_steps=6000,
                         learning_rate=12.0, image_size=1024, model_input_size=224, values_range=(0.0, 1.0),
                         crops_per_iteration=8, box_size=(0.90, 0.95), device='cpu', tol=1e-2,
                         eval_fn=None, seed=0, verbose=True):
    """Port of FeatureVisualizer._fa (scripts/synthesis/feature_viz.py). Defaults = the production
    `fa_hyperparameters` of example_config.yaml.

    objective   : callable, [0,1] image batch -> per-image score (B,); the loss is -mean(score).
    target_level: if given, the run tracks |full-image score - target| and keeps the closest image;
                  once the score overshoots the target the loss is flipped and damped (x -0.1), and
                  the loop stops early when within `tol` (exactly as in feature_viz.py).
    eval_fn     : optional callable used for the full-image evaluation/tracking instead of `objective`
                  (the controversial notebook tracks both models' scores).
    Returns dict(image (3,S,S) in [0,1], score, history (per-step full-image scores), steps).
    """
    torch.manual_seed(seed)
    if image_seed.shape[-1] != image_size:
        image_seed = F.interpolate(image_seed.unsqueeze(0), size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)[0]
    spectrum = inverse_image_spectrum(image_seed.to(device)).detach().requires_grad_(True)
    scaler = get_fft_scale(image_size, image_size, decay).to(device)
    optimizer = torch.optim.NAdam([spectrum], lr=learning_rate, betas=(0.95, 0.9))
    eval_fn = eval_fn or objective
    history, best_delta, best_image, best_score = [], float('inf'), None, None
    for step in range(total_steps):
        optimizer.zero_grad()
        image = fa_preconditioner(spectrum, scaler, values_range)
        loss, img = optimization_step(objective, image, box_size, noise, crops_per_iteration, model_input_size)
        with torch.no_grad():                       # full-canvas evaluation (bilinear resize to the model input)
            full = eval_fn(F.interpolate(img.unsqueeze(0), size=(model_input_size, model_input_size),
                                         mode='bilinear', align_corners=False, antialias=True))
        full_score = float(full[0]) if torch.is_tensor(full) and full.ndim else float(full)
        history.append(full_score)
        if target_level is not None:
            delta = abs(full_score - target_level)
            if delta < best_delta:                  # keep the image whose full-canvas score is closest to the target
                best_delta, best_image, best_score = delta, img.detach().clone(), full_score
                if best_delta < tol:
                    break
            if full_score > target_level:           # overshot: descend instead, damped, so the score settles onto the target
                loss = -0.1 * loss
        loss.backward()
        optimizer.step()
        if verbose and (step % 25 == 0):
            print(f'  step {step:4d}  full-image score {full_score:+.3f}', flush=True)
    if best_image is None:                          # no target: return the last image
        best_image, best_score = img.detach().clone(), history[-1]
    return dict(image=best_image.clamp(0, 1).cpu(), score=best_score, history=np.array(history), steps=len(history))


# ----------------------------------------------------------------------------------------------
# MACO (Fel et al.) -- plain-torch port of horama.maco used by scripts/synthesis/generate_macos.py
# ----------------------------------------------------------------------------------------------
def natural_magnitude_template(image_paths, image_size):
    """MACO fixes the Fourier *magnitude* to a natural-image template and optimizes only the phase.
    horama ships a template computed over a large natural-image set (downloaded at run time); here the
    template is the mean |rfft2| over calibration images, per color channel, at the working canvas size."""
    acc = None
    for p in image_paths:
        x = load_image01(p, image_size)
        mag = torch.fft.rfft2(x).abs()
        acc = mag if acc is None else acc + mag
    return acc / len(image_paths)


def _standardize(t):
    return (t - t.mean()) / (t.std() + 1e-4)


def maco_preconditioner(magnitude, phase, values_range=(0.0, 1.0)):
    ph = _standardize(phase)
    spec = torch.complex(torch.cos(ph) * magnitude, torch.sin(ph) * magnitude)
    spatial = _standardize(torch.fft.irfft2(spec))
    image = torch.sigmoid(recorrelate_colors(spatial))
    return image * (values_range[1] - values_range[0]) + values_range[0]


def maco(objective, magnitude, total_steps=4096, learning_rate=0.1, image_size=2048, model_input_size=224,
         noise=0.08, values_range=(0.0, 1.0), crops_per_iteration=12, box_size=(0.2, 0.25), device='cpu',
         eval_fn=None, seed=0, verbose=True):
    """Defaults = DEFAULT_MACO_PARAMS of scripts/synthesis/generate_macos.py. Returns dict(image,
    score history of the full canvas, transparency = accumulated |d loss / d image|)."""
    torch.manual_seed(seed)
    magnitude = magnitude.to(device)
    phase = (torch.randn_like(magnitude)).requires_grad_(True)   # only the phase is free; the magnitude stays the natural template
    optimizer = torch.optim.NAdam([phase], lr=learning_rate)
    transparency = torch.zeros(3, image_size, image_size, device=device)
    eval_fn = eval_fn or objective
    history = []
    for step in range(total_steps):
        optimizer.zero_grad()
        image = maco_preconditioner(magnitude, phase, values_range)
        loss, img = optimization_step(objective, image, box_size, noise, crops_per_iteration, model_input_size)
        loss.backward()
        transparency += img.grad.abs()              # where the objective pushed pixels: the MACO transparency mask
        optimizer.step()
        with torch.no_grad():
            history.append(float(eval_fn(F.interpolate(img.detach().unsqueeze(0), size=(model_input_size, model_input_size),
                                                       mode='bilinear', align_corners=False, antialias=True))[0]))
        if verbose and (step % 25 == 0):
            print(f'  step {step:4d}  full-image score {history[-1]:+.3f}', flush=True)
    final = maco_preconditioner(magnitude, phase, values_range).detach().clamp(0, 1).cpu()
    return dict(image=final, history=np.array(history), transparency=transparency.detach().cpu())
