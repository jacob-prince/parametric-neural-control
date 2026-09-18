#!/usr/bin/env python3
"""
adv_robustness_attack.py - Continuous adversarial robustness of the 10-model family,
measured by attacking the 250 encoding-model readouts directly.

For one DNN backbone (--model), this attacks each of its 25 fitted readouts
(5 monkeys x 5 channels).  A readout is the paper's exact encoding pipeline:

    image[0,1]  --normalize-->  frozen backbone  --hook(ingraph)-->  layer feature
        --xtransform(JIT PCA)-->  750-d  --readout_vec @ . + bias-->  scalar prediction

The whole chain is a differentiable torch graph (backbone frozen, input carries grad),
so we run PGD in [0,1] pixel space to push each readout's predicted response as high
(up) and as low (down) as possible inside an epsilon-ball, under both L-inf and L2
threat models.  The bidirectional "swing" (up - down), normalized by the readout's
natural response range, is the continuous robustness measure: robust models yield a
small swing per unit epsilon.

We also record the cheap first-order input-gradient sensitivity at the clean image.

Usage (cluster, one model):
    python adv_robustness_attack.py --model resnet50 --device cuda --out <dir>

Smoke test (fast, 1 monkey / 1 channel / 2 images):
    python adv_robustness_attack.py --model resnet50 --device cuda --out <dir> --smoke
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import glob
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root(start):
    """Walk up to the repo root (.../AccentuateVVS) - robust to where this script is
    placed (analyses/24/scripts OR take8/figure8/cluster)."""
    p = start
    for _ in range(8):
        if (p / "prj_control_fa").exists() or (p / "Closed-loop-visual-insilico").exists():
            return p
        p = p.parent
    return start.parents[2]


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)           # .../AccentuateVVS


def _import_backbone_deps():
    """Heavy imports (timm/clip/open_clip/circuit_toolkit) done lazily so the PGD
    and metric helpers can be imported without the model stack installed."""
    sys.path.insert(0, str(SCRIPT_DIR.parent / "synthesis"))  # was PROJECT_ROOT / "prj_control_fa" (snapshot vendored in scripts/synthesis)
    from models import load_model                     # PNC: getpass user-gate patch removed (models.py reads PNC_MODEL_BACKBONES)
    from core.layer_hook_utils import featureFetcher  # was circuit_toolkit.layer_hook_utils
    return load_model, featureFetcher


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("PNC_SOURCE_DATA", PROJECT_ROOT / "data"))  # was PROJECT_ROOT / "data"
PCA_PROJ_PATH = DATA_ROOT / "image_pca_projections"
STIMULI_ROOT = DATA_ROOT / "stimuli_encoding"

MONKEY_DIRS = {
    "red":    "red_20250428-20250430",
    "paul":   "paul_20250428-20250430",
    "venus":  "venus_250426-250429",
    "leap":   "leap_250426-250501",
    "three0": "three0_250426-250501",
}
MONKEY_UNITS = {
    "red":    [0, 2, 9, 15, 19],
    "paul":   [0, 8, 24, 40, 47],
    "venus":  [9, 79, 151, 331, 355],
    "leap":   [81, 282, 286, 306, 342],
    "three0": [56, 74, 120, 168, 204],
}
MODELS = [
    "AlexNet_training_seed_01", "clipag_vitb32", "dinov2_vitb14_reg", "radio_v2.5-b",
    "regnety_640", "resnet50", "resnet50_clip", "resnet50_dino", "resnet50_robust",
    "siglip2_vitb16",
]

# epsilon grid expressed as per-pixel budget in /255 units, common to both norms.
#   L-inf radius (pixel [0,1]) = e/255
#   L2   radius (pixel [0,1]) = (e/255) * sqrt(N)   (== an RMS-per-pixel budget of e/255)
# so the L-inf and L2 sensitivity curves share one interpretable x-axis.
DEFAULT_EPS_255 = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
IMG_PX = 224
N_PIX = 3 * IMG_PX * IMG_PX


# ---------------------------------------------------------------------------
# Seed images -> a common 224x224 [0,1] canvas (same for every model)
# ---------------------------------------------------------------------------
_TO_CANVAS = T.Compose([
    T.ToTensor(),
    T.Resize((IMG_PX, IMG_PX), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
])


def _resolve_seed(rel_path):
    """Config seed paths look like 'shared1000/shared0575_nsd43157.png'; the files
    live flat in stimuli_encoding.  Try direct, then basename."""
    cand = STIMULI_ROOT / rel_path
    if cand.exists():
        return cand
    cand = STIMULI_ROOT / os.path.basename(rel_path)
    if cand.exists():
        return cand
    hits = glob.glob(str(STIMULI_ROOT / "**" / os.path.basename(rel_path)), recursive=True)
    return Path(hits[0]) if hits else None


def load_canvas(seed_paths, device):
    imgs, names = [], []
    for rp in seed_paths:
        fp = _resolve_seed(rp)
        if fp is None:
            print(f"    [miss-seed] {rp}", flush=True)
            continue
        img = Image.open(fp).convert("RGB")
        imgs.append(_TO_CANVAS(img))
        names.append(os.path.basename(rp))
    if not imgs:
        return None, None
    x = torch.stack(imgs).to(device).clamp_(0, 1)
    return x, names


# ---------------------------------------------------------------------------
# Per-model normalization (extracted from the model's own preprocess pipeline)
# ---------------------------------------------------------------------------
def make_normalizer(preprocess, device):
    """Return a differentiable [0,1]-image -> model-input normalizer.

    We attack a common 224x224 [0,1] canvas, so we skip the pipeline's Resize/Crop
    (already applied to the canvas) and keep only Normalize.  Models without a
    Normalize step (e.g. RADIO) get identity."""
    mean = std = None
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, T.Normalize):
            mean = torch.as_tensor(t.mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
            std = torch.as_tensor(t.std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    if mean is None:
        return lambda x: x
    return lambda x: (x - mean) / std


# ---------------------------------------------------------------------------
# Readout registry for one model
# ---------------------------------------------------------------------------
# Cluster storage roots (in configs) -> local data mirrors, for portable/local runs.
# On the cluster the absolute paths exist, so _localize is a no-op there.
_PATH_REMAP = {
    os.environ.get("PNC_READOUT_EXPORT_ROOT", "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/Encoding_model_outputs"):   # key = cluster path baked into the configs
        str(DATA_ROOT / "encoding_model_outputs" / "Encoding_model_outputs"),
}


def _localize(path):
    if os.path.exists(path):
        return path
    for cluster_root, local_root in _PATH_REMAP.items():
        if path.startswith(cluster_root):
            cand = path.replace(cluster_root, local_root, 1)
            if os.path.exists(cand):
                return cand
    return path


def load_readout(monkey_dir, unit, model, device):
    """Load one readout: config, xtransform (JIT), readout_vec/bias, and meta stats."""
    pkl = (PCA_PROJ_PATH / monkey_dir / "posthoc_model_predict_PCA_popul_unit" /
           f"posthoc_prediction_NSDencimg_PCA_pop_unit_{monkey_dir}_unit{unit}_{model}.pkl")
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    cfg = d["config"]
    xt = torch.jit.load(_localize(cfg["xtransform_path"]), map_location=device).to(device).eval()

    # meta gives the natural-response statistics used to normalize the swing.
    meta = torch.load(_localize(cfg["meta_path"]), weights_only=False, map_location="cpu")
    # channel label indexes directly into the population/meta arrays (verified for all
    # monkeys); guard against a mislabeled config rather than crashing the whole job.
    n_pop = len(np.asarray(meta["q99_resp"]).ravel())
    col = _channel_column(monkey_dir, unit, n_pop)
    if col >= n_pop:
        print(f"  [skip] {monkey_dir} Ch{unit}: channel {col} >= meta pop {n_pop}", flush=True)
        return None
    def g(k):
        return float(np.asarray(meta[k]).ravel()[col])

    return {
        "monkey_dir": monkey_dir, "unit": unit, "model": model,
        "layer": cfg["layer_name"],
        "xtransform": xt,
        "readout_vec": d["readout_vec"].to(device).float().ravel(),   # (750,)
        "readout_bias": float(d["readout_bias"]),
        "range_q99q01": g("q99_resp") - g("q01_resp"),
        "range_q95q05": g("q95_resp") - g("q05_resp"),
        "range_minmax": g("max_resp") - g("min_resp"),
        "std_resp": g("std_resp"),
        "d2_test": g("D2_per_unit_test"),
        "reliability": g("reliability"),
        "ncsnr": g("ncsnr"),
    }


def _channel_column(monkey_dir, unit, n_pop):
    """The readout channels for a monkey are the fixed MONKEY_UNITS list; meta arrays
    are indexed by the unit id within the population (0..n_pop-1)."""
    # unit ids in this project index directly into the population arrays.
    return int(unit)


# ---------------------------------------------------------------------------
# PGD  (bidirectional, L-inf and L2)
# ---------------------------------------------------------------------------
def _project(x, x0, eps, norm):
    delta = x - x0
    if norm == "linf":
        delta = delta.clamp(-eps, eps)
    else:  # l2, per-image budget
        flat = delta.view(delta.size(0), -1)
        n = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        factor = (eps / n).clamp(max=1.0)
        delta = (flat * factor).view_as(delta)
    return (x0 + delta).clamp(0, 1)


def _rand_start(x0, eps, norm):
    if norm == "linf":
        d = torch.empty_like(x0).uniform_(-eps, eps)
    else:
        d = torch.randn_like(x0)
        flat = d.view(d.size(0), -1)
        d = (flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12) * eps).view_as(x0)
    return (x0 + d).clamp(0, 1)


def pgd(objective, x0, eps, norm, sign, steps, alpha, restarts):
    """Maximize sign*objective within an eps-ball of x0.  Returns best predictions (B,)."""
    best = torch.full((x0.size(0),), -np.inf, device=x0.device)      # best sign*pred
    best_pred = objective(x0).detach()                                # start from clean
    best = torch.maximum(best, sign * best_pred)
    for t in range(restarts + 1):           # trajectory 0 = clean start, then random restarts
        x = x0.clone() if t == 0 else _rand_start(x0, eps, norm).detach()
        for _s in range(steps):
            x.requires_grad_(True)
            pred = objective(x)
            loss = (sign * pred).sum()
            grad, = torch.autograd.grad(loss, x)
            with torch.no_grad():
                if norm == "linf":
                    x = x + alpha * grad.sign()
                else:
                    flat = grad.view(grad.size(0), -1)
                    g = (flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)).view_as(grad)
                    x = x + alpha * g
                x = _project(x, x0, eps, norm)
                cur = objective(x)
                improved = sign * cur > best
                best = torch.where(improved, sign * cur, best)
                best_pred = torch.where(improved, cur, best_pred)
            x = x.detach()
    return best_pred.detach()


@torch.no_grad()
def clean_predict(objective, x0):
    return objective(x0).detach()


def grad_sensitivity(objective, x0):
    """First-order input-gradient norms at the clean image (per image)."""
    x = x0.clone().requires_grad_(True)
    pred = objective(x)
    grad, = torch.autograd.grad(pred.sum(), x)
    flat = grad.view(grad.size(0), -1)
    return flat.norm(p=2, dim=1).detach(), flat.norm(p=1, dim=1).detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_model(model_name, device, eps_255, steps, restarts, out_dir,
              monkeys=None, channels=None, n_images=10, smoke=False, image_list=None, tag="",
              attack="pgd"):
    if smoke:
        monkeys = ["red"]; channels = [9]; n_images = 2
        eps_255 = [4.0]; steps = 10; restarts = 1
    # FGSM = the single-step special case of the same bounded-ascent attack: one full-eps
    # gradient step from the clean image, no random restart. The metric definitions (bidirectional
    # swing, normalization, eps grid, both norms) are IDENTICAL to PGD; only the inner optimizer
    # changes, which is exactly the attack-method-invariance question.
    if attack == "fgsm":
        steps, restarts = 1, 0

    # Held-out validation: a single fixed image set used for EVERY readout (item 1).
    shared_x = shared_names = None
    if image_list:
        shared_x, shared_names = load_canvas(image_list, device)
        if shared_x is None:
            sys.exit("no held-out images resolved from --image-list")
        n_images = shared_x.size(0)
        print(f"  [held-out] {n_images} fixed images used for every readout", flush=True)

    print(f"=== model={model_name} device={device} eps(/255)={eps_255} attack={attack} "
          f"steps={steps} restarts={restarts} smoke={smoke} ===", flush=True)

    load_model, featureFetcher = _import_backbone_deps()
    model, preprocess = load_model(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    normalizer = make_normalizer(preprocess, device)

    fetcher = featureFetcher(model, input_size=(3, IMG_PX, IMG_PX),
                             print_module=False, store_device=device)
    recorded = set()

    eps_255 = list(eps_255)
    rows = []
    mk_list = monkeys or list(MONKEY_DIRS.keys())

    for monkey in mk_list:
        monkey_dir = MONKEY_DIRS[monkey]
        ch_list = channels if channels is not None else MONKEY_UNITS[monkey]
        for unit in ch_list:
            r = load_readout(monkey_dir, unit, model_name, device)
            if r is None:
                print(f"  [miss] {monkey} Ch{unit}", flush=True); continue
            layer = r["layer"]
            if layer not in recorded:
                fetcher.record(layer, ingraph=True, store_device=device)
                recorded.add(layer)

            # images: fixed held-out set (item 1) OR the per-readout synthesis seeds
            if image_list:
                x0, names = shared_x, shared_names
            else:
                seed_paths = _config_seeds(monkey_dir, unit, model_name)[:n_images]
                x0, names = load_canvas(seed_paths, device)
                if x0 is None:
                    print(f"  [miss-seeds] {monkey} Ch{unit}", flush=True); continue

            rv, rb = r["readout_vec"], r["readout_bias"]
            def objective(x, layer=layer, rv=rv, rb=rb, xt=r["xtransform"]):
                model(normalizer(x))
                return xt(fetcher[layer]) @ rv + rb

            clean = clean_predict(objective, x0).cpu().numpy()
            gL2, gL1 = grad_sensitivity(objective, x0)
            gL2, gL1 = gL2.cpu().numpy(), gL1.cpu().numpy()

            base = dict(model=model_name, monkey=monkey, monkey_dir=monkey_dir,
                        channel=int(unit), layer=layer, attack=attack,
                        range_q99q01=r["range_q99q01"], range_q95q05=r["range_q95q05"],
                        range_minmax=r["range_minmax"], std_resp=r["std_resp"],
                        d2_test=r["d2_test"], reliability=r["reliability"], ncsnr=r["ncsnr"])

            for norm in ("linf", "l2"):
                for e255 in eps_255:
                    if norm == "linf":
                        eps = e255 / 255.0
                    else:
                        eps = (e255 / 255.0) * np.sqrt(N_PIX)
                    # FGSM: one step of the full eps (projection makes this exactly x0 + eps*sign(g)
                    # for L-inf and x0 + eps*g/||g|| for L2). PGD: standard 2.5*eps/steps step size.
                    alpha = eps if attack == "fgsm" else 2.5 * eps / steps
                    up = pgd(objective, x0, eps, norm, +1.0, steps, alpha, restarts).cpu().numpy()
                    dn = pgd(objective, x0, eps, norm, -1.0, steps, alpha, restarts).cpu().numpy()
                    for i, nm in enumerate(names):
                        rows.append({**base, "img_idx": i, "img_name": nm,
                                     "norm": norm, "eps_255": e255,
                                     "clean_pred": float(clean[i]),
                                     "adv_up": float(up[i]), "adv_dn": float(dn[i]),
                                     "grad_l2": float(gL2[i]), "grad_l1": float(gL1[i])})
            print(f"  [done] {monkey} Ch{unit} layer={layer} "
                  f"clean_mean={clean.mean():.3f} range={r['range_q99q01']:.3f}", flush=True)

    fetcher.cleanup()
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_smoke" if smoke else "") + (f"_{tag}" if tag else "")
    out_path = out_dir / f"adv_robustness_{model_name}{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path}  ({len(df)} rows)", flush=True)
    return out_path


def _config_seeds(monkey_dir, unit, model):
    pkl = (PCA_PROJ_PATH / monkey_dir / "posthoc_model_predict_PCA_popul_unit" /
           f"posthoc_prediction_NSDencimg_PCA_pop_unit_{monkey_dir}_unit{unit}_{model}.pkl")
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return d["config"]["seed_image_paths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(SCRIPT_DIR.parent / "results"))
    ap.add_argument("--eps-255", default=",".join(str(e) for e in DEFAULT_EPS_255))
    ap.add_argument("--attack", default="pgd", choices=["pgd", "fgsm"],
                    help="pgd = multi-step iterative attack; fgsm = single-step (steps=1, restarts=0, alpha=eps)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--monkeys", default=None, help="comma list, default all")
    ap.add_argument("--channels", default=None, help="comma list, default per-monkey")
    ap.add_argument("--image-list", default=None,
                    help="txt file (one image path per line) used for ALL readouts - held-out validation")
    ap.add_argument("--tag", default="", help="suffix appended to the output filename (e.g. monkey, for sharded runs)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    eps_255 = [float(x) for x in args.eps_255.split(",")]
    monkeys = args.monkeys.split(",") if args.monkeys else None
    channels = [int(x) for x in args.channels.split(",")] if args.channels else None
    image_list = None
    if args.image_list:
        image_list = [ln.strip() for ln in open(args.image_list) if ln.strip()]

    run_model(args.model, args.device, eps_255, args.steps, args.restarts, args.out,
              monkeys=monkeys, channels=channels, n_images=args.n_images, smoke=args.smoke,
              image_list=image_list, tag=args.tag, attack=args.attack)


if __name__ == "__main__":
    main()
