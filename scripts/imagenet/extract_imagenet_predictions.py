#!/usr/bin/env python3
"""Extract 50k ImageNet-val encoding predictions per model (runs on the cluster GPU).

For a model backbone, runs every ImageNet val image through the 25 encoding pipelines
(5 monkeys x 5 units): backbone -> hooked layer -> Xtfmer (JIT PCA) -> readout ->
predicted response. This is the natural-image cloud used by the benchmarking figure.

Self-contained within take8/cluster/ (vendored models.py / models_utils.py /
layer_hook_utils.py). Sources are: the take8 encoding cache (readout vec/bias + layer,
synced to the cluster) + the cluster's Xtfmer JIT transforms + model backbones +
ImageNet val. Nothing outside take8 is imported.

  KMP_DUPLICATE_LIB_OK=TRUE python extract_imagenet_predictions.py --model resnet50 \
      --device cuda --enc-cache <dir of take8 encoding_*.pkl> \
      --enc-outputs <cluster Encoding_model_outputs> --imagenet-root <.../imagenet> \
      --out <dir>   [--limit N]   # --limit for the small-scale validation
"""
import os
import re
import sys
import glob
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                         # vendored models.py / layer_hook_utils.py
# PNC: getpass username-gate patch removed (models.py reads PNC_MODEL_BACKBONES)
from layer_hook_utils import featureFetcher

MONKEY_DIRS = {'red': 'red_20250428-20250430', 'paul': 'paul_20250428-20250430',
               'venus': 'venus_250426-250429', 'leap': 'leap_250426-250501',
               'three0': 'three0_250426-250501'}
MONKEY_UNITS = {'red': [0, 2, 9, 15, 19], 'paul': [0, 8, 24, 40, 47],
                'venus': [9, 79, 151, 331, 355], 'leap': [81, 282, 286, 306, 342],
                'three0': [56, 74, 120, 168, 204]}


class Unit:
    __slots__ = ('monkey', 'unit', 'layer', 'xt', 'rv', 'rb', 'col')

    def __init__(self, monkey, unit, layer, xt, rv, rb, col):
        self.monkey, self.unit, self.layer = monkey, unit, layer
        self.xt, self.rv, self.rb, self.col = xt, rv, rb, col


def _find_xtfmer(enc_out, monkey_dir, model, unit, layer):
    d = Path(enc_out) / monkey_dir
    for ch in (f"Ch{unit}", f"Ch{unit:02d}", f"Ch{unit:03d}"):
        p = d / f"{monkey_dir}_{model}_{ch}_Xtfmer_{layer}_pca750_RidgeCV_JITscript.pt"
        if p.exists():
            return str(p)
    for p in d.glob(f"{monkey_dir}_{model}_Ch*_Xtfmer_{layer}_pca750_RidgeCV_JITscript.pt"):
        m = re.search(r'_Ch(\d+)_', p.name)
        if m and int(m.group(1)) == unit:
            return str(p)
    return None


def build_registry(model, enc_cache, enc_out, device):
    """25 units with readout+layer from the take8 encoding cache and Xtfmer from the cluster."""
    units, col = [], 0
    for monkey, mdir in MONKEY_DIRS.items():
        enc = pickle.load(open(os.path.join(enc_cache, f'encoding_{monkey}.pkl'), 'rb'))
        mods = list(enc['models']); us = list(enc['units'])
        for uid in MONKEY_UNITS[monkey]:
            ro = enc['readout'].get(f'{uid}|{model}')
            if ro is None:
                print(f"  [miss] {monkey} Ch{uid} readout"); col += 1; continue
            xt_path = _find_xtfmer(enc_out, mdir, model, uid, ro['layer'])
            if xt_path is None:
                print(f"  [miss] {monkey} Ch{uid} Xtfmer ({ro['layer']})"); col += 1; continue
            xt = torch.jit.load(xt_path, map_location=device).to(device).eval()
            units.append(Unit(monkey, uid, ro['layer'], xt,
                              np.asarray(ro['vec'], np.float32).ravel(), float(ro['bias']), col))
            col += 1
    by_layer = defaultdict(list)
    for u in units:
        by_layer[u.layer].append(u)
    print(f"  registry {len(units)}/25 units, {len(by_layer)} layers")
    return units, by_layer


@torch.no_grad()
def run(dnn, fetcher, by_layer, dl, n, device):
    preds = np.full((n, 25), np.nan, np.float32)
    done = 0
    for bi, (imgs, idx) in enumerate(dl):
        dnn(imgs.to(device, non_blocking=True))
        for layer, ul in by_layer.items():
            feat = fetcher[layer]
            for u in ul:
                pca = u.xt(feat).detach().cpu().numpy()
                vals = (pca @ u.rv + u.rb).astype(np.float32)
                for i, gi in enumerate(idx):
                    preds[int(gi), u.col] = vals[i]
        done += len(imgs)
        if bi % 50 == 0:
            print(f"  batch {bi}: {done}/{n}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--enc-cache', required=True, help='dir with take8 encoding_<monkey>.pkl')
    ap.add_argument('--enc-outputs', required=True, help='cluster Encoding_model_outputs dir')
    ap.add_argument('--imagenet-root', required=True, help='dir containing val/')
    ap.add_argument('--backbones', required=True, help='model_backbones dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='cap #images (validation)')
    a = ap.parse_args()

    import models as M
    M.ckptroot = a.backbones
    units, by_layer = build_registry(a.model, a.enc_cache, a.enc_outputs, a.device)
    if not units:
        sys.exit("no units")

    dnn, preprocess = M.load_model(a.model, device=a.device); dnn.eval()
    fetcher = featureFetcher(dnn, input_size=(3, 224, 224), device='cpu', store_device='cpu')
    for layer in by_layer:
        fetcher.record(layer, ingraph=False, store_device=a.device)
    dnn.to(a.device); fetcher.device = a.device

    val = os.path.join(a.imagenet_root, 'val')
    val = val if os.path.isdir(val) else a.imagenet_root
    ds = ImageFolder(val, transform=preprocess)
    order = list(range(len(ds)))[: (a.limit or len(ds))]

    class Sub(torch.utils.data.Dataset):
        def __len__(self): return len(order)
        def __getitem__(self, k):
            img, _ = ds[order[k]]; return img, order[k]
    dl = DataLoader(Sub(), batch_size=a.batch_size, shuffle=False,
                    num_workers=a.num_workers, pin_memory=True)
    n = len(ds)
    preds = run(dnn, fetcher, by_layer, dl, n, a.device)

    meta_mk, meta_u = [], []
    for mk in MONKEY_DIRS:
        for u in MONKEY_UNITS[mk]:
            meta_mk.append(mk); meta_u.append(u)
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f'imagenet_pred_{a.model}{"_lim%d" % a.limit if a.limit else ""}.pkl')
    pickle.dump(dict(model=a.model, unit_monkeys=np.array(meta_mk), unit_ids=np.array(meta_u),
                     imagenet_predictions=preds, n_images_run=len(order)),
                open(outp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    fetcher.cleanup()
    print(f"saved -> {outp}  ({len(order)} images)")


if __name__ == '__main__':
    main()
