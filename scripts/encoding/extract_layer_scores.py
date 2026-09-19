#!/usr/bin/env python3
"""Extract the per-unit layer-sweep R^2 tables from the encoding-fit ``*_pred_meta.pkl`` files.

The production fit (neural_regression_massprod_VVS_multimonkey_20250428-20250430_PCA4All.py)
writes, per subject and model,
    <encoding-models>/<subj>/model_outputs_pca4all/<subj>_<model>_sweep_regressors_layers_pred_meta.pkl
        = {'pred_dict':              {((layer, dimred), regressor): ndarray (n_images, n_units)},
           'D2_per_unit_dict':       {} (unused),
           'D2_per_unit_train_dict': {((layer, dimred), regressor): ndarray (n_units,)},
           'D2_per_unit_test_dict':  {((layer, dimred), regressor): ndarray (n_units,)}}

This script keeps only the two per-unit R^2 dictionaries and writes
    <out-dir>/<subj>__<subj>_<model>_sweep_regressors_layers_D2perunit.pkl
        = {'D2_per_unit_test_dict': ..., 'D2_per_unit_train_dict': ...}
which is exactly what source_data/model_features/_layer_scores/ holds (checked against the
shipped red_20250428-20250430 / resnet50 pair: identical keys, array-equal values, pickle
protocol 4). ``D2_per_unit_test_dict[((layer, 'pca750'), 'RidgeCV')]`` is the cross-validated
test R^2 per unit that the layer-selection rule (argmax over layers) and
scripts/preprocessing/build_layer_selection.py consume.

Torch-free: the shipped pred_meta pickles hold numpy arrays only. If a pickle holds torch
tensors, torch must be importable to unpickle it; tensors are converted with ``.cpu().numpy()``.

Usage:
    python extract_layer_scores.py --encoding-models <root with <subj>/model_outputs_pca4all> \
        --out <dir> [--subjects red_20250428-20250430,...]
    python extract_layer_scores.py --model-outputs <one model_outputs_pca4all dir> \
        --subject red_20250428-20250430 --out <dir>
"""
import argparse
import glob
import os
import pickle

import numpy as np

SUFFIX = "_sweep_regressors_layers_pred_meta.pkl"
KEEP = ("D2_per_unit_test_dict", "D2_per_unit_train_dict")


def _to_numpy(v):
    if hasattr(v, "detach"):          # torch tensor
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def extract_one(pred_meta_path, subject, out_dir):
    with open(pred_meta_path, "rb") as f:
        pred_meta = pickle.load(f)
    extract = {k: {kk: _to_numpy(vv) for kk, vv in pred_meta[k].items()} for k in KEEP}
    base = os.path.basename(pred_meta_path)
    assert base.startswith(subject + "_") and base.endswith(SUFFIX), base
    model = base[len(subject) + 1:-len(SUFFIX)]
    out_path = os.path.join(out_dir, f"{subject}__{subject}_{model}_sweep_regressors_layers_D2perunit.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(extract, f, protocol=4)
    n_layers = len(extract["D2_per_unit_test_dict"])
    n_units = len(next(iter(extract["D2_per_unit_test_dict"].values()))) if n_layers else 0
    print(f"{subject:24s} {model:28s} {n_layers:3d} (layer,dimred,regressor) keys x {n_units} units -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--encoding-models", help="root containing <subject>/model_outputs_pca4all/ "
                                              "(default layout of the production fit)")
    g.add_argument("--model-outputs", help="a single model_outputs_pca4all directory (needs --subject)")
    ap.add_argument("--subject", default=None, help="subject id, required with --model-outputs")
    ap.add_argument("--subjects", default=None, help="comma list of subject ids to process (default: all found)")
    ap.add_argument("--subdir", default="model_outputs_pca4all", help="per-subject output subdir of the fit")
    ap.add_argument("--out", required=True, help="output dir (e.g. source_data/model_features/_layer_scores)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    if args.model_outputs:
        if not args.subject:
            ap.error("--subject is required with --model-outputs")
        jobs.append((args.subject, args.model_outputs))
    else:
        subjects = args.subjects.split(",") if args.subjects else sorted(
            d for d in os.listdir(args.encoding_models)
            if os.path.isdir(os.path.join(args.encoding_models, d, args.subdir)))
        jobs = [(s, os.path.join(args.encoding_models, s, args.subdir)) for s in subjects]

    n = 0
    for subject, d in jobs:
        files = sorted(glob.glob(os.path.join(d, f"{subject}_*{SUFFIX}")))
        if not files:
            print(f"[warn] no {SUFFIX} files for {subject} in {d}")
        for p in files:
            extract_one(p, subject, args.out)
            n += 1
    print(f"wrote {n} extract(s) to {args.out}")


if __name__ == "__main__":
    main()
