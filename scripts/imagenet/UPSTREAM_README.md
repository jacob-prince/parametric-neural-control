# cluster - ImageNet-val prediction extraction (self-contained)

Computes the 50k ImageNet-val encoding predictions per model (the natural-image cloud used by
the benchmarking figure). Runs on a GPU cluster. Nothing here imports code outside this tree - the model/hook code is vendored below.

## What it produces
`imagenet_pred_<model>.pkl` per model: `imagenet_predictions` (50000 x 25 sites), plus
`unit_monkeys` / `unit_ids`. Synced back in and folded into the preproc cache.

## Recipe (per model, per site = monkey x unit)
backbone -> hooked layer -> Xtfmer (JIT PCA) -> readout(vec, bias) -> predicted response.
- readout vec/bias + layer come from the **encoding cache** (`encoding_<monkey>.pkl`,
  pushed to the cluster), so predictions use this tree's own readouts.
- Xtfmer JIT + model backbone + ImageNet val come from the cluster.

## Vendored (copied in, do not import from elsewhere)
| file | contents |
|---|---|
| `models.py` | backbone loaders (username gate patched at runtime) |
| `models_utils.py` | alexnet/realnet builders |
| `layer_hook_utils.py` | `featureFetcher` layer-hook helper |
| `extract_imagenet_predictions.py` | ImageNet-val prediction extractor |

## Cluster facts
- a GPU cluster reached via an SSH host alias; project staged under the configured project dir
- stage dir: `<project>/imagenet` (created by `sync.sh push`)
- ImageNet val: the cluster's `imagenet1k-256` dataset
- Encoding_model_outputs (Xtfmer JIT), model_backbones under `<project>`
- env: a conda env under `<project>/conda_envs/` (torch 2.10, timm, open_clip) - call by
  FULL PATH; `conda activate` does NOT work (env not on the default conda path)
- SLURM: submit to a GPU partition / account you have access to

## Run
```
./sync.sh push                                    # code + encoding caches -> cluster
LIMIT=256 sbatch --array=5,8 submit_extract.sh    # small validation (resnet50, resnet50_robust)
./sync.sh pull ; python test_reproduce.py         # check vs reference npz
sbatch submit_extract.sh ; ./sync.sh pull         # full 50k, all 10 models
```
python build_imagenet_cache.py                    # fold imagenet_out/*.pkl -> preproc cache

## Changelog
- Built package. Established 50k imagenet preds are not stored raw anywhere; a surviving
  all-10 copy (`fig4_combined_<model>.npz`) is used here as the validation reference only.
  Vendored models.py / models_utils.py / layer_hook_utils.py; wrote extract/sync/submit/test.
- Fixed env (env not on conda path -> call env python by full path) and the SLURM
  account/partition. Ran a small validation (256 imgs, resnet50 + resnet50_robust).
- Env fix worked - small validation (256 imgs) PASSED bit-exact vs the reference for resnet50
  AND resnet50_robust (corr=1.00000, max|Δ|=0.0000). Confirms the recipe + this tree's readouts
  reproduce the reference. Launched FULL extraction (all 10 models x 50k). Wrote
  `build_imagenet_cache.py` to fold outputs into the preproc cache.
- Full 50k x 10-model extraction COMPLETE + validated bit-exact. Folded via build_imagenet_cache.py.

## Controversial SEM
`extract_controversial_sem.py` (h5py, no GPU) computes per-(stim,unit) SEM from the ephys h5
(the day3 normalize_red recording); run on the cluster, sync `controversial_sem.npz` into
`preproc/cache/`, and `run_preproc.build_controversial` folds it into `controversial.pkl`.
- Extracted controversial SEM (recomputed-mean vs repavg corr=1.00000); folded into
  controversial table, SEM 140/140 matches the reference (max|Δ|=0).
- Gradient Fourier profiles extracted on a COMPUTE node (CPU partition - no login-node compute)
  via extract_gradient_freq.py + submit_gradient.sh; pulled + build_gradient_cache.py ->
  preproc/cache/gradient_freq.pkl (250 records + 969 natural profiles); validated against an
  independent reference (max|Δ|=0). Aggregation logic also validated locally on the red-unit9
  subset. (Raw grad_img for figS31/Fig7A viz: red-unit9 local; cross-site grad_avg for figS32
  is a follow-up.)
