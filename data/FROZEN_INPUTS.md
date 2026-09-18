# Frozen inputs

Files under `source_data/frozen_inputs/` are inputs to the figure pipeline that have no
producer in this repository that can be re-run from the public Tier A data. They are shipped
exactly as used for the paper; `scripts/preprocessing/build_all.py` copies them into
`preprocessed_data/` unchanged (its BUILD_MANIFEST.json records their SHA-256).

| file | what it is | why frozen | producing code |
|---|---|---|---|
| `fig1_image_cloud_encoding.png` | Fig. 1 embedding cloud of the 969 calibration images (ResNet-50 avgpool, 50 PCs) | rendered 2026-07-20 by a working version of the cloud plotter; the current script (`figures/main/fig1_cloud.py`) is visually equivalent, not bit-identical | `figures/main/fig1_cloud.py` + `scripts/embeddings/compute_embeddings.py` |
| `fig1_resnet50_pc50.npz` | 50-PC scores of calibration + every 10th accentuated image | needs the complete 27,720-image accentuated set (not in Tier A) and a GPU | `scripts/embeddings/compute_embeddings.py` |
| `leap_variant_similarity.json` | pixel correlation of the 277 duplicated Leap synthesis variant pairs (merge audit used by `pnc.preproc.loader`) | the original audit script was not kept; the pairs it lists are re-derivable from the Leap sweeps | none (see loader docstring for the rule) |
| `fig5_verification_data.csv` | join of the cluster attack outputs with control outcomes and gradient predictors per readout | produced in the previous analysis tree from its live loader | `scripts/provenance/` (README) |
| `fig5_bracket_bootstrap_p.csv` | seed-level hierarchical bootstrap p-values for the Fig. 5 model brackets | needs the previous tree's loader | `scripts/provenance/bracket_bootstrap.py` |
| `sup_hyperparam_hp_from_logs.json` | selected synthesis hyperparameters (noise, decay) per site-model, parsed from the production SLURM logs | the cluster `.err` logs are not distributed | none (log parser); consumed by `scripts/preprocessing/build_hp_tuning_cache.py` |
| `sup_advform_heldout_descriptors.csv` | per-axis held-out gradient descriptors (alternative spectral summaries) | produced in the previous analysis tree | `scripts/provenance/` (README) |
| `sup_advrob_fgsm_vs_pgd_eps.csv` | FGSM vs PGD epsilon comparison table | re-runnable from `cluster_outputs/` with `scripts/provenance/fgsm_vs_pgd.py` (kept frozen so the shipped tree does not depend on that script) | `scripts/provenance/fgsm_vs_pgd.py` |
| `sup_diet_refit.csv` | readout refits across the five encoding training diets (d0-d4) | needs the 750-d PCA features of every accentuated stimulus and the cluster gradient jobs | `scripts/gradients/build_diet_refits.py`, `pooled_gradmap_freq.py` |
| `sup_gradient_gallery_avg.pkl` | site-averaged input-gradient saliency maps, 10 seeds x 10 models | needs the full 250 site-model gradient maps (cluster only) | `scripts/gradients/cluster_grad_gallery_avg.py` |
| `sup_predicting_master.csv` | 29-column master table of candidate control predictors | merges outputs of several previous-tree analyses (Lambda, phase reliance, ImageNet margin) | `scripts/provenance/build_predicting_master.py` |
| `imagenet_predictions.pkl` | 50k ImageNet-val predictions through the 250 readouts | needs ImageNet-val, the exported readouts and a GPU | `scripts/imagenet/` |
| `grad_maps/*.pkl` (30) | full input-gradient maps and spectra for 3 sites x 10 models | needs the fitted models on the cluster and a GPU | `scripts/gradients/encoding_model_gradmap_freq_analysis.py` |
| `fig6_imagenet_thumbs/*.JPEG` (183) | ImageNet-val thumbnails shown in Fig. 6 | ImageNet-val is not redistributed | manual selection by disagreement index |
