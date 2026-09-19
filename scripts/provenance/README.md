# scripts/provenance -- producers of frozen csv inputs

These three scripts generated csv files that the public pipeline consumes as *frozen inputs*
(`$PNC_SOURCE_DATA/frozen_inputs/`). Two of them depend on the private take8 analysis tree
(`preproc.loader`, `compute`, `figure8`, ...) that is **not** part of this repository; they are
shipped unchanged (no cluster paths inside; only relative take8 paths) so the exact computation is
on record, not as runnable stages.

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `bracket_bootstrap.py` | seed-level hierarchical bootstrap p-values for the gradient-CV vs adversarial-sensitivity bracket comparisons (resamples the ~10 accentuation seeds per model-site, recomputes control r / slope, residualises, recomputes the oriented gap) | take8 `fig9_master.csv`, `compute` (SX_predicting) and `figure8.load_heldout_logauc` modules, take8 preproc cache | `bracket_bootstrap_p.csv` | no (CPU, `python bracket_bootstrap.py [reps]`) | `frozen_inputs/fig5_bracket_bootstrap_p.csv` | **no** -- needs the take8 loader/compute modules (`preproc.loader`) |
| `build_predicting_master.py` | the site-model master predictor table (n = 250): canonical predictors from the preproc cache + adversarial sensitivity + accentuation low-frequency content + Lambda / phase reliance | take8 preproc cache (`preproc.loader`), `SX_advrobustness/intermediate`, `27_freq/results`, `29_diffmaps`, `30_imagenet_probe` tables | `fig9_master.csv` | no | `frozen_inputs/sup_predicting_master.csv` | **no** -- needs the take8 loader and several private intermediate tables |
| `fgsm_vs_pgd.py` | FGSM-vs-PGD attack-method invariance: per-readout / per-model agreement of the sensitivity metrics, eps-resolved agreement, and the control-prediction comparisons | PGD and FGSM `adv_robustness_<model>_<monkey>.csv` dirs + `verification_data.csv` (all public) | `fgsm_vs_pgd.txt`, `fgsm_vs_pgd_table.csv`, `fgsm_vs_pgd_eps.csv` | no | `frozen_inputs/sup_advrob_fgsm_vs_pgd_eps.csv` | **yes.** Made runnable with argparse defaults under `PNC_SOURCE_DATA` (`cluster_outputs/fig5_ext_heldout`, `cluster_outputs/fig5_ext_heldout_fgsm`, `frozen_inputs/fig5_verification_data.csv`; output dir `--out-dir` / `PNC_OUTPUT`). Verified: `PNC_SOURCE_DATA=<release> python fgsm_vs_pgd.py --out-dir <tmp>` reproduces `sup_advrob_fgsm_vs_pgd_eps.csv` byte-for-byte (md5 `4a3f146ec221067f0c04ad1850be26ca`). |
