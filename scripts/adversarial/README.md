# scripts/adversarial -- adversarial-sensitivity attacks on the encoding readouts

As-run from `analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig5_adv/` (the two FGSM job
scripts are the take8 originals, see below). Edits: `prj_control_fa` import path -> the vendored
snapshot in `scripts/synthesis/`; `circuit_toolkit` -> `core`; `DATA_ROOT` overridable by
`PNC_SOURCE_DATA`; `SHARED_DIR`, `_MBK`, `CONFIG_ROOT` overridable by `PNC_SHARED1000_DIR`,
`PNC_MODEL_BACKBONES`, `PNC_ACCENT_CONFIG_ROOT`; job scripts point at `scripts/adversarial`,
`scripts/cluster/config.sh` and `$NS` (`PNC_ADV_OUT`). The `#SBATCH --output/--account/--partition`
lines are cluster-specific and must be edited by hand.

`adv_robustness_attack.py` resolves a readout as: `image_pca_projections/<subj>/posthoc_model_predict_PCA_popul_unit/posthoc_prediction_NSDencimg_PCA_pop_unit_<subj>_unit<u>_<model>.pkl`
(readout_vec/bias + config) and then the config's `xtransform_path`/`meta_path`, remapped from the
cluster root to `$PNC_SOURCE_DATA/encoding_model_outputs/Encoding_model_outputs` when present.

| script / job | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `adv_robustness_attack.py` | for one backbone, attacks each of its 25 readouts with PGD (or single-step FGSM, `--attack fgsm`) in [0,1] pixel space under L-inf and L2 budgets (eps ladder 0.5..64/255), up and down; records swing normalised by the readout's natural response range, and the clean-image input-gradient norms | `image_pca_projections` (Tier A), `Xtfmer_*_JITscript.pt` + `meta_*.pkl` (not public), backbone weights, `validation_stimuli/heldout_paths.txt` (100 held-out NSD shared1000 images) | `adv_robustness_<model>_<monkey>.csv` | GPU, `jobs/adv_robustness_heldout*.sh`, `jobs/adv_heldout_lowext_*.sh` (array 0-49) | `cluster_outputs/fig5_ext_heldout`, `fig5_ext_heldout_fgsm`, `fig5_ext_heldout_lowextra`, `fig5_ext_heldout_fgsm_lowextra` | no (Xtfmer/meta not public; the held-out images are cluster paths). Outputs are shipped. |
| `adv_visual_attack.py` | qualitative attack gallery: ladder/bisection search for the minimal L-inf eps that moves one readout by a target delta, saving the perturbed images | same as above | `advvis_<monkey>_Ch<u>_seed<k>_<model>.npz` | GPU, `jobs/adv_visual_paul8.sh`, `jobs/adv_visual_paul8_bisect.sh` | `cluster_outputs/fig5_advvis_exact` | no (same reason) |
| `build_heldout_stimuli.py` | resolves the 87 strictly-unused + 13 face shared1000 indices to image paths and renders a montage | cluster `shared1000/` dir (`PNC_SHARED1000_DIR`) | `heldout_paths.txt`, `heldout100_montage.png` | no | `validation_stimuli/heldout_paths.txt` (shipped) | only with a local shared1000 copy; its output is shipped |
| `heldout_gradmap_freq.py` | input-gradient radial Fourier profiles of every (site, model) readout on the 100 held-out images (per-image profiles kept) | accentuation configs (Tier A), readouts/Xtfmer (not public), backbones, held-out images | `<subj>_unit<u>_<model>_heldout_gradfreq.pkl` | GPU, `jobs/heldout_gradmap.sh` | `cluster_outputs/fig5_heldout_gradfreq` | no |
| `test_pgd_logic.py` | CPU unit tests of the PGD/metric helpers against closed-form linear optima | - | - | no | - | yes (`python test_pgd_logic.py`, torch only) |
| `validation_stimuli/*.txt` | index lists / resolved paths of the held-out validation images | - | - | - | - | - |
| `config.example.sh` | placeholder copy of the cluster config sourced by the jobs (same file as `scripts/cluster/config.example.sh`) | - | - | - | - | - |

FGSM job scripts: the brief anticipated reconstructing `jobs/adv_robustness_heldout_fgsm.sh` and
`jobs/adv_heldout_lowext_fgsm.sh` from their PGD counterparts, but the originals exist in
`analyses/preprint_figures/take8/supplementary/SX_advrobustness/cluster/jobs/` and were copied
instead (the attack script and `config.reference.sh` are byte-identical between take8 and take9).
Their headers say so; only their repo-relative paths were moved to this layout.
