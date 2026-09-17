#!/usr/bin/env bash
#SBATCH --job-name=heldout_flat
#SBATCH --account=kempner_konkle_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/heldout_flat_%A_%a.out
#SBATCH --error=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/heldout_flat_%A_%a.err

# item 1: gradient radial Fourier profiles (-> spectral flatness) on the 100 HELD-OUT images,
# same recipe as the seed caches. One monkey per array task.
# Submit from repo root: sbatch analyses/preprint_figures/take8/figure8/cluster/jobs/heldout_gradmap.sh

# PNC NOTE: the #SBATCH --output/--error paths above and --account/--partition are Harvard-cluster
# values; edit them for your cluster. Output root is $NS (override with PNC_ADV_OUT).
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${PROJECT_ROOT}/scripts/cluster/config.sh"  # was cluster/config.sh (copy scripts/cluster/config.example.sh -> config.sh)
module load python/3.12.5-fasrc01 2>/dev/null || module load python
source activate "$CONDA_PREFIX" 2>/dev/null || conda activate "$CONDA_PREFIX"
cd "$PROJECT_ROOT"

SUBJECTS=( leap_250426-250501 paul_20250428-20250430 red_20250428-20250430 three0_250426-250501 venus_250426-250429 )
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
F9=scripts/adversarial  # was analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig5_adv
NS=${PNC_ADV_OUT:?PNC_ADV_OUT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv

echo "======= heldout flatness | subject=$SUBJECT | $(hostname) | $(date) ======="
python $F9/heldout_gradmap_freq.py \
    --image-list "$F9/validation_stimuli/heldout_paths.txt" \
    --subjects "$SUBJECT" \
    --out-dir "$NS/heldout_gradfreq"
echo "======= done $(date) ======="
