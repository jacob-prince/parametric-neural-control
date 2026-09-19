#!/usr/bin/env bash
#SBATCH --job-name=adv_ho_fgsm
#SBATCH --account=kempner_konkle_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=cluster/logs/adv_ho_fgsm_%A_%a.out
#SBATCH --error=cluster/logs/adv_ho_fgsm_%A_%a.err

# FGSM control for item 1: adversarial sensitivity of the 250 readouts on the SAME 100 HELD-OUT
# NSD images (heldout_paths.txt), but with a single-step FGSM attack instead of iterative PGD, to
# confirm the sensitivity results do not depend on the attack method. Sharded (model x monkey) =
# 50 tasks, 5 readouts each. Single-step -> much faster than the PGD job.
# Submit from repo root: sbatch analyses/preprint_figures/take8/figure8/cluster/jobs/adv_robustness_heldout_fgsm.sh
# One (model,monkey): sbatch --array=27 <this>   (task = model_idx*5 + monkey_idx)

# PNC NOTE: this is the ORIGINAL take8 FGSM job script (analyses/preprint_figures/take8/supplementary/
# SX_advrobustness/cluster/jobs/), which was run against the same adv_robustness_attack.py; only the
# repo-relative paths were moved to the scripts/ layout (F8 -> scripts/adversarial, outputs -> $NS).
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${PROJECT_ROOT}/scripts/cluster/config.sh"  # was cluster/config.sh (copy scripts/cluster/config.example.sh -> config.sh)
module load python/3.12.5-fasrc01 2>/dev/null || module load python
source activate "$CONDA_PREFIX" 2>/dev/null || conda activate "$CONDA_PREFIX"
cd "$PROJECT_ROOT"

MODELS=(
    AlexNet_training_seed_01 clipag_vitb32 dinov2_vitb14_reg "radio_v2.5-b" regnety_640
    resnet50 resnet50_clip resnet50_dino resnet50_robust siglip2_vitb16
)
MONKEYS=( red paul venus leap three0 )
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 5))]}"
MONKEY="${MONKEYS[$((SLURM_ARRAY_TASK_ID % 5))]}"
F8=scripts/adversarial  # was analyses/preprint_figures/take8/figure8
NS=${PNC_ADV_OUT:?PNC_ADV_OUT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv

echo "======= adv heldout FGSM | model=$MODEL monkey=$MONKEY | $(hostname) | $(date) ======="
python $F8/adv_robustness_attack.py \
    --model "$MODEL" --device cuda \
    --attack fgsm \
    --monkeys "$MONKEY" --tag "$MONKEY" \
    --eps-255 0.5,1,2,4,8,16,32,64 \
    --image-list "$F8/validation_stimuli/heldout_paths.txt" \
    --out "$NS/ext_heldout_fgsm" \
    ${ADV_EXTRA:-}
echo "======= done $(date) ======="
