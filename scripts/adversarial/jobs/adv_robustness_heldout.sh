#!/usr/bin/env bash
#SBATCH --job-name=adv_heldout
#SBATCH --account=kempner_konkle_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/adv_heldout_%A_%a.out
#SBATCH --error=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/adv_heldout_%A_%a.err

# item 1: adversarial sensitivity of the 250 readouts on the 87 HELD-OUT NSD images
# (same fixed image set for every readout). Sharded (model x monkey) = 50 tasks, 5 readouts each.
# Submit from the CLUSTER repo-mirror root (needs prj_control_fa/, Closed-loop-visual-insilico/,
# circuit_toolkit/ + cluster/config.sh); outputs/logs land on konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv (durable).
# One (model,monkey): sbatch --array=27 <this>   (task = model_idx*5 + monkey_idx)

# PNC NOTE: the #SBATCH --output/--error paths above and --account/--partition are Harvard-cluster
# values; edit them for your cluster. Output root is $NS (override with PNC_ADV_OUT).
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
F9=scripts/adversarial  # was analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig5_adv
NS=${PNC_ADV_OUT:?PNC_ADV_OUT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv

echo "======= adv heldout | model=$MODEL monkey=$MONKEY | $(hostname) | $(date) ======="
python $F9/adv_robustness_attack.py \
    --model "$MODEL" --device cuda \
    --monkeys "$MONKEY" --tag "$MONKEY" \
    --eps-255 0.5,1,2,4,8,16,32,64 \
    --image-list "$F9/validation_stimuli/heldout_paths.txt" \
    --out "$NS/ext_heldout" \
    ${ADV_EXTRA:-}
echo "======= done $(date) ======="
