#!/usr/bin/env bash
#SBATCH --job-name=adv_vis_p8
#SBATCH --account=kempner_konkle_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/adv_vp8_%A_%a.out
#SBATCH --error=/n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv/logs/adv_vp8_%A_%a.err

# Qualitative adversarial-attack gallery for site paul unit 8, cat seed (idx 3), +2 z target.
# Ladder search for the minimal L-inf eps that raises each model's predicted response by the target
# (no fine-sweep exists for paul8). -> figure8/intermediate/advvis_exact/advvis_paul_Ch8_seed3_*.npz
# Submit from the CLUSTER repo-mirror root (needs prj_control_fa/, Closed-loop-visual-insilico/,
# circuit_toolkit/ + cluster/config.sh); outputs/logs land on konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv (durable).
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
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
F9=scripts/adversarial  # was analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig5_adv
NS=${PNC_ADV_OUT:?PNC_ADV_OUT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_adv
echo "Model: $MODEL  Node: $(hostname)  Start: $(date)"
python $F9/adv_visual_attack.py \
    --model "$MODEL" --monkey paul --unit 8 --seed-idx 3 \
    --target-delta 2.0 --steps 100 --restarts 2 --device cuda \
    --out $NS/advvis_exact
echo "Finish: $(date)"
