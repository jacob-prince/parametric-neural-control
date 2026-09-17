#!/bin/bash
#SBATCH --job-name=take9_inet
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_konkle_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-9
#SBATCH -o take9_inet_%A_%a.out
# PNC NOTE: the #SBATCH -o/--output, --partition and --account values above are Harvard-cluster values;
# edit them for your cluster. Paths below can be overridden with the PNC_* environment variables.
# Extract 50k ImageNet-val predictions for one model per array task.
# Env: the project's accentuate conda env, called by full path (not on the default conda path).
PY=${PNC_PYTHON:?PNC_PYTHON must be set (as-run value in the comment)}  # was the cluster conda env python
STAGE=${PNC_IMAGENET_STAGE:?PNC_IMAGENET_STAGE must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig6_imagenet (dir with extract_imagenet_predictions.py + encoding_<monkey>.pkl)
ENC=${PNC_READOUT_EXPORT_ROOT:?PNC_READOUT_EXPORT_ROOT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/Encoding_model_outputs
BB=${PNC_MODEL_BACKBONES:?PNC_MODEL_BACKBONES must be set (as-run value in the comment)}  # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones
INET=${PNC_IMAGENET_ROOT:?PNC_IMAGENET_ROOT must be set (as-run value in the comment)}   # was /n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k; tier1 imagenet1k-256 is gone (2026-07); kempner testbed copy (original JPEGs)
MODELS=(AlexNet_training_seed_01 clipag_vitb32 dinov2_vitb14_reg radio_v2.5-b regnety_640 \
        resnet50 resnet50_clip resnet50_dino resnet50_robust siglip2_vitb16)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
cd $STAGE
KMP_DUPLICATE_LIB_OK=TRUE $PY extract_imagenet_predictions.py --model $M \
  --enc-cache $STAGE --enc-outputs $ENC --imagenet-root $INET --backbones $BB \
  --out $STAGE/out --device cuda ${LIMIT:+--limit $LIMIT}
