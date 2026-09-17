#!/usr/bin/env bash
# cluster/config.sh — Shared configuration for the SLURM cluster pipeline
# Source this file from other scripts: source "$(dirname "$0")/config.sh"
#
# PNC NOTE: this is config.example.sh -- copy it to config.sh and replace every <PLACEHOLDER>.
# The as-run values (Harvard FASRC / Kempner cluster) are kept in the trailing "# was ..." comments.

set -euo pipefail

# ── SSH / Cluster ────────────────────────────────────────────────────────────
CLUSTER_HOST="<ssh-host-alias>"            # was "cannon"
CLUSTER_USER="<cluster-username>"          # was "jacobprince"
CLUSTER_PROJECT="<cluster-path-to-repo-mirror>"   # was "/n/holylabs/LABS/konkle_lab/Users/jacobprince/AccentuateVVS"

# ── Local paths ──────────────────────────────────────────────────────────────
LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── SLURM defaults ───────────────────────────────────────────────────────────
SLURM_ACCOUNT="<slurm-account>"            # was "kempner_konkle_lab"
SLURM_PARTITION="<gpu-partition>"          # was "kempner_h100"
SLURM_TIME="0-12:00:00"
SLURM_GPUS=1
SLURM_CPUS=16
SLURM_MEM="240G"

# ── Conda ────────────────────────────────────────────────────────────────────
CONDA_PREFIX="<path-to-conda-env>"         # was "/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/conda_envs/accentuate"

# ── Poll settings (seconds) ─────────────────────────────────────────────────
POLL_INITIAL=30
POLL_MAX=300
POLL_BACKOFF=1.5   # multiplicative factor each round

# ── Data symlink mapping ────────────────────────────────────────────────────
# Format: "local_subdir|cluster_target"
# These are created inside $CLUSTER_PROJECT/data/ on the cluster.
# Replace each <...> target with the directory holding that data on your cluster.
DATA_SYMLINKS=(
    "stimuli_encoding|<encoding-stimuli-dir>"                     # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025
    "model_features|<encoding-models-dir>"                        # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models
    "image_pca_projections|<encoding-models-dir>"                 # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models
    "model_predictions|<encoding-models-dir>"                     # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models
    "model_backbones|<model-backbones-dir>"                       # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/model_backbones
    "encoding_model_outputs/Encoding_model_outputs|<readout-export-dir>"   # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/Encoding_model_outputs
    "brain_data_encoding|<ephys-data-dir>"                        # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data
    "brain_data_control|<ephys-data-dir>"                         # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data
    "stimuli_control|<accentuation-outputs-dir>"                  # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs
    "accentuation_configs|<accentuation-configs-dir>"             # was /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs
)

# ── rsync excludes for code sync ─────────────────────────────────────────────
RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='data/'
    --exclude='cluster/logs/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='*.egg-info/'
    --exclude='notebooks/'
    --exclude='labbook/'
    --exclude='manuscript/'
    --exclude='demos/'
    --exclude='z-old/'
    --exclude='figures/'
    --exclude='*.png'
    --exclude='*.jpg'
    --exclude='*.jpeg'
    --exclude='*.gif'
    --exclude='*.svg'
    --exclude='*.npy'
    --exclude='*.npz'
    --exclude='*.h5'
    --exclude='*.nc'
    --exclude='*.pkl'
)
