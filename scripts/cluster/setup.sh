#!/usr/bin/env bash
# cluster/setup.sh - One-time setup on the Kempner cluster
# Creates project dirs, data symlink farm, and conda environment.
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "==> Running one-time cluster setup"

# Step 1: Sync code first
echo "--- Step 1: Syncing code ---"
bash "$(dirname "$0")/sync-code.sh"

# Step 2: Create directory structure + symlink farm on cluster
echo "--- Step 2: Creating directories and data symlinks ---"
SYMLINK_CMDS=""
for mapping in "${DATA_SYMLINKS[@]}"; do
    local_sub="${mapping%%|*}"
    target="${mapping##*|}"
    # Handle nested dirs (e.g., encoding_model_outputs/Encoding_model_outputs)
    SYMLINK_CMDS+="mkdir -p '${CLUSTER_PROJECT}/data/$(dirname "$local_sub")' && "
    SYMLINK_CMDS+="ln -sfn '${target}' '${CLUSTER_PROJECT}/data/${local_sub}' && "
done
# Remove trailing " && "
SYMLINK_CMDS="${SYMLINK_CMDS% && }"

ssh "$CLUSTER_HOST" bash -s <<REMOTE_SETUP
set -euo pipefail

# Create project structure
mkdir -p "${CLUSTER_PROJECT}/data"
mkdir -p "${CLUSTER_PROJECT}/cluster/logs"

# Create symlinks
${SYMLINK_CMDS}

echo "Symlinks created:"
ls -la "${CLUSTER_PROJECT}/data/"

# Verify at least one target is accessible
if [ -d "${LAB_STORAGE_CHECK_DIR:-}" ]; then   # PNC: set LAB_STORAGE_CHECK_DIR to your lab storage; was the literal /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation
    echo "  [OK] Lab storage accessible"
else
    echo "  [WARN] Lab storage not accessible from login node - this is normal, it should be accessible from compute nodes"
fi
REMOTE_SETUP

echo "--- Step 3: Creating conda environment ---"
ssh "$CLUSTER_HOST" bash -s <<CONDA_SETUP
set -euo pipefail

# Load conda
module load python/3.12.5-fasrc01 2>/dev/null || module load python

# Check if env already exists
if [ -d "${CONDA_PREFIX}" ]; then
    echo "Conda env already exists at ${CONDA_PREFIX}"
    echo "To recreate, delete it first: rm -rf ${CONDA_PREFIX}"
else
    echo "Creating conda env at ${CONDA_PREFIX}"
    mkdir -p "$(dirname "${CONDA_PREFIX}")"
    conda create --prefix "${CONDA_PREFIX}" python=3.11 -y
fi

# Activate and install
source activate "${CONDA_PREFIX}" 2>/dev/null || conda activate "${CONDA_PREFIX}"

echo "Installing requirements..."
pip install -r "${CLUSTER_PROJECT}/cluster/requirements.txt"

echo "Installing circuit_toolkit in editable mode..."
pip install -e "${CLUSTER_PROJECT}/circuit_toolkit"

echo "Conda env ready. Python: \$(python --version)"
echo "Torch: \$(python -c 'import torch; print(torch.__version__)')"
CONDA_SETUP

echo "==> Cluster setup complete!"
echo ""
echo "Verify with:"
echo "  ssh ${CLUSTER_HOST} 'ls -la ${CLUSTER_PROJECT}/data/'"
