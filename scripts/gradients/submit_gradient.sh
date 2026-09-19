#!/bin/bash
#SBATCH --job-name=take9_grad
#SBATCH --partition=shared
#SBATCH --account=konkle_lab
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o take9_grad_%j.out
# PNC NOTE: the #SBATCH -o/--output, --partition and --account values above are Harvard-cluster values;
# edit them for your cluster. Paths below can be overridden with the PNC_* environment variables.
# CPU-only aggregation of gradient Fourier profiles (no GPU, no login-node compute).
PY=${PNC_PYTHON:?PNC_PYTHON must be set (as-run value in the comment)}  # was the cluster conda env python
STAGE=${PNC_GRAD_STAGE:?PNC_GRAD_STAGE must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig5_gradmaps (dir containing extract_gradient_freq.py)
ENC=${PNC_ENCODING_MODEL_ROOT:?PNC_ENCODING_MODEL_ROOT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models
STIM=${PNC_ENCODING_STIM_DIR:?PNC_ENCODING_STIM_DIR must be set (as-run value in the comment)}  # was /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025
mkdir -p $STAGE/grad_out
cd $STAGE
KMP_DUPLICATE_LIB_OK=TRUE $PY extract_gradient_freq.py --enc-models $ENC --stim $STIM --out $STAGE/grad_out/gradient_freq.pkl
