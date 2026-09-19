#!/bin/bash
#SBATCH --job-name=take9_fig1emb
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_konkle_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /n/holylabs/LABS/konkle_lab/Lab/jacobprince/take9_cluster/fig1_embeddings/logs/take9_fig1emb_%j.out
# PNC NOTE: the #SBATCH -o/--output, --partition and --account values above are Harvard-cluster values;
# edit them for your cluster. Paths below can be overridden with the PNC_* environment variables.
# Fig 1 image-cloud embeddings: ResNet50 activations for the 969 encoding images (PCA fit)
# + accentuated stimuli (PCA transform) -> fig1_resnet50_pc50.npz.
# Submit from the CLUSTER repo-mirror root (needs circuit_toolkit/ + data/ symlinks).
# NOTE: output lands in the mirror's take9/preproc_data/; pull it to
# outputs_from_cluster/fig1_embeddings_out/ for comparison - the local npz is the
# frozen original (PCA sign/GPU nondeterminism makes bit-exact reproduction unlikely).
PY=${PNC_PYTHON:?PNC_PYTHON must be set (as-run value in the comment)}  # was the cluster conda env python
P=${PNC_REPO_ROOT:?PNC_REPO_ROOT must be set (as-run value in the comment)}  # was /n/holylabs/LABS/konkle_lab/Users/jacobprince/AccentuateVVS (repo mirror root)
cd $P
KMP_DUPLICATE_LIB_OK=TRUE $PY scripts/embeddings/compute_embeddings.py  # was analyses/preprint_figures/take9/cluster/scripts_to_cluster/fig1_embeddings/compute_embeddings.py (output dir via PNC_PREPROCESSED_DATA)
