# scripts/embeddings -- Fig. 1 image-cloud embeddings

Source: `take9/cluster/scripts_to_cluster/fig1_embeddings/`. Edits: `PROJECT_ROOT` is now the repo
root (`parents[2]`, was `parents[6]` in the cluster mirror), data dirs come from `PNC_SOURCE_DATA`,
the output dir from `PNC_PREPROCESSED_DATA`, `circuit_toolkit` -> `core`; `submit_embeddings.sh`
takes `PNC_PYTHON` / `PNC_REPO_ROOT`.

| script | what it computes | inputs | outputs | GPU/SLURM | produced (public data) | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `compute_embeddings.py` | ImageNet ResNet-50 activations at 4 layers for the 969 encoding images (PCA-50 fit) and for every 10th accentuated stimulus (PCA transform); saves the PC scores | `stimuli_encoding/` (Tier A), `stimuli_control/` (all accentuated stimuli; Tier A has only the displayed ones), torchvision ResNet-50 (downloaded) | `fig1_resnet50_pc50.npz` | GPU (CPU works, slowly), `submit_embeddings.sh` | `frozen_inputs/fig1_resnet50_pc50.npz` | partly: the encoding-image PCA fit runs from Tier A; the accentuated-stimulus cloud cannot be reproduced without the full 83 GB stimulus set (and PCA sign / GPU nondeterminism make bit-exact reproduction unlikely anyway, as the job header notes) |
| `submit_embeddings.sh` | SLURM wrapper | - | - | SLURM | - | - |
