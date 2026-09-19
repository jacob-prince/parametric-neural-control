# scripts/cluster -- SLURM plumbing used for the GPU stages

Copied from the private repo's `cluster/` (`config.sh` -> `config.example.sh` with placeholders,
`run.sh`, `setup.sh`, `requirements.txt` as-is). The job scripts in `scripts/adversarial/jobs/`,
`scripts/gradients/submit_gradient.sh`, `scripts/imagenet/submit_extract.sh` and
`scripts/embeddings/submit_embeddings.sh` source `scripts/cluster/config.sh` (copy
`config.example.sh` to `config.sh` and fill in the `<PLACEHOLDER>` values; the as-run Harvard FASRC /
Kempner values are kept in `# was` comments).

| file | what it does | inputs | outputs | GPU/SLURM | produced | runnable from Tier A? |
|---|---|---|---|---|---|---|
| `config.example.sh` | SSH host, cluster project path, SLURM account/partition/resources, conda env, data symlink map (`data/<local_subdir>` -> cluster storage) and rsync excludes | - | - | - | - | infrastructure; fill the placeholders for your cluster |
| `setup.sh` | one-time cluster setup: sync code, create `data/` symlink farm, create the conda env and `pip install -r requirements.txt` + `pip install -e circuit_toolkit` (the private toolkit; in this repo its used parts live in `core/`) | `config.sh`, `sync-code.sh` (not shipped) | - | ssh | - | needs `sync-code.sh`/`sync-results.sh`, which are not shipped (they only rsync the repo mirror) |
| `run.sh` | sync code -> `sbatch cluster/jobs/<id>.sh` -> poll -> sync results | `config.sh`, a job script | - | ssh + SLURM | - | same |
| `requirements.txt` | pip requirements of the `accentuate` conda env (torch, timm, open_clip, `clip` from GitHub, horama, h5py, boto3, ...) | - | - | - | - | yes |
