#!/usr/bin/env python3
"""Regenerate preprocessed_data/ from source_data/ -- every cache the figure scripts read.

    python scripts/preprocessing/build_all.py                # -> $PNC_PREPROCESSED_DATA (default preprocessed_data/)
    python scripts/preprocessing/build_all.py --out DIR      # build into another directory
    python scripts/preprocessing/build_all.py --only run_preproc,build_layer_selection
    python scripts/preprocessing/build_all.py --list

Stages run in dependency order, each in its own subprocess with a pinned environment
(PYTHONHASHSEED=0, single-threaded BLAS) so that the result is byte-identical to the shipped
preprocessed_data on the reference environment (see README, "Regenerating preprocessed_data").

Stage 0 copies the frozen inputs (source_data/frozen_inputs/, files with no in-repo producer;
see PROVENANCE.md there) into the output directory, because the builders and the figure
scripts read everything from one place.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pnc import paths  # noqa: E402

HERE = Path(__file__).resolve().parent
GRAD = REPO / 'scripts' / 'gradients'

# (name, script, extra args) -- dependency order.
STAGES = [
    ('run_preproc',                   HERE / 'run_preproc.py', []),                       # brain/encoding/predictions/exclusions/stimuli/controversial
    ('build_layer_selection',         HERE / 'build_layer_selection.py', []),
    ('build_floc_selectivity',        HERE / 'build_floc_selectivity.py', []),
    ('build_tuning_stability',        HERE / 'build_tuning_stability.py', []),
    ('build_axis_alignment',          HERE / 'build_axis_alignment.py', []),              # torch (raw PCA pickles)
    ('build_hp_tuning_cache',         HERE / 'build_hp_tuning_cache.py', []),             # from frozen hp_from_logs.json
    ('build_predicting_results',      HERE / 'build_predicting_results.py', []),          # from frozen sup_predicting_master.csv
    ('build_predicting_results_no_untrained', HERE / 'build_predicting_results.py', ['--exclude-untrained']),
    ('build_gradient_cache',          GRAD / 'build_gradient_cache.py', []),              # from cluster_outputs/fig5_grad_out
    ('build_advsens_outcome_reliability', HERE / 'build_advsens_outcome_reliability.py', []),
    ('fig5_heldout_analysis',         HERE / 'fig5_heldout_analysis.py', []),             # cluster_outputs + frozen verification csv
    ('fig5_outcome_ceiling_slope_resid9', HERE / 'fig5_outcome_ceiling_seedsplit.py', ['slope', '--resid', '9trained']),
    ('fig5_outcome_ceiling_slope_noresid', HERE / 'fig5_outcome_ceiling_seedsplit.py', ['slope', '--resid', 'none']),
    ('fig5_outcome_ceiling_r_resid9', HERE / 'fig5_outcome_ceiling_seedsplit.py', ['r', '--resid', '9trained']),
    ('build_fig2_cloud',              HERE / 'build_fig2_cloud.py', []),                  # torch
    ('build_fig3_embedding',          HERE / 'build_fig3_embedding.py', []),              # torch
    ('build_fig6_caches',             HERE / 'build_fig6_caches.py', []),                 # needs frozen imagenet_predictions.pkl
    ('build_sup_reliability_cache',   HERE / 'build_sup_reliability_cache.py', []),
    ('build_superstim_scatter_caches', HERE / 'build_superstim_scatter_caches.py', []),
]

FROZEN = [  # copied verbatim from source_data/frozen_inputs (see PROVENANCE.md there)
    'fig1_image_cloud_encoding.png', 'fig1_resnet50_pc50.npz', 'leap_variant_similarity.json',
    'fig5_verification_data.csv', 'fig5_bracket_bootstrap_p.csv', 'sup_hyperparam_hp_from_logs.json',
    'sup_advform_heldout_descriptors.csv', 'sup_advrob_fgsm_vs_pgd_eps.csv', 'sup_diet_refit.csv',
    'sup_gradient_gallery_avg.pkl', 'sup_predicting_master.csv', 'imagenet_predictions.pkl',
    'grad_maps', 'fig6_imagenet_thumbs',
]


def pinned_env(out_dir):
    # PNC_SOURCE_DATA passes through; PNC_PREPROCESSED_DATA is forced to out_dir so every builder writes there
    keep = ('PATH', 'HOME', 'TMPDIR', 'USER', 'LANG', 'LC_ALL', 'PNC_SOURCE_DATA')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(PNC_PREPROCESSED_DATA=str(out_dir), MPLBACKEND='Agg', PYTHONHASHSEED='0',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    return env


def stage_frozen(out_dir):
    src = paths.frozen_inputs()
    copied = []
    for name in FROZEN:
        s, d = src / name, out_dir / name
        if not s.exists():
            raise paths.MissingDataError(f'frozen input missing: {s}\n  -> python data/download_data.py --tier source')
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)             # replace wholesale so stale files from an earlier build do not linger
            shutil.copytree(s, d, ignore=shutil.ignore_patterns('._*', '.DS_Store'))
        else:
            shutil.copy2(s, d)
        copied.append(name)
    return copied


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_hashes(out_dir):
    # skip dot-entries (.build_logs) and the manifest itself, which is written after hashing
    rows = {}
    for p in sorted(out_dir.rglob('*')):
        if p.is_file() and not any(part.startswith('.') for part in p.relative_to(out_dir).parts) \
                and p.name != 'BUILD_MANIFEST.json':
            rows[str(p.relative_to(out_dir))] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=None, help='output directory (default: $PNC_PREPROCESSED_DATA or preprocessed_data/)')
    ap.add_argument('--only', help='comma-separated stage names (frozen inputs are always staged)')
    ap.add_argument('--skip-frozen', action='store_true', help='do not copy the frozen inputs')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args(argv)
    if args.list:
        for name, script, extra in STAGES:
            print(f'{name:40s} {script.relative_to(REPO)} {" ".join(extra)}')
        return 0
    out_dir = (args.out or paths.preprocessed_data()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES
    if args.only:
        wanted = {w.strip() for w in args.only.split(',')}
        stages = [s for s in STAGES if s[0] in wanted]
        unknown = wanted - {s[0] for s in stages}
        if unknown:
            raise SystemExit(f'unknown stage(s): {sorted(unknown)}')
    log_dir = out_dir / '.build_logs'
    log_dir.mkdir(exist_ok=True)
    records = []
    if not args.skip_frozen:
        t0 = time.time()
        copied = stage_frozen(out_dir)
        print(f'frozen inputs: {len(copied)} staged ({time.time() - t0:.0f}s)', flush=True)
    env = pinned_env(out_dir)
    for name, script, extra in stages:
        if not script.exists():
            raise SystemExit(f'{name}: {script} not found')
        print(f'{name} ...', end=' ', flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(script), *extra], cwd=str(REPO), env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (log_dir / f'{name}.log').write_text(r.stdout or '')
        records.append(dict(stage=name, script=str(script.relative_to(REPO)), args=extra,
                            returncode=r.returncode, duration_seconds=round(time.time() - t0, 1)))
        print(f"{'ok' if r.returncode == 0 else 'FAILED'} ({records[-1]['duration_seconds']}s)", flush=True)
        if r.returncode != 0:
            print((r.stdout or '')[-3000:])
            return 1                         # later stages read this one's output; stop here
    manifest = dict(output_dir=str(out_dir), python=sys.version.split()[0], stages=records,
                    files=write_hashes(out_dir))
    (out_dir / 'BUILD_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"done: {len(manifest['files'])} files in {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
