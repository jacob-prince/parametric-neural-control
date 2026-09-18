#!/usr/bin/env python3
"""Package source_data/ and preprocessed_data/ into the archives that go on Zenodo, and write
data/zenodo_manifest.json (archive name, tier, bytes, sha256, md5, member count).

    python data/build_manifest.py --archives DIR          # write archives + manifest
    python data/build_manifest.py --archives DIR --only preprocessed_data

Packaging rules (Zenodo default limits: 100 files / 50 GB per record):
  * large single files (HDF5) are uploaded as they are -- Zenodo serves HTTP range requests,
    so downloads resume per file and there is no 2x disk cost;
  * every many-file directory becomes one deterministic tar (sorted members, zeroed mtimes and
    owners; see _manifest_lib.write_tar) with a MEMBERS.json listing per-file sha256/md5;
    csv/json-heavy directories are gzip-compressed, binary-heavy ones are stored.
The manifest never contains URLs: download_data.py resolves files from the record id.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest_lib as L  # noqa: E402
sys.path.insert(0, str(L.REPO))
from pnc import paths  # noqa: E402

# (tier, relative path under the tier root, kind, compress)
#   kind 'file' -> uploaded raw; kind 'dir' -> tar of the directory;
#   kind 'glob:<pattern>' -> tar of the files in the directory matching the pattern
LAYOUT = [
    ('preprocessed_data', '.', 'dir', False),
    ('reference_figures', '.', 'dir', False),          # the 55 manuscript PNGs (pixel oracle for tolerance-mode tests)
    ('source_data', 'brain_data_control/vvs-accentuate_controlsessions_5monkeys_250504-250512.h5', 'file', False),
    ('source_data', 'brain_data_controversial/vvs_accentuate_day3_normalize_red_20250123-20250126.hdf5', 'file', False),
    ('source_data', 'brain_data_encoding/red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5', 'file', False),
    ('source_data', 'brain_data_encoding/paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5', 'file', False),
    ('source_data', 'brain_data_encoding/venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5', 'file', False),
    ('source_data', 'brain_data_encoding/leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5', 'file', False),
    ('source_data', 'brain_data_encoding/three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5', 'file', False),
    ('source_data', 'brain_data_encoding', 'glob:*_sessdata.pkl', False),   # the 22 per-day session pickles (mu/sigma, firing floors)
    ('source_data', 'encoding_model_outputs', 'dir', False),
    ('source_data', 'image_pca_projections', 'dir', False),
    ('source_data', 'model_predictions', 'dir', False),
    ('source_data', 'model_features', 'dir', False),
    ('source_data', 'stimuli_encoding', 'dir', False),
    ('source_data', 'stimuli_control', 'dir', False),
    ('source_data', 'stimuli_controversial', 'dir', False),
    ('source_data', 'accentuation_configs', 'dir', True),
    ('source_data', 'model_backbones', 'dir', False),
    ('source_data', 'cluster_outputs', 'dir', False),
    ('source_data', 'frozen_inputs', 'dir', False),
]


def archive_name(tier, rel, kind, compress):
    if kind == 'file':
        return f'{tier}__{Path(rel).name}'
    stem = tier if rel == '.' else f"{tier}__{rel.replace('/', '__')}"
    if kind.startswith('glob:'):
        stem += '__' + kind[5:].replace('*', '').replace('.', '_').strip('_')
    return stem + ('.tar.gz' if compress else '.tar')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archives', type=Path, required=True, help='directory to write archives into')
    ap.add_argument('--only', help='comma-separated archive names or tier names')
    ap.add_argument('--manifest', type=Path, default=L.MANIFEST_PATH)
    args = ap.parse_args(argv)
    args.archives.mkdir(parents=True, exist_ok=True)
    roots = {'source_data': paths.source_data(), 'preprocessed_data': paths.preprocessed_data(),
             'reference_figures': L.REPO / 'tests' / 'reference' / 'manuscript_png'}
    manifest = L.load_manifest(args.manifest) if args.manifest.exists() else dict(record_id=None, sandbox=False, archives={})
    wanted = {w.strip() for w in args.only.split(',')} if args.only else None
    for tier, rel, kind, compress in LAYOUT:
        name = archive_name(tier, rel, kind, compress)
        if wanted and not ({name, tier} & wanted):
            continue
        src = roots[tier] / rel
        if not src.exists():
            print(f'SKIP {name}: {src} does not exist')
            continue
        print(f'{name} ...', end=' ', flush=True)
        dst = args.archives / name
        if kind == 'file':
            # raw upload: the archive *is* the file; link rather than copy (no 2x disk cost)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            try:
                os.link(src.resolve(), dst)
            except OSError:
                os.symlink(src.resolve(), dst)
            members = None
        else:
            pattern = kind[5:] if kind.startswith('glob:') else None
            members = L.write_tar(dst, src, arc_prefix=(tier if rel == '.' else f'{tier}/{rel}'), compress=compress, pattern=pattern)
        sha, md5, n = L.digests(dst)
        manifest['archives'][name] = dict(tier=tier, path=rel, kind=kind, bytes=n, sha256=sha, md5=md5,
                                          n_members=(len(members) if members else 1))
        print(f'{n / 1e9:.2f} GB')
        L.save_manifest(manifest, args.manifest)
    total = sum(a['bytes'] for a in manifest['archives'].values())
    print(f"{len(manifest['archives'])} archives, {total / 1e9:.1f} GB -> {args.manifest}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
