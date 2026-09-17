#!/usr/bin/env python3
"""Download the data that accompanies the paper from Zenodo and verify every file.

    python data/download_data.py --tier preprocessed        # ~0.4 GB: enough to render every figure
    python data/download_data.py --tier source              # ~25 GB: regenerate preprocessed_data + run the demos
    python data/download_data.py --tier all
    python data/download_data.py --only source_data__stimuli_encoding.tar
    python data/download_data.py --verify-only              # re-check what is on disk
    python data/download_data.py --from-local /path/to/archives   # offline: use already-downloaded archives

Files land in source_data/ and preprocessed_data/ (override with PNC_SOURCE_DATA /
PNC_PREPROCESSED_DATA). Downloads are resumable (HTTP range requests), retried on transient
errors, and checked against the sha256 + md5 recorded in data/zenodo_manifest.json; archives
are extracted into a staging directory and moved into place file by file. A file that is already
present with the right size (and, unless --fast, the right sha256) is never downloaded or
overwritten; a mismatching file stops the run unless --force is given.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest_lib as L  # noqa: E402
sys.path.insert(0, str(L.REPO))
from pnc import paths  # noqa: E402

ZENODO = {False: 'https://zenodo.org', True: 'https://sandbox.zenodo.org'}
TIERS = {'preprocessed': ['preprocessed_data', 'reference_figures'], 'source': ['source_data'],
         'all': ['preprocessed_data', 'reference_figures', 'source_data']}


def file_url(record_id, name, sandbox=False):
    return f'{ZENODO[sandbox]}/api/records/{record_id}/files/{name}/content'


def http_download(url, dst, expected_bytes, retries=6):
    """Resumable download to `dst` (a .part file), with exponential backoff."""
    dst = Path(dst)
    for attempt in range(retries):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= expected_bytes:
            return
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, 'ab' if have else 'wb') as out:
                t0, done = time.time(), have
                while True:
                    chunk = resp.read(L.CHUNK)
                    if not chunk:
                        break
                    out.write(chunk); done += len(chunk)
                    if time.time() - t0 > 2:
                        print(f'\r    {done / 1e9:7.2f} / {expected_bytes / 1e9:.2f} GB', end='', flush=True); t0 = time.time()
            print('\r', end='')
            return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError) as e:
            code = getattr(e, 'code', None)
            if code is not None and code not in (408, 429, 500, 502, 503, 504):
                raise
            wait = min(60, 2 ** attempt)
            print(f'\n    transient error ({e}); retrying in {wait}s', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'giving up on {url}')


def verify(path, entry, fast=False):
    if not Path(path).exists() or Path(path).stat().st_size != entry['bytes']:
        return False
    if fast:
        return True
    sha, md5, _ = L.digests(path)
    return sha == entry['sha256'] and md5 == entry['md5']


def members_ok(root, members, fast=False):
    return all(verify(Path(root) / rel, m, fast) for rel, m in members.items())


def read_members(archive):
    with tarfile.open(archive, 'r:*') as tf:
        first = tf.next()
        if first is None or not first.name.endswith(L.MEMBERS_NAME):
            raise RuntimeError(f'{archive}: first member is not {L.MEMBERS_NAME}')
        return json.load(tf.extractfile(first)), first.name.rsplit('/', 1)[0]


def extract(archive, tier_root, members, prefix, force=False):
    """Extract into <tier_root>/.incoming/, then move members into place one by one."""
    staging = Path(tier_root) / '.incoming' / Path(archive).stem
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, 'r:*') as tf:
        tf.extractall(staging, filter='data')
    src_root = staging / prefix
    # prefix is '<tier>' or '<tier>/<relpath>'; strip the leading tier component
    rel_root = Path(*Path(prefix).parts[1:]) if len(Path(prefix).parts) > 1 else Path('.')
    for rel, m in members.items():
        s, d = src_root / rel, Path(tier_root) / rel_root / rel
        if not verify(s, m):
            raise RuntimeError(f'{archive}: extracted {rel} does not match its recorded checksum')
        if d.exists():
            if verify(d, m):
                continue
            if not force:
                raise RuntimeError(f'{d} exists with different content; re-run with --force to replace it')
        d.parent.mkdir(parents=True, exist_ok=True)
        os.replace(s, d)
    shutil.rmtree(staging, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tier', choices=sorted(TIERS), default='preprocessed')
    ap.add_argument('--only', help='comma-separated archive names (see data/zenodo_manifest.json)')
    ap.add_argument('--from-local', type=Path, help='directory holding already-downloaded archives (offline mode)')
    ap.add_argument('--cache', type=Path, default=None, help='where downloaded archives are kept (default: <source_data>/.archives)')
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--fast', action='store_true', help='size-only check of files already on disk')
    ap.add_argument('--force', action='store_true', help='replace files whose checksum differs')
    ap.add_argument('--keep-archives', action='store_true', help='do not delete tar archives after extraction')
    ap.add_argument('--manifest', type=Path, default=L.MANIFEST_PATH)
    args = ap.parse_args(argv)

    manifest = L.load_manifest(args.manifest)
    roots = {'source_data': paths.source_data(), 'preprocessed_data': paths.preprocessed_data(),
             'reference_figures': L.REPO / 'tests' / 'reference' / 'manuscript_png'}
    entries = {n: a for n, a in manifest['archives'].items() if a['tier'] in TIERS[args.tier]}
    if args.only:
        wanted = {w.strip() for w in args.only.split(',')}
        entries = {n: a for n, a in manifest['archives'].items() if n in wanted}
        if not entries:
            raise SystemExit(f'nothing in the manifest matches --only {args.only}')
    if not args.from_local and not args.verify_only and not manifest.get('record_id'):
        raise SystemExit('data/zenodo_manifest.json has no record_id yet: the Zenodo deposit has not been created. '
                         'Use --from-local DIR with archives obtained another way.')
    cache = args.cache or roots['source_data'] / '.archives'
    need_bytes = sum(a['bytes'] for a in entries.values())
    free = shutil.disk_usage(roots['source_data'] if roots['source_data'].exists() else L.REPO).free
    if not args.verify_only and free < 2.2 * need_bytes:
        print(f'warning: {free / 1e9:.1f} GB free, {need_bytes / 1e9:.1f} GB of archives selected (extraction needs ~2x)')

    problems = []
    for name, a in entries.items():
        tier_root = roots[a['tier']]
        if a['kind'] == 'file':
            dst = tier_root / a['path']
            if verify(dst, a, args.fast):
                print(f'{name}: ok'); continue
            if args.verify_only:
                problems.append(name); print(f'{name}: MISSING/MISMATCH'); continue
            src = args.from_local / name if args.from_local else None
            if src is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                part = dst.with_suffix(dst.suffix + '.part')
                print(f'{name}: downloading {a["bytes"] / 1e9:.2f} GB')
                http_download(file_url(manifest['record_id'], name, manifest.get('sandbox', False)), part, a['bytes'])
                src = part
            if not verify(src, a):
                problems.append(name); print(f'{name}: checksum mismatch after download'); continue
            if dst.exists() and not args.force and not verify(dst, a):
                raise SystemExit(f'{dst} exists with different content; re-run with --force')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src != dst:
                (os.replace if not args.from_local else shutil.copy2)(src, dst)
            print(f'{name}: ok')
            continue
        # directory archive
        archive = (args.from_local / name) if args.from_local else (cache / name)
        members = prefix = None
        if archive.exists() and verify(archive, a):
            members, prefix = read_members(archive)
        if members is not None and members_ok(tier_root / ('' if a['path'] == '.' else a['path']), members, args.fast):
            print(f'{name}: ok ({len(members)} files)'); continue
        if args.verify_only:
            problems.append(name); print(f'{name}: MISSING/MISMATCH'); continue
        if not (archive.exists() and verify(archive, a)):
            if args.from_local:
                problems.append(name); print(f'{name}: not found in {args.from_local}'); continue
            cache.mkdir(parents=True, exist_ok=True)
            print(f'{name}: downloading {a["bytes"] / 1e9:.2f} GB')
            part = archive.with_suffix(archive.suffix + '.part')
            http_download(file_url(manifest['record_id'], name, manifest.get('sandbox', False)), part, a['bytes'])
            if not verify(part, a):
                problems.append(name); print(f'{name}: checksum mismatch after download'); continue
            os.replace(part, archive)
        members, prefix = read_members(archive)
        extract(archive, tier_root, members, prefix, force=args.force)
        if not args.keep_archives and not args.from_local:
            archive.unlink()
        print(f'{name}: ok ({len(members)} files extracted)')
    if problems:
        print(f'{len(problems)} problem(s): {problems}')
        return 1
    print('all selected data present and verified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
