"""Shared helpers for the data tooling: hashing, deterministic tar archives, manifest I/O."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / 'zenodo_manifest.json'
MEMBERS_NAME = 'MEMBERS.json'
CHUNK = 1 << 20   # 1 MiB read size for hashing and downloads


def digests(path):
    """(sha256, md5, bytes) of a file."""
    h1, h2, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b''):
            h1.update(chunk); h2.update(chunk); n += len(chunk)
    return h1.hexdigest(), h2.hexdigest(), n


def iter_files(root, pattern=None):
    """Regular files under `root`, sorted, skipping macOS junk and build logs; `pattern` is an
    optional glob restricting the selection (relative to root, e.g. '*_sessdata.pkl')."""
    it = Path(root).rglob(pattern) if pattern else Path(root).rglob('*')
    for p in sorted(it):
        if p.is_file() and not p.name.startswith('._') and p.name not in ('.DS_Store',) \
                and '.build_logs' not in p.parts and not p.name.startswith('.'):
            yield p


def members_manifest(root, rel_to, pattern=None):
    # {relative path: bytes/sha256/md5}; this is what MEMBERS.json holds
    rows = {}
    for p in iter_files(root, pattern):
        sha, md5, n = digests(p)
        rows[str(p.relative_to(rel_to))] = dict(bytes=n, sha256=sha, md5=md5)
    return rows


def _normalize(ti):
    """Deterministic tar headers: fixed mtime, no owner, plain permission bits."""
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ''
    ti.mode = 0o755 if ti.isdir() else 0o644
    return ti


def write_tar(archive, root, arc_prefix, compress=False, pattern=None):
    """Write every file under `root` into `archive` as `<arc_prefix>/<relative path>`, with a
    MEMBERS.json (per-file bytes/sha256/md5) as the first member. Member order is sorted, so
    the archive bytes are a pure function of the file contents."""
    members = members_manifest(root, root, pattern)
    mode = 'w:gz' if compress else 'w'
    kwargs = dict(compresslevel=6, format=tarfile.PAX_FORMAT) if compress else dict(format=tarfile.PAX_FORMAT)
    if compress:
        # gzip header carries a timestamp; pin it so archive hashes are reproducible.
        import gzip
        raw = open(archive, 'wb')
        fobj = gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=raw, mtime=0)  # no name/time in the header
        tf = tarfile.open(fileobj=fobj, mode='w', format=tarfile.PAX_FORMAT)
    else:
        fobj = None
        tf = tarfile.open(archive, mode, **kwargs)
    with tf:
        # MEMBERS.json first, so download_data.read_members can read it without scanning the archive
        payload = json.dumps(members, indent=1, sort_keys=True).encode()
        ti = tarfile.TarInfo(f'{arc_prefix}/{MEMBERS_NAME}'); ti.size = len(payload); _normalize(ti)
        tf.addfile(ti, io.BytesIO(payload))
        # files only, one at a time, in sorted order (no directory entries)
        for rel in members:
            tf.add(Path(root) / rel, arcname=f'{arc_prefix}/{rel}', recursive=False, filter=_normalize)
    if fobj is not None:
        fobj.close()
        raw.close()
    return members


def load_manifest(path=MANIFEST_PATH):
    return json.loads(Path(path).read_text())


def save_manifest(m, path=MANIFEST_PATH):
    # sorted keys + trailing newline: the committed JSON diffs cleanly
    Path(path).write_text(json.dumps(m, indent=1, sort_keys=True) + '\n')
