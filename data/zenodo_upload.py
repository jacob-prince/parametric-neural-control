#!/usr/bin/env python3
"""Create or update a DRAFT Zenodo deposit holding the data archives, and record its id in
data/zenodo_manifest.json. This script never publishes: publishing (which mints the DOI and
freezes the files) is done by hand in the Zenodo web interface after review.

    export ZENODO_TOKEN=...          # personal access token with deposit:write (never committed)
    python data/zenodo_upload.py --archives DIR [--sandbox]          # create draft, upload everything
    python data/zenodo_upload.py --archives DIR --deposit 1234567     # add/replace files in an existing draft
    python data/zenodo_upload.py --metadata-only --deposit 1234567    # push title/authors/description only

The token is read from the ZENODO_TOKEN environment variable only. Uploads use the deposit
files API (one PUT per archive to the deposit bucket) and are verified against the md5 that
Zenodo reports. On success the script prints the draft URL; share the deposit's secret preview
link with reviewers while files are still changing.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _manifest_lib as L  # noqa: E402

API = {False: 'https://zenodo.org/api', True: 'https://sandbox.zenodo.org/api'}

METADATA = {
    'upload_type': 'dataset',
    'title': 'Source and preprocessed data for: Parametric neural control differentiates top '
             'neural network models of primate visual cortex',
    'creators': [
        {'name': 'Prince, Jacob S.', 'affiliation': 'Harvard University'},
        {'name': 'Wang, Binxu', 'affiliation': 'Harvard University; Kempner Institute'},
        {'name': 'Fel, Thomas', 'affiliation': 'Harvard University; Kempner Institute'},
        {'name': 'Jagadeesh, Akshay V.', 'affiliation': 'Harvard Medical School'},
        {'name': 'Vaziri, Parisa A.', 'affiliation': 'Harvard Medical School'},
        {'name': 'Alvarez, George A.', 'affiliation': 'Harvard University; Kempner Institute'},
        {'name': 'Livingstone, Margaret S.', 'affiliation': 'Harvard Medical School'},
        {'name': 'Konkle, Talia', 'affiliation': 'Harvard University; Kempner Institute'},
    ],
    'description': (
        'Neural recordings (trial-level HDF5), post-hoc encoding-model predictions, layer-selection '
        'scores, accentuation configs, cluster analysis outputs, displayed stimuli, and the preprocessed '
        'caches from which every figure of the paper is rendered. Download and verify with '
        'data/download_data.py from https://github.com/jacob-prince/parametric-neural-control; '
        'archives carry a MEMBERS.json with per-file SHA-256 checksums.'),
    'access_right': 'open',
    'license': 'cc-by-4.0',
    'keywords': ['macaque', 'visual cortex', 'encoding models', 'feature accentuation',
                 'deep neural networks', 'neural control'],
    'related_identifiers': [
        {'relation': 'isSupplementTo', 'identifier': 'https://github.com/jacob-prince/parametric-neural-control',
         'resource_type': 'software'},
    ],
}


def token():
    tok = os.environ.get('ZENODO_TOKEN')
    if not tok:
        raise SystemExit('set ZENODO_TOKEN (a Zenodo personal access token with deposit:write scope)')
    return tok


def request(method, url, tok, data=None, headers=None, raw=None):
    hdrs = {'Authorization': f'Bearer {tok}'}
    hdrs.update(headers or {})
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    if data is not None:
        hdrs['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as e:
        # include the response body: Zenodo's validation messages live there
        raise SystemExit(f'{method} {url} -> HTTP {e.code}: {e.read().decode(errors="replace")[:2000]}')


class _FileReader:
    """Streams a file for urllib without loading it into memory."""
    def __init__(self, path):
        self.fh = open(path, 'rb'); self.size = os.path.getsize(path)
    def read(self, n=-1):
        return self.fh.read(n)
    def __len__(self):
        return self.size   # urllib takes Content-Length from len() of a non-bytes body


def upload_file(bucket, path, tok):
    url = f"{bucket}/{urllib.parse.quote(Path(path).name)}"
    req = urllib.request.Request(url, data=_FileReader(path), method='PUT',
                                 headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/octet-stream',
                                          'Content-Length': str(os.path.getsize(path))})
    try:
        with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f'PUT {url} -> HTTP {e.code}: {e.read().decode(errors="replace")[:2000]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archives', type=Path, help='directory written by data/build_manifest.py')
    ap.add_argument('--deposit', type=int, help='existing draft deposit id (default: create a new draft)')
    ap.add_argument('--sandbox', action='store_true', help='use sandbox.zenodo.org')
    ap.add_argument('--metadata-only', action='store_true')
    ap.add_argument('--only', help='comma-separated archive names to upload')
    ap.add_argument('--manifest', type=Path, default=L.MANIFEST_PATH)
    args = ap.parse_args(argv)
    tok = token()
    api = API[args.sandbox]
    manifest = L.load_manifest(args.manifest)

    if args.deposit:
        dep = request('GET', f'{api}/deposit/depositions/{args.deposit}', tok)
    else:
        dep = request('POST', f'{api}/deposit/depositions', tok, data={})
        print(f"created draft deposit {dep['id']}")
    if dep.get('submitted'):
        raise SystemExit(f"deposit {dep['id']} is already published; this script only edits drafts "
                         "(create a new version in the web interface first)")
    request('PUT', f"{api}/deposit/depositions/{dep['id']}", tok, data={'metadata': METADATA})
    print('metadata set')

    if not args.metadata_only:
        if not args.archives:
            raise SystemExit('--archives DIR is required to upload files')
        bucket = dep['links']['bucket']
        # archives whose md5 Zenodo already holds are skipped, so a re-run only uploads what changed
        existing = {f['filename']: f for f in request('GET', f"{api}/deposit/depositions/{dep['id']}/files", tok)}
        wanted = {w.strip() for w in args.only.split(',')} if args.only else None
        for name, a in manifest['archives'].items():
            if wanted and name not in wanted:
                continue
            path = args.archives / name
            if not path.exists():
                raise SystemExit(f'{path} missing; run data/build_manifest.py first')
            if name in existing and existing[name].get('checksum', '').replace('md5:', '') == a['md5']:
                print(f'{name}: already uploaded'); continue
            print(f"{name}: uploading {a['bytes'] / 1e9:.2f} GB ...", flush=True)
            info = upload_file(bucket, path, tok)
            got = info.get('checksum', '').replace('md5:', '')
            if got != a['md5']:
                raise SystemExit(f'{name}: Zenodo reports md5 {got}, expected {a["md5"]}')
            print(f'{name}: ok')

    manifest['record_id'] = dep['id']          # record id == deposit id for the published record
    manifest['sandbox'] = bool(args.sandbox)
    L.save_manifest(manifest, args.manifest)
    print(f"draft: {dep['links'].get('html', '')}\n"
          "review it in the Zenodo web interface; publish there when the files are final "
          "(this script does not publish).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
