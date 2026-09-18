"""The data on disk matches data/zenodo_manifest.json: every archive's members are present with
the recorded sha256 (this is what download_data.py --verify-only checks). Skips until the
manifest has been built."""
import json
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / 'data'))
import _manifest_lib as L  # noqa: E402


def _manifest():
    m = L.load_manifest()
    if not m['archives']:
        pytest.skip('data/zenodo_manifest.json has no archives yet (run data/build_manifest.py)')
    return m


def test_manifest_is_well_formed():
    m = _manifest()
    for name, a in m['archives'].items():
        assert a['tier'] in ('source_data', 'preprocessed_data', 'reference_figures'), name
        assert len(a['sha256']) == 64 and len(a['md5']) == 32 and a['bytes'] > 0, name
    total = sum(a['bytes'] for a in m['archives'].values())
    assert len(m['archives']) <= 100, 'Zenodo default limit is 100 files per record'
    assert total < 50e9, 'Zenodo default limit is 50 GB per record'


@pytest.mark.slow
def test_data_on_disk_matches_manifest(paths):
    m = _manifest()
    hashes = json.loads((REPO / 'tests' / 'reference' / 'source_data_hashes.json').read_text()) \
        if (REPO / 'tests' / 'reference' / 'source_data_hashes.json').exists() else None
    if hashes is None:
        pytest.skip('tests/reference/source_data_hashes.json not built yet')
    missing, bad = [], []
    for tier, files in hashes.items():
        if tier.startswith('_'):
            continue
        root = paths['source'] if tier == 'source_data' else paths['cache']
        if not root.exists():
            pytest.skip(f'{tier} not present')
        for rel, h in files.items():
            p = root / rel
            if not p.exists():
                missing.append(rel); continue
            if p.stat().st_size != h['bytes'] or L.digests(p)[0] != h['sha256']:
                bad.append(rel)
    assert not missing and not bad, f'missing={missing[:10]} ({len(missing)}) mismatched={bad[:10]} ({len(bad)})'
