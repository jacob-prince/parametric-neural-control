"""The data tooling must never publish a deposit and must read the token from ZENODO_TOKEN only;
the downloader must verify checksums and refuse to clobber differing files."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / 'data'))
import _manifest_lib as L  # noqa: E402
import build_manifest  # noqa: E402
import download_data  # noqa: E402
import zenodo_upload  # noqa: E402


def test_uploader_has_no_publish_call():
    src = (REPO / 'data' / 'zenodo_upload.py').read_text()
    assert 'actions/publish' not in src
    assert 'ZENODO_TOKEN' in src and "environ.get('ZENODO_TOKEN')" in src
    assert 'token=' not in src.replace('token()', '')  # no token passed as a query parameter


def test_uploader_requires_env_token(monkeypatch):
    monkeypatch.delenv('ZENODO_TOKEN', raising=False)
    with pytest.raises(SystemExit):
        zenodo_upload.token()
    monkeypatch.setenv('ZENODO_TOKEN', 'abc')
    assert zenodo_upload.token() == 'abc'


def test_uploader_refuses_published_deposits(monkeypatch, tmp_path):
    monkeypatch.setenv('ZENODO_TOKEN', 'abc')
    calls = []

    def fake_request(method, url, tok, data=None, headers=None, raw=None):
        calls.append((method, url))
        return {'id': 7, 'submitted': True, 'links': {}}
    monkeypatch.setattr(zenodo_upload, 'request', fake_request)
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {}, 'record_id': None, 'sandbox': False}))
    with pytest.raises(SystemExit, match='already published'):
        zenodo_upload.main(['--deposit', '7', '--metadata-only', '--manifest', str(manifest)])
    assert all('publish' not in url for _, url in calls)


def test_uploader_draft_flow_records_record_id(monkeypatch, tmp_path):
    monkeypatch.setenv('ZENODO_TOKEN', 'abc')
    archive = tmp_path / 'x.tar'
    archive.write_bytes(b'payload')
    sha, md5, n = L.digests(archive)
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {'x.tar': dict(tier='preprocessed_data', path='.', kind='dir',
                                                               bytes=n, sha256=sha, md5=md5, n_members=1)},
                                    'record_id': None, 'sandbox': False}))
    calls = []

    def fake_request(method, url, tok, data=None, headers=None, raw=None):
        calls.append((method, url))
        if method == 'POST':
            return {'id': 42, 'submitted': False, 'links': {'bucket': 'https://b', 'html': 'https://zenodo.org/deposit/42'}}
        if url.endswith('/files'):
            return []
        return {'id': 42, 'submitted': False, 'links': {'bucket': 'https://b'}}
    monkeypatch.setattr(zenodo_upload, 'request', fake_request)
    monkeypatch.setattr(zenodo_upload, 'upload_file', lambda bucket, path, tok: {'checksum': f'md5:{md5}'})
    assert zenodo_upload.main(['--archives', str(tmp_path), '--manifest', str(manifest)]) == 0
    assert json.loads(manifest.read_text())['record_id'] == 42
    assert all('publish' not in url for _, url in calls)


def _make_tree(root):
    (root / 'a').mkdir(parents=True)
    (root / 'a' / 'one.txt').write_text('one')
    (root / 'a' / 'two.bin').write_bytes(b'\x00\x01' * 10)
    (root / 'a' / '.DS_Store').write_bytes(b'junk')


def test_tar_is_deterministic_and_carries_members(tmp_path):
    src = tmp_path / 'src'
    _make_tree(src)
    t1, t2 = tmp_path / 't1.tar', tmp_path / 't2.tar'
    m1 = L.write_tar(t1, src / 'a', 'source_data/a')
    m2 = L.write_tar(t2, src / 'a', 'source_data/a')
    assert t1.read_bytes() == t2.read_bytes()
    assert set(m1) == {'one.txt', 'two.bin'} and m1 == m2
    with tarfile.open(t1) as tf:
        names = tf.getnames()
    assert names[0] == 'source_data/a/MEMBERS.json' and '.DS_Store' not in ' '.join(names)


def test_download_from_local_extracts_and_verifies(tmp_path, monkeypatch):
    src = tmp_path / 'src'
    _make_tree(src)
    archives = tmp_path / 'archives'
    archives.mkdir()
    name = 'source_data__a.tar'
    L.write_tar(archives / name, src / 'a', 'source_data/a')
    sha, md5, n = L.digests(archives / name)
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {name: dict(tier='source_data', path='a', kind='dir',
                                                            bytes=n, sha256=sha, md5=md5, n_members=2)},
                                    'record_id': None, 'sandbox': False}))
    dest = tmp_path / 'source_data'
    monkeypatch.setenv('PNC_SOURCE_DATA', str(dest))
    monkeypatch.setenv('PNC_PREPROCESSED_DATA', str(tmp_path / 'preprocessed_data'))
    args = ['--tier', 'source', '--from-local', str(archives), '--manifest', str(manifest)]
    assert download_data.main(args) == 0
    assert (dest / 'a' / 'one.txt').read_text() == 'one'
    # second run: everything present -> no-op, still ok
    assert download_data.main(args) == 0
    assert download_data.main(args + ['--verify-only']) == 0
    # a differing file is never clobbered silently
    (dest / 'a' / 'one.txt').write_text('changed')
    with pytest.raises(RuntimeError, match='--force'):
        download_data.main(args)
    assert download_data.main(args + ['--force']) == 0
    assert (dest / 'a' / 'one.txt').read_text() == 'one'
    # a corrupted archive is rejected
    (archives / name).write_bytes(b'garbage')
    assert download_data.main(args) == 1


def test_download_without_record_id_needs_local_archives(tmp_path, monkeypatch):
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {'x': dict(tier='source_data', path='x', kind='file',
                                                           bytes=1, sha256='0', md5='0', n_members=1)},
                                    'record_id': None, 'sandbox': False}))
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 's'))
    with pytest.raises(SystemExit, match='record_id'):
        download_data.main(['--tier', 'source', '--manifest', str(manifest)])


def test_file_url_points_at_zenodo_records_api():
    assert download_data.file_url(123, 'a.tar') == 'https://zenodo.org/api/records/123/files/a.tar/content'
    assert download_data.file_url(123, 'a.tar', sandbox=True).startswith('https://sandbox.zenodo.org/')
