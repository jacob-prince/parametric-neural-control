"""More coverage of the data tooling: pattern archives, member manifests, archive naming,
resumable HTTP downloads against a local range-capable server, and downloader edge cases."""
import json
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / 'data'))
import _manifest_lib as L  # noqa: E402
import build_manifest  # noqa: E402
import download_data  # noqa: E402


def test_pattern_archive_selects_only_matching_files(tmp_path):
    src = tmp_path / 'enc'; src.mkdir()
    (src / 'red_x_sessdata.pkl').write_bytes(b'1'); (src / 'red.h5').write_bytes(b'2'); (src / '._red_x_sessdata.pkl').write_bytes(b'3')
    members = L.write_tar(tmp_path / 'a.tar', src, 'source_data/enc', pattern='*_sessdata.pkl')
    assert list(members) == ['red_x_sessdata.pkl']
    with tarfile.open(tmp_path / 'a.tar') as tf:
        assert tf.getnames() == ['source_data/enc/MEMBERS.json', 'source_data/enc/red_x_sessdata.pkl']


def test_gzip_archive_is_reproducible(tmp_path):
    src = tmp_path / 'cfg'; src.mkdir(); (src / 'a.yaml').write_text('a: 1\n')
    L.write_tar(tmp_path / '1.tar.gz', src, 'source_data/cfg', compress=True)
    L.write_tar(tmp_path / '2.tar.gz', src, 'source_data/cfg', compress=True)
    assert (tmp_path / '1.tar.gz').read_bytes() == (tmp_path / '2.tar.gz').read_bytes()
    members, prefix = download_data.read_members(tmp_path / '1.tar.gz')
    assert prefix == 'source_data/cfg' and list(members) == ['a.yaml']


def test_read_members_rejects_archives_without_a_manifest(tmp_path):
    with tarfile.open(tmp_path / 'bad.tar', 'w') as tf:
        p = tmp_path / 'f.txt'; p.write_text('x'); tf.add(p, arcname='source_data/x/f.txt')
    with pytest.raises(RuntimeError, match='MEMBERS.json'):
        download_data.read_members(tmp_path / 'bad.tar')


@pytest.mark.parametrize('tier,rel,kind,compress,expected', [
    ('preprocessed_data', '.', 'dir', False, 'preprocessed_data.tar'),
    ('source_data', 'image_pca_projections', 'dir', False, 'source_data__image_pca_projections.tar'),
    ('source_data', 'accentuation_configs', 'dir', True, 'source_data__accentuation_configs.tar.gz'),
    ('source_data', 'brain_data_encoding', 'glob:*_sessdata.pkl', False, 'source_data__brain_data_encoding__sessdata_pkl.tar'),
    ('source_data', 'brain_data_control/x.h5', 'file', False, 'source_data__x.h5'),
])
def test_archive_names(tier, rel, kind, compress, expected):
    assert build_manifest.archive_name(tier, rel, kind, compress) == expected


def test_layout_covers_every_tier_a_directory_once():
    rels = [(t, r) for t, r, k, c in build_manifest.LAYOUT]
    assert len(rels) == len(set(rels))
    tops = {r.split('/')[0] for t, r, k, c in build_manifest.LAYOUT if t == 'source_data'}
    for d in ('brain_data_encoding', 'brain_data_control', 'brain_data_controversial', 'image_pca_projections',
              'model_predictions', 'model_features', 'stimuli_encoding', 'stimuli_control', 'stimuli_controversial',
              'accentuation_configs', 'model_backbones', 'cluster_outputs', 'frozen_inputs', 'encoding_model_outputs'):
        assert d in tops, d


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b''
    fail_first = [0]

    def log_message(self, *a):
        pass

    def do_GET(self):
        if _RangeHandler.fail_first[0] > 0:
            _RangeHandler.fail_first[0] -= 1
            self.send_response(503); self.end_headers(); return
        start = 0
        rng = self.headers.get('Range')
        if rng:
            start = int(rng.split('=')[1].rstrip('-'))
        body = _RangeHandler.payload[start:start + 1000] if rng else _RangeHandler.payload[:1000]  # serve in 1000-byte pieces
        self.send_response(206 if rng else 200)
        self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def range_server():
    srv = HTTPServer(('127.0.0.1', 0), _RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f'http://127.0.0.1:{srv.server_port}/f'
    srv.shutdown()


def test_http_download_resumes_and_retries(range_server, tmp_path, monkeypatch):
    monkeypatch.setattr(download_data.time, 'sleep', lambda s: None)
    _RangeHandler.payload = bytes(range(256)) * 10          # 2560 bytes -> needs several range requests
    _RangeHandler.fail_first[0] = 2                          # two transient 503s first
    dst = tmp_path / 'f.part'
    dst.write_bytes(_RangeHandler.payload[:700])             # a partial file from an earlier attempt
    download_data.http_download(range_server, dst, len(_RangeHandler.payload))
    assert dst.read_bytes() == _RangeHandler.payload


def test_http_download_gives_up_after_persistent_failures(range_server, tmp_path, monkeypatch):
    monkeypatch.setattr(download_data.time, 'sleep', lambda s: None)
    _RangeHandler.payload = b'x' * 10
    _RangeHandler.fail_first[0] = 100
    with pytest.raises(RuntimeError):
        download_data.http_download(range_server, tmp_path / 'g.part', 10, retries=3)


def test_verify_checks_size_then_digests(tmp_path):
    p = tmp_path / 'f'; p.write_bytes(b'abc')
    sha, md5, n = L.digests(p)
    entry = dict(bytes=n, sha256=sha, md5=md5)
    assert download_data.verify(p, entry) and download_data.verify(p, entry, fast=True)
    assert not download_data.verify(p, dict(entry, bytes=4))
    assert not download_data.verify(p, dict(entry, sha256='0' * 64))
    assert download_data.verify(p, dict(entry, sha256='0' * 64), fast=True)


def test_downloader_rejects_unknown_only(tmp_path, monkeypatch):
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {'a.tar': dict(tier='source_data', path='a', kind='dir', bytes=1, sha256='0', md5='0', n_members=1)},
                                    'record_id': 1, 'sandbox': False}))
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 's'))
    with pytest.raises(SystemExit, match='--only'):
        download_data.main(['--only', 'zzz', '--manifest', str(manifest)])


def test_raw_file_tier_round_trip_from_local(tmp_path, monkeypatch):
    archives = tmp_path / 'archives'; archives.mkdir()
    (archives / 'source_data__big.h5').write_bytes(b'\x89HDF' * 100)
    sha, md5, n = L.digests(archives / 'source_data__big.h5')
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'archives': {'source_data__big.h5': dict(tier='source_data', path='brain_data_control/big.h5', kind='file',
                                                                              bytes=n, sha256=sha, md5=md5, n_members=1)},
                                    'record_id': None, 'sandbox': False}))
    dest = tmp_path / 'source_data'
    monkeypatch.setenv('PNC_SOURCE_DATA', str(dest)); monkeypatch.setenv('PNC_PREPROCESSED_DATA', str(tmp_path / 'p'))
    args = ['--tier', 'source', '--from-local', str(archives), '--manifest', str(manifest)]
    assert download_data.main(args) == 0
    assert (dest / 'brain_data_control' / 'big.h5').read_bytes() == b'\x89HDF' * 100
    assert download_data.main(args + ['--verify-only']) == 0
    (dest / 'brain_data_control' / 'big.h5').write_bytes(b'corrupt')
    assert download_data.main(args + ['--verify-only']) == 1
    with pytest.raises(SystemExit, match='--force'):
        download_data.main(args)
    assert download_data.main(args + ['--force']) == 0
