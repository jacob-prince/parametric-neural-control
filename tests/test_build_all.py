"""scripts/preprocessing/build_all.py: stage table, frozen inputs, pinned environment."""
import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import REPO

spec = importlib.util.spec_from_file_location('build_all', REPO / 'scripts' / 'preprocessing' / 'build_all.py')
build_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_all)


def test_every_stage_script_exists_and_is_unique():
    names = [s[0] for s in build_all.STAGES]
    assert len(names) == len(set(names))
    for name, script, extra in build_all.STAGES:
        assert script.exists(), f'{name}: {script}'
        assert all(isinstance(a, str) for a in extra)


def test_stage_order_respects_dependencies():
    names = [s[0] for s in build_all.STAGES]
    assert names[0] == 'run_preproc'
    assert names.index('build_advsens_outcome_reliability') > names.index('run_preproc')
    assert names.index('fig5_heldout_analysis') < names.index('fig5_outcome_ceiling_slope_resid9')
    assert names.index('build_fig6_caches') > names.index('run_preproc')


def test_frozen_inputs_match_the_provenance_document():
    doc = (REPO / 'data' / 'FROZEN_INPUTS.md').read_text()
    listed = {line.split('`')[1].split('/')[0] for line in doc.splitlines() if line.startswith('| `')}
    assert listed == set(build_all.FROZEN)


def test_pinned_env_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv('PNC_SOURCE_DATA', '/src')
    monkeypatch.setenv('PNC_PREPROCESSED_DATA', '/should/be/overridden')
    env = build_all.pinned_env(tmp_path)
    assert env['PNC_PREPROCESSED_DATA'] == str(tmp_path) and env['PNC_SOURCE_DATA'] == '/src'
    assert env['PYTHONHASHSEED'] == '0' and env['OPENBLAS_NUM_THREADS'] == '1' and env['MPLBACKEND'] == 'Agg'


def test_stage_frozen_copies_only_the_listed_inputs(monkeypatch, tmp_path):
    src = tmp_path / 'src' / 'frozen_inputs'
    src.mkdir(parents=True)
    for name in build_all.FROZEN:
        if name in ('grad_maps', 'fig6_imagenet_thumbs'):
            (src / name).mkdir(); (src / name / 'a.bin').write_bytes(b'1'); (src / name / '._junk').write_bytes(b'x')
        else:
            (src / name).write_bytes(b'data')
    (src / 'not_listed.txt').write_text('x')
    monkeypatch.setenv('PNC_SOURCE_DATA', str(tmp_path / 'src'))
    out = tmp_path / 'out'; out.mkdir()
    copied = build_all.stage_frozen(out)
    assert set(copied) == set(build_all.FROZEN)
    assert not (out / 'not_listed.txt').exists()
    assert (out / 'grad_maps' / 'a.bin').exists() and not (out / 'grad_maps' / '._junk').exists()
    (src / build_all.FROZEN[0]).unlink()
    with pytest.raises(build_all.paths.MissingDataError):
        build_all.stage_frozen(out)


def test_list_and_unknown_stage(capsys):
    assert build_all.main(['--list']) == 0
    assert 'run_preproc' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        build_all.main(['--only', 'no_such_stage', '--out', '/tmp/x', '--skip-frozen'])


def test_write_hashes_skips_logs_and_manifest(tmp_path):
    (tmp_path / 'a.pkl').write_bytes(b'a')
    (tmp_path / '.build_logs').mkdir(); (tmp_path / '.build_logs' / 'x.log').write_text('log')
    (tmp_path / 'BUILD_MANIFEST.json').write_text('{}')
    rows = build_all.write_hashes(tmp_path)
    assert list(rows) == ['a.pkl'] and rows['a.pkl']['bytes'] == 1
