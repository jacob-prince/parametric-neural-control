"""Documentation and metadata stay in step with the code: captions per figure, README commands
point at real files, citation authors match the Zenodo metadata, reference files agree with the
manifest."""
import json
import re
import sys

import pytest

from conftest import REPO
from pnc import manifest

sys.path.insert(0, str(REPO / 'data'))
import zenodo_upload  # noqa: E402


def test_captions_cover_every_figure_once():
    text = (REPO / 'figures' / 'CAPTIONS.md').read_text()
    stems = re.findall(r'^### .*?`([^`]+)`', text, flags=re.M)
    assert stems == manifest.all_output_names()
    bodies = re.split(r'^### .*$', text, flags=re.M)[1:]
    assert all(len(b.strip()) > 40 for b in bodies), 'empty caption'
    outside_math = re.sub(r'\$[^$]*\$', '', text)
    assert not re.findall(r'\\(?:textbf|cite\w*|sfref|sfrange|cref)\b', outside_math)


def test_readme_commands_refer_to_existing_files():
    text = (REPO / 'README.md').read_text()
    for rel in re.findall(r'`python ([\w./-]+\.py)', text):
        assert (REPO / rel).exists(), rel
    code_dirs = ('pnc', 'core', 'neural_regress', 'figures', 'notebooks', 'scripts', 'data', 'tests')
    for rel in re.findall(r'`([\w./-]+/)`', text):
        if rel.split('/')[0] in code_dirs:            # data-tree paths (source_data/..., cluster_outputs/) are not in git
            assert (REPO / rel).is_dir(), rel
    for rel in re.findall(r'src="([^"]+)"', text) + re.findall(r'\]\(([\w./-]+\.md)\)', text):
        assert (REPO / rel).exists(), rel
    for name in ('environment.yml', 'LICENSE'):
        assert name in text and (REPO / name).exists()
    docs = (REPO / 'docs' / 'REPRODUCIBILITY.md').read_text()
    for rel in re.findall(r'`python ([\w./-]+\.py)', docs):
        assert (REPO / rel).exists(), rel
    for name in ('pyproject.toml', 'requirements-scripts.txt', 'CITATION.cff'):
        assert (REPO / name).exists()


def test_readme_mentions_every_top_level_package():
    text = (REPO / 'README.md').read_text() + (REPO / 'docs' / 'REPRODUCIBILITY.md').read_text()
    for d in ('pnc/', 'core/', 'neural_regress/', 'figures/', 'notebooks/', 'scripts/', 'data/', 'tests/'):
        assert d in text, d


def test_readme_gifs_are_small_enough_for_github():
    for gif in (REPO / 'docs' / 'assets').glob('*.gif'):
        assert gif.stat().st_size < 10_000_000, gif.name


def test_citation_and_zenodo_metadata_agree():
    cff = (REPO / 'CITATION.cff').read_text()
    families = re.findall(r'family-names: (\S+)', cff)[:8]
    zen = [c['name'].split(',')[0] for c in zenodo_upload.METADATA['creators']]
    assert families == zen
    assert zenodo_upload.METADATA['license'] == 'cc-by-4.0' and zenodo_upload.METADATA['upload_type'] == 'dataset'
    assert 'repository-code: "https://github.com/jacob-prince/parametric-neural-control"' in cff
    assert 'MIT' in (REPO / 'LICENSE').read_text() and 'license: MIT' in cff


def test_reference_hash_files_are_well_formed():
    fig = json.loads((REPO / 'tests' / 'reference' / 'figure_pixel_hashes.json').read_text())['figures']
    assert list(fig) == manifest.all_output_names()
    for stem, r in fig.items():
        assert len(r['sha256']) == 64 and len(r['shape']) == 3 and r['shape'][2] == 4 and min(r['shape'][:2]) > 100
        assert (REPO / r['script']).exists(), stem
    pre = json.loads((REPO / 'tests' / 'reference' / 'preprocessed_data_hashes.json').read_text())['files']
    for name in ('brain_red.pkl', 'stimuli.pkl', 'gradient_freq.pkl', 'layer_selection.pkl', 'sup_reliability_channel_selection_cache.pkl',
                 'fig5_outcome_ceiling_seedsplit_resid9.csv', 'fig5_outcome_ceiling_seedsplit_noresid.csv', 'fig5_outcome_ceiling_seedsplit_r_resid9.csv'):
        assert name in pre, name
    assert 'z-old_sup_predicting_master_26col.csv' not in pre
    src = json.loads((REPO / 'tests' / 'reference' / 'source_data_hashes.json').read_text())['source_data']
    assert len(src) > 5000 and sum(v['bytes'] for v in src.values()) < 50e9
    assert any(k.startswith('stimuli_control/') for k in src) and any(k.startswith('frozen_inputs/') for k in src)


def test_environment_yml_lists_the_notebook_tooling():
    text = (REPO / 'environment.yml').read_text()
    for pkg in ('nbformat', 'nbclient', 'ipykernel', 'scikit-learn', 'pytorch'):
        assert re.search(rf'^\s*-\s*{pkg}\b', text, flags=re.M), pkg


def test_gitignore_blocks_data_and_private_fonts():
    text = (REPO / '.gitignore').read_text()
    for pat in ('source_data/', 'preprocessed_data/', 'outputs/', '*.h5', '*.pkl', '*.pt', '.DS_Store', '._*',
                'figures/assets/private/', 'tests/reference/manuscript_png/'):
        assert pat in text.splitlines(), pat
