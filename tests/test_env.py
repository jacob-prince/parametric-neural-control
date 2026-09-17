"""Report whether this interpreter is the reference (pixel-exact) environment. This test never
fails on version drift -- the pixel tests switch to tolerance mode instead -- but it fails if
the declared pins in environment.yml and conftest.REFERENCE_VERSIONS disagree with each other."""
import re
from pathlib import Path

from conftest import REFERENCE_VERSIONS, REPO, reference_env_status


def test_environment_yml_pins_match_reference_versions():
    text = (REPO / 'environment.yml').read_text()
    pins = dict(re.findall(r'^\s*-\s*([a-z0-9_-]+)=([0-9][0-9a-z.]*)\s*$', text, flags=re.M))
    assert pins['python'] == REFERENCE_VERSIONS['python']
    assert pins['numpy'] == REFERENCE_VERSIONS['numpy']
    assert pins['matplotlib'] == REFERENCE_VERSIONS['matplotlib']
    assert pins['pillow'] == REFERENCE_VERSIONS['pillow']
    assert pins['freetype'] == REFERENCE_VERSIONS['freetype']


def test_report_reference_environment(capsys):
    ok, details = reference_env_status()
    print(f"reference environment: {'YES (pixel-exact mode)' if ok else 'NO (tolerance mode)'} -- {details}")
    assert isinstance(ok, bool)
