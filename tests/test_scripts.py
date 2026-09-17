"""Every script under scripts/ compiles, and every argparse CLI answers --help. Scripts that need
GPU-only packages (timm, open_clip, clip, horama, yaml, ...) are only compiled."""
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

SCRIPTS = sorted(p for p in (REPO / 'scripts').rglob('*.py') if '__pycache__' not in p.parts)
OPTIONAL = {'timm', 'open_clip', 'clip', 'horama', 'yaml', 'boto3', 'cv2', 'skimage', 'imageio', 'einops',
            'dill', 'xarray', 'netCDF4', 'cuml', 'tqdm', 'sklearn', 'torch', 'torchvision', 'scipy', 'requests',
            'facenet_pytorch', 'pycocotools', 'kornia', 'lpips'}


@pytest.mark.parametrize('path', SCRIPTS, ids=[str(p.relative_to(REPO / 'scripts')) for p in SCRIPTS])
def test_script_compiles(path):
    py_compile.compile(str(path), doraise=True, cfile=None if False else str(Path('/dev/null')) if sys.platform != 'win32' else None)


CLI = [p for p in SCRIPTS if re.search(r'argparse|--help', p.read_text())]


@pytest.mark.parametrize('path', CLI, ids=[str(p.relative_to(REPO / 'scripts')) for p in CLI])
def test_script_help(path):
    r = subprocess.run([sys.executable, str(path), '--help'], cwd=str(REPO), text=True, capture_output=True,
                       timeout=300, env={**dict(__import__('os').environ), 'KMP_DUPLICATE_LIB_OK': 'TRUE'})
    if r.returncode != 0:
        m = re.search(r"No module named '([A-Za-z0-9_]+)", r.stderr)
        if m and m.group(1) in OPTIONAL:
            pytest.skip(f'optional dependency {m.group(1)} not installed')
    assert r.returncode == 0, r.stderr[-2000:]


def test_no_private_paths_outside_comments():
    bad = []
    for p in SCRIPTS + sorted((REPO / 'scripts').rglob('*.sh')):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            code = line.split('#', 1)[0]
            if re.search(r'/n/(holylabs|holylfs|home\d+|netscratch)|/Volumes/CORSAIR|/Users/jacobprince|binxuwang', code):
                bad.append(f'{p.relative_to(REPO)}:{n}: {line.strip()[:100]}')
    assert not bad, 'private paths in code (keep them only in comments):\n' + '\n'.join(bad)
