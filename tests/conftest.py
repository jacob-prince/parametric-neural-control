import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.dont_write_bytecode = True


def pytest_addoption(parser):
    parser.addoption('--run-raw', action='store_true', default=False,
                     help='also validate the large raw HDF5 / session sources (needs source_data)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-raw'):
        return
    skip = pytest.mark.skip(reason='needs --run-raw')
    for item in items:
        if 'raw' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def repo():
    return REPO
