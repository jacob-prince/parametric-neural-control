"""Filesystem layout. Every path the pipeline touches is derived from three roots, each
overridable through an environment variable so the same code runs against the downloaded
Zenodo trees (default, repo-relative) or any other location:

    PNC_SOURCE_DATA         raw inputs (default <repo>/source_data)
    PNC_PREPROCESSED_DATA   derived caches read by the figure scripts (default <repo>/preprocessed_data)
    PNC_OUTPUT              rendered figures and other outputs (default <repo>/outputs)

Both data trees are populated by `python data/download_data.py` (see README).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / 'figures' / 'assets'


# Subclass of FileNotFoundError so existing except-clauses still catch it; the message
# carries the download hint.
class MissingDataError(FileNotFoundError):
    pass


def _root(env, default):
    # `or`, not a dict default: an env var that is set but empty also falls back
    return Path(os.environ.get(env) or default).expanduser()


def source_data():
    return _root('PNC_SOURCE_DATA', REPO_ROOT / 'source_data')


def preprocessed_data():
    return _root('PNC_PREPROCESSED_DATA', REPO_ROOT / 'preprocessed_data')


def output_dir():
    return _root('PNC_OUTPUT', REPO_ROOT / 'outputs')


def cluster_outputs():
    """GPU/cluster results read directly at render time (source_data/cluster_outputs/<job>/)."""
    return source_data() / 'cluster_outputs'


def frozen_inputs():
    """Inputs with no in-repo producer (see source_data/frozen_inputs/PROVENANCE.md)."""
    return source_data() / 'frozen_inputs'


def require(path, hint=None):
    """Return `path` if it exists, else raise MissingDataError with a download hint."""
    path = Path(path)
    if not path.exists():
        # name the download tier that owns this path, so the hint gives the right --tier
        where = 'source_data' if str(path).startswith(str(source_data())) else 'preprocessed_data'
        tier = 'source' if where == 'source_data' else 'preprocessed'
        msg = (f"missing {path}\n  -> run `python data/download_data.py --tier {tier}`"
               + (f"\n  -> or {hint}" if hint else ''))
        raise MissingDataError(msg)
    return path
