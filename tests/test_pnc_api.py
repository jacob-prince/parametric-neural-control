"""The public surface of pnc is frozen: figure scripts are ported against it in parallel,
so a name may not silently disappear or change module. Regenerate the snapshot with
`python tests/test_pnc_api.py` only when a change is intended."""
import importlib
import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent / 'reference' / 'pnc_api.json'
MODULES = ['pnc.paths', 'pnc.utils', 'pnc.trim', 'pnc.manifest', 'pnc.famstats',
           'pnc.preproc.loader', 'pnc.preproc.pipeline', 'pnc.preproc.ceilings',
           'pnc.preproc.stimuli', 'pnc.preproc.encoding', 'pnc.preproc.floors',
           'pnc.preproc.exclusions', 'pnc.preproc.fft_utils', 'pnc.preproc.controversial']


def public_names():
    out = {}
    for name in MODULES:
        mod = importlib.import_module(name)
        out[name] = sorted(n for n in vars(mod) if not n.startswith('_')
                           and getattr(getattr(mod, n), '__module__', name) in (name, None)
                           or (not n.startswith('_') and n.isupper()))
    return out


def test_pnc_public_api_unchanged():
    expected = json.loads(SNAPSHOT.read_text())
    actual = public_names()
    missing = {m: sorted(set(expected[m]) - set(actual.get(m, []))) for m in expected}
    missing = {m: v for m, v in missing.items() if v}
    assert not missing, f'names removed from the frozen pnc API: {missing}'


if __name__ == '__main__':
    SNAPSHOT.write_text(json.dumps(public_names(), indent=1) + '\n')
    print(f'wrote {SNAPSHOT}')
