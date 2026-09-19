"""Directory listings must be sorted: the reference renders were produced on a filesystem
that returns sorted listings, and any `glob(...)[0]`, `pd.concat` over listed files or
first-wins loop would otherwise depend on the host filesystem."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN = ['pnc', 'figures', 'scripts/preprocessing']
PATTERN = re.compile(r'(?<![\w.])(glob\.glob|globmod\.glob|os\.listdir|\.iterdir|\.glob|\.rglob)\(')


def _unsorted_calls(text):
    bad = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in PATTERN.finditer(line):
            before = line[:m.start()]
            if 'sorted(' in before or line.lstrip().startswith('#') or 'def ' in before:
                continue
            bad.append((lineno, line.strip()))
    return bad


def test_every_directory_listing_is_sorted():
    offenders = []
    for top in SCAN:
        for path in sorted((REPO / top).rglob('*.py')):
            for lineno, line in _unsorted_calls(path.read_text()):
                offenders.append(f'{path.relative_to(REPO)}:{lineno}: {line}')
    assert not offenders, 'unsorted directory listings:\n' + '\n'.join(offenders)
