#!/usr/bin/env python3
"""Regenerate figures/CAPTIONS.md from the manuscript LaTeX sources (maintainer tool; the
manuscript tree is not part of this repository)."""
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnc import manifest  # noqa: E402

M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/Accentuate-VVS/Manuscript/02-Nature-Submission/take1')
spec = importlib.util.spec_from_file_location('supcap', M / 'supplement_captions.py')
supcap = importlib.util.module_from_spec(spec); spec.loader.exec_module(supcap)
CAP = supcap.CAPTIONS
NUM = {e['slug']: e['num'] for e in manifest.all_entries()}
INV_MAIN = {v: k for k, v in manifest.MAIN_FIGURES.items()}


def braces(s, start):
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start + 1:i]
    raise ValueError('unbalanced braces')


def tex2md(s):
    s = re.sub(r'(?<!\\)%[^\n]*', '', s)
    s = re.sub(r'\\sfrange\{([a-z0-9_]+)\}', lambda m: f"S{NUM[m.group(1)]}-S{NUM[m.group(1)] + 4}", s)
    s = re.sub(r'\\sfref\{([a-z0-9_]+)\}', lambda m: f"S{NUM.get(m.group(1), '?')}", s)
    s = re.sub(r'\\[cC]ref\{fig:([a-z_]+)\}', lambda m: f'Fig. {INV_MAIN.get(m.group(1), m.group(1))}', s)
    for _ in range(3):
        s = re.sub(r'\\textbf\{([^{}]*)\}', r'**\1**', s)
        s = re.sub(r'\\(?:emph|textit)\{([^{}]*)\}', r'*\1*', s)
        s = re.sub(r'\\texttt\{([^{}]*)\}', r'`\1`', s)
    s = re.sub(r'\s*\(?\\(?:citealp|citep|citet|cite)\{[^{}]*\}\)?', '', s)
    s = re.sub(r'\\url\{([^{}]*)\}', r'\1', s)
    s = s.replace('``', '"').replace("''", '"').replace('\\%', '%').replace('\\&', '&').replace('~', ' ') \
         .replace('\\,', ' ').replace('\\ ', ' ').replace('\\-', '')
    s = re.sub(r'\\(?:vspace|hspace)\*?\{[^{}]*\}', '', s)
    s = re.sub(r'\\label\{[^{}]*\}', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    out = ['# Figure captions', '',
           'Converted from the manuscript LaTeX sources (main.tex figure environments and the supplementary '
           'caption table); equations stay in TeX math notation. The notebooks under notebooks/figures/ carry '
           'these captions in their title cell.', '', '## Main figures', '']
    for i, name in manifest.MAIN_FIGURES.items():
        tex = (M / 'tex' / 'figures' / f'{name}.tex').read_text()
        m = re.search(r'\\(?:breakable)?caption', tex)
        k = tex.index('{', m.end())
        out += [f'### Figure {i}: `{name}`', '', tex2md(braces(tex, k)), '']
    out += ['## Supplementary figures', '']
    for e in manifest.all_entries():
        cap = tex2md(CAP.get(e['slug'], e['title']))
        if e['span']:
            for mk in manifest.SPAN_MONKEYS:
                stem = manifest.output_name(e['slug'], mk)
                out += [f"### Supplementary Figure S{int(stem[1:3])}: `{stem}`", '',
                        f"**{e['title']} ({mk}).** " + cap, '']
        else:
            out += [f"### Supplementary Figure S{e['num']}: `{manifest.output_name(e['slug'])}`", '', cap, '']
    (REPO / 'figures' / 'CAPTIONS.md').write_text('\n'.join(out))
    txt = '\n'.join(out)
    left = sorted(set(re.findall(r'\\[a-zA-Z]+', re.sub(r'\$[^$]*\$', '', txt))))
    print(f"{txt.count('### ')} captions written; tex commands left outside math: {left}")


if __name__ == '__main__':
    main()
