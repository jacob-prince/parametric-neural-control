#!/usr/bin/env python3
"""Generate one walkthrough notebook per manuscript figure under notebooks/figures/ (55 notebooks).

    python notebooks/_make_figure_notebooks.py            # (re)write all notebooks, keeping rendered outputs
    python notebooks/_make_figure_notebooks.py --check    # exit 1 if any committed notebook is stale

Each notebook lays out the figure script itself, in order: the script's header as an
introduction, the imports and configuration, every helper function in its own cell with its
docstring as the explanation, the body of main() inlined as the assembly (with the script's
parameters exposed in a cell of their own), the rendered figure, and the manuscript caption
under it. Code cells are verbatim slices of the script, so executing the notebook reproduces
the manuscript PNG pixel for pixel in the reference environment.

Explanatory text is generated from the script's docstrings and comments; a Markdown file
notebooks/figures/_narrative/<slug>.md can override or extend it, with sections
"## intro", "## setup", "## fn:<function name>", "## assemble", "## step:<n>" and "## closing".
--check compares cell sources and ids only (never outputs), and regeneration keeps the outputs
of every code cell whose source is unchanged.
"""
import argparse
import ast
import re
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnc import manifest  # noqa: E402

OUT = REPO / 'notebooks' / 'figures'
NARRATIVE = OUT / '_narrative'
KERNEL = dict(name='pnc', display_name='Python 3 (pnc)', language='python')
HELPER_MODULES = {'fig1_grid', 'fig1_cloud', 'fig2_data', 'fig4_panels', 'fig5_adv', 'fig5_grad', 'fig6_data'}


# ---------------------------------------------------------------- inputs
def captions():
    text = (REPO / 'figures' / 'CAPTIONS.md').read_text()
    out = {}
    for block in re.split(r'^### ', text, flags=re.M)[1:]:
        head, _, body = block.partition('\n')
        out[re.search(r'`([^`]+)`', head).group(1)] = (head.strip(), body.strip())
    return out


def narrative(slug):
    """Optional hand-written text per section, keyed by the '## ...' headings of the file."""
    p = NARRATIVE / f'{slug}.md'
    if not p.exists():
        return {}
    out, key = {}, None
    for line in p.read_text().splitlines():
        m = re.match(r'^## (.+?)\s*$', line)
        if m:
            key = m.group(1).strip(); out[key] = []
        elif key is not None:
            out[key].append(line)
    return {k: '\n'.join(v).strip() for k, v in out.items() if '\n'.join(v).strip()}


def notebook_spec():
    """[(notebook stem, output stem, script path, main kwargs)] in manuscript order."""
    spec = []
    for i, name in manifest.MAIN_FIGURES.items():
        spec.append((f'{i:02d}_{name}', name, REPO / 'figures' / 'main' / f'fig{i}_{name}.py', {}))
    for e in manifest.all_entries():
        script = REPO / 'figures' / 'supplementary' / f"sup_{e['slug']}.py"
        if e['span']:
            for mk in manifest.SPAN_MONKEYS:
                stem = manifest.output_name(e['slug'], mk)
                spec.append((stem, stem, script, {'monkey': mk}))
        else:
            stem = manifest.output_name(e['slug'])
            spec.append((stem, stem, script, {}))
    return spec


# ---------------------------------------------------------------- slicing the script
def _segment(lines, start, end):
    """Source lines start..end (1-based, inclusive), stripped of trailing blank lines."""
    seg = ''.join(lines[start - 1:end])
    return seg.rstrip('\n')


def _docstring_first_paragraph(node):
    doc = ast.get_docstring(node)
    if not doc:
        return ''
    return re.sub(r'\s+', ' ', doc.split('\n\n')[0]).strip()


def _header_to_markdown(doc):
    """Render the script's module docstring as prose: paragraphs stay paragraphs, the panel
    inventory lines ('  a  ...') become a bullet list."""
    out, bullets, para = [], [], []

    def flush_para():
        if para:
            out.append(' '.join(s.strip() for s in para)); para.clear()

    def flush_bullets():
        if bullets:
            out.append('\n'.join(bullets)); bullets.clear()
    for line in doc.splitlines():
        m = re.match(r'^\s{1,3}([a-z])\s{2,}(.*)$', line)
        if m:
            flush_para(); bullets.append(f'- **({m.group(1)})** {m.group(2).strip()}')
        elif bullets and re.match(r'^\s{4,}\S', line):
            bullets[-1] += ' ' + line.strip()
        elif not line.strip():
            flush_para(); flush_bullets()
        elif re.match(r'^\s{2,}[*-]\s', line) or re.match(r'^\s{2,}\S', line) and para == [] and bullets == [] and out and out[-1].endswith(':'):
            flush_para(); out.append(line.strip())
        else:
            flush_bullets(); para.append(line)
    flush_para(); flush_bullets()
    return '\n\n'.join(out)


def _split_main_body(lines, fn, kwargs):
    """The body of main() as a list of (title, code) chunks at top-level blank lines, dedented,
    with `return X` rewritten to `png = X` so the assembly runs at notebook top level."""
    body_start = fn.body[0].lineno
    # include the docstring only as prose, not code
    if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], 'value', None), ast.Constant) and isinstance(fn.body[0].value.value, str):
        body_start = fn.body[1].lineno if len(fn.body) > 1 else fn.end_lineno + 1
    nested = set()
    for n in fn.body:
        if isinstance(n, ast.FunctionDef):
            nested |= {x.lineno for x in ast.walk(n) if isinstance(x, ast.Return)}
    returns = {x.lineno for x in ast.walk(fn) if isinstance(x, ast.Return) and x.lineno not in nested}
    raw = lines[body_start - 1:fn.end_lineno]
    indent = len(raw[0]) - len(raw[0].lstrip())
    src = []
    for i, line in enumerate(raw, start=body_start):
        text = line[indent:] if line.strip() else '\n'
        if i in returns:
            m = re.match(r'^(\s*)return\b\s*(.*)$', text.rstrip('\n'))
            if m:
                expr = m.group(2) or 'None'
                note = '' if i == fn.body[-1].lineno else '   # (the script returns here in this mode)'
                text = f'{m.group(1)}png = {expr}{note}\n'
        src.append(text)
    # chunk at blank lines between top-level statements of the body
    tops = sorted({n.lineno - body_start for n in fn.body if n.lineno >= body_start})
    breaks = set()
    for k in tops:
        j = k - 1
        while j >= 0 and not src[j].strip():
            breaks.add(k); j -= 1
    chunks, cur = [], []
    for k, text in enumerate(src):
        if k in breaks and cur and any(t.strip() for t in cur):
            chunks.append(''.join(cur).strip('\n')); cur = []
        cur.append(text)
    if cur and any(t.strip() for t in cur):
        chunks.append(''.join(cur).strip('\n'))
    titled = []
    for c in chunks:
        first = c.splitlines()[0].strip()
        m = re.match(r'^#\s*[-=]*\s*(.*?)\s*[-=]*\s*$', first)
        titled.append((m.group(1).strip() if m and m.group(1).strip() else '', c))
    return titled


def _param_cell(fn, out_dir_expr, kwargs):
    """The assembly's inputs: out_dir plus every keyword parameter of main() at its default,
    overridden by the notebook's own kwargs (the span figures fix their monkey)."""
    lines = ['# Inputs of the assembly below (these are the parameters of main() in the script).',
             f'out_dir = {out_dir_expr}']
    args = fn.args
    defaults = dict(zip([a.arg for a in args.args][len(args.args) - len(args.defaults):], args.defaults))
    for a in args.args[1:]:
        val = kwargs[a.arg] if a.arg in kwargs else (ast.unparse(defaults[a.arg]) if a.arg in defaults else 'None')
        lines.append(f'{a.arg} = {val!r}' if a.arg in kwargs else f'{a.arg} = {val}')
    if len(args.args) > 1:
        lines.append('# Change any of them to render one of the variants described in the script header.')
    return '\n'.join(lines)


# ---------------------------------------------------------------- building a notebook
def build(nb_stem, stem, script, kwargs, cap):
    src = script.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    slug = script.stem
    nar = narrative(slug)
    head, body = cap
    rel = script.relative_to(REPO)
    cells, k = [], 0

    def cid():
        nonlocal k; k += 1; return f'{stem}-{k:02d}'

    # -- introduction
    doc = ast.get_docstring(tree) or ''
    intro = (f'# {head}\n\n'
             f'This notebook rebuilds the figure step by step from its script, `{rel}`. Every code cell '
             f'below is a verbatim slice of that file, in the order the file defines it: first the imports '
             f'and configuration, then each helper function with a short explanation, then the body of '
             f'`main()` laid out as the assembly of the panels, and finally the rendered figure with its '
             f'caption. Run the cells top to bottom; in the reference environment the PNG written at the end '
             f'is identical to the submitted manuscript file (see docs/REPRODUCIBILITY.md).\n\n')
    if nar.get('intro'):
        intro += nar['intro'] + '\n\n'
    if doc:
        intro += '## What the script says about itself\n\n' + _header_to_markdown(doc)
    cells.append(new_markdown_cell(intro.rstrip(), id=cid()))

    # -- environment (ours, not the script's)
    cells.append(new_markdown_cell(
        '## 1. Environment\n\n'
        'A few lines that are not in the script: they make the notebook render exactly like the '
        'command-line harness does (single-threaded BLAS, the Agg backend with matplotlib defaults, the '
        'repository on the import path) and check that the preprocessed data are present.', id=cid()))
    cells.append(new_code_cell(
        'import os, sys\n'
        'from pathlib import Path\n'
        '\n'
        '# Keep BLAS single-threaded: multithreaded OpenMP together with torch can kill the kernel on macOS.\n'
        "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):\n"
        "    os.environ.setdefault(_v, '1')\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        '\n'
        "# Render with the Agg backend and matplotlib's defaults, exactly like the figure scripts;\n"
        "# Jupyter's inline backend would otherwise change dpi and bounding boxes.\n"
        "os.environ['MPLBACKEND'] = 'Agg'\n"
        "import matplotlib; matplotlib.use('Agg'); matplotlib.rcdefaults()\n"
        '\n'
        "# Work from the repository root whether the notebook is opened there or in notebooks/figures/.\n"
        "REPO = Path.cwd() if (Path.cwd() / 'pnc').exists() else Path.cwd().parents[1]\n"
        'sys.path.insert(0, str(REPO))\n'
        'from pnc import paths\n'
        '\n'
        '# The figures read the preprocessed caches only; this raises a clear message if they are missing.\n'
        "paths.require(paths.preprocessed_data() / 'brain_red.pkl')   # python data/download_data.py --tier preprocessed\n"
        "OUT_DIR = paths.output_dir() / 'figures'\n"
        'OUT_DIR.mkdir(parents=True, exist_ok=True)', id=cid()))

    # -- walk the script
    prev_end = 0
    section = 2
    setup_run, setup_done, fn_intro_done = [], False, False
    main_fn = None

    def flush_setup():
        nonlocal setup_run, setup_done, section
        if not setup_run:
            return
        code = '\n\n'.join(setup_run).strip('\n')
        if not setup_done:
            text = (f'## {section}. Imports, style and configuration\n\n'
                    'The script starts by importing the study configuration and figure style from `pnc` '
                    '(colours, model names, the publication rcParams) and the cache loader, then fixes the '
                    'constants that the panels share: which sites and models are shown, layout sizes, colour '
                    'choices, and where the inputs live. `apply_figure_style()` is what makes every figure in '
                    'the paper look the same.')
            if nar.get('setup'):
                text += '\n\n' + nar['setup']
            section += 1; setup_done = True
        else:
            text = 'More configuration used by the functions below.'
        cells.append(new_markdown_cell(text, id=cid()))
        cells.append(new_code_cell(code, id=cid()))
        setup_run = []

    for node in tree.body:
        start = prev_end + 1
        if isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant) and isinstance(node.value.value, str) and prev_end == 0:
            prev_end = node.end_lineno; continue                      # module docstring: already used
        if isinstance(node, ast.If) and 'name' in ast.unparse(node.test):
            prev_end = node.end_lineno; continue                      # the CLI entry point is not part of the walkthrough
        seg = _segment(lines, start, node.end_lineno)
        # drop leading blank lines but keep leading comments
        seg = re.sub(r'^\n+', '', seg)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == 'main':
            main_fn = node; prev_end = node.end_lineno; continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            flush_setup()
            if not fn_intro_done:
                text = (f'## {section}. The building blocks\n\n'
                        'Each function below is one component of the figure: loading and shaping a cache, '
                        'computing a statistic, or drawing one panel into an axes it is handed. They are '
                        'defined here exactly as in the script and used by the assembly in the last section; '
                        'read them in order, the assembly calls them in roughly the same order.')
                if nar.get('blocks'):
                    text += '\n\n' + nar['blocks']
                cells.append(new_markdown_cell(text, id=cid())); section += 1; fn_intro_done = True
            argspec = ast.unparse(node.args) if isinstance(node, ast.FunctionDef) else ''
            desc = _docstring_first_paragraph(node)
            lead = re.findall(r'^#\s?(.*)$', seg.split(f'def {node.name}')[0] if isinstance(node, ast.FunctionDef) else '', flags=re.M)
            if not desc and lead:
                desc = ' '.join(l.strip() for l in lead if l.strip() and not set(l.strip()) <= set('-=#'))
            text = f'### `{node.name}({argspec})`' if isinstance(node, ast.FunctionDef) else f'### `class {node.name}`'
            if desc:
                text += '\n\n' + desc
            if nar.get(f'fn:{node.name}'):
                text += '\n\n' + nar[f'fn:{node.name}']
            cells.append(new_markdown_cell(text, id=cid()))
            cells.append(new_code_cell(seg, id=cid()))
        else:
            setup_run.append(seg)
        prev_end = node.end_lineno
    flush_setup()

    # -- assembly
    if main_fn is None:
        raise ValueError(f'{script}: no main()')
    mdoc = _docstring_first_paragraph(main_fn)
    text = (f'## {section}. Assembling the figure\n\n'
            'This is the body of `main()` from the script, laid out step by step: it builds the figure '
            'canvas, calls the building blocks to fill each panel, and saves the result with `save_fig`, '
            'which trims the white border so the file matches the submitted figure.')
    if mdoc:
        text += '\n\n' + mdoc
    if nar.get('assemble'):
        text += '\n\n' + nar['assemble']
    cells.append(new_markdown_cell(text, id=cid()))
    cells.append(new_code_cell(_param_cell(main_fn, 'str(OUT_DIR)', kwargs), id=cid()))
    for n, (title, code) in enumerate(_split_main_body(lines, main_fn, kwargs), start=1):
        step = f'**Step {n}**' + (f': {title}' if title else '')
        if nar.get(f'step:{n}'):
            step += '\n\n' + nar[f'step:{n}']
        cells.append(new_markdown_cell(step, id=cid()))
        cells.append(new_code_cell(code, id=cid()))

    # -- result and caption
    cells.append(new_markdown_cell(
        f'## {section + 1}. The figure\n\n'
        f'`png` now points at the rendered file. A downscaled preview follows (the full-resolution PNG is '
        f'the file itself, `{stem}.png` in the output directory).' + (('\n\n' + nar['closing']) if nar.get('closing') else ''),
        id=cid()))
    cells.append(new_code_cell(
        'import io\n'
        'from IPython.display import Image, display\n'
        'from PIL import Image as PILImage\n'
        '\n'
        '# A JPEG preview keeps the notebook small; open the PNG for full detail.\n'
        'print(png)\n'
        'im = PILImage.open(png).convert("RGB")\n'
        'im.thumbnail((1200, 1200))\n'
        'buf = io.BytesIO(); im.save(buf, "JPEG", quality=85, optimize=True)\n'
        'display(Image(data=buf.getvalue(), format="jpeg", width=900))', id=f'{stem}-show'))
    label = head.split(':')[0]
    # the caption body opens with its own bold title, so fold the figure number into that bold span
    cells.append(new_markdown_cell(f'**{label}. ' + body[2:] if body.startswith('**') else f'**{label}.** {body}', id=cid()))
    return new_notebook(cells=cells, metadata={'kernelspec': KERNEL, 'language_info': {'name': 'python'}})


# ---------------------------------------------------------------- driver
def render_all():
    caps = captions()
    return {OUT / f'{nb_stem}.ipynb': build(nb_stem, stem, script, kwargs, caps[stem])
            for nb_stem, stem, script, kwargs in notebook_spec()}


def skeleton(nb):
    """What --check compares: cell ids, types, sources and the kernelspec (never outputs)."""
    return ([(c.id, c.cell_type, c.source) for c in nb.cells], nb.metadata.get('kernelspec'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--only', help='comma-separated notebook stems')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {w.strip() for w in args.only.split(',')} if args.only else None
    stale, written = [], 0
    for path, gen in render_all().items():
        if wanted and path.stem not in wanted:
            continue
        nbformat.validate(gen)
        if args.check:
            if not path.exists() or skeleton(gen) != skeleton(nbformat.read(path, as_version=4)):
                stale.append(path.name)
            continue
        if path.exists():
            # keep the rendered outputs of every code cell whose id and source are unchanged
            old = {c.id: c for c in nbformat.read(path, as_version=4).cells}
            for c in gen.cells:
                o = old.get(c.id)
                if o is not None and c.cell_type == 'code' and o.cell_type == 'code' and o.source == c.source:
                    c.outputs, c.execution_count = o.outputs, o.execution_count
        nbformat.write(gen, path); written += 1
    if args.check:
        if stale:
            print('stale notebooks (run notebooks/_make_figure_notebooks.py):', stale); return 1
        print('all figure notebooks up to date'); return 0
    print(f'wrote {written} notebooks -> {OUT} (outputs kept where a cell is unchanged)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
