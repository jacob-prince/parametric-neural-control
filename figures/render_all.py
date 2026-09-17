#!/usr/bin/env python3
"""Render every manuscript figure (6 main + 49 supplementary) in isolated subprocesses.

    python figures/render_all.py                      # all 55 -> outputs/figures/
    python figures/render_all.py --only divergence,s17_control_slope_anova
    python figures/render_all.py --check              # also compare against tests/reference/figure_pixel_hashes.json

Each figure script is run as `python -m figures.<section>.<module> --out DIR [--monkey MK]`
with a minimal, fixed environment (Agg backend, private matplotlib config dir, hash seed 0,
single-threaded BLAS) so that renders do not depend on the caller's shell. Outputs are the
trimmed PNGs under their manuscript names (framework.png ... s49_predicting_summary.png)
plus render_manifest.json (per-figure script, duration, shape, pixel sha256, versions).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnc import manifest, paths  # noqa: E402

REFERENCE = REPO / 'tests' / 'reference' / 'figure_pixel_hashes.json'


def registry():
    """[(output stem, module, extra args)] for all 55 figures, manuscript order."""
    items = [(name, f'figures.main.fig{i}_{name}', []) for i, name in manifest.MAIN_FIGURES.items()]
    for e in manifest.all_entries():
        mod = f"figures.supplementary.sup_{e['slug']}"
        if e['span']:
            items += [(manifest.output_name(e['slug'], mk), mod, ['--monkey', mk]) for mk in manifest.SPAN_MONKEYS]
        else:
            items.append((manifest.output_name(e['slug']), mod, []))
    return items


def render_env(out_dir):
    """Environment for a figure subprocess: built from scratch, never inherited."""
    keep = ('PATH', 'HOME', 'TMPDIR', 'USER', 'LANG', 'LC_ALL', 'PNC_SOURCE_DATA',
            'PNC_PREPROCESSED_DATA', 'PNC_OUTPUT', 'PNC_HN_LIGHT_TTF')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    mpl = Path(out_dir) / '.mplconfig'
    mpl.mkdir(parents=True, exist_ok=True)
    env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(mpl), PYTHONHASHSEED='0',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    return env


def pixel_hash(png):
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(png).convert('RGBA'))
    return hashlib.sha256(arr.tobytes()).hexdigest(), list(arr.shape)


def versions():
    import matplotlib, numpy, PIL
    from matplotlib import ft2font, font_manager
    return dict(python=platform.python_version(), matplotlib=matplotlib.__version__,
                numpy=numpy.__version__, pillow=PIL.__version__,
                freetype=ft2font.__freetype_version__, platform=platform.platform(),
                machine=platform.machine(),
                helvetica_neue=font_manager.findfont('Helvetica Neue', fallback_to_default=False)
                if _has_font('Helvetica Neue') else None)


def _has_font(name):
    from matplotlib import font_manager
    try:
        font_manager.findfont(name, fallback_to_default=False)
        return True
    except ValueError:
        return False


def render(stem, module, extra, out_dir, env, timeout=1800):
    t0 = time.time()
    cmd = [sys.executable, '-m', module, '--out', str(out_dir), *extra]
    r = subprocess.run(cmd, cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    rec = dict(name=stem, module=module, args=extra, returncode=r.returncode,
               duration_seconds=round(time.time() - t0, 1))
    log_dir = Path(out_dir) / 'logs'
    log_dir.mkdir(exist_ok=True)
    (log_dir / f'{stem}.log').write_text(r.stdout or '')
    png = Path(out_dir) / f'{stem}.png'
    if r.returncode == 0 and png.exists():
        rec['sha256'], rec['shape'] = pixel_hash(png)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='comma-separated output stems (e.g. divergence,s17_control_slope_anova) or slugs')
    ap.add_argument('--out', type=Path, default=None, help='output directory (default: <PNC_OUTPUT>/figures)')
    ap.add_argument('--check', action='store_true', help='compare pixel hashes with tests/reference/figure_pixel_hashes.json')
    ap.add_argument('--keep-going', action='store_true', help='continue after a failure')
    args = ap.parse_args(argv)

    out_dir = (args.out or paths.output_dir() / 'figures').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = registry()
    if args.only:
        wanted = {w.strip() for w in args.only.split(',') if w.strip()}
        items = [it for it in items if it[0] in wanted or it[1].rsplit('_', 1)[-1] in wanted
                 or it[1].split('.')[-1].removeprefix('sup_') in wanted]
        if not items:
            raise SystemExit(f'nothing matched --only {args.only}')
    reference = json.loads(REFERENCE.read_text())['figures'] if args.check else None
    env = render_env(out_dir)
    records, failed = [], []
    for stem, module, extra in items:
        print(f'{stem} ...', end=' ', flush=True)
        rec = render(stem, module, extra, out_dir, env)
        if reference is not None and 'sha256' in rec:
            rec['matches_reference'] = rec['sha256'] == reference[stem]['sha256']
        records.append(rec)
        status = 'ok' if rec['returncode'] == 0 else 'FAILED'
        if reference is not None and rec['returncode'] == 0:
            status += ' / pixel-exact' if rec.get('matches_reference') else ' / DIFFERS'
        print(f"{status} ({rec['duration_seconds']}s)", flush=True)
        if rec['returncode'] != 0 or (reference is not None and not rec.get('matches_reference')):
            failed.append(stem)
            if rec['returncode'] != 0 and not args.keep_going:
                break
    summary = dict(output_dir=str(out_dir), versions=versions(), figures=records,
                   n_ok=sum(r['returncode'] == 0 for r in records), n_failed=len(failed), failed=failed)
    (out_dir / 'render_manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f"{summary['n_ok']}/{len(records)} rendered" + (f", {len(failed)} problem(s): {failed}" if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
