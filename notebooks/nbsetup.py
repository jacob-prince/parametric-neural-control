"""Shared setup for every notebook in this repository.

    from nbsetup import setup, show_figure          # from notebooks/figures/ or notebooks/demos/
    out_dir = setup()
    png = main(str(out_dir))                        # a figure script's main(): writes the trimmed PNG, returns its path
    show_figure(png)

A figure script's main(out_dir, **variant) writes <out_dir>/<manuscript name>.png, border-trimmed
so it matches the submitted file, and returns the path; keyword arguments select the variants
documented in the script's header (the notebooks render the defaults, which are the paper's figures).

setup() does what the command-line harness does before a figure script runs: pins BLAS to one
thread (multithreaded OpenMP together with torch can kill the Jupyter kernel on macOS), selects
the Agg backend with matplotlib's defaults (the inline backend would change dpi and bounding
boxes), puts the repository on the import path, checks that the requested data tier is present,
and returns the output directory. show_figure() displays a downscaled preview of a rendered PNG
followed by its manuscript caption.
"""
import io
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def setup(needs='preprocessed', single_thread=True):
    """Prepare the kernel and return the figure output directory (a Path).

    needs: 'preprocessed' (the caches every figure reads), 'source' (raw data, for the demos),
    or None to skip the data check.
    """
    if single_thread:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcdefaults()
    # the repository root (packages) and the demos folder (its _demo_utils helper) on the import path
    for extra in (REPO, HERE / 'demos'):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    from pnc import paths
    if needs == 'preprocessed':
        paths.require(paths.preprocessed_data() / 'brain_red.pkl', tier='preprocessed')
    elif needs == 'source':
        paths.require(paths.source_data() / 'stimuli_encoding', tier='source')
        os.environ.setdefault('PNC_MODEL_BACKBONES', str(paths.source_data() / 'model_backbones'))
    out_dir = paths.output_dir() / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def caption(stem):
    """The manuscript caption for an output stem (e.g. 'divergence', 's17_control_slope_anova')."""
    text = (REPO / 'figures' / 'CAPTIONS.md').read_text()
    for block in re.split(r'^### ', text, flags=re.M)[1:]:
        head, _, body = block.partition('\n')
        if re.search(r'`([^`]+)`', head).group(1) == stem:
            label = head.split(':')[0]
            body = body.strip()
            return f'**{label}. ' + body[2:] if body.startswith('**') else f'**{label}.** {body}'
    raise KeyError(stem)


def show_figure(png, width=900, max_px=1200):
    """Display a downscaled JPEG preview of `png` with its caption underneath.

    The full-resolution file (often tens of MB) stays on disk; the preview keeps the notebook small.
    """
    from IPython.display import Image, Markdown, display
    from PIL import Image as PILImage
    im = PILImage.open(png).convert('RGB')
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85, optimize=True)
    display(Image(data=buf.getvalue(), format='jpeg', width=width))
    display(Markdown(caption(Path(png).stem)))
