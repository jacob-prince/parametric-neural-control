"""
Shared constants, figure style and raw-data loaders for every figure script.

Ported verbatim from the take9 analysis pipeline (utils.py); the only changes are that
paths resolve through pnc.paths, save_fig writes to an explicit output directory and
trims the saved PNG (so it matches the manuscript file pixel for pixel), and the
face-masking hook used for talk renders is gone.
"""

import os
import glob as globmod
import pickle
import numpy as np
import h5py
from scipy import stats

from pnc import paths
from pnc.trim import trim

# =============================================================================
# PATHS
# =============================================================================

DATA_ROOT = str(paths.source_data())          # was /Volumes/CORSAIR/AccentuateVVS/data

BASE_PATH = os.path.join(DATA_ROOT, "image_pca_projections")
STIMULI_PATH = os.path.join(DATA_ROOT, "stimuli_encoding")
STIMULI_CONTROL_PATH = os.path.join(DATA_ROOT, "stimuli_control")
ENCODING_HDF5_DIR = os.path.join(DATA_ROOT, "brain_data_encoding")
CONTROL_HDF5_PATH = os.path.join(
    DATA_ROOT, "brain_data_control",
    "vvs-accentuate_controlsessions_5monkeys_250504-250512.h5")

# =============================================================================
# MONKEY / SESSION CONFIGURATION
# =============================================================================

MONKEY_DIRS = {
    'red': 'red_20250428-20250430',
    'paul': 'paul_20250428-20250430',
    'venus': 'venus_250426-250429',
    'leap': 'leap_250426-250501',
    'three0': 'three0_250426-250501',
}

ACCENT_DATE_PREFIXES = {
    'red': '02-05-2025',
    'paul': '03-05-2025',
    'venus': '05-05-2025',
    'leap': '06-05-2025',
    'three0': '06-05-2025',
}

ENCODING_HDF5_FILES = {
    'red': 'red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5',
    'paul': 'paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5',
    'venus': 'venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5',
    'leap': 'leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5',
    'three0': 'three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5',
}

MONKEY_RESPONSE_WINDOWS = {
    'red': 'rw100-400',
    'paul': 'rw100-400',
    'venus': 'rw80-250',
    'leap': 'rw80-250',
    'three0': 'rw80-250',
}

MONKEY_UNITS = {
    'red': [0, 2, 9, 15, 19],
    'paul': [0, 8, 24, 40, 47],
    'venus': [9, 79, 151, 331, 355],
    'leap': [81, 282, 286, 306, 342],
    'three0': [56, 74, 120, 168, 204],
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

ALL_MODELS = {
    'AlexNet_training_seed_01': 'AlexNet',
    'clipag_vitb32': 'CLIPAG ViT-B/32',
    'dinov2_vitb14_reg': 'DINOv2-ViT-B/14',
    'radio_v2.5-b': 'RADIO v2.5-B',
    'regnety_640': 'RegNetY-640',
    'resnet50': 'ResNet50',
    'resnet50_clip': 'ResNet50-CLIP',
    'resnet50_dino': 'ResNet50-DINO',
    'resnet50_robust': 'Robust ResNet50',
    'siglip2_vitb16': 'SigLIP2 ViT-B/16',
}

MODEL_SHORT_NAMES = {
    'AlexNet_training_seed_01': 'AlexNet',
    'clipag_vitb32': 'CLIPAG',
    'dinov2_vitb14_reg': 'DINOv2',
    'radio_v2.5-b': 'RADIO',
    'regnety_640': 'RegNetY',
    'resnet50': 'ResNet50',
    'resnet50_clip': 'RN50-CLIP',
    'resnet50_dino': 'RN50-DINO',
    'resnet50_robust': 'RN50-Robust',
    'siglip2_vitb16': 'SigLIP2',
}

# --- Model groups ---
ROBUST_MODELS = ['resnet50_robust', 'clipag_vitb32']
NON_ROBUST_CNNS = ['AlexNet_training_seed_01', 'resnet50', 'regnety_640',
                    'resnet50_clip', 'resnet50_dino']
NON_ROBUST_TRANSFORMERS = ['dinov2_vitb14_reg', 'radio_v2.5-b', 'siglip2_vitb16']

MODEL_PAIRS = {
    'clip_vs_robust': {
        'models': {'resnet50_clip': 'ResNet50-CLIP', 'resnet50_robust': 'Robust ResNet50'},
        'tag': 'clip_vs_robust',
    },
    'all_models': {
        'models': ALL_MODELS,
        'tag': 'all_models',
    },
}

# =============================================================================
# COLOR SCHEME
# =============================================================================

COLOR_ROBUST = '#E63946'
COLOR_CNN = '#457B9D'
COLOR_TRANSFORMER = '#2A9D8F'


def get_model_group(model):
    """Return group name: 'Robust', 'Non-robust CNN', 'Non-robust ViT'."""
    if model in ROBUST_MODELS:
        return 'Robust'
    elif model in NON_ROBUST_CNNS:
        return 'Non-robust CNN'
    else:
        return 'Non-robust ViT'


def get_model_group_color(model):
    """Return the GROUP hex color for a model (coarse: 3-way palette)."""
    if model in ROBUST_MODELS:
        return COLOR_ROBUST
    elif model in NON_ROBUST_CNNS:
        return COLOR_CNN
    return COLOR_TRANSFORMER


GROUP_ORDER = ['Robust', 'Non-robust CNN', 'Non-robust ViT']
GROUP_COLORS = {
    'Robust': COLOR_ROBUST,
    'Non-robust CNN': COLOR_CNN,
    'Non-robust ViT': COLOR_TRANSFORMER,
}

# --- Per-model ramp ---
# Ten distinct colors, one per model, ordered by grand control r (low → high).
# This is the palette used throughout the figures (cross-model heatmap,
# partition comparison, grand composite, etc.) and must remain stable so that
# a given model keeps its identity across every figure.
MODEL_COLORS = {
    'AlexNet_training_seed_01': '#1B0A2E',  # deep indigo
    'siglip2_vitb16':           '#4A0E6B',  # violet
    'resnet50_clip':            '#7B2D8E',  # purple
    'regnety_640':              '#C2185B',  # magenta
    'dinov2_vitb14_reg':        '#E53935',  # red
    'radio_v2.5-b':             '#F4511E',  # red-orange
    'resnet50':                 '#FB8C00',  # orange
    'resnet50_dino':            '#D4A017',  # gold
    'resnet50_robust':          '#8CCF50',  # light green
    'clipag_vitb32':            '#4DB86A',  # green
}

# Display order for legends / slopegraphs / heatmap axes (worst → best on
# grand control r, aIT Monkey R).
MODEL_ORDER = [
    'AlexNet_training_seed_01',
    'siglip2_vitb16',
    'resnet50_clip',
    'regnety_640',
    'dinov2_vitb14_reg',
    'radio_v2.5-b',
    'resnet50',
    'resnet50_dino',
    'resnet50_robust',
    'clipag_vitb32',
]


def get_model_color(model):
    """Return the per-model hex color (fine: 10-way palette).

    Falls back to the coarse group color if model is unknown.
    """
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return get_model_group_color(model)

MONKEY_COLORS = {
    'red': '#2F4858',    # aIT - dark slate
    'paul': '#2E86DE',   # cIT - dodgerblue
    'venus': '#E36BA0',  # V3/V4 - pink
    'leap': '#8896A6',   # STS·L - blue-grey
    'three0': '#B87F33', # STS·T - ochre
}

# =============================================================================
# FIGURE STYLE
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt

# -- Font --------------------------------------------------------------------
FONT_FAMILY = 'Helvetica Neue'     # NRJ: sans-serif, preferably Helvetica/Arial
FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 10
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 7
FONT_SIZE_SUPTITLE = 14

# -- Axes / spines -----------------------------------------------------------
SPINE_LINEWIDTH = 0.8
TICK_LENGTH_MAJOR = 4
TICK_LENGTH_MINOR = 2
TICK_WIDTH = 0.8
TICK_DIRECTION = 'out'
TICK_PAD = 3

# -- Lines / markers ---------------------------------------------------------
LINE_WIDTH = 1.5
MARKER_SIZE = 4
ERRORBAR_CAPSIZE = 2
ERRORBAR_LINEWIDTH = 1.0

# -- Figure ------------------------------------------------------------------
DPI = 300
FIG_FACECOLOR = 'white'

# -- Colormaps ---------------------------------------------------------------
CMAP_SEQUENTIAL = 'viridis'
CMAP_DIVERGING = 'RdBu_r'

# -- Standard figure widths (inches, for Nature-style) -----------------------
WIDTH_SINGLE_COL = 3.5
WIDTH_ONE_AND_HALF_COL = 5.5
WIDTH_DOUBLE_COL = 7.2


def apply_figure_style():
    """Apply publication figure style globally via rcParams.

    Call once at the top of every figure script:
        from pnc.utils import apply_figure_style
        apply_figure_style()
    """
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT_FAMILY, 'Helvetica', 'Arial', 'Avenir'],
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_SUPTITLE,
        'mathtext.default': 'regular',

        # Axes / spines
        'axes.linewidth': SPINE_LINEWIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 4,
        'axes.titlepad': 6,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',

        # Ticks
        'xtick.major.size': TICK_LENGTH_MAJOR,
        'xtick.minor.size': TICK_LENGTH_MINOR,
        'xtick.major.width': TICK_WIDTH,
        'xtick.minor.width': TICK_WIDTH * 0.6,
        'xtick.direction': TICK_DIRECTION,
        'xtick.major.pad': TICK_PAD,
        'ytick.major.size': TICK_LENGTH_MAJOR,
        'ytick.minor.size': TICK_LENGTH_MINOR,
        'ytick.major.width': TICK_WIDTH,
        'ytick.minor.width': TICK_WIDTH * 0.6,
        'ytick.direction': TICK_DIRECTION,
        'ytick.major.pad': TICK_PAD,

        # Lines / markers
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,

        # Figure
        'figure.dpi': 100,
        'figure.facecolor': FIG_FACECOLOR,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': FIG_FACECOLOR,
        'savefig.pad_inches': 0.05,
        'savefig.transparent': False,

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.0,

        # Grid (off by default)
        'axes.grid': False,

        # PDF / SVG text handling
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def light_font_properties(size=None):
    """FontProperties for the Helvetica Neue *Light* face used by a few labels.

    Apple's fonts cannot be redistributed, so the .ttf is not part of the repo. Resolution
    order: $PNC_HN_LIGHT_TTF -> figures/assets/private/HelveticaNeue-Light.ttf (gitignored;
    extract it from /System/Library/Fonts/HelveticaNeue.ttc on macOS for pixel-exact
    output) -> the installed 'Helvetica Neue' family at weight 'light' (tolerance mode).
    """
    from matplotlib.font_manager import FontProperties
    candidates = [os.environ.get('PNC_HN_LIGHT_TTF'),
                  str(paths.ASSETS / 'private' / 'HelveticaNeue-Light.ttf')]
    for c in candidates:
        if c and os.path.exists(c):
            return FontProperties(fname=c, size=size)
    return FontProperties(family=FONT_FAMILY, weight='light', size=size)


# -- Final-artwork sizing (NRJ) ----------------------------------------------
MM = 1.0 / 25.4                      # millimetres -> inches (author figures at final print size)
WIDTH_1COL_MM, WIDTH_2COL_MM = 88, 180        # Nature research-content column widths


def save_fig(fig, path_no_ext, dpi=DPI, bbox_inches='tight', do_trim=True, pdf=False):
    """Save `fig` as <path_no_ext>.png and return the path.

    The PNG is border-trimmed in place (pnc.trim.trim) so that the file on disk is
    pixel-identical to the manuscript copy; pass do_trim=False, or set PNC_NO_TRIM=1 in the
    environment, for the raw matplotlib canvas. pdf=True also writes <path_no_ext>.pdf (untrimmed).
    """
    output_base = os.path.abspath(os.fspath(path_no_ext))
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png = output_base + '.png'
    fig.savefig(png, dpi=dpi, facecolor=FIG_FACECOLOR, bbox_inches=bbox_inches)
    if pdf:
        fig.savefig(output_base + '.pdf', facecolor=FIG_FACECOLOR, bbox_inches=bbox_inches)
    if do_trim and os.environ.get('PNC_NO_TRIM', '').strip().lower() not in {'1', 'true', 'yes'}:
        trim(png)
    return png

# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def to_numpy(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def z_to_spks(z_values, mu_avg, sigma_avg, unit=None):
    """Convert z-scored values to spikes/s, clamped at 0."""
    if unit is not None:
        mu = mu_avg[unit]
        sigma = sigma_avg[unit]
    else:
        mu = mu_avg
        sigma = sigma_avg
    return np.maximum(0.0, z_values * sigma + mu)


# =============================================================================
# FIRING FLOOR
# =============================================================================
# A model's predicted accentuation response is in z units of the encoding
# normalization; predicted firing in spk/s = z*sigma + mu, so any prediction below
#
#     z_floor = -mu / sigma
#
# implies a physically impossible negative firing rate. Every predicted control score
# is raised to this floor before it enters a control metric, so predictions never
# assert sub-zero firing. mu/sigma and the resulting floor are computed once in
# preproc and read here from the common processed cache, so the clamp stays
# consistent with the standardization pipeline. (Predictions are stored RAW in the
# cache; the clamp is applied last, at the brain comparison -- inside
# loader.control_cloud / control_seed_rs, or via these accessors.)
#
# pnc.preproc.loader is imported lazily, per call: scripts that only want colors,
# style or the plain HDF5 loaders above must not pay for the brain cache.

def firing_z_floor(monkey, unit):
    """z at which predicted firing = 0 spk/s (-mu/sigma), from the common cache."""
    from pnc.preproc import loader as L
    return L.firing_floor(monkey, unit)


def clamp_to_floor(scores_z, monkey, unit):
    """Raise any predicted response (z) below the 0-spk/s floor up to the floor."""
    return np.maximum(np.asarray(scores_z, float), firing_z_floor(monkey, unit))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_encoding_pickle(monkey, unit, model):
    """Load posthoc PCA prediction pickle for encoding images."""
    monkey_dir = MONKEY_DIRS[monkey]
    pkl_name = (f"posthoc_prediction_NSDencimg_PCA_pop_unit_"
                f"{monkey_dir}_unit{unit}_{model}.pkl")
    pkl_path = os.path.join(BASE_PATH, monkey_dir,
                            "posthoc_model_predict_PCA_popul_unit", pkl_name)
    return load_pickle(pkl_path)


def load_accentuated_pickle(monkey, unit, model):
    """Load posthoc PCA prediction pickle for accentuated images."""
    monkey_dir = MONKEY_DIRS[monkey]
    pkl_name = (f"posthoc_prediction_PCA_pop_unit_"
                f"{monkey_dir}_unit{unit}_{model}.pkl")
    pkl_path = os.path.join(BASE_PATH, monkey_dir,
                            "posthoc_model_predict_PCA_popul_unit", pkl_name)
    return load_pickle(pkl_path)


def load_encoding_hdf5(monkey):
    """Load encoding session HDF5.

    Returns (stim_names, response_peak, trial_stim_names, trial_response_peak).
    """
    h5_path = os.path.join(ENCODING_HDF5_DIR, ENCODING_HDF5_FILES[monkey])
    with h5py.File(h5_path, 'r') as h5f:
        stim_names = np.array(h5f['repavg']['stimulus_name'], dtype=str)
        response_peak = np.array(h5f['repavg']['response_peak'], dtype=float)
        trial_stim_names = np.array(h5f['trials']['stimulus_name'], dtype=str)
        trial_response_peak = np.array(h5f['trials']['response_peak'], dtype=float)
    return stim_names, response_peak, trial_stim_names, trial_response_peak


def load_control_hdf5(monkey):
    """Load control session HDF5.

    Returns (stim_names, response_peak, trial_stim_names, trial_response_peak).
    """
    with h5py.File(CONTROL_HDF5_PATH, 'r') as h5f:
        # one group per monkey, named <monkey>_<dates>; match on the prefix
        matching_key = None
        for key in h5f.keys():
            if key.startswith(monkey.lower()):
                matching_key = key
                break
        if matching_key is None:
            raise ValueError(f"Monkey {monkey} not found in control HDF5")
        stim_names = np.array(h5f[matching_key]['repavg']['stimulus_name'], dtype=str)
        response_peak = np.array(h5f[matching_key]['repavg']['response_peak'], dtype=float)
        trial_stim_names = np.array(h5f[matching_key]['trials']['stimulus_name'], dtype=str)
        trial_response_peak = np.array(h5f[matching_key]['trials']['response_peak'], dtype=float)
    return stim_names, response_peak, trial_stim_names, trial_response_peak


def load_encoding_mu_sigma(monkey):
    """Load per-session mu/sigma from encoding sessdata PKLs, average across days.

    Returns (mu_avg, sigma_avg) each shape (n_neurons,).
    """
    rw = MONKEY_RESPONSE_WINDOWS[monkey]
    pattern = os.path.join(ENCODING_HDF5_DIR,
                           f"{monkey}_*_hp0_bs0_zs1_{rw}_sessdata.pkl")
    pkls = sorted(globmod.glob(pattern))
    if not pkls:
        raise FileNotFoundError(
            f"No encoding sessdata PKLs for {monkey}: {pattern}")

    all_mu, all_sigma = [], []
    for p in pkls:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        nm = d['neuron_meta']
        all_mu.append(np.array(nm['mu'], dtype=np.float64))
        all_sigma.append(np.array(nm['sigma'], dtype=np.float64))

    mu_avg = np.mean(all_mu, axis=0)
    sigma_avg = np.mean(all_sigma, axis=0)
    return mu_avg, sigma_avg


def build_trial_index(trial_stim_names):
    """Map stimulus name -> list of trial indices."""
    idx = {}
    for i, name in enumerate(trial_stim_names):
        idx.setdefault(name, []).append(i)
    return idx


def compute_sem_for_unit(stim_names, trial_index, trial_responses, unit):
    """Compute SEM across trials for each stimulus."""
    sems = np.zeros(len(stim_names))
    n_reps = np.zeros(len(stim_names), dtype=int)
    for j, name in enumerate(stim_names):
        indices = trial_index.get(name, [])
        n_reps[j] = len(indices)
        # SEM needs at least two trials; single-trial stimuli keep SEM 0
        if len(indices) > 1:
            trial_vals = trial_responses[indices, unit]
            sems[j] = np.std(trial_vals, ddof=1) / np.sqrt(len(indices))
    return sems, n_reps


def parse_stimulus_name(stim_name):
    """Parse accentuated stimulus name into components.

    Returns dict with model, unit, seed, target, score or None if unparseable.
    """
    if 'score' not in stim_name:
        return None
    # layout: <model>_RidgeCV_unit_<u>_img_<seed>_level_<target>_score_<score>.png
    try:
        model = stim_name.split('_RidgeCV')[0]
        suffix = stim_name.split('_RidgeCV')[1]
        parts = suffix.split('_')
        unit_id = int(parts[2])
        seed = int(parts[4])
        target = float(parts[6])
        score = float(parts[8].replace('.png', ''))
        return {'model': model, 'unit': unit_id, 'seed': seed,
                'target': target, 'score': score}
    except (IndexError, ValueError):
        return None
