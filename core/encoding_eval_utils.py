"""
encoding_eval_utils.py
======================
Utilities for visualising encoding model quality:
  measured neural responses (resp_mat from the encoding session HDF5)
  vs model-predicted responses (from posthoc prediction pkl),
  with train / test split shown as separate colours.

Key functions
-------------
load_encoding_data(session_id)
    Returns a merged DataFrame (one row per stimulus) containing both
    the actual neural responses and all model predictions, plus split flags.

fig_encoding_scatter(df, unit_id, model_name, ...)
    Scatter: x = predicted response, y = neural response.
    Train and test stimuli drawn with distinct colours.
    Annotates Pearson r and R² for each split.

fig_encoding_scatter_grid(df, unit_ids, model_names, ...)
    Grid version: rows = units, cols = models.
"""

from __future__ import annotations

import os
import sys
import pickle as pkl
from os.path import join
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from core.data_utils import load_from_hdf5, extract_neural_data_dict_2025apr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EPHYS_DIR    = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data"
ENCODING_DIR = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"

_BOOL_MAP = {1: True, 0: False, "1": True, "0": False,
             "True": True, "False": False, True: True, False: False}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_encoding_data(session_id: str) -> pd.DataFrame:
    """
    Load neural responses and model predictions for the encoding session.

    Loads the session HDF5 (for actual neural responses) and the posthoc
    prediction pkl (for predicted responses + train/test flags), then joins
    them on stimulus name.

    Parameters
    ----------
    session_id : str
        E.g. "red_20250428-20250430"

    Returns
    -------
    df : pd.DataFrame
        One row per stimulus.  Columns include:
          stimulus_name, is_train, is_test, is_nsd, image_fps,
          neural_resp_unit_{u}         for u in 0..n_units-1
          pred_resp_{model}_unit_{u}   for all models × units
    """
    # --- neural responses from HDF5 ---
    h5_path = join(EPHYS_DIR,
                   f"{session_id}_vvs-encodingstimuli_z1_rw100-400.h5")
    data     = load_from_hdf5(h5_path)
    data_dict = extract_neural_data_dict_2025apr(data)

    resp_mat     = data_dict["resp_mat"]          # (n_stims, n_units)
    stim_names   = data_dict["stimulus_names"]    # array of str
    n_stims, n_units = resp_mat.shape

    # Decode bytes → str if needed
    stim_names = [s.decode("utf-8") if isinstance(s, bytes) else s
                  for s in stim_names]

    df_neural = pd.DataFrame({"stimulus_name": stim_names})
    for u in range(n_units):
        df_neural[f"neural_resp_unit_{u}"] = resp_mat[:, u]

    print(f"Loaded HDF5 {session_id}: {n_stims} stims, {n_units} units")

    # --- model predictions + split flags from pkl ---
    posthoc_dir = join(ENCODING_DIR, session_id, "posthoc_model_predict")
    pkl_path    = join(posthoc_dir,
                       f"encoding_stim_info_w_pred_resp_{session_id}.pkl")
    df_pred = pd.read_pickle(pkl_path)

    # Normalise boolean split columns
    for col in ["is_train", "is_test", "is_nsd", "is_floc",
                "is_OO", "is_normalizer"]:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].map(_BOOL_MAP).astype(bool)

    print(f"Loaded predictions pkl: {len(df_pred)} rows, "
          f"{len(df_pred.columns)} cols")

    # --- join on stimulus_name ---
    # The pkl uses bare filenames; the HDF5 stimulus_names may include paths.
    # Normalise both to basename for robustness.
    df_pred["_key"]   = df_pred["stimulus_name"].apply(os.path.basename)
    df_neural["_key"] = df_neural["stimulus_name"].apply(os.path.basename)

    df = pd.merge(df_neural, df_pred, on="_key",
                  how="inner", suffixes=("_neural", ""))
    df = df.drop(columns=["_key", "stimulus_name_neural"], errors="ignore")
    df = df.rename(columns={"stimulus_name": "stimulus_name"})

    print(f"Merged: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Single scatter panel (one unit × one model)
# ---------------------------------------------------------------------------
_SPLIT_COLORS = {"train": "#4878CF", "test": "#FF8C00", "other": "#888888"}
_SPLIT_MARKERS = {"train": "o", "test": "^", "other": "s"}


def fig_encoding_scatter(
    df: pd.DataFrame,
    unit_id: int,
    model_name: str,
    ax: plt.Axes | None = None,
    show: bool = True,
    s: float = 30,
    alpha: float = 0.6,
) -> plt.Figure:
    """
    Scatter: x = predicted response, y = actual neural response.
    Train and test stimuli shown with different colours and markers.

    Parameters
    ----------
    df         : merged DataFrame from load_encoding_data()
    unit_id    : which neural unit to plot
    model_name : model key, e.g. "resnet50_robust"
    ax         : existing Axes to draw into (creates new figure if None)
    show       : call plt.show() at the end
    """
    yvar = f"neural_resp_unit_{unit_id}"
    xvar = f"pred_resp_{model_name}_unit_{unit_id}"

    if yvar not in df.columns:
        raise KeyError(f"Column '{yvar}' not in df")
    if xvar not in df.columns:
        raise KeyError(f"Column '{xvar}' not in df")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    else:
        fig = ax.get_figure()

    # --- define splits ---
    splits = {
        "train": df["is_train"].fillna(False),
        "test":  df["is_test"].fillna(False),
    }
    # anything in neither goes to "other"
    in_split = splits["train"] | splits["test"]
    splits["other"] = ~in_split

    # pre-compute global axis limits
    x_all = df[xvar].dropna()
    y_all = df[yvar].dropna()
    pad  = 0.05 * (max(x_all.max(), y_all.max()) -
                   min(x_all.min(), y_all.min()))
    vmin = min(x_all.min(), y_all.min()) - pad
    vmax = max(x_all.max(), y_all.max()) + pad
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # --- plot each split ---
    stat_lines = []
    for split_name, mask in splits.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        x, y = sub[xvar], sub[yvar]
        valid = x.notna() & y.notna()

        ax.scatter(x[valid], y[valid],
                   color=_SPLIT_COLORS[split_name],
                   marker=_SPLIT_MARKERS[split_name],
                   s=s, alpha=alpha, zorder=2,
                   label=f"{split_name} (N={valid.sum()})")

        if valid.sum() > 1:
            corr, _ = pearsonr(x[valid], y[valid])
            r2      = r2_score(y[valid], x[valid])
            stat_lines.append(
                f"{split_name}: r={corr:.2f}, R²={r2:.2f}, N={valid.sum()}")

    # --- y=x reference line ---
    ax.axline((0, 0), slope=1, linestyle="--", color="gray", lw=1, zorder=1)

    # --- stats annotation (top-left) ---
    if stat_lines:
        ax.text(0.04, 0.96, "\n".join(stat_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", fc="white",
                          ec="lightgray", alpha=0.85))

    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax.set_xlabel(f"Predicted response\n({model_name})", fontsize=9)
    ax.set_ylabel(f"Neural response (unit {unit_id})", fontsize=9)
    ax.set_title(f"{model_name} | unit {unit_id}", fontsize=10)

    if standalone:
        fig.tight_layout()
        if show:
            plt.show()
    return fig


# ---------------------------------------------------------------------------
# Grid: rows = units, cols = models
# ---------------------------------------------------------------------------
def fig_encoding_scatter_grid(
    df: pd.DataFrame,
    unit_ids: Sequence[int],
    model_names: Sequence[str],
    ax_size: float = 4.0,
    show: bool = True,
) -> plt.Figure:
    """
    Grid of scatter panels: rows = units, cols = models.

    Parameters
    ----------
    df          : merged DataFrame from load_encoding_data()
    unit_ids    : list of unit indices to show (one row each)
    model_names : list of model keys (one col each)
    ax_size     : size (inches) of each panel
    """
    n_rows = len(unit_ids)
    n_cols = len(model_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(ax_size * n_cols, ax_size * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, unit_id in enumerate(unit_ids):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            try:
                fig_encoding_scatter(df, unit_id, model_name,
                                     ax=ax, show=False)
            except KeyError as e:
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="red")

    fig.suptitle("Encoding model: neural vs predicted response", fontsize=13)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Quick CLI usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from circuit_toolkit.plot_utils import saveallforms

    parser = argparse.ArgumentParser(
        description="Plot encoding model scatter for a session")
    parser.add_argument("--session", default="red_20250428-20250430")
    parser.add_argument("--units",   nargs="+", type=int, default=[0, 2, 9, 15, 19])
    parser.add_argument("--models",  nargs="+",
                        default=["resnet50_robust", "resnet50"])
    parser.add_argument("--figdir",  default=(
        "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/"
        "figures/peer_review_export"))
    args = parser.parse_args()

    os.makedirs(args.figdir, exist_ok=True)

    df = load_encoding_data(args.session)

    # Grid figure
    fig = fig_encoding_scatter_grid(df, args.units, args.models, show=False)
    saveallforms(args.figdir,
                 f"encoding_scatter_grid_{args.session}", figh=fig)
    plt.close(fig)
    print(f"Saved grid → {args.figdir}")

    # Per-unit per-model individual figures
    for unit_id in args.units:
        for model_name in args.models:
            fig = fig_encoding_scatter(df, unit_id, model_name, show=False)
            saveallforms(args.figdir,
                         f"encoding_scatter_{args.session}"
                         f"_U{unit_id}_{model_name}", figh=fig)
            plt.close(fig)
            print(f"  Saved U{unit_id} {model_name}")
