"""
accentuated_image_utils.py
==========================
Utilities for loading and visualising accentuated image sequences.

Level normalization
-------------------
Levels are normalized to the natural image response distribution per unit:
  level 0   = 1st  percentile of natural responses
  level 1   = 99th percentile of natural responses
  sweep     = -0.25 → +1.5  (11 uniform steps, step=0.175)
  raw range ≈ [-1.85, +6.24]  (p99-p1 ≈ 4.62 raw units, p1 ≈ -0.69)

Key functions
-------------
load_accentuated_sequence(df, model_name, unit_id, img_id, ...)
    Return images and metadata for one (model, unit, img) trajectory
    sorted by level, with optional level subsampling.

plot_level_row(images, levels, ...)
    Show images as a single row, labelled by level.

plot_level_grid(df, model_names, unit_id, img_ids, ...)
    Grid of rows: one row per (model_name x img_id) combination.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_accentuated_sequence(
    df,
    model_name: str,
    unit_id: int,
    img_id: int,
    n_levels: int | None = None,
    level_indices: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, object]:
    """
    Load a sequence of accentuated images for one (model, unit, img) trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Accentuated stimulus dataframe with columns:
        model_name, unit_id, img_id, level, score, filepath, ...
    model_name : str
        Accentuator model (e.g. 'resnet50_robust').
    unit_id : int
        Target unit index.
    img_id : int
        Seed image index.
    n_levels : int, optional
        Evenly subsample this many levels across the full range.
        If None, use all levels.
    level_indices : list[int], optional
        Explicit integer indices into the sorted level list to select.
        Overrides n_levels.

    Returns
    -------
    images : list[np.ndarray]
        List of (H, W, C) uint8 images sorted by level.
    levels : np.ndarray
        Corresponding level values.
    df_seq : pd.DataFrame
        Filtered and sorted sub-dataframe (includes score, pred_resp cols, etc.)
    """
    df_seq = df.query(
        "model_name == @model_name and unit_id == @unit_id and img_id == @img_id"
    ).sort_values("level").reset_index(drop=True)

    if len(df_seq) == 0:
        raise ValueError(
            f"No rows found for model_name={model_name!r}, "
            f"unit_id={unit_id}, img_id={img_id}"
        )

    # Subsample levels
    if level_indices is not None:
        df_seq = df_seq.iloc[level_indices].reset_index(drop=True)
    elif n_levels is not None and n_levels < len(df_seq):
        idx = np.round(np.linspace(0, len(df_seq) - 1, n_levels)).astype(int)
        df_seq = df_seq.iloc[idx].reset_index(drop=True)

    images = [np.array(Image.open(fp).convert("RGB")) for fp in df_seq["filepath"]]
    levels = df_seq["level"].to_numpy(dtype=float)

    return images, levels, df_seq


# ---------------------------------------------------------------------------
# Single-row plot
# ---------------------------------------------------------------------------
def plot_level_row(
    images: list[np.ndarray],
    levels: np.ndarray,
    title: str = "",
    label_key: str = "level",          # "level" or "score"
    scores: np.ndarray | None = None,
    ax_size: float = 2.0,
    show: bool = True,
) -> plt.Figure:
    """
    Display a row of images labelled by their accentuation level.

    Parameters
    ----------
    images : list of (H,W,C) arrays
    levels : 1-D array of level values
    title  : row title (y-label on the leftmost panel)
    label_key : what to show under each image ("level" or "score")
    scores : optional score values (used if label_key="score")
    ax_size : size of each square panel in inches
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(ax_size * n, ax_size + 0.5))
    if n == 1:
        axes = [axes]

    for k, (ax, img, lv) in enumerate(zip(axes, images, levels)):
        ax.imshow(img)
        ax.axis("off")
        val  = scores[k] if (label_key == "score" and scores is not None) else lv
        ax.set_title(f"{val:.2f}", fontsize=7, pad=2)

    if title:
        axes[0].set_ylabel(title, fontsize=8, rotation=0,
                           ha="right", va="center", labelpad=4)
    fig.tight_layout(pad=0.3)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-row grid
# ---------------------------------------------------------------------------
def plot_level_grid(
    df,
    model_names: list[str],
    unit_id: int,
    img_ids: list[int],
    n_levels: int = 9,
    level_indices: Sequence[int] | None = None,
    label_key: str = "level",
    ax_size: float = 1.8,
    show: bool = True,
) -> plt.Figure:
    """
    Grid of image rows: rows = (model_name, img_id) combinations,
    cols = level steps.

    Parameters
    ----------
    df : accentuated stimulus dataframe
    model_names : list of accentuator model names (one per row group)
    unit_id : target unit
    img_ids : list of seed image ids to show (one row per img_id per model)
    n_levels : number of evenly-spaced levels to subsample (if level_indices is None)
    level_indices : explicit level indices (overrides n_levels)
    label_key : "level" or "score"
    ax_size : panel size in inches
    """
    rows = [(mn, iid) for mn in model_names for iid in img_ids]
    n_rows = len(rows)

    # Determine n_cols from first valid sequence
    _imgs, _lvls, _ = load_accentuated_sequence(
        df, rows[0][0], unit_id, rows[0][1],
        n_levels=n_levels, level_indices=level_indices)
    n_cols = len(_imgs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(ax_size * n_cols, ax_size * n_rows + 0.4),
        squeeze=False,
    )

    for row_idx, (model_name, img_id) in enumerate(rows):
        try:
            images, levels, df_seq = load_accentuated_sequence(
                df, model_name, unit_id, img_id,
                n_levels=n_levels, level_indices=level_indices,
            )
        except ValueError as e:
            print(f"  Warning: {e}")
            continue

        scores = df_seq["score"].to_numpy(dtype=float)

        for col_idx, (ax, img, lv) in enumerate(zip(axes[row_idx], images, levels)):
            ax.imshow(img)
            ax.axis("off")
            val = scores[col_idx] if label_key == "score" else lv
            if row_idx == 0:                        # level label on top row only
                ax.set_title(f"{val:.2f}", fontsize=6, pad=2)
            if col_idx == 0:                        # row label on left
                short_mn = model_name.replace("_training_seed_01", "").replace("_", "\n")
                ax.set_ylabel(f"{short_mn}\nimg {img_id}",
                              fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=4)

    fig.suptitle(f"Accentuated images — unit {unit_id}", fontsize=11, y=1.01)
    fig.tight_layout(pad=0.2)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Quick convenience wrapper
# ---------------------------------------------------------------------------
def show_sequence(
    df,
    model_name: str,
    unit_id: int,
    img_id: int,
    n_levels: int = 9,
    **kwargs,
) -> plt.Figure:
    """One-liner: load + plot a single accentuation trajectory."""
    images, levels, _ = load_accentuated_sequence(
        df, model_name, unit_id, img_id, n_levels=n_levels)
    return plot_level_row(
        images, levels,
        title=f"{model_name} | U{unit_id} img{img_id}",
        **kwargs,
    )
