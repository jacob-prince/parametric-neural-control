## intro

Supplementary Figure S21 asks whether the ranking of models by control is the same in every recording area. Per-area, per-model mean control r is computed for the five areas (one per animal: aIT, cIT, V3/V4 and two STS areas). Panel a plots each model's mean across the areas as a line, with adversarially trained models thicker; panel b is the area-by-area matrix of Spearman rank correlations between the per-model orderings, with the strongest pair boxed. Data come from `loader.control_table()`.

## setup

`MONK` fixes the area order and `AREA_LABEL` names each area together with its animal so the two STS areas are told apart. `RHO_CMAP` is the mako colormap for rho between 0 and 1. Note that in the code `panel_b` draws the line plot (caption panel A) and `panel_a` draws the heatmap (caption panel B); the assembly places them accordingly.

## fn:build

Pivots the control table to a model-by-area matrix of mean control r, computes Spearman rho for each of the 10 area pairs into a symmetric matrix with ones on the diagonal, and also keeps the raw per-site r values per (model, area) for the faint dots in the line plot.

## fn:panel_a

The heatmap: rho values printed in each cell (white text on dark cells), the aIT-cIT pair boxed in orange in both off-diagonal cells, area labels colored by animal and a colorbar.

## fn:panel_b

The line plot: models ordered by grand-mean control r, each drawn as a line with markers across the five areas in the model color, thicker for adversarially trained models, with the individual per-site r values behind as faint jittered dots. The legend handles are created here but placed by the assembly.

## assemble

Two axes in one row, the line plot twice the width of the heatmap. Titles and letters share one baseline computed from the taller of the two axes. The model legend is placed below the line plot, and a one-line summary of mean rho and the aIT-cIT rho is written under the heatmap.

## step:1

Runs `build`, computes the mean and sample SD of rho over the 10 pairs and pulls out the aIT-cIT value, and prints them.

## step:2

Creates the figure and two axes, drawing the line plot on the left and the heatmap on the right.

## step:3

Draws the canvas, writes the two titles and letters at a shared height, places the five-column model legend under panel a and the rho summary line under panel b.

## step:4

Saves the figure.

## closing

In panel a the lines should barely cross: the same two thick lines stay on top in every area while the whole bundle dips in aIT and rises in V3/V4. In panel b the boxed cell is the strongest agreement, and the summary line gives the average over all pairs.
