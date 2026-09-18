## intro
The main-figure gradient-spectra analyses are repeated across a ladder of five encoding-readout training diets, to ask whether fitting the readout on more, or cleaner, data changes the gradient spectra or how well their participation ratio (PR) predicts control. The diets range from calibration images only (d0, the synthesis-time readouts used in the main figure) through adding the pair's own accentuations (d1), all site-targeted accentuations (d2) and every accentuated stimulus (d3), to everything except the pair's own control-scored images (d4, zero circularity). Panel a shows per-model mean held-out spectra per diet, panel b the PR-versus-control scatter per diet, panel c the model- and site-level correlations across diets for the 10/9/7 subsets, and panel d readout predictivity along the ladder. Inputs are the frozen `sup_diet_refit.csv`, the per-diet gradient profile directories listed in `GRADDIR` (d0 reuses `fig5_heldout_gradfreq`), and `L.control_table()`.

## setup
`DIETS` and `DIET_TITLE` name the rungs; `GRADDIR` maps each to its profile directory. `FMIN` and `FMAX` fix the band, `EXTREMES` marks the untrained and adversarially trained models, and `SUBSET_COL` colours the 10/9/7 lines to match the other figures. The mean fitting-set sizes printed in the panel-a titles are computed from the CSV, not hard-coded.

## fn:spectral_pr
Participation ratio of a radial power profile over the band; its docstring notes it is an exact reparameterization of CoV.

## fn:load_profiles
Reads one directory of profile pickles into a dictionary keyed by (monkey, unit, model), averaging over images.

## fn:site_resid
Subtracts each site's mean over the nine trained models, keyed by monkey and unit.

## fn:pr_scatter
A compact version of the main-figure scatter: site dots, model means with 95% CIs (t times SEM; diamonds for the extremes), a dashed site-level fit and a bold model-level fit, and the two correlations boxed. Returns the model-level and site-level r.

## assemble
Axes are placed by hand in three rows. Row 1 has five spectra panels, one per diet; row 2 has five matching scatters; row 3 has two summary panels (model level, site level), a predictivity panel, and a model legend. Letters a to d mark the four groups.

## step:1
Reads the refit CSV and the control table and residualizes control slope.

## step:2
Loads the gradient profiles for every diet and computes the mean numbers of natural and accentuated fitting images per diet for the titles.

## step:3
Creates the figure and draws all three rows. Row 1 overlays per-model mean normalized spectra with the adversarially trained models drawn thicker. Row 2 attaches a PR to every control-table row for that diet and calls `pr_scatter`, while collecting model and site r for the 10, 9 and 7 subsets into a small table. Row 3 plots those correlations against diet, plots mean natural-test r and own-accentuation r with SEM error bars from the refit CSV, and draws the model legend.

## step:4
Places the four panel letters.

## step:5
Prints the model-level correlation table (diet by subset) and saves the figure.

## closing
Look for sameness: the spectra in panel a should overlay almost perfectly from column to column, the scatters in panel b should look alike, and the lines in panel c should be flat across the diets. Panel d shows how much the readout itself changes along the ladder.
