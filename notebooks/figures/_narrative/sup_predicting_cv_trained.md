## intro
Among the nine trained models (225 encoding axes), what predicts held-out parametric control? The figure pits gradient spectral PR (top row) against adversarial sensitivity (bottom row), each alongside natural-image encoding accuracy. Panels a and d give 10-fold cross-validated R squared for the row's predictor alone, encoding accuracy alone, and both together, against a split-half ceiling. Panels b and e test the row's predictor alone under progressively harder held-out regimes: random folds, whole models, whole sites, whole animals. Panels c and f scatter measured against predicted control slope with the model itself held out. Inputs are `L.control_table()`, the frozen `sup_predicting_master.csv` (encoding accuracy and adversarial sensitivity) and the held-out gradient profiles in `fig5_heldout_gradfreq`. Untrained is excluded because it is an out-of-distribution extreme that inflates every correlation and breaks leave-one-model-out on the residualized outcome.

## setup
`TRAINED` is the analysed family; `--include-untrained` adds Untrained (filename tag `_incl_untr`) but the site residual always references the nine trained models. `ROWS` names the two predictors and their colours, `REGIMES` the four held-out designs, and `OUTC` the two outcomes. Models are ordinary least squares with one global slope per predictor and no per-animal terms, which is the only design that can predict an unseen animal.

## blocks
Table-building helpers, a small cross-validation kit (`_oof`, `_r2`, `folds_for`), and a ceiling estimate that recomputes the outcome from split halves of the seeds.

## fn:heldout_pr
PR per readout from the held-out radial profiles, over the main-figure band.

## fn:build_table
Merges the control table with the master CSV, attaches PR, residualizes both outcomes against each site's mean over the nine trained models, then keeps only `TRAINED` rows with complete predictors.

## fn:_oof
Out-of-fold predictions for an OLS model: predictors are z-scored using training-fold statistics, an intercept is added, and the fold's held-out rows are predicted.

## fn:folds_for
Random 10-fold splits for the first regime; otherwise one fold per model, per site, or per animal.

## fn:_control_cloud_by_seed
Rebuilds the (predicted, measured) pairs of one readout's accentuation sweep, grouped by seed, with predictions clamped at the firing floor, as `loader.control_cloud` does. Seeds are the experiment's repeat unit.

## fn:ceiling_resid
The explainable-variance ceiling for the residualized outcome. Seeds are split in half at random, the outcome is recomputed from each half per readout, each half is residualized within site, the halves are correlated across readouts and Spearman-Brown corrected, and the result is averaged over 100 partitions. Because cross-validated R squared is bounded by outcome reliability, this correlation is the R squared ceiling directly.

## assemble
Two rows of three hand-placed axes. In each row the left panel is grouped bars per outcome, the middle panel is bars per held-out regime (outcomes as solid and faded), and the right panel is the unseen-model scatter with a 1:1 line; letters run a to c and d to f.

## step:1
Configures the variant, builds the table, estimates both ceilings, then fills two dictionaries: `A` with 10-fold R squared for predictor alone, encoding alone and both, and `B` with the predictor-alone R squared under each regime, for both outcomes; prints everything.

## step:2
Creates the figure and fixes the row and column geometry.

## step:3
Loops over the two predictor rows and draws the three panels for each: the predictor-comparison bars with dashed ceiling segments and a legend, the regime bars, and the held-out-model scatter with its R squared and, in the top row only, the model colour key. Letters are placed per row.

## step:4
Saves the figure.

## closing
Follow the bars in b and e from left to right: PR should stay high all the way to held-out animals, while adversarial sensitivity and encoding accuracy fall away under the harder regimes. The scatter in c should hug the diagonal more closely than the one in f.
