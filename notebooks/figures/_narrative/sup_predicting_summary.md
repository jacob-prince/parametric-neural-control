## intro
A summary of the predictor comparison: a leaderboard of single predictors of site-residualized control, contrasting the trained models (n = 225 site-models) with the conventionally trained subset (n = 175), plus the best predictor subset evaluated by repeated 10-fold cross-validation. Panel a shows each predictor's site-level correlation with control slope and with control r as a dumbbell, the full family filled and the conventionally trained subset open, with a star when the subset stays significant. Panel b scatters the best model's cross-validated predictions against measured control slope, one colour per monkey. Nothing is computed here: every statistic is read from `sup_predicting_results_no_untrained.pkl` (or `sup_predicting_results.pkl` with `--all-models`), produced by `scripts/preprocessing/build_predicting_results.py`.

## setup
`PRED2CAT` groups each predictor by what it measures and `CAT_COLORS` colours the groups; `ASHORT` and `PRED_DESC` supply the short labels and one-line descriptions for the key. `MK_LEG` maps monkeys to their area labels. The canonical render excludes Untrained (`EXCL`). One thing to know when reading the code: the results pickle indexes the full analysed family under the key `10` even when that family is the nine trained models.

## fn:_load
Loads the pickle, sets the family label, builds the leaderboard list (the pickle's predictor groups plus a few extras if they are missing), and prepares the marker-legend handles with the two sample sizes.

## fn:kfold_best_model
Pulls the cached best-subset result for control slope: out-of-fold predictions, measured values, monkey labels, cross-validated R squared, the predicted-versus-measured r, the explainable ceiling and the chosen predictors.

## fn:panel_leaderboard
Two side-by-side columns (slope, then r) sharing one predictor order, sorted by the full family's correlation magnitude with slope. Each value is multiplied by the sign of the full family's correlation so the full set reads positive and a reversal in the conventionally trained subset dips left of zero. Stars mark predictors with p < 0.05 in that subset; the two gradient-spectral labels are bolded.

## fn:panel_best
Scatter of predicted versus measured control slope, coloured by monkey, with a per-monkey regression line spanning that monkey's measured range and a dashed identity line on a square axis.

## fn:draw_desc_box
The boxed key: predictors listed under their category, each with its short label and description.

## assemble
A single row of three cells: the leaderboard (split internally into slope and r), the predictor key, and the best-model scatter. Titles, the stats line, the two legends and the letters are positioned after a draw from the axes' actual positions.

## step:1
Loads the results, creates the figure and grid, and draws the three panels.

## step:2
Renders once, then adds the two panel titles, the R squared / r / ceiling-fraction line above panel b, the marker legend strip under the leaderboard, the monkey legend and the best-subset footnote under panel b, and the letters.

## step:3
Saves the figure and prints the chosen subset, its cross-validated R squared and the ceiling.

## closing
The gradient-spectral predictors should lead the leaderboard and keep their stars, while the adversarial-sensitivity dumbbell has its open marker crossing zero. The caption gives R squared = 0.51 and r = 0.72 for the best model, 60% of the 0.85 ceiling; treat that as an optimistic bound, since the same folds were used to select and to score the subset. `--all-models` renders the ten-model variant.
