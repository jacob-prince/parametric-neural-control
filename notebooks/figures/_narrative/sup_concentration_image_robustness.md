## intro
Gradient spectral PR is meant to be a property of a fitted encoding axis, so it should not depend on which images are used to probe it. This figure tests that. Panel a compares per-axis PR measured on the 10 synthesis seeds with PR measured on the 100 held-out NSD images; panel b checks that the 10-model ranking is preserved; panel c shows that held-out PR still predicts control within site, with a leave-one-model-out cross-validated fit; panel d asks how many probe images are needed for the estimate and its association with control to stabilize. Inputs are the per-image held-out profiles in `fig5_heldout_gradfreq`, the descriptor table `sup_advform_heldout_descriptors.csv`, and the seed-based profiles from `L.load_gradient_freq()`. The module docstring also describes a specification-curve analysis (panels e and f) that this version of the script does not draw.

## setup
`FMIN` and `FMAX` fix the band. `--gradsum pr|cv` selects PR or CoV through `_configure`, which also sets `SUMM`, the active band summary. Inside `main`, `KS` lists the probe-image counts tried in panel d (1, 2, 5, 10, 25, 100) and `NBOOT` the number of bootstrap draws per count. `FIG4_MODEL_ORDER` fixes the legend order in panel a.

## fn:cv_lin
Coefficient of variation of band power along the last axis, optionally on the amplitude spectrum. `pr_lin` is the participation ratio over the same band; the two are monotone transforms of each other, so every correlation flips sign between them.

## fn:parse_pkl
Recovers (monkey, unit, model) from a profile filename.

## fn:load
Reads the descriptor table, stacks the per-image profiles into an array of shape (250 axes, 100 images, 158 bins) aligned to the table rows, computes the seed-based summary per axis from `load_gradient_freq`, and tags each row with its model group.

## fn:make_demean
Builds a fast within-site demeaning function from integer site codes using `bincount`, so within-site residuals can be recomputed hundreds of times in the bootstrap.

## fn:lomo_r2
Leave-one-model-out R squared: ordinary least squares on the summary plus monkey dummies, fitted with one model's 25 axes held out at a time, scored on the pooled out-of-fold predictions.

## assemble
All statistics are computed first, then a single row of four square panels is drawn in order a to d, and letters are anchored just above each title after a draw.

## step:1
Configures the variant, loads the data, builds site codes and the demeaning function, builds monkey dummies, averages the per-image spectra, asserts that the recomputed CoV matches the descriptor table (an alignment check), and computes the canonical held-out summary per axis.

## step:2
Defines the within-site correlation helper, then computes the panel statistics: the seed-versus-held-out correlation and per-model ranks with their Spearman rho, the within-site r and leave-one-model-out R squared for both outcomes, and the panel-d bootstrap, which for each probe count draws two random image subsets, recomputes PR from each, and records split-half reliability and the within-site correlation with each outcome. The figure and its four axes are created at the end.

## step:3
A small styling helper for tick sizes and spine widths.

## step:4
Defines the boxed-legend helper and draws the panels: a is the scatter with an identity line and model legend; b is the rank-rank plot with model names placed to the side (flipped inward or outward depending on the metric); c is the within-site scatter with a fitted line and a stats box; d plots reliability and the two control correlations against probe count with 95% CI bands. Letters are placed after a draw.

## step:5
Saves the figure and prints the headline numbers for each panel.

## closing
The caption reports r = 0.98 between seed and held-out PR, Spearman rho = 0.93 for the model ranking, within-site r of -0.64 (slope) and -0.60 (control r), and leave-one-model-out R squared of 0.45 and 0.38. In panel d the curves should flatten within a handful of images. `--gradsum cv` renders the CoV companion.
