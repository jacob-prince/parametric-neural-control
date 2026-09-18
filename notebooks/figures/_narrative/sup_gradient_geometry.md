## intro
A standalone version of the gradient-geometry content of main Fig. 5, laid out exactly like the adversarial-sensitivity supplement (S39) with adversarial sensitivity replaced by the gradient spectral participation ratio (PR). Panel a is a mosaic for the cat seed at Monkey P ch8: input-gradient saliency maps on top, and below them each model's accentuation of that seed toward the site, matched to a common achieved response; columns are ordered by per-model mean PR. Panel b overlays each model's gradient frequency spectrum on the 100 held-out NSD images. Panels c to e plot PR against site-residualized control slope for the three model subsets. Inputs are the held-out radial profiles in `fig5_heldout_gradfreq`, `fig5_heldout_table.csv`, the frozen `sup_advsens_outcome_reliability.csv`, the grad-map cache behind `L.load_grad_maps`, and the accentuation-sweep PNGs.

## setup
`ACC_MONKEY`, `ACC_UNIT` and `SEED_IDX` fix the example site and seed (paul, 8, 3). `FMIN` and `FMAX` (1 and 112) fix the spectral band. `ROBUST`, `UNTR` and `EXTREMES` define the subsets and decide which models get diamond markers.

## blocks
Small spectral and image helpers feed one `build` function that does every computation and all the drawing.

## fn:normalize_row
Divides a radial profile by its band sum, for plotting spectra on a common scale.

## fn:spectral_pr
The participation ratio over the band: squared sum of power over sum of squared power, which equals the number of bins divided by (1 + CoV squared).

## fn:grad_to_saliency
L2 magnitude of a three-channel gradient, scaled by its 99th percentile and clipped to [0, 1].

## fn:load_heldout_profiles
One mean radial gradient power profile per (monkey, unit, model), averaged over the 100 held-out images.

## fn:accent_curve
Scans the accentuation directory for one model, site and seed and returns the sweep as sorted (level, achieved score, path) triples; `load_accent_at` picks the image whose score is closest to a target.

## fn:_pr_inset
Horizontal bars of per-model mean PR with SEM whiskers, sorted ascending, placed as an inset in the lower-left region of panel c.

## fn:build
Computes PR per readout, merges it with control slope, residualizes against each site's mean over the nine trained models, and averages profiles per model. The mosaic target response is the smallest of the trained models' maximum achieved responses; Untrained is capped below it and flagged with an asterisk when its maximum falls more than 0.15 below the target. The scatters use the main-figure convention: faint site dots, model means with 95% CIs (t times SEM), a dashed site-level fit and a bold model-level fit, with the outcome reliability stated under each title.

## assemble
A two-row outer grid: the top row is the 2 x (n+1) mosaic with a seed column, and the bottom row is four equal columns for panels b to e. The mosaic title and panel letters are placed last.

## step:1
Calls `build()` and saves the figure.

## closing
In the mosaic, PR falls from left to right and the adversarially trained maps at the right hug object surfaces while the maps at the left are diffuse and high-frequency. Panels c to e should all slope downward, lower PR going with stronger control; the caption reports model-level r of -0.90 for all 10 models, -0.46 without the adversarially trained pair, and -0.93 for the 7 conventionally trained models.
