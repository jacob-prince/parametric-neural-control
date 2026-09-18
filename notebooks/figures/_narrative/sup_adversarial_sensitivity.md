## intro
Only two backbones were adversarially trained, but adversarial sensitivity can be measured for every one of the 250 fitted encoding axes, because each backbone plus its scalar readout is a differentiable image-to-response function that can be attacked with pixel-space PGD. This figure asks whether that sensitivity predicts parametric control. Panel a shows, for one example site (Monkey P ch8) and the cat seed, the smallest perturbation that raises each model's prediction by +2 z; panel b sweeps perturbation strength; panels c to e plot the sensitivity metric (log-epsilon AUC on the 100 held-out NSD images) against site-residualized control slope for three model subsets. Inputs are cluster PGD outputs (`fig5_advvis_exact`, `fig5_ext_heldout`, `fig5_ext_heldout_lowextra`), the held-out table `fig5_heldout_table.csv`, and the frozen reliability file `sup_advsens_outcome_reliability.csv`.

## setup
`ACC_MONKEY`, `ACC_UNIT` and `SEED_TAG` pick the example site and seed for the mosaic (paul, 8, seed3, the same as the main-figure mosaic). `AUC_EPS_LO` and `AUC_EPS_HI` (0.125 and 16, in units of 1/255) define the epsilon window the metric integrates over; changing them changes the x-axis of panels c to e and the shaded band in panel b together. `ROBUST`, `UNTR` and `EXTREMES` define the model subsets.

## blocks
Three loaders build the mosaic records, the sweep curves and the AUC metric; a small inset helper draws the per-model bar summary; `build` does all the drawing.

## fn:load_gallery
Reads one npz per model for the example site and seed: the clean image, the adversarial image, the epsilon that was used, and the clean and adversarial predictions. Its docstring notes that epsilon was found by bisection so the achieved change lands at +2 z within 0.05. Records are sorted by epsilon.

## fn:_heldout_raw
Concatenates the `adv_robustness_*.csv` files from the two held-out directories, skips smoke-test files, and keeps only the L-infinity rows.

## fn:load_heldout_curves
Defines normalized swing as (`adv_up` - `adv_dn`) / `range_q99q01`, then averages it per model, site and epsilon. This is the raw material for panel b and extends below the integration window.

## fn:load_heldout_logauc
The sensitivity metric. Restricts to the epsilon window, then integrates normalized swing over log2(epsilon) with the trapezoid rule and divides by the window width, so every octave of perturbation strength carries equal weight and the result matches the log axis in panel b.

## fn:_sensitivity_inset
Horizontal bars of each model's mean AUC with SEM whiskers, sorted from least to most sensitive, placed as a square inset in the top-right corner of panel c.

## fn:build
Merges the held-out table with the AUC metric, residualizes control slope against each site's mean over the nine trained models, then draws the mosaic, the sweep and the three scatters. In each scatter, faint dots are sites, large markers are model means with 95% CIs (t times SEM over the model's sites; diamonds for the untrained and adversarially trained models), the dashed line is the site-level fit and the bold line the model-level fit. Each panel's subtitle reports the split-half reliability of the outcome for that subset.

## assemble
A two-row outer grid: the top row is a 2 x (n+1) mosaic (seed column, then one column per model, adversarial image above perturbation), ordered by per-model mean sensitivity with the most sensitive on the left; the bottom row is four equal columns for panels b to e. The mosaic title and panel letters are added last.

## step:1
Calls `build()` and saves the figure.

## closing
Read the mosaic left to right: epsilon grows as sensitivity falls, and the adversarially trained models end up at the right with the largest perturbations. In panels c to e the positive relationship present with all 10 models should weaken once the two adversarially trained models are removed, and be absent among the 7 conventionally trained models.
