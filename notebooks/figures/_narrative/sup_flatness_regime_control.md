## intro
Accentuation stimuli were synthesized under a Fourier-domain regularizer whose strength (an augmentation-noise / spectral-decay rung) was selected automatically per site-model. A sceptic could argue that low-PR gradients predict control only because they suit that optimizer. This figure stratifies the 250 site-models by their selected regime, five discrete rungs, and re-evaluates the PR-control relationship within regimes. Panel a plots PR against control score coloured by regime with a fit per regime; panel b plots the per-regime correlation for both control score and control slope, with the pooled and within-regime correlations as reference lines. Inputs are `L.control_table()`, the seed-based gradient profiles from `L.load_gradient_freq()`, and the selected hyperparameters in `sup_hyperparam_hp_tuning_selected.csv`.

## setup
`FMIN` and `FMAX` fix the band and `--gradsum cv` switches the summary from PR to CoV (filename tag `_cv`). Regimes are ordered by their spectral-decay alpha and coloured from viridis. A per-regime fit line is drawn only when the regime holds at least eight site-models, and per-regime markers are filled only when p < 0.05.

## fn:_cv
The band summary for one profile: participation ratio by default, coefficient of variation in the alternative mode.

## fn:load
Merges the control table (control r and slope), the per-axis summary, and each site-model's selected noise and decay values; the regime is the (noise, decay) pair.

## fn:within_regime_r
The partial correlation of the summary with a control outcome after removing regime means from both variables. Its p-value uses n minus the number of regimes minus one degrees of freedom, the correct count once five regime means have been partialled out.

## assemble
A single row with a wider left panel for the scatter and a narrower right panel for the per-regime summary; letters are placed as figure text.

## step:1
Sets the mode, loads and merges the data, orders the regimes by alpha and assigns colours.

## step:2
Computes the pooled correlation, the within-regime correlation with its p-value, and the per-regime correlation with p-value, each for control r and control slope.

## step:3
Creates the figure. Panel a scatters PR against control score by regime, adds per-regime fits, and boxes the pooled and within-regime correlations (the box moves to whichever corner is free for the chosen metric). Panel b places a circle (control r) and a square (control slope) per regime, draws the within-regime and pooled reference lines, and adds two legends: the regime colours with their site counts, and the marker convention.

## step:4
Places the panel letters.

## step:5
Saves the figure and prints the pooled, within-regime and per-regime statistics.

## closing
All five regimes should sit below zero in panel b, and the dashed within-regime line should be close to the dotted pooled line; the caption reports an attenuation only from -0.47 to -0.44 for control score. In PR mode the y-axis is negative because PR runs opposite to CoV.
