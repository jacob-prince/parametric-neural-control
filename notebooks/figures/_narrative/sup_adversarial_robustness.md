## intro
This figure checks that the adversarial-sensitivity results do not depend on how the attack is implemented, by comparing the multi-step PGD attack used in production with a single-step FGSM attack. Panel a plots the agreement between the two attacks at each perturbation strength; panel b plots the size of the attack effect against strength for three training families under both attacks; panel c plots the correlation between sensitivity (log-epsilon AUC over [0.125, 16]/255, the main-figure measure) and control slope for the 10, 9 and 7 model subsets, PGD beside FGSM. Inputs are the cluster CSVs in `fig5_ext_heldout*` and `fig5_ext_heldout_fgsm*`, the frozen agreement table `sup_advrob_fgsm_vs_pgd_eps.csv`, and `fig5_heldout_table.csv`.

## setup
`AUC_EPS_LO` and `AUC_EPS_HI` set the integration window for the sensitivity metric. `C_PGD` and `C_FGSM` colour the two attacks throughout, and `SUBSET_COL` colours the 10/9/7 subset labels to match the main figure. `ROBUST`, `UNTR` and `EXTREMES` define the subsets.

## blocks
Three loaders reshape the raw attack CSVs (per-axis AUC, per-family means, per-axis sweeps); three panel functions draw a, b and c.

## fn:load_logauc
The main-figure sensitivity recipe, parameterized by the directories to read and the output column name so it can be run once for PGD and once for FGSM: keep L-infinity rows within the epsilon window, compute normalized swing, average per readout and epsilon, then integrate over log2(epsilon) and divide by the window width.

## fn:swing_vs_eps
Averages normalized swing within three training families (Untrained, Conventional, Adv-trained) at each epsilon, for one attack directory.

## fn:_per_axis_swings
Mean normalized swing per model, site and epsilon, without the window restriction; feeds the per-model agreement curves.

## fn:panel_agreement
For each model, correlates PGD and FGSM swings across its 25 axes at every epsilon and draws a faint curve; the dashed grey curve is the mean of those within-model curves and the solid black curve is the pooled agreement over all 250 axes from the frozen table. The comment explains why the pooled curve can sit below the within-model mean: the two attacks re-rank models at intermediate and high epsilon.

## fn:panel_sweep
Log-log plot of family-mean swing against epsilon, solid circles for PGD and dashed open squares for FGSM, with a colour key and a line-style key as two separate legends.

## fn:panel_trend
Horizontal bars of Pearson r between sensitivity and control slope, for PGD and FGSM side by side, at the model level (per-model means) and the site level (site residuals, computed within the subset being plotted). Each bar is multiplied by the sign of the full 10-model correlation at that level so the full set reads positive; a subset that reverses sign dips left of zero. Bar lengths are floored at -0.08 so a small reversal stays visible without stretching the axis.

## assemble
A single row of three panels at 62 mm height. Panel c is nudged right and narrowed so its long y labels clear panel b. Each panel gets a title stating what is plotted and an italic grey subtitle in plain language; letters are placed after a draw.

## step:1
Reads the frozen agreement table (L-infinity rows, sorted by epsilon) and builds the per-axis frame: control slope from the held-out table merged with PGD and FGSM AUC.

## step:2
Creates the figure and the three axes, then repositions panel c.

## step:3
Draws the three panels and sets their titles and subtitles.

## step:4
Renders once and places the panel letters relative to each axes' position.

## step:5
Saves the figure.

## closing
In panel a the agreement should dip at intermediate budgets (the caption puts the minimum near 0.63) but stay well above zero. In panel b the green adversarially trained curves sit lowest under both attacks. In panel c the full-set and nine-trained bars are similar for PGD and FGSM, while the seven-model bars collapse toward zero, with a small sign reversal for FGSM.
