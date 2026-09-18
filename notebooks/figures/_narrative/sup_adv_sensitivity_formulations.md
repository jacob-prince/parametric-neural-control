## intro
A robustness sweep for the endpoint-dependence result: adversarial sensitivity correlates with control across all 10 models, but the relationship disappears once the two adversarially trained models are removed. The figure asks whether that holds for every reasonable formulation of adversarial sensitivity, under both PGD and FGSM, and whether the gradient spectral participation ratio (PR) survives the same test. Panel a is a heatmap of oriented Pearson r between each sensitivity formulation and control slope, for PGD and FGSM and for the 10/9/7 model subsets, at the model level and the site level. Panel b does the same for a family of attack-independent gradient-spectral summaries. Panel c gives exact permutation p-values for the seven conventionally trained models. Inputs are the PGD and FGSM cluster CSVs, the held-out radial gradient profiles in `fig5_heldout_gradfreq`, `fig5_heldout_table.csv` and the descriptor table `sup_advform_heldout_descriptors.csv`; the full statistics matrix is also written next to the figure as a CSV.

## setup
`_BASE_FORMULATIONS` lists every measure with a kind and its arguments; `EPS_GRID` (0.125 to 64/255) supplies the per-epsilon swing rows and `FIG_DEFAULT` names the boxed main-figure measure. `FMIN` and `FMAX` fix the spectral band. `_configure` handles the two flags: `--outcome slope|r` picks the control outcome, and `--gradsum pr|cv` decides whether PR or CoV is the featured spectral measure (under `pr`, the CoV-family rows are swapped for PR counterparts). `SUBSET_COL` and the accent colours match the other supplementary figures.

## blocks
The helpers fall into three groups: builders of per-readout series (attack formulations and spectral summaries), the statistics of the main-figure recipe (residualization, subsets, exact permutation tests), and the drawing primitives (heat blocks, row labels, p strips).

## fn:curve_table
Per readout and epsilon, the mean normalized perturbation effect. `norm` selects the threat model, `up_only` uses the upward change alone (`adv_up` - `clean_pred`) instead of the full swing, and `denom` chooses the response normalizer.

## fn:log_auc
Trapezoid integral of the effect over log2(epsilon) within a window, divided by the window width. `lin_auc` is the same over linear epsilon.

## fn:load_heldout_profiles
One mean radial gradient power profile per readout, averaged over the 100 held-out images.

## fn:load_spectral_summaries
Collects every spectral summary as a per-readout Series: the descriptor-table measures (CoV, Wiener flatness, log-log decay slope, low-frequency fraction, centroid) plus measures computed here from the profiles over the same band: Gini coefficient, normalized entropy, participation ratio, median frequency, low-frequency fractions at several cutoffs, a log low/high power ratio, 1/f-weighted power, and CoV and PR variants that drop the three lowest bins or coarsely rebin.

## fn:build_series
Returns a dictionary mapping each measure kind to a callable that produces its per-readout Series for one attack's raw CSVs, together with the main curve table. Attack-independent kinds (gradient norms, spectral summaries) are included so the same interface serves every row.

## fn:site_residual
Control outcome minus its per-site mean over the nine trained models.

## fn:exact_p
One-sided exact permutation test at the model level: every relabeling of the seven model means, counting permutations at least as extreme in the direction of that measure's own all-10 correlation. A sign-flipped correlation therefore gets a p near 1.

## fn:exact_p_site
The same relabeling test at the site level. Model identities are permuted as columns of a site x model predictor matrix and the site-level r is recomputed for every permutation, using precomputed sums so the loop is cheap.

## fn:compute_matrix
Loops over every formulation and attack (shared kinds once, attack kinds for both PGD and FGSM), records site and model r for each subset, and the two exact p-values for the seven-model subset.

## fn:_oriented_grid
Builds the heatmap values, multiplying each measure's row by the sign of its own all-10 correlation at that level, so keeping the full-set relationship reads positive and a reversal reads negative.

## fn:heat_sens
Panel a block: seaborn heatmap with annotated values, a white divider between PGD and FGSM, white rules at the group breaks in `SENS_BREAKS`, a dark rule above the attack-independent gradient-norm rows (whose FGSM half is masked), and a black box around the main-figure default.

## fn:heat_spec
Panel b block: the spectral rows for the three subsets, with breaks separating concentration, location and low-frequency emphasis, and a black box around the featured spectral measure.

## fn:pstrip
One column of exact p-values on a log axis, rows aligned with the heat block: filled steel dots for PGD, open gold circles for FGSM, violet dots for spectral rows, a dashed line at p = 0.05, and a faint guide from each point to 1.

## assemble
The layout is fixed with `fig.add_axes` at three quarters of the two-column width. Panel a is two heat blocks (model level, site level) in the upper part; panel b is two narrower three-column blocks below with a shared row height; panel c is four p strips at the right, two aligned with a and two with b. A horizontal colour bar and a subset key finish the figure.

## step:1
Configures the variant, builds the PGD and FGSM series, reads the held-out table, residualizes the outcome, computes the full statistics matrix, writes it to CSV and prints it, then creates the figure and all axes.

## step:2
Draws the four heat blocks and their row labels, with the featured measure in bold.

## step:3
Draws the four p strips, aligns the panel-c title with panel a's titles after a draw, and adds the footnote explaining the one-sided test.

## step:4
Adds the colour bar, the subset key at the top, and the rotated group labels on the left.

## step:5
Places the panel letters.

## step:6
Saves the figure with the variant tag.

## closing
Read down the 7 columns. In panel a nearly every sensitivity row fades toward zero or turns negative for both attacks, and in panel c those dots sit right of the dashed p = 0.05 line. In panel b the spectral rows keep their sign in the 7 column and their p-values sit to the left of the line. `--outcome r` and `--gradsum cv` render the companion variants with filename tags.
