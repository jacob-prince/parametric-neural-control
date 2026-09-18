## intro
The previous figure showed that the automatically selected synthesis regularization tracks control fidelity. This one asks whether the adversarially trained models' control advantage survives once those hyperparameters are accounted for. A single joint linear model with model, site, augmentation-noise and spectral-decay terms is fit to each control outcome, and the part attributed to the two knobs is subtracted, leaving model and site structure in place. Panels A and B show the adjusted outcomes per model; panel C compares the raw adversarial-versus-conventional gap with the adjusted gap from an ANCOVA.

## setup
`COVARS` names the two regularization knobs regressed out. `OUTCOMES` lists the two control measures, `ROBUST` the two adversarially trained models, and `UNTR` the untrained baseline, which is shown in the per-model panels but never enters the family tests. `C_ADJ` is the orange used for adjusted bars.

## blocks
`load` prepares the table with residualized outcomes; `per_model_panel` and `summary_panel` each draw one kind of panel. The statistics come from the shared `pnc.famstats` helpers `partial_residuals` and `adv_gap`.

## fn:load
Merges the control table with the hyperparameter cache and, for each outcome, adds a column of partial residuals from the joint model with the knob effects removed but model and site effects kept.

## fn:per_model_panel
For one outcome, orders models by their adjusted mean and draws each model's site-model residuals as a jittered swarm. Two bars sit on each column: gray for the raw mean and black for the adjusted mean, so the shift from adjustment is visible. Adversarially trained model names are bold.

## fn:summary_panel
For each outcome, draws two bars: the raw adversarial-minus-conventional gap among the nine trained models (p from a within-site paired Wilcoxon over the 25 sites) and the hp-adjusted gap (p from an ANCOVA with site fixed effects and site-clustered errors). Significance stars sit above each bar.

## assemble
A single row of three panels, the two per-model swarms wider than the summary. The table is loaded, panels drawn, then letters, title, subtitle and a legend for the two mean bars are added after a draw pass.

## step:1
Loads the merged table with residualized outcomes.

## step:2
Creates the figure and grid and fills the control `r` panel, the control slope panel, and the summary.

## step:3
Places letters, the title and a subtitle noting the collinearity of the two knobs, adds a legend distinguishing raw from adjusted means, and prints the raw and adjusted gaps with their p-values to the console.

## step:4
Saves and closes the figure; `png` receives the path.

## closing
Compare the gray and black bars within each column of panels A and B: adjustment moves the means a little but leaves the two bold models at the top. In panel C both the raw and adjusted gaps should be positive and starred, which is the evidence that the advantage is not an artifact of regularization settings.
