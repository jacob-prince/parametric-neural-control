## intro
Axis alignment measures how much of a parametric sweep actually lands on the encoding axis, expressed as a per-dimension displacement ratio and averaged over the ten seeds of each site-model. If adversarially trained models simply produce better-aligned sweeps, that alone might explain their control advantage. This figure first shows the association between alignment and control across the trained site-models, then repeats the per-model comparison with alignment regressed out, and finally compares the raw and alignment-adjusted family gaps. Alignment comes from the loader's axis-alignment cache and the outcomes from the control table.

## setup
`OUTCOMES` lists control `r` and control slope, `ROBUST` the two adversarially trained models, and `UNTR` the untrained baseline, which is excluded from the scatter panels and the family tests but shown in the per-model panels. `C_ADJ` colors the adjusted bars.

## blocks
`build` assembles the table and residuals; `scatter_panel`, `per_model_panel` and `summary_panel` draw the three panel types, using the shared `adv_gap` and `partial_residuals` helpers for statistics.

## fn:build
Computes the alignment ratio for every (site, model, seed) record, averages it per site-model, and joins it to the control outcomes, keeping only rows where both outcomes are finite. It then adds partial residuals for each outcome from a joint model with model and site terms, removing only the alignment effect.

## fn:scatter_panel
Plots alignment against one outcome for the trained site-models, colored by model with adversarially trained points outlined in black, and adds a pooled fit with Pearson `r`, p-value and n in a corner box.

## fn:per_model_panel
Orders all ten models by their adjusted mean and shows each model's residuals as a swarm, with a gray bar for the raw mean and a black bar for the alignment-adjusted mean.

## fn:summary_panel
For each outcome, bars for the raw adversarial-minus-conventional gap (within-site Wilcoxon p) and the alignment-adjusted gap (ANCOVA with site fixed effects and clustered errors), with significance stars.

## assemble
Two grid rows: the top has the two alignment scatters, the bottom the two per-model panels and the summary. The table is built once, the trained subset is taken for the scatters, panels are drawn, and letters, title and a mean-bar legend are placed after a draw pass.

## step:1
Builds the full table and extracts the trained-model subset used by the scatter panels.

## step:2
Creates the two grids and fills the alignment-versus-`r` and alignment-versus-slope scatters (with titles), the two adjusted per-model panels, and the summary.

## step:3
Places letters and the figure title, adds the raw-versus-adjusted mean legend between the rows, and prints the raw and adjusted gaps with p-values.

## step:4
Saves and closes the figure; `png` receives the path.

## closing
The scatters should show a weak positive slope, enough to justify adjusting. In the bottom row the model ordering should be preserved after adjustment, with the two bold models still highest, and all four summary bars positive and starred.
