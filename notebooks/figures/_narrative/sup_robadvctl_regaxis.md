## intro
Each accentuation run selected its own synthesis regularization automatically: an augmentation-noise level and a spectral-decay exponent. This figure asks whether those selected settings covary with control fidelity across the 250 site-models. Each panel plots one regularization knob against one control outcome, with a pooled fit and three correlations: the pooled Pearson `r`, a within-model partial `r`, and a within-site partial `r`. The data are the study control table merged with the hyperparameter-selection cache.

## setup
`HP_CSV` names the selected-hyperparameter table and `HP_HINT` the script that builds it. `JW` sets the small horizontal jitter applied to each knob so identical selected values do not stack. The four panels are fixed in `main`: noise and decay on the x-axis, control `r` and control slope on the y-axis.

## blocks
`load` merges the tables, `_wm_r` and `_ws_r` compute the two partial correlations, and `scatter_panel` draws one cell. `collinearity_panel` is defined but not drawn.

## fn:load
Merges the control table with the hyperparameter table on monkey, model and channel, producing one row per site-model with its selected noise, decay and control outcomes.

## fn:_wm_r
Centers both the predictor and the outcome within each model, then correlates the residuals. This is the association that remains after model identity is removed.

## fn:_ws_r
Does the same centering within each site (monkey plus channel), so the correlation reflects variation across models at the same site.

## fn:scatter_panel
Scatters the site-models colored by model with a little jitter, fits and draws a pooled least-squares line, and writes the pooled `r` with its p-value and the two partial correlations in a box at the lower right, which is clear of the upward-trending cloud.

## fn:collinearity_panel
Would plot augmentation noise against spectral decay with their correlation; the panel was removed from the figure and the function is no longer called.

## assemble
The canvas is a two-by-two grid, narrower and taller than a standard two-column figure. Rows are the two knobs and columns the two outcomes. Panels are drawn first, then letters, a title and a subtitle are added.

## step:1
Loads the merged table, creates the grid, and fills the four scatter panels: noise versus `r`, noise versus slope, decay versus `r`, decay versus slope.

## step:2
After a draw pass, places a bold letter on each panel and writes the figure title and a subtitle stating the sample and the plotting convention.

## step:3
Saves and closes the figure; `png` receives the path.

## closing
All four panels should trend upward, and the within-model and within-site correlations in each box show that the trend is not only a between-model effect. This motivates the next figure, which regresses these knobs out of the control outcomes.
