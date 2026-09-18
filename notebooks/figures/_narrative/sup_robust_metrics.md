## intro

Supplementary Figure S18 checks that the adversarially trained advantage in control does not depend on the scoring choice. The main text scores control as Pearson r on seed-averaged sweeps; here the same comparison is re-run with the control slope (panel a), the coefficient of determination R-squared (panel b) and per-seed control r, where each seed sweep is scored on its own (panel c). In every panel one dot is one observation, colored by monkey, and the group center with a 95% CI is overlaid for the adversarially trained, conventionally trained and untrained groups. Slopes and R-squared come from `loader.control_table()`; per-seed r values from `loader.control_seed_rs()`.

## setup

`GROUPS` fixes the three training families and `GROUP_COLOR` their colors (green, crimson and the Untrained model color). `_group` assigns each model to a family using `ROBUST_MODELS` and the untrained model name. `MK_LABEL` gives the monkey legend labels with each animal's area.

## blocks

`load` gathers the data; `_strip` draws one grouped strip panel and runs the group test; `_boot_median_ci` supplies the CI for the median variant; `_brackets` and `_one_bracket` annotate the result; `_spines` and `_sig` are small formatting helpers.

## fn:load

Copies the control table with a group column added, then walks every monkey, unit and model, collecting the per-seed control r values into one pool per group (with matching monkey labels for coloring) and a per-site-model mean over seeds.

## fn:_boot_median_ci

Percentile bootstrap (5000 resamples) of the median, used for R-squared because its heavy negative tail makes a t-based interval on the mean misleading.

## fn:_strip

For each group, plots the finite values as jittered dots colored by monkey, then a large group marker at the center (mean with a t-based CI, or median with the bootstrap CI when `central='median'`) with the whisker drawn on top. It runs a two-sided Mann-Whitney U test between the adversarially and conventionally trained groups, with Untrained excluded, and returns the p-value and the two group centers.

## fn:_one_bracket

Draws a bracket between two group positions at an axis-relative height and labels it with significance stars and the difference of centers.

## fn:_brackets

Places the single adversarially versus conventionally trained bracket near the top of the panel.

## assemble

Three equal panels in one row at two-column width. Each panel is built by `_strip` and then given panel-specific y-limits; panel b switches to a symlog y-axis so the negative R-squared tail fits. Panel letters and two figure-level legends (group colors on the left, monkey colors on the right) are placed above the panels.

## step:1

Loads the control table with groups and the per-seed r pools.

## step:2

Creates the figure and a 1 by 3 GridSpec.

## step:3

Draws the three panels: slope with the y-axis cropped to -0.35 to 1.5, R-squared with group medians on a symlog axis and a dashed zero line, and per-seed r cropped to -0.55 to 1.5. Each gets its bracket and its result is stored in `key`; finally the letters and the two legends are added.

## step:4

Saves the figure and prints `key`, the per-panel p-values and group centers.

## closing

In each panel compare the green and crimson center markers and read the bracket label: the difference and its significance are the result. Panel b's symlog axis makes the negative R-squared tail visible; note that its centers are medians, unlike the means in panels a and c.
