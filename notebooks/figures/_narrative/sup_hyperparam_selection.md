## intro

Supplementary Figure S6 documents how the synthesis regularization was chosen for each of the 250 model-site pairs (25 sites by 10 models). Rather than hand-tuning, an automated search walked a five-rung ladder of (augmentation-noise, spectral-decay alpha) regimes and stopped at the first rung whose synthesis success rate exceeded 0.95. Panels a and b show where every model-site landed on the ladder, colored first by animal and then by model; panel c summarizes each model's mean rung; panel d shows the success rate actually achieved at the selected rung. The data are a single cached table of the selection outcome per model-site, `sup_hyperparam_hp_tuning_selected.csv`, rebuilt from the production logs.

## setup

`LADDER` lists the five (noise, alpha) pairs in rung order, with rung 0 the least aggressive, and `STOP` is the 0.95 success-rate target. `MONKEY_REGION` and `MONKEY_NAME` supply the legend labels, and `SHORT` relabels the untrained AlexNet as "Untrained". Changing `STOP` only moves the reference line and the reported fraction in panel d; it does not rerun the selection.

## blocks

`load` attaches a rung index to each row, then three panel functions draw the figure: `panel_a` draws the ladder (called twice with different colorings), `panel_b` the per-model bars and `panel_c` the success-rate histogram. Note that the code's `panel_b` and `panel_c` correspond to caption panels C and D.

## fn:load

Reads the cache and maps each row's (noise, decay) pair onto its `LADDER` index, stored as a new `rung` column. Values are rounded to two decimals before lookup so float noise in the CSV cannot break the match.

## fn:spread_ties

A helper that dodges points sharing a discrete value so they do not overlap. It is defined but not used by the current panels, which dodge with a fixed linear spread instead.

## fn:panel_a

Draws the ladder as five horizontal bands, one per rung, with every model-site on that rung spread evenly across the band and given a small vertical jitter; the count per rung is printed at the right. With `color_by='monkey'` dots take the animal's color; with `color_by='model'` they are first sorted by `MODEL_ORDER` and colored by model, so the model colors form contiguous runs. The y-axis is inverted so rung 0 sits on top, and `show_yaxis=False` lets the second copy share the first copy's tick labels.

## fn:panel_b

Mean selected rung per model as bars, sorted ascending and colored by model, with an orange triangle above the two adversarially trained models.

## fn:panel_c

Histogram of the achieved success rate at the selected rung across all model-sites, with a dashed red line at `STOP`. It returns the fraction of model-sites at or above that target, which is also printed on the panel.

## assemble

A single-row GridSpec of four axes at two-column width, with the two ladder panels given more width than the bar chart and histogram. Panels are drawn left to right, then spines are trimmed and titles and letters are placed against the measured axis positions.

## step:1

Loads the table and prints a few diagnostics: the number of model-sites, the number of distinct regimes, the correlation between the noise and decay settings (they move together along the ladder), and the count per rung.

## step:2

Creates the figure and four axes, then draws the ladder twice (by animal, then by model with its y-axis hidden), the per-model bars and the success histogram, keeping the returned fraction for the final printout.

## step:3

Hides the top and right spines on all four axes.

## step:4

Draws the canvas so axis positions are final, then writes a title above each panel and a bold letter at its top left; panel b's letter is tucked closer because it has no y-axis labels.

## step:5

Saves the figure and prints the fraction of model-sites whose achieved success rate met the 0.95 target.

## closing

Most dots sit on the top two rungs of the ladder in panels a and b, and every model spreads across several rungs. In panel d, look at how much of the histogram lies to the right of the dashed line: the procedure returns the best available regime, so a minority of model-sites fall short of the target.
