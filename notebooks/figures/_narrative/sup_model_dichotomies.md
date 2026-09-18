## intro
With ten models in the study there are many ways to split them into groups. This figure asks whether any natural split, convolutional versus transformer, supervised versus self-supervised, language-aligned versus vision-only, separates models on parametric control the way adversarial training does. Each split is tested within site: for every recording site the mean outcome of each group's models is computed, and the per-site group differences are compared across the 25 sites with a Wilcoxon signed-rank test. The top row uses control `r` and the bottom row control slope, both read from the study control table.

## setup
`PANELS` is the heart of the script: four tuples naming a dichotomy, its two group labels, and the model lists on each side. Only the nine trained models appear; `UNTR` names the untrained baseline that is excluded. `ADV` holds the two adversarially trained models, which sit on opposite sides of the first three splits and are therefore removed for a second p-value in those panels. The `resid` parameter switches to site-residualized outcomes and adds a suffix to the output name.

## blocks
`build` shapes the data, `paired` runs the test, `fmt_delta` formats the effect, and `panel` draws one split.

## fn:paired
Takes the site-by-model pivot table and two model lists, averages each group within site, and returns the mean of the per-site differences, the Wilcoxon p-value, and the number of sites.

## fn:fmt_delta
Prints a signed difference with two decimals, but adds digits when the value would otherwise round to zero, so a tiny nonzero effect is not displayed as exactly zero.

## fn:build
Pivots the control table into sites by models for the requested outcome. With `resid` true it subtracts each site's mean over the nine trained models, removing between-site level differences.

## fn:panel
Draws one dichotomy. Each model's 25 site values are shown as faint dots and its mean as a larger dot, both in the model's color, spread within its group's column. A black bar with 95 percent confidence interval marks the group mean over the 25 per-site group means. The annotation reports the difference and p-value, colored red when significant, and for the three non-adversarial splits adds the p-value recomputed with the two adversarially trained models removed.

## assemble
The figure is a two-by-four grid: one row per outcome, one column per dichotomy. Panels are drawn first with a shared y-range per row, then letters, column titles and two legends are added after a draw pass.

## step:1
Builds the two pivot tables, sets a common y-limit per row with headroom for the annotation, and calls `panel` for each of the eight cells, collecting the statistics in `stats_out`.

## step:2
Places a bold letter on each panel and the dichotomy title above the top row, then adds a marker key (site-model, model mean, group mean with CI) and a nine-model color key along the bottom.

## step:3
Saves and closes the figure, then prints the within-site test results for every panel, including the excluding-adversarial p-values where they apply.

## closing
Look for the one panel in each row where the black group bars clearly separate: it should be the adversarial training column. In the other three the two groups overlap, and any hint of a difference disappears once the adversarially trained models are excluded. Set `resid` to `True` to render the site-residualized variant.
