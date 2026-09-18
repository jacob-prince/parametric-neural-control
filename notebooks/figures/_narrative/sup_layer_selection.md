## intro

Supplementary Figure S5 shows how the encoding layer was chosen for each of the 250 model-site pairs and whether the chosen depth follows the cortical hierarchy. Panel a has one small plot per model: a validation R-squared curve against relative layer depth for every site, colored by area, with the selected peak marked. Panel b swarms the selected depth per model, panel c summarizes it as a mean with range (reproducing the manuscript's Table 3), and panel d plots the mean selected depth per recorded area with one line per model plus a grand mean. Panels e and f give per-model Spearman rank correlations between selected depth and area position along the ventral stream, first at the coarse area level and then with Monkey V's five sites resolved by their position along the probe. Everything comes from `loader.load_layer_selection()`, whose entries are keyed by (monkey, model, unit) and carry the curve, the peak depth and R-squared, the area, and the probe depth.

## setup

`AREA_COLOR` maps each area to its animal's color (the two STS animals share one), `AREA_ORDER` runs posterior to anterior with STS last, and `VENTRAL_AREAS` is the subset used for the hierarchy correlations. Models come from the loader config, and the adversarially trained models are drawn in bold wherever model names appear.

## blocks

After `load`, the helpers are one renderer per panel, with three small support functions: `_venus_order` and `_area_mean` for the depth panels, and `_stars` and `_draw_rho_bars` shared by the two correlation panels.

## fn:load

Reads the cache, lists the models, extracts the area and probe depth per site, and returns the sites sorted by area order so all panels iterate in the same sequence.

## fn:draw_curves

One model's panel a: a faint line per site and a dot at its selected peak, both in the area color, with fixed axis ranges so all ten panels are comparable.

## fn:draw_peak_by_model

Panel b: jittered selected depths per model colored by area, with a black median bar overlaid by a thinner bar in the model's color.

## fn:draw_table3

Panel c: per model, the mean selected depth as a dot and the min-to-max range as a line, sorted by mean, with model names as tick labels. Returns the rows so the assembly can print the table.

## fn:_venus_order

Orders Monkey V's five sites from deepest (V3) to shallowest (V4) by their probe depth, which is the posterior-to-anterior order used in panel f.

## fn:draw_area_depth

Panel d: for each model, the mean selected depth in each area of `AREA_ORDER` as a thin colored line, plus the grand mean across models in black, with the ventral areas lightly shaded.

## fn:ventral_rho

Spearman correlation between selected depth and the coarse ventral rank (V3, V4, cIT, aIT) across all sites in those areas, for one model.

## fn:fine_rho

The same correlation with a finer ordering: Monkey V's five sites take ranks zero to four in probe order, then cIT and aIT sites take the next two ranks.

## fn:_draw_rho_bars

Horizontal bars of per-model correlation sorted ascending, colored by model, with a significance label beside each bar and a zero line.

## assemble

The figure is two-column width with a three-row GridSpec. The top row is a nested 2 by 5 block of per-model curve panels; the middle row holds panels b and c side by side; the bottom row holds panel d and the two correlation panels. The area legend sits at the top right, and letters are placed after a canvas draw.

## step:1

Loads the data and derives the model list, site order, area and depth lookups.

## step:2

Creates the figure and all three rows of axes, drawing every panel in turn (curves, swarm, table, area depth, and the two correlation bars), then adds the title and the area legend. Note that the axes for panels e and f are named `axF` and `axG` in the code; the letters follow the caption.

## step:3

Draws the canvas and places the six panel letters relative to the measured axes positions.

## step:4

Saves the figure, then prints the Table 3 reproduction (mean and range per model) and each model's coarse and fine Spearman correlations with significance.

## closing

In panel a the curves rise and then plateau, with the marked peaks at intermediate to late depths. In panel d the black grand-mean line rises toward the anterior areas, and panels e and f show which models carry a significant depth-by-area correlation.
