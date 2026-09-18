## intro

Supplementary Figure S8 looks at the structure of synthesis failures rather than their overall rate. A stimulus fails when its own post-hoc prediction misses the target by more than a fraction tau of the channel's target range; because the target grid is shared across models within a channel, the criterion is comparable across models. Panel a shows success rate as a model-by-level heatmap at the strict (2%) and lenient (5%) tolerances; panel b shows how large the misses are among the failures. The data come from the per-monkey exclusion caches via `loader.load_exclusions`, pooled over all sites and seeds in the five animals.

## setup

`THRESHOLDS` holds the two tolerances in display order, strict then lenient. `SHORT` relabels the untrained AlexNet, `FAIL_C` is the histogram color, and the layout constants (`LEFT`, `RIGHT`, `TOP`, `BOTTOM`, the width ratios and inner spacings) control the two-block layout described under `build_layout`.

## blocks

`load_all` and `success_matrix` prepare the data; `draw_success_heatmap` and `draw_error_hist` draw the two panels; `build_layout` handles the geometry so the heatmap cells come out square.

## fn:load_all

Concatenates generating model, ordinal level and relative error over the five monkeys. The error is stored once per threshold for convenience, but it is the same raw quantity in both slots; the threshold only enters when success is judged.

## fn:success_matrix

For one tolerance, the percentage of stimuli with relative error at or below it, in a matrix of models (in `MODEL_ORDER`) by ordinal level, pooled across sites and seeds.

## fn:draw_success_heatmap

Plots one success matrix on a red-yellow-green scale fixed from 0 to 100% with square cells. Model names appear on the y-axis of the first heatmap only, colored by model with the adversarially trained models in bold.

## fn:draw_error_hist

Takes only the failures at the strict tolerance (error above 2% of range), so the success spike at zero is excluded, and histograms the miss as a percentage of range. The upper edge is the 99th percentile rounded up to a multiple of 5, with larger misses clipped into the last bin; a dashed orange line marks the 5% tolerance.

## fn:build_layout

Builds the figure at a requested height: an outer two-column grid (panel a block, histogram) with a nested three-column grid inside the first block for the two heatmaps and a slim colorbar. It returns the physical width of a heatmap box so the caller can solve for the figure height that makes the cells exactly square.

## assemble

Because the heatmap cells are square and their count is fixed, the figure height is solved from the measured heatmap width in a probe pass and the real figure is built in a second pass. The heatmaps and their shared colorbar fill the left block and the histogram fills the right; panel letters are anchored to the top of each block.

## step:1

Loads the data, computes the success matrices at both tolerances, runs the probe layout to measure the heatmap width, solves the figure height, then builds the real figure and draws both heatmaps, the colorbar (labeled above the bar so it does not intrude on panel b), the failure histogram and the panel letters.

## step:2

Saves the figure.

## step:3

Prints the output path.

## closing

In panel a, read the heatmaps row by row: most cells are green, and the red cells concentrate at the highest drive levels and in particular models. Comparing the two heatmaps shows how much the lenient tolerance recovers. Panel b shows that most failing misses are small, with a long right tail.
