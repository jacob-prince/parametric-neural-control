## intro

Supplementary Figure S9 asks whether an accentuation sweep moves the image along the site's encoding axis or mostly sideways in feature space. Each sweep's net displacement in the 750-dimensional PCA feature space is split into an on-axis component along the fitted encoding axis (DM) and an off-axis residual pooled over the remaining 749 dimensions (DOM). Panel a is a toy schematic of the decomposition; panel b is the distribution over sweeps of the per-dimension ratio DM / (DOM / sqrt(749)); panel c compares the on-axis displacement to the sorted spectrum of off-axis displacements; panel d breaks the ratio out by model. All quantities come precomputed from `loader.load_axis_alignment()`.

## setup

`C_ONAXIS` (orange) and `C_OFFAXIS` (blue) are used consistently for DM and DOM across all panels. The number of off-axis dimensions `n_off` is read from the cache rather than hard-coded, and `ymax_D` (40) is the ceiling at which the ratio is clipped in panels b and d.

## blocks

One drawing function per panel, plus `_model_xticks`, which formats the model names along the x-axis of panel d.

## fn:draw_schematic

Draws a synthetic latent space: a gray point cloud elongated along a fixed encoding axis, an 11-level sweep colored blue to red from suppress to drive, and the net displacement between its end points decomposed into an orange on-axis arrow (DM) and a blue perpendicular arrow (DOM) meeting at a right-angle mark. Nothing here is data; it only fixes the geometry in the reader's mind.

## fn:draw_ratio_hist

Histogram of the per-dimension ratio across sweeps, clipped at 40, with a dotted reference at 1 (where the on-axis displacement equals the root-mean-square per-dimension off-axis displacement) and a dashed orange line at the median.

## fn:draw_spectrum

Plots, on log-log axes, the median and interquartile band of the sorted off-axis displacement spectrum across sweeps, and places the median DM with its interquartile range as a single orange point at the left, so the encoding axis can be compared with the most-modulated off-axis dimensions.

## fn:draw_ratio_by_model

One column per model: all sweeps as faint jittered dots (clipped at the ceiling), a black median bar overlaid with the model's color, and the ten seeds of that model's representative channel as larger black-edged dots. A dashed line marks the overall median and a dotted line the reference at 1.

## fn:_model_xticks

Writes the model short names as rotated x tick labels in the model colors, bold for the adversarially trained models.

## assemble

A single row of four axes at two-column width, with the schematic widest and set to equal aspect. Panels are drawn in order a to d, spines trimmed on the three data panels, and then titles and letters are placed at a common height, each letter positioned just left of its title's measured extent.

## step:1

Unpacks the cache: DM and DOM per sweep, the sorted off-axis spectrum, the model, monkey and unit labels, the off-axis dimension count and the representative-channel exemplars.

## step:2

Computes the per-dimension ratio, the medians of the raw and per-dimension ratios and the fraction of sweeps in which DM exceeds the largest off-axis displacement, then creates the figure and draws all four panels.

## step:3

Trims spines on panels b to d, draws the canvas, and writes each panel's title and letter at a shared baseline.

## step:4

Saves the figure and prints the per-dimension median ratio, the raw DM:DOM median and the percentage of sweeps in which the encoding axis is the most-modulated dimension.

## closing

The raw DM:DOM ratio is below 1 only because DOM pools 749 dimensions; the per-dimension ratio in panels b and d sits far above the reference at 1 for every model, including Untrained. In panel c, compare the orange point against the left end of the blue curve: the encoding axis typically outstrips even the single most-modulated off-axis dimension.
