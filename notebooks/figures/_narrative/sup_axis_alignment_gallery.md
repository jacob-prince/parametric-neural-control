## intro

Supplementary Figure S10 is the visual companion to the axis-alignment statistics: it shows accentuation sweeps for example channels projected into a three-dimensional read-out-aligned embedding, where the vertical direction is the fitted encoding axis and the horizontal and depth directions are the top two principal components of the residual off-axis space. Each cell shows one channel of one model with all ten seed sweeps drawn over the natural calibration images. The top two rows show the most representative channel per model, the bottom two rows the most axis-aligned channel per model. The projected coordinates, hulls and predicted responses come precomputed from `loader.load_axis_alignment()`.

## setup

`AXIS_TEAL` is the encoding-axis color shared with Figure 3. `top5` and `bot5` split `MODEL_ORDER` into the two rows of each block, and `TOPC` fixes where the top of each cell's content sits inside its window so the top dots of a row line up.

## blocks

`exemplar_bounds` and `draw_exemplar` handle one cell; `draw_exemplar_row` lays out five cells at a shared height; `place_row_names` writes model names above them; `_teal_axis` draws the small encoding-axis arrow in each cell.

## fn:_teal_axis

Draws a short vertical teal arrow from the origin of the cell, with a dot at its base, as the in-panel reminder of the encoding-axis direction.

## fn:exemplar_bounds

Returns the bounding box of everything to be drawn in one cell: the calibration cloud, all trajectory points and the seed positions.

## fn:draw_exemplar

Draws one cell: the calibration images as gray dots inside a shaded convex hull, then each seed's sweep as a thin black polyline with dots colored by predicted firing rate on a red-blue scale normalized to the calibration cloud, white diamonds at the natural seed positions, and the teal axis marker. Axes are equal-aspect and hidden.

## fn:draw_exemplar_row

Places five square cells at one height. Each window is sized to its own content (with 14% padding) but anchored so the content top sits a fixed fraction below the window top, which is what keeps the rows visually aligned despite different scales.

## fn:place_row_names

Writes each model's short name centered above its cell, in the model color and bold for the adversarially trained models.

## assemble

The canvas is a 4 by 5 grid built by hand with `fig.add_axes` so cells can be physically square: two rows for the representative channels, a larger gap, then two rows for the most-aligned channels. A rotated tag on the left names each two-row block, and a compact key beneath the title explains the three axes of the projection.

## step:1

Loads the cache, pulls out the two exemplar dictionaries (representative and most-aligned) and splits the model list into two rows of five.

## step:2

Computes the cell geometry, draws the four rows, adds the title, places the model names above each row's aligned content top, writes the two rotated block tags and draws the axis key (teal vertical arrow for the encoding axis, gray arrows for the horizontal and depth off-axis components).

## step:3

Saves the figure.

## closing

Read each cell vertically: the sweeps climb from blue at the bottom to red at the top along the encoding axis, while the calibration cloud stays comparatively flat. The bottom block shows the cleanest cases; the top block shows what a typical channel looks like for each model.
