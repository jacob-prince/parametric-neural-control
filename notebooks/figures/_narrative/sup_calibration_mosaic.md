## intro

Supplementary Figure S1 shows every one of the 969 calibration images used to fit the encoding models, so a reader can see what the models were trained on and which images drove IT most. The mosaic is split into three blocks by image source (NSD natural scenes, segmented objects and animals on white, and fLoc localizer images). Within each block the thumbnails are sorted from strongest to weakest mean response across the five aIT sites of Monkey R, and each thumbnail carries a red-blue border encoding that response. Image membership and source flags come from `loader.load_stimuli()`, responses from the Monkey R brain cache, and the thumbnails from `DATA_ROOT/stimuli_encoding`.

## setup

`THUMB`, `BORDER`, `OUTLINE` and `GAP` set the pixel geometry of one cell, and `COLS` the number of thumbnails per row. `SOURCES` lists the three blocks as (flag column, header label, expected count, header color); the counts are asserted at run time and must add up to 969. `BASE_CMAP` is the red-blue map, later re-centred on the median rate.

## fn:stretched_cmap

Remaps a diverging colormap so its white midpoint lands at a chosen fraction of the axis instead of at one half. This lets the colorbar keep an even linear scale from zero to the maximum while the color neutral point sits at the median firing rate.

## fn:firing_rates

Recovers firing rate in spikes per second for every calibration image by inverting the cached per-unit z-scores with each unit's mean and standard deviation, then averages across the five Monkey R sites.

## fn:resolve_paths

For each source flag, matches the basename of every image path in the stimulus table against the files on disk, attaches the image's mean rate, and sorts high to low. Anything not found on disk is returned in a `missing` list.

## fn:load_thumb

Downsamples one image to the thumbnail size and pads it to a square on white so the grid stays regular for non-square sources.

## fn:compose

Paints one block as a row-major array: a thin dark outline, then the firing-rate colored border, then the thumbnail, for each item. Returns the pixel array and the number of rows it occupies.

## assemble

The figure is two-column width and 232 mm tall, with one GridSpec row per source block. Row heights are proportional to each block's row count plus a fixed header allowance, so the three blocks share the same thumbnail scale. The title and a horizontal colorbar sit in the band above the first block.

## step:1

Computes the per-image rates, resolves the file paths per source, and exits if any image is missing. Sets up the color scale: a linear norm from zero to a rounded 98th-percentile maximum, and the colormap stretched so white falls at the median rate.

## step:2

Composes the three blocks in `SOURCES` order, asserting that each block has the expected count and that the total is 969.

## step:3

Creates the figure and the GridSpec with height ratios derived from each block's row count.

## step:4

Draws each block as an image with all axis furniture hidden, and adds the source label on the left and the count on the right above it, in the block's color.

## step:5

Adds the figure title from the manifest and the horizontal firing-rate colorbar, with ticks every ten spikes per second and an extension arrow for values above the maximum.

## step:6

Saves the figure, closes it, and prints the block counts and output path.

## closing

Look at the leading edge of the segmented-object and fLoc blocks: the warmest borders cluster there and they are mostly human faces, consistent with the face-patch targeting of Monkey R's array. The NSD block is far more uniform in color.
