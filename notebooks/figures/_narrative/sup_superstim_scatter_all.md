## intro
This is the pooled companion to the Monkey V super-stimulus figure: the same layout, but drawing the strongest super-stimuli from all 25 sites across the five animals. Super-stimuli are accentuated images whose measured control response fell outside the site's natural calibration range, restricted to those measured with at least three repeats. The mosaic on the left shows seventy super-drive and thirty super-suppress images; the scatter on the right places them against a pooled gray calibration cloud in z units. As before, the numbers come from a preprocessed cache.

## setup
`CACHE` names the all-sites pickle and `BUILD_HINT` the preprocessing script that creates it. `MIN_REPS`, `N_DRIVE_DS` and `N_SUPP_DS` control the selection, `MOSAIC_BOX` and `SCAT_RECT` the two halves, and the slot lists fix where thumbnails are called out. The scatter rectangle starts a little higher than in the Monkey V version to leave room for the legend below it.

## blocks
Same helper set as the single-monkey figure: path fixing, image display, mosaic, callouts, scatter, and a `render` that selects and composes.

## fn:_relocate
Re-roots each cached image path under the current stimulus directory, keeping the folder and file names.

## fn:_load_img
Shows an image with ticks removed, or a light gray placeholder if the file is absent.

## fn:draw_mosaic
Lays out the hundred tiles, splits them into an upper drive block and a lower suppress block with a separator line and rotated labels, and borders each tile in the generating model's color.

## fn:add_callout
Insets one thumbnail at a slot and connects it to its scatter point with a dashed model-colored leader.

## fn:draw_scatter
Draws the pooled calibration cloud with whiskers and a dotted identity line, then the super-stimuli as model-colored dots with whiskers, extending the y-range downward for the suppress tail. The highest-measured drive cases are called out to the upper-left slots and the lowest-measured suppress cases to the lower-right slots.

## fn:render
Applies the repeat filter, picks the top drive and bottom suppress stimuli, builds the mosaic and scatter, adds the title and model legend, and saves at fixed canvas size.

## assemble
`main` loads the cache, fixes the paths, and calls `render` once with the pooled cloud and the all-sites title.

## step:1
Loads the pickle, re-roots image paths, and renders; `png` receives the output path.

## closing
Because sites are pooled, the calibration cloud is denser and the super-stimuli come from several animals and areas. Compare the mix of border colors here with the Monkey V figure to see which models dominate the extremes when every site is considered.
