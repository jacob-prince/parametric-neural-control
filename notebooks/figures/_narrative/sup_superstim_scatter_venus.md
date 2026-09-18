## intro
This figure shows the most extreme super-stimuli for the V3/V4 sites of Monkey V, both as images and as points in the predicted-versus-measured plane. Super-stimuli are accentuated images whose measured response fell outside the site's natural calibration range, with at least three trial repeats so the estimate is not a single noisy trial. The left half is a ten-by-ten mosaic of the strongest cases, split into super-drive at the top and super-suppress at the bottom; the right half plots the same stimuli against the gray calibration cloud. Everything is read from a preprocessed cache; the script only lays it out.

## setup
`CACHE` points at the per-stimulus table and calibration cloud built by the preprocessing script named in `BUILD_HINT`. `MIN_REPS` is the repeat threshold, and `N_DRIVE_DS` and `N_SUPP_DS` set how many drive and suppress stimuli fill the mosaic (they sum to the hundred cells). `MOSAIC_BOX` and `SCAT_RECT` fix the two halves in figure fractions, and `UL_SLOTS` and `LR_SLOTS` are the pre-chosen positions for the thumbnail callouts in the scatter's empty corners.

## blocks
The helpers work from the outside in: `_relocate` fixes file paths, `_load_img` shows one image, `draw_mosaic` fills the left half, `draw_scatter` and `add_callout` fill the right half, and `render` selects the stimuli and calls the rest.

## fn:_relocate
The cache stores absolute paths from the machine that built it. This re-roots each path under the current stimulus directory, keeping only the folder and file names.

## fn:_load_img
Displays an image in an axes with ticks removed, or fills the axes light gray if the file is missing, so a missing image is visible but does not stop the render.

## fn:draw_mosaic
Places a hundred small axes in a ten-by-ten grid inside `MOSAIC_BOX`. When `split_after` is given, it leaves a gap after that many rows, draws a separator line, and labels the upper block super-drive and the lower block super-suppress. Each tile's border takes the color of the model that generated the image.

## fn:add_callout
Adds one thumbnail as an inset at a given slot, borders it in the model color, and draws a dashed leader from the stimulus's point in the scatter to the thumbnail center.

## fn:draw_scatter
Plots the gray calibration cloud with SEM whiskers and a dotted identity line, then the super-stimuli as larger model-colored dots with their own whiskers. When `suppress` is true the y-range is extended downward so the suppress points have room. It then calls out the highest-measured drive stimuli to the upper-left slots and the lowest-measured suppress stimuli to the lower-right slots, and returns the list of models present for the legend.

## fn:render
Filters the table to stimuli with enough repeats, takes the top drive cases by measured response and the bottom suppress cases, builds the mosaic item list in that order, draws both halves, adds the monkey label and a model legend, and saves.

## assemble
`main` is a single step: load the cache, fix the image paths, and hand the calibration cloud and stimulus table to `render` with the drive-plus-suppress layout enabled.

## step:1
Loads the pickle, re-roots every image path with `_relocate`, and renders the figure; `png` receives the saved path.

## closing
Notice that the calibration cloud hugs the identity line while the super-stimuli sit well outside it, above for drive and below for suppress. The mosaic borders and the dot colors share the model palette, so you can read off which models produced the extremes.
