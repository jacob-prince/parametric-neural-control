## intro

Figure 1 is the overview of the closed loop: record responses to a calibration image set, fit an encoding axis per model and site, synthesize accentuated stimuli along that axis, and present them back to the animal. The loop itself is a hand-drawn schematic (`figures/assets/schematic_fig1.png`). This script does not draw any data; it composites two data ingredients onto that schematic: a cloud of the 969 calibration images laid out in ResNet50 feature space, and a tall grid of accentuated stimuli that shows what "walking along the encoding axis" looks like for one site.

Both ingredients have their own builder modules next to this script in `figures/main/`. `fig1_grid.py` renders the accentuation grid on the fly: seven feature levels (rows, drive at the top, suppress at the bottom) for three seed images (columns), from the RN50-Robust accentuations of Monkey P unit 8, with borders colored by achieved response and the seed row in light yellow. `fig1_cloud.py` renders the calibration-image cloud as thumbnails in PC1 by PC2 of ResNet50 avgpool features. The cloud PNG shipped in `preprocessed_data` is a frozen input, though: it came from an older version of `fig1_cloud.py`, and this script reads the frozen file rather than calling the builder. Rerunning `fig1_cloud` gives a visually equivalent but not bit-identical cloud.

## setup

The constants place the three layers in figure fractions on the wide `CANVAS`: `SCHEM_BOX` centers the square schematic, `CLOUD_BOX` pushes the cloud into the right margin, and `MOSAIC_CX`, `MOSAIC_CY`, `MOSAIC_H` fix the center and height of the grid on the left (its width is derived later from the rendered aspect ratio). `WHITE_THR` is the near-white threshold used twice: to make the schematic's background transparent and to crop the grid's white margins. To move a layer, edit its box; to change which site the grid shows, edit `UNIT`, `SEED_IMG_IDS` and `PER_SEED_LEVELS` inside `fig1_grid.py`.

## fn:keyed_schematic

Loads the schematic as RGBA and sets alpha to zero wherever all three color channels are at or above `WHITE_THR`. The effect is that the cloud shows through the blank paper of the schematic while the line art, arrows, markers and text stay opaque on top of it.

## fn:add_image

Adds a borderless axes at the given figure-fraction box, draws the image into it, and hides both the axis decoration and the axes background patch. The `zorder` argument is what stacks the three layers.

## assemble

The composite is a stack of three full-bleed image axes ordered by `zorder`: the cloud at the bottom, the color-keyed schematic in the middle, and the opaque accentuation grid on top. The grid's box is computed from its cropped pixel dimensions so it keeps its native aspect ratio at the requested height; the conversion multiplies by the canvas aspect because figure fractions are not square.

## step:1

Rebuilds the accentuation grid by calling `fig1_grid.main`, writing it as a temporary PNG in the output directory, and checks that the frozen cloud PNG exists (`paths.require` raises with a hint if it is missing).

## step:2

Opens both ingredients, crops the grid to its non-white content (the white margins are opaque and would otherwise cover schematic text), and derives `MOSAIC_BOX` from the cropped width-to-height ratio.

## step:3

Creates the canvas and stacks the three layers with `add_image`: cloud, keyed schematic, then grid.

## step:4

Saves the composite at 200 dpi without tight cropping. The ingredient builders applied the shared figure style, which sets a tight save bbox, so the script resets that setting first so the canvas dimensions are preserved.

## closing

In the rendered figure, look for the calibration cloud sitting behind the phase-1 recording icon on the right and the accentuation grid on the left, with prediction increasing from bottom to top. Only the grid is regenerated on each run; if the frozen cloud is missing, `fig1_cloud.py` can produce a substitute that looks the same but is not pixel-identical to the manuscript render.
