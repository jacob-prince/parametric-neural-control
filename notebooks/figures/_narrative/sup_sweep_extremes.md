## intro

Supplementary Figure S11 shows the two ends of the accentuation sweep, the strongest suppress and strongest drive images, for every one of the 25 recorded sites under four representative models (ResNet50, DINOv2, RN50-Robust and CLIPAG). The 25 sites are split into two side-by-side blocks so the image cells stay large, and the seed image rotates across (site, model) pairs so the whole seed set is sampled. Image pixels are read straight from the accentuated PNGs under `stimuli_control`; only the site list comes from the loader.

## setup

`MODELS` names the four models shown, `DIRPREF` maps each monkey to its accentuation directory prefix, and `THUMB` (224 px) is the cell resolution. `C_SUPP` and `C_DRIVE` are the two ends of the red-blue level colormap used for image borders, and `MISSING` collects any (site, model) pairs with no files so a blank cell is drawn instead of failing.

## fn:acc_dir

Builds the accentuation directory path for one monkey and model from `DIRPREF`.

## fn:extreme_files

Globs the PNGs for one (model, unit, seed), parses target level and achieved score from each filename, and returns the lowest-level and highest-level file. Where a level was re-synthesized and has duplicates, the copy whose achieved score is closest to its target is kept. Returns a pair of None if nothing is found.

## fn:thumb

Loads a PNG as a 224-pixel RGB array, or a flat light-gray placeholder when the path is None.

## assemble

A single large GridSpec holds both blocks: rows are the site slots of three monkey groups per side (5, 5 and 3) with thin spacer rows between groups, and columns are four models times two sub-columns (S and D) per side with spacer columns and a wider middle gap. Cells are filled block by block, then labels for channels, monkey regions and models are written into the figure margins.

## step:1

Reads the channel table, lists the units per monkey, defines the two side blocks (Monkey V's five sites are split three left and two right to balance the page), and builds the row and column ratio lists plus lookup tables from (block, site) and (model, S/D) to grid indices.

## step:2

Creates the figure at two-column width and about 0.79 of that in height, with the shared GridSpec.

## step:3

Loops over sides, monkey blocks, sites and models, advancing a seed counter each time so the seed cycles through all ten. For each pair it fetches the extreme files, draws the suppress and drive thumbnails with blue and red borders, and adds S and D headers on the top row, channel tags at the left of each row, a rotated monkey and region label beside each block, and model names above each side.

## step:4

Writes the figure title from the manifest.

## step:5

Saves the figure and prints any missing (site, model) pairs.

## closing

Scan a row to compare how the four models push the same site toward its extremes, and scan a column to see one model's suppress and drive images across sites. The bold model names mark the adversarially trained models.
