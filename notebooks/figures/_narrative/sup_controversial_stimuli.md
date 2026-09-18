## intro
This figure simply shows every stimulus from the controversial-accentuation experiment so a reader can inspect them. There are 140 images: seven aIT sites in Monkey R, ten seed images each, and two target directions. The top mosaic holds the ResNet50-favoring images and the bottom mosaic the ResNet50-Robust-favoring ones, with rows indexing sites and columns indexing seeds. Structure comes from the loader's controversial table; the images are raw experiment inputs on disk.

## setup
`STIM_DIR` is the folder of raw stimulus PNGs. `C_R50` and `C_ROBUST` take the two models' colors from the shared palette for the tile borders and headers. `NIMG` is the number of seed columns, and `SAT_BOOST` raises image saturation slightly for display.

## blocks
Only two helpers: `resolve` finds a file and `square` prepares it for display. The `block` function that fills one mosaic is defined inside the assembly.

## fn:resolve
Maps a stimulus basename to its path in `STIM_DIR` and raises immediately if the file is missing, so an incomplete set is caught before anything is drawn.

## fn:square
Opens an image, boosts its saturation by `SAT_BOOST`, and center-crops it to a square array.

## assemble
The figure height is computed from the tile width so that every tile is square and the two mosaics stack vertically to fill a portrait page, with fixed millimeter bands reserved for the title, the mid header and a bottom margin. Data are checked first, the two blocks drawn, then title and headers placed.

## step:1
Loads the controversial table, extracts stimulus names, sites and seed ids, flags each image as ResNet50-favoring or Robust-favoring from its name, asserts there are seventy of each, and resolves every image path up front.

## step:2
Defines `block`, which fills one mosaic with tiles ordered by seed id, colored borders, a site label at the left of each row and a column number above the first row. It then computes the page geometry, builds the two nested grids, and draws the top block with the ResNet50 mask and the bottom block with the Robust mask.

## step:3
Writes the figure title near the top edge and a colored header above each mosaic.

## step:4
Saves and closes the figure, prints the stimulus and site counts, and sets `png` to the output path.

## closing
Scan down a column to see how the same seed was pushed in opposite directions for different sites, and compare the same cell in the two mosaics to see what the two models disagree about. Borders and headers use the same orange and green as the results figures.
