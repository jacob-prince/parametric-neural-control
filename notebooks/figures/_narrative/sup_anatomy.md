## intro

Supplementary Figure S3 shows where the electrodes are. It stacks three raster rows of intra-operative CT and structural MRI sections: Monkey R's chronic microwire array in anterior IT, Monkey V's Neuropixels track in V3/V4, and Monkey T's Neuropixels track in STS. Arrowheads in the images mark the array tip or the probe entry and target. There is no data processing here; the script lays out two image assets from `figures/assets/anatomical` at their native aspect ratios.

## setup

`RED` and `COMPOSITE` are the two source images; the composite holds Monkey V on its top half and Monkey T on its bottom half. `GUTTER_MM` reserves a left strip for the panel letter and region title, `IMG_W_MM` is the remaining image width, and `GAP_MM` with the two padding constants sets the vertical spacing. The figure height is not fixed; it is computed from the image aspect ratios.

## fn:load_rows

Loads the Monkey R image, splits the composite into its top (Monkey V) and bottom (Monkey T) halves, and returns the three rows as (letter, title, RGB array) tuples in display order.

## assemble

Each row is drawn at the common image width, so its height in millimetres follows from its pixel aspect ratio. The total figure height is the sum of those heights plus gaps and padding. Rows are placed top to bottom as free axes, each with its letter and a two-line region title in the left gutter.

## step:1

Loads the rows and computes each row's display height and the total figure height in millimetres.

## step:2

Creates the figure at two-column width and the computed height.

## step:3

Converts the gutter and image widths into figure fractions.

## step:4

Walks down the page: for each row, adds an axes of the right height, draws the image without interpolation inside a thin dark frame, and writes the panel letter and region title in the gutter, then steps the cursor down by the row height and gap.

## step:5

Saves the figure and prints the output path with the figure height that was used.

## closing

The figure exists to be looked at rather than measured: find the white arrowhead at the array tip in row a and the green arrowheads marking probe entry and target in rows b and c.
