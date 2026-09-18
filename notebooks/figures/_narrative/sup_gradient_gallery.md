## intro
A reference gallery of input-gradient saliency maps: ten seed images as rows, the ten encoding models as columns, plus a left column showing the seeds themselves. Each cell is a model's gradient saliency for that seed, computed as the L2 magnitude of the gradient across colour channels, normalized per map and then averaged over the 25 recorded sites. The maps are read from the frozen `sup_gradient_gallery_avg.pkl`; seed thumbnails come from `STIMULI_PATH`, except the dog row, which reuses the schematic seed crop from `figures/assets/seed.png`.

## setup
`SEED_IMAGES` lists the seed filenames in the row order of the pickle, and `DOG_SEED_IDX` marks the row that uses the schematic crop. The colour scale is shared across every cell: `vmax` is the 99th percentile over all models' maps.

## fn:seed_thumb
Returns the thumbnail for a seed row, substituting the schematic crop only in the dog slot.

## assemble
A grid of ten rows by eleven columns filling a tall two-column canvas. Rows are filled seed by seed; the model column headers are placed as figure text after a draw so they can be centred on each column, with compound names wrapped onto two lines.

## step:1
Loads the pickle and pulls out the model list, the saliency arrays and the per-model site counts; computes the shared `vmax`.

## step:2
Creates the figure and the grid.

## step:3
For each seed row, shows the thumbnail in the first column and then each model's map in magma on the common scale, with thin uniform frames; then adds the bold, model-coloured column headers.

## step:4
Writes the figure title, including the number of sites averaged.

## step:5
Saves the figure and prints the site counts.

## closing
Because the colour scale is shared, brightness can be compared across columns: look at where each model's sensitivity lands relative to the seed content, and how that changes from the left columns to the two adversarially trained models.
