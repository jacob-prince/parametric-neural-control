## intro

Figure 2 introduces feature accentuation: take a natural seed image and nudge it along a site's encoding axis so that the model's predicted firing goes down or up in graded steps. Panels a and b are hand-drawn schematics from `figures/assets` (how synthesis works, and what axis-aligned accentuation means in feature space). Panel c shows one real sweep for a V4 site of Monkey V under RN50-Robust, from suppressed on the left through the seed in the middle to driven on the right. Panel d shows the entire accentuation dataset as a 3D point cloud in a shared readout basis: encoding-axis projection vertically, two residual PCs horizontally, with the natural calibration images as a gray cloud. Panel e shows example sweeps for one site per cortical area (aIT, cIT, V3, V4, STS) and four models.

Panels c and e read the raw synthesized images from `DATA_ROOT/stimuli_control`. Panel d reads a preprocessed cache through the small helper module `fig2_data.py` next to this script; its only job is `load_cloud`, which unpickles `preprocessed_data/fig2_cloud.pkl` (built by `scripts/preprocessing/build_fig2_cloud.py`, which aligns every monkey, unit and model combination into the shared basis and pools them).

## setup

`C_MONKEY`, `C_UNIT`, `C_MODEL`, `C_SEED` choose the single sweep in panel c, and `C_SUPPRESS` and `C_DRIVE` are the level indices shown on either side of the seed (four each). For panel e, `E_MODELS` lists the four models (rows), `LEVEL_PICK` the four level indices (columns), and `GRID_SITES` the five sites with their grid position, region label and seed. `THUMB`, `BORDER` and `GAP` are pixel sizes for the pasted mosaics, `CMAP` is the red-blue border colormap, and `CYAN` is the encoding-axis color shared with Figure 3. Swapping entries in `GRID_SITES` or `E_MODELS` is the easiest way to show other sites or models.

## blocks

The helpers come in three groups: image utilities that locate and load accentuated stimuli (`_accent_dir`, `_floor`, `load_sweep`, `_thumb`, `_bordered`), the two mosaic builders for panels c and e, and the drawing helpers for panel d (`draw_cloud` and the `_pc_widget` basis glyph).

## fn:_accent_dir

Builds the path to one monkey and model's accentuation directory from the date-prefix and monkey-directory tables in `pnc.utils`.

## fn:_floor

Returns the site's firing floor in z units from the loader; achieved scores are clamped at this floor so border colors never encode a predicted rate below zero spikes.

## fn:load_sweep

Lists the accentuated images for one monkey, model, unit and seed, parses the target level and achieved score from each filename, skips hyperparameter-tuning files, clamps the score at the floor, and returns the sweep sorted from lowest to highest target.

## fn:build_sweep_strip

Assembles panel c as a single PIL strip: the chosen suppress levels, the seed image (`figures/assets/seed.png`) with a yellow border, then the chosen drive levels. Border colors are normalized over the achieved scores of the shown levels only, so the coldest and warmest borders are the extremes of this strip. Returns the strip and the number of suppress and drive levels actually found.

## fn:build_mosaic

Builds one panel-e mosaic for a site: rows are the four `E_MODELS`, columns the four `LEVEL_PICK` levels. Border colors are normalized over all sixteen shown scores, so colors are comparable within a mosaic but not across mosaics.

## fn:_pc_widget

Draws the three-arrow readout-basis glyph (encoding projection up, residual PC2 diagonal, residual PC1 right) in a physically square inset, sized in inches so the arrows are not distorted by the panel's aspect ratio.

## fn:draw_cloud

Draws panel d on a 3D axes. The natural-image cloud is shifted so its centroid coincides with the centroid of the accentuated points, then plotted in gray; the accentuated points are colored by predicted level with a diverging norm centered at zero. Axis limits are set from the 0.5 to 99.5 percentiles of all points with a small margin, the box is stretched vertically, the view is nearly edge-on, and all axis furniture is hidden. Returns the colormap and norm so the assembly can draw a matching colorbar.

## assemble

The figure is a tall canvas whose bottom half is a 2 by 3 grid (`de`): cell (0, 0) holds the 3D cloud and the other five cells hold the region mosaics. The top half is placed by hand: the two schematics as free axes (b overlaps a and sits in front), and the wide sweep strip below them. All annotation is done after a canvas draw so it can read the actual axes positions.

## step:1

Creates the figure and grid, places the two schematic images with their titles, builds and displays the panel-c strip, draws the cloud into the top-left grid cell, and fills the remaining cells with one `build_mosaic` call per entry in `GRID_SITES`.

## step:2

Draws the canvas to fix positions, then adds every label: the "decreased / seed / increased firing" captions under the strip, the panel-d title with the dataset size, its colorbar, the vertical cyan "encoding axis" text and arrow, the "natural image distribution" tag, the basis glyph, the panel-e title, region labels in each monkey's color, model labels in each model's color rotated beside the rows, and the panel letters.

## step:3

Saves the figure through `save_fig`, closes it, and prints how many suppress and drive levels made it into the strip.

## closing

Follow the border colors: within every strip and mosaic they run from blue (suppressed) through the yellow seed to red (driven). In panel d the colored accentuated points extend well beyond the gray natural cloud along the vertical encoding axis, which is the whole point of the design.
