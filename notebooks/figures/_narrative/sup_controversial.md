## intro
The controversial-accentuation experiment forces two encoding models to disagree. For each of seven aIT sites in Monkey R, images were synthesized to drive ResNet50's predicted response while suppressing ResNet50-Robust's, and vice versa, then shown back to the animal. Panel A is a procedure schematic, panel B shows example stimuli for four sites around a prediction-space scatter, panel C plots model prediction against measured response for three sites, and panel D summarizes the per-site correlations for both models. Neural responses are standardized trial averages from the loader's controversial table; the stimulus images are raw experiment inputs.

## setup
`C_R50` and `C_ROBUST` are the orange and green used for the two models throughout, with faded variants for stimuli made for the opposing model. `B_UNITS` picks the four example sites for the mosaics, `C_UNITS` the three for the scatter row, and `ALL_UNITS` the seven that enter the summary. `K_MOSAIC` sets how many seed images each mosaic shows. The canvas is a fixed 180 by 150 mm and is saved without a tight bounding box.

## blocks
`_df` and `_img` load data and images; `draw_outcome` is a tiny schematic helper used by `panel_a`; the other `panel_*` functions each draw one lettered panel.

## fn:draw_outcome
Draws a two-bar cartoon of a predicted response, one bar per model, with HIGH and LOW labels over the taller and shorter bar. It illustrates a stimulus goal, not data.

## fn:_df
Flattens the controversial table into one row per stimulus: site, seed image id, which model it was made to favor (from the stimulus name), the two target-model scores used during synthesis, the two Lasso encoder predictions used for evaluation, the neural response with SEM, and the local image path.

## fn:_img
Opens an image, converts to RGB and center-crops it to a square; returns nothing on failure so a missing file leaves an empty tile.

## fn:panel_b
Chooses the shared seed columns by ranking seeds on mean controversy (how strongly each favors its intended model, averaged over both directions and the four example sites), so the two mosaics correspond seed for seed. Draws the ResNet50-preferring mosaic on the left and the Robust-preferring mosaic on the right with colored borders and site labels, and between them a scatter of every stimulus's ResNet50 score against its Robust score, circles for one direction and squares for the other, with an identity line for reference.

## fn:panel_c
For each of the three example sites, plots both models' Lasso predictions against the measured response with SEM whiskers. Full color marks stimuli made for the plotting model's own family and faded color the opposing family; a least-squares line per model and the two correlations are boxed in the corner.

## fn:panel_d
For each of the seven sites, computes the Pearson correlation between the neural response and each model's predictions over all twenty stimuli, draws them as paired points joined by a gray line, adds mean diamonds and a Wilcoxon signed-rank bracket with stars. Site labels on the right are spread apart to avoid overlap.

## fn:panel_a
Places the two stimulus-goal cartoons from `draw_outcome` on either side of the procedure schematic image, nudging the image so it does not collide with the rotated model labels.

## assemble
Five explicit horizontal bands set the layout: panel A, a small gap, panel B, a header band, and the bottom row shared by C (wider) and D. Panels are drawn first, then letters, section headers and the shared marker legend are placed after a draw pass.

## step:1
Loads the table, builds the banded grid, and draws panels A, B, C and D, reserving the bottom of the C and D cells for the legend.

## step:2
After a draw pass, places the four letters, writes the headers above C and D, and adds a four-entry legend under panel C explaining the marker and shade combinations.

## step:3
Saves at the exact canvas size with no tight bounding box; `png` receives the path.

## closing
In panel B the scatter should split cleanly into two off-diagonal clusters, showing the synthesis worked. In panels C and D look at the sign of the correlations: the green Robust predictions should track the neural response positively while the orange ResNet50 predictions run negative, which is what the summary's Wilcoxon test formalizes.
