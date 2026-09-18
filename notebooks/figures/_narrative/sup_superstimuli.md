## intro
This figure asks how often accentuation pushed a site beyond anything seen during calibration, and which models did it most. A super-stimulus is defined per site: an accentuated image whose measured control response lands above the 99th or below the 1st percentile of that site's natural-image responses. The data are the calibration responses (to set the percentiles) and the control-phase clouds for every site-model. Panel A is a schematic of the definition, panel B ranks the models by their super-stimulus proportion, and panel C splits that proportion into the drive and suppress tails.

## setup
`ROBUST` holds the two adversarially trained models, which are bolded on the axes. `C_DRIVE` and `C_SUPP` are the red and blue used for the upper and lower tails throughout. The percentile cut-offs themselves (1 and 99) are written directly in `build`.

## blocks
`build` produces one tidy table; the three `panel_*` functions each draw one axes from it.

## fn:build
For every site, it takes the natural (calibration) measured responses under the first model in the config, which is enough because the measured distribution does not depend on the model, and computes the 1st and 99th percentiles. Then for each model it loads the control cloud and records the percentage of that model's accentuated stimuli above q99 (drive), below q01 (suppress), and their sum. Site-models with an empty control cloud are skipped.

## fn:panel_a
Draws a standard normal density purely as a schematic, shades the two tails beyond the 1st and 99th percentiles in the tail colors, and labels the gray middle as the natural range. No data enter this panel.

## fn:panel_b
Horizontal bars of the per-model mean super-stimulus percentage, ordered from lowest to highest, with the individual site values jittered on top as small dark dots. Model names are colored by model and the adversarially trained ones are bold.

## fn:panel_c
Paired vertical bars of the mean super-drive and super-suppress percentages per model, ordered by total proportion from highest to lowest, so you can see whether a model's excess comes from driving, suppressing, or both.

## assemble
The canvas is a single row of three panels of slightly different widths. Data are built first and a short summary is printed, then the panels are drawn left to right, then titles and letters are placed after a draw pass so they sit relative to the final axes positions.

## step:1
Calls `build`, then prints the per-model mean super-stimulus percentage, the means for the adversarially trained versus other trained models, and a check that the top two models are the two adversarially trained ones.

## step:2
Creates the figure and a one-by-three grid, and fills the schematic, the ranked proportion bars and the drive-versus-suppress decomposition.

## step:3
Forces a draw so axes positions are known, then writes a centered title above each panel and a bold letter at its top-left.

## step:4
Saves the figure, closes it, and sets `png` to the output path.

## closing
In panel B the two bold models should sit at the top with the Untrained baseline at the bottom, and the spread of site dots shows how variable the proportion is across sites. Panel C tells you whether that lead is carried mostly by super-drive or shared with super-suppress.
