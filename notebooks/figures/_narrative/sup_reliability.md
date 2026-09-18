## intro

Supplementary Figure S2 documents how the 25 targeted electrodes were chosen and how trustworthy their responses are. Panel a plots the Spearman-Brown corrected split-half reliability of every channel on each animal's array or probe, with the five selected sites highlighted. Panel b converts each selected site's reliability into a calibration noise ceiling, the highest correlation any model could reach. Panel c checks that tuning is stable across days within the calibration phase and across the calibration and control phases. Panel d shows the 5 by 5 tuning-correlation matrix among the selected channels per animal, to confirm they are reliable but not redundant. Panels a, b and d read the channel-selection cache (`preprocessed_data/sup_reliability_channel_selection_cache.pkl`, built from the raw encoding files because the all-channel data are not in the standard preproc cache); panel c reads `loader.load_tuning_stability()`.

## setup

`MONKEYS` sets the left-to-right order, with `TAG` and `REGION` providing the one-letter animal codes and area labels used in tick labels. `CMAP` for panel d is a purple-orange diverging map, deliberately not the red-blue reserved elsewhere for accentuation levels. `W` is the bar width for the paired bars in panel c.

## fn:calibration_ceilings

For each selected channel, takes the square root of its clipped split-half reliability as the calibration-phase noise ceiling. This uses exactly the reliabilities plotted in panel a, so every animal has a valid ceiling here (control-phase ceilings are a separate supplementary figure).

## fn:load_channel_selection

Unpickles the channel-selection cache; the error hint names the script that builds it.

## fn:panel_a

Per animal, scatters all finite channel reliabilities in light gray, then the five selected sites in the animal's color, and draws a dotted line at the lowest reliability among the selected five.

## fn:panel_b

One bar per selected site grouped by animal, with a hatched placeholder and an "n/a" tag for any site whose ceiling is undefined. Draws a dashed line at the mean ceiling and returns that mean.

## fn:panel_c

Paired bars per site: the cross-day correlation (hatched, translucent) and the cross-phase correlation (solid), with a legend explaining the two fills.

## fn:panel_d

Five small heatmaps, one per animal, of the tuning correlation among the selected channels, with each value printed in the cell (white text on strong colors) and the diagonal outlined. Returns the last image handle for the shared colorbar and the list of axes.

## assemble

The figure is two-column width and 198 mm tall, a four-row GridSpec with panels a, b and c each taking a full row and the row of five heatmaps at the bottom. Panel titles and letters are added after a canvas draw so they can be placed relative to the actual axes.

## step:1

Loads the tuning-stability data, the channel-selection cache, and derives the per-site calibration ceilings.

## step:2

Creates the figure and GridSpec, draws panels a, b and c into the first three rows, draws the heatmap row, and nudges the heatmaps down slightly so their titles clear panel c's tick labels.

## step:3

Hides the top and right spines of the three bar-and-scatter panels.

## step:4

Draws the canvas, adds a colorbar to the right of the heatmap row, then places a centered title and a bold letter above each of the four panels.

## step:5

Saves the figure, closes it, and prints the mean calibration noise ceiling.

## closing

In panel a the highlighted sites sit at the top of each gray cloud, and the dotted line marks how far down the selection reached. Panel b's dashed line is the mean ceiling across all 25 sites (0.88 in the caption). In panel d, off-diagonal values well below one mean the selected channels have distinct tuning.
