## intro

Supplementary Figure S20 shows the same per-site control r as Figure S19 but emphasizes the ranking of models within each site. One point per model is drawn at each of the 25 sites, colored by model and grouped by animal and area along the x-axis; the two adversarially trained models are drawn larger with black edges, and the single best model at each site is ringed. The data come from `loader.control_table()`.

## setup

`ROBUST` holds the two adversarially trained models, `UNTR` names the untrained baseline, `AREA` and `TAG` label each animal, and `RING` is the color of the best-at-site ring.

## fn:build

Pivots the control table into a 25 by 10 matrix of control r. For each site it ranks the models and records whether an adversarially trained model is first or in the top two, and the rank of the best such model. It also computes a per-area advantage: mean r of the two adversarially trained models minus the mean of the other trained models, with Untrained excluded. These summaries are printed rather than plotted.

## fn:panel_a

Lays the sites out along x grouped by monkey with a gap between groups, jitters the ten model dots within each site, rings the best model, and draws a colored bar and label under each animal group. A faint zero line and a y-range from just below the minimum r to 1 finish the panel.

## assemble

One axis in a two-column-width figure, with a six-column model legend above it that also shows the ring marker, and a single centered title rather than panel letters.

## step:1

Runs `build` to get the models, monkeys, site list, per-site r values, per-area advantages and rank statistics.

## step:2

Creates the figure, draws the panel, assembles the legend handles (black-edged markers for the adversarially trained models) and writes the title.

## step:3

Saves the figure and prints how often an adversarially trained model ranked first or in the top two, and the per-area advantage.

## closing

Look for where the rings fall: at most sites they land on one of the two black-edged dots. Rings on other colors mark the exceptions.
