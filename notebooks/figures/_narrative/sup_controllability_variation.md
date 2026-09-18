## intro

Supplementary Figure S19 asks how much controllability varies from site to site. For each of the 25 recorded sites it shows the control r of all ten models as faint dots and the site mean as a colored bar, with sites sorted from least to most controllable and colored by the animal and its cortical area. A dashed line gives the grand mean and a bracket at the right spans the min-to-max range of the site means. The per-site, per-model control r values come from `loader.control_table()`.

## setup

`MONKEY_AREA` and `MONKEY_TAG` label each animal with its area and one-letter tag; `MONKEY_ORDER` fixes the legend order. Coloring is by monkey rather than by area so the two STS animals stay distinguishable.

## fn:build

Groups the control table by site, keeps the finite per-model r values, computes each site's mean and returns the sites sorted by mean along with the array of means.

## fn:draw

Draws the sorted sites: jittered per-model dots in the animal's color at low alpha, a thicker bar at the site mean, and a dashed grand-mean line stopping at the right edge of the data. It then adds the range bracket beyond the last site with its label, the SD (sample SD of the site means) beneath it, colored bold site tags on the x-axis and an area legend, and returns the SD, grand mean and range.

## assemble

A single axis placed by hand fills a two-column-width figure. `draw` does all the plotting; the manifest title is written above the axis after a canvas draw.

## step:1

Builds the sorted site list, creates the figure and axis, and draws the panel, keeping the returned summary statistics.

## step:2

Draws the canvas and centers the title above the axis.

## step:3

Saves the figure and prints the SD, grand mean, range and number of sites.

## closing

Follow the colored bars from left to right: the low end is dominated by one area and the high end by others, which is the point of the figure. The bracket and SD label at the right give the spread in one glance.
