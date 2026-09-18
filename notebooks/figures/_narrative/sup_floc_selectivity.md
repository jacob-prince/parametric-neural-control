## intro

Supplementary Figure S4 characterizes what each of the 25 sites likes, using the fLoc localizer categories (faces, bodies, objects, scenes, characters, scrambled). Panel a gives one small panel per site, arranged as five rows (animals) by five columns (sites): the density of calibration natural-image responses in gray, with the six category responses overlaid as jittered dots and each category's mean as a triangle below the axis. The preferred category (largest mean) is highlighted and named in the panel title. Panel b distils each site to a single bar: the d-prime of its preferred category against all other categories, colored by category, with a Welch t-test star above and the animals bracketed by area below. Everything comes from `loader.load_floc_selectivity()`.

## setup

`MONKEYS` fixes the row order (aIT and cIT first, then V3/V4 and the two STS animals), with `MONKEY_TAG` and `MONKEY_REGION` for labels. `DOMAINS` and `DOMAIN_COLORS` fix the category order and colors used in both panels and the legend. `FIGHEIGHT_MM` sets the page height.

## fn:domain_means

Mean fLoc response per category present in the data, ignoring NaN.

## fn:preferred

The category with the largest mean.

## fn:pref_and_stats

Returns the preferred category, its d-prime against the pooled responses of every other category (difference of means over the pooled standard deviation), and the p-value of a Welch t-test of preferred versus other.

## fn:sig_stars

Converts a p-value into one, two or three asterisks at the usual thresholds, or an empty string.

## fn:draw_panel

Draws one site's in-context panel. The natural-image density is a Gaussian kernel estimate normalized to a peak of one; each category's responses are placed at a fixed height with a little vertical jitter; the preferred category gets larger dots with black edges; and the mean triangles sit just below the axis. Only the y-axis is hidden, so the response scale stays readable.

## fn:draw_summary

Draws panel b: one bar per site grouped by animal with a gap between animals, stars above significant bars, tick labels in each animal's color, a p-value key in the corner, and colored region brackets under the tick labels.

## assemble

The figure is two-column width with a 6 by 5 GridSpec: five rows of five site panels, then a taller sixth row spanning all columns for the summary. The title, the shared legend and the two panel letters are placed after the axes exist.

## step:1

Loads the fLoc selectivity data.

## step:2

Creates the figure and the GridSpec with the summary row weighted taller than the site rows.

## step:3

Fills the five by five grid with `draw_panel`, showing the x-axis label only on the last row and putting the animal and area label on the first panel of each row.

## step:4

Draws the summary panel across the bottom row, adds the title from the manifest and a legend (natural-image density, the six categories, and the preferred-domain marker), and places the panel letters with b aligned to the top of the summary panel.

## step:5

Saves the figure and prints the path.

## closing

Compare rows: the aIT and cIT sites, which were targeted to face patches, mostly show "Faces" in their titles and tall, starred red bars in panel b, whereas the V3/V4 and STS rows are more mixed in both preferred category and selectivity strength.
