## intro
This figure reports the controversial-accentuation experiment site by site. Each of the 140 stimuli (7 aIT sites in Monkey R, 10 seeds, 2 target models) was synthesized so that one encoding model predicted a high response while the other predicted a low one, forcing standard ResNet50 and ResNet50-Robust to disagree about the same image. Panels a to g plot the measured, z-scored response of each site against each model's prediction (Lasso readouts), and panel h collects the per-site correlations for both models and compares them with a Wilcoxon signed-rank test. Everything comes from one table returned by `L.load_controversial_table()`, which carries the measured responses and their SEMs, both models' predictions, and per-site and pooled noise ceilings.

## setup
`C_R50` and `C_ROBUST`, with their faded variants, are copied from the companion controversial figure (S36) so legend markers match across the two figures. The site list is not hard-coded: `UNITS` is filled from the table at load time. The font-size constants are the only things a reader is likely to adjust.

## blocks
Two small helpers load and correlate, and two panel functions draw a site scatter and the summary.

## fn:_load
Reads the controversial table into module-level arrays that the panel helpers use. A stimulus is tagged as ResNet50-favoring when its name contains `max_r50`; everything else is Robust-favoring. It also pulls the per-site ceiling (`ncr`) and the pooled ceiling (`ncr_pool`), both expressed as Pearson r.

## fn:_r
A guarded Pearson correlation: returns NaN unless both inputs vary and there are at least five points.

## fn:scatter
Draws one site. Marker shape encodes the predicting model (circles for ResNet50, squares for Robust); the marker is fully coloured when the stimulus was synthesized in that model's favour and faded otherwise. Grey error bars give the SEM of the measured response, one least-squares line is fitted per model, and the two correlations are boxed in the top right. Note that each correlation pools both stimulus families for the site, matching S36. The per-site noise ceiling is printed as a small subtitle.

## fn:summary
Plots the seven per-site correlations for each model at two x positions, joins paired values with grey lines, marks group means with diamonds and a connecting black line, and annotates the mean values. Dotted lines show plus and minus the pooled noise ceiling. The Wilcoxon signed-rank test is run on the seven paired correlations and its p-value sits above a bracket. Returns the two r vectors and the p-value so the assembly can print them.

## assemble
The canvas is a 2 x 4 grid. The first seven cells (top row, then bottom row left to right) hold the site scatters in `UNITS` order and the bottom-right cell holds the summary. A title, a four-entry legend and an n-count note form a header; after a first draw their measured heights are used to re-space them so the three vertical gaps are equal.

## step:1
Loads the table, creates the figure and grid, draws the seven scatters (x labels only on the bottom row and on the top-right panel, which sits above the summary rather than another scatter), draws the summary, adds the header text and legend, places the panel letters, then renders once and re-flows the header so the gaps match.

## step:2
Saves the figure and prints the two mean correlations, the Wilcoxon p-value and the number of sites.

## closing
In panels a to g the green Robust fit should rise with the measured response and the orange ResNet50 fit should fall. In panel h the green points sit above zero and the orange points below it; the caption gives means of -0.32 for ResNet50 and +0.38 for RN50-Robust with Wilcoxon p = 0.016. The dotted ceiling lines show how much of the achievable correlation each model reaches.
