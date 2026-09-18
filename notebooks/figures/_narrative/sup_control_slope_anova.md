## intro

Supplementary Figure S17 repeats the Figure 4 test of whether models differ in how well they control neural responses, but using the control slope (the OLS slope of measured against predicted responses over a site-model pair's accentuated stimuli) in place of control r. Panel a shows the per-model distribution of slope over the 25 sites with a repeated-measures ANOVA across models; panel b shows the same ANOVA run separately within each recording area. The per-site slopes come from `loader.control_table()`.

## setup

`REGION_ORDER` and `REGION_LABEL` fix the area order and labels for panel b, distinguishing the two STS animals as STS(T) and STS(L). `ROBUST` holds the two adversarially trained models and `SHORT` relabels the untrained AlexNet.

## fn:rm_anova

A one-way repeated-measures ANOVA on a sites-by-models matrix: total variance is split into a model (condition) term, a site (subject) term and a residual, and F is the model mean square over the residual mean square with degrees of freedom k minus 1 and (n minus 1)(k minus 1). Returns F, both degrees of freedom and the p-value.

## fn:build

Pivots the control table into the 25 by 10 slope matrix, asserts it is complete, runs the ANOVA on the full matrix and again within each area's five sites.

## fn:panel_a

Orders models by mean slope from low to high, and for each draws a faint violin, jittered per-site dots (adversarially trained models ringed in dark gray) and a black mean bar. The y-range is extended upward to leave room for the ANOVA annotation and a small legend.

## fn:panel_b

Bars of the per-area F statistic in the animal's color, with significance stars above each bar and the shared degrees of freedom in the y-label.

## assemble

Two axes in one row at two-column width, the violin panel about two and a half times wider than the bar panel. Panels are drawn, spines trimmed, and titles and letters placed after a canvas draw.

## step:1

Runs `build` and prints the overall F, degrees of freedom and p, followed by the same statistics for each area.

## step:2

Creates the figure and both axes, draws the two panels and hides the top and right spines.

## step:3

Writes a title above each panel and a bold letter at its top left.

## step:4

Saves the figure.

## closing

In panel a, the two ringed models sit at the right and Untrained at the left; the annotation carries the ANOVA result. In panel b every bar carries stars, meaning the model effect holds within each area on its own.
