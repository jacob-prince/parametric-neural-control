## intro

Figure 4 scales the Figure 3 comparison up to all ten models and all 25 sites. Panel a shows each model's encoding score on held-out natural images (bar = mean over sites, dots = sites), panel b the same for parametric control on each model's own accentuated stimuli, and panel c a slopegraph connecting encoding to control per model family (adversarially trained, conventionally trained, untrained). Panel d repeats the story per animal and brain area: control bars with each model's encoding mean overlaid as a faint dash, and a per-area slopegraph beneath. The only data source is `loader.control_table()`, one row per model-site pair with `encoding_r` and `control_r`.

The panel renderers live in `fig4_panels.py` next to this script so that panel d inherits exactly the same styling as a, b and c. `bar_panel` draws per-model mean bars with SEM error bars and jittered site dots (dots that fall inside a bar are drawn white so they stay visible). `slope_panel` draws one thin encoding-to-control line per site colored by family, conventionally trained first so the sparser adversarial and untrained points draw on top, then family means as diamonds with SEM. `two_line_title` keeps a large first title row at a fixed height so the panel letters align. The module also defines the three families: `ADV` (ResNet50-Robust and CLIPAG), `UNTR` (the untrained AlexNet) and `CONV` (everything else).

## setup

`MONKEY_ORDER` fixes the column order of panel d, `D_ROWS` names its two rows (control bars, then the summary slopegraph), and `FS_D` is the smaller tick font used there. `YLIM` is computed at run time from the full range of both scores with a small margin and applied to every panel, so heights are comparable across the whole figure.

## fn:_tint

Blends a model color toward white; used for the encoding-mean dash in panel d so it reads as context rather than data.

## fn:region_of

Returns the brain-area label for a monkey straight from the table's `region` column, joining with a slash if a monkey spans two areas.

## assemble

The figure is two-column width and 132 mm tall. The top row is a 1 by 3 GridSpec for a, b and c. Before panel d is placed, the canvas is drawn and text extents are measured: the right edge of panel c's family labels becomes the right edge of the d grid, and the x-position of panel a's y-label becomes the anchor for d's row labels. Panel d is then a 2 by 5 GridSpec, one column per monkey.

## step:1

Loads the control table and keeps only the columns this figure needs.

## step:2

Creates the figure and the top-row GridSpec with three axes.

## step:3

Computes the shared `YLIM` and draws panels a and b with `bar_panel`, encoding score on the left and control on the right.

## step:4

Draws panel c with `slope_panel` and decorates it. Then measures the a, b and c text extents, builds the panel-d grid, and fills each monkey column: the top axes get control bars without dots, model names hanging below the baseline, and each model's encoding mean plus or minus one standard deviation as a tinted dash (keyed with small arrows in the first column only); the bottom axes get a compact slopegraph. Each column is headed by its brain area and monkey letter in the monkey's color.

## step:5

Draws the canvas again, adds the two row labels down the left edge of panel d at the measured x-position, and places the panel-d title just above the tallest column header. Prints the vertical gap between the a and b tick labels and that title so it can be checked.

## step:6

Places the four panel letters, using the top of the measured title text as the baseline for a, b, c and of the panel-d title for d.

## step:7

Saves the figure and prints the path.

## closing

Panel a is nearly flat: all ten models reach similar encoding scores (the caption gives 0.67 plus or minus 0.07 across model means). Panel b is lower and much more spread (0.45 plus or minus 0.15). In panel c only the adversarially trained family stays high under control, and panel d shows the same pattern within each animal and area.
