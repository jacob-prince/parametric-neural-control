## intro

Supplementary Figure S7 asks how often feature accentuation actually hit its target. For every synthesized stimulus, the generating model's own post-hoc prediction for the saved image is compared to the intended target, and the miss is expressed as a fraction of the channel's target range. Panel a shows the ten seed images the sweeps start from; panel b is the histogram of misses with the 2% and 5% tolerances marked; panel c gives the per-model success rate at 5%; panel d shows how success depends on the drive level. The miss values come from the per-monkey exclusion caches via `loader.load_exclusions`, which also carry the generating model and the ordinal level of each stimulus.

## setup

`TAU` (0.05) is the primary tolerance and `TAU2` (0.02) the stricter secondary one. `SEED_IMAGES` lists the ten seed files in sweep order, and seed 4 (the dog) is drawn from the shared cropped asset `seed.png` so it matches every other figure that shows it. `MONKEYS` lists the five animals whose caches are pooled.

## fn:seed_thumb

Loads one seed image as a 224-pixel RGB thumbnail, substituting the canonical cropped asset for the dog seed.

## assemble

An outer two-row GridSpec splits the canvas into a short strip for the seed images and a taller row for the three analysis panels. The seed strip is a nested 1 by 10 grid; the lower row is a 1 by 3 grid with the histogram narrowest. Panel letters and a bold suptitle are placed after a canvas draw. Note that the code names the lower axes `axA`, `axB` and `axC`; they are caption panels b, c and d.

## step:1

Loads the exclusion cache for each monkey, concatenates the generating model, relative error and ordinal level across animals, and drops stimuli with a non-finite error.

## step:2

Computes the success flags at both tolerances and the per-model rate, then creates the figure and fills the top strip with the ten seed thumbnails, each labeled by its seed index.

## step:3

Draws the lower row: the miss histogram clipped at 30% of range with dashed threshold lines and an inset that magnifies the y-axis about 80-fold to show the tail; the per-model success bars with a dotted line at the overall rate; and success versus ordinal level, one faint line per model plus the bold pooled line. A level is plotted for a model only if it has at least ten stimuli there.

## step:4

Adds the suptitle and places the four panel letters against the measured axis positions.

## step:5

Saves the figure.

## step:6

Prints the number of stimuli kept, the overall success rates at 5% and 2%, and the per-model rate at 5%.

## closing

The histogram in panel b is a spike near zero, and the inset is where the tail becomes visible. In panel d, the pooled black line stays near ceiling through the middle levels and drops at the extremes, especially the strongest drive levels; the faint per-model lines show which models account for that drop.
