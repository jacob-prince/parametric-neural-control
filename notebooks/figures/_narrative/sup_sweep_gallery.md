## intro

Supplementary Figures S12 to S16 are one gallery per monkey showing how all ten models accentuate three of that animal's recorded sites, each site paired with a different seed image. Within a gallery, the three (site, seed) groups sit side by side; within a group, rows are the ten models and columns are four target levels along each model's own accentuation axis (strong suppress, weak suppress, weak drive, strong drive). Because the columns walk the same sweep for every model, the grid exposes how differently each model reshapes the same image. The pixels come straight from the raw accentuated PNGs under `stimuli_control` and the seed thumbnails from `stimuli_encoding`.

## setup

`GROUPS_BY_MONKEY` lists the three (unit, seed) pairs shown per monkey, using seeds 3, 4 and 7 for every animal. `LEVELS` picks ordinal levels 0, 1, 6 and 10 of the 11 in a sweep, and `COL_COLORS` samples the red-blue level colormap for the column borders. `SAT` applies a 15% saturation boost to every thumbnail, and `THUMB` sets the 224-pixel cell size. The `monkey` parameter selects which of the five figures to render; None renders all five.

## blocks

The small helpers resolve and load files: `acc_dir`, `_level_score`, `_level` and `sweep_files` find the eleven PNGs of a sweep, and `_saturate`, `load_thumb` and `seed_thumb` turn files into display arrays. `build_gallery` does all the drawing for one monkey.

## fn:sweep_files

Globs the PNGs for one (model, unit, seed), parses the target level and achieved score from each filename, and returns the files sorted from suppress to drive. A few sweeps were re-synthesized and left duplicates at the same target level; the copy whose achieved score is closest to the target is kept.

## fn:load_thumb

Opens an accentuated PNG, resizes it to the cell size if needed and applies the saturation boost.

## fn:seed_thumb

Loads the seed image for a group at cell size with the same saturation; the middle group (seed 4, the dog) uses the shared cropped asset `seed.png` so it matches other figures.

## fn:build_gallery

Renders one monkey's figure. It first loads all thumbnails for the three groups and checks that each sweep has all eleven levels and truly spans from a negative to a positive level; sweeps that fail get gray placeholders and set `span_ok` to False. It then lays out a 1 by 3 parent grid with a nested 10 by 4 grid per group, colors each cell border by column, labels the columns on the top row and the models at the far left, and places the seed thumbnail with a site label above each group. The output name is taken from the manifest span for this monkey.

## assemble

`main` is a thin loop: for each requested monkey it calls `build_gallery`, collects the output path and the span check, and reports any missing files at the end.

## step:1

Loops over the selected monkeys, renders each gallery, prints its path and whether every sweep spanned suppress to drive, then prints the accumulated issue list and the overall check.

## closing

Read each group down a column to compare models at one level, or across a row to follow one model's sweep. Set `monkey` to `'red'`, `'paul'`, `'venus'`, `'leap'` or `'three0'` to render S12 through S16; the seed thumbnails at the top of each group identify the base image.
