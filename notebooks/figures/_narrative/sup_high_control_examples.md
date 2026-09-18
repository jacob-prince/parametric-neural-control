## intro
Figures S22 to S26 are a gallery of the best-controlled site-model pairs, one figure per macaque. For each monkey the script picks the four site-model combinations with the highest control correlation and shows, side by side, what the accentuation sweep looked like and how well the model's predicted responses matched the measured control-phase responses. The image strips come from the raw control stimuli on disk; the scatter draws on the cached per-stimulus predictions and the control-session responses. The point is to make the abstract control score concrete: these are the cases where a parametric sweep through a model's feature axis produced a matching parametric sweep in the neuron.

## setup
`MONKEYS` lists the five animals and `MONKEY_NUM` pairs each with its supplementary figure number, taken from the manifest rather than hardcoded. `N_EX` sets how many examples (rows) each figure shows, `N_STRIP` how many images are sampled from each sweep for the strip, and `N_ROW` how many seed sweeps are shown per example. `CMAP` is the diverging red-blue map used for both the image borders and the scatter dots, so blue always means suppress and red always means drive.

## blocks
The helpers split into two groups: those that select and shape the data (`top_examples`, `cloud_named`, `_seed_r2`, `top_seed_sweeps`) and those that find the stimulus images on disk (`image_dir`, `resolve_image`). `render` then does all of the drawing for one monkey.

## fn:top_examples
Reads the study-wide control table, keeps the rows for one monkey that have a defined control correlation, and returns the top `n` (monkey, unit, model) triples sorted by control `r`. This is the only place where the choice of examples is made.

## fn:cloud_named
Rebuilds the predicted-versus-measured cloud for one site-model, but keeps the stimulus name, seed and feature level alongside each point so the sweep can be reconstructed later. Predictions are floored at the site's firing floor, matching the convention used for the control score elsewhere in the paper, and stimuli with no recorded response are dropped.

## fn:image_dir
Locates the accentuation image folder for a given monkey and model under the control stimulus root by name pattern.

## fn:resolve_image
Turns a control stimulus name into a file path inside that folder, falling back to a glob if the exact name is not present. Returns nothing if the image is missing, which `render` records rather than failing.

## fn:_seed_r2
Computes an R-squared for a single seed sweep, comparing floored predictions to measured responses with the loader's one-minus-residual-over-total convention. It is used only to rank seeds, not for any reported statistic.

## fn:top_seed_sweeps
Groups the cloud by seed, ranks the seeds by `_seed_r2`, and returns the best `n` as level-sorted lists of (level, stimulus name). These become the image rows.

## fn:render
Draws one monkey's figure. Each example gets a row split into a two-by-six image grid on the left and a square scatter on the right. The grid shows six evenly spaced levels from each of the two best seed sweeps, with a border colored by feature level. The scatter plots every accentuated stimulus for that site-model, links the levels of each seed with a thin dotted line, colors dots by predicted response, and overlays a least-squares fit in the model's color with the pair's Pearson `r` and slope annotated. A shared colorbar and a title identifying the monkey and area sit at the top, and the function returns the saved path plus the key numbers and any missing images.

## assemble
`main` is short: it loops over the five monkeys, calls `top_examples` to choose that monkey's four best site-models, and hands them to `render`. In the notebook the `monkey` parameter is set to one animal, so only that figure is built.

## step:1
Iterates over `MONKEY_NUM`, skips animals that do not match `monkey`, and renders one gallery per selected animal. With `monkey` set, `png` is the single output path; with `monkey` left as `None` it would be the list of all five.

## closing
Look at how the image strips change smoothly from the blue (suppress) end to the red (drive) end, and how the dots in the scatter follow the fit line with the seed connectors running roughly parallel to it. Set `monkey` to `'red'`, `'paul'`, `'venus'`, `'leap'` or `'three0'` to render S22 through S26 respectively.
