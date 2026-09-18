## intro
Not every synthesized accentuation reached its intended target. A stimulus is called a failure when the generating model's own post-hoc prediction for the saved image misses the target by more than a tolerance, expressed as a fraction of the channel's target range. Simply dropping each model's failures would compare models on different stimulus sets, so the paper considers several exclusion regimes. This figure lays them out and checks that the per-model control results do not depend on the choice. Row A shows how much each regime keeps, row B summarizes the regimes in a table, and row C recomputes control `r` under each regime. Exclusion masks come precomputed from the loader; the control clouds are rebuilt from cached predictions and control-session responses.

## setup
`THRESHOLDS` lists the two tolerances shown (strict then lenient) and `TAU_PRIMARY` the one used for the control outcomes in row C. `CRITERIA` defines the four regimes with a label, a table definition and a color. `MIN_LEVELS` is the number of kept levels a sweep needs to count as usable, and `MIN_STIM` the number of kept stimuli a site-model needs before a control `r` is computed.

## blocks
`load_all`, `_setup` and `compute` build module-level retention statistics; `_sweeps_ge` and `_fair_seedconds` are the two counting helpers behind the table; `control_by_regime` recomputes the outcomes; the five `draw_*` functions each render one component.

## fn:load_all
Concatenates the exclusion records of all five monkeys: generating model, unit, seed, level order, whether the stimulus was presented, and the boolean keep-mask for every threshold and regime.

## fn:_setup
Populates the module-level arrays from `load_all`, builds the sweep and seed-condition keys, and computes the retention statistics for both thresholds. It runs from `main` so importing the module does no work.

## fn:_sweeps_ge
Counts sweeps (monkey, unit, model, seed) that retain at least `K` levels under a mask.

## fn:_fair_seedconds
Counts seed-conditions (monkey, unit, seed) in which every model retains at least `K` levels, which is the number of conditions where all models can be compared on equal footing.

## fn:compute
For one threshold and each regime, records the mask, the number and percentage of stimuli kept, the usable-sweep count, the fair seed-condition count, and the per-model retention percentage.

## fn:control_by_regime
For every site-model and regime at the primary tolerance, rebuilds the control cloud from the kept stimuli exactly as the loader does (own-model prediction, firing floor applied, measured response) and stores the Pearson `r`, skipping site-models with fewer than `MIN_STIM` kept stimuli or no variance.

## fn:draw_footing
Heatmap of per-model retention percentage (regimes by models) at one threshold, with the value printed in each cell and an optional colorbar.

## fn:draw_retention
Paired bars of overall retention per regime, solid for the strict threshold and hatched for the lenient one, colored by regime.

## fn:draw_completion
Bars of the percentage of synthesized stimuli actually presented, per monkey. Unpresented stimuli are excluded under every regime, which matters for the two STS animals.

## fn:draw_table
A five-column table listing each regime's definition, stimuli kept, usable sweeps and fair seed-conditions at both thresholds, with row shading by regime. The table is pinned to fill its axes exactly so the row gaps stay even.

## fn:draw_regime_outcomes
One subplot per regime with a fixed model order and a shared y-range: each model's site-model control `r` values as a swarm with a black mean bar. The shared axis is what makes shifts between regimes visible.

## assemble
The figure is three rows with hand-set vertical boundaries rather than a shared grid, because the rotated tick labels of row A and the titles of row C overhang by different amounts. Row A holds two heatmaps, the retention bars and the completion bars; row B is the table; row C is the four regime subplots. Statistics are computed first, then the rows are drawn top to bottom, then title and letters.

## step:1
Runs `_setup` to load masks and compute retention, then `control_by_regime` to recompute control `r` under each regime.

## step:2
Creates the canvas, defines the three row bands, and draws row A (both heatmaps, retention, completion), row B (table) and row C (regime outcomes).

## step:3
Writes the figure title at the top left.

## step:4
After a draw pass, places the row letters, then prints for each regime the range of per-model mean `r`, the Spearman rank correlation of model means against the no-exclusion regime, and the smallest per-model site-model count.

## step:5
Saves and closes the figure; `png` receives the path.

## closing
In row C, read across the four subplots: the model order and the lead of the two bold models should look essentially the same in every regime, even in the intersect regime that keeps far fewer stimuli. Row A explains why the regimes differ so much in retention, and the completion bars show which animals contribute fewer presented stimuli.
