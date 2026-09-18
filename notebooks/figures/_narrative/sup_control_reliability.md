## intro
Before interpreting control scores, we need to know how reliable the control-phase responses are and how close each model's score comes to the best achievable. This figure estimates a control-phase noise ceiling for each site-model with the NSD-style estimator: trial noise is taken from same-day repeats of the calibration anchor images shown during the control session, and the ceiling is the maximum Pearson correlation attainable given that noise and the repeat counts of the accentuated stimuli. Panel A shows how many repeats each accentuated stimulus received, panel B the ceilings per model, and panel C control `r` divided by the ceiling. Ceilings are defined for the three animals with sufficient same-day anchor repeats.

## setup
`CEIL_MONKEYS` lists the three animals (R, P, V) for which a control-session ceiling exists; the two STS animals are omitted. `NC_FLOOR` excludes site-models whose ceiling is at or below 0.1 from the normalization so the ratio does not blow up. `EX_MK`, `EX_UNIT` and `EX_MODEL` identify the Figure 3 example site used by `example_ceiling`.

## blocks
The first group of helpers rebuilds the ceiling from trial-level data (`_acc_trials`, `_anchor_trial_noise`, `_nsd_ceiling`, `example_ceiling`); the second reads precomputed tables (`ceiling_table`, `all_repeat_counts`); the `panel_*` functions draw. Note that the function names do not match the figure letters: `panel_b` draws panel A, `panel_c` draws B, and `panel_d` draws C.

## fn:_acc_trials
Collects, for one site-model, the trial-level z-scored responses to each of its accentuated stimuli in the control session, along with the full trial arrays needed to find the anchor repeats.

## fn:_anchor_trial_noise
Groups anchor trials by stimulus and recording day, takes the within-group variance for groups with at least two trials, and averages. Returns NaN unless at least `min_groups` such groups exist.

## fn:_nsd_ceiling
Given per-stimulus trial lists and the noise variance, computes the noise-corrected signal variance, the noise-ceiling SNR, and the resulting maximum Pearson `r`. It requires at least `min_stim` stimuli with two or more repeats and a positive noise variance.

## fn:example_ceiling
Runs the three helpers above for the Figure 3 example site to reproduce its stored ceiling from scratch. It is defined for reference and is not called in the assembled figure.

## fn:ceiling_table
Stacks the loader's per-monkey ceiling tables for the three animals into one long table of site, model and `nc_r`.

## fn:all_repeat_counts
Counts how many times each accentuated stimulus was shown in each control session, pooled over the three animals. This is a presentation count and does not use responses.

## fn:panel_a
Ranked per-stimulus mean control response with trial SEM shading for the example site. It is not used in the assembled figure.

## fn:panel_b
Histogram of repeat counts with integer bins, plus a dashed line and label at the mean. This is figure panel A.

## fn:panel_c
For each model, a swarm of per-site ceilings colored by animal, with a thick black-and-model-colored median bar; small triangles at the top mark the two adversarially trained models. Returns the per-model medians. This is figure panel B.

## fn:panel_d
Reads the control table, keeps site-models with a ceiling above `NC_FLOOR`, divides control `r` by the ceiling, and plots the ratio per model sorted by mean, with the mean printed above each column and the Untrained model drawn faint. A dotted line at one marks the ceiling. This is figure panel C.

## assemble
The layout is two rows: the top row holds the repeat histogram and the ceiling swarm side by side, and the bottom row is the full-width normalized-control panel. Data are gathered first, panels drawn, then titles and letters placed after a draw pass.

## step:1
Builds the ceiling table and the pooled repeat counts, and prints the number of stimuli, mean repeats and range.

## step:2
Creates the nested grid, draws the three panels, collects the per-model medians and normalized means, and removes the top and right spines.

## step:3
After a draw pass, writes each panel's title above it and a bold letter at its top-left.

## step:4
Saves the figure and prints the per-model median ceilings and the per-model mean ceiling-normalized control.

## closing
Panel B should show ceilings clustered high for every model, including Untrained, which tells you the measurements themselves are reliable regardless of which model generated the stimuli. In panel C compare the ordering of models with the raw control ranking; the adversarially trained models should still lead once the ceiling is accounted for.
