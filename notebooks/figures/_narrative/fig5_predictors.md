## intro

Figure 5 asks what property of a model predicts how well its encoding axis controls a site. Two candidates are compared: adversarial sensitivity (how far a small pixel perturbation can push the predicted response) and the spectral structure of the encoding axis's input gradient (whether its energy is concentrated in a few spatial-frequency bands or spread thin). Panel a is a 4 by 10 image mosaic for one cIT site of Monkey P and the cat seed: per model, the minimally perturbed image that reaches the attack target, the perturbation itself, the input-gradient saliency map, and an accentuation matched to a common achieved response, with models ordered by gradient spectral flatness so the adversarially trained ones sit on the right. Panel b plots sensitivity against perturbation strength, panel c the model-mean gradient power spectra with an inset of the concentration metric, panel d a dual scatter of site-residualized control outcome against each predictor, and panel e summary correlations across model subsets against a split-half noise ceiling.

Two helper modules next to this script supply the data. `fig5_adv.py` reads the cluster PGD outputs: the minimal-epsilon attack records for the mosaic, the per-epsilon normalized-swing sweeps on 100 held-out NSD images (`load_heldout_curves`), and the log-epsilon area under that curve that serves as the sensitivity measure (`load_heldout_logauc`), plus the per-readout outcome table. `fig5_grad.py` supplies the gradient side: spectral flatness and normalization of radial profiles, saliency maps from gradient magnitude, the per-site gradient data merged with control outcomes (`load_data`), the seed and accentuation image loaders, and `site_config`, which resolves the example site and seed.

## setup

Five knobs define the variant: `SITE` (`paul8` by default, `red9` is the alternative), `SEED_IDX` (3 is the cat), `OUTCOME` (`slope` or `r`), `RESID` (`9trained` residualizes against the nine trained models, `10` against all, `none` uses the absolute outcome) and `GRADMETRIC` (`pr`, `cv` or `flatness`). `_configure` re-binds all derived labels and column names from these. The default combination writes the plain manuscript filename; any other combination gets a descriptive tag. `FMIN` and `FMAX` from the gradient helper bound the frequency band used by every spectral metric.

## blocks

The helpers move from configuration, through the two spectral metrics and data loaders, to the statistics (`williams_p`, `pair_stats`, `_stars`, `_load_bootp`), and finally the three panel renderers (`_flatness_inset_nonat`, `_scatter`, `_summary_bars`) and `build`, which assembles everything.

## fn:_site_resid

Subtracts each site's mean outcome so the scatter compares models within a site. With `RESID` set to `9trained` the mean is taken over the nine trained models only, so the untrained baseline is measured against the trained reference rather than pulling it down; `none` returns the column unchanged.

## fn:spectral_cv

Coefficient of variation of the radial gradient power spectrum within the analysis band: a spectrum with a few dominant bands has a high value.

## fn:spectral_pr

Participation ratio of the same spectrum: the effective number of frequency bands carrying power. It is an exact transform of the coefficient of variation, so the two metrics rank models identically; lower values mean greater concentration.

## fn:_load_heldout_profiles

Reads, per monkey, unit and model, the mean radial gradient profile over the 100 held-out NSD images. This is the same probe set the adversarial analysis uses and was never used for synthesis, so the gradient metric is not circular.

## fn:load_gallery

Loads the per-model minimal-epsilon attack record (clean image, adversarial image, epsilon used, predictions) for one site and seed.

## fn:williams_p

Two-sided test for whether two correlations that share a variable differ, here the correlation of control with sensitivity versus with the gradient metric, given their own correlation and n.

## fn:pair_stats

For each model subset (all 10, the 9 trained, the 7 conventionally trained) and each level (across models, across sites), computes the correlation of each predictor with the outcome and a Williams p for the gap between them. Both correlations are first oriented to the sign they have in the full 10-model set, so a predictor that flips sign in a subset reads as negative and the tested gap widens rather than shrinks.

## fn:_load_bootp

Reads an optional frozen table of seed-level bootstrap p-values for the bracket stars (available for the default residualization only); when absent the Williams p is used.

## fn:_flatness_inset_nonat

A horizontal bar inset inside the spectra panel listing the per-model gradient metric, sorted, with model names backed in white so they stay readable over the curves.

## fn:_scatter

One half of panel d: site-level dots (diamonds for the untrained and adversarially trained models), model means with 95% t-based confidence intervals in both directions, a dashed site-level fit and a solid model-level fit, and a box giving both correlations.

## fn:_summary_bars

Panel e: horizontal bars of the oriented correlations, gray for adversarial sensitivity and black for the gradient metric, in six slots (model level and site level, each for the 10, 9 and 7 model subsets). Each slot gets the split-half ceiling as a dashed line with a shaded band. A bar that dips just below zero marks a predictor whose sign flipped in that subset. Significance stars come from the bootstrap table when available, otherwise from the Williams test, and are drawn as a bracket or above the bars depending on space.

## fn:build

Loads and merges everything, computes the held-out spectral metrics per site, residualizes the outcome, ranks models by flatness, gathers the mosaic ingredients, lays out the figure and draws all five panels with their titles and letters. Returns the finished figure.

## assemble

`build` does the whole layout: an outer two-row GridSpec with the mosaic on top (four rows by ten model columns, row labels on the far left, model names as column headers) and a four-cell analysis band below, where the third cell is split again into the two halves of panel d. After drawing, panel b is widened to match panel c so their plotted extents are equal. `main` only configures, calls `build`, and saves.

## step:1

Configures the variant, builds the figure, decides whether this is the default render or a tagged variant, saves, and prints the settings used.

## closing

In panel a, watch the perturbation and gradient rows: the adversarially trained models on the right show coherent, low-frequency structure while the rest are diffuse and high-frequency. In panel e the black gradient bars stay positive even for the seven conventionally trained models, whereas the gray sensitivity bars do not (the caption gives site-level r of 0.26 versus 0.05). Use `--site`, `--seed-img`, `--outcome`, `--resid` and `--gradmetric` to render the variants.
