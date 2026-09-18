## intro

Figure 6 turns to benchmarking. The first question is whether pooling every model's accentuations into one big stimulus set sharpens or blurs the differences between models. Panel a plots each model's brain predictivity under four regimes (its own personalized accentuations, the pooled set, the pooled set minus its own stimuli, and the pooled set minus everything the adversarially trained models generated), and panel b shows the distribution of predicted activation per model for the personalized and pooled sets side by side. The second question is whether carefully chosen natural images could do the same job. Panel c shows two prediction planes for an example model pair (ResNet50 against ResNet50-Robust, for the Figure 3 aIT site): ImageNet validation images chosen to maximize disagreement, and the accentuated sweeps. Panels d, e and f quantify the disagreement for that pair, across all channel and model-pair combinations, and as a function of image budget.

All data come through `fig6_data.py` next to this script, which only unpickles three caches built by `scripts/preprocessing/build_fig6_caches.py`: `load_separation` (per-channel predictivity under each regime plus the violin samples), `load_efficiency` (disagreement across all combinations and budget curves) and `load_pair` (everything for the worked example). It also names the frozen ImageNet thumbnail folder `THUMB_DIR_INET`. Note that the script's internal axis names run a to h while the rendered figure uses six letters, as in the caption: b spans both violin panels and c spans both planes.

## setup

`PAIR_PRESETS` defines the worked-example pair: `rn50` (ResNet50 versus ResNet50-Robust, the default) or `clip` (RN50-CLIP versus CLIPAG), chosen with `--preset`. `CH` is the example channel (Monkey R unit 9, the Figure 3 site), `PER_MODEL_BUDGET` is the per-model image count matched between accentuated stimuli and best-case ImageNet subsets, and the three `COL_` constants color the ImageNet cloud, the best-case ImageNet subset and the accentuated set throughout.

## fn:plot_violin_panel

Draws one half-violin per model (the left half only), a short median tick, and every individual point jittered to the right. The jitter uses a per-model seeded generator so a change in one model's sample count never reshuffles another's dots, and the point cloud is rasterized to keep the PDF small.

## fn:_ci95

Mean with a t-based 95% confidence interval, or NaN bounds for fewer than two values.

## fn:_accent_local_path

Finds the accentuated image on disk whose achieved score is closest to a cached score for a given source model and unit, searching every monkey's directory; used to fetch thumbnails for the accentuated plane.

## fn:_band

Median and 25th to 75th percentile band across the columns of a budget array, for the curves in the last panel.

## assemble

The canvas is 14 by 9.6 inches with an outer two-row GridSpec. The top row holds the regime chart (wider) and the two violin panels sharing a y-axis. The bottom row holds the two square prediction planes and a three-panel stack on the right for the disagreement violins, the across-combination summary and the budget curve. Panels are filled top to bottom, and letters are placed last from measured positions.

## step:1

Resolves the preset into `MODEL_A` and `MODEL_B`.

## step:2

Loads the three caches, checks that the thumbnail folder exists, builds the layout, and draws the two violin panels for the personalized and pooled samples.

## step:3

Panel a: for every model with data in all four regimes, computes the mean and 95% CI over channels per regime and draws a connected line with error bars and points.

## step:4

Adds the model labels on the right of panel a, stacked in order of personalized score with thin leader lines to each model's last point so they never overlap.

## step:5

Annotates each regime with the standard deviation of the model means (the untrained model excluded) and decorates panel a. Then computes the shared square limits for the two planes from the outer percentiles of the ImageNet predictions and the sweep predictions.

## step:6

Defines `_plane`, which draws a prediction plane: identity line, the faint ImageNet cloud, zero lines, equal aspect, and axis labels in each model's color.

## step:7

Draws both planes. On the accentuated plane each sweep is a thin black line with dots colored by achieved score on a fixed red-blue scale and a white diamond at the level nearest zero; the ImageNet cloud is left faint behind it with a small tag.

## step:8

Defines `_place_thumb_grid`, which tiles thumbnails into a corner of a plane. It walks a grid of candidate cells from the corner inward, skips any cell that would cover a data point, resolves each item's image (cached relative path, rebuilt from the ImageNet index if the cached path is stale, or located on disk for accentuated stimuli), and draws the thumbnail with a colored frame plus a square marker at the item's predicted position.

## step:9

Places the thumbnails on both planes (B-preferring in the upper left, A-preferring in the lower right). Then draws panel d: violins of per-image disagreement for the best ImageNet subset (the top and bottom `PER_MODEL_BUDGET` by signed difference) against all accentuated points, with mean bars and a star at each maximum. Panel e: the same comparison across all channel and pair combinations, with mean and SEM diamonds, and the fraction of combinations where the accentuated set wins. Panel f: median and interquartile band of mean disagreement against per-model budget on a log axis, with a dotted line at the working budget. Finally the six panel letters, the save, and a printout of the regime SDs and the winning fraction.

## closing

Read the SD annotations across panel a: separation is largest for personalized accentuations and shrinks under pooling, yet the two adversarially trained models stay on top even when all their stimuli are removed. In panel d the accentuated set roughly doubles the disagreement of the best-case ImageNet subset (the caption gives means of 1.30 versus 0.60), and panel e shows this holds in 90.5% of combinations. Pass `--preset clip` to render the other example pair.
