## intro
This figure makes the gradient spectral participation ratio (PR) concrete by walking through its computation for one seed image (the cat) at one recording site (Monkey V, V3/V4, unit 331). Panel a shows the seed beside a 2 x 5 mosaic of each model's accentuation of that seed toward the site, all matched to a common achieved response. Panel b shows, one row per model, the four steps from input gradient to radial power profile and the PR that summarizes it. Gradient maps come from the cache behind `L.load_grad_maps`, accentuation thumbnails from the sweep PNGs under `STIMULI_CONTROL_PATH`, and the seed from `STIMULI_PATH`.

## setup
`MONKEY`, `UNIT` and `SEED` pick the site and the seed index; `SEED_FILE` names the image. `FMIN` and `FMAX` (1 and 112) define the frequency band over which PR is computed. The rest of the constants are the canvas geometry in millimetres: `S` and `E_W` size the image tiles and the radial-power axis, `DX` offsets the right block of five models, `B_ROW0_TOP` sets where panel b starts, and `A_SEED*`, `THUMB` and `ROW_BOT` lay out panel a. `--gradsum cv` switches to the coefficient-of-variation variant, which also draws a 1 +/- CoV band.

## fn:accent_curve
Scans the accentuation directory for one model, site and seed, matching filenames with a regular expression, and returns the sweep as sorted (level, achieved score, path) triples.

## fn:load_accent_at
Picks the sweep image whose achieved response is closest to a target, resizes it, and returns the image with its score.

## fn:seed_thumb
Loads the seed image (trying png then jpg) at thumbnail size, with a flat grey placeholder if neither exists.

## fn:row_data
The spectral pipeline for one gradient map, following `pnc.preproc.fft_utils` exactly. The display image is 0.5 plus the gradient divided by its standard deviation; the grayscale is the mean over RGB and is what gets Fourier transformed; `image_fourier_power` returns the shifted 2D power spectrum; the radial profile is averaged over rings. PR is the squared sum of band power divided by the sum of squared power (or CoV in the alternative mode), and the profile is divided by its band mean so the mean sits at 1 and all rows can share one axis.

## assemble
All axes are placed in absolute millimetre coordinates. Panel b is drawn first as two side-by-side blocks of five models in figure-4 order (the `MODEL_COLORS` order), each row four tiles plus a radial-power axis; then panel a is drawn above it: the seed image and the 2 x 5 accentuation mosaic; letters and panel titles come last.

## step:1
Loads the gradient maps for the site, runs `row_data` on the cat-seed gradient of every model, splits the models into left and right blocks of five, sets shared y-limits from all profiles, loads the seed thumbnail, and finds the accentuation target: the largest response reachable by every trained model (the smallest of the per-model maxima, excluding Untrained). Each model's thumbnail is the sweep image closest to that target.

## step:2
Defines a one-line helper that converts a row index into the bottom y coordinate of that row in panel b.

## step:3
Creates the figure and draws panel b. For each model row: the RGB gradient, the grayscale with a signed-square-root stretch clipped at the 98th percentile so faint structure is visible, the log10 power spectrum in magma with white rings marking three radii, and the log-log radial profile with a dashed line at the mean and the PR value boxed. Column titles, row labels and the frequency axis label follow, and the seed image for panel a is placed.

## step:4
Fills the panel-a mosaic: each thumbnail framed in its model colour, labelled above, with the achieved z printed in the corner. Panel letters and titles are added, the panel-a title stating the matched target.

## step:5
Saves the figure, then prints the matched target, the per-model PR values in figure-4 order, and each thumbnail's achieved z.

## closing
Compare the radial profiles across rows: the adversarially trained models (CLIPAG, RN50-Robust) concentrate their power at low frequencies and have lower PR, whereas the conventionally trained models are more broadband with higher PR. Note that nothing in panel b depends on the accentuated images; PR is a property of the encoding model alone.
