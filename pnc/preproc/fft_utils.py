"""Fourier-power utilities for the input-gradient spectrum analysis (Fig 7 flatness).

VENDORED VERBATIM from the Closed-loop-visual-insilico toolkit,
`core/fft_utils.py` (Binxu Wang), so that the gradient-flatness computation is
reproducible standalone inside take9 without depending on that external repo.

These are the exact functions used by the upstream generator
(`cluster/encoding_model_gradmap_freq_analysis.py`) that produced the per-(site,model)
`*_grad_maps_freq_profiles.pkl` files cached under `preproc/cache/grad_maps/`:
the per-seed input gradient (3xHxW) is reduced to grayscale by MEAN over color
channels, 2D-FFT'd (no mean subtraction), power = |fftshift(F)|^2, then radially
averaged about center ((W-1)/2, (H-1)/2). Verified to reproduce the cached
`profiles` to ~1e-8 (see qc). Spectral flatness = geometric/arithmetic mean of the
radial profile over the 1-111 cyc/img band.
"""
import numpy as np


def image_fourier_power(img, return_shifted_spectrum=False):
    """2D Fourier power spectrum of an image.
    img: 2D grayscale (H,W) or RGB (H,W,3). RGB is reduced to grayscale by mean over channels.
    Returns power (H,W); if return_shifted_spectrum, also the fftshift-centered power.
    """
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img
    img_gray = img_gray.astype(np.float32)
    F = np.fft.fft2(img_gray)
    F_shift = np.fft.fftshift(F)  # center low-freq at center
    power = np.abs(F) ** 2
    power_shift = np.abs(F_shift) ** 2
    if return_shifted_spectrum:
        return power, power_shift
    return power


def fourier_power_radial_profile_with_counts(power):
    """Radial (frequency) average of a 2D (fftshift-centered) power spectrum.
    Returns radial_prof (mean power per integer radius bin) and bincounts (pixels per bin).
    """
    y, x = np.indices(power.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    r_int = r.astype(np.int32)
    bincounts = np.bincount(r_int.ravel())
    radial_prof = np.bincount(r_int.ravel(), power.ravel()) / bincounts
    return radial_prof, bincounts


def gradient_radial_profile(grad_chw):
    """Convenience for take9: (3,H,W) input-gradient -> its radial Fourier power profile
    (the per-seed `profiles` row in the cached grad_maps pkls). Matches the upstream recipe:
    grad_img_t[i].permute(1,2,0) -> image_fourier_power(...) -> radial-with-counts."""
    grad_hwc = np.asarray(grad_chw).transpose(1, 2, 0)
    _, power_shift = image_fourier_power(grad_hwc, return_shifted_spectrum=True)
    prof, bincounts = fourier_power_radial_profile_with_counts(power_shift)
    return prof, bincounts


def spectral_flatness(profile, fmin=1, fmax=112):
    """Wiener spectral flatness (geometric/arithmetic mean) of a radial power profile
    over the [fmin, fmax) frequency-bin band. 1 = white/flat, low = concentrated."""
    p = np.clip(np.asarray(profile, float)[fmin:fmax], 1e-30, None)
    return float(np.exp(np.mean(np.log(p))) / p.mean())
