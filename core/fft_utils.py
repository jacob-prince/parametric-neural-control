import numpy as np

def image_fourier_power(img, return_shifted_spectrum=False):
    """
    Compute the 2D Fourier power spectrum of an image.
    Args:
        img (np.ndarray): 2D grayscale (H, W) or RGB (H, W, 3) image as ndarray.
        return_shifted_spectrum (bool): If True, also return shifted (centered) spectrum.
    Returns:
        power (np.ndarray): 2D power spectrum (H, W)
        (optionally) shifted_spectrum (np.ndarray): Shifted (centered) power spectrum
    """
    # If RGB, convert to grayscale
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


def fourier_power_radial_profile(power):
    """
    Compute the radial (frequency) average of a 2D power spectrum.
    Args:
        power (np.ndarray): 2D power spectrum (H, W)
    Returns:
        radial_prof (np.ndarray): 1D radial profile (mean power at each radius)
    """
    y, x = np.indices(power.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    # Bin the radii
    r = r.astype(np.int32)
    # Compute mean for each radius
    radial_prof = np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel())
    return radial_prof


def fourier_power_radial_profile_with_counts(power):
    """
    Compute the radial (frequency) average of a 2D power spectrum.
    Args:
        power (np.ndarray): 2D power spectrum (H, W)
    Returns:
        radial_prof (np.ndarray): 1D radial profile (mean power at each radius)
        bincounts (np.ndarray): 1D array, the count of pixels in each radius bin
    """
    y, x = np.indices(power.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    # Bin the radii
    r_int = r.astype(np.int32)
    # Compute mean for each radius, and also pixel count per radius bin
    bincounts = np.bincount(r_int.ravel())
    radial_prof = np.bincount(r_int.ravel(), power.ravel()) / bincounts
    return radial_prof, bincounts

def image_fourier_power_diff(img1, img2):
    """
    Compute the difference in Fourier power spectrum between two images.
    Returns the absolute and signed difference.
    Args:
        img1, img2 (np.ndarray): Two images, same shape.
    Returns:
        diff_abs (np.ndarray): Absolute difference (same shape as input)
        diff_signed (np.ndarray): Signed difference (img1 - img2)
    """
    pow1 = image_fourier_power(img1)
    pow2 = image_fourier_power(img2)
    diff_signed = pow1 - pow2
    diff_abs = np.abs(diff_signed)
    return diff_abs, diff_signed

# Example usage:
# img = path_to_nparray(img_path)
# power, power_shift = image_fourier_power(img, return_shifted_spectrum=True)
# prof = fourier_power_radial_profile(power_shift)
