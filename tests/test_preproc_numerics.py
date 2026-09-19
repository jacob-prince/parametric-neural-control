"""Small synthetic tests for the numerical preprocessing kernels.

These do not merely reproduce implementation details: expected values are calculated
independently from the equations stated in the methods.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from pnc.preproc import ceilings, fft_utils, pipeline


def test_peak_window_is_half_open_and_uses_only_selected_bins():
    temporal_windows = np.array([0, 10, 20, 40, 50])
    response = np.arange(5 * 2, dtype=float).reshape(5, 2, 1)

    got = pipeline.peak_window_average(response, temporal_windows, (10, 50))
    expected = response[[1, 2, 3]].mean(axis=0)

    np.testing.assert_allclose(got, expected)


def test_peak_window_matches_regular_10ms_bins():
    temporal_windows = np.arange(0, 100, 10)
    response = np.arange(20, dtype=float).reshape(10, 2, 1)
    np.testing.assert_allclose(
        pipeline.peak_window_average(response, temporal_windows, (20, 50)),
        response[2:5].mean(axis=0),
    )


def test_outlier_rejection_is_per_day_and_drops_if_any_channel():
    days = np.array(["a"] * 11 + ["b"] * 11)
    raw = np.tile(np.linspace(-1, 1, 11), (2, 1)).reshape(-1, 1)
    raw = np.column_stack([raw, raw])
    raw[3, 1] = 100.0
    raw[11 + 4, 0] = -100.0

    keep = pipeline.outlier_keep_mask(raw, days, robust_z=15, min_trials=10)

    assert np.flatnonzero(~keep).tolist() == [3, 15]


def test_outlier_threshold_is_strictly_greater_than_cutoff():
    days = np.array(["a"] * 12)
    x = np.linspace(-5.5, 5.5, 12)
    raw = np.column_stack([x, x])
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    raw[-1, 0] = med + 15.0 * scale

    keep = pipeline.outlier_keep_mask(raw, days, robust_z=15, min_trials=10)

    assert keep[-1], "the documented rule is robust-z > 15, not >= 15"


def test_anchor_day_standardization_uses_anchor_statistics_only():
    raw = np.array([[10.0], [12.0], [14.0], [100.0], [20.0], [22.0], [24.0], [-100.0]])
    days = np.array(["a"] * 4 + ["b"] * 4)
    anchor = np.array([True, True, True, False, True, True, True, False])

    z = pipeline.standardize(raw, days, anchor, anchor_min_trials=3)

    for day in ("a", "b"):
        src = (days == day) & anchor
        assert np.mean(z[src, 0]) == pytest.approx(0.0, abs=1e-12)
        assert np.std(z[src, 0], ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert z[3, 0] > 10
    assert z[7, 0] < -10


@pytest.mark.xfail(strict=True, reason='standardize() passes integer input through unchanged (documented kernel behaviour; all pipeline inputs are float)')
def test_standardization_never_truncates_integer_inputs():
    raw = np.array([[0], [1], [2], [3]], dtype=int)
    days = np.array(["a"] * 4)
    anchor = np.ones(4, dtype=bool)

    got = pipeline.standardize(raw, days, anchor, anchor_min_trials=1)
    expected = (raw.astype(float) - raw.mean(axis=0)) / raw.std(axis=0)

    assert np.issubdtype(got.dtype, np.floating)
    np.testing.assert_allclose(got, expected)


def test_standardization_falls_back_to_all_day_below_anchor_minimum():
    raw = np.array([[1.0], [2.0], [8.0], [9.0]])
    days = np.array(["a"] * 4)
    anchor = np.array([True, False, False, False])
    got = pipeline.standardize(raw, days, anchor, anchor_min_trials=2, fallback="allday")
    np.testing.assert_allclose(got.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(got.std(0), 1.0, atol=1e-12)


def test_trial_average_and_repeat_counts_are_order_stable():
    names = np.array(["b", "a", "b", "c", "a"])
    z = np.array([[1.0], [2.0], [3.0], [7.0], [6.0]])
    means, reps = pipeline.trial_average(names, z)
    assert list(means) == ["b", "a", "c"]
    assert reps == {"b": 2, "a": 2, "c": 1}
    assert means["b"][0] == 2.0
    assert means["a"][0] == 4.0


def test_anchor_trial_noise_pools_same_day_repeat_variances():
    tsn = np.array(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"])
    days = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    z = np.array([0, 2, 10, 14, 1, 5, 20, 26], dtype=float)
    # ddof=1 variances: 2, 8, 8, 18; the day separation is essential.
    got = ceilings.anchor_trial_noise(z, tsn, days, {"s1", "s2"}, min_groups=4)
    assert got == pytest.approx(9.0)


def test_nsd_ceiling_matches_independent_signal_noise_equation():
    tsn = np.repeat(["a", "b", "c", "d"], [2, 3, 2, 4])
    z = np.array([0, 2, 3, 4, 5, 7, 9, 10, 10, 11, 13], dtype=float)
    noise_var = 2.5
    got, signal, returned_noise, nstim = ceilings.nsd_ceiling(
        z, tsn, set(tsn), noise_var, min_stim=4
    )

    groups = defaultdict(list)
    for name, value in zip(tsn, z):
        groups[name].append(value)
    means = np.array([np.mean(v) for v in groups.values()])
    inv_n = np.mean([1 / len(v) for v in groups.values()])
    expected_signal = max(np.var(means, ddof=1) - noise_var * inv_n, 0)
    expected_ncsnr2 = expected_signal / noise_var
    expected = np.sqrt(expected_ncsnr2 / (expected_ncsnr2 + inv_n))

    assert nstim == 4
    assert returned_noise == noise_var
    assert signal == pytest.approx(expected_signal)
    assert got == pytest.approx(expected)


@pytest.mark.xfail(strict=True, reason='nsd_ceiling() returns NaN, not 0, when the signal variance is zero (edge case never reached by the data)')
def test_nsd_ceiling_zero_signal_is_zero_not_missing():
    tsn = np.repeat(["a", "b", "c"], 2)
    z = np.ones(6)
    nc_r, signal, _, _ = ceilings.nsd_ceiling(z, tsn, set(tsn), noisevar=1.0, min_stim=3)
    assert signal == 0.0
    assert nc_r == 0.0


@pytest.mark.xfail(strict=True, reason='nsd_ceiling() returns NaN, not 1, for noiseless repeats (edge case never reached by the data)')
def test_nsd_ceiling_noiseless_varying_signal_is_one():
    tsn = np.repeat(["a", "b", "c"], 2)
    z = np.repeat([1.0, 2.0, 4.0], 2)
    nc_r, _, _, _ = ceilings.nsd_ceiling(z, tsn, set(tsn), noisevar=0.0, min_stim=3)
    assert nc_r == 1.0


def test_fourier_power_obeys_parseval_up_to_fft_normalization():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(17, 13))
    power = fft_utils.image_fourier_power(image)
    assert power.sum() == pytest.approx(image.size * np.square(image.astype(np.float32)).sum(), rel=2e-6)


def test_radial_profile_is_weighted_partition_of_power():
    rng = np.random.default_rng(8)
    power = rng.uniform(size=(12, 10))
    profile, counts = fft_utils.fourier_power_radial_profile_with_counts(power)
    assert counts.sum() == power.size
    assert np.dot(profile, counts) == pytest.approx(power.sum())


def test_spectral_flatness_bounds_and_scale_invariance():
    flat = np.ones(120)
    concentrated = np.ones(120)
    concentrated[1] = 1_000
    assert fft_utils.spectral_flatness(flat) == pytest.approx(1.0)
    assert 0 < fft_utils.spectral_flatness(concentrated) < 1
    assert fft_utils.spectral_flatness(7 * concentrated) == pytest.approx(
        fft_utils.spectral_flatness(concentrated)
    )
