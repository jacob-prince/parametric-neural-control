"""Spectral-cache validation (gradient-frequency, gradient-map and ImageNet prediction caches)."""
from __future__ import annotations

import numpy as np
import pytest

from pnc.preproc import fft_utils


def test_gradient_summary_has_complete_site_model_grid(load_pickle_file, cache, monkeys, models):
    data = load_pickle_file(cache / "gradient_freq.pkl")
    gradients = data["gradients"]
    keys = [(g["monkey"], int(g["unit"]), g["model"]) for g in gradients]
    assert len(keys) == 250
    assert len(set(keys)) == 250
    assert set(g["model"] for g in gradients) == set(models)
    assert set(g["monkey"] for g in gradients) == set(monkeys)
    assert np.array_equal(data["freqs"], np.arange(len(data["freqs"])))
    assert np.all(data["bincounts"] > 0)
    assert np.asarray(data["natural_profiles_per_image"]).shape == (969, len(data["freqs"]))
    assert np.isfinite(data["natural_profiles_per_image"]).all()
    assert np.all(np.asarray(data["natural_profiles_per_image"]) >= 0)
    for item in gradients:
        assert item["n_seeds"] == 10
        assert np.asarray(item["profile_mean"]).shape == data["freqs"].shape
        assert np.isfinite(item["profile_mean"]).all()
        assert np.all(np.asarray(item["profile_mean"]) >= 0)


def test_cached_gradient_profile_recomputes_from_raw_gradient_map(load_pickle_file, cache):
    files = sorted((cache / "grad_maps").glob("*.pkl"))
    assert files, "no gradient-map cache found"
    data = load_pickle_file(files[0])
    profile, counts = fft_utils.gradient_radial_profile(data["grad_img"][0])
    np.testing.assert_allclose(profile, data["profiles"][0], rtol=2e-7, atol=1e-8)
    np.testing.assert_array_equal(counts, data["bincounts"])


def test_spectral_participation_ratio_and_cov_are_exact_transforms(load_pickle_file, cache):
    data = load_pickle_file(cache / "gradient_freq.pkl")
    nband = 111
    for item in data["gradients"]:
        power = np.asarray(item["profile_mean"], float)[1:112]
        cv = power.std() / power.mean()
        pr = power.sum() ** 2 / np.square(power).sum()
        assert pr == pytest.approx(nband / (1 + cv**2), rel=1e-12)
        assert 1 <= pr <= nband


def test_imagenet_prediction_cache_dimensions_and_labels(load_pickle_file, cache, models):
    data = load_pickle_file(cache / "imagenet_predictions.pkl")
    pred = np.asarray(data["imagenet_predictions"])
    assert pred.shape == (50_000, 25, 10)
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert list(data["models"]) == list(models)
    assert len(set(zip(data["unit_monkeys"], map(int, data["unit_ids"])))) == 25
