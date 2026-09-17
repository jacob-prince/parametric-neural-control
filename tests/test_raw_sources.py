"""Optional checks against the large raw sources (enable with ``--run-raw``)."""
from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from pnc.preproc import pipeline


@pytest.mark.raw
@pytest.mark.manuscript
def test_control_hdf5_response_windows_and_channel_dimensions(source, monkeys):
    from pnc import utils

    expected = {"red": (100, 400), "paul": (100, 400), "venus": (80, 250),
                "leap": (80, 250), "three0": (80, 250)}
    with h5py.File(utils.CONTROL_HDF5_PATH, "r") as handle:
        for monkey in monkeys:
            key = next(key for key in handle if key.startswith(monkey))
            group = handle[key]
            assert tuple(map(int, group["neuron_metadata/peak_respwindow"][:])) == expected[monkey]
            response = group["trials/response_temporal"]
            assert response.ndim == 3
            assert response.shape[2] > max(utils.MONKEY_UNITS[monkey])
            assert response.shape[1] == len(group["trials/stimulus_name"])


@pytest.mark.raw
def test_cached_control_trials_rebuild_from_epoch_extraction_outliers_and_zscoring(
    source, brain, monkeys
):
    """End-to-end audit of the highest-risk preprocessing chain against raw HDF5."""
    from pnc import utils

    def decode(values):
        return np.array([
            value.decode() if isinstance(value, bytes) else str(value) for value in values
        ])

    with h5py.File(utils.CONTROL_HDF5_PATH, "r") as handle:
        for monkey in monkeys:
            cached = brain(monkey)
            cfg = cached["config"]
            units = list(map(int, cached["units"]))
            key = next(key for key in handle if key.startswith(monkey))
            group = handle[key]
            windows = np.asarray(group["neuron_metadata/temporal_windows"][:])
            peak_window = tuple(map(int, group["neuron_metadata/peak_respwindow"][:]))
            names = np.asarray(group["trials/stimulus_name"][:], dtype=str)
            days = decode(group["trials/session_dates"][:])
            temporal = np.asarray(group["trials/response_temporal"][:, :, units], float)

            raw = pipeline.peak_window_average(temporal, windows, peak_window)
            keep = pipeline.outlier_keep_mask(
                raw, days, cfg["outlier_robust_z"], cfg["outlier_min_trials"],
                cfg["outlier_drop_if_any_channel"],
            )
            raw, names, days = raw[keep], names[keep], days[keep]
            anchors = np.isin(names, cached["calibration"]["stim"])
            rebuilt = pipeline.standardize(
                raw, days, anchors, method=cfg["standardize"],
                anchor_min_trials=cfg["anchor_min_trials"], fallback=cfg["fallback"],
            )

            assert int((~keep).sum()) == int(cached["n_trials_dropped"])
            np.testing.assert_array_equal(names.astype(object), cached["control"]["trial_stim"])
            np.testing.assert_array_equal(days.astype(object), cached["control"]["trial_day"])
            np.testing.assert_allclose(rebuilt, cached["control"]["trial_z"], atol=3e-5)


@pytest.mark.raw
def test_cached_encoding_responses_match_raw_hdf5_without_reordering(source, brain, monkeys):
    from pnc import utils

    for monkey in monkeys:
        cached = brain(monkey)
        units = list(map(int, cached["units"]))
        names, repavg, trial_names, trials = utils.load_encoding_hdf5(monkey)
        np.testing.assert_array_equal(names.astype(object), cached["calibration"]["stim"])
        np.testing.assert_allclose(repavg[:, units], cached["calibration"]["resp_z"], atol=1e-7)
        np.testing.assert_array_equal(trial_names.astype(object), cached["calibration"]["trial_stim"])
        np.testing.assert_allclose(trials[:, units], cached["calibration"]["trial_z"], atol=1e-7)


@pytest.mark.raw
@pytest.mark.manuscript
def test_raw_encoding_metadata_boolean_counts(source):
    """Pinpoint the mixed bool/string source columns that must be decoded semantically."""
    from pnc import utils

    data = utils.load_encoding_pickle("red", 0, "AlexNet_training_seed_01")["df"]

    def semantic_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "t", "yes"}

    assert sum(map(semantic_bool, data["is_train"])) == 774
    assert sum(map(semantic_bool, data["is_test"])) == 195
    assert sum(map(semantic_bool, data["is_normalizer"])) == 100


@pytest.mark.raw
def test_firing_floor_formula_recomputes_from_session_pickles(source, brain, monkeys):
    import glob
    from pnc import utils

    for monkey in monkeys:
        units = list(map(int, brain(monkey)["units"]))
        pattern = str(
            source / "brain_data_encoding" /
            f"{monkey}_*_hp0_bs0_zs1_{utils.MONKEY_RESPONSE_WINDOWS[monkey]}_sessdata.pkl"
        )
        files = sorted(glob.glob(pattern))
        assert files
        zero_z = {unit: [] for unit in units}
        observed = {unit: [] for unit in units}
        for filename in files:
            with open(filename, "rb") as handle:
                session = pickle.load(handle)
            mu = np.asarray(session["neuron_meta"]["mu"], float)
            sigma = np.asarray(session["neuron_meta"]["sigma"], float)
            trial_z = np.asarray(session["peak_resp"], float)
            for unit in units:
                zero_z[unit].append(-mu[unit] / sigma[unit])
                observed[unit].append(np.nanmin(trial_z[unit]))
        expected = np.array([
            min(np.mean(zero_z[unit]), np.min(observed[unit])) for unit in units
        ])
        np.testing.assert_allclose(brain(monkey)["firing_floor"], expected, atol=1e-12)
