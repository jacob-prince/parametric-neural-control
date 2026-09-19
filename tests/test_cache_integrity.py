"""Internal consistency tests for the canonical take9 preprocessing caches."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pnc.preproc import exclusions, stimuli


EXPECTED_UNITS = {
    "red": [0, 2, 9, 15, 19],
    "paul": [0, 8, 24, 40, 47],
    "venus": [9, 79, 151, 331, 355],
    "leap": [81, 282, 286, 306, 342],
    "three0": [56, 74, 120, 168, 204],
}


def _manual_anchor_noise(values, names, days, anchors, min_groups):
    groups = {}
    for value, name, day in zip(values, names, days):
        if name in anchors and np.isfinite(value):
            groups.setdefault((name, day), []).append(float(value))
    variances = [np.var(group, ddof=1) for group in groups.values() if len(group) >= 2]
    return float(np.mean(variances)) if len(variances) >= min_groups else np.nan


def _manual_nsd_ceiling(values, names, stimulus_set, noise_var, min_stim):
    groups = {}
    for value, name in zip(values, names):
        if name in stimulus_set and np.isfinite(value):
            groups.setdefault(name, []).append(float(value))
    repeated = [group for group in groups.values() if len(group) >= 2]
    if len(repeated) < min_stim or not (np.isfinite(noise_var) and noise_var > 0):
        return np.nan, np.nan, len(repeated)
    means = np.array([np.mean(group) for group in repeated])
    inv_n = float(np.mean([1 / len(group) for group in repeated]))
    data_var = float(np.var(means, ddof=1))
    if not data_var > 0:
        return np.nan, 0.0, len(repeated)
    signal_var = max(data_var - noise_var * inv_n, 0.0)
    ratio2 = signal_var / noise_var
    return float(np.sqrt(ratio2 / (ratio2 + inv_n))), signal_var, len(repeated)


@pytest.mark.manuscript
def test_study_design_dimensions(brain, encoding, predictions, monkeys, models):
    assert len(monkeys) == 5
    assert len(models) == 10
    assert sum(len(brain(monkey)["units"]) for monkey in monkeys) == 25

    for monkey in monkeys:
        b, e, p = brain(monkey), encoding(monkey), predictions(monkey)
        assert b["units"].tolist() == EXPECTED_UNITS[monkey]
        assert list(e["models"]) == list(models)
        assert list(p["pred_models"]) == list(models)
        assert e["pred"].shape == (5, 10, 969)
        assert p["pred"].shape[1:] == (10, 5)
        assert np.isfinite(e["pred"]).all()
        assert np.isfinite(p["pred"]).all()


@pytest.mark.manuscript
def test_calibration_partition_and_source_counts(load_pickle_file, cache, monkeys):
    cal = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    for monkey in monkeys:
        use = cal["monkey"] == monkey
        assert int(use.sum()) == 969
        assert int(np.asarray(cal["is_train"][use], bool).sum()) == 774
        assert int(np.asarray(cal["is_test"][use], bool).sum()) == 195
        assert int(np.asarray(cal["is_normalizer"][use], bool).sum()) == 100
        assert int(np.asarray(cal["is_nsd"][use], bool).sum()) == 649
        assert int(np.asarray(cal["is_floc"][use], bool).sum()) == 61
        assert int(np.asarray(cal["is_OO"][use], bool).sum()) == 259


@pytest.mark.manuscript
def test_target_schedule_recomputed_from_calibration_percentiles(brain, predictions, monkeys):
    for monkey in monkeys:
        b, p = brain(monkey), predictions(monkey)
        for ui, unit in enumerate(b["units"]):
            q01, q99 = np.percentile(b["calibration"]["resp_z"][:, ui], [1, 99])
            bandwidth = q99 - q01
            expected = np.linspace(q01 - 0.25 * bandwidth, q99 + 0.50 * bandwidth, 11)
            got = np.unique(p["gen_level"][p["gen_unit"] == unit])
            assert len(got) == 11
            np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason='Monkey L: 277 design cells were synthesized twice; the duplicate variants are merged downstream by pnc.preproc.loader._leap_merge_map, not in the cache')
@pytest.mark.manuscript
def test_every_sweep_has_one_stimulus_at_each_of_11_levels(brain, predictions, monkeys):
    problems = []
    for monkey in monkeys:
        p = predictions(monkey)
        frame = pd.DataFrame({
            "model": p["gen_model"], "unit": p["gen_unit"],
            "seed": p["gen_seed"], "level": p["gen_level"], "stim": p["stim"],
        })
        key = ["model", "unit", "seed", "level"]
        duplicated = frame.duplicated(key, keep=False)
        duplicate_cells = frame.loc[duplicated, key].drop_duplicates()
        sizes = frame.groupby(["model", "unit", "seed"], sort=False).size()
        bad_sizes = sizes[sizes != 11]
        shown = set(brain(monkey)["control"]["stim"])
        presented = frame[frame["stim"].isin(shown)]
        presented_duplicate_cells = presented.loc[
            presented.duplicated(key, keep=False), key
        ].drop_duplicates()
        if duplicated.any() or len(bad_sizes):
            problems.append(
                f"{monkey}: {len(duplicate_cells)} duplicate design cells "
                f"({len(presented_duplicate_cells)} have multiple presented variants); "
                f"{len(bad_sizes)} sweeps not length 11"
            )
    assert not problems, "; ".join(problems)


def test_prediction_names_parse_and_match_cached_generator_metadata(predictions, monkeys):
    for monkey in monkeys:
        p = predictions(monkey)
        for i, name in enumerate(p["stim"]):
            parsed = stimuli.parse_accentuated(str(name))
            assert parsed is not None, f"unparseable: {name}"
            assert parsed["model"] == p["gen_model"][i]
            assert parsed["unit"] == int(p["gen_unit"][i])
            assert parsed["seed"] == int(p["gen_seed"][i])
            assert parsed["target"] == pytest.approx(float(p["gen_level"][i]), abs=1e-6)
            assert parsed["score"] == pytest.approx(float(p["score"][i]), abs=1e-6)


def test_trial_averages_and_repeat_counts_reconstruct_exactly(brain, monkeys):
    for monkey in monkeys:
        control = brain(monkey)["control"]
        by_name = {}
        for i, name in enumerate(control["trial_stim"]):
            by_name.setdefault(name, []).append(i)
        assert set(control["stim"]) == set(by_name)
        for i, name in enumerate(control["stim"]):
            idx = by_name[name]
            assert int(control["n_reps"][i]) == len(idx)
            np.testing.assert_allclose(
                control["resp_z"][i],
                np.asarray(control["trial_z"])[idx].mean(axis=0),
                atol=2e-6,
            )


def test_anchor_day_zscores_are_zero_mean_unit_sd_per_session(brain, monkeys):
    for monkey in monkeys:
        control = brain(monkey)["control"]
        days = np.asarray(control["trial_day"])
        kinds = np.asarray(control["trial_kind"])
        z = np.asarray(control["trial_z"], float)
        for day in np.unique(days):
            anchors = (days == day) & (kinds == "calibration")
            if anchors.sum() >= 5:
                np.testing.assert_allclose(z[anchors].mean(0), 0, atol=1e-5)
                np.testing.assert_allclose(z[anchors].std(0), 1, atol=1e-5)


def test_cached_noise_ceilings_recompute_independently_from_single_trials(brain, monkeys, models):
    for monkey in monkeys:
        b = brain(monkey)
        control = b["control"]
        names = np.asarray(control["trial_stim"])
        days = np.asarray(control["trial_day"])
        trial_z = np.asarray(control["trial_z"], float)
        anchors = set(names[np.asarray(control["trial_kind"]) == "calibration"])
        cache = pd.DataFrame(b["ceilings"])

        for ui, unit in enumerate(b["units"]):
            noise = _manual_anchor_noise(
                trial_z[:, ui], names, days, anchors,
                b["config"]["ceiling_min_anchor_groups"],
            )
            for model in models:
                generated = {
                    name for name in names
                    if name.startswith(model + "_RidgeCV") and f"_unit_{int(unit)}_" in name
                }
                nc_r, signal, nstim = _manual_nsd_ceiling(
                    trial_z[:, ui], names, generated, noise,
                    b["config"]["ceiling_min_stim"],
                )
                row = cache[(cache["unit"] == unit) & (cache["model"] == model)]
                assert len(row) == 1
                row = row.iloc[0]
                assert int(row.n_stim) == nstim
                if np.isnan(nc_r):
                    assert np.isnan(row.nc_r)
                else:
                    # Cached trials are float32 whereas ceilings were computed before
                    # serialization in float64; tolerate only that quantization boundary.
                    assert row.nc_r == pytest.approx(nc_r, abs=2e-6)
                    assert row.signal_var == pytest.approx(signal, abs=2e-6)
                    assert row.noise_var == pytest.approx(noise, abs=2e-6)


def test_firing_floors_are_finite_and_never_above_observed_or_session_floor(brain, monkeys):
    for monkey in monkeys:
        b = brain(monkey)
        assert np.isfinite(b["firing_floor"]).all()
        assert np.all(b["firing_floor"] <= b["firing_floor_avg"] + 1e-12)
        assert np.all(b["firing_floor"] <= b["firing_floor_obs_min"] + 1e-12)


def test_prediction_cache_is_raw_and_not_preclamped(brain, predictions, monkeys):
    for monkey in monkeys:
        b, p = brain(monkey), predictions(monkey)
        assert p.get("clamped_to_floor") is False
        models = list(p["pred_models"])
        units = list(p["target_units"])
        floors = dict(zip(map(int, b["units"]), map(float, b["firing_floor"])))
        for i, (model, unit) in enumerate(zip(p["gen_model"], p["gen_unit"])):
            achieved = p["pred"][i, models.index(model), units.index(unit)]
            assert np.isfinite(achieved)
        # Floors are metadata only in this cache; applying them must be an explicit,
        # reversible operation at comparison time.
        assert all(np.isfinite(list(floors.values())))


def test_exclusion_masks_recompute_from_documented_formulas(brain, predictions, monkeys):
    for monkey in monkeys:
        b, p = brain(monkey), predictions(monkey)
        shown = {
            n for n, kind in zip(b["control"]["stim"], b["control"]["kind"])
            if kind == "accentuated"
        }
        floors = dict(zip(map(int, b["units"]), map(float, b["firing_floor"])))
        result = exclusions.compute(p, shown, floors)
        expected_present = np.isin(p["stim"], list(shown))
        np.testing.assert_array_equal(result["present"], expected_present)
        np.testing.assert_allclose(
            result["relerr"],
            np.abs(result["target"] - result["achieved"])
            / np.array([
                np.ptp(result["target"][result["gen_unit"] == unit]) or 1.0
                for unit in result["gen_unit"]
            ]),
            rtol=1e-6,
            atol=1e-7,
        )
        for tau in result["thresholds"]:
            np.testing.assert_array_equal(
                result["masks"][tau]["permodel"],
                result["present"] & (result["relerr"] <= tau),
            )
