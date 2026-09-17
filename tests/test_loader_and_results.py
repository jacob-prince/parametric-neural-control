"""Independent checks that loader outputs are the statistics the figures claim to use."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pnc.preproc import loader


def test_control_cloud_is_personalized_diagonal_and_floor_clamped(brain, predictions, monkeys, models):
    for monkey in monkeys:
        b, p = brain(monkey), predictions(monkey)
        for unit in b["units"]:
            floor = loader.firing_floor(monkey, int(unit))
            for model in models:
                x, y = loader.control_cloud(monkey, int(unit), model)
                assert len(x) == len(y)
                assert len(x) > 0
                assert np.isfinite(x).all() and np.isfinite(y).all()
                assert np.all(x >= floor)

                mi = list(p["pred_models"]).index(model)
                ui = list(p["target_units"]).index(unit)
                selected = (p["gen_model"] == model) & (p["gen_unit"] == unit)
                response = loader._acc_response(monkey, int(unit))
                expected = [p["pred"][i, mi, ui] for i in np.flatnonzero(selected)
                            if p["stim"][i] in response]
                np.testing.assert_allclose(x, np.maximum(expected, floor))


def test_observational_clouds_equal_unclamped_encoding_sources(
    brain, encoding, monkeys, models
):
    """Audit the loader transform directly, without requiring any particular value range."""
    for monkey in monkeys:
        b, e = brain(monkey), encoding(monkey)
        for unit in b["units"]:
            ui = list(e["units"]).index(unit)
            for model in models:
                mi = list(e["models"]).index(model)
                source = dict(zip(e["stim"], e["pred"][ui, mi]))

                anchor_x, anchor_y = loader.anchor_cloud(monkey, int(unit), model)
                anchor_rows = [
                    (name, b["control"]["resp_z"][i, ui])
                    for i, (name, kind) in enumerate(
                        zip(b["control"]["stim"], b["control"]["kind"])
                    )
                    if kind == "calibration" and name in source
                ]
                np.testing.assert_allclose(anchor_x, [source[name] for name, _ in anchor_rows])
                np.testing.assert_allclose(anchor_y, [value for _, value in anchor_rows])

                keep = loader._test_names(monkey)
                encoding_x, encoding_y = loader.encoding_cloud(monkey, int(unit), model)
                encoding_rows = [
                    (name, value)
                    for name, value in zip(
                        b["calibration"]["stim"], b["calibration"]["resp_z"][:, ui]
                    )
                    if name in source and name in keep
                ]
                np.testing.assert_allclose(
                    encoding_x, [source[name] for name, _ in encoding_rows]
                )
                np.testing.assert_allclose(encoding_y, [value for _, value in encoding_rows])


def test_loader_measures_match_independent_formulas():
    pred = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    measured = 0.25 + 1.5 * pred
    r, slope, r2 = loader._measures(pred, measured)
    assert r == pytest.approx(stats.pearsonr(pred, measured).statistic)
    assert slope == pytest.approx(1.5)
    expected_r2 = 1 - np.square(measured - pred).sum() / np.square(measured - measured.mean()).sum()
    assert r2 == pytest.approx(expected_r2)


def test_control_table_has_one_row_per_site_model_and_recomputes_metrics(monkeys, models):
    table = loader.control_table()
    assert len(table) == 25 * 10
    assert not table.duplicated(["monkey", "unit", "model"]).any()
    assert set(table["monkey"]) == set(monkeys)
    assert set(table["model"]) == set(models)

    for row in table.itertuples():
        pred, measured = loader.control_cloud(row.monkey, row.unit, row.model)
        r, slope, r2 = loader._measures(pred, measured)
        assert row.n == len(pred)
        assert row.control_r == pytest.approx(r, abs=1e-12)
        assert row.control_slope == pytest.approx(slope, abs=1e-12)
        assert row.control_R2 == pytest.approx(r2, abs=1e-12)
