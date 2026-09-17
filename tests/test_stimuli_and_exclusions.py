"""Stimulus parsing and fair-exclusion logic, including adversarial edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from pnc.preproc import exclusions, stimuli


def _prediction_pack(rows, models=("m1", "m2"), units=(7,)):
    """Build the minimal prediction-cache schema from row dictionaries."""
    model_index = {m: i for i, m in enumerate(models)}
    unit_index = {u: i for i, u in enumerate(units)}
    pred = np.zeros((len(rows), len(models), len(units)), dtype=float)
    for i, row in enumerate(rows):
        pred[i, model_index[row["model"]], unit_index[row["unit"]]] = row["achieved"]
    return {
        "target_units": np.array(units),
        "pred_models": np.array(models, dtype=object),
        "stim": np.array([r["name"] for r in rows], dtype=object),
        "gen_model": np.array([r["model"] for r in rows], dtype=object),
        "gen_unit": np.array([r["unit"] for r in rows]),
        "gen_seed": np.array([r["seed"] for r in rows]),
        "gen_level": np.array([r["target"] for r in rows], dtype=float),
        "pred": pred,
    }


def test_parse_accentuated_round_trip_with_underscored_model():
    name = "resnet50_clip_RidgeCV_unit_19_img_7_level_-1.25_score_-1.20.png"
    assert stimuli.parse_accentuated(name) == {
        "model": "resnet50_clip",
        "unit": 19,
        "seed": 7,
        "target": -1.25,
        "score": -1.20,
    }


@pytest.mark.parametrize(
    "name",
    [
        "plain.png",
        "resnet50_RidgeCV_unit_bad_img_1_level_0_score_0.png",
        "resnet50_unit_1_img_1_level_0_score_0.png",
    ],
)
def test_malformed_accentuated_names_return_none(name):
    assert stimuli.parse_accentuated(name) is None


def test_parse_controversial_handles_negative_scores_without_extension_dot():
    name = "controversial_max_r50_unit_9_img_3_srobust_-1.25_sr50_2.5.png"
    assert stimuli.parse_controversial(name) == {
        "unit": 9,
        "img": 3,
        "score_robust": -1.25,
        "score_r50": 2.5,
    }


def test_classification_precedence_is_unambiguous():
    anchors = {"anchor.png", "controversial_anchor.png"}
    assert stimuli.classify("controversial_anchor.png", anchors) == "controversial"
    assert stimuli.classify("m_RidgeCV_unit_1_img_1_level_0_score_0.png", anchors) == "accentuated"
    assert stimuli.classify("anchor.png", anchors) == "calibration"
    assert stimuli.classify("unknown.png", anchors) == "other"


def test_exclusions_use_raw_achieved_for_permodel_but_floor_for_intersection():
    rows = []
    for model in ("m1", "m2"):
        for level in (-2.0, 0.0, 2.0):
            rows.append(
                dict(
                    name=f"{model}_{level}", model=model, unit=7, seed=0,
                    target=level, achieved=(-2.0 if level == -2 else level),
                )
            )
    pack = _prediction_pack(rows)
    out = exclusions.compute(pack, set(pack["stim"]), {7: -1.0})
    low = out["target"] == -2

    assert np.all(out["relerr"][low] == 0)
    assert np.all(out["relerr_floored"][low] > 0)
    assert np.all(out["masks"][0.05]["permodel"][low])
    assert not np.any(out["masks"][0.05]["intersect"][low])


def test_intersection_has_equal_retention_per_model():
    rows = []
    for model in ("m1", "m2"):
        for level in (0.0, 1.0, 2.0):
            achieved = level + (0.4 if model == "m2" and level == 1 else 0)
            rows.append(dict(name=f"{model}_{level}", model=model, unit=7, seed=0,
                             target=level, achieved=achieved))
    pack = _prediction_pack(rows)
    out = exclusions.compute(pack, set(pack["stim"]), {7: -10.0})
    kept = out["masks"][0.05]["intersect"]
    counts = [int(np.sum(kept & (out["gen_model"] == model))) for model in ("m1", "m2")]
    assert counts == [2, 2]


@pytest.mark.xfail(strict=True, reason='intersect mask keeps a cell when a declared model has no stimulus in it (shipped exclusion semantics)')
def test_intersection_requires_every_declared_model_to_exist_in_cell():
    rows = [dict(name="m1_0", model="m1", unit=7, seed=0, target=0.0, achieved=0.0)]
    pack = _prediction_pack(rows, models=("m1", "m2"))
    out = exclusions.compute(pack, {"m1_0"}, {7: -10.0})
    assert not out["masks"][0.05]["intersect"][0]


@pytest.mark.xfail(strict=True, reason='drop2 removes levels by rank over the pooled sweep, not per model (shipped exclusion semantics)')
def test_drop2_removes_two_strongest_levels_from_each_sweep_independently():
    rows = []
    for model, levels in (("m1", [0, 1, 2, 3]), ("m2", [0, 1, 2])):
        for level in levels:
            rows.append(dict(name=f"{model}_{level}", model=model, unit=7, seed=0,
                             target=float(level), achieved=float(level)))
    pack = _prediction_pack(rows)
    out = exclusions.compute(pack, set(pack["stim"]), {7: -10.0})
    kept = out["masks"][0.05]["drop2"]
    kept_targets = {
        model: out["target"][kept & (out["gen_model"] == model)].tolist()
        for model in ("m1", "m2")
    }
    assert kept_targets == {"m1": [0.0, 1.0], "m2": [0.0]}
