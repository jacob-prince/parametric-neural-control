"""Partition isolation, model-selection provenance, and cross-validation validity."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pnc.preproc import loader


def _bool_array(values):
    """Decode mixed bool/string/0-1 metadata by value, never Python truthiness."""
    truthy = {"1", "true", "t", "yes"}
    return np.array([
        bool(value) if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in truthy
        for value in values
    ])


def test_train_and_test_are_disjoint_exhaustive_and_identical_across_animals(
    load_pickle_file, cache, monkeys
):
    cal = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    reference = None
    for monkey in monkeys:
        use = cal["monkey"] == monkey
        names = np.asarray(cal["stimulus_name"][use], str)
        train = _bool_array(cal["is_train"][use])
        test = _bool_array(cal["is_test"][use])
        assert not np.any(train & test)
        assert np.all(train | test)
        partition = {name: (bool(tr), bool(te)) for name, tr, te in zip(names, train, test)}
        if reference is None:
            reference = partition
        else:
            assert partition == reference


def test_normalizers_are_balanced_between_train_and_test(load_pickle_file, cache, monkeys):
    cal = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    for monkey in monkeys:
        use = cal["monkey"] == monkey
        train = _bool_array(cal["is_train"][use])
        test = _bool_array(cal["is_test"][use])
        normalizer = _bool_array(cal["is_normalizer"][use])
        assert int(normalizer.sum()) == 100
        assert int((normalizer & train).sum()) == 50
        assert int((normalizer & test).sum()) == 50


@pytest.mark.xfail(strict=True, reason='one calibration image (UFH_159) shown as a control-phase anchor is not in the declared normalizer set')
def test_control_calibration_anchors_are_exactly_the_declared_normalizer_set(
    load_pickle_file, cache, brain, monkeys
):
    cal = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    for monkey in monkeys:
        use = cal["monkey"] == monkey
        names = np.asarray(cal["stimulus_name"][use], str)
        normalizer = _bool_array(cal["is_normalizer"][use])
        declared = set(names[normalizer])
        shown = {
            str(name) for name, kind in zip(
                brain(monkey)["control"]["stim"], brain(monkey)["control"]["kind"]
            ) if kind == "calibration"
        }
        assert shown <= declared
        if len(shown) == 100:
            assert shown == declared


def test_encoding_cloud_uses_only_the_heldout_complement_of_train(
    load_pickle_file, cache, brain, encoding, monkeys, models
):
    calmeta = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    for monkey in monkeys:
        use = calmeta["monkey"] == monkey
        names = np.asarray(calmeta["stimulus_name"][use], object)
        train = _bool_array(calmeta["is_train"][use])
        expected_names = set(names[~train])
        assert loader._test_names(monkey) == expected_names

        b, e = brain(monkey), encoding(monkey)
        for unit in b["units"]:
            ui = list(e["units"]).index(unit)
            measured = dict(zip(b["calibration"]["stim"], b["calibration"]["resp_z"][:, ui]))
            for model in models:
                mi = list(e["models"]).index(model)
                predicted = dict(zip(e["stim"], e["pred"][ui, mi]))
                x, y = loader.encoding_cloud(monkey, int(unit), model, split="test")
                expected_order = [name for name in b["calibration"]["stim"] if name in expected_names]
                assert len(x) == len(expected_names)
                np.testing.assert_allclose(x, [predicted[name] for name in expected_order])
                np.testing.assert_allclose(y, [measured[name] for name in expected_order])


@pytest.mark.xfail(strict=True, reason='anchor_cloud uses all 100 re-presented calibration anchors, not only the 50 held-out ones')
def test_cross_session_generalization_excludes_training_normalizer_images(
    load_pickle_file, cache, brain, monkeys, models
):
    """The 100 daily anchors normalize sessions; only their 50 test images evaluate generalization."""
    calmeta = load_pickle_file(cache / "stimuli.pkl")["calibration"]
    for monkey in monkeys:
        use = calmeta["monkey"] == monkey
        names = np.asarray(calmeta["stimulus_name"][use], str)
        train = _bool_array(calmeta["is_train"][use])
        heldout = set(names[~train])
        control = brain(monkey)["control"]
        expected = {
            str(name) for name, kind in zip(control["stim"], control["kind"])
            if kind == "calibration" and name in heldout
        }
        assert 0 < len(expected) <= 50
        for unit in brain(monkey)["units"]:
            for model in models:
                anchor_x, anchor_y = loader.anchor_cloud(monkey, int(unit), model)
                shared_x, shared_y = loader.shared_cloud(monkey, int(unit), model)
                assert len(anchor_x) == len(anchor_y) == len(expected)
                assert len(shared_x) == len(shared_y) == len(expected)


@pytest.mark.raw
def test_pca_readout_predictions_are_linear_and_partition_dimensions_are_valid(source, monkeys):
    from pnc import utils

    # One source fit per animal is sufficient to verify the serialized schema and equation;
    # all 250 cached fit identities are checked elsewhere.
    for monkey in monkeys:
        unit = utils.MONKEY_UNITS[monkey][0]
        data = utils.load_encoding_pickle(monkey, unit, "resnet50")
        frame = data["df"]
        train = _bool_array(frame["is_train"])
        test = _bool_array(frame["is_test"])
        features = np.asarray(data["PCA_resp"], float)
        weights = np.asarray(data["readout_vec"], float).ravel()
        bias = float(np.asarray(data["readout_bias"]))
        serialized_prediction = np.asarray(data["target_unit_resp"], float).ravel()
        assert features.shape == (969, 750)
        assert int(train.sum()) == 774 > features.shape[1]
        assert int(test.sum()) == 195
        assert not np.any(train & test)
        assert np.isfinite(features).all()
        np.testing.assert_allclose(features @ weights + bias, serialized_prediction, atol=2e-4)


@pytest.mark.raw
def test_accentuation_seed_images_are_heldout_normalizers_never_fit_images(source, monkeys):
    from pnc import utils

    reference = None
    for monkey in monkeys:
        unit = utils.MONKEY_UNITS[monkey][0]
        data = utils.load_encoding_pickle(monkey, unit, "resnet50")
        frame = data["df"]
        names = np.asarray(frame["stimulus_name"], str)
        train = _bool_array(frame["is_train"])
        test = _bool_array(frame["is_test"])
        normalizer = _bool_array(frame["is_normalizer"])
        seed_names = [os.path.basename(path) for path in data["config"]["seed_image_paths"]]
        assert len(seed_names) == len(set(seed_names)) == 10
        index = {name: i for i, name in enumerate(names)}
        assert all(name in index for name in seed_names)
        idx = np.array([index[name] for name in seed_names])
        assert not train[idx].any()
        assert test[idx].all()
        assert normalizer[idx].all()
        if reference is None:
            reference = seed_names
        else:
            assert seed_names == reference
