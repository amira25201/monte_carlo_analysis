"""Unit tests on synthetic data; the real dataset is never needed to run these."""

import numpy as np
import pandas as pd
import pytest

import config
import mc_analysis as mc


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.RandomState(0)
    n = 120
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "P_ID": rng.randint(10**6, 10**7, n),
            "L2_input": rng.rand(n),
            "Usage_L2": rng.rand(n),
            "L2_attitude": rng.rand(n),
            "Phonological_Awareness": rng.rand(n),
            "Proficiency_L2": rng.rand(n),
            "Age_in_months": rng.randint(120, 400, n),
            "L1": rng.choice(["Polish", "Chinese", "Arabic"], n),
            "Attn_check_BLP": 1.0,
            "Attn_check_input": 1.0,
        }
    )
    signal = 0.5 * df["L2_input"] + 0.3 * df["Usage_L2"]
    df["Grammar"] = (signal + 0.05 * rng.randn(n)).clip(0, 1)
    df["Vocabulary"] = (signal + 0.05 * rng.randn(n)).clip(0, 1)
    df["Pronunciation_discrimination"] = rng.rand(n)
    df["Pronunciation_stress"] = rng.rand(n)
    df["Pronunciation_comprehension"] = rng.rand(n)
    df["Pronunciation_TOTAL"] = df[["Pronunciation_discrimination", "Pronunciation_stress", "Pronunciation_comprehension"]].mean(axis=1)
    return df


@pytest.fixture(scope="module")
def fitted(synthetic):
    X, num, cat = mc.split_predictors(synthetic)
    return mc.train_target_pipelines(synthetic, X, mc.build_preprocessor(num, cat), n_estimators=30)


def test_targets_ids_outcomes_and_flags_are_not_predictors(synthetic):
    X, num, cat = mc.split_predictors(synthetic)
    for col in config.TARGETS + config.EXCLUDED_PREDICTORS:
        assert col not in X.columns, col
    assert "L2_input" in num and "L1" in cat


def test_sub_scores_cannot_leak_into_pronunciation_model(fitted):
    ft = fitted["Pronunciation_TOTAL"]
    assert "Pronunciation_stress" not in ft.X_train.columns
    # Without the sub-scores the synthetic pronunciation target is pure noise, so the model cannot fit it.
    assert ft.r2 < 0.3


def test_each_pipeline_owns_its_preprocessor(fitted):
    preps = [ft.pipe.named_steps["prep"] for ft in fitted.values()]
    assert len({id(p) for p in preps}) == len(preps)


def test_bootstrap_predictions_shape_and_seed(fitted):
    ft = fitted["Grammar"]
    prep = ft.pipe.named_steps["prep"]
    X_tr = np.asarray(prep.transform(ft.X_train), dtype=np.float32)
    X_te = np.asarray(prep.transform(ft.X_test), dtype=np.float32)
    a = mc.bootstrap_predictions(ft.pipe.named_steps["model"], X_tr, ft.y_train, X_te, n_boot=4, seed=1)
    b = mc.bootstrap_predictions(ft.pipe.named_steps["model"], X_tr, ft.y_train, X_te, n_boot=4, seed=1)
    assert a.shape == (4, len(ft.y_test))
    np.testing.assert_array_equal(a, b)


def test_predict_total_is_mean_of_models_and_vectorised(fitted):
    models = [ft.pipe for ft in fitted.values()]
    X = fitted["Grammar"].X_test
    total = mc.predict_total(models, X)
    expected = np.mean([m.predict(X) for m in models], axis=0)
    np.testing.assert_allclose(total, expected)
    assert total.shape == (len(X),)


def test_ceilings_use_test_rows_and_total_is_reachable(fitted):
    ceilings = mc.compute_ceilings(fitted).set_index("target")["ceiling"]
    for ft in fitted.values():
        assert ceilings[ft.target] == pytest.approx(ft.pipe.predict(ft.X_test).max())
    total = mc.predict_total([ft.pipe for ft in fitted.values()], fitted["Grammar"].X_test)
    assert ceilings["TOTAL_L2"] == pytest.approx(total.max())


def test_population_baseline_is_median_and_mode(synthetic):
    X, _, _ = mc.split_predictors(synthetic)
    x0 = mc.population_baseline(X)
    assert len(x0) == 1
    assert x0["L2_input"].iloc[0] == pytest.approx(X["L2_input"].median())
    assert x0["L1"].iloc[0] == X["L1"].mode().iloc[0]


def test_bounds_come_from_observed_range(synthetic):
    X, _, _ = mc.split_predictors(synthetic)
    b = mc.observed_bounds(X, ["L2_input", "not-a-column"])
    assert b == {"L2_input": (float(X["L2_input"].min()), float(X["L2_input"].max()))}


def test_joint_search_respects_bounds_and_only_edits_editable(fitted, synthetic):
    ft = fitted["Grammar"]
    X, _, _ = mc.split_predictors(synthetic)
    x0 = mc.population_baseline(X)
    editable = ["L2_input", "Usage_L2"]
    bounds = {"L2_input": (0.2, 0.6), "Usage_L2": (0.0, 1.0)}
    goal = float(ft.pipe.predict(ft.X_test).max())
    y, x_best = mc.joint_search(ft.pipe.predict, x0, goal, editable, bounds, mc.SearchSettings(steps=200, patience=50, multistart=3))
    assert 0.2 <= x_best["L2_input"].iloc[0] <= 0.6
    untouched = [c for c in X.columns if c not in editable]
    pd.testing.assert_frame_equal(x_best[untouched], x0[untouched])
    assert y >= float(ft.pipe.predict(x0)[0]) - 1e-9


def test_changes_table_handles_no_change():
    x0 = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    empty = mc.changes_vs_baseline(x0, x0.copy(), ["a", "b"])
    assert empty.empty and list(empty.columns) == ["feature", "baseline", "needed", "change"]
    moved = mc.changes_vs_baseline(x0, pd.DataFrame([{"a": 1.5, "b": 2.0}]), ["a", "b"])
    assert moved.iloc[0].feature == "a" and moved.iloc[0].change == pytest.approx(0.5)


def test_editable_features_exclude_traits_and_self_ratings():
    for fixed in ["Age_in_months", "Non.Verbal_Intelligence", "Visual_Working_Memory", "Phonological_Working_Memory", "Proficiency_L2"]:
        assert fixed not in config.EDITABLE_FEATURES

