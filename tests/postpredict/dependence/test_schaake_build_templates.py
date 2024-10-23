# Tests for postpredict.dependence.TimeDependencePostprocessor.transform

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from postpredict.dependence import Schaake


def test_schaake_build_templates_equal_weights(obs_data, wide_model_out):
    ss = Schaake()
    ss.target_data_train = obs_data
    ss.key_cols = ["location", "age_group"]
    ss.time_col = "date",
    ss.obs_col = "value"
    ss.feat_cols = ["location", "age_group"]
    
    ss._build_train_X_Y(1, 4)
    
    # Get a whole lot of templates, check that the relative frequency of the
    # selected train_Y's is about even
    # specifically, we test that all marginal frequencies are within a 99.999% CI,
    # a tolerance of about 0.0044.
    # not perfect; expect that this test will fail in roughly 24/100000 runs
    n_times = 1000
    templates = ss._build_templates(pl.concat([wide_model_out] * n_times))
    freq_diffs = (
        templates["value_shift_p1"]
        .value_counts()
        .with_columns(
            (pl.col("count") / (wide_model_out.shape[0] * n_times) - 1 / 24).alias("freq_diff")
        )
    )
    assert all(np.abs(freq_diffs["freq_diff"]) < 4.417 * np.sqrt(1/24 * (1 - 1/24) / (wide_model_out.shape[0] * n_times)))


def test_schaake_build_templates_unequal_weights(obs_data, wide_model_out):
    # we test with an artificial scheme where weights are proportional to
    # population size if age groups match, 0 otherwise
    class PopSizeWeighter():
        def __init__(self) -> None:
            pass
        
        def get_weights(self, train_X, test_X):
            weights = (
                (np.expand_dims(test_X["age_group"].to_numpy(), -1) == np.expand_dims(train_X["age_group"].to_numpy(), 0))
                * np.expand_dims(train_X["population"].to_numpy(), 0)
            )
            weights = weights / np.sum(weights, axis = 1)[:, np.newaxis]
            return weights
    
    ss = Schaake(weighter = PopSizeWeighter())
    ss.target_data_train = obs_data
    ss.key_cols = ["location", "age_group"]
    ss.time_col = "date",
    ss.obs_col = "value"
    ss.feat_cols = ["location", "age_group", "population"]
    
    ss._build_train_X_Y(0, 4)
    
    # Get a whole lot of templates, check that the relative frequency of the
    # selected train_Y's is about equal to the expected frequencies
    # specifically, we test that all marginal frequencies are within a 99.999% CI,
    # a tolerance of about 0.0044.
    # not perfect; expect that this test will fail in some runs (rarely)
    n_times = 1000
    test_wmo = pl.concat([wide_model_out] * n_times)
    templates = ss._build_templates(test_wmo)
    
    for age_group in ["young", "old"]:
        expected_freqs = (
            obs_data
            .with_columns(
                weight_unnormalized = (pl.col("age_group") == age_group).cast(int) * (pl.col("date") < datetime.strptime("2020-01-14", "%Y-%m-%d") + timedelta(6)).cast(int) * pl.col("population")
            )
            .with_columns(
                weight = pl.col("weight_unnormalized") / pl.col("weight_unnormalized").sum(),
                value_shift_p0 = pl.col("value")
            )
        )
        actual_freqs = (
            templates.filter(test_wmo["age_group"] == age_group)
            ["value_shift_p0"]
            .value_counts()
            .with_columns(
                (pl.col("count") / (wide_model_out.shape[0] * n_times / 2)).alias("freq")
            )
            .sort(by = "value_shift_p0")
        )
        freq_diffs = (
            expected_freqs
            .join(actual_freqs, on = "value_shift_p0", how = "left")
            .with_columns(
                freq_diff = pl.col("weight") - pl.col("freq")
            )
        )
        # nulls occur in exactly rows with weight 0; we did not sample things with
        # probability 0 and did sample things with non-zero probability
        assert all(freq_diffs.filter(pl.col("weight") == 0)["freq"].is_null())
        assert all(~freq_diffs.filter(pl.col("weight") > 0)["freq"].is_null())
        # in rows with non-zero weight, empirical frequencies are close to expected
        assert all(
            np.abs(freq_diffs.filter(pl.col("weight") > 0)["freq_diff"])
            < 4.417 * np.sqrt(freq_diffs.filter(pl.col("weight") > 0)["weight"]
                                * (1 - freq_diffs.filter(pl.col("weight") > 0)["weight"])
                                / (wide_model_out.shape[0] * n_times / 2))
        )


def test_schaake_build_templates_reproducible(obs_data, wide_model_out):
    ss = Schaake(rng = np.random.default_rng(42))
    ss.target_data_train = obs_data
    ss.key_cols = ["location", "age_group"]
    ss.time_col = "date",
    ss.obs_col = "value"
    ss.feat_cols = ["location", "age_group"]
    ss._build_train_X_Y(1, 4)
    n_times = 1000
    templates_1 = ss._build_templates(pl.concat([wide_model_out] * n_times))

    ss = Schaake(rng = np.random.default_rng(42))
    ss.target_data_train = obs_data
    ss.key_cols = ["location", "age_group"]
    ss.time_col = "date",
    ss.obs_col = "value"
    ss.feat_cols = ["location", "age_group"]
    ss._build_train_X_Y(1, 4)
    templates_2 = ss._build_templates(pl.concat([wide_model_out] * n_times))

    assert np.all(templates_1 == templates_2)
