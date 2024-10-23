# Tests for postpredict.dependence.TimeDependencePostprocessor._build_train_X_Y

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


def test_build_train_X_Y_positive_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "date"]

    tdp._build_train_X_Y(1, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 12 + ["b"] * 12,
        "age_group": (["young"] * 6 + ["old"] * 6) * 2,
        "date": [datetime.strptime("2020-01-14", "%Y-%m-%d") + timedelta(i) for i in range(6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_p1": list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)),
        "value_shift_p2": list(range(12, 18)) + list(range(22, 28)) + list(range(32, 38)) + list(range(42, 48)),
        "value_shift_p3": list(range(13, 19)) + list(range(23, 29)) + list(range(33, 39)) + list(range(43, 49)),
        "value_shift_p4": list(range(14, 20)) + list(range(24, 30)) + list(range(34, 40)) + list(range(44, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_Y_nonnegative_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "date"]

    tdp._build_train_X_Y(0, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 12 + ["b"] * 12,
        "age_group": (["young"] * 6 + ["old"] * 6) * 2,
        "date": [datetime.strptime("2020-01-14", "%Y-%m-%d") + timedelta(i) for i in range(6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_p0": list(range(10, 16)) + list(range(20, 26)) + list(range(30, 36)) + list(range(40, 46)),
        "value_shift_p1": list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)),
        "value_shift_p2": list(range(12, 18)) + list(range(22, 28)) + list(range(32, 38)) + list(range(42, 48)),
        "value_shift_p3": list(range(13, 19)) + list(range(23, 29)) + list(range(33, 39)) + list(range(43, 49)),
        "value_shift_p4": list(range(14, 20)) + list(range(24, 30)) + list(range(34, 40)) + list(range(44, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_Y_negative_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "date"]

    tdp._build_train_X_Y(-1, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 10 + ["b"] * 10,
        "date": [datetime.strptime("2020-01-14", "%Y-%m-%d") + timedelta(i) for i in range(1, 6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_m1": list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35)) + list(range(40, 45)),
        "value_shift_p0": list(range(11, 16)) + list(range(21, 26)) + list(range(31, 36)) + list(range(41, 46)),
        "value_shift_p1": list(range(12, 17)) + list(range(22, 27)) + list(range(32, 37)) + list(range(42, 47)),
        "value_shift_p2": list(range(13, 18)) + list(range(23, 28)) + list(range(33, 38)) + list(range(43, 48)),
        "value_shift_p3": list(range(14, 19)) + list(range(24, 29)) + list(range(34, 39)) + list(range(44, 49)),
        "value_shift_p4": list(range(15, 20)) + list(range(25, 30)) + list(range(35, 40)) + list(range(45, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_Y_mask(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "date"]

    mask = (obs_data["date"] <= datetime.strptime("2020-01-15", "%Y-%m-%d")) \
        | (obs_data["date"] >= datetime.strptime("2020-01-19", "%Y-%m-%d"))
    tdp._build_train_X_Y(1, 4, obs_mask = mask)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 6 + ["b"] * 6,
        "age_group": (["young"] * 3 + ["old"] * 3) * 2,
        "date": [datetime.strptime(d, "%Y-%m-%d") for d in ["2020-01-14", "2020-01-15", "2020-01-19"]] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_p1": [11, 12, 16] + [21, 22, 26] + [31, 32, 36] + [41, 42, 46],
        "value_shift_p2": [12, 13, 17] + [22, 23, 27] + [32, 33, 37] + [42, 43, 47],
        "value_shift_p3": [13, 14, 18] + [23, 24, 28] + [33, 34, 38] + [43, 44, 48],
        "value_shift_p4": [14, 15, 19] + [24, 25, 29] + [34, 35, 39] + [44, 45, 49]
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_Y_pit_templates(obs_data, wide_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "reference_date"]
    tdp.wide_horizon_cols = ["horizon1", "horizon2", "horizon3"]

    tdp._build_train_X_Y(1, 3, obs_mask=None, wide_model_out=wide_model_out,
                         reference_time_col="reference_date",
                         pit_templates=True)
    
    # only include 2020-01-15; expect all rows for 2020-01-22 to be dropped
    # due to null values
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 2 + ["b"] * 2,
        "age_group": ["young", "old"] * 2,
        "reference_date": [datetime.strptime("2020-01-15", "%Y-%m-%d")] * 4
    })
    
    # For a/young, a/old, b/young, b/old, what proportion of samples are <=
    # the observed value at each horizon?  see fixtures in conftest.py
    expected_train_Y = pl.DataFrame({
        "pit_horizon1": [0.8, 1.0, 1.0, 1.0],
        "pit_horizon2": [0.8, 1.0, 1.0, 1.0],
        "pit_horizon3": [0.2, 0.9, 0.9, 1.0]
    })
    
    # check each separately up to row order
    assert_frame_equal(tdp.train_X, expected_train_X, check_row_order=False)
    assert_frame_equal(tdp.train_Y, expected_train_Y, check_row_order=False)
    
    # check joined results up to row order
    assert_frame_equal(
        pl.concat([expected_train_X, expected_train_Y], how="horizontal"),
        pl.concat([tdp.train_X, tdp.train_Y], how="horizontal"),
        check_row_order=False
    )


def test_build_train_X_Y_pit_templates_mask(obs_data, wide_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _build_train_X_Y method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    tdp.target_data_train = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "reference_date"]
    tdp.wide_horizon_cols = ["horizon1", "horizon2", "horizon3"]
    
    # we keep everything other than the b/young combination
    mask = (obs_data["location"] != "b") | (obs_data["age_group"] != "young")

    tdp._build_train_X_Y(1, 3, obs_mask=mask, wide_model_out=wide_model_out,
                         reference_time_col="reference_date",
                         pit_templates=True)
    
    # only include 2020-01-15; expect all rows for 2020-01-22 to be dropped
    # due to null values
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 2 + ["b"],
        "age_group": ["young", "old", "old"],
        "reference_date": [datetime.strptime("2020-01-15", "%Y-%m-%d")] * 3
    })
    
    # For a/young, a/old, b/old, what proportion of samples are <=
    # the observed value at each horizon?  see fixtures in conftest.py
    expected_train_Y = pl.DataFrame({
        "pit_horizon1": [0.8, 1.0, 1.0],
        "pit_horizon2": [0.8, 1.0, 1.0],
        "pit_horizon3": [0.2, 0.9, 1.0]
    })
    
    # check each separately up to row order
    assert_frame_equal(tdp.train_X, expected_train_X, check_row_order=False)
    assert_frame_equal(tdp.train_Y, expected_train_Y, check_row_order=False)
    
    # check joined results up to row order
    assert_frame_equal(
        pl.concat([expected_train_X, expected_train_Y], how="horizontal"),
        pl.concat([tdp.train_X, tdp.train_Y], how="horizontal"),
        check_row_order=False
    )
