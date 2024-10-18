# Tests for postpredict.dependence.TimeDependencePostprocessor._pivot_horizon

from datetime import datetime
from itertools import product

import numpy as np
import polars as pl
import pytest
from postpredict.dependence import TimeDependencePostprocessor


def test_pivot_horizon_positive_horizon(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _pivot_horizon method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=long_model_out,
        reference_time_col="reference_date",
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns
    expected_cols = set(tdp.key_cols + tdp.feat_cols +
                        ["reference_date", "output_type", "output_type_id"] +
                        [f"postpredict_horizon{h}" for h in range(1, 4)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    group_horizon_iterable = product(
        [datetime.strptime(d, "%Y-%m-%d") for d in ["2020-01-15", "2020-01-22"]],
        ["a", "b"],
        ["young", "old"],
        range(1, 4)
    )
    for reference_date, location, age_group, h in group_horizon_iterable:
        expected_values = (
            long_model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group) &
                (pl.col("horizon") == h)
            )
            [:, "value"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        actual_values = (
            wide_model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group)
            )
            [:, f"postpredict_horizon{h}"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        
        assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)



def test_pivot_horizon_negative_horizon(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _pivot_horizon method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    
    model_out = long_model_out.with_columns(horizon = pl.col("horizon") - 2)
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=model_out,
        reference_time_col="reference_date",
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns
    expected_cols = set(tdp.key_cols + tdp.feat_cols +
                        ["reference_date", "output_type", "output_type_id"] +
                        [f"postpredict_horizon{h}" for h in range(-1, 2)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    group_horizon_iterable = product(
        [datetime.strptime(d, "%Y-%m-%d") for d in ["2020-01-15", "2020-01-22"]],
        ["a", "b"],
        ["young", "old"],
        range(-1, 2)
    )
    for reference_date, location, age_group, h in group_horizon_iterable:
        expected_values = (
            model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group) &
                (pl.col("horizon") == h)
            )
            [:, "value"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        actual_values = (
            wide_model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group)
            )
            [:, f"postpredict_horizon{h}"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        
        assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)


def test_pivot_horizon_diff_sample_count_by_group(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _pivot_horizon method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    
    model_out = long_model_out.filter(
        ((pl.col("location") == "a") & (pl.col("age_group") == "young") & (pl.col("output_type_id") < 5)) |
        ((pl.col("location") == "a") & (pl.col("age_group") == "old") & (pl.col("output_type_id") > 2)) |
        ((pl.col("location") == "b") & (pl.col("age_group") == "young") & (pl.col("output_type_id") < 7)) |
        ((pl.col("location") == "b") & (pl.col("age_group") == "old") & (pl.col("output_type_id") < 4))
    )
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=model_out,
        reference_time_col="reference_date",
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns
    expected_cols = set(tdp.key_cols + tdp.feat_cols +
                        ["reference_date", "output_type", "output_type_id"] +
                        [f"postpredict_horizon{h}" for h in range(1, 4)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    group_horizon_iterable = product(
        [datetime.strptime(d, "%Y-%m-%d") for d in ["2020-01-15", "2020-01-22"]],
        ["a", "b"],
        ["young", "old"],
        range(1, 4)
    )
    for reference_date, location, age_group, h in group_horizon_iterable:
        expected_values = (
            model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group) &
                (pl.col("horizon") == h)
            )
            [:, "value"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        actual_values = (
            wide_model_out
            .filter(
                (pl.col("reference_date") == reference_date) &
                (pl.col("location") == location) &
                (pl.col("age_group") == age_group)
            )
            [:, f"postpredict_horizon{h}"]
            .to_numpy()
            .flatten()
            .tolist()
        )
        
        assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)


def test_pivot_horizon_diff_sample_count_by_horizon_same_group_errors(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _pivot_horizon method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    
    model_out = long_model_out.filter(
        ((pl.col("location") == "a") & (pl.col("age_group") == "young") &
            ((pl.col("output_type_id") < 5) | ((pl.col("output_type_id") < 7) & (pl.col("horizon") == 2)))) |
        ((pl.col("location") == "a") & (pl.col("age_group") == "old") & (pl.col("output_type_id") > 2)) |
        ((pl.col("location") == "b") & (pl.col("age_group") == "young") & (pl.col("output_type_id") < 7)) |
        ((pl.col("location") == "b") & (pl.col("age_group") == "old") & (pl.col("output_type_id") < 4))
    )
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    with pytest.raises(ValueError):
        tdp._pivot_horizon(
            model_out=model_out,
            reference_time_col="reference_date",
            horizon_col="horizon",
            idx_col="output_type_id",
            pred_col="value"
        )


def test_pivot_horizon_missing_horizon_errors(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _pivot_horizon method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor(rng = np.random.default_rng(42))
    
    model_out = long_model_out.filter(
        ~((pl.col("reference_date") == datetime.strptime("2020-01-15", "%Y-%m-%d")) &
          (pl.col("location") == "a") & (pl.col("age_group") == "young") &
          (pl.col("horizon") == 2))
    )
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    with pytest.raises(ValueError):
        tdp._pivot_horizon(
            model_out=model_out,
            reference_time_col="reference_date",
            horizon_col="horizon",
            idx_col="output_type_id",
            pred_col="value"
        )
