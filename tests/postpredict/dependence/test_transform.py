# Tests for postpredict.dependence.TimeDependencePostprocessor.transform

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


def test_transform(obs_data, long_model_out, templates, long_expected_final, monkeypatch, mocker):
    # this tests the full transformation pipeline defined in
    # TimeDependencePostprocessor, *other than* the _build_templates method,
    # which is to be implemented by a subclass of the abstract base class.
    # For this test, we use the fixed templates defined as a test fixture.
    
    # define a concrete subclass of TimeDependencePostprocessor whose
    # _build_templates method returns the templates fixture
    class TestPostprocessor(TimeDependencePostprocessor):
        def fit(self, df, key_cols=None, time_col="date", obs_col="value", feat_cols=["date"]):
            pass
        
        
        def _build_templates(self, wide_model_out):
            return templates
    
    tdp = TestPostprocessor(rng = np.random.default_rng(42))
    tdp.df = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    
    # perform the transform operation
    actual_final = tdp.transform(
        model_out=long_model_out,
        reference_time_col="reference_date",
        horizon_col="horizon",
        pred_col="value",
        idx_col="output_type_id"
    )
    
    # Challenge: tdp.transform performs the transformation on the
    # groups defined by combinations of location and age_group in a random
    # order!  As a result, the output_type_ids are not assigned values in a
    # consistent order.
    #
    # In order to check equality with expected results, we force the
    # output_type_id in long_expected_final to start at the index that was used
    # for actual_final within each reference_date/location/age_group combination.
    # To maintain a valid test that indices are distinct across different
    # location/age_group combinations, we first check that the minimal index in
    # actual_final is distinct across the different groups.
    assert all(
        actual_final["reference_date", "location", "age_group", "output_type_id"]
        .group_by("reference_date", "location", "age_group")
        .min()
        ["output_type_id"]
        .value_counts()
        ["count"] == 1
    )
    
    # now, update the indices in the output_type_id column
    min_actual_index = (
        actual_final["reference_date", "location", "age_group", "output_type_id"]
        .group_by("reference_date", "location", "age_group")
        .min()
        .with_columns(actual_min_output_type_id = pl.col("output_type_id"))
        .drop("output_type_id")
    )
    min_expected_index = (
        long_expected_final["reference_date", "location", "age_group", "output_type_id"]
        .group_by("reference_date", "location", "age_group")
        .min()
        .with_columns(expected_min_output_type_id = pl.col("output_type_id"))
        .drop("output_type_id")
    )
    long_expected_final = (
        long_expected_final
        .join(min_actual_index, on=["reference_date", "location", "age_group"], how="left")
        .join(min_expected_index, on=["reference_date", "location", "age_group"], how="left")
        .with_columns(
            output_type_id = pl.col("output_type_id") - pl.col("expected_min_output_type_id") + pl.col("actual_min_output_type_id")
        )
        .drop("actual_min_output_type_id", "expected_min_output_type_id")
    )
    
    # assert equality with expected results, within each location/age_group combination
    assert_frame_equal(
        actual_final,
        long_expected_final,
        check_row_order=False
    )
