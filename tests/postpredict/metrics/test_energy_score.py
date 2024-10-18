# Tests for postpredict.metrics.energy_score

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from postpredict.metrics import energy_score


def test_energy_score():
    model_out_wide = pl.concat([
        pl.DataFrame({
            "location": "a",
            "date": datetime.strptime("2024-10-01", "%Y-%m-%d"),
            "output_type": "sample",
            "output_type_id": [1.0, 2.0, 3.0, 4.0],
            "horizon1": [5.0, 7.7, 18.0, 10.0],
            "horizon2": [3.0, 4.0, 10.0, 6.0],
            "horizon3": [4.4, 1.0, 12.0, 9.0]
        }),
        pl.DataFrame({
            "location": "b",
            "date": datetime.strptime("2024-10-08", "%Y-%m-%d"),
            "output_type": "sample",
            "output_type_id": [5.0, 6.0, 7.0, 8.0],
            "horizon1": [6.0, 4.0, 5.0, 2.0],
            "horizon2": [12.0, 0.0, 15.0, 6.0],
            "horizon3": [16.6, 21.0, 32.0, -1.0]
        })
    ])
    obs_data_wide = pl.DataFrame({
        "location": ["a", "a", "b", "b"],
        "date": [datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d"),
                 datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d")],
        "value": [3.0, 4.0, 0.0, 7.2],
        "value_lead1": [4.0, 10.0, 7.2, 9.6],
        "value_lead2": [10.0, 5.0, 9.6, 10.0],
        "value_lead3": [5.0, 2.0, 10.0, 14.1]
    })
    
    # expected scores calculated in R using the scoringRules package:
    # library(scoringRules)
    # X <- matrix(
    #     data = c(5.0, 7.7, 18.0, 10.0, 3.0, 4.0, 10.0, 6.0, 4.4, 1.0, 12.0, 9.0),
    #     nrow = 3, ncol = 4,
    #     byrow = TRUE
    # )
    # y <- c(4.0, 10.0, 5.0)
    # print(es_sample(y, X), digits = 20)
    # X <- matrix(
    #     data = c(6.0, 4.0, 5.0, 2.0, 12.0, 0.0, 15.0, 6.0, 16.6, 21.0, 32.0, -1.0),
    #     nrow = 3, ncol = 4,
    #     byrow = TRUE
    # )
    # y <- c(9.6, 10.0, 14.1)
    # print(es_sample(y, X), digits = 20)
    expected_scores_df = pl.DataFrame({
        "location": ["a", "b"],
        "date": [datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d")],
        "energy_score": [5.8560677725938221627, 5.9574451598773787708]
    })
    
    actual_scores_df = energy_score(model_out_wide=model_out_wide,
                                    obs_data_wide=obs_data_wide,
                                    key_cols=["location", "date"],
                                    pred_cols=["horizon1", "horizon2", "horizon3"],
                                    obs_cols=["value_lead1", "value_lead2", "value_lead3"],
                                    reduce_mean=False)
    
    assert_frame_equal(actual_scores_df, expected_scores_df, check_row_order=False, atol=1e-19)
    
    expected_mean_score = np.mean([5.8560677725938221627, 5.9574451598773787708])
    actual_mean_score = energy_score(model_out_wide=model_out_wide,
                                     obs_data_wide=obs_data_wide,
                                     key_cols=["location", "date"],
                                     pred_cols=["horizon1", "horizon2", "horizon3"],
                                     obs_cols=["value_lead1", "value_lead2", "value_lead3"],
                                     reduce_mean=True)
    assert actual_mean_score == pytest.approx(expected_mean_score)
