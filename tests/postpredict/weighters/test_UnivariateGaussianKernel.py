# Tests for postpredict.metrics.energy_score

import numpy as np
import pytest
from postpredict.weighters import UnivariateGaussianKernel
from scipy.stats import norm


def test_UnivariateGaussianKernel():
    test_X = np.arange(-3, 4)[:, np.newaxis]
    train_X = np.arange(-10, 11)[:, np.newaxis]
    
    def get_one_weights_row(test_x, train_X):
        w = norm.pdf(train_X, loc=test_x, scale=np.sqrt(2.4))
        return (w / np.sum(w)).reshape(1, train_X.shape[0])
    
    expected_weights = np.concat(
        [get_one_weights_row(test_x, train_X) for test_x in test_X],
        axis = 0
    )
    
    weighter = UnivariateGaussianKernel(h = 2.4)
    actual_weights = weighter.get_weights(train_X, test_X)
    
    assert actual_weights == pytest.approx(expected_weights)
