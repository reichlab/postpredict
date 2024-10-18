import collections

import numpy as np


class Parameter(collections.UserDict):
    def __init__(self, value: np.ndarray, trainable: bool = True):
        self.value = value
        self.trainable = trainable


class EqualWeighter():
    def __init__(self) -> None:
        pass
    
    
    def get_weights(self, train_X, test_X):
        """
        Compute training set observation weights.
        
        Parameters
        ----------
        train_X: dataframe or array of shape (n_train, p)
            Training set features used for weighting. There is one row for each
            training set instance and one column for each of the p features.
        test_X: dataframe or array of shape (n_test, p)
            Test set features used for weighting. There is one row for each test
            set instance and one column for each of the p features.
        
        Returns
        -------
        numpy array of shape (n_test, n_train) with weights for each training set
        instance, where weights sum to 1 within each row.
        """
        n_train = train_X.shape[0]
        n_test = test_X.shape[0]
        return np.full((n_test, n_train), 1 / n_train)


class UnivariateGaussianKernel():
    def __init__(self, h):
        self.parameters = {
            "h": Parameter(
                    value = h,
                    trainable = True
                )
        }
    
    
    def get_weights(self, train_X, test_X):
        """
        Compute training set observation weights.
        
        Parameters
        ----------
        train_X: dataframe or array of shape (n_train, 1)
            Training set features used for weighting. There is one row for each
            training set instance and one column for a single feature.
        test_X: dataframe or array of shape (n_test, 1)
            Test set features used for weighting. There is one row for each
            test set instance and one column for a single feature.
        
        Returns
        -------
        numpy array of shape (n_test, n_train) with weights for each training set
        instance, where weights sum to 1 within each row.
        """
        n_train = train_X.shape[0]
        prop_weights = np.exp(-0.5 / self.parameters["h"].value * (test_X - train_X.reshape(1, n_train))**2)
        return prop_weights / np.sum(prop_weights, axis = 1, keepdims = True)
