import abc

import numpy as np
import polars as pl

from postpredict import weighters
from postpredict.util import argsort_random_tiebreak


class TimeDependencePostprocessor(abc.ABC):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng


    @abc.abstractmethod
    def fit(self, df, key_cols=None, time_col="date", obs_col="value", feat_cols=["date"], **kwargs):
        """
        Fit a model for temporal dependence across prediction horizons.

        Returns
        -------
        None
        """


    @abc.abstractmethod
    def _build_templates(self, wide_model_out):
        """
        Build dependence templates.
        
        Parameters
        ----------
        wide_model_out: pl.DataFrame
            Model output with sample predictions that do not necessarily capture
            temporal dependence.
        
        Returns
        -------
        templates: np.ndarray
            Dependence templates of shape (wide_model_out.shape[0], self.train_Y.shape[1])
        """


    def transform(self, model_out: pl.DataFrame,
                  reference_time_col: str = "reference_date",
                  horizon_col: str = "horizon", pred_col: str = "value",
                  idx_col: str = "output_type_id"):
        """
        Apply a postprocessing transformation to sample predictions to induce
        dependence across time in the predictive samples.
        
        Parameters
        ----------
        model_out: pl.DataFrame
            polars dataframe with sample predictions that do not necessarily
            capture temporal dependence.
        reference_time_col: str
            name of column in model_out that records the reference time for
            predictions
        horizon_col: str
            name of column in model_out that records the prediction horizon
        pred_col: str
            name of column in model_out with predicted values (samples)
        idx_col: str
            name of column in model_out with sample indices
        
        Returns
        -------
        A copy of the model_out parameter, with sample indices updated so that
        they reflect the estimated temporal dependence structure.
        """
        # pivot model_out from long to wide format
        wide_model_out = self._pivot_horizon(model_out, reference_time_col,
                                             horizon_col, idx_col, pred_col)
        min_horizon = model_out[horizon_col].min()
        max_horizon = model_out[horizon_col].max()
        
        # extract train_X and train_Y from observed data (self.df)
        self._build_train_X_Y(min_horizon, max_horizon)
        
        # perform the transformation, one group at a time
        transformed_wide_model_out = (
            wide_model_out
            .group_by(*(self.key_cols + [reference_time_col]))
            .map_groups(self._transform_one_group)
        )
        
        # unpivot back to long format
        pivot_index = [c for c in model_out.columns if c not in [horizon_col, pred_col]]
        transformed_model_out = (
            transformed_wide_model_out
            .unpivot(
                index = pivot_index,
                on = self.wide_horizon_cols,
                variable_name = horizon_col,
                value_name = pred_col
            )
            .with_columns(
                # convert horizon columns back to original values and data type
                # this is inverting an operation that was done in _pivot_horizon just before the pivot
                pl.col(horizon_col)
                .str.slice(len("postpredict_") + len(horizon_col), None) # keep everything after f"postpredict_{horizon_col}" prefix
                .cast(model_out[horizon_col].dtype)
            )
        )

        return transformed_model_out


    def _transform_one_group(self, wide_model_out):
        templates = self._build_templates(wide_model_out)
        transformed_model_out = self._apply_shuffle(
            wide_model_out = wide_model_out,
            value_cols = self.wide_horizon_cols,
            templates = templates
        )
        return transformed_model_out
    
    
    def _apply_shuffle(self,
                      wide_model_out: pl.DataFrame,
                      value_cols: list[str],
                      templates: np.ndarray) -> pl.DataFrame:
        """
        Given a collection of samples and an equally-sized collection of
        "dependence templates", shuffle the samples to match the rankings in the
        dependence templates. It is assumed that samples are exchangeable,
        i.e. this function should be called with samples for a single
        observational unit (e.g., one location/age group combination).
        
        Parameters
        ----------
        wide_model_out: polars dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        value_cols: character vector of columns in `wide_model_out` that contain
        predicted values over time. These should be given in temporal order.
        templates: numpy array of shape (wide_model_out.shape[0], len(value_cols))
        containing dependence templates.
        
        Returns
        -------
        A copy of the `wide_model_out` argument, with values in the `value_cols`
        columns shuffled to match the rankings in the `templates`.
        
        Notes
        -----
        The argument `wide_model_out` should be in "semi-wide" form, where each
        row corresponds to one sample for one observational unit. Here, an
        observational unit is defined by a combination of keys such as location
        and/or age group. For each such unit and sample, the predictive samples
        should be in a set of columns given in temporal order; for example,
        these might be called `horizon1` through `horizon4`.
        """
        col_orderings = {
            c: argsort_random_tiebreak(templates[:, i], rng = self.rng) \
            for i, c in enumerate(value_cols)
        }

        shuffled_wmo = wide_model_out.clone()
        for c in value_cols:
            shuffled_wmo = shuffled_wmo.with_columns(pl.col(c).sort().alias(c))
            shuffled_wmo[col_orderings[c], c] = shuffled_wmo[c]
        
        return shuffled_wmo


    def _build_train_X_Y(self, min_horizon, max_horizon):
        """
        Build training set data frames self.train_X with features and
        self.train_Y with observed values in windows from min_horizon to
        max_horizon around each time point.
        
        Parameters
        ----------
        min_horizon: int
            minimum prediction horizon
        max_horizon: int
            maximum prediction horizon
        
        Returns
        -------
        None
        
        Notes
        -----
        This method sets self.shift_varnames, self.train_X, and self.train_Y,
        and it updates self.df to have new columns.
        
        It expects the object to have the properties self.df, self.key_cols,
        self.time_col, self.obs_col, and self.feat_cols set already.
        """
        self.shift_varnames = []
        for h in range(min_horizon, max_horizon + 1):
            if h < 0:
                shift_varname = self.obs_col + "_shift_m" + str(abs(h))
            else:
                shift_varname = self.obs_col + "_shift_p" + str(abs(h))
            
            if shift_varname not in self.shift_varnames:
                self.shift_varnames.append(shift_varname)
                self.df = self.df.with_columns(
                    pl.col(self.obs_col)
                    .shift(-h)
                    .over(self.key_cols, order_by=self.time_col)
                    .alias(shift_varname)
                )
        
        df_dropnull = self.df.drop_nulls()
        self.train_X = df_dropnull[self.feat_cols]
        self.train_Y = df_dropnull[self.shift_varnames]


    def _pivot_horizon(self, model_out, reference_time_col, horizon_col,
                       idx_col, pred_col):
        """
        Pivot horizon column wider, overwriting sample indices along the way
        to reflect temporal dependence across horizons within key groups.
        """
        # check that within each group defined by self.key_cols and
        # reference_time_col, each horizon appears the same number of times.
        min_horizon = model_out[horizon_col].min()
        max_horizon = model_out[horizon_col].max()
        expected_horizons = list(range(min_horizon, max_horizon + 1))

        # To try to avoid column name collisions, we do a little namespacing
        # with the prefix "postpredict_"
        horizon_counts = (
            model_out
            .group_by(self.key_cols + [reference_time_col, horizon_col])
            .agg(pl.col(horizon_col).len().alias("postpredict_horizon_count"))
        )
        
        # all horizons from min_horizon to max_horizon are present within all key_col groups
        all_groups_match_expected = (
            horizon_counts[self.key_cols + [reference_time_col, horizon_col]]
            .group_by(self.key_cols + [reference_time_col])
            .all()
            .with_columns(
                pl.col(horizon_col)
                .map_elements(lambda x: list(set(x).symmetric_difference(expected_horizons)) == [], return_dtype=bool)
                .alias("matches_expected_horizons")
            )
            ["matches_expected_horizons"].all()
        )
        if not all_groups_match_expected:
            raise ValueError("Within each key group, model_out must contain predictions at all integer horizons from the smallest to the largest present.")

        # within each key_col and reference_time group, each horizon appears the same number of times
        n_unique_horizon_counts = (
            horizon_counts[self.key_cols + [reference_time_col, "postpredict_horizon_count"]]
            .group_by(self.key_cols)
            .n_unique()
            ["postpredict_horizon_count"]
        )
        if any(n_unique_horizon_counts > 1):
            raise ValueError("Within each key and reference_time group, model_out must contain the same numer of predictions for each horizon.")

        # replace sample indices to have repeated values across horizons within each key and reference_time group,
        # no repeated values across key groups
        horizon_count_by_group = (
            horizon_counts[self.key_cols + [reference_time_col, "postpredict_horizon_count"]]
            .group_by(self.key_cols + [reference_time_col])
            .agg(pl.col("postpredict_horizon_count").first())
        )
        model_out = (
            model_out
            .join(
                horizon_count_by_group.with_columns(
                    postpredict_horizon_cum_count = pl.col("postpredict_horizon_count").cum_sum() - pl.col("postpredict_horizon_count")
                ),
                on = self.key_cols + [reference_time_col]
            )
            .with_columns(
                output_type_id = pl.arange(
                    pl.col("postpredict_horizon_cum_count").first(),
                    pl.col("postpredict_horizon_cum_count").first() + pl.col("postpredict_horizon_count").first()
                )
                .over(self.key_cols + [reference_time_col, horizon_col])
            )
            .drop(["postpredict_horizon_count", "postpredict_horizon_cum_count"])
        )
        
        # perform pivot operation, save resulting new column names to self
        self.wide_horizon_cols = [f"postpredict_{horizon_col}{h}" for h in range(min_horizon, max_horizon + 1)]
        wide_model_out = (
            model_out
            .with_columns(("postpredict_" + horizon_col + pl.col(horizon_col).cast(str)).alias(horizon_col))
            .pivot(
                on=horizon_col,
                index=None,
                values=pred_col
            )
        )
        
        return wide_model_out



class Schaake(TimeDependencePostprocessor):
    def __init__(self, weighter=weighters.EqualWeighter(),
                 rng: np.random.Generator = np.random.default_rng()) -> None:
        self.weighter = weighter
        super().__init__(rng)


    def fit(self, df, key_cols=None, time_col="date", obs_col="value", feat_cols=["date"]):
        """
        Fit a Schaake shuffle model for temporal dependence across prediction
        horizons. In practice this just involves saving the input arguments for
        later use; the Schaake shuffle does not require any parameter estimation.
        
        Parameters
        ----------
        df: polars dataframe with training set observations.
        key_cols: names of columns in `df` used to identify observational units,
        e.g. location or age group.
        time_col: name of column in `df` that contains the time index.
        obs_col: name of column in `df` that contains observed values.
        feat_cols: names of columns in `df` with features
        
        Returns
        -------
        None
        """
        self.df = df
        self.key_cols = key_cols
        self.time_col = time_col
        self.obs_col = obs_col
        self.feat_cols = feat_cols
    
    
    def _build_templates(self, wide_model_out):
        """
        Build dependence templates for use with the Schaake shuffle.
        """
        # weights shape is (n_test, n_train)
        weights = self.weighter.get_weights(self.train_X, wide_model_out[:, self.feat_cols])

        # draw one sample from each distribution in a batch of (n_test,)
        # categorical distributions, each over the n_train categories
        # representing sequences in rows of self.train_Y
        selected_inds = [self.rng.choice(weights.shape[1], size=1, p=weights[i, :])[0] \
                         for i in range(weights.shape[0])]

        # get the templates
        templates = self.train_Y[selected_inds, :]
        return templates
