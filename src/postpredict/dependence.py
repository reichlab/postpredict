import abc

import numpy as np
import polars as pl

from postpredict import weighters
from postpredict.metrics import marginal_pit
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
        templates: pl.DataFrame
            Dependence templates of shape (wide_model_out.shape[0], self.train_Y.shape[1])
        """


    def transform(self, model_out: pl.DataFrame,
                  obs_mask: np.ndarray | None = None,
                  pit_templates: bool = False,
                  return_long_format: bool = True):
        """
        Apply a postprocessing transformation to sample predictions to induce
        dependence across time in the predictive samples.
        
        Parameters
        ----------
        model_out: pl.DataFrame
            polars dataframe with sample predictions that do not necessarily
            capture temporal dependence.
        obs_mask: np.ndarray | None
            mask to use for observed data. The primary use case is to support
            cross-validation. If None, all observed data are used to form
            dependence templates. Otherwise, `obs_mask` should be a boolean
            array of shape (self.target_data_train.shape[0], ). Rows of self.target_data_train where obs_mask
            is True will be used, while rows of self.target_data_train where obs_mask is False
            will not be used.
        pit_templates: bool
            If False (default), templates are based on observed values. If True,
            templates are based on PIT values for past forecasts.
        return_long_format: bool
            If True, return long format. If False, return wide format with
            horizon pivoted into columns.
        
        Returns
        -------
        A copy of the model_out parameter, with sample indices updated so that
        they reflect the estimated temporal dependence structure.
        """
        # pivot model_out from long to wide format
        wide_model_out = self._pivot_horizon(model_out)
        min_horizon = model_out[self.horizon_col].min()
        max_horizon = model_out[self.horizon_col].max()
        
        if self.model_out_train is not None:
            wide_model_out_train = self._pivot_horizon(self.model_out_train)
        else:
            wide_model_out_train = None
        
        # extract train_X and train_Y from observed data (self.target_data_train)
        # and/or past forecasts (wide_model_out_train)
        self._build_train_X_Y(min_horizon, max_horizon, obs_mask,
                              wide_model_out_train, self.reference_time_col,
                              pit_templates)
        
        # perform the transformation, one group at a time
        transformed_wide_model_out = (
            wide_model_out
            .group_by(*(self.key_cols + [self.reference_time_col]))
            .map_groups(self._transform_one_group)
        )
        
        if not return_long_format:
            return transformed_wide_model_out
        
        # unpivot back to long format
        pivot_index = [c for c in model_out.columns if c not in [self.horizon_col, self.pred_col]]
        transformed_model_out = (
            transformed_wide_model_out
            .unpivot(
                index = pivot_index,
                on = self.wide_horizon_cols,
                variable_name = self.horizon_col,
                value_name = self.pred_col
            )
            .with_columns(
                # convert horizon columns back to original values and data type
                # this is inverting an operation that was done in _pivot_horizon just before the pivot
                pl.col(self.horizon_col)
                .str.slice(len("postpredict_") + len(self.horizon_col), None) # keep everything after f"postpredict_{horizon_col}" prefix
                .cast(model_out[self.horizon_col].dtype)
            )
        )
        
        return transformed_model_out


    def _transform_one_group(self, wide_model_out):
        templates = self._build_templates(wide_model_out).to_numpy()
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


    def _build_train_X_Y(self, min_horizon: int, max_horizon: int,
                         obs_mask: np.ndarray | None = None,
                         wide_model_out: pl.DataFrame | None = None,
                         reference_time_col: str | None = None,
                         pit_templates: bool = False) -> None:
        """
        Build training set data frames self.train_X with features and
        self.train_Y with candidate dependence templates based on either
        observed values in windows from min_horizon to max_horizon around each
        time point or PIT values derived from past forecasts.
        
        Parameters
        ----------
        min_horizon: int
            minimum prediction horizon
        max_horizon: int
            maximum prediction horizon
        obs_mask: np.ndarray | None
            mask to use for observed data. The primary use case is to support
            cross-validation. If None, all observed data are used to form
            dependence templates. Otherwise, `obs_mask` should be a boolean
            array of shape (self.target_data_train.shape[0], ). Rows of self.target_data_train where obs_mask
            is True will be used, while rows of self.target_data_train where obs_mask is False
            will not be used.
        wide_model_out: pl.DataFrame
            polars dataframe with sample predictions that do not necessarily
            capture temporal dependence, in wide format with horizons in columns.
            Only needed if pit_templates = True.
        reference_time_col: str
            name of column in wide_model_out that records the reference time for
            predictions. Only needed if pit_templates = True
        pit_templates: bool
            If False (default), templates are based on observed values. If True,
            templates are based on PIT values for past forecasts.
        
        Returns
        -------
        None
        
        Notes
        -----
        This method sets self.shift_varnames, self.train_X, and self.train_Y,
        and it updates self.target_data_train to have new columns.
        
        It expects the object to have the properties self.target_data_train, self.key_cols,
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
                self.target_data_train = self.target_data_train.with_columns(
                    pl.col(self.obs_col)
                    .shift(-h)
                    .over(self.key_cols, order_by=self.time_col)
                    .alias(shift_varname)
                )
        
        if obs_mask is None:
            obs_mask = True
        df_mask_and_dropnull = self.target_data_train.filter(obs_mask).drop_nulls()

        if pit_templates:
            pit_values = (
                marginal_pit(
                    model_out_wide = wide_model_out,
                    obs_data_wide = df_mask_and_dropnull.with_columns(pl.col(self.time_col).alias(reference_time_col)),
                    index_cols = self.key_cols + [reference_time_col],
                    pred_cols = self.wide_horizon_cols,
                    obs_cols = self.shift_varnames
                )
                .join(
                    wide_model_out[list(set(self.key_cols + [reference_time_col] + self.feat_cols))].unique(),
                    on=self.key_cols + [reference_time_col],
                    how="left"
                )
            )
            train_X_Y_source = pit_values
            train_Y_cols = [f"pit_{pred_c}" for pred_c in self.wide_horizon_cols]
        else:
            train_X_Y_source = df_mask_and_dropnull
            train_Y_cols = self.shift_varnames
        
        self.train_X = train_X_Y_source[self.feat_cols]
        self.train_Y = train_X_Y_source[train_Y_cols]


    def _pivot_horizon(self, model_out):
        """
        Pivot horizon column wider, overwriting sample indices along the way
        to reflect temporal dependence across horizons within key groups.
        """
        # check that within each group defined by self.key_cols and
        # reference_time_col, each horizon appears the same number of times.
        min_horizon = model_out[self.horizon_col].min()
        max_horizon = model_out[self.horizon_col].max()
        expected_horizons = list(range(min_horizon, max_horizon + 1))

        # To try to avoid column name collisions, we do a little namespacing
        # with the prefix "postpredict_"
        horizon_counts = (
            model_out
            .group_by(self.key_cols + [self.reference_time_col, self.horizon_col])
            .agg(pl.col(self.horizon_col).len().alias("postpredict_horizon_count"))
        )
        
        # all horizons from min_horizon to max_horizon are present within all key_col groups
        all_groups_match_expected = (
            horizon_counts[self.key_cols + [self.reference_time_col, self.horizon_col]]
            .group_by(self.key_cols + [self.reference_time_col])
            .all()
            .with_columns(
                pl.col(self.horizon_col)
                .map_elements(lambda x: list(set(x).symmetric_difference(expected_horizons)) == [], return_dtype=bool)
                .alias("matches_expected_horizons")
            )
            ["matches_expected_horizons"].all()
        )
        if not all_groups_match_expected:
            raise ValueError("Within each key group, model_out must contain predictions at all integer horizons from the smallest to the largest present.")

        # within each key_col and reference_time group, each horizon appears the same number of times
        n_unique_horizon_counts = (
            horizon_counts[self.key_cols + [self.reference_time_col, "postpredict_horizon_count"]]
            .group_by(self.key_cols)
            .n_unique()
            ["postpredict_horizon_count"]
        )
        if any(n_unique_horizon_counts > 1):
            raise ValueError("Within each key and reference_time group, model_out must contain the same numer of predictions for each horizon.")

        # replace sample indices to have repeated values across horizons within each key and reference_time group,
        # no repeated values across key groups
        horizon_count_by_group = (
            horizon_counts[self.key_cols + [self.reference_time_col, "postpredict_horizon_count"]]
            .group_by(self.key_cols + [self.reference_time_col])
            .agg(pl.col("postpredict_horizon_count").first())
        )
        model_out = (
            model_out
            .join(
                horizon_count_by_group.with_columns(
                    postpredict_horizon_cum_count = pl.col("postpredict_horizon_count").cum_sum() - pl.col("postpredict_horizon_count")
                ),
                on = self.key_cols + [self.reference_time_col]
            )
            .with_columns(
                output_type_id = pl.arange(
                    pl.col("postpredict_horizon_cum_count").first(),
                    pl.col("postpredict_horizon_cum_count").first() + pl.col("postpredict_horizon_count").first()
                )
                .over(self.key_cols + [self.reference_time_col, self.horizon_col])
            )
            .drop(["postpredict_horizon_count", "postpredict_horizon_cum_count"])
        )
        
        # perform pivot operation, save resulting new column names to self
        self.wide_horizon_cols = [f"postpredict_{self.horizon_col}{h}" for h in range(min_horizon, max_horizon + 1)]
        wide_model_out = (
            model_out
            .with_columns(("postpredict_" + self.horizon_col + pl.col(self.horizon_col).cast(str)).alias(self.horizon_col))
            .pivot(
                on=self.horizon_col,
                index=None,
                values=self.pred_col
            )
        )
        
        return wide_model_out



class Schaake(TimeDependencePostprocessor):
    def __init__(self, weighter=weighters.EqualWeighter(),
                 rng: np.random.Generator = np.random.default_rng()) -> None:
        self.weighter = weighter
        super().__init__(rng)


    def fit(self,
            target_data_train: pl.DataFrame,
            model_out_train: pl.DataFrame,
            key_cols: list[str] | None = None,
            time_col: str = "date",
            obs_col: str = "value",
            reference_time_col: str = "reference_date",
            horizon_col: str = "horizon",
            pred_col: str = "value",
            idx_col: str = "output_type_id",
            feat_cols: list[str] = ["date"]) -> None:
        """
        Fit a Schaake shuffle model for temporal dependence across prediction
        horizons. In practice this just involves saving the input arguments for
        later use; the Schaake shuffle does not require any parameter estimation.
        
        Parameters
        ----------
        target_data_train: pl.DataFrame
            training set observations of target data.
        model_out_train: pl.DataFrame
            training set predictions
        key_cols: list[str] | None
            names of columns in `target_data_train` and `model_out_train` used
            to identify observational units, e.g. location or age group.
        time_col: str
            name of column in `target_data_train` that contains the time index.
        obs_col: str
            name of column in `target_data_train` that contains observed values.
        reference_time_col: str
            name of column in `model_out_train` that contains the reference time
            for model predictions
        horizon_col: str
            name of column in `model_out_train` that contains the prediction
            horizon relative to the reference time
        pred_col: str
            name of column in model_out with predicted values (samples)
        idx_col: str
            name of column in model_out with sample indices
        feat_cols: list[str]
            names of columns in `target_data_train` and `model_out_train` with features
        
        Returns
        -------
        None
        """
        self.target_data_train = target_data_train
        self.model_out_train = model_out_train
        self.key_cols = key_cols
        self.time_col = time_col
        self.obs_col = obs_col
        self.reference_time_col = reference_time_col
        self.horizon_col = horizon_col
        self.pred_col = pred_col
        self.idx_col = idx_col
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
