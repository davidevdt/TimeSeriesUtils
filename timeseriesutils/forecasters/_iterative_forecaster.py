from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

class IterativeForecaster(BaseEstimator, RegressorMixin):
    """
    A custom forecasting model that iteratively predicts future values using
    a provided regression model. Supports adding lagged features for better forecasting.

    Parameters
    ----------
    model : estimator object
        The machine learning model to use for forecasting.
    
    pre_trained : bool, optional (default=False)
        If True, the model is pre-trained and doesn't need to be fit again.
    """

    def __init__(self, model, pre_trained=False):
        self.model = model
        self.pre_trained = pre_trained

    def fit(self, X, y):
        """Fit the model to the training data."""
        if not self.pre_trained:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return self.model.predict(X)

    def score(self, *args, **kwargs):
        """Return the R^2 score of the model."""
        return self.model.score(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        """Get parameters for this estimator."""
        return self.model.get_params(*args, **kwargs)

    def forecast(self, y_train=None, y_test=None, X_train=None, X_test=None,
                horizon=1, steps=1, single_step_forecasts=False, freq=None,
                lags_dict=None, max_lags=None, target_name=None, return_X=False,
                last_obs_date=None):
        """
        Generate forecasts for a specified number of steps ahead.

        Parameters
        ----------
        y_train : pd.Series or None
            The training target series.
        y_test : pd.Series or None
            The test target series.
        X_train : pd.DataFrame or None
            The training feature dataframe.
        X_test : pd.DataFrame or None
            The test feature dataframe.
        horizon : int, optional (default=1)
            Number of steps ahead to start forecasting. Automatically set to 1 if the provided value islower than that.  
        steps : int, optional (default=1)
            Number of forecasting steps to perform.
        single_step_forecasts : bool, optional (default=False)
            Whether to forecast one step at a time or all steps at once.
        freq : pd.DateOffset or None
            Frequency of the data (e.g., daily, monthly).
        lags_dict : dict or None
            A dictionary of columns and their respective lags to include in the model.
        max_lags : int or None
            Maximum number of lags to consider.
        target_name : str or None
            Name of the target variable.
        return_X : bool, optional (default=False)
            Whether to return the expanded dataset along with the forecasts.
        last_obs_date : pd.Timestamp or None
            The last observed date in the training data.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the forecasted values.
        """

        if y_train is None and X_train is None and X_test is None:
            raise ValueError("Either y_train, X_train, or X_test must be provided.")

        # Validate and ensure consistency between train and test sets
        if X_train is not None and X_test is not None:
            self._validate_consistency(X_train, X_test, "X_train", "X_test")

        if y_train is not None and y_test is not None:
            self._validate_consistency(y_train, y_test, "y_train", "y_test")

        # Min. horizon = 1 (we start forecasting after the last observed value)
        if horizon <= 0:
            horizon = 1 

        # Determine max_lags if not provided
        if max_lags is None:
            if lags_dict is not None:
                max_lags = max(max(lags) for lags in lags_dict.values())
            else:
                max_lags = 0 

        if target_name is None:
            target_name = 'Target'

        # Determine forecasting dates and prepare the forecasting dataframe
        X_forecast, forecasting_dates = self._prepare_forecasting_data(
            X_train, X_test, freq, horizon, steps, max_lags, last_obs_date, target_name, y_train, y_test
        )

        # Dictionary to store forecasted values
        forecasted_values = {"Time": [], target_name: []}

        # Iteratively forecast
        for date in forecasting_dates:
            new_row = X_forecast.loc[[date]].drop(columns=target_name)
            row_idx = X_forecast.index.get_loc(date)

            if lags_dict is not None:
                new_row = self._add_lagged_features(new_row, X_forecast, row_idx, lags_dict, horizon)

            prediction = self.model.predict(new_row)[0]
            forecasted_values["Time"].append(date)
            forecasted_values[target_name].append(prediction)

            if not single_step_forecasts:
                X_forecast.loc[date, target_name] = prediction

        y_pred = pd.DataFrame(forecasted_values).set_index("Time")

        if return_X:
            return X_forecast.loc[forecasting_dates], y_pred
        return y_pred


    def _validate_consistency(self, train_set, test_set, train_name, test_name):
        """Validate frequency and continuity of train and test sets."""
        if train_set.index[-1] - train_set.index[-2] != test_set.index[1] - test_set.index[0]:
            raise ValueError(f"{train_name} and {test_name} must have the same frequency.")
        if test_set.index[0] <= train_set.index[-1]:
            raise ValueError(f"{test_name} must start after {train_name}.")
        if not all(train_set.columns == test_set.columns):
            raise ValueError(f"{train_name} and {test_name} must have the same columns.")

    def _prepare_forecasting_data(self, X_train=None, X_test=None, y_train=None, y_test=None,
                                freq=None, horizon=1, steps=1, max_lags=None, last_obs_date=None, target_name="Target"):
        """Prepare forecasting dates and the forecasting dataframe."""

        if X_train is None and X_test is None and y_train is None:
            raise ValueError("At least one of X_train, X_test, or y_train must be provided.")

        # Initialize forecasting dates and data frame
        forecasting_dates = None
        X_forecast = None

        # Determine target name if not specified
        if target_name is None:
            target_name = 'Target'
            if lags_dict:
                if X_test or X_train:
                    if X_test:
                        target_name = [t for t in lags_dict.keys() if t not in X_test.columns]
                elif X_train:
                    target_name = [t for t in lags_dict.keys() if t not in X_train.columns]

                if target_name:
                    target_name = target_name[0]

        # Determine frequency if not provided
        if freq is None or isinstance(freq, str):
            if X_train is not None:
                freq = X_train.index[-1] - X_train.index[-2]
            elif y_train is not None:
                freq = y_train.index[-1] - y_train.index[-2]
            else: 
                raise ValueError("Can't infer freq. Please provide a valid one.")

        # Determine last observed date if not provided
        if last_obs_date is None or X_train is not None or y_train is not None:
            if X_train is not None:
                last_obs_date = X_train.index[-1]
            elif y_train is not None:
                last_obs_date = y_train.index[-1]

        # Handle X_test if available
        if X_test is not None:
            forecasting_dates = X_test.index
            X_forecast = X_test.copy()
            steps = len(X_test)

        # Handle X_train if available and if X_test is not available or X_forecast is None
        if X_train is not None:
            if X_forecast is None:
                X_forecast = pd.concat([X_train.iloc[-(max_lags + (horizon-1)):].copy(), pd.DataFrame(index=forecasting_dates)], axis=0)
            else:
                X_forecast = pd.concat([X_train.iloc[-(max_lags + (horizon-1)):].copy(), X_forecast], axis=0)

        # Handle y_train if available and if X_forecast has not been fully constructed
        if y_train is not None:
            if forecasting_dates is None:
                forecasting_dates = pd.date_range(start=last_obs_date + (freq * horizon), periods=steps, freq=freq)
            if X_forecast is None:
                X_forecast = pd.DataFrame(
                    index=pd.concat([y_train.iloc[-(max_lags + (horizon-1)):].index, forecasting_dates.to_series()], axis=0).index)
            combined_y = pd.concat([y_train, y_test]) if y_test is not None else y_train.iloc[-(max_lags + (horizon-1)):]
            X_forecast[target_name] = combined_y.reindex(X_forecast.index)

        # If X_forecast is still None (which shouldn't be), create an empty DataFrame with the forecast dates
        if X_forecast is None:
            X_forecast = pd.DataFrame(index=forecasting_dates)

        # If y_train was not provided, fill target column with NaNs
        if target_name not in X_forecast.columns:
            X_forecast[target_name] = np.nan

        return X_forecast, forecasting_dates

    def _add_lagged_features(self, new_row, X_forecast, row_idx, lags_dict, horizon):
        """Add lagged features to the new_row dataframe."""
        lagged_cols = {}
        for col, lags in lags_dict.items():
            for lag in lags:
                lagged_cols[f'{col}_lag{lag}'] = X_forecast.iloc[row_idx - (lag + (horizon-1))][col]
        return pd.concat([new_row, pd.DataFrame(lagged_cols, index=new_row.index)], axis=1)
