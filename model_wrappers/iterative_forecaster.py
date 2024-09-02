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
            Number of steps ahead to start forecasting.
        
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

        # Validate input consistency
        if X_train is not None and X_test is not None:
            if X_train.index[-1] - X_train.index[-2] != X_test.index[1] - X_test.index[0]:
                raise ValueError("X_train and X_test must have the same frequency.")
            if X_test.index[0] <= X_train.index[-1]:
                raise ValueError("X_test must start after X_train.")
            if not all(X_train.columns == X_test.columns):
                raise ValueError("X_train and X_test must have the same columns.")

        if y_train is not None and y_test is not None:
            if y_train.index[-1] - y_train.index[-2] != y_test.index[1] - y_test.index[0]:
                raise ValueError("y_train and y_test must have the same frequency.")
            if y_test.index[0] <= y_train.index[-1]:
                raise ValueError("y_test must start after y_train.")

        if max_lags is None:
            max_lags = 0

        if lags_dict is not None:
            for lags in lags_dict.values():
                if max(lags) > max_lags:
                    max_lags = max(lags)

        if target_name is None:
            target_name = 'Target'

        # Determine forecasting dates and prepare the forecasting dataframe
        if X_test is not None:
            steps = len(X_test)
            if freq is None:
                freq = X_test.index[-1] - X_test.index[-2]
            forecasting_dates = X_test.index
            X_forecast = X_test.copy()
        elif X_train is not None:
            if freq is None:
                freq = X_train.index[-1] - X_train.index[-2]
            last_obs_date = X_train.index[-1]
            forecasting_dates = pd.date_range(start=last_obs_date + (freq * horizon), periods=steps, freq=freq)
            X_forecast = X_train.iloc[-max_lags:].copy()
            X_forecast = pd.concat([X_forecast, pd.DataFrame(index=forecasting_dates)], axis=0)
        else:
            raise ValueError("X_train or X_test must be provided.")

        if y_train is not None:
            if freq is None:
                freq = y_train.index[-1] - y_train.index[-2]
            if last_obs_date is None:
                last_obs_date = y_train.index[-1]
            y_combined = pd.concat([y_train, y_test]) if y_test is not None else y_train
            X_forecast[target_name] = y_combined.reindex(X_forecast.index)

        else:
            X_forecast[target_name] = np.nan

        forecasted_values = {"Time": [], target_name: []}

        for date in forecasting_dates:
            new_row = X_forecast.loc[[date]].drop(columns=target_name)
            row_idx = X_forecast.index.get_loc(date)

            if lags_dict is not None:
                lagged_cols = {}
                for col, lags in lags_dict.items():
                    for lag in lags:
                        lagged_cols[f'{col}_lag{lag}'] = X_forecast.iloc[row_idx - lag][col]
                lags_df = pd.DataFrame(lagged_cols, index=[date])
                new_row = pd.concat([new_row, lags_df], axis=1)

            prediction = self.model.predict(new_row)[0]
            forecasted_values["Time"].append(date)
            forecasted_values[target_name].append(prediction)

            if not single_step_forecasts:
                X_forecast.loc[date, target_name] = prediction

        y_pred = pd.DataFrame(forecasted_values).set_index("Time")

        if return_X:
            return X_forecast, y_pred
        return y_pred
