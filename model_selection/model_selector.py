import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

class ModelSelector:
    """
    A class to evaluate and select the best machine learning model using cross-validation and hyperparameter tuning.

    This class facilitates the comparison of different models with various hyperparameter configurations using cross-validation.
    It supports applying feature and target transformations before and after model fitting and prediction. The class selects
    the best model based on a specified scoring metric and provides methods for obtaining results and making predictions.

    Parameters:
    - models: A list or dictionary of models to evaluate.
    - param_grids: A dictionary of parameter grids for each model.
    - cv_splitter: A cross-validation splitter object (e.g., TimeSeriesSplit).
    - scorer: The scoring metric to use ('r2', 'mse', or a callable function).
    - X_pre_valid_transformers: List of transformers to apply before validation.
    - X_transformers: List of transformers to apply to the features (during cross-validation).
    - y_transformers: List of transformers to apply to the target variable (during cross-validation).
    """

    def __init__(self, models, param_grids, cv_splitter, scorer='r2', 
                 X_pre_valid_transformers=None, X_transformers=None, y_transformers=None):
        """
        Initialize the ModelSelector class.

        Parameters:
        - models: A list or dictionary of models to evaluate.
        - param_grids: A dictionary of parameter grids for each model.
        - cv_splitter: A cross-validation splitter object.
        - scorer: The scoring metric ('r2', 'mse', or a callable function).
        - X_pre_valid_transformers: List of transformers to apply before validation.
        - X_transformers: List of transformers to apply to the features.
        - y_transformers: List of transformers to apply to the target variable.
        """
        if isinstance(models, list):
            models = {model.__class__.__name__: model for model in models}
        self.models = models if isinstance(models, dict) else {models.__class__.__name__: models}

        if isinstance(param_grids, list):
            param_grids = {model_name: param_grids for model_name in self.models}
        self.param_grids = param_grids if isinstance(param_grids, dict) else {list(self.models.keys())[0]: param_grids}

        self.cv_splitter = cv_splitter
        self.scorer = scorer
        self.results = {}
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.X_pre_valid_transformers = X_pre_valid_transformers
        self.X_transformers = X_transformers
        self.y_transformers = y_transformers

    def calculate_score(self, y_true, y_pred):
        """Calculate the score based on the selected metric."""
        if self.scorer == 'r2':
            return r2_score(y_true, y_pred)
        elif self.scorer == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative because we want to maximize
        elif callable(self.scorer):
            return self.scorer(y_true, y_pred)
        else:
            raise ValueError("Unsupported scorer. Use 'r2', 'mse', or a callable function.")

    def evaluate_model(self, model, X, y):
        """Evaluate a model using cross-validation."""
        scores = []

        for train_index, val_index in self.cv_splitter.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Apply X transformers
            if self.X_transformers:
                for transformer in self.X_transformers:
                    X_train = transformer.fit_transform(X_train)
                    X_val = transformer.transform(X_val)

            # Apply y transformers
            if self.y_transformers:
                for transformer in self.y_transformers:
                    y_train = transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Inverse transform y_pred if necessary
            if self.y_transformers:
                for transformer in reversed(self.y_transformers):
                    y_pred = transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            # Calculate score
            score = self.calculate_score(y_val, y_pred)
            scores.append(score)

        return np.mean(scores)

    def fit(self, X, y):
        """Fit the best model on the entire dataset."""

        if self.X_pre_valid_transformers:
            for transformer in self.X_pre_valid_transformers:
                X = transformer.fit_transform(X)

                if transformer.__class__.__name__ == 'FeatureLagger':
                    y = transformer.get_target(X, y)

        for model_name, model in self.models.items():
            param_grid = self.param_grids[model_name]
            for params in ParameterGrid(param_grid):
                model.set_params(**params)
                avg_score = self.evaluate_model(model, X, y)

                # Store results
                self.results[(model_name, tuple(params.items()))] = avg_score

                # Update best model if applicable
                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_model = model
                    self.best_params = params

        # Fit the best model on the entire dataset
        if self.X_transformers:
            for transformer in self.X_transformers:
                X = transformer.fit_transform(X)

        if self.y_transformers:
            for transformer in reversed(self.y_transformers):
                y = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()

        self.best_model.set_params(**self.best_params)
        self.best_model.fit(X, y)

        return self

    def get_best_model(self):
        """Return the best model."""
        return self.best_model

    def get_results(self):
        """Return the results of the cross-validation."""
        return self.results

    def predict(self, X):
        """Predict using the best model."""
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            idx = X.index

        if self.X_pre_valid_transformers:
            for transformer in self.X_pre_valid_transformers:
                X = transformer.transform(X)

                if transformer.__class__.__name__ == 'FeatureLagger':
                    idx = X.index

        if self.X_transformers:
            for transformer in self.X_transformers:
                X = transformer.transform(X)

        y_pred = self.best_model.predict(X)

        if self.y_transformers:
            for transformer in reversed(self.y_transformers):
                y_pred = transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            y_pred = pd.Series(y_pred, index=idx)

        return y_pred
