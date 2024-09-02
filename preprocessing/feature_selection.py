import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelection:

    def __init__(self, models, params_grids, cv_splitter, scorer='r2', 
                 X_pre_valid_transformers=None, X_transformers=None, y_transformers=None):
        """
        A class for performing feature selection using cross-validation.

        Parameters:
        - models: A list or dictionary of models to evaluate.
        - params_grids: A dictionary of parameter grids for each model.
        - cv_splitter: A cross-validation splitter object.
        - scorer: The scoring metric ('r2', 'mse', or a callable function).
        - X_pre_valid_transformers: List of transformers to apply before validation.
        - X_transformers: List of transformers to apply to the features (during cross-validation).
        - y_transformers: List of transformers to apply to the target variable (during cross-validation).
        """

        if isinstance(models, list):
            models = {model.__class__.__name__: model for model in models}
        self.models = models if isinstance(models, dict) else [models]
        self.params_grids = params_grids
        self.cv_splitter = cv_splitter
        self.scorer = scorer
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.selected_features = None
        self.results = {}
        self.X_pre_valid_transformers = X_pre_valid_transformers
        self.X_transformers = X_transformers
        self.y_transformers = y_transformers

    def select_features(self, X, y):
        """
        Fit the model using cross-validation and perform feature selection.

        Parameters:
        - X: Input features.
        - y: Target variable.
        """

        if self.X_pre_valid_transformers:
            if isinstance(X, pd.DataFrame):
                cols = X.columns

            for transformer in self.X_pre_valid_transformers:
                X = transformer.fit_transform(X)

                if transformer.__class__.__name__ == 'FeatureLagger':
                    y = transformer.get_target(X, y)

        for model_name, model in self.models.items():
            param_grid = self.params_grids[model_name]

            for params in ParameterGrid(param_grid):
                model.set_params(**params)
                scores = []

                for train_idx, test_idx in self.cv_splitter.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    if self.X_transformers:
                        for transformer in self.X_transformers:
                            X_train = transformer.fit_transform(X_train)
                            X_test = transformer.transform(X_test)

                    if self.y_transformers:
                        for transformer in self.y_transformers:
                            y_train = transformer.fit_transform(y_train)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if self.y_transformers:
                        for transformer in reversed(self.y_transformers):
                            y_pred = transformer.inverse_transform(y_pred)

                    score = self.calculate_score(y_test, y_pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                self.results[(model_name, tuple(params.items()))] = avg_score

                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_model = model
                    self.best_params = params

        # Fit best model on entire dataset
        if self.X_transformers:
            for transformer in self.X_transformers:
                X = transformer.fit_transform(X)

        if self.y_transformers:
            for transformer in reversed(self.y_transformers):
                y = transformer.fit_transform(y)

        self.best_model.set_params(**self.best_params)
        self.best_model.fit(X, y)

        selector = SelectFromModel(self.best_model, prefit=True)
        self.selected_features = cols[selector.get_support()].tolist()

        return self.selected_features

    def get_X(self, X):
        """
        Get the input features with selected features only.

        Parameters:
        - X: Input features.

        Returns:
        - X: Transformed features with selected columns.
        """
        return X[self.selected_features]

    def get_best_score(self):
        """Return the best score obtained."""
        return self.best_score

    def get_results(self):
        """Return the results of the cross-validation."""
        return self.results

    def calculate_score(self, y_true, y_pred):
        """
        Calculate the score based on the selected metric.

        Parameters:
        - y_true: Actual values.
        - y_pred: Predicted values.

        Returns:
        - score: Calculated score.
        """
        if self.scorer == 'r2':
            return r2_score(y_true, y_pred)
        elif self.scorer == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative because we want to maximize
        elif callable(self.scorer):
            return self.scorer(y_true, y_pred)
        else:
            raise ValueError("Unsupported scorer. Use 'r2', 'mse', or a callable function.")
