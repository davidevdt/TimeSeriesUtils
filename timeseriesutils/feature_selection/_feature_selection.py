import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator

class FeatureSelector(ModelSelector):
    """
    A class for feature selection that extends ModelSelector. It uses a feature selector to identify
    and select important features from the dataset based on the best model obtained during model selection.

    Parameters
    ----------
    feature_selector : type or None, optional
        A feature selector class or instance. If a class, it should be callable with the `estimator` and
        `prefit` parameters. If None, defaults to `SelectFromModel`.
        
    models_dict : dict
        A dictionary where keys are model names and values are dictionaries containing:
            - 'model': The model instance (e.g., from scikit-learn).
            - 'param_grid': A dictionary of parameter grids for the model.
            - 'X_transformers': A list of transformers to apply to X.
            - 'y_transformers': A list of transformers to apply to y.
        
    cv_splitter : object
        An instance of a cross-validation splitter (e.g., `KFold`, `StratifiedKFold`) used for model validation.
        
    scorer : str or callable, default='r2'
        A scoring method or function to evaluate model performance. Can be 'r2', 'mse', 'rmse', or a custom callable.
        
    Attributes
    ----------
    selected_features : list of str or None
        The names of the selected features identified by the feature selector. Updated after calling `select_features`.

    Methods
    -------
    select_features(X, y=None, n_jobs=1)
        Selects features based on the best model and the specified feature selector. Requires `X` to be a pandas DataFrame.

    get_feature_selector
        Returns the feature selector instance or class used for feature selection.

    set_feature_selector
        Sets a new feature selector instance or class for feature selection.
    """

    def __init__(
        self,
        feature_selector=SelectFromModel, 
        models_dict=None, 
        cv_splitter=None, 
        scorer='r2'
    ):
        """
        Initialize the FeatureSelector.

        Parameters
        ----------
        feature_selector : type or None
            A feature selector class or instance. If a class, it should be callable with the `estimator` and `prefit` parameters.
        
        models_dict : dict
            A dictionary where keys are model names and values are dictionaries containing:
                - 'model': The model instance (e.g., from scikit-learn).
                - 'param_grid': A dictionary of parameter grids for the model.
                - 'X_transformers': A list of transformers to apply to X.
                - 'y_transformers': A list of transformers to apply to y.
        
        cv_splitter : object
            An instance of a cross-validation splitter used for model validation.
        
        scorer : str or callable, default='r2'
            A scoring method or function to evaluate model performance.
        
        """
        super().__init__(models_dict, cv_splitter, scorer)
        self._feature_selector = feature_selector
        self.selected_features = None 

    def select_features(self, X, y=None, n_jobs=1, verbose=False, refit=False):
        """
        Selects features based on the best model and the specified feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing features. Must be a pandas DataFrame.
        
        y : array-like, optional
            The target values. Not used in this method but required for compatibility with `ModelSelector.fit`.
        
        n_jobs : int, default=1
            The number of jobs to run in parallel for model fitting.

        verbose : bool, default = False 
            Whether to print progress during model fitting.

        refit : bool, default = False
            Whether to refit the best model on the data.

        Returns
        -------
        selected_features : list of str
            The names of the selected features as identified by the feature selector. If the feature selector is not set
            or if no best model is found, an exception will be raised.
        
        Raises
        ------
        ValueError
            If `X` is not a pandas DataFrame or if `feature_selector` is not set.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if self._feature_selector is None:
            raise ValueError("Feature selector must be set before calling select_features")

        cols = X.columns
        if self.best_model is None or refit:
            if y is None:
                raise ValueError("y must be provided if refit is True or models have not been fitted yet.")
            self.fit(X, y, n_jobs, verbose)

        if self.X_transformers and self.best_model_name in self.X_transformers and self.X_transformers[self.best_model_name]:
            for transformer in self.X_transformers[self.best_model_name]: 
                idx = X.index 
                X = transformer.transform(X)
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=transformer.get_feature_names_out(), index=idx)
                cols = X.columns

        if isinstance(self._feature_selector, type):
            self._feature_selector = self._feature_selector(estimator=self.best_model, prefit=True)
        else:
            self._feature_selector.set_params(estimator=self.best_model, prefit=True)

        if not hasattr(self.feature_selector, 'get_support'):
            raise ValueError("Feature selector must have a `get_support` method")

        self.selected_features = cols[self.feature_selector.get_support()].tolist()
        return self.selected_features

    @property
    def feature_selector(self):
        """
        Get the current feature selector.

        Returns
        -------
        feature_selector : type or instance
            The feature selector instance or class.
        """
        return self._feature_selector

    @feature_selector.setter
    def set_feature_selector(self, new_feature_selector):
        """
        Set a new feature selector.

        Parameters
        ----------
        new_feature_selector : type or instance
            The new feature selector class or instance to use for feature selection.
        """
        self._feature_selector = new_feature_selector
