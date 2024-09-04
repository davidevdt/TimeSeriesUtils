import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, clone
from functools import partial
from joblib import Parallel, delayed

class ModelSelector:
    """
    A class for selecting the best model from a set of candidates based on cross-validated performance.
    The class allows for parallelized model selection, with support for preprocessing pipelines for both features (X) and targets (y)
    within each cross-validation fold.

    Attributes
    ----------
    models : dict
        A dictionary where keys are model names and values are dictionaries containing 'model' (model instance), 
        'param_grid' (parameter grids), 'X_transformers' (transformers for X), and 'y_transformers' (transformers for y).
    
    cv_splitter : object
        A cross-validation splitter object (e.g., from scikit-learn) that provides train/test indices.
    
    scorer : str or callable, optional
        A scoring function or string name of the scoring method ('r2', 'mse', 'rmse'). Default is 'r2'.
    
    best_model : object
        The model instance that performed the best during the cross-validation process.
    
    best_model_name : str
        The name of the best-performing model.
    
    best_params : dict
        The best parameters for the best-performing model.
    
    best_score : float
        The best score achieved by any model during cross-validation.
    
    results : dict
        A dictionary where the keys are tuples of (model_name, params) and the values are the average scores from cross-validation.

    Methods
    -------
    fit(X, y, n_jobs=1)
        Fits the models to the data, performing cross-validation to find the best model and parameters.
    
    predict(X)
        Predicts the target values using the best-performing model.
    
    get_best_score()
        Returns the best score achieved during model selection.
    
    get_results()
        Returns the cross-validation results for all models and parameter combinations.
    
    get_best_model()
        Returns the best-performing model instance.
    
    calculate_score(y_true, y_pred, scorer='r2')
        Static method to calculate the score of the model predictions.
    """
    
    def __init__(self, models_dict, cv_splitter, scorer='r2'):
        """
        Initializes the ModelSelector with the given models, parameter grids, and other settings.

        Parameters
        ----------
        models_dict : dict
            A dictionary where keys are model names and values are dictionaries containing:
                - 'model': The model instance (e.g., from scikit-learn).
                - 'param_grid': A dictionary of parameter grids for the model.
                - 'X_transformers': A list of transformers to apply to X.
                - 'y_transformers': A list of transformers to apply to y.
        
        cv_splitter : object
            A cross-validation splitter that provides train/test indices.
        
        scorer : str or callable, optional
            The scoring function or string name of the scoring method ('r2', 'mse', 'rmse'). Default is 'r2'.
        """
        
        self.models = {}
        self.params_grids = {}
        self.X_transformers = {}
        self.y_transformers = {}
        
        for model_name, settings in models_dict.items():
            self.models[model_name] = settings['model']
            self.params_grids[model_name] = settings['param_grid']
            self.X_transformers[model_name] = settings.get('X_transformers', [])
            self.y_transformers[model_name] = settings.get('y_transformers', [])
        
        self.cv_splitter = cv_splitter
        self.scorer = scorer
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = float('-inf')
        self.results = {}

    @staticmethod
    def __fit(X, y, model, params, score_function, X_transformers, y_transformers, train_idx, test_idx):
        """
        Fits the model on the training data and evaluates it on the test data.

        Parameters
        ----------
        X : DataFrame
            The feature data.
        
        y : Series or DataFrame
            The target data.
        
        model : object
            The model instance to fit.
        
        params : dict
            The parameters to set for the model.
        
        score_function : callable
            The function to compute the score.
        
        X_transformers : list or None
            The list of transformers to apply to X before fitting.
        
        y_transformers : list or None
            The list of transformers to apply to y before fitting.
        
        train_idx : array-like
            The indices for the training data.
        
        test_idx : array-like
            The indices for the test data.

        Returns
        -------
        score : float
            The score of the model on the test data.
        """
        
        model.set_params(**params)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if X_transformers:
            for transformer in X_transformers:
                if transformer:
                    idx = X_train.index
                    idx_test = X_test.index
                    X_train = transformer.fit_transform(X_train)
                    X_test = transformer.transform(X_test)
                    if not isinstance(X_train, pd.DataFrame):
                        X_train = pd.DataFrame(X_train, columns=transformer.get_feature_names_out(), index=idx)
                        X_test = pd.DataFrame(X_test, columns=transformer.get_feature_names_out(), index=idx_test)

        if y_transformers:
            for transformer in y_transformers:
                if transformer:
                    y_train = transformer.fit_transform(y_train)

        if not isinstance(y_train, np.ndarray):
            y_train = y_train.to_numpy()

        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test).reshape(-1, 1)

        if y_transformers:
            for transformer in reversed(y_transformers):
                if transformer:
                    y_pred = transformer.inverse_transform(y_pred)
        
        score = score_function(y_test, y_pred)
        return score

    def fit(self, X, y, n_jobs=1, verbose=False):
        """
        Fits all models on the data, performing cross-validation to find the best model and parameters.

        Parameters
        ----------
        X : DataFrame
            The feature data to fit the models on.
        
        y : Series or DataFrame
            The target data to fit the models on.
        
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is 1 (no parallelization).

        verbose : bool, optional
            Whether to print progress during model fitting. Default is False.

        Returns
        -------
        self : ModelSelector
            The fitted ModelSelector instance.
        """
        
        score_function = partial(self.calculate_score, scorer=self.scorer)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        for model_name, model in self.models.items():
            param_grid = self.params_grids[model_name]
            X_transformers = self.X_transformers[model_name]
            y_transformers = self.y_transformers[model_name]

            for params in ParameterGrid(param_grid):
                if verbose:
                    print(f"Fitting {model_name} with params: {params}")

                scores = Parallel(n_jobs=n_jobs)(
                    delayed(self.__fit)
                    (X, y, model, params, score_function, X_transformers, y_transformers, train_index, test_index)
                    for train_index, test_index in self.cv_splitter.split(X)
                )

                avg_score = np.mean(scores)
                self.results[(model_name, tuple(params.items()))] = avg_score
                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_model = clone(model).set_params(**params)
                    self.best_params = params
                    self.best_model_name = model_name

        # Fit best model on entire dataset
        if self.X_transformers[self.best_model_name]:
            for transformer in self.X_transformers[self.best_model_name]:
                if transformer:
                    idx = X.index
                    X = transformer.fit_transform(X)
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X, columns=transformer.get_feature_names_out(), index=idx)

        if self.y_transformers[self.best_model_name]:
            for transformer in self.y_transformers[self.best_model_name]:
                if transformer:
                    y = transformer.fit_transform(y)

        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        self.best_model.fit(X, y.ravel()) 
        return self

    def predict(self, X):
        """
        Predicts the target values using the best-performing model.

        Parameters
        ----------
        X : Series or DataFrame
            The feature data to make predictions on.

        Returns
        -------
        y_pred : Series
            The predicted target values.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet or if X is not a valid input type.
        """

        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit first.")

        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("X must be a pandas Series or DataFrame")

        idx = X.index

        if self.X_transformers[self.best_model_name]:
            for transformer in self.X_transformers[self.best_model_name]:
                if transformer:
                    X = transformer.transform(X)
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X, columns=transformer.get_feature_names_out(), index=idx)

        y_pred = self.best_model.predict(X).reshape(-1, 1)

        if self.y_transformers[self.best_model_name]:
            for transformer in reversed(self.y_transformers[self.best_model_name]):
                if transformer:
                    y_pred = transformer.inverse_transform(y_pred)

        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.to_numpy()

        y_pred = pd.Series(y_pred.ravel(), index=idx)
        return y_pred

    @staticmethod
    def calculate_score(y_true, y_pred, scorer='r2'):
        """
        Calculates the score of the model predictions.

        Parameters
        ----------
        y_true : array-like
            The true target values.
        
        y_pred : array-like
            The predicted target values.
        
        scorer : str or callable, optional
            The scoring function or method name ('r2', 'mse', 'rmse'). Default is 'r2'.

        Returns
        -------
        score : float
            The calculated score.
        
        Raises
        ------
        ValueError
            If the scorer is not supported.
        """
        
        if scorer == 'r2':
            return r2_score(y_true, y_pred)
        elif scorer == 'mse':
            return -mean_squared_error(y_true, y_pred)
        elif scorer == 'rmse':
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif callable(scorer):
            return scorer(y_true, y_pred)
        else:
            raise ValueError("Unsupported scorer. Use 'r2', 'mse', or a callable function.")
