import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureLagger(BaseEstimator, TransformerMixin):
    """
    A transformer that lags specified features in a DataFrame and optionally applies transformations to the data
    before lagging. This is useful for time series data where past values of a feature are used as inputs 
    for a model predicting future values.

    Parameters
    ----------
    lags_dict : dict, optional (default=None)
        A dictionary where keys are column names and values are lists of lags to apply.
        Example: {'column1': [1, 2], 'column2': [3]} will create columns with 1st and 2nd lag for 'column1'
        and the 3rd lag for 'column2'.
    
    target_name : str, optional (default=None)
        The name of the target column. This is the column that may be excluded from transformations
        and that will be used for certain calculations (like calculating the target series after lagging).
    
    transformations : list of transformers, optional (default=None)
        A list of transformation objects (such as StandardScaler or other transformers from scikit-learn) 
        to be applied to the features before lagging. If a single transformer is provided, it will be 
        converted to a list automatically.

    Attributes
    ----------
    max_lags : int
        The maximum number of lags specified in the lags_dict.
    
    lags_from_train_data : pandas DataFrame
        Stores the last few rows from the training data to ensure continuity during transformation.
    
    col_order : list of str
        Stores the original order of columns before transformations and lagging.
    
    feature_names : pandas Index
        Stores the names of the features after lagging and transformation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> from your_module import FeatureLagger
    >>> data = pd.DataFrame({
    ...     'feature1': range(10),
    ...     'feature2': range(10, 20),
    ...     'target': range(20, 30)
    ... })
    >>> lags_dict = {'feature1': [1, 2], 'feature2': [1]}
    >>> fl = FeatureLagger(lags_dict=lags_dict, target_name='target', transformations=[StandardScaler()])
    >>> fl.fit(data)
    >>> transformed_data = fl.transform(data)
    >>> print(transformed_data)
    """

    def __init__(self, lags_dict=None, target_name=None, transformations=None):
        """
        Initialize the FeatureLagger transformer with a dictionary specifying the lags for each column and optional transformations.

        Parameters
        ----------
        lags_dict : dict, optional (default=None)
            A dictionary where keys are column names and values are lists of lags to apply.
            Example: {'column1': [1, 2], 'column2': [3]} will create columns with 1st and 2nd lag for 'column1'
            and the 3rd lag for 'column2'.
        
        target_name : str, optional (default=None)
            The name of the target column. This is the column that may be excluded from transformations
            and that will be used for certain calculations (like calculating the target series after lagging).
        
        transformations : list of transformers, optional (default=None)
            A list of transformation objects (such as StandardScaler or other transformers from scikit-learn) 
            to be applied to the features before lagging. If a single transformer is provided, it will be 
            converted to a list automatically.
        """
        self.lags_dict = lags_dict if lags_dict else {}
        self.target_name = target_name
        self.transformations = transformations
        if self.transformations and not isinstance(self.transformations, list):
            self.transformations = [self.transformations]

        self.max_lags = 0
        for k, v in self.lags_dict.items():
            if max(v) > self.max_lags:
                self.max_lags = max(v)
        self.lags_from_train_data = None
        self.col_order = []
        self.feature_names = None

    def get_target(self, X, y=None, dropna=True):
        """
        Retrieve the target column after dropping rows affected by lagging.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame.
        
        y : pandas Series, optional (default=None)
            The target values. If provided, it will be used instead of extracting the target from X.
        
        dropna : bool, optional (default=True)
            Whether to drop NaN values resulting from lagging.

        Returns
        -------
        pandas Series
            The target column after accounting for lags.
        """
        if y is not None:
            target = y.dropna() if dropna else y
        else:
            target = X[self.target_name].dropna().copy() if dropna else X[self.target_name].copy()

        return target.iloc[self.max_lags:]

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        This method calculates the transformations on the data and stores the last few rows
        of the training data for continuity during transformation.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame.
        
        y : pandas Series, optional
            The target values. Not used in this method.

        Returns
        -------
        self : object
            Returns self.
        """
        self.col_order = [c for c in X.columns if c != self.target_name]

        if self.transformations:
            for transformation in self.transformations:
                transformation.fit(X[self.col_order])

        self.lags_from_train_data = X.iloc[-self.max_lags:, :]
        return self

    def transform(self, X):
        """
        Apply transformations and create lagged features.

        This method applies the specified transformations to the features and then generates
        lagged versions of the specified columns.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame.

        Returns
        -------
        pandas DataFrame
            The transformed DataFrame with lagged features.
        """
        X_transformed = X[self.col_order].copy()
        if self.transformations:
            for transformation in self.transformations:
                X_transformed = transformation.transform(X_transformed)

            X_transformed = pd.DataFrame(X_transformed, columns=self.col_order, index=X.index)
            X_transformed[self.target_name] = X[self.target_name].copy()

        new_columns = []

        X_transformed = pd.concat([self.lags_from_train_data, X_transformed], axis=0)

        for column, lags in self.lags_dict.items():
            if column in X.columns:
                for lag in lags:
                    new_columns.append(pd.DataFrame({f'{column}_lag{lag}': X_transformed[column].shift(lag)}))

        if self.target_name in X.columns and self.target_name in self.lags_dict:
            X_transformed = X_transformed.drop(self.target_name, axis=1)
        X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        X_transformed = X_transformed.dropna()
        X_transformed = X_transformed.iloc[self.max_lags:]
        self.feature_names = X_transformed.columns
        return X_transformed

    def get_feature_names_out(self):
        """
        Get the names of the features after transformation and lagging.

        Returns
        -------
        pandas Index
            The names of the transformed and lagged features.
        """
        return self.feature_names
