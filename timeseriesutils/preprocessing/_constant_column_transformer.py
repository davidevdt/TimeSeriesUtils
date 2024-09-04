from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AddConstantColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that adds a constant value column to the input DataFrame or numpy array.

    This transformer is useful for adding a bias term or constant feature to a dataset.

    Parameters
    ----------
    column_name : str, default='constant'
        The name of the column to be added to the DataFrame.

    constant_value : numeric, default=1
        The constant value to be added in the new column.

    Attributes
    ----------
    feature_names_in_ : list of str or None
        The feature names of the input DataFrame, if the input is a DataFrame. This attribute is set during
        the `transform` method.

    Methods
    -------
    fit(X, y=None)
        Does nothing and returns self. This transformer does not require fitting.

    transform(X)
        Adds a column with a constant value to the input data. If the input is a DataFrame and the column
        already exists, it returns the input data unchanged. Otherwise, it adds the new column with the
        specified constant value.

    get_feature_names_out(input_features=None)
        Returns the feature names of the transformed data. If the input is a DataFrame, it returns the
        original feature names plus the new column name. If the input is not a DataFrame, it returns
        an empty list plus the column name.
    """
    
    def __init__(self, column_name='constant', constant_value=1):
        """
        Initialize the AddConstantColumnTransformer.

        Parameters
        ----------
        column_name : str, default='constant'
            The name of the column to be added to the DataFrame.

        constant_value : numeric, default=1
            The constant value to be added in the new column.
        """
        self.column_name = column_name
        self.constant_value = constant_value
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        This transformer does not require fitting. Returns self.

        Parameters
        ----------
        X : {array-like, DataFrame}
            Input data, not used in this method.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : AddConstantColumnTransformer
            The transformer object.
        """
        return self

    def transform(self, X):
        """
        Add a constant value column to the input data.

        Parameters
        ----------
        X : {array-like, DataFrame}
            Input data to which the constant column will be added.

        Returns
        -------
        transformed_X : {array-like, DataFrame}
            The input data with the added constant column. If the input is a DataFrame and the column
            already exists, it returns the input data unchanged. Otherwise, it adds the new column
            with the specified constant value.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            if self.column_name in X.columns:
                return X
            X[self.column_name] = self.constant_value
            return X
        else: 
            return np.hstack([X, np.ones((X.shape[0], 1))])

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names of the transformed data.

        Parameters
        ----------
        input_features : list of str, optional
            List of feature names for the input data. Ignored if input is None.

        Returns
        -------
        feature_names : list of str
            The feature names of the transformed data. Includes the original feature names plus the new column name.
        """
        if input_features is None:
            input_features = []
        return input_features + self.feature_names_in_
