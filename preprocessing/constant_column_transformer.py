from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AddConstantColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that adds a new column with a constant value to a pandas DataFrame.

    This transformer can be used in a scikit-learn pipeline. It adds a specified column
    to the input DataFrame with a constant value provided during initialization.

    Parameters
    ----------
    column_name : str, default='constant'
        The name of the column to be added to the DataFrame.
        
    constant_value : any, default=1
        The constant value to fill in the new column.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from your_module import AddConstantColumnTransformer
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> transformer = AddConstantColumnTransformer(column_name='B', constant_value=5)
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       A  B
    0  1  5
    1  2  5
    2  3  5
    """
    
    def __init__(self, column_name='constant', constant_value=1):
        """
        Initialize the AddConstantColumnTransformer.

        Parameters
        ----------
        column_name : str, default='constant'
            The name of the column to be added to the DataFrame.
            
        constant_value : any, default=1
            The constant value to fill in the new column.
        """
        self.column_name = column_name
        self.constant_value = constant_value

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        This transformer does not require fitting, so this method simply returns self.

        Parameters
        ----------
        X : pandas DataFrame
            The input data.
            
        y : None, optional
            Ignored, exists for compatibility with scikit-learn's fit method signature.

        Returns
        -------
        self : object
            Returns self.
        """
        # This transformer does not need fitting, so we just return self
        return self

    def transform(self, X):
        """
        Transform the input data by adding a new column with a constant value.

        Parameters
        ----------
        X : pandas DataFrame
            The input data to transform. Must be a pandas DataFrame.

        Returns
        -------
        X : pandas DataFrame
            The transformed DataFrame with the new column added.

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame.
        """
        # Check if X is a pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Add a new column with the constant value
            X[self.column_name] = self.constant_value
            return X
        else:
            raise ValueError("Input must be a pandas DataFrame.")
