import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

class TsUtils:
    """
    A utility class containing static methods for common time series data manipulation tasks.
    """

    @staticmethod
    def calculate_periodic_change(data, column, offset=pd.DateOffset(years=1),
                                  prev_suffix='PrevPeriodValue', change_suffix='Period_Change', perc_change_suffix='Period_%Change'):
        """
        Calculate the change for a specific column over a defined periodic offset in a pandas DataFrame.

        Parameters
        ----------
        data : pandas DataFrame
            The DataFrame containing the time series data.
        
        column : str
            The name of the column for which the periodic change is to be calculated.
        
        offset : pandas DateOffset, default=pd.DateOffset(years=1)
            The offset to use for calculating the previous period's value.
        
        prev_suffix : str, default='PrevPeriodValue'
            Suffix for the column containing the previous period's value.
        
        change_suffix : str, default='Period_Change'
            Suffix for the column containing the periodic change (difference).
        
        perc_change_suffix : str, default='Period_%Change'
            Suffix for the column containing the periodic percentage change.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the original column, the previous period's value, the periodic change,
            and the periodic percentage change.
        """
        prev_data_df = pd.DataFrame(
            {f'{column}_{prev_suffix}': data[column].to_numpy()},
            index=data.index + offset
        )
        prev_data_df = pd.concat([data[column], prev_data_df], axis=1, join='inner')
        prev_data_df[f'{column}_{change_suffix}'] = prev_data_df[column] - prev_data_df[f'{column}_{prev_suffix}']
        prev_data_df[f'{column}_{perc_change_suffix}'] = prev_data_df[f'{column}_{change_suffix}'] / prev_data_df[f'{column}_{prev_suffix}']
        return prev_data_df.iloc[:, -2:]

    def is_stationary(series: pd.Series, alpha: float = 0.05, verbose: bool = False) -> bool:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller (ADF) test.
    
        A time series is considered stationary if it has constant mean, variance, and autocovariance over time.
        The ADF test checks for the presence of a unit root, which is a strong indication of non-stationarity.
        
        Parameters:
        -----------
        series : pd.Series
            The time series data to be tested for stationarity.
        alpha : float, optional
            The significance level to determine stationarity. Default is 0.05 (5% significance).
        verbose : bool, optional
            If True, prints the test statistic, p-value, and critical values from the ADF test. Default is False.
        
        Returns:
        --------
        bool
            Returns True if the time series is stationary, otherwise False.
        
        Example:
        --------
        >>> data = pd.Series([1, 2, 1.5, 1.8, 1.2, 1.4])
        >>> is_stationary(data, verbose=True)
        Test Statistic: -2.34567
        p-value: 0.1234
        Critical Values:
        1%: -3.5
        5%: -2.9
        10%: -2.6
        False
        """
        
        # Perform the Augmented Dickey-Fuller test
        adf_test = adfuller(series, autolag='AIC')
    
        test_statistic = adf_test[0]  # Test statistic value
        p_value = adf_test[1]         # p-value
        critical_values = adf_test[4] # Dictionary of critical values
    
        # Verbose option to print detailed test information
        if verbose:
            print(f"Test Statistic: {test_statistic}")
            print(f"p-value: {p_value}")
            print("Critical Values:")
            for key, value in critical_values.items():
                print(f"   {key}: {value}")
    
        # Determine stationarity: if p-value is less than alpha, we reject the null hypothesis (stationary)
        return p_value < alpha

    @staticmethod
    def standardize_frequencies(df, target_frequency='W', method='ffill'):
        """
        Standardize the frequencies of a time series DataFrame to a specified frequency.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        target_frequency : str, default='W'
            The target frequency to which the DataFrame should be standardized. For example, 'W' for weekly, 'M' for monthly.
        
        method : str, default='ffill'
            The method to use for filling missing values after resampling. Options include 'ffill' (forward fill), 'bfill' (backward fill),
            or any method recognized by pandas' resample method.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the index standardized to the target frequency and missing values filled according to the specified method.
        """
        # Resample the DataFrame to the target frequency
        df_resampled = df.resample(target_frequency).apply(method)
        
        return df_resampled

    @staticmethod
    def resample_data(df, rule, method='mean'):
        """
        Resample the time series DataFrame to a different frequency.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        rule : str
            The resampling rule. For example, 'D' for daily, 'M' for monthly, 'A' for annual.
        
        method : str, default='mean'
            The method to use for resampling. Options include 'mean', 'sum', 'max', 'min', etc.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the resampled data.
        """
        return df.resample(rule).apply(method)

    @staticmethod
    def compute_rolling_statistics(df, column, window, method='mean'):
        """
        Compute rolling statistics for a specific column in a pandas DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        column : str
            The name of the column for which rolling statistics are to be computed.
        
        window : int
            The window size for the rolling computation.
        
        method : str, default='mean'
            The statistic to compute. Options include 'mean', 'std', 'sum', etc.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the rolling statistics.
        """
        if method == 'mean':
            return df[column].rolling(window=window).mean()
        elif method == 'std':
            return df[column].rolling(window=window).std()
        elif method == 'sum':
            return df[column].rolling(window=window).sum()
        else:
            raise ValueError("Unsupported method. Choose from 'mean', 'std', 'sum'.")

    @staticmethod
    def fill_missing_values(df, method='ffill'):
        """
        Fill missing values in the time series DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        method : str, default='ffill'
            The method to use for filling missing values. Options include 'ffill' (forward fill), 'bfill' (backward fill), and 'linear' (linear interpolation).

        Returns
        -------
        pandas DataFrame
            A DataFrame with missing values filled.
        """
        return df.fillna(method=method) if method in ['ffill', 'bfill'] else df.interpolate(method=method)

    @staticmethod
    def extract_time_features(df, features=['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']):
        """
        Extract time-based features from the DataFrame's index.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        features : list of str, default=['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
            List of time-based features to extract. Options include 'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'.

        Returns
        -------
        pandas DataFrame
            A DataFrame with extracted time-based features.
        """
        time_features = {}
        if 'year' in features:
            time_features['year'] = df.index.year
        if 'month' in features:
            time_features['month'] = df.index.month
        if 'day' in features:
            time_features['day'] = df.index.day
        if 'weekday' in features:
            time_features['weekday'] = df.index.weekday
        if 'hour' in features:
            time_features['hour'] = df.index.hour
        if 'minute' in features:
            time_features['minute'] = df.index.minute
        if 'second' in features:
            time_features['second'] = df.index.second
        return pd.DataFrame(time_features, index=df.index)

    @staticmethod
    def create_lag_features(df, column, lags):
        """
        Create lagged features for a specific column in the DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        column : str
            The name of the column for which lagged features are to be created.
        
        lags : int or list of int
            The lag periods for which features are to be created. Can be a single integer or a list of integers.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the original column and the created lagged features.
        """
        if isinstance(lags, int):
            lags = [lags]
        lagged_features = {f'{column}_lag_{lag}': df[column].shift(lag) for lag in lags}
        return pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)

    @staticmethod
    def convert_timezone(df, target_timezone='UTC'):
        """
        Convert the time series DataFrame's index to a specified timezone.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        target_timezone : str, default='UTC'
            The target timezone to convert to. This can be any timezone recognized by pandas.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the index converted to the target timezone.
        """
        return df.tz_convert(target_timezone)

    @staticmethod
    def localize_timezone(df, timezone):
        """
        Localize the time series DataFrame's index to a specified timezone.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame containing the time series data.
        
        timezone : str
            The timezone to localize to. This can be any timezone recognized by pandas.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the index localized to the specified timezone.
        """
        return df.tz_localize(timezone)