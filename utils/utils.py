import pandas as pd

class Utils:
    """
    A utility class containing static methods for common data manipulation tasks.
    """

    @staticmethod
    def yoy_change(data, col_name, offset=pd.tseries.offsets.DateOffset(years=1),
                   prev_suffix='PrevYearValue', change_suffix='YoY_Change', perc_change_suffix='YoY_%Change'):
        """
        Calculate Year-over-Year (YoY) change for a specific column in a pandas DataFrame.

        This method computes the YoY change, which includes the difference between the current value
        and the value from the same period in the previous year, as well as the percentage change.

        Parameters
        ----------
        data : pandas DataFrame
            The DataFrame containing the data.
            
        col_name : str
            The name of the column for which YoY change is to be calculated.
            
        offset : pandas DateOffset, default=pd.tseries.offsets.DateOffset(years=1)
            The offset to use for calculating the previous year's value. Typically, this is set to one year.
            
        prev_suffix : str, default='PrevYearValue'
            Suffix for the column containing the previous year's value.
            
        change_suffix : str, default='YoY_Change'
            Suffix for the column containing the YoY change (difference).
            
        perc_change_suffix : str, default='YoY_%Change'
            Suffix for the column containing the YoY percentage change.

        Returns
        -------
        pandas DataFrame
            A DataFrame with the original column, the previous year's value, the YoY change,
            and the YoY percentage change.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from your_module import Utils
        >>> data = pd.DataFrame({'value': [100, 200, 150, 300]}, 
                                index=pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']))
        >>> result = Utils.yoy_change(data, 'value')
        >>> print(result)
                    value  value_PrevYearValue  value_YoY_Change  value_YoY_%Change
        2021-01-01    200                100.0             100.0               1.0
        2022-01-01    150                200.0             -50.0              -0.25
        2023-01-01    300                150.0             150.0               1.0
        """
        prev_data_df = pd.DataFrame(
            {f'{col_name}_{prev_suffix}': data[col_name].to_numpy()},
            index=data.index + offset
        )
        prev_data_df = pd.concat([data[col_name], prev_data_df], axis=1, join='inner')
        prev_data_df[f'{col_name}_{change_suffix}'] = prev_data_df[col_name] - prev_data_df[f'{col_name}_{prev_suffix}']
        prev_data_df[f'{col_name}_{perc_change_suffix}'] = prev_data_df[f'{col_name}_{change_suffix}'] / prev_data_df[f'{col_name}_{prev_suffix}']
        return prev_data_df
