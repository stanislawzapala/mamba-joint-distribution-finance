from arch import arch_model
import numpy as np
import pandas as pd
from scipy.stats import gennorm, rankdata
import seaborn as sns
import matplotlib.pyplot as plt

def sort_data_by_date(data):
    """
    Sort the DataFrame by date in ascending order.

    Parameters:
    data (pd.DataFrame): A DataFrame containing stock price data with a 'date' column.

    Returns:
    pd.DataFrame: The sorted DataFrame.
    """
    return data.sort_values(by='date', ascending=True)

def calculate_returns(data):
    """
    Calculate simple and log returns for a given DataFrame.

    Parameters:
    data (pd.DataFrame): A DataFrame containing stock price data with a 'close' column.

    Returns:
    pd.DataFrame: The original DataFrame with added 'returns' and 'log_returns' columns.
    """
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    return data

def add_garch_residuals(df: pd.DataFrame, column: str = 'log_returns') -> pd.DataFrame:
    """
    Fit a GARCH(1,1) model to the specified column of a DataFrame and return the standardized residuals.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to fit the GARCH model on. Default is 'log_returns'.

    Returns:
    pd.DataFrame: The DataFrame with the added 'garch_residuals' column.
    """
    series = df[column] * 100
    model = arch_model(series.dropna(), vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    df['garch_residuals'] = garch_fit.std_resid
    return df


def normalize_to_uniform(df: pd.DataFrame, column: str = 'log_returns', method: str = 'gennorm') -> pd.DataFrame:
    """
    Normalize the specified columns of a DataFrame to a uniform distribution [0,1].

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to normalize. Default is 'log_returns'.
    method (str): The normalization method. Default is 'gennorm'.

    Returns:
    pd.DataFrame: The DataFrame with normalized columns.
    """
    df_copy = df.copy()

    valid_data = df_copy[column].dropna()

    if method == 'gennorm':
        beta, loc, scale = gennorm.fit(valid_data)
        uniform_data = gennorm.cdf(valid_data, beta, loc=loc, scale=scale)

    elif method == 'ecdf':
        uniform_data = rankdata(valid_data, method = 'ordinal') / (len(valid_data) + 1)
    
    else: raise ValueError("Invalid method. Choose 'gennorm' or 'ecdf'.")

    df_copy.loc[valid_data.index, 'uniform_returns'] = uniform_data

    return df_copy

