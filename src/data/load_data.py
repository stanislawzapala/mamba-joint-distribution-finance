import pandas as pd
from pathlib import Path


def load_raw_djia(folder_path: str = 'data/raw'):
    """
    Load raw data from CSV files in a folder.

    Parameters:
    folder_path (str): The path to the folder containing CSV files.

    Returns:
    dictionary: A dictionary where keys are stock symbols and values are DataFrames containing the stock data.
    """

    path = Path(folder_path)
    csv_files = path.glob("*.csv")

    data_dict = {}

    for file in csv_files:
        ticker = file.stem
        df = pd.read_csv(file, index_col='date', parse_dates=['date'], skiprows=[1])
        data_dict[ticker] = df

    return data_dict


    

