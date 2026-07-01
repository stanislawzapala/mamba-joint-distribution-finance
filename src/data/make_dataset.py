import os
import pandas as pd
from pathlib import Path

from src.data.load_data import load_raw_djia
from src.data.preprocess import sort_data_by_date, calculate_returns

def main():
    print("Loading raw DJIA data...")

    input_folder = Path("data/raw/djia29")
    output_folder = Path("data/processed")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print(f"Input folder: {input_folder}")
    data_dict = load_raw_djia(input_folder)
    
    if not data_dict:
        print("No CSV files found in the input folder.")
        return
    
    for ticker, df in data_dict.items():
        print(f"Preprocessing data for {ticker}...")
        df = sort_data_by_date(df)
        df = calculate_returns(df)

        output_file = output_folder / f"{ticker}_preprocessed.csv"
        df.to_csv(output_file)
        print(f"Saved preprocessed data for {ticker} to {output_file}")


if __name__ == "__main__":
    main()