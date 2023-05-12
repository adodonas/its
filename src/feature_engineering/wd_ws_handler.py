import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def process_file(file_path, prefix):
    logging.info(f"Processing file {file_path}...")

    # load csv file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # validate required columns
    required_columns = ['station', 'timestamp', 'rh', 'wd', 'ws']
    df.columns.values[0:5] = required_columns
    if not all(column in df.columns for column in required_columns):
        logging.error(f"File {file_path} is missing required columns.")
        return None

    # convert 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')

    # check if 'rh', 'wd', 'ws' are numeric and if not, convert them
    for column in ['rh', 'wd', 'ws']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # drop rows with any invalid (NaN) values in 'rh', 'wd', 'ws' and 'timestamp' columns
    df.dropna(subset=['timestamp', 'rh', 'wd', 'ws'], inplace=True)

    # separate numeric and non-numeric columns for resampling
    df_numeric = df[['timestamp', 'rh', 'wd', 'ws']].set_index('timestamp')
    df_non_numeric = df[['timestamp', 'station']].set_index('timestamp')

    # resample the data to aggregate from 10 minutes to 1 hour
    df_numeric = df_numeric.resample('H').mean()
    df_non_numeric = df_non_numeric.resample('H').first()

    # merge the resampled dataframes
    df_resampled = pd.concat([df_non_numeric, df_numeric], axis=1).reset_index()

    df_resampled['area'] = prefix

    return df_resampled


def merge_and_aggregate_csvs(directory):
    # initialize an empty list to store dataframes
    df_list = []

    prefix = None
    # iterate through all csv files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            prefix = filename.split("_")[0]
            # construct full file path
            file_path = os.path.join(directory, filename)
            df = process_file(file_path, prefix)
            if df is not None:
                df_list.append(df)

    # merge all dataframes in the list
    merged_df = pd.concat(df_list, ignore_index=True)

    # save merged dataframe to a csv file
    merged_df.to_excel(f'{directory}/{prefix}_merged.xlsx', index=False)


directory = '../../data/air_sviva_gov_il/4y/wd_ws'

for item in os.listdir(directory):
    # construct full item path
    item_path = os.path.join(directory, item)

    if os.path.isdir(item_path):
        merge_and_aggregate_csvs(item_path)
