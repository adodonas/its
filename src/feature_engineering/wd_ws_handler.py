import os

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def process_file(file_path, prefix):
    logging.info(f"Processing file {file_path}...")

    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = ['station', 'timestamp', 'rh', 'wd', 'ws']
    df.columns.values[0:5] = required_columns
    if not all(column in df.columns for column in required_columns):
        logging.error(f"File {file_path} is missing required columns.")
        return None

    # Convert 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')

    # Check if 'rh', 'wd', 'ws' are numeric and if not, convert them
    for column in ['rh', 'wd', 'ws']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with any invalid (NaN) values in 'rh', 'wd', 'ws' and 'timestamp' columns
    df.dropna(subset=['timestamp', 'rh', 'wd', 'ws'], inplace=True)

    # Convert wind direction to radians
    df['wd'] = np.deg2rad(df['wd'])

    # Compute x and y components of the wind
    df['x_component'] = df['ws'] * np.cos(df['wd'])
    df['y_component'] = df['ws'] * np.sin(df['wd'])

    # Set 'timestamp' as the index
    df.set_index('timestamp', inplace=True)

    # Separate numeric and non-numeric columns for resampling
    df_numeric = df[['rh', 'x_component', 'y_component', 'ws']]
    df_non_numeric = df[['station']]

    # Resample the data to aggregate from 10 minutes to 1 hour
    df_numeric = df_numeric.resample('H').mean()
    df_non_numeric = df_non_numeric.resample('H').first()

    # Calculate the average wind direction from the average x and y components
    df_numeric['avg_wind_direction'] = np.rad2deg(np.arctan2(df_numeric['y_component'], df_numeric['x_component']))

    # Handle the situation where the average direction is negative
    df_numeric['avg_wind_direction'] = df_numeric['avg_wind_direction'].apply(lambda x: x + 360 if x < 0 else x)

    # Merge the resampled dataframes
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
