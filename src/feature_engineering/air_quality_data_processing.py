import os
import numpy as np
import pandas as pd
import glob
import logging
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def load_and_process_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        return df
    except Exception as e:
        logging.error(f"Error occurred in load_and_process_file: {e}")


def merge_and_process_files(df1, second_folder_files, prefix):
    all_merged_dfs = []  # list to hold all merged DataFrames
    except_columns = ['station', 'area', 'timestamp']
    for second_file in second_folder_files:
        df2 = pd.read_excel(second_file)
        df_merged = pd.merge(df1, df2, on='timestamp', how='outer')

        for col in df_merged.columns:
            if col not in except_columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        df_merged.dropna(inplace=True)

        df_merged = add_day_area_and_rearrange_columns(df_merged, prefix)

        lockdown_dates = [("2020-03-17", "2020-04-19"), ("2020-09-11", "2020-10-13"), ("2020-12-27", "2021-02-07")]

        df_merged['lockdown'] = df_merged['timestamp'].apply(lambda date: is_in_lockdown(date, lockdown_dates))

        all_merged_dfs.append(df_merged)  # add this DataFrame to the list

    return all_merged_dfs


def label_encode(df):
    le = LabelEncoder()
    df['area'] = le.fit_transform(df['area'])
    return df


def add_day_area_and_rearrange_columns(df_merged, prefix):
    # Add 'day_of_week' column
    df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
    df_merged = df_merged.sort_values('timestamp')
    df_merged = df_merged.reset_index(drop=True)
    station = df_merged['station']
    df_merged = df_merged.drop('station', axis=1)  # drop station column
    df_merged['station'] = station  # add it back, this will add it at the end
    df_merged['area'] = prefix  # add area column
    return df_merged


def is_in_lockdown(date, lockdown_dates):
    for start, end in lockdown_dates:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start <= date <= end:
            return 1
    return 0



def process_folders(no_no2_nox_so2_folder, wd_ws_folder):
    try:
        first_folder_files = glob.glob(os.path.join(no_no2_nox_so2_folder, '*.xlsx'))
        second_folder_dirs = [d for d in os.listdir(wd_ws_folder) if os.path.isdir(os.path.join(wd_ws_folder, d))]

        final_df_list = []

        for first_file in first_folder_files:
            prefix = os.path.splitext(os.path.basename(first_file))[0].lower()
            prefix = prefix.split('_')[0]

            df1 = load_and_process_file(first_file)

            for second_dir in second_folder_dirs:
                if second_dir.lower() == prefix:
                    second_folder_files = glob.glob(os.path.join(wd_ws_folder, second_dir, '*.xlsx'))
                    all_merged_dfs = merge_and_process_files(df1, second_folder_files, prefix)
                    final_df_list.extend(all_merged_dfs)

        final_df = pd.concat(final_df_list, ignore_index=True)
        final_df = label_encode(final_df)
        final_df_normal = final_df.copy()
        columns_to_normalize = ['NO', 'NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws']

        scaler = MinMaxScaler()
        final_df_normal[columns_to_normalize] = scaler.fit_transform(final_df_normal[columns_to_normalize])

        os.makedirs('results', exist_ok=True)

        final_df.to_excel('results/final_merged.xlsx', index=False)
        final_df_normal.to_excel('results/final_normal_merged.xlsx', index=False)

    except Exception as e:
        logging.error(f"Error occurred in process_folders: {e}")


first_folder = '../../data/air_sviva_gov_il/4y/no_no2_nox_so2'
second_folder = '../../data/air_sviva_gov_il/4y/wd_ws'

process_folders(first_folder, second_folder)
