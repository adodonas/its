import os

import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_and_process_file(file_path):
    df = pd.read_excel(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    return df


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


def plot_data(df):
    # Set 'timestamp' as the DataFrame's index
    df.set_index('timestamp', inplace=True)

    # Select the columns you want to resample
    cols_to_resample = ['NO', 'NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws']
    resampled_df = df[cols_to_resample].resample('M').mean()

    # Resample the columns you want to include but not take the mean of
    # Here, we're taking the most frequent value ('mode') each month
    cols_to_include = ['area', 'day_of_week']
    for col in cols_to_include:
        resampled_df[col] = df[col].resample('M').apply(lambda x: x.mode()[0] if not x.empty else np.nan)

    # Reset the index
    resampled_df.reset_index(inplace=True)

    columns_to_plot = ['NO', 'NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws', 'area', 'day_of_week']

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(20, 30))

    for i, col in enumerate(columns_to_plot):
        axs[i].plot(resampled_df['timestamp'], resampled_df[col])
        axs[i].set_title(col)

    # Automatically adjust subplot parameters to give specified padding
    fig.tight_layout()

    plt.show()


def process_folders(no_no2_nox_so2_folder, wd_ws_folder):
    first_folder_files = glob.glob(os.path.join(no_no2_nox_so2_folder, '*.xlsx'))
    second_folder_dirs = [d for d in os.listdir(wd_ws_folder) if os.path.isdir(os.path.join(wd_ws_folder, d))]

    final_df_list = []  # list to hold all final DataFrames from each prefix

    for first_file in first_folder_files:
        prefix = os.path.splitext(os.path.basename(first_file))[0].lower()
        prefix = prefix.split('_')[0]

        df1 = load_and_process_file(first_file)

        for second_dir in second_folder_dirs:
            if second_dir.lower() == prefix:
                second_folder_files = glob.glob(os.path.join(wd_ws_folder, second_dir, '*.xlsx'))
                all_merged_dfs = merge_and_process_files(df1, second_folder_files, prefix)
                final_df_list.extend(all_merged_dfs)  # add all DataFrames from this prefix to the final list

    final_df = pd.concat(final_df_list, ignore_index=True)  # concatenate all DataFrames into one
    final_df = label_encode(final_df)
    final_df_normal = final_df.copy()  # create a copy of the original DataFrame
    columns_to_normalize = ['NO', 'NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws']

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    final_df_normal[columns_to_normalize] = scaler.fit_transform(final_df_normal[columns_to_normalize])

    final_df.to_excel('final_merged.xlsx', index=False)  # save the final DataFrame to a single file
    final_df_normal.to_excel('final_normal_merged.xlsx', index=False)


first_folder = 'W:/ws/afeka/פרויקט_גמר/its/data/air_sviva_gov_il/4y/no_no2_nox_so2'
second_folder = 'W:/ws/afeka/פרויקט_גמר/its/data/air_sviva_gov_il/4y/wd_ws'

process_folders(first_folder, second_folder)
