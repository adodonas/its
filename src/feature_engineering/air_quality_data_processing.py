import glob
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import textwrap

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


def perform_ttest(lockdown_data, jewish_holiday_data):
    logging.info("Performing t-test...")
    t_stat, p_value = stats.ttest_ind(lockdown_data, jewish_holiday_data)
    logging.info(f"t-statistic: {t_stat}, p-value: {p_value}")
    return t_stat, p_value


def create_hypothesis_text(p_value):
    hypothesis_result = ""
    if p_value <= 0.05:
        hypothesis_result = "The p-value is less than or equal to 0.05, which suggests that the NO levels during lockdowns and Jewish holidays are significantly different."
        logging.info(hypothesis_result)
    else:
        hypothesis_result = "The p-value is greater than 0.05, which suggests that the NO levels during lockdowns and Jewish holidays are not significantly different."
        logging.info(hypothesis_result)

    # Wrap text to 150 characters
    wrapper = textwrap.TextWrapper(width=150)
    word_list = wrapper.wrap(text=hypothesis_result)
    hypothesis_result = '\n'.join(word_list)
    return hypothesis_result


def create_plots(final_df, lockdown_data, jewish_holiday_data, t_stat, p_value, hypothesis_result):
    logging.info("Creating visualization...")

    # First plot
    plt.figure(figsize=(10, 6))
    plt.plot(final_df['timestamp'], final_df['NO'], label='NO levels')
    plt.fill_between(final_df['timestamp'], final_df['NO'], where=final_df['lockdown'] == 1, color='green', alpha=0.3,
                     label='Lockdown period')
    plt.fill_between(final_df['timestamp'], final_df['NO'], where=final_df['jewish_holiday'] == 1, color='red',
                     alpha=0.3, label='Jewish holiday period')
    plt.xlabel('Timestamp\n\n' + hypothesis_result)
    plt.legend()
    plt.savefig(f'NO_levels_during_periods_t_{t_stat}_p_{p_value}.png')

    # Second plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Lockdown', 'Jewish Holiday'], [lockdown_data.mean(), jewish_holiday_data.mean()], color=['green', 'red'])
    plt.ylabel('Average NO levels')
    plt.title('Comparison of average NO levels during lockdowns and Jewish holidays')
    plt.xlabel('Condition\n\n' + hypothesis_result)
    plt.legend()
    plt.savefig(f'Average_NO_levels_during_periods_t_{t_stat}_p_{p_value}.png')

    logging.info("Plots saved.")


def analyse_and_visualize(final_df):
    logging.info("Starting analysis and visualization...")

    # Extract data for lockdowns and Jewish holidays
    logging.info("Extracting data for lockdowns and Jewish holidays...")
    lockdown_data = final_df[final_df['lockdown'] == 1]['NO']
    jewish_holiday_data = final_df[final_df['jewish_holiday'] == 1]['NO']

    # Perform t-test
    t_stat, p_value = perform_ttest(lockdown_data, jewish_holiday_data)

    # Create hypothesis text
    hypothesis_result = create_hypothesis_text(p_value)

    # Create and save plots
    create_plots(final_df, lockdown_data, jewish_holiday_data, t_stat, p_value, hypothesis_result)

    logging.info("Analysis and visualization complete.")


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

        jewish_holidays_dates = [
            ("2018-09-18", "2018-09-19"), ("2019-10-08", "2019-10-09"), ("2020-09-27", "2020-09-28"),
            ("2021-09-15", "2021-09-16"), ("2022-10-04", "2022-10-05"),  # Yom Kippur
            ("2018-05-19", "2018-05-20"), ("2019-06-08", "2019-06-10"), ("2020-05-28", "2020-05-30"),
            ("2021-05-16", "2021-05-18"), ("2022-06-04", "2022-06-06"),  # Shavuot
            ("2018-03-30", "2018-04-07"), ("2019-04-19", "2019-04-27"), ("2020-04-08", "2020-04-16"),
            ("2021-03-27", "2021-04-04"), ("2022-04-15", "2022-04-23"),  # Passover
            ("2018-09-09", "2018-09-11"), ("2019-09-29", "2019-10-01"), ("2020-09-18", "2020-09-20"),
            ("2021-09-06", "2021-09-08"), ("2022-09-25", "2022-09-27"),  # Rosh Hashana
            ("2018-09-23", "2018-10-02"), ("2019-10-13", "2019-10-22"), ("2020-10-02", "2020-10-11"),
            ("2021-09-20", "2021-09-29"), ("2022-10-09", "2022-10-18")  # Sukkot
        ]

        df_merged['lockdown'] = df_merged['timestamp'].apply(lambda date: is_within_date_range(date, lockdown_dates))
        df_merged['jewish_holiday'] = df_merged['timestamp'].apply(
            lambda date: is_within_date_range(date, jewish_holidays_dates))

        analyse_and_visualize(df_merged)

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


def is_within_date_range(date, date_ranges):
    for start, end in date_ranges:
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
