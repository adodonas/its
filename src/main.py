import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")


def read_and_clean_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl').iloc[:-10]
    df_cleaned = df.drop([0, 1, 3])
    df_cleaned.columns = df_cleaned.iloc[0]
    df_cleaned = df_cleaned.reset_index(drop=True)

    nan_column = df_cleaned.columns[df_cleaned.columns.isna()].tolist()[0]
    df_cleaned = df_cleaned.rename(columns={nan_column: 'Timestamp'}).drop(0).reset_index(drop=True)
    df_cleaned.replace(0, np.nan, inplace=True)

    columns_to_exclude = ['Timestamp', 'PM2.5']
    for column in df_cleaned.columns.difference(columns_to_exclude):
        df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')

    df_cleaned = df_cleaned.dropna(axis=0, how='any').reset_index(drop=True)

    return df_cleaned


def validate_timestamps(df):
    unique_formats = set()
    date_pattern = re.compile(r'\d{1,2}:\d{2} \d{1,2}/\d{1,2}/\d{2}')

    for date_str in df['Timestamp']:
        if not date_pattern.match(date_str):
            unique_formats.add(date_str)

    return unique_formats


def aggregate_data(df, freq='D'):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M %d/%m/%Y', errors='coerce')
    df = df.set_index('Timestamp')
    df_aggregated = df.resample(freq).mean().reset_index()

    if freq == 'M':
        df_aggregated['Timestamp'] = df_aggregated['Timestamp'].apply(lambda date: date.replace(day=1))

    return df_aggregated


def convert_date_format(date_string):
    return pd.to_datetime(date_string).strftime('%Y-%m-%d %H:%M')


def process_and_plot_data(df):
    plt.figure(figsize=(20, 18))

    variables = ['NO', 'NO2', 'NOX', 'O3', 'PM10', 'RH', 'Temp', 'WD', 'WS', 'STT']
    for index, var in enumerate(variables):
        plt.subplot(3, 4, index + 1)
        sns.lineplot(data=df, x='Timestamp', y=var)
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.title(f'{var} Monthly (Normalized)')

    plt.tight_layout()
    plt.show()


def main():
    file_path1 = '../data/IEP/דוח תחנה 01_01_2020.xlsx'
    file_path2 = '../data/IEP/דוח תחנה 01_06_2020.xlsx'

    df1 = read_and_clean_data(file_path1)
    df2 = read_and_clean_data(file_path2)
    combined_df = pd.concat([df1, df2], axis=0)
    unique_formats = validate_timestamps(combined_df)

    if len(unique_formats) > 0:
        print(f"Warning: Found inconsistent date formats: {unique_formats}")

    df_monthly = aggregate_data(combined_df, freq='M').dropna()

    # Process and merge STT data
    df_stt = pd.read_excel('data/file_b9f4265a-22a5-4883-91b7-6106f6a12fac.xlsx', engine='openpyxl').iloc[19:]
    df_stt.columns = df_stt.iloc[0]
    df_stt = df_stt.reset_index(drop=True).drop(0).reset_index(drop=True)
    df_stt['תקופה'] = df_stt[df_stt.columns[0]].apply(convert_date_format)
    exclude_column = 'תקופה'
    for column in df_stt.columns.difference([exclude_column]):
        df_stt[column] = pd.to_numeric(df_stt[column], errors='coerce')

    df_stt['STT'] = df_stt.sum(axis=1)
    columns_to_keep = ['תקופה', 'STT']
    filtered_df = df_stt[columns_to_keep].rename(columns={'תקופה': 'Timestamp'})

    start_date = '2020-01-01'
    end_date = '2020-10-01'
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
    filtered_df = filtered_df[
        (filtered_df['Timestamp'] >= pd.Timestamp(start_date)) & (filtered_df['Timestamp'] <= pd.Timestamp(end_date))]

    df_monthly = df_monthly.set_index('Timestamp')
    filtered_df = filtered_df.set_index('Timestamp')
    df_monthly = pd.concat([df_monthly, filtered_df], axis=1)

    # Normalize the data
    scaler = MinMaxScaler()
    columns_to_normalize = ['NO', 'NO2', 'NOX', 'O3', 'PM10', 'RH', 'Temp', 'WD', 'WS', 'STT']
    df_monthly_normalized = df_monthly.copy()
    df_monthly_normalized[columns_to_normalize] = scaler.fit_transform(df_monthly[columns_to_normalize])

    # Plot the data
    process_and_plot_data(df_monthly_normalized)

    # Display the DataFrame
    print(df_monthly)

if __name__ == '__main__':
    main()
