import logging
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class IsraeliEnvProtectionDataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_paths = self.__get_excel_files_in_folder()
        logging.info(f"Found {len(self.file_paths)} Excel files in the folder.")
        self.combined_df = self.__read_and_clean_data()

    def __get_excel_files_in_folder(self):
        return [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if file.endswith('.xlsx')]

    def __read_and_clean_data(self):
        dfs = []
        for index, file_path in enumerate(self.file_paths, start=1):
            logging.info(f"Processing file {index}/{len(self.file_paths)}: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl')
            df_cleaned = self.__set_timestamp_column(df)

            for column in df_cleaned.columns.difference(['Timestamp']):
                df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')

            dfs.append(df_cleaned)

        # Convert Timestamp to a common datetime format
        for df in dfs:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M %d/%m/%Y', errors='coerce')

        combined_df = reduce(lambda left, right:
                             pd.merge(left, right, on='Timestamp', how='outer'), dfs)
        combined_df = combined_df.sort_values(by='Timestamp').reset_index(drop=True)
        logging.info("Finished processing all files.")
        return combined_df

    def __set_timestamp_column(self, df_cleaned):
        if 'Timestamp' not in df_cleaned.columns:
            nan_columns = df_cleaned.columns[df_cleaned.columns.isna()].tolist()
            if len(nan_columns) > 0:
                nan_column = nan_columns[0]
                df_cleaned = df_cleaned.rename(columns={nan_column: 'Timestamp'}).drop(0).reset_index(drop=True)
            else:
                logging.warning("No 'Timestamp' column found. Please check the data.")
        df_cleaned.replace(0, np.nan, inplace=True)
        return df_cleaned

    def __validate_timestamps(self):
        unique_formats = set()
        date_pattern = re.compile(r'\d{1,2}:\d{2} \d{1,2}/\d{1,2}/\d{2}')

        for date_str in self.combined_df['Timestamp']:
            if not date_pattern.match(date_str):
                unique_formats.add(date_str)

        return unique_formats

    def __aggregate_data(self, freq='H'):  # Changing the frequency to 'H' for hourly data
        df = self.combined_df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M %d/%m/%Y', errors='coerce')
        df['Timestamp'] = df['Timestamp'].dt.strftime(
            '%H:%M %d/%m/%Y')  # Formatting the Timestamp back to the desired format
        df = df.set_index('Timestamp')
        df_aggregated = df.resample(freq).mean().reset_index()

        if freq == 'M':
            df_aggregated['Timestamp'] = df_aggregated['Timestamp'].apply(lambda date: date.replace(day=1))

        return df_aggregated

    def process_data(self):
        aggregated_data = self.__aggregate_data().dropna()
        logging.info("Aggregated data processing complete.")
        return aggregated_data
