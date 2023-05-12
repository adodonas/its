import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_paths = self.get_excel_files_in_folder()
        self.combined_df = self.read_and_clean_data()

    def get_excel_files_in_folder(self):
        return [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if file.endswith('.xlsx')]

    def read_and_clean_data(self):
        dfs = []

        for file_path in self.file_paths:
            df = pd.read_excel(file_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M %d/%m/%Y', errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)

            for column in df.columns.difference(['Timestamp']):
                df[column] = pd.to_numeric(df[column], errors='coerce')

            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.dropna(inplace=True)
        combined_df.sort_values(by='Timestamp', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Group by Timestamp and calculate the mean for each group
        combined_df = combined_df.groupby('Timestamp', as_index=False).mean()

        return combined_df

    def normalize_data(self):
        df_normalized = self.combined_df.copy()
        scaler = MinMaxScaler()

        for column in df_normalized.columns.difference(['Timestamp']):
            df_normalized[column] = scaler.fit_transform(df_normalized[[column]])

        return df_normalized

    def add_features(self):
        df_with_features = self.combined_df.copy()

        # Extract day of the week, month, and hour from the 'Timestamp' column
        df_with_features['Day_of_week'] = df_with_features['Timestamp'].dt.dayofweek
        df_with_features['Month'] = df_with_features['Timestamp'].dt.month
        df_with_features['Hour'] = df_with_features['Timestamp'].dt.hour

        return df_with_features

    def plot_normalized_data(self, df_normalized=None, figsize=(12, 8), colors=None):
        if df_normalized is None:
            df_normalized = self.normalize_data()

        columns_to_plot = df_normalized.columns.difference(['Timestamp'])

        if colors is None:
            colors = ['blue', 'green', 'red', 'purple', 'orange']

        num_columns = len(columns_to_plot)
        fig, axs = plt.subplots(num_columns, 1, figsize=figsize, sharex=True)
        fig.tight_layout(pad=3)

        for idx, column in enumerate(columns_to_plot):
            axs[idx].plot(df_normalized['Timestamp'], df_normalized[column], color=colors[idx % len(colors)])
            axs[idx].set_ylabel(column)
            axs[idx].grid(True)

        axs[-1].set_xlabel('Timestamp')
        plt.show()
