import ast
import os
from os import environ

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(env_path)

batch_size = int(environ.get('BATCH_SIZE'))
padding_value = float(environ.get('PADDING_VALUE'))
test_size = float(environ.get('TEST_SIZE'))


class DataPreparation():

    def __init__(self, data_col_name:str):
        self.data_col_name = data_col_name

    def do_all_data_preparation(self, df_x, df_y):

        x_df, y_df = self.convert_data_to_list(df_x, df_y,
                                           self.data_col_name)

        X_train, X_val, y_train, y_val = train_test_split(x_df,
                                                          y_df,
                                                          test_size=test_size,
                                                          random_state=42)

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_function)

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=self.collate_function)

        return train_loader, val_loader

    @staticmethod
    def convert_data_to_list(x_df: pd.DataFrame, y_df: pd.DataFrame, data_col_name:str):

        x_df[data_col_name] = x_df[data_col_name].apply(lambda x: list(map(float, ast.literal_eval(x))))
        y_df[data_col_name] = y_df[data_col_name].apply(lambda x: list(map(float, ast.literal_eval(x))))

        assert all(len(x) == len(y) for x, y in zip(x_df[data_col_name], y_df[data_col_name])),\
            "Sequence lengths do not match, check the files"

        return x_df[data_col_name].tolist(), y_df[data_col_name].tolist()

    @staticmethod
    def collate_function(batch):
        x, y = zip(*batch)

        x_padded = pad_sequence(x, batch_first=True, padding_value=padding_value)
        y_padded = pad_sequence(y, batch_first=True, padding_value=padding_value)

        return x_padded, y_padded


class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self._prepare_data()

    def _prepare_data(self):

        all_x_data = np.concatenate(self.x_data).reshape(-1, 1)
        all_y_data = np.concatenate(self.y_data).reshape(-1, 1)

        self.scaler_x.fit(all_x_data)
        self.scaler_y.fit(all_y_data)

        self.scaled_tok = [self.scaler_x.transform(np.array(seq).reshape(-1, 1)) for seq in self.x_data]
        self.scaled_dynamo = [self.scaler_y.transform(np.array(seq).reshape(-1, 1)) for seq in self.y_data]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.scaled_tok[idx])
        y = torch.FloatTensor(self.scaled_dynamo[idx])
        return x, y
