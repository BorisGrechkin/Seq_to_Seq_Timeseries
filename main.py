import pandas as pd

from Scripts.data_preprocess import DataPreparation
from Scripts.model_training import train_model

tok_df_path = 'Data/tokogramms_all.csv'
dynamo_df_path = 'Data/dynamogramms_all.csv'

if __name__ == '__main__':

    x_df = pd.read_csv(tok_df_path)
    y_df = pd.read_csv(dynamo_df_path)

    data_prep = DataPreparation('data')
    train_loader, val_loader = data_prep.do_all_data_preparation(x_df, y_df)
    print('Data preparation done')

    train_model(train_loader, val_loader)
