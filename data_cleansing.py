import numpy as np
import pandas as pd

def data_cleansing(data_df):
    data_df = data_df.replace(-1, np.NaN)
    delete_cols = ['ps_reg_03', 'ps_car_03_cat', 'ps_car_05_cat']
    data_df.drop(delete_cols, axis = 1)
    for colname, col in data_df.iteritems():
        col.replace(np.NaN, col.mean())

    headers = list(data_df.columns.values)
    cat_headers = list(filter(lambda x : 'cat' in x, headers))

    for h in headers:
        if h in cat_headers:
            #convert cat to one-hot-key vector
            hot_key = pd.get_dummies(pd.Series(data_df[h]))
            data_df = pd.concat([data_df,hot_key],axis=1)
            data_df = data_df.drop([h],axis=1)
        else:
            #rescale this column
            max_val = data_df[h].max()
            data_df[h] /= max_val
    return data_df

def get_train_data(train_path):
    train_df = pd.read_csv(train_path)
    del train_df['id']
    target = train_df['target']
    del train_df['target']
    train_df = data_cleansing(train_df)
    return (train_df.as_matrix(), target.values)

def get_test_data(test_path):
    test_df = pd.read_csv(test_path)
    test_id = test_df['id']
    del test_df['id']
    return (test_id, data_cleansing(test_df).as_matrix())