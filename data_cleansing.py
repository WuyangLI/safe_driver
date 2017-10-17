import numpy as np
import pandas as pd
from feature_selection import get_selected_data, get_model_for_data_selection

def feature_engineering(data_df):
    delete_cols = ['ps_reg_03', 'ps_car_03_cat', 'ps_car_05_cat']
    data_df.drop(delete_cols, axis = 1)
    for colname,col in data_df.iteritems():
        col.replace(-1, col.value_counts().idxmax())

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
    return data_df.as_matrix()


def get_train_data(train_path, thresh):
    train_raw_df = pd.read_csv(train_path)
    del train_raw_df['id']
    target = train_raw_df['target'].as_matrix()
    del train_raw_df['target']
    train_df = feature_engineering(train_raw_df)
    model_for_feature_selection = get_model_for_data_selection(train_df, target)
    selected_data = get_selected_data(thresh, model_for_feature_selection, train_df)
    return (selected_data, target, model_for_feature_selection)

def get_test_data(test_path, thresh, model_for_feature_selection):
    test_df = pd.read_csv(test_path)
    test_id = test_df['id'].as_matrix()
    del test_df['id']
    return (test_id, get_selected_data(thresh, model_for_feature_selection, feature_engineering(test_df)))
