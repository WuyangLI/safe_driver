import numpy as np
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from data_cleansing import get_train_data


def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def get_model_for_data_selection(data_df, label_df):
    model_for_feature_selection = XGBClassifier()
    model_for_feature_selection.fit(data_df, label_df)
    return model_for_feature_selection

def find_threshold(data_path="../input/train.csv"):
    # load data
    (train_df, label_df) = get_train_data(data_path)
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(train_df, label_df, test_size=0.33, random_state=7)
    # fit model on all training data
    model_for_feature_selection = get_model_for_data_selection(X_train, y_train)
    # make predictions for test data and evaluate
    y_pred = model_for_feature_selection.predict(X_test)
    predictions = [round(value) for value in y_pred]
    gini = gini_normalized(y_test, predictions)
    print("gini: %.4f" % gini)
    # Fit model using each importance as a threshold
    thresholds = sort(model_for_feature_selection.feature_importances_)
    print(thresholds)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model_for_feature_selection, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        gini = gini_normalized(y_test, predictions)
        print("Thresh=%.3f, n=%d, Gini: %.4f" % (thresh, select_X_train.shape[1], gini))


def get_selected_data(thresh, model_for_feature_selection, data_df):
    selection = SelectFromModel(model_for_feature_selection, thresh)
    selected_data = selection.transform(data_df)
    return selected_data

if __name__=="__main__":
    find_threshold()