from data_cleansing import get_test_data, get_train_data
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import pandas as pd


def gini(y_true, y_pred):
    """ Simple implementation of the (normalized) gini score in numpy.
        Fully vectorized, no python loops, zips, etc. Significantly
        (>30x) faster than previous implementions

        Credit: https://www.kaggle.com/jpopham91/
    """

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


def normalized_gini(y_true, y_pred):
    ng = gini(y_true, y_pred) / gini(y_true, y_true)
    return ng

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = normalized_gini(labels, preds)
    return [('gini', gini_score)]

def train_model(K):
    (X, y) = get_train_data("../input/train.csv")
    test_set = get_test_data("../input/test.csv")
    skf = StratifiedKFold(n_splits=K)
    xgb_preds = []
    for train_index, valid_index in skf.split(X, y):
        train_X, valid_X = X[train_index], X[valid_index]
        train_y, valid_y = y[train_index], y[valid_index]

        xgb_params = {'eta': 0.05, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic',
                  'eval_metric': 'auc', 'seed': 99, 'silent': True}

        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(test_set)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(xgb_params, d_train, 2000, watchlist, feval=gini_xgb, maximize=True, verbose_eval=50,
                          early_stopping_rounds=100)

        xgb_pred = model.predict(d_test)
        xgb_preds.append(list(xgb_pred))

    preds = []
    for i in range(len(xgb_preds[0])):
        sum = 0
        for j in range(K):
            sum += xgb_preds[j][i]
        preds.append(sum / K)

    id_test = pd.Series(range(0, len(preds)))
    output = pd.DataFrame({'id': id_test, 'target': preds})
    output.to_csv("{}-foldCV_avg_sub.csv".format(K), index=False)

if __name__ == "__main__":
    train_model(5)
