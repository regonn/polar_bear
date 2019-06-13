import numpy as np
import pandas as pd
import optuna
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

int_dtype_list = ['int8', 'int16', 'int32',
                  'int64', 'uint8', 'uint16', 'uint32', 'uint64']
float_dtype_list = ['float16', 'float32', 'float64']


def _target_data(train_df: pd.DataFrame, target_col: str) -> pd.Series:
    """Get target column and data from train data

    Extended description of function.

    Parameters
    ----------
    train_df : pd.DataFrame
        train data
    target_col : str
        target column name

    Returns
    -------
    pd.Series

    >>> import pandas as pd
    >>> data = pd.DataFrame({"param": [1, 2, 3], "target": [1, 0, 1]})
    >>> _target_data(data, "target")
       y1:target
    0          1
    1          0
    2          1
    """
    target_series = train_df[target_col]
    target_series.name = "y1:" + target_col
    return target_series


def convert_series(series: pd.Series, threshold_one_hot=0.3) -> pd.DataFrame:
    dtype = series.dtype
    value_counts = series.value_counts()
    value_counts_number = value_counts.shape[0]
    rows_count = len(series)
    return_df = pd.DataFrame()

    if dtype in int_dtype_list:
        if value_counts_number < (rows_count * threshold_one_hot):
            mode_value = value_counts.index[0]
            series[np.isnan(series)] = mode_value
            one_hot_df = pd.get_dummies(series, prefix=series.name)
            for one_hot_label, one_hot_content in one_hot_df.iteritems():
                return_df[one_hot_label] = one_hot_content
    elif dtype in float_dtype_list:
        if value_counts_number < (rows_count * threshold_one_hot):
            mode_value = series.value_counts().index[0]
            series[np.isnan(series)] = mode_value
            one_hot_df = pd.get_dummies(series, prefix=series.name)
            for one_hot_label, one_hot_content in one_hot_df.iteritems():
                return_df[one_hot_label] = one_hot_content
        else:
            mean = series.mean()
            series[np.isnan(series)] = mean
            return_df[series.name + "_float"] = series
    elif (dtype == 'object') or (dtype == 'bool'):
        if value_counts_number < (rows_count * threshold_one_hot):
            mode_value = series.value_counts().index[0]
            series[pd.isnull(series)] = mode_value
            one_hot_df = pd.get_dummies(series, prefix=series.name)
            for one_hot_label, one_hot_content in one_hot_df.iteritems():
                return_df[one_hot_label] = one_hot_content
    return return_df


def _make_return_df(train_df, test_df, target_col, threshold_one_hot) -> pd.DataFrame:
    return_df = pd.DataFrame()
    feature_column_index = 1

    for label, series in train_df.iteritems():
        if (label == target_col):
            continue

        series = pd.concat([series, test_df[label]])

        value_counts = series.value_counts()
        value_counts_number = value_counts.shape[0]

        if (value_counts_number == 1):
            continue

        converted_df = convert_series(series, threshold_one_hot)

        for converted_label, converted_content in converted_df.iteritems():
            return_df["x" + str(feature_column_index)
                      + ":" + converted_label] = converted_content
            feature_column_index += 1

    return return_df


def _wrapper_objective(train_df, target_series, target_col, test_df):
    target_dtype = target_series.dtype
    n_estimators = 10
    if target_dtype in float_dtype_list:
        if target_series.value_counts().shape[0] < 10:
            rf = ensemble.RandomForestClassifier(n_estimators=n_estimators)
            rf_type = 'classifier'
        else:
            rf = ensemble.RandomForestRegressor(n_estimators=n_estimators)
            rf_type = 'regressor'
    else:
        rf = ensemble.RandomForestClassifier(n_estimators=n_estimators)
        rf_type = 'classifier'

    def objective(trial):
        threshold_one_hot = trial.suggest_int('threshold_one_hot', 0, 100) * 0.01
        return_df = _make_return_df(
            train_df, test_df, target_col, threshold_one_hot)
        X_train, X_test, y_train, y_test = train_test_split(
            return_df[0:len(train_df)].values, target_series.values, test_size=0.2, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        if rf_type == 'classifier':
            return 1.0 - metrics.accuracy_score(y_test, y_pred)
        if rf_type == 'regressor':
            return 1.0 - metrics.r2_score(y_test, y_pred)
    return objective


def clean(train_df, test_df, target_col, threshold_one_hot=None):
    target_series = _target_data(train_df, target_col)

    if threshold_one_hot is None:
        study = optuna.create_study()
        study.optimize(_wrapper_objective(
            train_df, target_series, target_col, test_df), n_trials=100, timeout=10*60)
        return_df = _make_return_df(
            train_df, test_df, target_col, study.best_params['threshold_one_hot'] * 0.01)
    else:
        return_df = _make_return_df(
            train_df, test_df, target_col, threshold_one_hot)

    return return_df[0:len(train_df)], target_series, return_df[len(train_df):]
