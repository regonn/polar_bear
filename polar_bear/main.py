import numpy as np
import pandas as pd

int_dtype_list = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
float_dtype_list = ['float16', 'float32', 'float64', 'float128']


def clean(train_df, test_df, target_col, threshold_one_hot=0.1):
    return_df = pd.DataFrame()
    rows_count = len(train_df) + len(test_df)
    feature_column_index = 1

    for label, content in train_df.iteritems():
        if label == target_col:
            target_df = pd.DataFrame()
            target_df["y1:" + label] = content
            continue

        content = pd.concat([content, test_df[label]])
        dtype = content.dtype

        if dtype in int_dtype_list:
            value_counts = content.value_counts().shape[0]
            if value_counts < (rows_count * threshold_one_hot):
                mode = content.value_counts().index[0]
                content[np.isnan(content)] = mode
                one_hot_df = pd.get_dummies(content, prefix=label)
                for one_hot_label, one_hot_content in one_hot_df.iteritems():
                    return_df["x" + str(feature_column_index) + ":" + one_hot_label] = one_hot_content
                    feature_column_index += 1
        elif dtype in float_dtype_list:
            value_counts = content.value_counts().shape[0]
            if value_counts < (rows_count * threshold_one_hot):
                mode = content.value_counts().index[0]
                content[np.isnan(content)] = mode
                one_hot_df = pd.get_dummies(content, prefix=label)
                for one_hot_label, one_hot_content in one_hot_df.iteritems():
                    return_df["x" + str(feature_column_index) + ":" + one_hot_label] = one_hot_content
                    feature_column_index += 1
            else:
                mean = content.mean()
                content[np.isnan(content)] = mean
                return_df["x" + str(feature_column_index) + ":" + label + "_float"] = content
                feature_column_index += 1
        elif (dtype == 'object') or (dtype == 'bool'):
            value_counts = content.value_counts().shape[0]
            if value_counts < (rows_count * threshold_one_hot):
                mode = content.value_counts().index[0]
                content[pd.isnull(content)] = mode
                one_hot_df = pd.get_dummies(content, prefix=label)
                for one_hot_label, one_hot_content in one_hot_df.iteritems():
                    return_df["x" + str(feature_column_index) + ":" + one_hot_label] = one_hot_content
                    feature_column_index += 1

    return return_df[0:len(train_df)], target_df, return_df[len(train_df):]
