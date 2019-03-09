import numpy as np
import pandas as pd

int_dtype_list = ['int8', 'int16', 'int32',
                  'int64', 'uint8', 'uint16', 'uint32', 'uint64']
float_dtype_list = ['float16', 'float32', 'float64']


def _detect_type(content, threshold_category_rate):
    dtype = content.dtype
    rows_count = len(content)
    value_counts = content.value_counts()
    value_counts_number = value_counts.shape[0]
    if dtype in int_dtype_list:
        if all(key in [0, 1] for key in content.value_counts().keys()):
            return 'binary'
        elif value_counts_number < (rows_count * threshold_category_rate):
            return 'category'
        else:
            return 'numerical'
    if dtype in float_dtype_list:
        return 'numerical'
    if (dtype == 'bool') or (dtype == 'object'):
        return 'category'


def make_model_definition_for_ludwig(df, target_col, threshold_category_rate=0.3):
    return_dict = {}
    input_features = []
    rows_count = len(df)
    target_df = df[target_col]

    return_dict['output_features'] = [
        {'name': target_col, 'type': _detect_type(target_df, threshold_category_rate)}]

    for label, content in df.iteritems():
        value_counts = content.value_counts()
        value_counts_number = value_counts.shape[0]

        if (value_counts_number == 1) or (value_counts_number == rows_count) or (label == target_col):
            continue

        if (content.dtype == 'object') and (value_counts_number > (rows_count * threshold_category_rate)):
            continue

        input_features.append(
            {'name': label, 'type': _detect_type(content, threshold_category_rate)})

    return_dict['input_features'] = input_features

    print(return_dict)
    return return_dict
