# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
import unittest
import polar_bear as pb
import pandas as pd


class TestPolarBear(unittest.TestCase):

    def test_convert_series(self):
        train_df = pd.read_csv('tests/train.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_column = 'class_text'
        converted_train_df, converted_test_df = pb.convert_series(train_df[target_column], test_df[target_column])
        expected_converted_train_df = pd.read_csv('expected_converted_train.csv')
        expected_converted_test_df = pd.read_csv('expected_converted_test.csv')
        self.assertListEqual(converted_train_df.columns.to_list(), expected_converted_train_df.columns.to_list())
        self.assertListEqual(converted_test_df.columns.to_list(), expected_converted_test_df.columns.to_list())

        converted_train_with_nan_df, converted_test_with_nan_df = pb.convert_series(train_df[target_column], test_df[target_column], include_dummy_na=True)
        expected_converted_train_with_nan_df = pd.read_csv('expected_converted_with_nan_train.csv')
        expected_converted_test_with_nan_df = pd.read_csv('expected_converted_with_nan_test.csv')
        self.assertListEqual(converted_train_with_nan_df.columns.to_list(), expected_converted_train_with_nan_df.columns.to_list())
        self.assertListEqual(converted_test_with_nan_df.columns.to_list(), expected_converted_test_with_nan_df.columns.to_list())

    def test_clean(self):
        train_df = pd.read_csv('tests/train.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_col = 'target'
        cleaned_train_df, target_series, cleaned_test_df = pb.clean(
            train_df, test_df, target_col, 0.5)

        self.assertEqual(len(cleaned_test_df.dropna()), len(test_df))
        self.assertEqual(len(cleaned_train_df), len(target_series))
        self.assertEqual(len(cleaned_train_df.columns),
                         len(cleaned_test_df.columns))
        self.assertEqual(target_series.name, "y1:" + target_col)

        # Check no effect for original dataframe
        new_train_df = pd.read_csv('tests/train.csv')
        self.assertListEqual(
            new_train_df.columns.to_list(), train_df.columns.to_list())

        expected_train_df = pd.read_csv('tests/expected_train.csv')
        expected_test_df = pd.read_csv('tests/expected_test.csv')
        self.assertListEqual(cleaned_train_df.columns.to_list(),
                             expected_train_df.columns.to_list())
        self.assertListEqual(cleaned_test_df.columns.to_list(),
                             expected_test_df.columns.to_list())

    def test_clean_optuna_classifier(self):
        train_df = pd.read_csv('tests/train.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_col = 'target'
        cleaned_train_df, target_series, cleaned_test_df = pb.clean(
            train_df, test_df, target_col)
        self.assertEqual(len(cleaned_test_df.dropna()), len(test_df))
        self.assertEqual(len(cleaned_train_df), len(target_series))
        self.assertEqual(len(cleaned_train_df.columns),
                         len(cleaned_test_df.columns))
        self.assertEqual(target_series.name, "y1:" + target_col)

    def test_clean_optuna_regressor(self):
        train_df = pd.read_csv('tests/train_regression.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_col = 'target'
        cleaned_train_df, target_series, cleaned_test_df = pb.clean(
            train_df, test_df, target_col)
        self.assertEqual(len(cleaned_test_df.dropna()), len(test_df))
        self.assertEqual(len(cleaned_train_df), len(target_series))
        self.assertEqual(len(cleaned_train_df.columns),
                         len(cleaned_test_df.columns))
        self.assertEqual(target_series.name, "y1:" + target_col)

    def test_ludwig(self):
        train_df = pd.read_csv('tests/train.csv')

        model_definition = pb.make_model_definition_for_ludwig(
            train_df, 'target')
        self.assertEqual(model_definition, {'output_features': [{'name': 'target', 'type': 'binary'}], 'input_features': [{'name': 'height', 'type': 'numerical'}, {
                         'name': 'class', 'type': 'numerical'}, {'name': 'class_text', 'type': 'category'}, {'name': 'switch', 'type': 'category'}]})

if __name__ == '__main__':
    unittest.main()
