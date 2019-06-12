# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
from unittest import TestCase
import polar_bear as pb
import pandas as pd


class TestPolarBear(TestCase):
    def test_clean(self):
        train_df = pd.read_csv('tests/train.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_col = 'target'
        cleaned_train_df, target_df, cleaned_test_df = pb.clean(
            train_df, test_df, target_col, 0.5)

        self.assertEqual(len(cleaned_test_df.dropna()), len(test_df))
        self.assertEqual(len(cleaned_train_df), len(target_df))
        self.assertEqual(len(cleaned_train_df.columns),
                         len(cleaned_test_df.columns))
        self.assertEqual(target_df.columns[0], "y1:" + target_col)

        expected_train_df = pd.read_csv('tests/expected_train.csv')
        expected_test_df = pd.read_csv('tests/expected_test.csv')
        print(cleaned_train_df.columns.to_list())
        print(expected_train_df.columns.to_list())
        self.assertListEqual(cleaned_train_df.columns.to_list(), expected_train_df.columns.to_list())
        self.assertListEqual(cleaned_test_df.columns.to_list(), expected_test_df.columns.to_list())

    def test_clean_optuna(self):
        train_df = pd.read_csv('tests/train.csv')
        test_df = pd.read_csv('tests/test.csv')
        target_col = 'target'
        cleaned_train_df, target_df, cleaned_test_df = pb.clean(
            train_df, test_df, target_col)
        self.assertEqual(len(cleaned_test_df.dropna()), len(test_df))
        self.assertEqual(len(cleaned_train_df), len(target_df))
        self.assertEqual(len(cleaned_train_df.columns),
                         len(cleaned_test_df.columns))
        self.assertEqual(target_df.columns[0], "y1:" + target_col)
