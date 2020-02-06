import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os


class TestingExercise5_07(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

        np.random.seed(10)
        samples = np.random.randint(0, len(self.df), 10)
        self.df_test = self.df.iloc[samples]
        self.df = self.df.drop(samples)

        self.model = DecisionTreeClassifier()
        self.model = self.model.fit(self.df[set(self.df.columns) - {'diagnosis'}], self.df.diagnosis)

    def test_dataframe_shape(self):
        expected_shape = (559, 31)
        actual_shape = self.df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_train_set_accuracy(self):
        train_accuracy = self.model.score(self.df[set(self.df.columns)-{'diagnosis'}], self.df.diagnosis)
        self.assertEqual(train_accuracy, 1)

    def test_test_set_accuracy(self):
        test_accuracy = self.model.score(self.df_test[set(self.df_test.columns)-{'diagnosis'}], self.df_test.diagnosis)
        self.assertAlmostEqual(test_accuracy, 0.9, places=1)


if __name__ == '__main__':
    unittest.main()
