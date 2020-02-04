import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

np.random.seed(10)
samples = np.random.randint(0, len(df), 10)
df_test = df.iloc[samples]
df = df.drop(samples)

model = DecisionTreeClassifier()
model = model.fit(df[set(df.columns)-{'diagnosis'}], df.diagnosis)


class TestingExercise42(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (559, 31)
        actual_shape = df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_train_set_accuracy(self):
        train_accuracy = model.score(df[set(df.columns)-{'diagnosis'}], df.diagnosis)
        self.assertEqual(train_accuracy, 1)

    def test_test_set_accuracy(self):
        test_accuracy = model.score(df_test[set(df_test.columns)-{'diagnosis'}], df_test.diagnosis)
        self.assertAlmostEqual(test_accuracy, 0.9, places=1)


if __name__ == '__main__':
    unittest.main()
