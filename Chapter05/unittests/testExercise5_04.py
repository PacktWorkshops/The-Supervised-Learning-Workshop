import unittest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import os


class TestingExercise5_04(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

        self.df_test = self.df.iloc[430]
        self.df = self.df.drop([430])  # Remove the sample

        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.df[['mean radius',
                                                                                           'worst radius']],
                                                                                  self.df.diagnosis, test_size=0.2,
                                                                                  random_state=123)

        self.model = KNN(n_neighbors=3)
        self.model.fit(X=self.train_X, y=self.train_y)

    def test_dataframe_shape(self):
        expected_shape = (568, 31)
        actual_shape = self.df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_validation_accuracy(self):
        actual_accuracy = self.model.score(X=self.valid_X, y=self.valid_y)
        self.assertAlmostEqual(actual_accuracy, 0.93859649, places=4)

    def test_train_accuracy(self):
        actual_accuracy = self.model.score(X=self.train_X, y=self.train_y)
        self.assertAlmostEqual(actual_accuracy, 0.93832599, places=4)

    def test_test_smaple_prediction(self):
        pred = self.model.predict(self.df_test[['mean radius', 'worst radius']].values.reshape((-1, 2)))[0]
        truth = self.df_test.diagnosis
        self.assertEqual(pred, 'benign')
        self.assertEqual(truth, 'malignant')


if __name__ == '__main__':
    unittest.main()
