import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os


class TestingExercise5_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets',
                                      'linear_classifier.csv'))

        self.model = LinearRegression()
        self.model.fit(self.df.x.values.reshape((-1, 1)), self.df.y.values.reshape((-1, 1)))

    def test_dataframe_shape(self):
        expected_shape = (22, 3)
        actual_shape = self.df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_linreg_params(self):
        actual_model_coeffs = (self.model.coef_[0][0], self.model.intercept_[0])
        expected_model_coeffs = (1.63634014, 5.50840010)
        self.assertAlmostEqual(actual_model_coeffs[0], expected_model_coeffs[0], places=4)
        self.assertAlmostEqual(actual_model_coeffs[1], expected_model_coeffs[1], places=4)

    def test_linreg_accuracy(self):
        y_pred = self.model.predict(self.df.x.values.reshape((-1, 1)))
        pred_labels = []

        for _y, _y_pred in zip(self.df.y, y_pred):
            if _y < _y_pred:
                pred_labels.append('o')
            else:
                pred_labels.append('x')
        self.df['Pred Labels'] = pred_labels
        actual_accuracy = np.mean(self.df.labels == self.df['Pred Labels'])
        expected_accuracy = 0.90909091
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, places=4)


if __name__ == '__main__':
    unittest.main()
