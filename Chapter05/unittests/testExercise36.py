import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets',
                         'linear_classifier.csv'))

model = LinearRegression()
model.fit(df.x.values.reshape((-1, 1)), df.y.values.reshape((-1, 1)))


class TestingExercise36(unittest.TestCase):
    def test_dataframe_shape(self):
        expected_shape = (22, 3)
        actual_shape = df.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_linreg_params(self):
        actual_model_coeffs = (model.coef_[0][0], model.intercept_[0])
        expected_model_coeffs = (1.63634014, 5.50840010)
        self.assertAlmostEqual(actual_model_coeffs[0], expected_model_coeffs[0], places=4)
        self.assertAlmostEqual(actual_model_coeffs[1], expected_model_coeffs[1], places=4)

    def test_linreg_accuracy(self):
        y_pred = model.predict(df.x.values.reshape((-1, 1)))
        pred_labels = []

        for _y, _y_pred in zip(df.y, y_pred):
            if _y < _y_pred:
                pred_labels.append('o')
            else:
                pred_labels.append('x')
        df['Pred Labels'] = pred_labels
        actual_accuracy = np.mean(df.labels == df['Pred Labels'])
        expected_accuracy = 0.90909091
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, places=4)


if __name__ == '__main__':
    unittest.main()
