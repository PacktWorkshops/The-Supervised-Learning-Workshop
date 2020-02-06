import unittest
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import os


class TestingActivity7_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.house_prices_reg = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'boston_house_prices_regression.csv'))
        self.titanic_clf = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic_classification.csv'))

        with open(os.path.join(ROOT_DIR, '../..', 'Chapter06', 'Saved Models/stacked_linear_regression.pkl'),
                  'rb') as f:
            self.reg = pickle.load(f)

        with open(os.path.join(ROOT_DIR, '../..', 'Chapter06', 'Saved Models/random_forest_clf.pkl'), 'rb') as f:
            self.rf = pickle.load(f)

    def test_dataset_shape(self):
        self.assertEqual(self.house_prices_reg.shape, (102, 18))
        self.assertEqual(self.titanic_clf.shape, (891, 10))

    def test_mae_rmse(self):
        X = self.house_prices_reg.drop(columns=['y'])
        y = self.house_prices_reg['y'].values
        y_pred = self.reg.predict(X)
        self.assertAlmostEqual(mean_absolute_error(y, y_pred), 2.87408434, places=4)
        self.assertAlmostEqual(sqrt(mean_squared_error(y, y_pred)), 4.50458398, places=4)


if __name__ == '__main__':
    unittest.main()
