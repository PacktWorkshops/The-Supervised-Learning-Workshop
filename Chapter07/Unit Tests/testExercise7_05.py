import unittest
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import os


class TestingActivity7_04(unittest.TestCase):
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

    def test_feature_importance(self):
        self.assertAlmostEqual(self.rf.feature_importances_[0], 0.15641979, places=2)
        self.assertAlmostEqual(self.rf.feature_importances_[1], 0.41157803, places=2)


if __name__ == '__main__':
    unittest.main()
