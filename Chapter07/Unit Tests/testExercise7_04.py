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

    def test_grid_search(self):
        X = self.titanic_clf.iloc[:, :-1].values
        y = self.titanic_clf.iloc[:, -1].values

        param_dist = {"n_estimators": list(range(10, 210, 10)),
                      "max_depth": list(range(3, 20)),
                      "max_features": list(range(1, 10)),
                      "min_samples_split": list(range(2, 11)),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        rf_rand = RandomForestClassifier()

        n_iter_search = 60
        random_search = RandomizedSearchCV(rf_rand, param_distributions=param_dist, scoring='accuracy',
                                           n_iter=n_iter_search, cv=5)
        random_search.fit(X, y)

        self.assertEqual(len(random_search.cv_results_), 19)
        self.assertEqual(len(list(random_search.cv_results_.values())[0]), 60)


if __name__ == '__main__':
    unittest.main()
