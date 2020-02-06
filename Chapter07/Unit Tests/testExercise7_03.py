import unittest
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os


class TestingActivity7_03(unittest.TestCase):
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

    def test_kfcv(self):
        X = self.titanic_clf.iloc[:, :-1].values
        y = self.titanic_clf.iloc[:, -1].values
        skf = StratifiedKFold(n_splits=5)
        scores = []

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            rf_skf = RandomForestClassifier(**self.rf.get_params())

            rf_skf.fit(X_train, y_train)
            y_pred = rf_skf.predict(X_val)

            scores.append(accuracy_score(y_val, y_pred))

        self.assertAlmostEqual(scores[0], 0.61452514, places=4)
        self.assertAlmostEqual(np.mean(scores), 0.71056069, places=4)


if __name__ == '__main__':
    unittest.main()
