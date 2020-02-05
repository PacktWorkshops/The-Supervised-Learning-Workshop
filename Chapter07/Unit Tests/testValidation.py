import unittest
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

house_prices_reg = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'boston_house_prices_regression.csv'))
titanic_clf = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic_classification.csv'))

with open(os.path.join(ROOT_DIR, '../..', 'Chapter06', 'Saved Models/stacked_linear_regression.pkl'), 'rb') as f:
    reg = pickle.load(f)

with open(os.path.join(ROOT_DIR, '../..', 'Chapter06', 'Saved Models/random_forest_clf.pkl'), 'rb') as f:
    rf = pickle.load(f)


class TestingEvaluation(unittest.TestCase):
    def test_dataset_shape(self):
        self.assertEqual(house_prices_reg.shape, (102, 18))
        self.assertEqual(titanic_clf.shape, (891, 10))

    def test_mae_rmse(self):
        X = house_prices_reg.drop(columns=['y'])
        y = house_prices_reg['y'].values
        y_pred = reg.predict(X)
        self.assertAlmostEqual(mean_absolute_error(y, y_pred), 2.87408434, places=4)
        self.assertAlmostEqual(sqrt(mean_squared_error(y, y_pred)), 4.50458398, places=4)

    def test_accuracy_precision_recall(self):
        X = titanic_clf.iloc[:, :-1].values
        y = titanic_clf.iloc[:, -1].values
        y_pred = rf.predict(X)
        self.assertAlmostEqual(accuracy_score(y, y_pred), 0.64646465, places=4)
        self.assertEqual(confusion_matrix(y_pred=y_pred, y_true=y)[0][0], 547)
        self.assertAlmostEqual(precision_score(y, y_pred), 0.93548387, places=4)
        self.assertAlmostEqual(recall_score(y, y_pred), 0.08479532, places=4)
        self.assertAlmostEqual(f1_score(y, y_pred), 0.15549598, places=4)

    def test_kfcv(self):
        X = titanic_clf.iloc[:, :-1].values
        y = titanic_clf.iloc[:, -1].values
        skf = StratifiedKFold(n_splits=5)
        scores = []

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            rf_skf = RandomForestClassifier(**rf.get_params())

            rf_skf.fit(X_train, y_train)
            y_pred = rf_skf.predict(X_val)

            scores.append(accuracy_score(y_val, y_pred))

        self.assertAlmostEqual(scores[0], 0.61452514, places=4)
        self.assertAlmostEqual(np.mean(scores), 0.71056069, places=4)


if __name__ == '__main__':
    unittest.main()
