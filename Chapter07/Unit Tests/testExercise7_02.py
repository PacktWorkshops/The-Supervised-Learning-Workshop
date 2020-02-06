import unittest
import pandas as pd
import pickle
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
import os


class TestingActivity7_02(unittest.TestCase):
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

    def test_accuracy_precision_recall(self):
        X = self.titanic_clf.iloc[:, :-1].values
        y = self.titanic_clf.iloc[:, -1].values
        y_pred = self.rf.predict(X)
        self.assertAlmostEqual(accuracy_score(y, y_pred), 0.64646465, places=4)
        self.assertEqual(confusion_matrix(y_pred=y_pred, y_true=y)[0][0], 547)
        self.assertAlmostEqual(precision_score(y, y_pred), 0.93548387, places=4)
        self.assertAlmostEqual(recall_score(y, y_pred), 0.08479532, places=4)
        self.assertAlmostEqual(f1_score(y, y_pred), 0.15549598, places=4)


if __name__ == '__main__':
    unittest.main()
