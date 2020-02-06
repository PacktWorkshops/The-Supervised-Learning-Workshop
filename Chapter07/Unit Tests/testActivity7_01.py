import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import os


class TestingActivity7_01(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis'].map({'malignant': 1, 'benign': 0}.get).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    def test_dataset_shape(self):
        self.assertEqual(self.X_train.shape, (455, 30))

    def test_model_scores(self):
        meta_gbc = GradientBoostingClassifier()
        param_dist = {
            'n_estimators': list(range(10, 210, 10)),
            'criterion': ['mae', 'mse'],
            'max_features': ['sqrt', 'log2', 0.25, 0.3, 0.5, 0.8, None],
            'max_depth': list(range(1, 10)),
            'min_samples_leaf': list(range(1, 10))
        }
        rand_search_params = {
            'param_distributions': param_dist,
            'scoring': 'accuracy',
            'n_iter': 100,
            'cv': 5,
            'return_train_score': True,
            'n_jobs': -1,
            'random_state': 11
        }
        random_search = RandomizedSearchCV(meta_gbc, **rand_search_params)
        random_search.fit(self.X_train, self.y_train)
        idx = np.argmax(random_search.cv_results_['mean_test_score'])
        final_params = random_search.cv_results_['params'][idx]
        train_X, val_X, train_y, val_y = train_test_split(self.X_train, self.y_train, test_size=0.15, random_state=11)
        gbc = GradientBoostingClassifier(**final_params)
        gbc.fit(train_X, train_y)

        final_threshold = 0.05
        pred_probs_test = np.array([each[1] for each in gbc.predict_proba(self.X_test)])
        preds_test = (pred_probs_test > final_threshold).astype(int)

        self.assertEqual(train_X.shape, (386, 30))
        self.assertEqual(preds_test[0], 1)


if __name__ == '__main__':
    unittest.main()
