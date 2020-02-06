import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
import os


class TestingActivity5_02(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'breast-cancer-data.csv'))

        X, y = self.df[[c for c in self.df.columns if c != 'diagnosis']], self.df.diagnosis

        skb_model = SelectKBest(k=2)
        X_new = skb_model.fit_transform(X, y)

        # get the k - best column names
        mask = skb_model.get_support()  # list of booleans
        self.selected_features = []  # The list of your K best features

        for bool, feature in zip(mask, self.df.columns):
            if bool:
                self.selected_features.append(feature)

        diagnoses = [
            'benign',  # 0
            'malignant',  # 1
        ]
        self.output = [diagnoses.index(diag) for diag in self.df.diagnosis]

        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(self.df[self.selected_features],
                                                                                  self.output, test_size=0.2,
                                                                                  random_state=123)

        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(self.train_X, self.train_y)

    def test_best_features(self):
        self.assertEqual(self.selected_features[0], 'worst perimeter')
        self.assertEqual(self.selected_features[1], 'worst concave points')

    def test_validation_accuracy(self):
        valid_accuracy = self.model.score(self.valid_X, self.valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.93859649, places=4)

    def test_random_feat_model_accruacy(self):
        selected_features = [
            'mean radius',  # List features here
            'mean texture',
            'compactness error'
        ]
        train_X, valid_X, train_y, valid_y = train_test_split(self.df[selected_features], self.output,
                                                              test_size=0.2, random_state=123)
        model = LogisticRegression(solver='liblinear')
        model.fit(train_X, train_y)
        valid_accuracy = model.score(valid_X, valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.88596491, places=4)

    def test_all_feats_model_accruacy(self):
        selected_features = [
            feat for feat in self.df.columns if feat != 'diagnosis'  # List features here
        ]
        train_X, valid_X, train_y, valid_y = train_test_split(self.df[selected_features], self.output,
                                                              test_size=0.2, random_state=123)
        model = LogisticRegression(solver='liblinear')
        model.fit(train_X, train_y)
        valid_accuracy = model.score(valid_X, valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.98245614, places=4)


if __name__ == '__main__':
    unittest.main()
