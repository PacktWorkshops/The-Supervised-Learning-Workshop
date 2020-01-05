import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'breast-cancer-data.csv'))

X, y = df[[c for c in df.columns if c != 'diagnosis']], df.diagnosis

skb_model = SelectKBest(k=2)
X_new = skb_model.fit_transform(X, y)

# get the k - best column names
mask = skb_model.get_support() #list of booleans
selected_features = [] # The list of your K best features

for bool, feature in zip(mask, df.columns):
    if bool:
        selected_features.append(feature)

diagnoses = [
    'benign', # 0
    'malignant', # 1
]
output = [diagnoses.index(diag) for diag in df.diagnosis]

train_X, valid_X, train_y, valid_y = train_test_split(df[selected_features], output,
                                                      test_size=0.2, random_state=123)

model = LogisticRegression(solver='liblinear')
model.fit(train_X, train_y)


class TestingActivity12(unittest.TestCase):
    def test_best_features(self):
        self.assertEqual(selected_features[0], 'worst perimeter')
        self.assertEqual(selected_features[1], 'worst concave points')

    def test_validation_accuracy(self):
        valid_accuracy = model.score(valid_X, valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.93859649)

    def test_random_feat_model_accruacy(self):
        selected_features = [
            'mean radius',  # List features here
            'mean texture',
            'compactness error'
        ]
        train_X, valid_X, train_y, valid_y = train_test_split(df[selected_features], output,
                                                              test_size=0.2, random_state=123)
        model = LogisticRegression(solver='liblinear')
        model.fit(train_X, train_y)
        valid_accuracy = model.score(valid_X, valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.88596491)

    def test_all_feats_model_accruacy(self):
        selected_features = [
            feat for feat in df.columns if feat != 'diagnosis'  # List features here
        ]
        train_X, valid_X, train_y, valid_y = train_test_split(df[selected_features], output,
                                                              test_size=0.2, random_state=123)
        model = LogisticRegression(solver='liblinear')
        model.fit(train_X, train_y)
        valid_accuracy = model.score(valid_X, valid_y)
        self.assertAlmostEqual(valid_accuracy, 0.98245614)


if __name__ == '__main__':
    unittest.main()
