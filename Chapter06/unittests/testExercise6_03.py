import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os


class TestingExercise6_03(unittest.TestCase):
    def setUp(self) -> None:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data = pd.read_csv(os.path.join(ROOT_DIR, '..', 'Datasets', 'titanic.csv'))

        train, val = train_test_split(self.data, test_size=0.2, random_state=11)

        self.x_train = self.preprocess(train)
        self.y_train = train['Survived'].values

        self.x_val = self.preprocess(val)
        self.y_val = val['Survived'].values

    @staticmethod
    def preprocess(data):
        def fix_age(age):
            if np.isnan(age):
                return -1
            elif age < 1:
                return age * 100
            else:
                return age

        data.loc[:, 'Gender'] = data.Gender.apply(lambda s: int(s == 'female'))
        data.loc[:, 'Age'] = data.Age.apply(fix_age)

        embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C', 'Emb_Q', 'Emb_S']]
        cols = ['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare']

        return pd.concat([data[cols], embarked], axis=1).values

    def test_dataframe_shape(self):
        expected_shape = (712, 9)
        actual_shape = self.x_train.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_random_forest(self):
        rf_params = {
            'n_estimators': 100,
            'criterion': 'entropy',
            'max_features': 0.5,
            'min_samples_leaf': 10,
            'random_state': 11,
            'n_jobs': -1
        }
        rf = RandomForestClassifier(**rf_params)
        rf.fit(self.x_train, self.y_train)
        rf_preds_train = rf.predict(self.x_train)
        rf_preds_val = rf.predict(self.x_val)
        train_accuracy = accuracy_score(y_true=self.y_train, y_pred=rf_preds_train)
        val_accuracy = accuracy_score(y_true=self.y_val, y_pred=rf_preds_val)
        self.assertAlmostEqual(train_accuracy, 0.82303371, places=4)
        self.assertAlmostEqual(val_accuracy, 0.86033520, places=4)


if __name__ == '__main__':
    unittest.main()
