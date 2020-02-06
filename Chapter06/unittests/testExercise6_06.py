import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import os


class TestingExercise6_06(unittest.TestCase):
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

    def test_stacking(self):
        x_train_with_metapreds = np.zeros((self.x_train.shape[0], self.x_train.shape[1] + 2))
        x_train_with_metapreds[:, :-2] = self.x_train
        x_train_with_metapreds[:, -2:] = -1
        self.assertEqual(x_train_with_metapreds[0][0], 3)

        kf = KFold(n_splits=5, random_state=11)
        for train_indices, val_indices in kf.split(self.x_train):
            kfold_x_train, kfold_x_val = self.x_train[train_indices], self.x_train[val_indices]
            kfold_y_train, kfold_y_val = self.y_train[train_indices], self.y_train[val_indices]
            svm = LinearSVC(random_state=11, max_iter=1000)
            svm.fit(kfold_x_train, kfold_y_train)
            svm_pred = svm.predict(kfold_x_val)
            knn = KNeighborsClassifier(n_neighbors=4)
            knn.fit(kfold_x_train, kfold_y_train)
            knn_pred = knn.predict(kfold_x_val)
            x_train_with_metapreds[val_indices, -2] = svm_pred
            x_train_with_metapreds[val_indices, -1] = knn_pred

        x_val_with_metapreds = np.zeros((self.x_val.shape[0], self.x_val.shape[1] + 2))
        x_val_with_metapreds[:, :-2] = self.x_val
        x_val_with_metapreds[:, -2:] = -1
        self.assertEqual(x_val_with_metapreds[0][0], 3)

        svm = LinearSVC(random_state=11, max_iter=1000)
        svm.fit(self.x_train, self.y_train)
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(self.x_train, self.y_train)
        svm_pred = svm.predict(self.x_val)
        knn_pred = knn.predict(self.x_val)
        x_val_with_metapreds[:, -2] = svm_pred
        x_val_with_metapreds[:, -1] = knn_pred

        lr = LogisticRegression(random_state=11)
        lr.fit(x_train_with_metapreds, self.y_train)
        lr_preds_train = lr.predict(x_train_with_metapreds)
        lr_preds_val = lr.predict(x_val_with_metapreds)

        train_accuracy_st = accuracy_score(y_true=self.y_train, y_pred=lr_preds_train)
        val_accuracy_st = accuracy_score(y_true=self.y_val, y_pred=lr_preds_val)
        train_accuracy_kn = accuracy_score(y_true=self.y_train, y_pred=svm.predict(self.x_train))
        val_accuracy_kn = accuracy_score(y_true=self.y_val, y_pred=svm_pred)
        train_accuracy_sv = accuracy_score(y_true=self.y_train, y_pred=knn.predict(self.x_train))
        val_accuracy_sv = accuracy_score(y_true=self.y_val, y_pred=knn_pred)
        self.assertAlmostEqual(train_accuracy_st, 0.78651685, places=2)
        self.assertAlmostEqual(val_accuracy_st, 0.88826816, places=2)
        self.assertAlmostEqual(train_accuracy_kn, 0.69241573, places=2)
        self.assertAlmostEqual(val_accuracy_kn, 0.76536313, places=2)
        self.assertAlmostEqual(train_accuracy_sv, 0.79213483, places=2)
        self.assertAlmostEqual(val_accuracy_sv, 0.67039106, places=2)


if __name__ == '__main__':
    unittest.main()
